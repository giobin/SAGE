import base64
import io
import json
import os
import re
import unicodedata

import gymnasium as gym
import numpy as np

from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
from PIL import Image

from src.config import GuidanceType


def get_prompt(config: DictConfig) -> Tuple[str, bool]:
    """
    Returns the prompt based on the configuration.
    Disambiguates between text and image prompts automatically based on the configuration.
    Automatically handles CoT (Chain of Thought) prompts if specified in the configuration.
    
    Args:
        prompts_config (DictConfig): The configuration object containing the prompts path and type.
        
    Returns:
        str: The unformatted prompt string.
        bool: Whether the prompt is for an image or text.
    """
    prompts_config = config.env.prompt
    # Force image mode for certain envs which only support visual observations
    image_mode = config.env.image_obs or config.env.env_id.startswith("MiniGrid") or config.env.env_id.startswith("BabyAI")
    
    prompts_path = prompts_config.prompts_path
    use_cot_prompt = prompts_config.get("use_cot_prompt", False)
    # if guidance_type requires for prob distribution we need to set the distribution prompt
    if config.training.guidance_model.guidance_type in [GuidanceType.distribution, GuidanceType.full]:
        use_prompt_for_distribution = True
    else:
        use_prompt_for_distribution = False

    key = (
        # Distribution prompts take absolute priority
        "image_cot_distribution_prompt" if use_prompt_for_distribution and image_mode else
        "text_cot_distribution_prompt" if use_prompt_for_distribution and not image_mode else

        # Normal prompts
        "image_cot_prompt" if image_mode and use_cot_prompt else
        "image_prompt" if image_mode else
        "text_cot_prompt" if use_cot_prompt else
        "text_prompt"
    )

    prompt_path = prompts_config.get(key, None)
    if prompt_path is None:
        raise ValueError(f"{key.replace('_', ' ').title()} is not specified in the configuration.")

    # Apply prompt_suffix from guidance model config (e.g. "_qwen" -> image_cot_prompt_qwen.md)
    prompt_suffix = config.training.guidance_model.get("prompt_suffix", "")
    if prompt_suffix:
        base, ext = os.path.splitext(prompt_path)
        variant_path = f"{base}{prompt_suffix}{ext}"
        variant_full = os.path.join(prompts_path, variant_path)
        if os.path.exists(variant_full):
            prompt_path = variant_path

    full_prompt_path = os.path.join(prompts_path, prompt_path)
    if not os.path.exists(full_prompt_path):
        print(f"Prompt file {full_prompt_path} does not exist.")
        raise FileNotFoundError(f"Please provide a valid prompt file.")
    
    with open(full_prompt_path, 'r') as f:
        prompt = f.read()
    
    # Custom formatting for specific environments
    if config.env.env_id == "gym_cards/CardMaze-v0":
        from src.environments.gym_cards.envs.cardmaze import format_prompt
        prompt = format_prompt(prompt, config.env.env_args)

    return prompt.strip(), image_mode


def get_messages(prompt: str, splits: List[str] = ["system", "user", "assistant"], merge_system: bool = False, img_tag: str = "{image}") -> List[Dict[str, str]]:
    """
    Converts a prompt string into a list of messages for each split.
    
    Args:
        prompt (str): The prompt string.
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user", "assistant"].
        merge_system (bool): Whether to merge the system messages into the user messages. Defaults to False.
        img_tag (str): The tag to use for images. Defaults to "{image}".
        
    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """
    
    def _get_content(text: str, img_tag: str = "{image}"):
        if not img_tag in text:
            return text.strip()
        else:
            content = []
            for part in text.split(img_tag):
                if part:
                    content.append({"type": "text", "text": part.strip()})
                content.append({"type": "image"})
            return content[:-1]
    
    messages = []
    for split in splits:
        start_tag = f"<{split}>"
        end_tag = f"</{split}>"

        start_idx = prompt.find(start_tag)
        end_idx = prompt.find(end_tag)
        if end_idx == -1:
            end_tag = f"<\\{split}>"
            end_idx = prompt.find(end_tag)
        
        # Skip if the split is not in the prompt (e.g. no system prompt)
        if start_idx == -1 and end_idx == -1:
            continue
        messages.append({
            "role": split,
            "content": _get_content(prompt[start_idx + len(start_tag):end_idx].strip(), img_tag=img_tag)
        })
    
    # If no splits at all, assume the whole prompt is a user message
    if len(messages) == 0:
        messages.append({
            "role": "user",
            "content": _get_content(prompt.strip(), img_tag=img_tag)
        })
        
    if merge_system:
        for i, message in enumerate(messages):
            if message["role"] == "system" and messages[i+1] and messages[i+1]["role"] == "user":
                if isinstance(messages[i+1]["content"], str) and isinstance(message["content"], str):
                    messages[i+1]["content"] = message["content"] + " " + messages[i+1]["content"]
                elif isinstance(messages[i+1]["content"], dict) and isinstance(message["content"], dict):
                    messages[i+1]["content"].update(message["content"])

        messages = [message for message in messages if message["role"] != "system"]

    return messages


def format_image(img: Image.Image | os.PathLike) -> str:
    """
    Converts a PIL Image to a base64-encoded PNG data URI.
    Args:
        img (Image.Image | os.PathLike): The image to convert. Can be a PIL Image or a path to an image file.
    Returns:
        str: The base64-encoded PNG data URI of the image.
    """
    if isinstance(img, os.PathLike):
        img = Image.open(img)
    img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_data_uri = f'data:image/png;base64,{img_base64}'
    return image_data_uri

def format_messages(messages: List[Dict[str, str]], images: List[List[str]]) -> List[List[Dict[str, str]]]:
    """
    Formats the messages to include the actual image URLs in the user messages.
    This function assumes that the images are provided in the same order the empty image messages appear in the user messages.
    Args:
        messages (list[dict[str, str]]): The list of messages to format.
        images (list[list[str]]): The list of image URLs to include in the messages (can also be base64 strings).
    Returns:
        list[list[dict[str, str]]]: A list of messages with image URLs.
    """
    all_messages_with_images_url = []
    for i, message in enumerate(messages):
        new_message = []
        img_index = 0
        for turn in message:
            if turn['role'] != 'user':
                new_message.append(turn)
            else:
                new_turn = {'role':'user', 'content':[]}
                for one_content in turn['content']:
                    if one_content['type'] != 'image':
                        new_turn['content'].append(one_content)
                    else:
                        one_new_content = {
                            'type':'image_url',
                            'image_url':{'url': images[i][img_index]}
                        }
                        new_turn['content'].append(one_new_content)
                        img_index += 1
                new_message.append(new_turn)
        all_messages_with_images_url.append(new_message)
    return all_messages_with_images_url

# discrete action extraction from text
ACTION_TAG_RE = re.compile(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.IGNORECASE | re.DOTALL)
BOX_TAG_RE = re.compile(r"<\s*\|begin_of_box\|\s*>(.*?)<\s*\|end_of_box\|\s*>", re.IGNORECASE | re.DOTALL)
PROBS_TAG_RE = re.compile(r"<\s*probs\s*>(.*?)<\s*/\s*probs\s*>", re.IGNORECASE | re.DOTALL)
INT_RE = re.compile(r"[+-]?\d+")
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _normalize_digits(s: str) -> str:
    """Convert all Unicode digits to ASCII '0'-'9'."""
    out = []
    for ch in s:
        try:
            out.append(str(unicodedata.decimal(ch)))
        except Exception:
            out.append(ch)
    return "".join(out)


def _last_int_in_text(s: str) -> Optional[int]:
    """Return the last integer found in the string, or None."""
    matches = list(INT_RE.finditer(s))
    if not matches:
        return None
    return int(matches[-1].group(0))


def _is_allowed(idx: int, allowed) -> bool:
    """
    Check if idx is allowed assuming 'allowed' is always a gym.spaces.Discrete.
    If allowed is None, treat all indices as allowed.
    """
    if allowed is None:
        return True
    # Assume gym.spaces.Discrete or an object with .contains(int) method.
    try:
        return bool(allowed.contains(int(idx)))
    except Exception:
        return False


def _space_dimension(space: Optional[gym.Space]) -> Optional[int]:
    """
    Infer the flat dimension of a Gym space.
    """
    if space is None:
        return None
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    if hasattr(space, "shape"):
        try:
            return int(np.prod(space.shape))
        except Exception:
            return None
    if hasattr(space, "n"):
        try:
            return int(space.n)  # type: ignore[arg-type]
        except Exception:
            return None
    return None


def _default_distribution(action_idx: Optional[int], dim: Optional[int]) -> Optional[np.ndarray]:
    """
    Build a one-hot distribution centered on action_idx.
    """
    if action_idx is None or dim is None:
        return None
    if action_idx < 0 or action_idx >= dim:
        return None
    dist = np.zeros(dim, dtype=np.float32)
    dist[action_idx] = 1.0
    return dist


def _flatten_numeric_values(value) -> List[float]:
    """
    Recursively flatten nested numeric containers into a list of floats.
    """
    if value is None:
        return []
    if isinstance(value, (float, int)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return _flatten_numeric_values(value.tolist())
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    if isinstance(value, dict):
        items = value.items()
        try:
            items = sorted(value.items(), key=lambda kv: int(kv[0]))
        except Exception:
            pass
        flattened: List[float] = []
        for _, v in items:
            flattened.extend(_flatten_numeric_values(v))
        return flattened
    if isinstance(value, str):
        return [float(match.group(0)) for match in FLOAT_RE.finditer(value)]
    return []


def _validate_distribution_candidate(raw_values, expected_dim: Optional[int], verbose: bool) -> Optional[np.ndarray]:
    """
    Convert raw probability values into a validated numpy vector.
    """
    floats = _flatten_numeric_values(raw_values)
    if not floats:
        return None
    if expected_dim is not None and len(floats) != expected_dim:
        if verbose:
            print(f"Distribution length mismatch: expected {expected_dim}, got {len(floats)}")
        return None
    probs = np.asarray(floats, dtype=np.float32)
    if not np.all(np.isfinite(probs)):
        if verbose:
            print("Distribution contains non-finite values.")
        return None
    if np.any(probs < 0):
        if verbose:
            print("Distribution contains negative values.")
        return None
    total = float(probs.sum())
    if not np.isclose(total, 1.0, atol=5e-3):
        if verbose:
            print(f"Distribution does not sum to 1 (sum={total}).")
        return None
    return probs


def _parse_distribution_from_text(text: str, expected_dim: Optional[int], verbose: bool) -> Optional[np.ndarray]:
    """
    Extract probability distributions from <probs>...</probs> tags.
    """
    for block in PROBS_TAG_RE.findall(text):
        probs = _validate_distribution_candidate(block, expected_dim, verbose)
        if probs is not None:
            return probs
    return None


def _sample_allowed(allowed):
    """
    Sample an action from the allowed Discrete space.
    Returns None if 'allowed' is None or sampling fails.
    """
    if allowed is None:
        return None
    try:
        # gym.spaces.Discrete.sample() returns an int-like type.
        return int(allowed.sample())
    except Exception:
        return None
    
def _extract_from_tags(t: str, pattern: re.Pattern, allowed) -> Optional[int]:
    """Generic tag extractor for action-like tags."""
    tags = pattern.findall(t)
    if tags:
        for tag_content in reversed(tags):
            idx = _last_int_in_text(tag_content)
            if idx is not None and _is_allowed(idx, allowed):
                return idx
        sampled = _sample_allowed(allowed)
        if sampled is not None:
            return sampled
        return None
    return None

def extract_action_index(
    text: str,
    allowed=None,  # Assume gym.spaces.Discrete; no need to handle other cases.
    verbose: bool = False,
    return_source: bool = False,
) -> Union[Optional[int], Tuple[Optional[int], str]]:
    """
    Extract an action index from a model response.

    Strategy:
    1) If <action>...</action> tags exist, use the LAST tag and the LAST integer inside it.
    2) If <begin_of_box>...<end_of_box> special tokens exist (used by GLM-4.1v), use the LAST integer inside the last box.
    3) Otherwise, use the LAST integer in the whole text (ignoring fenced/inline code).

    Validation:
    - The extracted index must be allowed by the provided Discrete space.

    Fallback:
    - If we FAIL to extract a VALID index (none found or not allowed),
      we SAMPLE uniformly from the allowed Discrete space.
    - If 'allowed' is None, return None on failure.

    If return_source is True, returns (action, source) where source is one of:
      "action_tag", "box_tag", "fallback_int", "sampled", "none"
    """
    vprint = print if verbose else lambda *args, **kwargs: None

    def _ret(idx, source):
        return (idx, source) if return_source else idx

    if not text:
        # No text: try sampling if possible.
        vprint("No text provided, sampling from allowed actions.")
        return _ret(_sample_allowed(allowed), "sampled")

    t = _normalize_digits(text)

    # 1) <action>...</action>
    idx = _extract_from_tags(t, ACTION_TAG_RE, allowed)
    if idx is not None:
        vprint(f"Extracted action index from <action> tag: {idx}")
        return _ret(idx, "action_tag")

    # 2) <|begin_of_box|>...</|end_of_box|>
    idx = _extract_from_tags(t, BOX_TAG_RE, allowed)
    if idx is not None:
        vprint(f"Extracted action index from <|begin_of_box|> tag: {idx}")
        return _ret(idx, "box_tag")

    # 3) Fallback: remove fenced and inline code, then search for the last integer.
    t_fallback = FENCED_CODE_RE.sub(" ", t)
    t_fallback = INLINE_CODE_RE.sub(" ", t_fallback)
    idx = _last_int_in_text(t_fallback)

    if idx is not None and _is_allowed(idx, allowed):
        vprint(f"Extracted action index from fallback (last integer): {idx}")
        return _ret(idx, "fallback_int")

    # Final fallback: sample from allowed actions if possible.
    vprint("No valid action index found, sampling from allowed actions.")
    return _ret(_sample_allowed(allowed), "sampled")

# continuous action extraction from text
def extract_action_and_thinking(
    json_str,
    allowed,
    verbose: bool = False,
    guidance_type: GuidanceType = GuidanceType.action,
    distribution_space: Optional[gym.Space] = None,
):
    """
    Extract the chosen action and explanation from a model output.

    Parameters:
        json_str (str): Output string from the model (may be JSON or plain text).
        allowed (gym.Space): The (Discrete) action space.
        verbose (bool): If True, print debug info.
        guidance_type (GuidanceType): Whether a probability distribution is expected.
        distribution_space (gym.Space | None): Optional gym space describing the probability vector.

    Returns:
        tuple: (action, explanation, distribution, failed_distribution_parses, action_parse_source)
            - action: int or None
            - explanation: str or ""
            - distribution: Optional[np.ndarray] of probabilities over discrete actions
            - failed_distribution_parses: int flag indicating missing/invalid distributions
            - action_parse_source: str indicating how the action was extracted
              ("action_tag", "box_tag", "fallback_int", "sampled", "json", "none")
    """
    vprint = print if verbose else (lambda *args, **kwargs: None)
    expected_distribution = guidance_type in (GuidanceType.distribution, GuidanceType.full)
    distribution_dim = _space_dimension(distribution_space)
    if distribution_dim is None:
        distribution_dim = _space_dimension(allowed)

    try:
        data = json.loads(json_str)
        parsed_json = isinstance(data, dict) and "action" in data and "explanation" in data
        if verbose and parsed_json:
            print(f"Parsed JSON: {data}")
    except json.JSONDecodeError:
        data = None
        parsed_json = False

    action: Optional[Union[int, np.ndarray]] = None
    explanation = str(json_str)
    distribution: Optional[np.ndarray] = None
    distribution_parsed = False
    action_parse_source = "none"

    if parsed_json:
        action_val = data["action"]
        explanation = str(data["explanation"])
        action_parse_source = "json"
        try:
            action = int(action_val)
        except Exception as e:
            vprint(f"Failed casting Discrete action: {e}")
            action = None
            action_parse_source = "none"

        if expected_distribution:
            for key in ("distribution", "probabilities", "probs"):
                if key in data:
                    distribution = _validate_distribution_candidate(data[key], distribution_dim, verbose)
                    distribution_parsed = distribution is not None
                    break
    else:
        result = extract_action_index(text=json_str, allowed=allowed, verbose=verbose, return_source=True)
        if isinstance(result, tuple):
            action, action_parse_source = result
        else:
            action = result
            action_parse_source = "none"
        if expected_distribution:
            distribution = _parse_distribution_from_text(json_str, distribution_dim, verbose)
            distribution_parsed = distribution is not None

    if expected_distribution and not distribution_parsed:
        alternate = _parse_distribution_from_text(json_str, distribution_dim, verbose)
        if alternate is not None:
            distribution = alternate
            distribution_parsed = True

    failed_distribution_parses = 0
    if expected_distribution and not distribution_parsed:
        failed_distribution_parses = 1
        distribution = _default_distribution(
            int(action) if isinstance(action, (int, np.integer)) else None,
            distribution_dim,
        )

    return action, explanation, distribution, failed_distribution_parses, action_parse_source

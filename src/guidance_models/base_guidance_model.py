import torch

from typing import List, Optional, Dict, Any
from omegaconf import DictConfig
from PIL import Image
from abc import ABC, abstractmethod
from tqdm import tqdm

from src.guidance_models.common import get_prompt
from src.environments.common import get_action_description
from src.utils import tensor_to_pil
from src.config import VLMCacheType

class BaseGuidanceModel(ABC):
    def __init__(self, guidance_model_config: DictConfig, verbose: bool = False, **kwargs) -> None:
        """
        Initialize the base guidance model.
        Base class does not implement the model init logic, but just initializes the cache
        
        Parameters:
            guidance_model_config (DictConfig): The configuration for the guidance model.
            verbose (bool): Whether to print verbose output during model loading and inference.
        """
        
        self.verbose = verbose
        self._vprint = print if self.verbose else lambda *args, **kwargs: None

        self.guidance_model_config = guidance_model_config
        self.vlm_generation_call_count = 0
        self.vlm_generation_prompt_count = 0
        self.use_cache = guidance_model_config.cache.use_cache
        if self.use_cache:
            if guidance_model_config.cache.cache_type == VLMCacheType.vector:
                from src.guidance_models.cache.vector_cache import VectorCache
                self.cache = VectorCache(max_size=guidance_model_config.cache.cache_size, vector_size=guidance_model_config.cache.cache_vector_size)
                self.use_vector_cache = True
            elif guidance_model_config.cache.cache_type == VLMCacheType.single:
                from src.guidance_models.cache import LRUCache
                self.cache = LRUCache(max_size=guidance_model_config.cache.cache_size)
                self.use_vector_cache = False
            else:
                raise ValueError(f"Unsupported cache type: {guidance_model_config.cache.cache_type}. Supported types are: {VLMCacheType.vector}, {VLMCacheType.single}.")
            
        # Must be implemented in subclasses that call external model backends.
        self.model_id = None
        self.img_tag = None
        self.seed = None
        self.device = None
        self.dtype = None
        self.tokenizer = None
        self.sampling_params = None
        
    def __del__(self):
        """
        Destructor should always delete the VLM and free up resources, implement in subclass
        """
        if hasattr(self, "cache"):
            del self.cache
    
    @abstractmethod
    def _generate_no_cache(self, prompts: str | List[str], images: Optional[List] = None, **kwargs) -> List[str]:
        """
        Generate text from the given prompts and images.
        
        Parameters:
            prompts (str or list): The prompts to generate text for. A list of prompts can be provided for batch generation.
            images (list): A list of PIL images to use for the prompts. Defaults to None.
            kwargs: Additional keyword arguments to pass to the model for inference (e.g. temperature, top_p, max_tokens).
            
        Returns:
            list: The generated text for each prompt. Includes the probability of the "Yes" and "No" tokens if return_probs is True.
        """
        pass

    def _generate(self, prompts, images=None, **kwargs):
        """
        Generate text from the given prompts and images, using caching to avoid redundant computations.
        
        Parameters:
            prompts (str or list): The prompts to generate text for. A list of prompts can be provided for batch generation.
            images (list): A list of PIL images to use for the prompts. Defaults to None.
            kwargs: Additional keyword arguments to pass to the model for inference (e.g. temperature, top_p, max_tokens).
            
        Returns:
            list: The generated text for each prompt. Includes the probability of the "Yes" and "No" tokens if return_probs is True.
        """
        if not self.use_cache:
            self._vprint("Cache is disabled, generating responses without caching.")
            self._increment_vlm_prompt_count(prompts)
            return self._generate_no_cache(prompts, images=images, **kwargs)
        responses, prompts_for_model, images_for_model = self.cache.get_responses(prompts, images=images)
        if len(prompts_for_model) == 0:
            return responses
                
        prompts = [ prompts_for_model[i] for i in sorted(prompts_for_model.keys()) ]
        images = [ images_for_model[i] for i in sorted(images_for_model.keys()) ] if images is not None else None
        self._vprint(f"Generating {len(prompts)} responses.")
        self._increment_vlm_prompt_count(prompts)
        generated_texts = self._generate_no_cache(prompts, images=images, **kwargs)
        # Match the generated texts back to the original order of prompts
        for i, text in zip(sorted(prompts_for_model.keys()), generated_texts):
            responses[i] = text
            if self.use_cache:
                img_key = tuple(img.tobytes() for img in images_for_model[i]) if len(images_for_model) > 0 else None
                if self.use_vector_cache:
                    self.cache.set((prompts_for_model[i], img_key), text, only_if_not_full=True)
                else:
                    self.cache.set((prompts_for_model[i], img_key), text)

        self._vprint(f"Generated {len(generated_texts)} responses with vLLM model {self.model_id} on device {self.device}.")
        return responses

    def _increment_vlm_prompt_count(self, prompts: str | List[str]) -> None:
        """Increment the prompt counter for each actual VLM invocation."""
        if isinstance(prompts, str):
            prompt_count = 1
        else:
            prompt_count = len(prompts)
        self.vlm_generation_prompt_count += prompt_count
        self.vlm_generation_call_count += 1
        return

    def populate_prompts(
        self,
        config: DictConfig,
        states: torch.Tensor,
        infos: Optional[dict] = None,
        history: Optional[List] = None,
    ) -> tuple[List[str], Optional[List[List[Image.Image]]]]:
        """
        Populate the prompts with the current state of the environment.
        
        Args:
            config (DictConfig): The configuration object containing the model and environment settings.
            states (torch.Tensor): The current states (batch of observations) from the environment. Can be symbolic or image-based.
            infos (Optional[dict]): Additional information from the environment (required for ALFWorld to populate the prompts).
            history (Optional[list]): Per-env deque/list of dicts with keys {"obs", "action"}.
            
        Returns:
            list[str]: The populated prompts for each state in the batch.
            list[list[Image.Image]] or None: 2D list of shape (n_prompts, n_images_per_prompt) if any
        """
        prompt, image_mode = get_prompt(config)

        states_images = states['image'] if isinstance(states, dict) and 'image' in states else states
        batch_size = states_images.shape[0] if isinstance(states, dict) else states.shape[0]
        stack_size = states_images.shape[1] if len(states_images.shape) == 5 else 1
        images_raw = states['image'] if isinstance(states, dict) and 'image' in states else states

        def _extract_frame(single_state):
            imgs = single_state["image"] if isinstance(single_state, dict) and "image" in single_state else single_state
            if torch.is_tensor(imgs):
                frame = imgs[-1] if imgs.ndim == 4 else imgs
            else:
                frame = imgs[-1] if hasattr(imgs, "ndim") and imgs.ndim == 4 else imgs
            return tensor_to_pil(frame)

        history_marker = "## History:"
        history_enabled = history is not None
        history_text_only = bool(getattr(config.training.guidance_model, "just_text_history", False))
        if history_enabled and history_marker not in prompt:
            raise ValueError(f"Prompt must contain '{history_marker}' when history is enabled.")

        def _format_action(action_val: int) -> str:
            desc = get_action_description(config.env.env_id, action_val)
            return f"{action_val} ({desc})" if desc else f"{action_val}"

        def _insert_history(p: str, history_lines: str) -> str:
            if history_marker not in p:
                return p
            return p.replace(history_marker, f"{history_marker}\n{history_lines}", 1)

        if image_mode:
            # Image prompts: keep current observation images, insert history text, and optionally prepend history images.
            prompts = []
            images = []
            if stack_size > 1:
                images_prompt = (
                    f"The observation is provided as a stack of {stack_size} images, representing the state of the world "
                    f"from the timestep t-{stack_size - 1} to t (the current timestep)."
                )
                for i in range(stack_size - 1):
                    images_prompt += f"\n- Timestep t-{stack_size - 1 - i}: {{image}}"
                images_prompt += "\n- Current timestep: {image}"
                prompt_base = prompt.replace("{image}", images_prompt)
            else:
                prompt_base = prompt

            for i in range(batch_size):
                current_imgs = (
                    [tensor_to_pil(images_raw[i][s]) for s in range(stack_size)]
                    if stack_size > 1
                    else [tensor_to_pil(images_raw[i])]
                )
                hist_entries = history[i] if history is not None else []
                if hist_entries:
                    if history_text_only:
                        hist_lines = "\n".join(
                            f"- t-{len(hist_entries) - idx}: action={_format_action(entry['action'])}"
                            for idx, entry in enumerate(hist_entries)
                        )
                        hist_imgs = []
                    else:
                        hist_lines = "\n".join(
                            f"- t-{len(hist_entries) - idx}: {{image}} -> action={_format_action(entry['action'])}"
                            for idx, entry in enumerate(hist_entries)
                        )
                        hist_imgs = [_extract_frame(entry["obs"]) for entry in hist_entries]
                else:
                    hist_lines = "no previous history available."
                    hist_imgs = []

                prompt_i = _insert_history(prompt_base, hist_lines)

                if isinstance(states, dict) and 'mission' in states:
                    # If the environment provides a mission, include it in the prompt.
                    missions = states['mission']
                    assert len(missions) == batch_size, "Missions must match the batch size of the states."

                    # Special handling for ALFWorld to include admissible_commands, text_observation from infos
                    if infos is not None and config.env.env_id.startswith("Alfworld"):
                        admissible_commands = infos.get('admissible_commands', [[] for _ in range(batch_size)])
                        admissible_commands_strs = []
                        for cmds in admissible_commands:
                            cmds_str = "\n".join([f"{i}: {cmd}" for i, cmd in enumerate(cmds)])
                            admissible_commands_strs.append(cmds_str)

                        prompt_i = prompt_i.format(
                            mission=missions[i],
                            image="{image}",
                            admissible_commands=admissible_commands_strs[i],
                        )
                    else:
                        if "direction" in states:
                            from src.environments.text_descriptions import MINIGRID_DIRECTIONS
                            direction = MINIGRID_DIRECTIONS[states['direction'][i].item()]
                            prompt_i = prompt_i.format(mission=missions[i], image="{image}", direction=direction)
                        else:
                            prompt_i = prompt_i.format(mission=missions[i], image="{image}")

                prompts.append(prompt_i)
                images.append(([] if history_text_only else hist_imgs) + current_imgs)
        else:
            # Text prompts: insert history after the marker and format the current observation.
            images = None

            if config.env.prompt.use_text_description:
                from src.environments.text_descriptions import get_text_obs
                try:
                    states = get_text_obs(states, config)
                except NotImplementedError as e:
                    import warnings
                    warnings.warn("Text description for the environment is not implemented. Using raw observation tensor instead.")
                    states = [f"{state.tolist()}" for state in states]

            prompts = []
            for i, state in enumerate(states):
                hist_entries = history[i] if history is not None else []
                if hist_entries:
                    if history_text_only:
                        hist_lines = "\n".join(
                            f"- t-{len(hist_entries) - idx}: action={_format_action(entry['action'])}"
                            for idx, entry in enumerate(hist_entries)
                        )
                    else:
                        hist_lines = "\n".join(
                            f"- t-{len(hist_entries) - idx}: obs={entry['obs']} -> action={_format_action(entry['action'])}"
                            for idx, entry in enumerate(hist_entries)
                        )
                else:
                    hist_lines = "no previous history available."
                prompt_i = prompt.format(observation=state)
                prompt_i = _insert_history(prompt_i, hist_lines)
                prompts.append(prompt_i)
        return prompts, images
        
    def provide_guidance(self, prompts: torch.Tensor, images: Optional[List[Image.Image]] = None, **kwargs) -> Dict[str, List[str | Image.Image]]:
        """
        Provide guidance (choose an action) based on the current state of the environment using a VLM.
        
        Args:
            config (DictConfig): The configuration object containing the model and environment settings.
            states (torch.Tensor): The current states (batch of observations) from the environment. Can be symbolic or image-based.
            **kwargs: Additional arguments for inference (e.g. temperature, top_p).
            
        Returns:
            Dict[str, List[str | Image.Image]]: A dictionary mapping from "responses" to the generated responses, "prompts" to the prompts used for generation, and "images" to the images (if any) used as input.
        """

        responses = self._generate(prompts, images=images, **kwargs)
        return {"responses": responses, "prompts": prompts, "images": images}

    def provide_reward_shaping_preferences(
        self,
        difference_prompts: List[str],
        preference_prompt_templates: List[str],
        *,
        difference_images: Optional[List[List[Image.Image]]] = None,
        difference_kwargs: Optional[Dict[str, Any]] = None,
        preference_kwargs: Optional[Dict[str, Any]] = None,
        max_batch_size: int = 8,
    ) -> Dict[str, List[str]]:
        """Query the VLM for reward-shaping preferences using a two-step prompt pipeline.

        The caller prepares fully formatted difference prompts (with images) and preference prompt
        templates containing an ``{analysis}`` placeholder. This helper first collects the
        intermediate analyses, then injects them into the preference templates before requesting the
        final preference labels.

        Args:
            difference_prompts: Batched prompts asking the model to compare two observations.
            preference_prompt_templates: Batched preference prompts containing an ``{analysis}``
                placeholder that will be replaced with the generated analysis text.
            difference_images: Optional batched image payloads aligned with ``difference_prompts``.
            difference_kwargs: Optional per-stage generation kwargs for the analysis step.
            preference_kwargs: Optional per-stage generation kwargs for the preference step.
            max_batch_size: Maximum number of prompts processed per generation batch.

        Returns:
            dict: With keys ``analysis`` (list[str]), ``preference_responses`` (list[str]),
            ``filled_preference_prompts`` (list[str]).
        """

        if len(difference_prompts) != len(preference_prompt_templates):
            raise ValueError(
                "difference_prompts and preference_prompt_templates must have the same length"
            )

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        difference_kwargs = difference_kwargs or {}
        preference_kwargs = preference_kwargs or {}

        total_prompts = len(difference_prompts)
        batch_count = len(range(0, total_prompts, max_batch_size))

        analysis_responses: List[str] = []
        for start in tqdm(
            range(0, total_prompts, max_batch_size),
            desc="Reward shaping analysis",
            total=batch_count,
        ):
            end = min(start + max_batch_size, total_prompts)
            batch_prompts = difference_prompts[start:end]
            batch_images = (
                difference_images[start:end] if difference_images is not None else None
            )
            batch_responses = self._generate(
                batch_prompts,
                images=batch_images,
                **difference_kwargs,
            )
            analysis_responses.extend(batch_responses)

        filled_preference_prompts = [
            template.replace("{analysis}", analysis_responses[idx])
            for idx, template in enumerate(preference_prompt_templates)
        ]

        preference_responses: List[str] = []
        for start in tqdm(
            range(0, total_prompts, max_batch_size),
            desc="Reward shaping preference",
            total=batch_count,
        ):
            end = min(start + max_batch_size, total_prompts)
            batch_prompts = filled_preference_prompts[start:end]
            batch_responses = self._generate(
                batch_prompts,
                images=None,
                **preference_kwargs,
            )
            preference_responses.extend(batch_responses)

        return {
            "analysis": analysis_responses,
            "preference_responses": preference_responses,
            "filled_preference_prompts": filled_preference_prompts,
        }

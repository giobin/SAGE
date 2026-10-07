"""Re-implementation of RL-VLM-F reward shaping utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import deque

import hydra
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from gymnasium.spaces import Dict as GymDict
from tqdm import tqdm

from src.config import EnvironmentConfig, PromptConfig, RewardModelConfig, RewardShapingConfig
from src.utils import detach_to_cpu, tensor_to_pil
from src.environments.common import _is_minigrid


def _fanin_init(tensor: torch.Tensor) -> torch.Tensor:
    """Initializer matching the original RL-VLM-F fan-in scheme."""

    size = tensor.size()
    if tensor.ndim < 2:
        raise ValueError("fanin_init requires a tensor with at least two dimensions")
    if tensor.ndim == 2:
        fan_in = size[0]
    else:
        fan_in = int(np.prod(size[1:]))
    bound = 1.0 / np.sqrt(fan_in)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def _resolve_initializer(name: str) -> Any:
    if not name:
        return nn.init.xavier_uniform_
    norm = name.lower()
    if norm == "fanin":
        return _fanin_init
    if hasattr(nn.init, name):
        return getattr(nn.init, name)
    if hasattr(nn.init, norm):
        return getattr(nn.init, norm)
    raise ValueError(f"Unsupported initializer '{name}' for reward model")


def _resolve_activation(name: Optional[str], default: str) -> nn.Module:
    target = name or default
    norm = target.lower()
    if norm == "identity":
        return nn.Identity()
    if hasattr(nn, target):
        activation_cls = getattr(nn, target)
    elif hasattr(nn, target.capitalize()):
        activation_cls = getattr(nn, target.capitalize())
    else:
        raise ValueError(f"Unsupported activation '{target}' for reward model")
    if not issubclass(activation_cls, nn.Module):
        raise ValueError(f"Activation '{target}' is not a torch.nn.Module subclass")
    return activation_cls()


class RewardModelCNN(nn.Module):
    """Configurable CNN used to approximate reward preferences."""

    def __init__(self, input_shape: Tuple[int, int, int], cfg: RewardModelConfig, use_text: bool = False) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.cfg = cfg
        self.use_text = use_text

        if not (
            len(cfg.conv_kernel_sizes)
            == len(cfg.conv_channels)
            == len(cfg.conv_strides)
            == len(cfg.conv_paddings)
        ):
            raise ValueError(
                "Reward model convolution configuration lists must have equal length"
            )

        self.hidden_activation = _resolve_activation(cfg.hidden_activation, "ReLU")
        self.output_activation = _resolve_activation(cfg.output_activation, "Tanh")
        initializer = _resolve_initializer(cfg.hidden_init)

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()

        in_channels = channels
        for out_channels, kernel_size, stride, padding in zip(
            cfg.conv_channels,
            cfg.conv_kernel_sizes,
            cfg.conv_strides,
            cfg.conv_paddings,
        ):
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            initializer(conv.weight)
            nn.init.zeros_(conv.bias)
            self.conv_layers.append(conv)
            self.conv_norm_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        with torch.no_grad():
            probe = torch.zeros(1, channels, height, width)
            for conv in self.conv_layers:
                probe = conv(probe)
        flattened = int(np.prod(probe.shape[1:]))

        # Optional text embedding for MiniGrid mission conditioning
        if self.use_text:
            from src.agents.ac_cnn_text import SimpleTokenizer
            self._text_embedding_size = 32
            self._tokenizer = SimpleTokenizer(vocab_size=500)
            self._word_embedding = nn.Embedding(500, 32, padding_idx=0)
            self._text_rnn = nn.GRU(32, self._text_embedding_size, batch_first=True)

        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        last_dim = flattened + (self._text_embedding_size if self.use_text else 0)
        for hidden_dim in cfg.hidden_sizes:
            fc = nn.Linear(last_dim, hidden_dim)
            fc.weight.data.uniform_(-cfg.init_w, cfg.init_w)
            fc.bias.data.uniform_(-cfg.init_w, cfg.init_w)
            self.fc_layers.append(fc)
            self.fc_norm_layers.append(nn.BatchNorm1d(hidden_dim))
            last_dim = hidden_dim

        self.output_layer = nn.Linear(last_dim, cfg.output_size)
        self.output_layer.weight.data.uniform_(-cfg.init_w, cfg.init_w)
        self.output_layer.bias.data.uniform_(-cfg.init_w, cfg.init_w)

    def _embed_missions(self, missions: List[str]) -> torch.Tensor:
        """Encode mission strings into a fixed-size embedding via GRU."""
        tokenized = [self._tokenizer.encode(m) for m in missions]
        padded = self._tokenizer.pad_sequences(tokenized)
        device = next(self.parameters()).device
        token_ids = torch.tensor(padded, dtype=torch.long, device=device)
        embedded = self._word_embedding(token_ids)
        _, hidden = self._text_rnn(embedded)
        return hidden[-1]  # (batch, text_embedding_size)

    def forward(self, inputs: torch.Tensor, missions: Optional[List[str]] = None) -> torch.Tensor:
        x = inputs
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.cfg.batch_norm_conv:
                x = self.conv_norm_layers[idx](x)
            x = self.hidden_activation(x)

        x = x.view(x.size(0), -1)

        if self.use_text:
            if missions is None:
                raise ValueError("RewardModelCNN with use_text=True requires missions argument")
            text_embed = self._embed_missions(missions)
            x = torch.cat([x, text_embed], dim=-1)

        for idx, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.cfg.batch_norm_fc:
                x = self.fc_norm_layers[idx](x)
            x = self.hidden_activation(x)

        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


@dataclass
class RewardShapingBatch:
    """Snapshot of the data collected during a PPO iteration."""

    rewards: torch.Tensor
    obs: Any
    actions: torch.Tensor
    dones: torch.Tensor


@dataclass
class RewardShapingTransition:
    """Snapshot of a single transition kept in the global preference buffer."""

    observation: Any
    reward: float
    done: bool


@dataclass
class RewardShapingPreferenceSample:
    """Structured record of a VLM preference query result."""

    first: RewardShapingTransition
    second: RewardShapingTransition
    label: int
    analysis: str
    raw_response: str
    filled_prompt: str
    difference_prompt: str


class RewardShapingManager:
    """Coordinates reward-shaping updates driven by VLM preferences."""

    def __init__(
        self,
        cfg: RewardShapingConfig,
        env_cfg: EnvironmentConfig,
        prompt_cfg: PromptConfig,
        envs,
        experiment_dir: str,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.prompt_cfg = prompt_cfg
        self.experiment_dir = experiment_dir
        self.device = device

        base_prompt_path = prompt_cfg.reward_shaping_prompts_path or prompt_cfg.prompts_path
        self.prompts_path = Path(hydra.utils.to_absolute_path(base_prompt_path))
        self.goal_description = (prompt_cfg.reward_shaping_goal or "").strip()

        self._difference_prompt_template = self._load_prompt_template(
            self.prompts_path / prompt_cfg.reward_shaping_difference_prompt
        )
        self._preference_prompt_template = self._load_prompt_template(
            self.prompts_path / prompt_cfg.reward_shaping_preference_prompt
        )

        self._pending_batches: List[RewardShapingBatch] = []

        self.transition_capacity = getattr(self.cfg, "buffer_capacity", 10000)
        self.dataset_capacity = getattr(self.cfg, "dataset_capacity", 5000)

        self._transition_buffer: deque[RewardShapingTransition] = deque(maxlen=self.transition_capacity)
        self._preference_dataset: deque[RewardShapingPreferenceSample] = deque(maxlen=self.dataset_capacity)
        self._total_feedback: int = 0

        self._last_update_global_step: Optional[int] = None
        self._reward_model_ensemble = nn.ModuleList()
        self._reward_model_optimizer: Optional[torch.optim.Optimizer] = None
        self._reward_model_input_shape: Optional[Tuple[int, int, int]] = None
        self._reward_model_param_count: float = 0.0

        if not self.env_cfg.image_obs and not _is_minigrid(self.env_cfg.env_id):
            raise NotImplementedError("Reward shaping currently supports only image observations.")

        rm_cfg = self.cfg.reward_model
        channels, height, width = self._infer_image_shape_from_env(envs)
        self._reward_model_input_shape = (channels, height, width)
        self._reward_model_use_text = _is_minigrid(self.env_cfg.env_id)
        for _ in range(max(1, rm_cfg.ensemble_size)):
            model = RewardModelCNN(self._reward_model_input_shape, rm_cfg, use_text=self._reward_model_use_text).to(self.device)
            self._reward_model_ensemble.append(model)

        parameters = chain.from_iterable(model.parameters() for model in self._reward_model_ensemble)
        self._reward_model_optimizer = torch.optim.Adam(parameters, lr=rm_cfg.learning_rate)
        self._reward_model_param_count = float(
            sum(param.numel() for param in self._reward_model_ensemble.parameters())
        )
        print(f"Reward model ensemble initialized with {self._reward_model_param_count} parameters.")

        self.env_description = (getattr(prompt_cfg, "reward_shaping_env_description", "") or "").strip()
        self.goal_description = (getattr(prompt_cfg, "reward_shaping_goal", "") or "").strip()
    
    def process_iteration(
        self,
        batch: RewardShapingBatch,
        iteration: int,
        global_step: int,
        vlm_guidance_model: Optional[Any] = None,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Stage batch data, optionally trigger updates, and report diagnostics."""
        if not self.cfg.enabled:
            return None, {}
        
        # Transitions should be accumulated regardless of update triggering
        self._pending_batches.append(batch)
        self._accumulate_transitions(batch)

        diagnostics: Dict[str, float] = {
            "reward_shaping/pending_batches": float(len(self._pending_batches)),
            "reward_shaping/alpha": float(self.cfg.alpha),
            "reward_shaping/buffer_size": float(len(self._transition_buffer)),
            "reward_shaping/dataset_size": float(len(self._preference_dataset)),
            "reward_shaping/reward_model_params": self._reward_model_param_count,
            "reward_shaping/update_triggered": 0.0,
            "reward_shaping/last_update_step": float(self._last_update_global_step or -1),
            "reward_shaping/total_feedback": float(self._total_feedback),
        }

        # Only update on specified intervals
        if self._should_update(global_step):
            if self.cfg.max_feedback is not None and self._total_feedback >= self.cfg.max_feedback:
                print(
                    "Max feedback reached; skipping reward shaping update "
                    f"({self._total_feedback} >= {self.cfg.max_feedback})."
                )
                diagnostics["reward_shaping/feedback_cap_reached"] = 1.0
                diagnostics["reward_shaping/update_triggered"] = 0.0
                self._pending_batches.clear()
            else:
                max_pairs = self.cfg.max_pairs_per_update
                if self.cfg.max_feedback is not None:
                    remaining = self.cfg.max_feedback - self._total_feedback
                    max_pairs = remaining if max_pairs is None else min(max_pairs, remaining)

                sampled_pairs = self._sample_transition_pairs(max_pairs=max_pairs)
                print(f"Reward shaping update at iteration {iteration} with {len(sampled_pairs)} sampled transition pairs.")
                diagnostics["reward_shaping/sample_pairs"] = float(len(sampled_pairs))
                if max_pairs is not None:
                    self._total_feedback += int(max_pairs)
                    diagnostics["reward_shaping/total_feedback"] = float(self._total_feedback)

                if not sampled_pairs:
                    print("No sampled transition pairs available for reward shaping update.")
                    diagnostics["reward_shaping/preferences_total"] = 0.0
                    diagnostics["reward_shaping/preferences_valid"] = 0.0
                    diagnostics["reward_shaping/preferences_invalid"] = 0.0
                    diagnostics["reward_shaping/preferences_skipped_pairs"] = 0.0
                    diagnostics["reward_shaping/preferences_response_count"] = 0.0
                else:
                    (
                        difference_prompts,
                        preference_templates,
                        difference_images,
                        retained_pairs,
                        skipped_pairs,
                    ) = self._prepare_reward_shaping_queries(sampled_pairs)

                    diagnostics["reward_shaping/preferences_skipped_pairs"] = float(skipped_pairs)
                    diagnostics["reward_shaping/preferences_total"] = float(len(retained_pairs))

                    if retained_pairs:
                        try:
                            print(f"Querying VLM for {len(retained_pairs)} reward shaping preferences...")
                            query_outputs = vlm_guidance_model.provide_reward_shaping_preferences(
                                difference_prompts,
                                preference_templates,
                                difference_images=difference_images,
                            )
                        except NotImplementedError:
                            print("VLM guidance model does not support reward shaping preferences (likely an oracle).")
                            diagnostics["reward_shaping/vlm_missing"] = 1.0
                            diagnostics["reward_shaping/preferences_valid"] = 0.0
                            diagnostics["reward_shaping/preferences_invalid"] = float(len(retained_pairs))
                            diagnostics["reward_shaping/preferences_response_count"] = 0.0
                        else:
                            preference_responses = query_outputs.get("preference_responses", [])
                            analysis = query_outputs.get("analysis", [])
                            filled_prompts = query_outputs.get("filled_preference_prompts", [])

                            parsed_labels = [
                                self._parse_preference_response(resp) for resp in preference_responses
                            ]
                            diagnostics["reward_shaping/preferences_response_count"] = float(
                                len(parsed_labels)
                            )

                            valid_count = sum(label is not None for label in parsed_labels)
                            total_pairs = len(retained_pairs)
                            invalid_count = max(0, total_pairs - valid_count)
                            diagnostics["reward_shaping/preferences_valid"] = float(valid_count)
                            diagnostics["reward_shaping/preferences_invalid"] = float(
                                invalid_count
                            )

                            for idx, label in enumerate(parsed_labels):
                                if label is None:
                                    continue

                                first, second = retained_pairs[idx]
                                sample = RewardShapingPreferenceSample(
                                    first=first,
                                    second=second,
                                    label=label,
                                    analysis=analysis[idx] if idx < len(analysis) else "",
                                    raw_response=preference_responses[idx],
                                    filled_prompt=filled_prompts[idx] if idx < len(filled_prompts) else "",
                                    difference_prompt=difference_prompts[idx]
                                    if idx < len(difference_prompts)
                                    else "",
                                )
                                self._preference_dataset.append(sample)

                            print(f"Added {valid_count} valid preference samples to the dataset.")
                            diagnostics["reward_shaping/update_triggered"] = 1.0
                            diagnostics["reward_shaping/last_update_step"] = float(global_step)
                    else:
                        print("No retained pairs after preparing reward shaping queries.")
                        diagnostics["reward_shaping/preferences_valid"] = 0.0
                        diagnostics["reward_shaping/preferences_invalid"] = 0.0
                        diagnostics["reward_shaping/preferences_response_count"] = 0.0
                        diagnostics["reward_shaping/update_triggered"] = 0.0

                self._pending_batches.clear()
                self._last_update_global_step = global_step

                training_payload = self._prepare_preference_training_data()
                if training_payload is None:
                    diagnostics["reward_shaping/preferences_dataset"] = 0.0
                    diagnostics["reward_shaping/train_skipped"] = 1.0
                else:
                    if self._reward_model_use_text:
                        first_batch, second_batch, labels, skipped_samples, first_m, second_m = training_payload
                    else:
                        first_batch, second_batch, labels, skipped_samples = training_payload
                        first_m, second_m = None, None
                    dataset_size = labels.numel()
                    diagnostics["reward_shaping/preferences_dataset"] = float(dataset_size)
                    diagnostics["reward_shaping/preferences_dropped_for_training"] = float(skipped_samples)
                    train_metrics = self._train_reward_model(first_batch, second_batch, labels, first_m, second_m)
                    diagnostics.update(train_metrics)
        else:
            diagnostics["reward_shaping/update_triggered"] = 0.0

        # Shaped rewards are always computed since they need to override the env rewards
        shaped_rewards: Optional[torch.Tensor] = None
        if self.cfg.relabel_rollout:
            predicted_rewards = self._predict_shaped_rewards(batch)
            if predicted_rewards is not None:
                diagnostics["reward_shaping/predicted_reward_mean"] = float(
                    predicted_rewards.mean().item()
                )
                diagnostics["reward_shaping/predicted_reward_std"] = float(
                    predicted_rewards.std(unbiased=False).item()
                )
            env_rewards = batch.rewards.to(self.device)
            shaped_rewards = self.mix_rewards(env_rewards, predicted_rewards)
            diagnostics["reward_shaping/shaped_reward_mean"] = float(
                shaped_rewards.mean().item()
            )
            diagnostics["reward_shaping/shaped_reward_std"] = float(
                shaped_rewards.std(unbiased=False).item()
            )

        return shaped_rewards, diagnostics

    def _should_update(self, global_step: int) -> bool:
        interval = max(1, int(self.cfg.update_interval))
        if self._last_update_global_step is None:
            return global_step >= interval
        return (global_step - self._last_update_global_step) >= interval

    def mix_rewards(self, env_rewards: torch.Tensor, shaped_rewards: torch.Tensor) -> torch.Tensor:
        """Interpolates environment and shaped rewards according to alpha."""
        return (1.0 - self.cfg.alpha) * env_rewards + self.cfg.alpha * shaped_rewards

    def _accumulate_transitions(self, batch: RewardShapingBatch) -> None:
        """Copy the current PPO batch into the global transition buffer."""
        rewards_cpu = batch.rewards.detach().cpu()
        dones_cpu = batch.dones.detach().cpu()

        num_steps, num_envs = rewards_cpu.shape
        for t in range(num_steps):
            for env_idx in range(num_envs):
                reward = float(rewards_cpu[t, env_idx].item())
                done = bool(dones_cpu[t, env_idx].item())
                observation = self._slice_observation(batch.obs, t, env_idx)

                transition = RewardShapingTransition(
                    observation=observation,
                    reward=reward,
                    done=done,
                )
                self._transition_buffer.append(transition)

    def _slice_observation(self, obs: Any, step: int, env_idx: int) -> Any:
        """Extract a single observation from the batch in a device-agnostic way."""
        if isinstance(obs, dict):
            return {
                key: detach_to_cpu(value[step, env_idx])
                for key, value in obs.items()
            }
        return detach_to_cpu(obs[step, env_idx])

    def _sample_transition_pairs(
        self, max_pairs: Optional[int] = None
    ) -> Sequence[tuple[RewardShapingTransition, RewardShapingTransition]]:
        """Return up to `max_pairs` random, distinct transition pairs from the buffer."""
        buffer_len = len(self._transition_buffer)
        if buffer_len < 2:
            return []

        pair_limit = max_pairs if max_pairs is not None else self.cfg.max_pairs_per_update
        sample_size = min(buffer_len, pair_limit * 2) if pair_limit is not None else 64 # default to 64 (32 pairs) to avoid calling the VLM with too many samples
        # ensure even number of samples
        if sample_size % 2 == 1:
            sample_size -= 1
        if sample_size < 2:
            return []

        indices = np.random.choice(buffer_len, size=sample_size, replace=False)
        temp_buffer_list = list(self._transition_buffer) 
        
        pairs = []
        # Shuffle selected indices to pair them up randomly
        np.random.shuffle(indices)
        
        for idx in range(0, sample_size, 2):
            idx_a = indices[idx]
            idx_b = indices[idx + 1]
            
            a = temp_buffer_list[idx_a]
            b = temp_buffer_list[idx_b]
            
            if a is b:
                continue
            pairs.append((a, b))
                
        return pairs

    def _load_prompt_template(self, path: Path) -> str:
        """Read a prompt template from disk and perform minimal normalization."""

        if not path.exists():
            raise FileNotFoundError(f"Reward shaping prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _prepare_reward_shaping_queries(
        self,
        pairs: Sequence[tuple[RewardShapingTransition, RewardShapingTransition]],
    ) -> tuple[
        List[str],
        List[str],
        Optional[List[List[Image.Image]]],
        List[tuple[RewardShapingTransition, RewardShapingTransition]],
        int,
    ]:
        """Build prompt payloads for VLM preference queries."""
        
        def _get_minigrid_goal(first: RewardShapingTransition, second: RewardShapingTransition):
            mission_first = first.observation.get("mission", None)
            mission_second = second.observation.get("mission", None)
            if mission_first is None or mission_second is None:
                raise ValueError("No mission found in minigrid observation for reward shaping.")
            goal_str = f"The goal in the first observation is: '{mission_first.strip()}'.\nThe goal in the second observation is: '{mission_second.strip()}'."
            return goal_str

        difference_prompts: List[str] = []
        preference_templates: List[str] = []
        difference_images: List[List[Image.Image]] = []
        retained_pairs: List[tuple[RewardShapingTransition, RewardShapingTransition]] = []
        skipped_pairs = 0

        for i, (first, second) in enumerate(pairs):
            first_images = self._transition_to_image_payload(first)
            second_images = self._transition_to_image_payload(second)

            if not first_images or not second_images:
                skipped_pairs += 1
                continue
            
            # Skip if the two images are identical using L2 norm
            diff = np.linalg.norm(
                np.array(first_images[-1], dtype=np.float32) - np.array(second_images[-1], dtype=np.float32)
            )
            if diff < 1e-3:
                skipped_pairs += 1
                continue

            difference_prompt = self._difference_prompt_template
            difference_prompt = difference_prompt.replace("{env_description}", self.env_description)
            goal_description = _get_minigrid_goal(first, second) if _is_minigrid(self.env_cfg.env_id) else self.goal_description
            difference_prompt = difference_prompt.replace("{goal_description}", goal_description)
            difference_prompt = difference_prompt.replace("{image_1}", "{image}")
            difference_prompt = difference_prompt.replace("{image_2}", "{image}")

            preference_template = self._preference_prompt_template
            if goal_description:
                preference_template = preference_template.replace(
                    "{goal_description}", goal_description
                )
            else:
                preference_template = preference_template.replace(
                    "{goal_description}", ""
                )

            # CardMaze specific
            if "n_cards" in self.env_cfg:
                difference_prompt = difference_prompt.replace("{n_cards}", str(self.env_cfg.n_cards))
                preference_template = preference_template.replace("{n_cards}", str(self.env_cfg.n_cards))
            
            difference_prompts.append(difference_prompt)
            preference_templates.append(preference_template)
            difference_images.append([first_images[-1], second_images[-1]])
            retained_pairs.append((first, second))

        return difference_prompts, preference_templates, difference_images, retained_pairs, skipped_pairs

    def _preference_label_to_class(self, label: int) -> Optional[int]:
        """Map VLM label {-1, 0, 1} to CrossEntropy class index."""

        if label == 0:
            return 0
        if label == 1:
            return 1
        return None

    def _observation_to_model_tensor(self, observation: Any) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[str]]]:
        """Convert a stored observation into a normalized CHW image tensor.

        When the reward model uses text conditioning (MiniGrid), returns (tensor, mission_str).
        Otherwise returns just the tensor.
        """

        channels, height, width = self._reward_model_input_shape
        mission = None
        if isinstance(observation, dict):
            if "image" not in observation:
                return (None, None) if self._reward_model_use_text else None
            image_source = observation["image"]
            if self._reward_model_use_text:
                mission = str(observation.get("mission", ""))
        else:
            image_source = observation

        _none = (None, None) if self._reward_model_use_text else None

        tensor = self._ensure_tensor(image_source)
        if tensor is None:
            return _none

        tensor = tensor.float()

        # MEMORY FIX: Handle uint8 restoration
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] not in (channels, 1, 3):
                if tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1)
                else:
                    tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
            elif tensor.shape[0] in (1, 3) and tensor.shape[0] != channels:
                if tensor.shape[-1] == channels:
                    tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        else:
            return _none

        if tensor.shape[0] != channels or tensor.shape[1] != height or tensor.shape[2] != width:
            if tensor.numel() != channels * height * width:
                return _none
            tensor = tensor.view(channels, height, width)

        tensor = tensor.clamp(0.0, 1.0)
        return (tensor, mission) if self._reward_model_use_text else tensor

    def _transition_to_model_tensor(self, transition: RewardShapingTransition):
        """Helper that extracts the image tensor (and optionally mission) from a stored transition."""

        return self._observation_to_model_tensor(transition.observation)

    def _prepare_preference_training_data(
        self,
    ) -> Optional[tuple]:
        """Convert stored preference samples into tensors suitable for training.

        Returns (first_batch, second_batch, label_tensor, skipped) or
        (first_batch, second_batch, label_tensor, skipped, first_missions, second_missions)
        when text conditioning is active.
        """

        if not self._preference_dataset or self._reward_model_input_shape is None:
            return None

        first_images: List[torch.Tensor] = []
        second_images: List[torch.Tensor] = []
        first_missions: List[str] = []
        second_missions: List[str] = []
        labels: List[int] = []
        skipped = 0

        if self.cfg.buffer_sample_size is not None:
            dataset_samples = np.random.choice(
                len(self._preference_dataset),
                size=min(len(self._preference_dataset), self.cfg.buffer_sample_size),
                replace=False,
            )
            dataset_iterable = [self._preference_dataset[idx] for idx in dataset_samples]
            print(f"Sampling {len(dataset_iterable)} preference samples for training from dataset of size {len(self._preference_dataset)}.")
        else:
            dataset_iterable = self._preference_dataset
            print(f"Using entire preference dataset for training ({len(self._preference_dataset)} samples).")

        for sample in dataset_iterable:
            class_label = self._preference_label_to_class(sample.label)
            if class_label is None:
                skipped += 1
                continue

            first_result = self._transition_to_model_tensor(sample.first)
            second_result = self._transition_to_model_tensor(sample.second)

            if self._reward_model_use_text:
                first_tensor, first_m = first_result
                second_tensor, second_m = second_result
            else:
                first_tensor = first_result
                second_tensor = second_result

            if first_tensor is None or second_tensor is None:
                skipped += 1
                continue

            first_images.append(first_tensor)
            second_images.append(second_tensor)
            labels.append(class_label)
            if self._reward_model_use_text:
                first_missions.append(first_m)
                second_missions.append(second_m)

        if not labels:
            return None

        first_batch = torch.stack(first_images, dim=0)
        second_batch = torch.stack(second_images, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        if self._reward_model_use_text:
            return first_batch, second_batch, label_tensor, skipped, first_missions, second_missions
        return first_batch, second_batch, label_tensor, skipped

    def _train_reward_model(
        self,
        first_batch: torch.Tensor,
        second_batch: torch.Tensor,
        labels: torch.Tensor,
        first_missions: Optional[List[str]] = None,
        second_missions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """One epoch of supervised updates over the preference dataset."""

        dataset_size = labels.shape[0]
        if dataset_size == 0:
            return {"reward_shaping/train_skipped": 1.0}

        num_members = len(self._reward_model_ensemble)
        ce_loss = nn.CrossEntropyLoss()

        batch_size = min(self.cfg.batch_size, dataset_size)
        benches = int(np.ceil(dataset_size / batch_size))
        num_epochs = max(1, int(getattr(self.cfg, "num_epochs", 1)))

        loss_per_member: List[List[float]] = [[] for _ in range(num_members)]
        correct_per_member = torch.zeros(num_members, dtype=torch.float64)
        total_per_member = torch.zeros(num_members, dtype=torch.float64)
        accuracy_threshold = getattr(self.cfg, "accuracy_threshold", None)
        epochs_completed = 0

        for epoch_idx in range(num_epochs):
            epoch_correct = torch.zeros(num_members, dtype=torch.float64)
            epoch_total = torch.zeros(num_members, dtype=torch.float64)
            permutation = torch.randperm(dataset_size)
            epoch_desc = f"Training reward model epoch {epoch_idx + 1}/{num_epochs}"
            for batch_idx in tqdm(range(benches), desc=epoch_desc):
                start = batch_idx * batch_size
                end = min(start + batch_size, dataset_size)
                indices = permutation[start:end]

                batch_first = first_batch[indices].to(self.device)
                batch_second = second_batch[indices].to(self.device)
                batch_labels = labels[indices].to(self.device)
                batch_first_m = [first_missions[i] for i in indices] if first_missions else None
                batch_second_m = [second_missions[i] for i in indices] if second_missions else None

                self._reward_model_optimizer.zero_grad()
                loss = torch.tensor(0.0, device=self.device)

                for member_idx, model in enumerate(self._reward_model_ensemble):
                    model.train()
                    logits_first = model(batch_first, missions=batch_first_m)
                    logits_second = model(batch_second, missions=batch_second_m)
                    logits = torch.cat([logits_first, logits_second], dim=-1)

                    member_loss = ce_loss(logits, batch_labels)
                    loss = loss + member_loss
                    loss_per_member[member_idx].append(member_loss.item())

                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == batch_labels).sum().double().cpu()
                    correct_per_member[member_idx] += correct
                    total = batch_labels.size(0)
                    total_per_member[member_idx] += total
                    epoch_correct[member_idx] += correct
                    epoch_total[member_idx] += total

                loss.backward()
                self._reward_model_optimizer.step()

            epochs_completed += 1

            if accuracy_threshold is not None and float(accuracy_threshold) >= 0.0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    epoch_accuracies = np.array(
                        [
                            (epoch_correct[idx].item() / epoch_total[idx].item())
                            if epoch_total[idx] > 0 else 0.0
                            for idx in range(num_members)
                        ]
                    )
                epoch_mean_acc = float(np.mean(epoch_accuracies)) if epoch_accuracies.size else 0.0
                print(
                    f"Epoch {epoch_idx + 1} accuracy mean {epoch_mean_acc:.4f}"
                    f" (threshold {float(accuracy_threshold):.4f})."
                )
                if epoch_mean_acc >= float(accuracy_threshold):
                    print("Early stopping reward model training due to reaching accuracy threshold.")
                    break

        avg_loss = float(
            np.mean([np.mean(member_losses) for member_losses in loss_per_member if member_losses])
        ) if any(loss_per_member) else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            accuracies = np.array(
                [
                    (correct_per_member[idx].item() / total_per_member[idx].item())
                    if total_per_member[idx] > 0 else 0.0
                    for idx in range(num_members)
                ]
            )
            print(f"Reward model training completed over {dataset_size} samples with final accuracy mean {np.mean(accuracies) if accuracies.size else 0.0:.4f} and std {np.std(accuracies) if accuracies.size else 0.0:.4f}.")

        metrics: Dict[str, float] = {
            "reward_shaping/train_batches": float(benches * epochs_completed),
            "reward_shaping/train_loss": avg_loss,
            "reward_shaping/train_accuracy_mean": float(np.mean(accuracies) if accuracies.size else 0.0),
            "reward_shaping/train_accuracy_std": float(np.std(accuracies) if accuracies.size else 0.0),
            "reward_shaping/train_skipped": 0.0,
        }
        metrics["reward_shaping/train_epochs"] = float(epochs_completed)

        return metrics

    def _predict_shaped_rewards(self, batch: RewardShapingBatch) -> Optional[torch.Tensor]:
        """Run ensemble inference over the current batch to produce shaped rewards."""

        num_steps, num_envs = batch.rewards.shape
        candidate_images: List[torch.Tensor] = []
        candidate_missions: List[str] = []
        candidate_indices: List[Tuple[int, int]] = []

        for step in tqdm(range(num_steps), desc="Predicting shaped rewards"):
            for env_idx in range(num_envs):
                observation = self._slice_observation(batch.obs, step, env_idx)
                result = self._observation_to_model_tensor(observation)
                if self._reward_model_use_text:
                    tensor, mission = result
                else:
                    tensor = result
                    mission = None
                if tensor is None:
                    continue
                candidate_images.append(tensor)
                candidate_indices.append((step, env_idx))
                if self._reward_model_use_text:
                    candidate_missions.append(mission)

        if not candidate_images:
            return None

        stacked = torch.stack(candidate_images, dim=0).to(self.device)
        missions_arg = candidate_missions if self._reward_model_use_text else None
        with torch.no_grad():
            ensemble_outputs: List[torch.Tensor] = []
            for model in self._reward_model_ensemble:
                model.eval()
                outputs = model(stacked, missions=missions_arg).squeeze(-1)
                ensemble_outputs.append(outputs)
            averaged = torch.stack(ensemble_outputs, dim=0).mean(dim=0)

        shaped = torch.zeros_like(batch.rewards, device=self.device)
        for value, (step, env_idx) in zip(averaged, candidate_indices):
            shaped[step, env_idx] = value
        return shaped

    def _transition_to_image_payload(self, transition: RewardShapingTransition) -> List[Image.Image]:
        """Convert a stored transition observation into a list of PIL images."""

        observation = transition.observation

        if isinstance(observation, dict):
            if "image" in observation:
                image_source = observation["image"]
            else:
                print("No 'image' key found in observation dict for reward shaping.")
                return []
        else:
            image_source = observation

        tensor = self._ensure_tensor(image_source)
        if tensor is None:
            return []

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        frames: List[Image.Image] = []

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim == 3:
            frames.append(tensor_to_pil(tensor))
        elif tensor.ndim == 4:
            for idx in range(tensor.shape[0]):
                frame = tensor[idx]
                if frame.ndim == 2:
                    frame = frame.unsqueeze(0)
                frames.append(tensor_to_pil(frame))
        else:
            print(f"Unsupported tensor shape for reward shaping image extraction: {tensor.shape}")
            return []

        return frames

    def _ensure_tensor(self, value: Any) -> Optional[torch.Tensor]:
        """Convert supported array types into a CPU torch tensor."""

        if isinstance(value, torch.Tensor):
            return value.detach().cpu() 
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, list):
            try:
                return torch.tensor(value, dtype=torch.float32)
            except Exception as e:
                print("Failed to convert list to tensor for reward shaping.")
                print(e)
                return None
        return None

    def _infer_image_shape_from_env(self, envs) -> Tuple[int, int, int]:
        """Infer (C, H, W) from the environment's observation space."""

        space = envs.single_observation_space
        if isinstance(space, GymDict):
            if "image" not in space.spaces:
                raise ValueError("Expected 'image' key in Dict observation space for reward shaping.")
            space = space["image"]

        shape = space.shape
        if len(shape) == 2:
            height, width = shape
            channels = 1
        elif len(shape) == 3:
            if shape[0] in (1, 3):
                channels, height, width = shape
            elif shape[-1] in (1, 3):
                height, width, channels = shape
            else:
                raise ValueError(f"Unsupported 3D image shape for reward shaping: {shape}")
        elif len(shape) == 4:
            if shape[-1] in (1, 3):
                stack, height, width, per_channel = shape
                channels = stack * per_channel
            elif shape[1] in (1, 3):
                stack, per_channel, height, width = shape
                channels = stack * per_channel
            else:
                raise ValueError(f"Unsupported stacked image shape for reward shaping: {shape}")
        else:
            raise ValueError(f"Unsupported observation shape for reward shaping: {shape}")

        return int(channels), int(height), int(width)

    def _parse_preference_response(self, response: str) -> Optional[int]:
        """Extract the preference label (-1, 0, or 1) from a VLM response."""

        if not response:
            return None

        candidates = re.findall(r"-?\d+", response)
        for token in reversed(candidates):
            if token in {"-1", "0", "1"}:
                return int(token)
        return None
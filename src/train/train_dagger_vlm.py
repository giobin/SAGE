import copy
import inspect
import math
import os
import sys
import time
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.config import (
    DaggerBetaScheduleType,
    GPUConfig,
    GuidanceModelType,
    GuidanceType,
    WandBLoggerMode,
    register_configs,
)
from src.environments import make_env
from src.guidance_models.common import extract_action_and_thinking
from src.train.common import (
    append_metrics,
    batch_index,
    evaluate,
    get_gpu_device_map,
    obs_to_tensor,
    save_checkpoint,
)
from src.utils import (
    detach_to_cpu,
    extract_hydra_suffix,
    get_agent_class,
    get_exp_name,
    get_guidance_model_class,
    set_seed,
)
from src.wandb_compat import wandb


def _configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        cpu_value = value.detach().cpu()
        if _is_float_image_tensor(cpu_value):
            return cpu_value.round().clamp(0, 255).to(torch.uint8)
        return cpu_value.clone()
    if isinstance(value, np.ndarray):
        array_value = np.array(value, copy=True)
        if _is_float_image_array(array_value):
            return np.rint(array_value).clip(0, 255).astype(np.uint8)
        return array_value
    if isinstance(value, dict):
        return {k: _clone_cpu(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _is_image_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) >= 3 and (shape[-1] in (1, 3) or shape[-3] in (1, 3))


def _is_float_image_tensor(value: torch.Tensor) -> bool:
    if not torch.is_floating_point(value) or not _is_image_shape(tuple(value.shape)):
        return False
    if value.numel() == 0:
        return False
    min_value = float(value.min().item())
    max_value = float(value.max().item())
    return min_value >= 0.0 and 1.0 < max_value <= 255.0


def _is_float_image_array(value: np.ndarray) -> bool:
    if not np.issubdtype(value.dtype, np.floating) or not _is_image_shape(value.shape):
        return False
    if value.size == 0:
        return False
    min_value = float(np.min(value))
    max_value = float(np.max(value))
    return min_value >= 0.0 and 1.0 < max_value <= 255.0


def _stack_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {k: _stack_values([v[k] for v in values]) for k in first.keys()}
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        try:
            return np.stack(values, axis=0)
        except ValueError:
            return np.array(values, dtype=object)
    if isinstance(first, (str, bytes)):
        return np.array(values, dtype=object)
    try:
        return torch.as_tensor(values)
    except Exception:
        return np.array(values, dtype=object)


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if torch.is_tensor(obs):
        return obs.to(device)
    return obs


def _batch_size_from_obs(obs: Any) -> int:
    if isinstance(obs, dict):
        first = next(iter(obs.values()))
        return int(first.shape[0] if hasattr(first, "shape") else len(first))
    return int(obs.shape[0] if hasattr(obs, "shape") else len(obs))


def _action_to_cpu_tensor(action: Any) -> torch.Tensor:
    if torch.is_tensor(action):
        return action.detach().cpu().clone()
    return torch.as_tensor(action).detach().cpu().clone()


class DaggerReplayDataset:
    def __init__(self, max_size: int, seed: int) -> None:
        if max_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {max_size}.")
        self.max_size = int(max_size)
        self._storage: list[tuple[Any, torch.Tensor]] = []
        self._next_idx = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, obs: Any, action: Any) -> None:
        item = (_clone_cpu(obs), _action_to_cpu_tensor(action))
        if len(self._storage) < self.max_size:
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item
            self._next_idx = (self._next_idx + 1) % self.max_size

    def add_batch(self, obs: Any, actions: torch.Tensor | np.ndarray) -> None:
        batch_size = _batch_size_from_obs(obs)
        for env_idx in range(batch_size):
            self.add(batch_index(obs, env_idx), actions[env_idx])

    def sample(self, batch_size: int) -> tuple[Any, torch.Tensor]:
        if len(self._storage) == 0:
            raise RuntimeError("Cannot sample from an empty DAgger replay dataset.")
        indices = self._rng.integers(0, len(self._storage), size=int(batch_size))
        items = [self._storage[int(i)] for i in indices]
        obs = _stack_values([item[0] for item in items])
        actions = torch.stack([item[1] for item in items], dim=0)
        return obs, actions


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def compute_beta(config: DictConfig, iteration_idx: int) -> float:
    schedule = _enum_value(config.training.beta_schedule)
    beta_start = float(config.training.beta_start)
    beta_end = float(config.training.beta_end)

    if schedule == DaggerBetaScheduleType.exponential.value:
        beta = beta_start * (float(config.training.beta_decay) ** iteration_idx)
        beta = max(beta, beta_end)
    elif schedule == DaggerBetaScheduleType.indicator.value:
        beta = beta_start if iteration_idx == 0 else beta_end
    elif schedule == DaggerBetaScheduleType.linear.value:
        warmup_iters = max(1, int(config.training.beta_warmup_iters))
        frac = min(1.0, iteration_idx / warmup_iters)
        beta = beta_start + frac * (beta_end - beta_start)
    elif schedule == DaggerBetaScheduleType.constant.value:
        beta = beta_start
    else:
        raise ValueError(f"Unknown DAgger beta schedule: {schedule}")

    return float(np.clip(beta, 0.0, 1.0))


def _teacher_kwargs(
    *,
    config: DictConfig,
    envs: gym.vector.SyncVectorEnv,
    obs: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    model_type = _enum_value(config.training.guidance_model.model_type)
    if model_type in (GuidanceModelType.oracle.value, GuidanceModelType.random.value):
        kwargs["envs"] = envs
        kwargs["action_space"] = envs.single_action_space
        if model_type == GuidanceModelType.oracle.value:
            kwargs["random_guidance_prob"] = float(
                config.training.guidance_model.get("random_guidance_prob", 0.0)
            )
    return kwargs


def query_teacher_actions(
    *,
    config: DictConfig,
    envs: gym.vector.SyncVectorEnv,
    vlm_guidance_model: Any,
    obs: Any,
    infos: Optional[dict],
    device: torch.device,
    history: Optional[list[deque]],
) -> torch.Tensor:
    prompts, images = vlm_guidance_model.populate_prompts(
        config,
        states=obs,
        infos=infos,
        history=history,
    )
    guidance_provision = vlm_guidance_model.provide_guidance(
        prompts=prompts,
        images=images,
        **_teacher_kwargs(config=config, envs=envs, obs=obs),
    )
    parsed = [
        extract_action_and_thinking(
            response,
            allowed=envs.single_action_space,
            guidance_type=GuidanceType.action,
        )
        for response in guidance_provision["responses"]
    ]
    teacher_actions = [action for action, *_ in parsed]

    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        return torch.tensor(
            [int(action) for action in teacher_actions],
            dtype=torch.long,
            device=device,
        )
    if isinstance(envs.single_action_space, gym.spaces.Box):
        action_shape = envs.single_action_space.shape
        action_array = np.stack(
            [
                np.asarray(action, dtype=np.float32).reshape(action_shape)
                for action in teacher_actions
            ],
            axis=0,
        )
        return torch.as_tensor(action_array, dtype=torch.float32, device=device)

    raise NotImplementedError(
        f"DAgger-VLM does not support action space {type(envs.single_action_space)}."
    )


def _history_action(action: torch.Tensor | np.ndarray | Any, env_idx: int, discrete_env: bool) -> Any:
    value = action[env_idx] if hasattr(action, "__getitem__") else action
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if discrete_env or value.numel() == 1:
            return int(value.reshape(-1)[0].item()) if discrete_env else float(value.reshape(-1)[0].item())
        return value.reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        if discrete_env or value.size == 1:
            return int(value.reshape(-1)[0]) if discrete_env else float(value.reshape(-1)[0])
        return value.reshape(-1).tolist()
    return int(value) if discrete_env else value


@hydra.main(version_base=None, config_path="../../config", config_name="dagger_qwen35")
def main(config: DictConfig) -> None:
    _configure_output_streams()
    print(OmegaConf.to_yaml(config, resolve=True))

    if not config.vlm_guidance:
        raise ValueError("DAgger-VLM requires vlm_guidance=true so a teacher can label states.")
    if config.training.agent.get("use_memory", False):
        raise NotImplementedError("DAgger-VLM v1 supports feed-forward AC agents only; use_memory=true is not supported.")
    if _enum_value(config.training.guidance_model.guidance_type) != GuidanceType.action.value:
        raise ValueError("DAgger-VLM labels teacher actions only; set training.guidance_model.guidance_type=action.")

    if config.cuda and torch.cuda.is_available():
        rl_gpu_id, vlm_gpu_ids = get_gpu_device_map(config.get("gpu_device_map", {}))
        config.gpu_device_map = GPUConfig(rl_gpu_id=rl_gpu_id, vlm_gpu_ids=vlm_gpu_ids)
    else:
        config.gpu_device_map = GPUConfig(rl_gpu_id=None, vlm_gpu_ids=None)

    env_args = OmegaConf.to_container(config.env.get("env_args", {}), resolve=True)
    shared_memory = env_args.pop("shared_memory", None)
    if config.env.env_id.startswith("Alfworld"):
        shared_memory = False

    config.env.prompt.prompts_path = hydra.utils.to_absolute_path(config.env.prompt.prompts_path)

    Guidance_model_class = get_guidance_model_class(
        _enum_value(config.training.guidance_model.model_type),
        config.training.guidance_model.use_vllm,
    )
    print(
        f"Using guidance model class: {Guidance_model_class.__name__} "
        f"for model {config.training.guidance_model.model_id}"
    )
    model_args = dict(config.training.guidance_model.get("model_args", {}))
    guidance_model_type = _enum_value(config.training.guidance_model.model_type)
    if guidance_model_type == GuidanceModelType.oracle.value and "FrozenLake" in config.env.env_id:
        map_dim = config.env.env_args.get("random_map_dimension", None)
        if map_dim is None:
            map_name = config.env.env_args.get("map_name", None)
            if map_name is None:
                raise ValueError("FrozenLake oracle requires random_map_dimension or map_name.")
            map_dim = [int(n) for n in map_name.split("x") if n.isdigit()]
            assert len(map_dim) == 2, f"Could not parse map dimensions from map_name '{map_name}'."
            model_args["grid_size"] = tuple(map_dim)
        else:
            model_args["grid_size"] = (map_dim, map_dim)

    needs_envs = guidance_model_type == GuidanceModelType.oracle.value
    vlm_guidance_model = None
    if not needs_envs:
        if (
            guidance_model_type == GuidanceModelType.transformers.value
            and config.training.guidance_model.use_vllm
        ):
            model_args["verbose"] = config.logger.vllm_verbose
        vlm_guidance_model = Guidance_model_class(
            model_id=config.training.guidance_model.model_id,
            seed=config.seed,
            gpu_ids=config.get("gpu_device_map", {}).get("vlm_gpu_ids", None),
            env_id=config.env.env_id,
            guidance_model_config=config.training.guidance_model,
            enforce_eager=config.training.guidance_model.enforce_eager,
            **model_args,
        )

    supports_shared_memory = "shared_memory" in inspect.signature(gym.vector.SyncVectorEnv).parameters
    sync_env_kwargs = (
        {"shared_memory": shared_memory}
        if shared_memory is not None and supports_shared_memory
        else {}
    )
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=config.env.env_id,
                idx=i,
                capture_video=config.logger.capture_video,
                use_visual_obs=config.env.image_obs,
                stack_size=config.env.stack_size,
                seed=config.get("seed", None),
                **env_args,
            )
            for i in range(config.env.num_envs)
        ],
        **sync_env_kwargs,
    )

    eval_seed = config.get("eval_seed", config.get("seed", 0) + 10000)
    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=config.env.env_id,
                idx=0,
                capture_video=False,
                use_visual_obs=config.env.image_obs,
                stack_size=config.env.stack_size,
                seed=eval_seed + i,
                **env_args,
            )
            for i in range(config.training.eval_num_envs)
        ],
        **sync_env_kwargs,
    )

    discrete_env = isinstance(envs.single_action_space, gym.spaces.Discrete)
    # Sampled when a teacher answer cannot be parsed into a valid action; seeded for reproducibility
    envs.single_action_space.seed(config.seed)

    if vlm_guidance_model is None:
        model_args["envs"] = envs
        model_args["agent_cfg"] = config.training.agent
        vlm_guidance_model = Guidance_model_class(
            model_id=config.training.guidance_model.model_id,
            seed=config.seed,
            gpu_ids=config.get("gpu_device_map", {}).get("vlm_gpu_ids", None),
            env_id=config.env.env_id,
            guidance_model_config=config.training.guidance_model,
            enforce_eager=config.training.guidance_model.enforce_eager,
            **model_args,
        )

    config.training.batch_size = int(config.env.num_envs * config.env.num_steps)
    config.training.num_iterations = max(
        1,
        int(math.ceil(config.env.total_timesteps / config.training.batch_size)),
    )

    config.exp_name = get_exp_name(config=config)
    hydra_experiment_name = HydraConfig.get().runtime.output_dir
    run_name_suffix = extract_hydra_suffix(hydra_experiment_name)
    run_name = f"{run_name_suffix}_{config.exp_name}"

    wandb.init(
        project=config.logger.wandb_project_name,
        entity=config.logger.wandb_entity,
        sync_tensorboard=False,
        config=OmegaConf.to_container(config, resolve=True, enum_to_str=True),
        name=run_name,
        dir=hydra_experiment_name,
        monitor_gym=True,
        save_code=True,
        mode=config.logger.wandb_mode.value,
    )
    if config.logger.wandb_mode != WandBLoggerMode.disabled:
        if config.logger.wandb_logging_frequency % config.env.num_envs != 0:
            raise ValueError(
                f"config.logger.wandb_logging_frequency ({config.logger.wandb_logging_frequency}) "
                f"must be a multiple of config.env.num_envs ({config.env.num_envs})"
            )

    set_seed(config.seed, config.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    if device.type == "cuda":
        print(f"cuda:{config.gpu_device_map.rl_gpu_id}")
        device = torch.device(f"cuda:{config.gpu_device_map.rl_gpu_id}")
    print(f"DAgger student will be using device: {device}")

    Agent = get_agent_class(config.training.method_name, config.training.agent.agent_type, discrete_env=discrete_env)
    agent = Agent(envs, config.training.agent).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.training.learning_rate, eps=1e-5)
    print(f"n_agent_parameters", sum(p.numel() for p in agent.parameters() if p.requires_grad))

    dataset = DaggerReplayDataset(max_size=config.training.buffer_size, seed=config.seed)
    history_len = config.training.guidance_model.get("history_len", 0)
    vlm_history = [deque(maxlen=history_len) for _ in range(config.env.num_envs)] if history_len > 0 else None
    rollout_log_interval = int(config.training.get("rollout_log_interval", 0))

    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=config.seed)
    next_obs = obs_to_tensor(next_obs, device)
    next_done = torch.zeros(config.env.num_envs, dtype=torch.float32, device=device)

    for iteration in range(1, config.training.num_iterations + 1):
        beta_i = compute_beta(config, iteration - 1)
        expert_action_count = 0
        bc_losses: list[float] = []
        iteration_start_time = time.time()
        rollout_start_time = time.time()
        rollout_teacher_seconds = 0.0
        rollout_env_seconds = 0.0
        vlm_calls_before_iter = vlm_guidance_model.vlm_generation_call_count
        vlm_prompts_before_iter = vlm_guidance_model.vlm_generation_prompt_count
        print(
            f"iteration={iteration}/{config.training.num_iterations} start, "
            f"global_step={global_step}, beta={beta_i:.4f}, dataset_size={len(dataset)}",
            flush=True,
        )

        for step in range(config.env.num_steps):
            global_step += config.env.num_envs
            if vlm_history is not None:
                for env_idx in range(config.env.num_envs):
                    if bool(next_done[env_idx].item()):
                        vlm_history[env_idx].clear()
                prev_obs_for_history = detach_to_cpu(next_obs)
                valid_prev_obs = ~next_done.bool()
            else:
                prev_obs_for_history = None
                valid_prev_obs = None

            with torch.no_grad():
                teacher_query_start_time = time.time()
                teacher_actions = query_teacher_actions(
                    config=config,
                    envs=envs,
                    vlm_guidance_model=vlm_guidance_model,
                    obs=next_obs,
                    infos=infos,
                    device=device,
                    history=vlm_history,
                )
                rollout_teacher_seconds += time.time() - teacher_query_start_time
                dataset.add_batch(next_obs, teacher_actions)

                if "infos" in inspect.signature(agent.get_action_and_value).parameters:
                    student_action = agent.get_action_and_value(next_obs, infos=infos)[0]
                else:
                    student_action = agent.get_action_and_value(next_obs)[0]

                expert_mask = torch.rand(config.env.num_envs, device=device) < beta_i
                action = student_action.clone()
                teacher_for_rollout = teacher_actions.to(device=action.device, dtype=action.dtype)
                action[expert_mask] = teacher_for_rollout[expert_mask]
                expert_action_count += int(expert_mask.sum().item())

            env_step_start_time = time.time()
            next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rollout_env_seconds += time.time() - env_step_start_time
            next_done_np = np.logical_or(terminations, truncations)

            if vlm_history is not None:
                for env_idx in range(config.env.num_envs):
                    if valid_prev_obs is None or bool(valid_prev_obs[env_idx].item()):
                        env_obs_cpu = batch_index(prev_obs_for_history, env_idx)
                        vlm_history[env_idx].append(
                            {
                                "obs": env_obs_cpu,
                                "action": _history_action(action, env_idx, discrete_env),
                            }
                        )
                    if bool(next_done_np[env_idx]):
                        vlm_history[env_idx].clear()

            if "episode" in infos:
                episode_dones = np.array(infos["_episode"], dtype=bool)
                episodic_returns = np.array(infos["episode"]["r"])[episode_dones]
                episodic_lengths = np.array(infos["episode"]["l"])[episode_dones]
                if episodic_returns.size > 0:
                    wandb.log(
                        {
                            "charts/episodic_return": float(np.mean(episodic_returns)),
                            "charts/episodic_length": float(np.mean(episodic_lengths)),
                        },
                        step=global_step,
                    )

            next_obs = obs_to_tensor(next_obs_raw, device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if (
                rollout_log_interval > 0
                and (
                    step == 0
                    or (step + 1) % rollout_log_interval == 0
                    or step + 1 == config.env.num_steps
                )
            ):
                elapsed_rollout = time.time() - rollout_start_time
                completed_steps = step + 1
                calls_this_iter = (
                    vlm_guidance_model.vlm_generation_call_count - vlm_calls_before_iter
                )
                prompts_this_iter = (
                    vlm_guidance_model.vlm_generation_prompt_count - vlm_prompts_before_iter
                )
                seconds_per_call = (
                    rollout_teacher_seconds / calls_this_iter if calls_this_iter > 0 else 0.0
                )
                print(
                    f"iteration={iteration}/{config.training.num_iterations}, "
                    f"rollout_step={completed_steps}/{config.env.num_steps}, "
                    f"global_step={global_step}, dataset_size={len(dataset)}, "
                    f"vlm_prompts_iter={prompts_this_iter}, vlm_calls_iter={calls_this_iter}, "
                    f"teacher_seconds={rollout_teacher_seconds:.2f}, "
                    f"seconds_per_vlm_call={seconds_per_call:.2f}, "
                    f"elapsed_rollout_seconds={elapsed_rollout:.2f}",
                    flush=True,
                )

        agent.train()
        rollout_seconds = time.time() - rollout_start_time
        bc_start_time = time.time()
        for _ in range(config.training.bc_updates_per_iter):
            mb_obs, mb_actions = dataset.sample(config.training.bc_batch_size)
            mb_obs = _obs_to_device(mb_obs, device)
            mb_actions = mb_actions.to(device)
            if discrete_env:
                mb_actions = mb_actions.long().view(-1)
            else:
                mb_actions = mb_actions.float().view(config.training.bc_batch_size, *envs.single_action_space.shape)

            _, _, logprob, _, _ = agent.get_action_and_value(mb_obs, action=mb_actions)
            if logprob.ndim > 1:
                logprob = logprob.sum(dim=-1)
            bc_loss = -logprob.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.training.max_grad_norm)
            optimizer.step()
            bc_losses.append(float(bc_loss.detach().cpu().item()))
        bc_seconds = time.time() - bc_start_time

        if (
            config.training.get("eval_frequency", 0) > 0
            and global_step // config.training.get("eval_frequency", 0)
            > (global_step - config.training.batch_size) // config.training.get("eval_frequency", 0)
        ):
            episodic_returns, throughput, elapsed_time = evaluate(agent, config, eval_envs)
            wandb.log(
                {
                    "eval/episodic_return": float(np.mean(episodic_returns)),
                    "eval/throughput": throughput,
                    "eval/elapsed_time": elapsed_time,
                },
                step=global_step,
            )
            append_metrics({"global_step": global_step, "eval_return": float(np.mean(episodic_returns)),
                            "vlm_calls": vlm_guidance_model.vlm_generation_prompt_count})

        if (
            config.training.save_model
            and config.training.get("checkpoint_frequency", 0) > 0
            and global_step // config.training.get("checkpoint_frequency", 0)
            > (global_step - config.training.batch_size) // config.training.get("checkpoint_frequency", 0)
        ):
            save_checkpoint(agent, iteration=iteration, subfolder_name="checkpoints", pt_name="agent")

        expert_fraction = expert_action_count / float(config.training.batch_size)
        elapsed_since_start = time.time() - start_time
        iteration_seconds = time.time() - iteration_start_time
        iteration_vlm_calls = vlm_guidance_model.vlm_generation_call_count - vlm_calls_before_iter
        iteration_vlm_prompts = vlm_guidance_model.vlm_generation_prompt_count - vlm_prompts_before_iter
        log_payload = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/beta": beta_i,
            "charts/dataset_size": len(dataset),
            "charts/expert_action_fraction": expert_fraction,
            "losses/bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
            "charts/SPS": global_step / elapsed_since_start if elapsed_since_start > 0 else 0.0,
            "charts/vlm_prompt_calls": vlm_guidance_model.vlm_generation_prompt_count,
            "charts/vlm_call_count": vlm_guidance_model.vlm_generation_call_count,
            "charts/iteration_seconds": iteration_seconds,
            "charts/rollout_seconds": rollout_seconds,
            "charts/teacher_query_seconds": rollout_teacher_seconds,
            "charts/env_step_seconds": rollout_env_seconds,
            "charts/bc_update_seconds": bc_seconds,
            "charts/iteration_vlm_prompt_calls": iteration_vlm_prompts,
            "charts/iteration_vlm_call_count": iteration_vlm_calls,
            "charts/vlm_seconds_per_call": (
                rollout_teacher_seconds / iteration_vlm_calls
                if iteration_vlm_calls > 0
                else 0.0
            ),
        }
        wandb.log(log_payload, step=global_step)
        print(
            f"iteration={iteration}, global_step={global_step}, beta={beta_i:.4f}, "
            f"dataset_size={len(dataset)}, bc_loss={log_payload['losses/bc_loss']:.4f}, "
            f"iteration_seconds={iteration_seconds:.2f}, "
            f"vlm_seconds_per_call={log_payload['charts/vlm_seconds_per_call']:.2f}",
            flush=True,
        )

    episodic_returns, throughput, elapsed_time = evaluate(agent, config, eval_envs)
    wandb.log(
        {
            "eval/episodic_return": float(np.mean(episodic_returns)),
            "eval/throughput": throughput,
            "eval/elapsed_time": elapsed_time,
        },
        step=global_step,
    )
    append_metrics({"global_step": global_step, "eval_return": float(np.mean(episodic_returns)),
                    "vlm_calls": vlm_guidance_model.vlm_generation_prompt_count})

    if config.training.save_model:
        save_checkpoint(agent, iteration=None, subfolder_name="models", pt_name="agent")

    wandb.finish()
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    register_configs()
    main()

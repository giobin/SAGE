import time
import inspect
from typing import Optional
from collections import deque

import copy
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from gymnasium.spaces import Dict as DictSpace
from torch.distributions.categorical import Categorical

from src.utils import set_seed, get_agent_class, get_guidance_model_class, get_exp_name, extract_hydra_suffix, detach_to_cpu
from src.config import register_configs, WandBLoggerMode
from src.environments import make_env
from src.train.common import evaluate, save_checkpoint, append_metrics, get_gpu_device_map, obs_to_tensor, flatten_batch_obs, batch_index
from src.train.guidance import get_vlm_guidance
from src.train.reward_shaping import RewardShapingManager, RewardShapingBatch
from src.config import GPUConfig, GuidanceModelType, BCoefScheduleType, GuidanceType
from src.wandb_compat import wandb

@hydra.main(version_base=None, config_path="../../config", config_name="sage_qwen35")
def main(config: DictConfig) -> None:
    """
    Train a PPO agent, optionally with entropy-gated teacher guidance (SAGE) or VLM reward shaping (RL-VLM-F).
    Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    """
    print(OmegaConf.to_yaml(config, resolve=True))

    rl_gpu_id, vlm_gpu_ids = get_gpu_device_map(config.get("gpu_device_map", {}))
    config.gpu_device_map = GPUConfig(rl_gpu_id=rl_gpu_id, vlm_gpu_ids=vlm_gpu_ids)
    
    env_args = OmegaConf.to_container(config.env.get("env_args", {}), resolve=True)
    shared_memory = env_args.pop("shared_memory", None)

    # ALFWorld observations are not compatible with shared memory
    if config.env.env_id.startswith("Alfworld"):
        shared_memory = False

    # Instantiate the VLM teacher BEFORE env creation: vLLM must initialize CUDA before
    # AI2-THOR (ALFWorld) restores the VirtualGL LD_PRELOAD, whose faker libs break CUDA.
    vlm_guidance_model = None
    if config.vlm_guidance:
        config.env.prompt.prompts_path = hydra.utils.to_absolute_path(config.env.prompt.prompts_path)

        Guidance_model_class = get_guidance_model_class(config.training.guidance_model.model_type.value, config.training.guidance_model.use_vllm)
        print(f"Using guidance model class: {Guidance_model_class.__name__} for model {config.training.guidance_model.model_id} of type {config.training.guidance_model.model_type.value}")
        model_args = dict(config.training.guidance_model.get("model_args", {}))
        # Special case: for the oracle guidance model on FrozenLake, we need to pass the grid size
        if config.training.guidance_model.model_type == GuidanceModelType.oracle and "FrozenLake" in config.env.env_id:
            map_dim = config.env.env_args.get("random_map_dimension", None)
            if map_dim is None:
                map_name = config.env.env_args.get("map_name", None)
                if map_name is None:
                    raise ValueError("For FrozenLake oracle guidance model, either 'random_map_dimension' or 'map_name' must be specified in the environment config.")
                map_dim = [int(n) for n in map_name.split("x") if n.isdigit()]
                assert len(map_dim) == 2, f"Could not parse map dimensions from map_name '{map_name}'."
                model_args["grid_size"] = tuple(map_dim)
            else:
                model_args["grid_size"] = (map_dim, map_dim)
        needs_envs = config.training.guidance_model.model_type == GuidanceModelType.oracle
        if not needs_envs:
            if config.training.guidance_model.model_type == GuidanceModelType.transformers and config.training.guidance_model.use_vllm:
                model_args["verbose"] = config.logger.vllm_verbose
            vlm_guidance_model = Guidance_model_class(model_id=config.training.guidance_model.model_id,
                                                        seed=config.seed,
                                                        gpu_ids=config.get("gpu_device_map", {}).get("vlm_gpu_ids", None),
                                                        env_id=config.env.env_id,
                                                        guidance_model_config=config.training.guidance_model,
                                                        enforce_eager = config.training.guidance_model.enforce_eager,
                                                        **model_args)

    # env setup
    supports_shared_memory = "shared_memory" in inspect.signature(gym.vector.SyncVectorEnv).parameters
    sync_env_kwargs = (
        {"shared_memory": shared_memory}
        if shared_memory is not None and supports_shared_memory
        else {}
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=config.env.env_id,
                  idx=i,
                  capture_video=config.logger.capture_video,
                  use_visual_obs=config.env.image_obs,
                  stack_size=config.env.stack_size,
                  seed=config.get("seed", None),
                  **env_args)
         for i in range(config.env.num_envs)],
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

    # Deferred guidance model init for oracle models that need env access.
    if config.vlm_guidance and vlm_guidance_model is None:
        model_args["envs"] = envs
        model_args["agent_cfg"] = config.training.agent
        vlm_guidance_model = Guidance_model_class(model_id=config.training.guidance_model.model_id,
                                                    seed=config.seed,
                                                    gpu_ids=config.get("gpu_device_map", {}).get("vlm_gpu_ids", None),
                                                    env_id=config.env.env_id,
                                                    guidance_model_config=config.training.guidance_model,
                                                    enforce_eager = config.training.guidance_model.enforce_eager,
                                                    **model_args)

    reward_shaping_manager: Optional[RewardShapingManager] = None

    config.training.batch_size = int(config.env.num_envs * config.env.num_steps)
    config.training.minibatch_size = int(config.training.batch_size // config.training.num_minibatches)
    config.training.num_iterations = config.env.total_timesteps // config.training.batch_size

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
        # check that the logging frequency is a multiple of the config.env.num_envs
        if config.logger.wandb_logging_frequency % config.env.num_envs != 0:
            raise ValueError(f"config.logger.wandb_logging_frequency ({config.logger.wandb_logging_frequency}) must be a multiple of config.env.num_envs ({config.env.num_envs})")

    # TRY NOT TO MODIFY: seeding
    set_seed(config.seed, config.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    if device.type == "cuda":
        device = torch.device(f"cuda:{config.gpu_device_map.rl_gpu_id}")
    print(f"RL agent device: {device}")

    Agent = get_agent_class(config.training.method_name, config.training.agent.agent_type, discrete_env=discrete_env)
    agent = Agent(envs, config.training.agent).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config.training.learning_rate, eps=1e-5)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters() if p.requires_grad)}")

    # Setup guidance action counter if using VLM guidance
    if config.vlm_guidance:
        from collections import defaultdict
        agent.guidance_action_counter = defaultdict(int)

    if config.vlm_guidance and config.training.reward_shaping.enabled:
        reward_shaping_manager = RewardShapingManager(
            cfg=config.training.reward_shaping,
            env_cfg=config.env,
            prompt_cfg=config.env.prompt,
            envs=envs,
            experiment_dir=hydra_experiment_name,
            device=device,
        )
        reward_model_ready = False # don't use shaped rewards until the first reward model update is done

    # ALGO Logic: Storage setup
    # Support both Box and Dict observation spaces
    if isinstance(envs.single_observation_space, DictSpace):
        obs = {}
        for key, space in envs.single_observation_space.spaces.items():
            if hasattr(space, "shape") and space.shape is not None:
                obs[key] = torch.zeros((config.env.num_steps, config.env.num_envs) + space.shape, device=device)
            else:
                # Text or discrete spaces are stored as object arrays
                obs[key] = np.empty((config.env.num_steps, config.env.num_envs), dtype=object)
    else:
        obs = torch.zeros((config.env.num_steps, config.env.num_envs) + envs.single_observation_space.shape).to(device) # [steps, envs, 300, 300, 3]
    actions = torch.zeros((config.env.num_steps, config.env.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    rewards = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    if config.training.reward_shaping.enabled:
        true_rewards = torch.zeros_like(rewards)
    dones = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    values = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    actions_from_guidance = torch.zeros((config.env.num_steps, config.env.num_envs)).to("cpu", dtype=torch.bool)
    # Conservative value estimates from critic ensemble (v_min) for advantage computation on guided transitions
    v_min_values = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    v_mean_values = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    # Episode return tracking for BC return filtering
    episode_returns_buffer = torch.zeros((config.env.num_steps, config.env.num_envs)).to(device)
    episode_return_acc = np.zeros(config.env.num_envs, dtype=np.float32)
    episode_start_step = np.zeros(config.env.num_envs, dtype=np.int64)
    if config.training.agent.get("use_memory", False):
        memories = torch.zeros((config.env.num_steps, config.env.num_envs, agent.memory_size)).to(device)
        current_memory = torch.zeros((config.env.num_envs, agent.memory_size)).to(device)

    vlm_distribs = torch.zeros((config.env.num_steps, config.env.num_envs) + (int(envs.single_action_space.n),)).to(device)  # store VLM distributions to compute KL divergence if needed
    agent_distribs = torch.zeros((config.env.num_steps, config.env.num_envs) + (int(envs.single_action_space.n),)).to(device)  # store agent distributions to compute KL divergence if needed

    history_len = config.training.guidance_model.get("history_len", 0) if config.vlm_guidance else 0
    vlm_history = [deque(maxlen=history_len) for _ in range(config.env.num_envs)] if history_len > 0 else None

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=config.seed)
    next_obs = obs_to_tensor(next_obs, device)
    next_done = torch.zeros(config.env.num_envs).to(device)

    # wandb logging table
    guidance_steps_table = wandb.Table(columns=["global_step", "high_entropy_envs", "entropy", "prompt", "images", "suggested_action_text", "suggested_action_int"])

    supports_infos = "infos" in inspect.signature(agent.get_action_and_value).parameters

    for iteration in range(1, config.training.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if config.training.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.training.num_iterations
            lrnow = frac * config.training.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Annealing the behaviour cloning coefficient if instructed to do so.
        # We do this only if bc_schedule is set to linear, otherwise we keep it constant
        if config.vlm_guidance and config.training.bc_coef > 0.0 and config.training.bc_schedule == BCoefScheduleType.linear:
            frac = 1.0 - (iteration - 1.0) / config.training.num_iterations
            bc_coef_now = frac * config.training.bc_coef
        else:
            bc_coef_now = config.training.bc_coef

        # Reset episode return buffer for this rollout (don't reset episode_return_acc — carries over for episodes spanning rollout boundaries)
        episode_returns_buffer.zero_()
        episode_start_step[:] = 0

        for step in range(0, config.env.num_steps):
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
            if isinstance(next_obs, dict):
                for key, value in next_obs.items():
                    obs[key][step] = value
            else:
                obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if config.training.agent.get("use_memory", False):
                    current_memory = current_memory * (1 - next_done.unsqueeze(-1)) # set to zero the memory for the done envs
                    if supports_infos:
                        action, agent_probs, logprob, entropy, value, next_memory = agent.get_action_and_value(
                            next_obs,
                            memory=current_memory,
                            infos=infos,
                        )
                    else:
                        action, agent_probs, logprob, entropy, value, next_memory = agent.get_action_and_value(
                            next_obs,
                            memory=current_memory,
                        )
                    memories[step] = current_memory
                    current_memory = next_memory
                else:
                    if supports_infos:
                        action, agent_probs, logprob, entropy, value = agent.get_action_and_value(
                            next_obs,
                            infos=infos,
                        )
                    else:
                        action, agent_probs, logprob, entropy, value = agent.get_action_and_value(next_obs)

                # If the policy is unsure, the guidance model suggests the action
                if config.vlm_guidance and config.training.guidance_model.do_policy_guidance:
                    action, logprob, high_entropy_mask, suggested_action_text, guidance_steps_table, prompt, image, vlm_distribution = get_vlm_guidance(
                        envs=envs,
                        logprob=logprob,
                        entropy=entropy,
                        config=config,
                        vlm_guidance_model=vlm_guidance_model,
                        global_step=global_step,
                        step=step,
                        agent=agent,
                        next_obs=next_obs,
                        action=action,
                        probs=agent_probs,
                        guidance_steps_table=guidance_steps_table,
                        guidance_type=config.training.guidance_model.guidance_type,
                        infos=infos,
                        history=vlm_history,
                        memory=current_memory if config.training.agent.get("use_memory", False) else None,
                    )

                    # save the vlm_distribution if required by guidance type
                    if config.training.guidance_model.guidance_type in [GuidanceType.distribution, GuidanceType.full]:
                        # check if we have any high entropy envs at this step (i.e. vlm_distribution is not empty)
                        # convert VLM list -> [num_envs, num_actions] tensor
                        if len(vlm_distribution) > 0:
                            vlm_distribution_tensor = torch.as_tensor(
                                vlm_distribution,
                                device=vlm_distribs.device,
                                dtype=vlm_distribs.dtype,
                            )
                            vlm_distribs[step][high_entropy_mask] = vlm_distribution_tensor
                        agent_distribs[step] = agent_probs.probs

                values[step] = value.flatten()

                # Store conservative (min) and mean value estimates from critic ensemble
                if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
                    ensemble_kwargs = {}
                    if supports_infos:
                        ensemble_kwargs['infos'] = infos
                    if config.training.agent.get("use_memory", False):
                        ensemble_kwargs['memory'] = current_memory
                    v_ensemble = agent.get_value_ensemble(next_obs, **ensemble_kwargs)
                    v_min_values[step] = v_ensemble.min(dim=-1).values.flatten()
                    v_mean_values[step] = v_ensemble.mean(dim=-1).flatten()
                else:
                    v_min_values[step] = value.flatten()
                    v_mean_values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            if config.vlm_guidance and config.training.guidance_model.do_policy_guidance:
                actions_from_guidance[step] = (
                    high_entropy_mask.cpu()
                    if config.vlm_guidance
                    else torch.zeros_like(high_entropy_mask, dtype=torch.bool, device="cpu")
                )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            if config.training.reward_shaping.enabled:
                true_rewards[step] = copy.deepcopy(rewards[step])
            next_obs = obs_to_tensor(next_obs, device) # when in image_obs, this has shape (n_envs, (n_stack), H, W, C)
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

            # Track episode returns for BC return filtering
            episode_return_acc += reward
            done_np = next_done.cpu().numpy() if torch.is_tensor(next_done) else next_done
            for env_idx in range(config.env.num_envs):
                if done_np[env_idx]:
                    ep_start = int(episode_start_step[env_idx])
                    ep_return = float(episode_return_acc[env_idx])
                    episode_returns_buffer[ep_start:step + 1, env_idx] = ep_return
                    episode_return_acc[env_idx] = 0.0
                    episode_start_step[env_idx] = step + 1

            if vlm_history is not None:
                # Append the transition (state, action) used to reach the new next_obs
                for env_idx in range(config.env.num_envs):
                    env_obs_cpu = batch_index(prev_obs_for_history, env_idx) if isinstance(prev_obs_for_history, dict) or torch.is_tensor(prev_obs_for_history) else prev_obs_for_history
                    if valid_prev_obs is None or bool(valid_prev_obs[env_idx].item()):
                        vlm_history[env_idx].append(
                            {
                                "obs": env_obs_cpu,
                                "action": int(action[env_idx].item()) if torch.is_tensor(action) else int(action[env_idx]),
                            }
                        )
                    if bool(next_done[env_idx].item()):
                        vlm_history[env_idx].clear()

            if "episode" in infos:
                episode_dones = infos["_episode"]  # This tells which envs ended an episode this step
                episode_dones = np.array(episode_dones, dtype=bool)

                episodic_returns = np.array(infos["episode"]["r"])[episode_dones]
                episodic_lengths = np.array(infos["episode"]["l"])[episode_dones]

                mean_return = np.mean(episodic_returns)
                mean_length = np.mean(episodic_lengths)

                wandb.log({"charts/episodic_return": mean_return, "charts/episodic_length": mean_length}, step=global_step)

        # End of step loop - at this point buffers are all full
        shaped_rewards = None
        shaping_logs: dict[str, float] = {}
        if reward_shaping_manager is not None:
            batch = RewardShapingBatch(
                rewards=true_rewards,
                obs=obs,
                actions=actions,
                dones=dones,
            )
            shaped_rewards, shaping_logs = reward_shaping_manager.process_iteration(
                batch=batch,
                iteration=iteration,
                global_step=global_step,
                vlm_guidance_model=vlm_guidance_model,
            )
            if shaping_logs["reward_shaping/update_triggered"] == 1.0:
                reward_model_ready = True # one update done, we can now use shaped rewards
            env_reward_mean = float(true_rewards.mean().item())
            env_reward_std = float(true_rewards.std(unbiased=False).item())
            shaping_logs["reward_shaping/env_reward_mean"] = env_reward_mean
            shaping_logs["reward_shaping/env_reward_std"] = env_reward_std
            if shaped_rewards is not None and reward_model_ready:
                # Potential based reward shaping
                if config.training.reward_shaping.use_pbrs:
                    shaped_rewards[:-1] = config.training.gamma * shaped_rewards[1:] - shaped_rewards[:-1]
                
                rewards.copy_(shaped_rewards)
                shaped_flat = shaped_rewards.detach().view(-1).cpu().numpy()
                true_flat = true_rewards.detach().view(-1).cpu().numpy()
                if shaped_flat.size > 1 and np.std(shaped_flat) > 1e-8 and np.std(true_flat) > 1e-8:
                    corr = float(np.corrcoef(shaped_flat, true_flat)[0, 1])
                    shaping_logs["reward_shaping/env_reward_corr"] = corr
                    shaping_logs["reward_shaping/env_reward_corr_valid"] = 1.0
                else:
                    shaping_logs["reward_shaping/env_reward_corr_valid"] = 0.0

        # bootstrap value if not done
        with torch.no_grad():
            if config.training.agent.get("use_memory", False):
                current_memory = current_memory * (1 - next_done.unsqueeze(-1))

            # Compute V_min and V_mean for the final bootstrap step
            if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
                ensemble_kwargs_bootstrap = {}
                if supports_infos:
                    ensemble_kwargs_bootstrap['infos'] = infos
                if config.training.agent.get("use_memory", False):
                    ensemble_kwargs_bootstrap['memory'] = current_memory
                v_all = agent.get_value_ensemble(next_obs, **ensemble_kwargs_bootstrap)
                next_value_min = v_all.min(dim=-1).values.reshape(1, -1)
                next_value_mean = v_all.mean(dim=-1).reshape(1, -1)
            else:
                if config.training.agent.get("use_memory", False):
                    next_value = agent.get_value(next_obs, memory=current_memory).reshape(1, -1)
                else:
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                next_value_min = next_value
                next_value_mean = next_value

            # Single asymmetric GAE pass for ALL transitions:
            #   V_min for bootstrap V(s') — prevents overestimation compounding (TD3/SAC principle)
            #   V_mean for baseline V(s) — unbiased variance reduction
            # When num_critic_heads=1, V_min == V_mean == V, so this is standard GAE.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.env.num_steps)):
                if t == config.env.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_bootstrap = next_value_min
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_bootstrap = v_min_values[t + 1]

                baseline = v_mean_values[t]
                delta = rewards[t] + config.training.gamma * next_bootstrap * nextnonterminal - baseline
                advantages[t] = lastgaelam = delta + config.training.gamma * config.training.gae_lambda * nextnonterminal * lastgaelam

            # Returns for value regression targets: use V_mean as base
            returns = advantages + v_mean_values

        # flatten the batch (PPO)
        b_obs = flatten_batch_obs(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = v_mean_values.reshape(-1)
        b_actions_from_guidance = actions_from_guidance.reshape(-1)
        b_episode_returns = episode_returns_buffer.reshape(-1)
        b_vlm_distribs = vlm_distribs.reshape((-1,) + (int(envs.single_action_space.n),))
        b_agent_distribs = agent_distribs.reshape((-1,) + (int(envs.single_action_space.n),))
        if config.training.agent.get("use_memory", False):
            b_memories = memories.reshape((-1, agent.memory_size))

        # Optimizing the policy and value network
        b_inds = np.arange(config.training.batch_size)
        clipfracs = []
        for epoch in range(config.training.update_epochs):
            np.random.shuffle(b_inds)   
            for start in range(0, config.training.batch_size, config.training.minibatch_size):
                end = start + config.training.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = batch_index(b_obs, mb_inds) if isinstance(b_obs, dict) else b_obs[mb_inds]
                if config.training.agent.get("use_memory", False):
                    _, _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, action=b_actions[mb_inds], memory=b_memories[mb_inds])
                else:
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.training.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # Data-partitioned loss: PPO on unguided, Advantage-Weighted BC on guided
                mb_guided = b_actions_from_guidance[mb_inds].to(device).bool()
                mb_unguided = ~mb_guided

                # --- PPO loss on UNGUIDED transitions only ---
                # Guided transitions have meaningless importance ratios (logprob was overwritten by VLM)
                if mb_unguided.any():
                    ratio_ung = ratio[mb_unguided]
                    adv_ung = mb_advantages[mb_unguided]
                    if config.training.norm_adv and adv_ung.numel() > 1:
                        adv_ung = (adv_ung - adv_ung.mean()) / (adv_ung.std() + 1e-8)
                    pg_loss1 = -adv_ung * ratio_ung
                    pg_loss2 = -adv_ung * torch.clamp(ratio_ung, 1 - config.training.clip_coef, 1 + config.training.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Entropy bonus only for unguided (at guided steps, we want the policy to concentrate)
                    entropy_loss = entropy[mb_unguided].mean()
                else:
                    pg_loss = torch.tensor(0.0, device=device)
                    entropy_loss = torch.tensor(0.0, device=device)

                # --- Advantage-Weighted BC on GUIDED transitions only ---
                # Weights teacher actions by how good they were (advantage), so bad guidance is suppressed.
                # Connects to MARWIL (Wang et al., 2018) and AWR (Peng et al., 2019).
                awbc_loss = torch.tensor(0.0, device=device)
                if mb_guided.any() and config.vlm_guidance and bc_coef_now > 0:
                    adv_g = mb_advantages[mb_guided]

                    # CRR filtering (Wang et al., 2020)
                    awbc_filter = config.training.awbc_filter
                    if awbc_filter == "exp":
                        # CRR exponential (Eq. 4): exp(A / τ) — soft weighting (AWBC).
                        weights = torch.exp(adv_g / config.training.awbc_temperature).clamp(max=20.0)
                    elif awbc_filter == "binary":
                        # CRR binary (Eq. 3): indicator 1[A > 0] — unit-scale weights.
                        weights = (adv_g > 0).float()
                    elif awbc_filter == "none":
                        # Plain BC
                        weights = torch.ones_like(adv_g)
                    else:
                        raise ValueError(f"Unknown awbc_filter '{awbc_filter}', expected 'exp', 'binary' or 'none'.")

                    # Episode return filter: zero out BC for transitions from failed episodes.
                    if config.training.get("awbc_return_filter", False):
                        ep_returns_mb = b_episode_returns[mb_inds][mb_guided]
                        return_threshold = config.training.get("bc_return_threshold", 0.0)
                        return_mask = (ep_returns_mb > return_threshold).float()
                        weights = weights * return_mask

                    if config.training.guidance_model.guidance_type == GuidanceType.action or config.training.guidance_model.guidance_type == GuidanceType.full:
                        awbc_loss = -(weights.detach() * newlogprob[mb_guided]).mean()
                    elif config.training.guidance_model.guidance_type == GuidanceType.distribution:
                        # KL divergence variant for distribution guidance
                        vlm_distr_mb = b_vlm_distribs[mb_inds][mb_guided]
                        agent_distr_mb = b_agent_distribs[mb_inds][mb_guided]
                        vlm_dist = Categorical(probs=vlm_distr_mb)
                        agent_dist = Categorical(probs=agent_distr_mb)
                        kl_div = torch.distributions.kl.kl_divergence(vlm_dist, agent_dist)
                        awbc_loss = (weights.detach() * kl_div).mean()

                # --- Value loss ---
                # By default, computed on ALL transitions (critic learns from everything).
                # When exclude_value_loss_when_guided is True, only unguided transitions
                # contribute so that guided steps use pure BC and unguided steps use pure PPO.
                newvalue = newvalue.view(-1)
                if config.training.get("exclude_value_loss_when_guided", False) and mb_unguided.any():
                    vl_mask = mb_unguided
                elif config.training.get("exclude_value_loss_when_guided", False) and not mb_unguided.any():
                    vl_mask = None  # all guided → no value loss
                else:
                    vl_mask = None  # use all transitions

                if vl_mask is not None:
                    vl_newvalue = newvalue[vl_mask]
                    vl_returns = b_returns[mb_inds][vl_mask]
                    vl_old_values = b_values[mb_inds][vl_mask]
                else:
                    vl_newvalue = newvalue
                    vl_returns = b_returns[mb_inds]
                    vl_old_values = b_values[mb_inds]

                if config.training.get("exclude_value_loss_when_guided", False) and not mb_unguided.any():
                    v_loss = torch.tensor(0.0, device=device)
                elif config.training.clip_vloss:
                    v_loss_unclipped = (vl_newvalue - vl_returns) ** 2
                    v_clipped = vl_old_values + torch.clamp(
                        vl_newvalue - vl_old_values,
                        -config.training.clip_coef,
                        config.training.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - vl_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((vl_newvalue - vl_returns) ** 2).mean()

                # --- Ensemble value loss (train detached heads on same targets) ---
                v_ensemble_loss = torch.tensor(0.0, device=device)
                if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
                    ensemble_kwargs_mb = {}
                    if config.training.agent.get("use_memory", False):
                        ensemble_kwargs_mb['memory'] = b_memories[mb_inds]
                    v_ensemble = agent.get_value_ensemble(mb_obs, **ensemble_kwargs_mb)
                    v_ensemble_preds = v_ensemble[:, 1:]  # exclude main critic (already in v_loss)
                    if config.training.get("exclude_value_loss_when_guided", False) and mb_unguided.any():
                        v_ensemble_loss = 0.5 * ((v_ensemble_preds[mb_unguided] - b_returns[mb_inds][mb_unguided].unsqueeze(-1)) ** 2).mean()
                    elif config.training.get("exclude_value_loss_when_guided", False) and not mb_unguided.any():
                        v_ensemble_loss = torch.tensor(0.0, device=device)
                    else:
                        v_ensemble_loss = 0.5 * ((v_ensemble_preds - b_returns[mb_inds].unsqueeze(-1)) ** 2).mean()

                # --- Total loss ---
                loss = (pg_loss
                        - config.training.ent_coef * entropy_loss
                        + config.training.vf_coef * (v_loss + v_ensemble_loss)
                        + bc_coef_now * awbc_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.training.max_grad_norm)
                optimizer.step()

            if config.training.target_kl is not None and approx_kl > config.training.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if shaping_logs:
            wandb.log(shaping_logs, step=global_step)

        # Eval frequency in terms of global steps: we check if the current step has passed the evaluation frequency
        # Here we are at the end of an iteration, which increases global_step by config.batch_size steps, so we can check if in the last iteration we have passed the evaluation frequency.
        # e.g. if global_step is 12, batch_size is 4 and eval_frequency is 10, we find that 12 // 10 = 1 > (12 - 4) // 10 = 0,
        # which means that in the last iteration we have passed the evaluation frequency of 10.
        vlm_calls = vlm_guidance_model.vlm_generation_prompt_count if config.vlm_guidance else 0
        if config.training.get("eval_frequency", 0) > 0 and global_step // config.training.get("eval_frequency", 0) > (global_step - config.training.batch_size) // config.training.get("eval_frequency", 0):
            episodic_returns, throughput, elapsed_time = evaluate(agent, config, eval_envs)
            wandb.log({"eval/episodic_return": np.mean(episodic_returns),
                       "eval/throughput": throughput,
                       "eval/elapsed_time": elapsed_time}, step=global_step)
            append_metrics({"global_step": global_step, "eval_return": float(np.mean(episodic_returns)), "vlm_calls": vlm_calls})

        # Save the model checkpoint if the frequency is set and the condition is met
        if config.training.save_model and config.training.get("checkpoint_frequency", 0) > 0 and global_step // config.training.get("checkpoint_frequency", 0) > (global_step - config.training.batch_size) // config.training.get("checkpoint_frequency", 0):
            save_checkpoint(agent, global_step, subfolder_name="checkpoints", pt_name="agent", iteration=iteration)
            if reward_shaping_manager is not None:
                save_checkpoint(reward_shaping_manager._reward_model_ensemble, global_step, subfolder_name="checkpoints", pt_name="reward_model", iteration=iteration)
                

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        guided_mask_flat = b_actions_from_guidance.bool()
        log_payload = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/bc_coef": bc_coef_now,
            "losses/value_loss": v_loss.item(),
            "losses/awbc_loss": awbc_loss.item(),
            "losses/ensemble_value_loss": v_ensemble_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/guided_fraction": guided_mask_flat.float().mean().item(),
            "charts/guided_mean_advantage": b_advantages[guided_mask_flat].mean().item() if guided_mask_flat.any() else 0.0,
        }
        # AWBC filter rate: fraction of guided transitions with positive advantage
        if guided_mask_flat.any():
            guided_advs = b_advantages[guided_mask_flat]
            log_payload["charts/awbc_positive_adv_rate"] = (guided_advs > 0).float().mean().item()
        # Return filter pass rate (when enabled)
        if config.training.get("awbc_return_filter", False) and guided_mask_flat.any():
            guided_returns = b_episode_returns[guided_mask_flat]
            threshold = config.training.get("bc_return_threshold", 0.0)
            log_payload["charts/return_filter_pass_rate"] = (guided_returns > threshold).float().mean().item()
        # Conservative value sanity check
        if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
            v_min_flat = v_min_values.reshape(-1)
            v_mean_flat = v_mean_values.reshape(-1)
            log_payload["charts/v_mean"] = v_mean_flat.mean().item()
            log_payload["charts/v_min"] = v_min_flat.mean().item()
            log_payload["charts/v_gap"] = (v_mean_flat - v_min_flat).mean().item()
        if config.vlm_guidance:
            log_payload["charts/vlm_prompt_calls"] = vlm_guidance_model.vlm_generation_prompt_count
            log_payload["charts/vlm_call_count"] = vlm_guidance_model.vlm_generation_call_count
        wandb.log(log_payload, step=global_step)

        print(f"global_step: {global_step}, SPS: {int(global_step / (time.time() - start_time))}, vlm_calls: {vlm_calls}")

    if config.vlm_guidance:
        table = wandb.Table(
            data=[[action, count] for action, count in agent.guidance_action_counter.items()],
            columns=["action", "count"]
        )
        wandb.log({
            "final_guidance_action_plot": wandb.plot.bar(
                table, "action", "count", title="Final Guidance Model Actions"
            )
        })
        wandb.log({"guidance_steps_table": guidance_steps_table})

    # Evaluate the agent at the end of training
    episodic_returns, throughput, elapsed_time = evaluate(agent, config, eval_envs)
    wandb.log({"eval/episodic_return": np.mean(episodic_returns),
               "eval/throughput": throughput,
               "eval/elapsed_time": elapsed_time}, step=global_step)
    append_metrics({"global_step": global_step, "eval_return": float(np.mean(episodic_returns)),
                    "vlm_calls": vlm_guidance_model.vlm_generation_prompt_count if config.vlm_guidance else 0})

    if config.training.save_model:
        save_checkpoint(agent, iteration=None, subfolder_name="models", pt_name="agent")
        if reward_shaping_manager is not None:
            save_checkpoint(reward_shaping_manager._reward_model_ensemble, iteration=None, subfolder_name="models", pt_name="reward_model")

    wandb.finish()
    envs.close()
    eval_envs.close()

if __name__ == "__main__":
    register_configs()
    main()

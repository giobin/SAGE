import torch

from src.guidance_models.common import extract_action_and_thinking
from src.guidance_models.base_guidance_model import BaseGuidanceModel
from src.train.common import batch_index
from src.agents.base_agents import AC_Agent
from src.config import GuidanceModelType, GuidanceType, UncertaintyMethod

import gymnasium as gym
from types import SimpleNamespace
from gymnasium.vector import SyncVectorEnv
from omegaconf import DictConfig
from typing import Optional, Tuple
from src.wandb_compat import wandb


def get_vlm_guidance(envs: SyncVectorEnv,
                     logprob: torch.Tensor,
                     entropy: torch.Tensor,
                     config: DictConfig,
                     vlm_guidance_model: BaseGuidanceModel,
                     global_step: int,
                     step: int,
                     agent: AC_Agent,
                     next_obs: torch.Tensor,
                     action: torch.Tensor,
                     probs: torch.Tensor,
                     guidance_steps_table: wandb.Table,
                     guidance_type: GuidanceType,
                     infos: Optional[dict] = None,
                     history: Optional[list] = None,
                     memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], wandb.Table]:
    """
    Provides VLM guidance to the agent based on the entropy of the policy for a given state. 
    Can be plugged into a model-free RL training loop, suggested actions will overwrite the agent's actions.

    Args:
        envs (SyncVectorEnv): The environment(s) to interact with.
        logprob (torch.Tensor): Log probabilities of the actions taken by the agent.
        entropy (torch.Tensor): Entropy of the policy for the current state.
        config (DictConfig): Configuration object containing the model and environment settings.
        vlm_guidance_model (BaseGuidanceModel): The VLM guidance model to use.
        global_step (int): The global step in the training process.
        step (int): The current step in the training process.
        agent (AC_Agent): The agent to provide guidance for.
        next_obs (torch.Tensor): The next observations from the environment.
        action (torch.Tensor): The actions taken by the agent.
        probs (torch.Tensor): Probabilities of the actions taken by the agent.
        guidance_steps_table (wandb.Table): A table to log the guidance steps for visualization in WandB.
        guidance_type (GuidanceType): The type of guidance to provide (action, distribution, full).
        infos (Optional[dict]): Additional information from the environment (required for ALFWorld to populate the prompts).
        history (Optional[list]): Per-env deque/list of previous {"obs", "action"} entries to include in the prompt.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], wandb.Table]:
            - action (torch.Tensor): The actions to take, possibly modified by the guidance model (one for each vector environment).
            - logprob (torch.Tensor): Updated log probabilities of the actions after applying guidance.
            - high_entropy_mask (torch.Tensor): A mask indicating which actions were guided by the VLM.
            - suggested_action_text (list[str]): The VLM completions for the suggested actions.
            - guidance_steps_table (wandb.Table): Updated table with the guidance steps for visualization in WandB.
    """
    
    suggested_action_text, guidance_provision_prompts, guidance_provision_images, vlm_distribution = [], [], [], []

    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise NotImplementedError(f"Unsupported action space: {type(envs.single_action_space)}")
    # Normalized entropy H / log|A| in [0, 1]
    max_entropy = torch.log(
        torch.tensor(envs.single_action_space.n, device=logprob.device, dtype=logprob.dtype)
    )
    normalized_entropy = entropy / max_entropy

    # Uncertainty gating: select which environments need VLM guidance
    uncertainty_method = config.training.guidance_model.get("uncertainty_method", UncertaintyMethod.entropy)
    guidance_threshold = config.training.guidance_model.guidance_threshold
    v_std = None  # will be set if value disagreement is computed
    normalized_v_std = None
    high_disagreement_mask = None

    if uncertainty_method == UncertaintyMethod.entropy:
        # Gate on normalized policy entropy only (default, backward-compatible)
        high_entropy_mask = normalized_entropy >= guidance_threshold

    elif uncertainty_method == UncertaintyMethod.value_disagreement:
        # Gate on critic ensemble disagreement only (requires num_critic_heads > 1)
        if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
            with torch.no_grad():
                vd_kwargs = {}
                if memory is not None:
                    vd_kwargs['memory'] = memory
                if infos is not None:
                    vd_kwargs['infos'] = infos
                _, v_std, _ = agent.get_value_stats(next_obs, **vd_kwargs)
            # Normalize v_std to [0,1] within the batch so the same threshold scale works
            v_std_max = v_std.max()
            normalized_v_std = v_std / (v_std_max + 1e-8) if v_std_max > 0 else torch.zeros_like(v_std)
            high_entropy_mask = normalized_v_std >= guidance_threshold
        else:
            # Fallback to entropy if no ensemble
            high_entropy_mask = normalized_entropy >= guidance_threshold

    elif uncertainty_method == UncertaintyMethod.dual:
        # Gate on BOTH entropy AND value disagreement (filters false positives)
        high_entropy_mask = normalized_entropy >= guidance_threshold
        if hasattr(agent, 'num_critic_heads') and agent.num_critic_heads > 1:
            with torch.no_grad():
                vd_kwargs = {}
                if memory is not None:
                    vd_kwargs['memory'] = memory
                if infos is not None:
                    vd_kwargs['infos'] = infos
                _, v_std, _ = agent.get_value_stats(next_obs, **vd_kwargs)
            v_std_max = v_std.max()
            normalized_v_std = v_std / (v_std_max + 1e-8) if v_std_max > 0 else torch.zeros_like(v_std)
            high_disagreement_mask = normalized_v_std >= guidance_threshold
            high_entropy_mask = high_entropy_mask & high_disagreement_mask

    else:
        raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

    # Always guiding the agent under high uncertainty might lead to overfitting/always using the same strategy,
    # so we introduce a "kappa" parameter to control how many actions we guide.
    # This works similarly to epsilon-greedy exploration, where we guide only a fraction of the actions.
    # kappa = 1.0 means we always guide, kappa = 0.0 means we never guide.
    kappa = config.training.guidance_model.get("kappa", 1.0)
    if kappa < 1.0:
        high_entropy_mask = high_entropy_mask & (torch.rand_like(normalized_entropy) < kappa)

    # Build prompts and images for every env (returned to the caller even when no env is guided)
    guidance_provision_prompts, guidance_provision_images = vlm_guidance_model.populate_prompts(
        config,
        states=next_obs,
        infos=infos,
        history=history,
    )
    
    if high_entropy_mask.any():
        # Entropy is high, use the guidance model to suggest an action
        indices = high_entropy_mask.nonzero(as_tuple=True)[0]
        ent_vals = normalized_entropy[indices]
        prompts = [guidance_provision_prompts[i] for i in indices.tolist()] 
        images = [guidance_provision_images[i] for i in indices.tolist()] if guidance_provision_images is not None else None

        if config.training.guidance_model.cache.use_cache:
            cache_stats_before_guidance = vlm_guidance_model.cache.cache_stats.copy() if vlm_guidance_model.cache is not None else None
        kwargs = {}
        if config.training.guidance_model.model_type == GuidanceModelType.oracle or config.training.guidance_model.model_type == GuidanceModelType.random:
            env_indices = indices.tolist()
            if hasattr(envs, "envs"):
                hent_envs = SimpleNamespace(envs=[envs.envs[i] for i in env_indices])
                kwargs["envs"] = hent_envs
            else:
                kwargs["envs"] = envs
                kwargs["env_indices"] = env_indices
            kwargs["action_space"] = envs.single_action_space
            if config.training.guidance_model.model_type == GuidanceModelType.oracle:
                kwargs["random_guidance_prob"] = float(config.training.guidance_model.get("random_guidance_prob", 0.0))
            
        # Get guidance from the VLM model
        guidance_provision = vlm_guidance_model.provide_guidance(prompts=prompts, images=images, **kwargs)
        suggested_action_text = guidance_provision["responses"]
        if config.training.guidance_model.model_type == GuidanceModelType.oracle:
            random_action_count = int(guidance_provision.get("random_action_count", 0))
            wandb.log({"charts/oracle_random_guidance_actions": random_action_count}, step=global_step)

        if config.training.guidance_model.cache.use_cache:
            cache_stats_after_guidance = vlm_guidance_model.cache.cache_stats.copy() if vlm_guidance_model.cache is not None else None

        action_thinking_and_maybe_distribution = [
            extract_action_and_thinking(
                action_text,
                allowed=envs.single_action_space,
                guidance_type=guidance_type,
            )
            for action_text in suggested_action_text
        ]
        suggested_action_int_cpu = [a for a, *_ in action_thinking_and_maybe_distribution]
        reasoning = [r for _, r, *_ in action_thinking_and_maybe_distribution]
        vlm_distribution = [d for _, _, d, *_ in action_thinking_and_maybe_distribution]
        suggested_action_int = torch.tensor(
            suggested_action_int_cpu,
            device=action.device
        )
        if guidance_type == GuidanceType.action or guidance_type == GuidanceType.full:
            # Overwrite the action and logprob with the suggested action from the guidance model
            action[indices] = suggested_action_int
            updated_logprob = probs.log_prob(action)   # tricky, this can be shape (B,) for Discrete or (B, action_dim) for Box
            if updated_logprob.ndim > 1:
                # sum over action dimensions to get a (B,) tensor
                updated_logprob = updated_logprob.sum(-1)
            logprob[indices] = updated_logprob[indices]

        # Update the action counter
        for a in suggested_action_int:
            agent.guidance_action_counter[a.item()] += 1

        # Wandb logging
        if global_step % config.logger.wandb_logging_frequency == 0:
            # save just the first environment suggested action text
            guidance_steps_table.add_data(global_step, 
                                        str(indices.tolist()), 
                                        str(ent_vals.tolist()),
                                        guidance_provision["prompts"][0] if len(guidance_provision["prompts"]) > 0 else "",
                                        [wandb.Image(guidance_provision["images"][0][i]) for i in range(len(guidance_provision["images"][0]))] if guidance_provision_images is not None else None,
                                        reasoning[0] if len(reasoning) > 0 else "",
                                        str(suggested_action_int_cpu) if len(suggested_action_int_cpu) > 0 else "")

            if config.training.guidance_model.cache.use_cache and cache_stats_before_guidance is not None and cache_stats_after_guidance is not None:
                hits = cache_stats_after_guidance['hits'] - cache_stats_before_guidance['hits']
                misses = cache_stats_after_guidance['misses'] - cache_stats_before_guidance['misses']
                wandb.log({
                    "cache/hits": hits,
                    "cache/misses": misses,
                    "cache/hit_ratio": hits / len(indices),
                    "cache/overall_hit_ratio": cache_stats_after_guidance['hit_ratio'],
                    "cache/size": cache_stats_after_guidance['size'],
                    **({"cache/avg_vector_length": cache_stats_after_guidance['avg_vector_length'],
                        "cache/avg_diversity": cache_stats_after_guidance['avg_diversity']} 
                    if config.training.guidance_model.cache.cache_type.value == "vector" else {})
                }, step=global_step)

            # Log guidance action counts
            wandb.log({f"guidance_action_count/{a}": c for a, c in agent.guidance_action_counter.items()}, step=global_step)
            wandb.log({"charts/high_entropy_actions": high_entropy_mask.sum().item()}, step=global_step)

            # Log value disagreement metrics if using ensemble-based gating
            if v_std is not None:
                wandb.log({
                    "charts/mean_value_std": v_std.mean().item(),
                    "charts/max_value_std": v_std.max().item(),
                    "charts/normalized_value_std_mean": normalized_v_std.mean().item(),
                }, step=global_step)
                if high_disagreement_mask is not None:
                    wandb.log({
                        "charts/high_disagreement_ratio": high_disagreement_mask.float().mean().item(),
                    }, step=global_step)

    else:
        if config.training.guidance_model.model_type == GuidanceModelType.oracle:
            wandb.log({"charts/oracle_random_guidance_actions": 0}, step=global_step)
        if global_step % config.logger.wandb_logging_frequency == 0:
            wandb.log({"charts/high_entropy_actions": 0}, step=global_step)   
        
        reasoning = [""]  # No reasoning since no guidance was provided
    
    return action, logprob, high_entropy_mask, reasoning, guidance_steps_table, guidance_provision_prompts, guidance_provision_images, vlm_distribution

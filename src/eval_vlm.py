import time
import hydra
import torch
import numpy as np
import gymnasium as gym

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from collections import defaultdict


from src.utils import set_seed, get_guidance_model_class
from src.config import register_configs, WandBLoggerMode, GuidanceModelType
from src.environments import make_env
from src.train.common import obs_to_tensor, append_metrics
from src.guidance_models.common import extract_action_and_thinking
from collections import Counter
from src.wandb_compat import wandb

MAX_EPISODE_STEPS = 1000 # Safety limit, increase if an environment requires more steps

@hydra.main(version_base=None, config_path="../config", config_name="vlm_policy_qwen35")
def main(config: DictConfig) -> None:
    """
    Evaluate a VLM (or oracle) directly as a zero-shot policy (VLM-as-Policy baseline).
    """
    print(OmegaConf.to_yaml(config, resolve=True))
    set_seed(config.seed, config.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    config.env.prompt.prompts_path = hydra.utils.to_absolute_path(config.env.prompt.prompts_path)
    
    env = make_env(env_id=config.env.env_id, idx=0, 
                  capture_video=config.logger.capture_video, 
                  use_visual_obs=config.env.image_obs,
                  stack_size=config.env.stack_size,
                  seed=config.get("seed", None),
                  **config.env.get("env_args", {}))()
    env = gym.vector.SyncVectorEnv([lambda: env])
    # Sampled when a teacher answer cannot be parsed into a valid action; seeded for reproducibility
    env.single_action_space.seed(config.seed)

    Guidance_model_class = get_guidance_model_class(config.training.guidance_model.model_type.value, config.training.guidance_model.use_vllm)
    print(f"Using guidance model class: {Guidance_model_class.__name__} for model {config.training.guidance_model.model_id} of type {config.training.guidance_model.model_type.value}")
    model_args = dict(config.training.guidance_model.get("model_args", {}))
    if config.training.guidance_model.model_type == GuidanceModelType.oracle:
        model_args["envs"] = env
    
    vlm_guidance_model = Guidance_model_class(model_id=config.training.guidance_model.model_id, 
                                                seed=config.seed,
                                                gpu_ids=config.get("gpu_device_map", {}).get("vlm_gpu_ids", None),
                                                guidance_model_config=config.training.guidance_model,
                                                env_id=config.env.env_id,
                                                enforce_eager = config.training.guidance_model.enforce_eager,
                                                **model_args
                                                )
    vlm_guidance_model.guidance_action_counter = defaultdict(int)
    parse_source_counter = Counter()  # Track how actions were parsed (action_tag, fallback_int, sampled, etc.)

    config.exp_name = f"VLM_policy_zero_shot__{config.env.env_id}__{config.training.guidance_model.model_id.split('/')[-1]}__{config.seed}"
    run_name = f"{config.exp_name}__{int(time.time())}"
    hydra_experiment_name = HydraConfig.get().runtime.output_dir
    
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
    
    # Run evaluation episodes using only the VLM guidance model as a zero-shot policy
    episodic_returns = []
    steps = []
    for episode in range(config.training.eval_episodes):
        start_time = time.time()
        next_obs, _ = env.reset(seed=config.seed + episode)
        next_obs = obs_to_tensor(next_obs, device)
        done = False
        step = 0
        _return = 0.0
        while not done and step < MAX_EPISODE_STEPS:
            if step % 16 == 0:
                print(f"Episode {episode}, Step {step}, Return {_return:.2f}")
            step += 1
            cache_stats_before_guidance = vlm_guidance_model.cache.cache_stats.copy() if config.training.guidance_model.cache.use_cache else None
            kwargs = {}
            if config.training.guidance_model.model_type == GuidanceModelType.oracle or config.training.guidance_model.model_type == GuidanceModelType.random:
                kwargs = {"envs": env}
            prompts, images = vlm_guidance_model.populate_prompts(config, states=next_obs)
            guidance_provision = vlm_guidance_model.provide_guidance(prompts=prompts, images=images, **kwargs)
            suggested_action_text = guidance_provision["responses"]
            cache_stats_after_guidance = vlm_guidance_model.cache.cache_stats.copy() if config.training.guidance_model.cache.use_cache else None
            action, reasoning, _, _, parse_source = [
                extract_action_and_thinking(
                    action_text,
                    allowed=env.single_action_space,
                    verbose=False,
                    guidance_type=config.training.guidance_model.guidance_type,
                )
                for action_text in suggested_action_text
            ][0]
            parse_source_counter[parse_source] += 1
            assert isinstance(action, int), f"Action must be an integer, got {type(action)}"
            
            vlm_guidance_model.guidance_action_counter[action] += 1
            if cache_stats_before_guidance is not None and cache_stats_after_guidance is not None:
                hits = cache_stats_after_guidance['hits'] - cache_stats_before_guidance['hits']
                misses = cache_stats_after_guidance['misses'] - cache_stats_before_guidance['misses']
                wandb.log({
                    "cache/hits": hits,
                    "cache/misses": misses,
                    "cache/overall_hit_ratio": cache_stats_after_guidance['hit_ratio'],
                    "cache/size": cache_stats_after_guidance['size'],
                    **({"cache/avg_vector_length": cache_stats_after_guidance['avg_vector_length'],
                        "cache/avg_diversity": cache_stats_after_guidance['avg_diversity']} 
                    if config.training.guidance_model.cache.cache_type.value == "vector" else {})})
                
            wandb.log({f"guidance_action_count/{a}": c for a, c in vlm_guidance_model.guidance_action_counter.items()})
            
            next_obs, reward, terminations, truncations, infos = env.step([action])
            next_obs = obs_to_tensor(next_obs, device)
            done = np.logical_or(terminations, truncations)
            _return += reward[0]
            
        print(f"Episode {episode} finished in {step} steps with return {_return:.2f} in {time.time() - start_time:.2f} seconds.")
        episodic_returns.append(_return)
        steps.append(step)

        total_parses = sum(parse_source_counter.values())
        wandb.log({"eval/return": _return,
                   "eval/steps": step,
                   "eval/avg_return": np.mean(episodic_returns),
                   "eval/avg_steps": np.mean(steps),
                   **{f"parse_source/{k}": v for k, v in parse_source_counter.items()},
                   "parse_source/action_tag_rate": parse_source_counter.get("action_tag", 0) / max(total_parses, 1),
                   "parse_source/no_action_tag_rate": 1.0 - parse_source_counter.get("action_tag", 0) / max(total_parses, 1),
                   })
    
    total_parses = sum(parse_source_counter.values())
    action_tag_count = parse_source_counter.get("action_tag", 0)
    print(f"\n{'='*60}")
    print(f"Average return over {len(episodic_returns)} episodes: {np.mean(episodic_returns):.2f}")
    print(f"Average steps per episode: {np.mean(steps):.2f}")
    print(f"Guidance action counts: {dict(vlm_guidance_model.guidance_action_counter)}")
    print(f"\nAction parse source breakdown ({total_parses} total):")
    for source, count in parse_source_counter.most_common():
        print(f"  {source:20s}: {count:5d} ({count/total_parses*100:.1f}%)")
    print(f"\n  <action> tag success rate: {action_tag_count}/{total_parses} ({action_tag_count/max(total_parses,1)*100:.1f}%)")
    print(f"  NO <action> tag (fallback/sampled): {total_parses - action_tag_count}/{total_parses} ({(total_parses - action_tag_count)/max(total_parses,1)*100:.1f}%)")
    print(f"{'='*60}")
    append_metrics({"global_step": 0, "eval_return": float(np.mean(episodic_returns)), "vlm_calls": vlm_guidance_model.vlm_generation_prompt_count})

if __name__ == "__main__":
    register_configs()
    main()

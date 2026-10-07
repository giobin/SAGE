from enum import Enum
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

############################################
# Enums for custom types in configurations #
############################################

class BCoefScheduleType(Enum):
    """
    Enum for different types of behaviour cloning coefficient schedules.
    """
    constant = "constant" # Constant behaviour cloning coefficient (no schedule)
    linear = "linear"     # Linear decay from initial value to almost 0 over the course of training

class DaggerBetaScheduleType(Enum):
    """
    Enum for DAgger expert-action mixing schedules.
    """
    exponential = "exponential" # beta_i = beta_start * beta_decay ** i
    indicator = "indicator"     # beta_0 = beta_start, beta_i = beta_end afterwards
    linear = "linear"           # Linear decay from beta_start to beta_end over beta_warmup_iters
    constant = "constant"       # Constant beta_start for all iterations

class AgentType(Enum):
    """
    Enum for different types of agents.
    """
    mlp = "mlp"           # Multi-layer perceptron
    cnn = "cnn"           # Convolutional neural network
    cnn_text = "cnn_text" # Convolutional neural network with text embeddings (e.g., for ALFWorld)
    
class MethodName(Enum):
    """
    Enum for different RL methods.
    """
    ppo = "ppo"               # Proximal Policy Optimization
    dagger_vlm = "dagger_vlm" # DAgger-style VLM imitation baseline

class ActivationType(Enum):
    """
    Enum for different activation functions.
    """
    ReLU = "ReLU"          # Rectified Linear Unit
    SiLU = "SiLU"          # Sigmoid Linear Unit (Swish)
    Tanh = "Tanh"          # Hyperbolic Tangent

class GuidanceModelType(Enum):
    """
    Enum for different guidance models.
    """
    transformers = "transformers" # Huggingface/transformers models
    oracle = "oracle"             # Heuristic based oracle (no VLM used)
    random = "random"             # Random guidance model (no VLM used)

class GuidanceType(Enum):
    """
    Enum for different types of guidance.
    """
    action = "action"                   # Guidance for action selection
    distribution = "distribution"       # Guidance for distribution over actions
    full = "full"                       # Guidance for both action selection and distribution

class UncertaintyMethod(Enum):
    """
    Enum for different uncertainty estimation methods used to gate VLM guidance.
    """
    entropy = "entropy"                         # Gate on policy entropy only (default, used in the paper)
    value_disagreement = "value_disagreement"   # Gate on critic ensemble disagreement only (requires num_critic_heads > 1)
    dual = "dual"                               # Gate on BOTH entropy AND value disagreement (requires num_critic_heads > 1)
    
class VLMCacheType(Enum):
    """
    Enum for different types of VLM caches.
    """
    single = "single"  # Single response cache for storing one response per key
    vector = "vector"  # Vector cache for storing multiple responses per key
    
class WandBLoggerMode(Enum):
    """
    Enum for Weights and Biases (WandB) logging modes.
    """
    disabled = "disabled" # No WandB logging
    online   = "online"   # Runs are syncronized as they happen (requires internet connection)
    offline  = "offline"  # Runs are recorded but must later be syncronized with `wandb sync`

#################################################
# Dataclasses for config structure and defaults #
#################################################

# Environment configuration

@dataclass
class PromptConfig:
    """
    Configuration for VLM prompt templates.
    """
    prompts_path: str = MISSING           # Path to the folder with the prompts (relative to the root of the repository).
    text_prompt: str = "text_prompt.md"   # Default text prompt file name (relative to prompts_path).
    image_prompt: str = "image_prompt.md" # Default image prompt file name (relative to prompts_path).
    use_text_description: bool = False    # If True, use a detailed formatted text description of the observation instead of a simple observation vector (coming from src.environments.text_descriptions).
    
    use_cot_prompt: bool = True                   # If True, use Chain-of-Thought prompt variant (provided in the config).
    text_cot_prompt: str = "text_cot_prompt.md"   # Default text CoT prompt file name (relative to prompts_path).
    image_cot_prompt: str = "image_cot_prompt.md" # Default image CoT prompt file name (relative to prompts_path).

    text_cot_distribution_prompt: str = "text_cot_distribution_prompt.md"   # Default text CoT prompt file name for distribution guidance (relative to prompts_path).
    image_cot_distribution_prompt: str = "image_cot_distribution_prompt.md" # Default image CoT prompt file name for distribution guidance (relative to prompts_path).
    # Reward-shaping specific prompt assets (defaults fall back to policy guidance path when None)
    reward_shaping_prompts_path: str = "data/prompts/reward_shaping" # Path to the folder with the reward-shaping specific prompts (relative to the root of the repository).
    reward_shaping_difference_prompt: str = "difference_prompt.md"   # Default difference prompt file name for reward shaping (relative to prompts_path).
    reward_shaping_preference_prompt: str = "preference_prompt.md"   # Default preference prompt file name for reward shaping (relative to prompts_path).
    reward_shaping_env_description: Optional[str] = None             # Description of the environment for reward shaping in natural language.
    reward_shaping_goal: Optional[str] = None                        # Description of the goal for reward shaping in natural language.

@dataclass
class EnvironmentConfig:
    """
    Configuration for the RL environment.
    """
    env_id: str = MISSING             # Environment ID (from Gymnasium)
    total_timesteps: int = 100000     # Total timesteps for training (number of individual steps)
    num_steps: int = 128              # Number of steps per environment per rollout
    num_envs: int = 4                 # Number of vectorized environments
    image_obs: bool = False           # If True, use image observations instead of vector observations
    stack_size: int = 1               # Number of frames to stack for image observations (if image_obs is True)
    prompt: PromptConfig = field(default_factory=PromptConfig)       # Prompt configuration for VLM guidance
    env_args: Optional[Dict[str, Any]] = field(default_factory=dict) # Additional env-specific arguments (e.g., is_slippery for FrozenLake)

@dataclass
class CacheConfig:
    """
    Configuration for the cache used in VLM guidance.
    """
    use_cache: bool = True                                  # If True, use a cache to save common responses to avoid repeated calls to the VLM for actions that are already known (e.g. for the same observation).
    cache_size: int = 128                                   # Size of the cache used to save common responses to avoid repeated calls to the VLM for actions that are already known (e.g. for the same observation).
    cache_type: VLMCacheType = VLMCacheType.vector          # Type of cache to use (single response or vector cache)
    cache_vector_size: int = 5                              # Size of the vector cache (number of responses to store per key). Only used if cache_type is VLMCacheType.vector.

@dataclass
class GuidanceModelConfig:
    """
    Configuration for the Vision-Language guidance model.
    """
    model_type: GuidanceModelType = GuidanceModelType.transformers # Type of model to use
    model_id: str = "google/gemma-3-12b-it"                        # Hugging Face model ID
    use_vllm: bool = True                                          # If True, use vLLM for inference (for Transformers models)
    enforce_eager: bool = False                                    # If True, use eager mode for vLLM which is debuggable (if False, use compiled mode)
    
    do_policy_guidance: bool = True                                # If True, use the VLM to provide policy guidance under uncertainty.
    guidance_threshold: float = 0.1                                # Normalized uncertainty threshold [0,1] above which the agent will be guided by the VLM. Applied to whichever metric is selected by uncertainty_method.
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.entropy  # Method for computing uncertainty to gate VLM guidance.
    kappa: float = 1.0                                             # Probability of querying the teacher in a high-uncertainty state (1.0 = always, as in the paper).
    random_guidance_prob: float = 0.0                              # For oracle guidance: probability to replace oracle action with a random env action (seeded by experiment seed).
    
    cache: CacheConfig = field(default_factory=CacheConfig) # Cache configuration for the VLM guidance model
    guidance_type: GuidanceType = GuidanceType.action  # Type of guidance to provide (action selection, distribution over actions, or both)
    prompt_suffix: str = ""  # Suffix appended to prompt filenames (e.g. "_qwen" loads image_cot_prompt_qwen.md instead of image_cot_prompt.md)
    history_len: int = 0  # Number of past (obs, action) pairs to include in the VLM prompt. 0 disables history.
    just_text_history: bool = False  # If True, include only text actions in history (no history images).
    
    model_args: Dict[str, Any] = field(default_factory=lambda: {
        "gpu_memory_utilization": 0.75, 
        "json_output": False
    }) # Additional model-specific kwargs (e.g., max_new_tokens, temperature, top_p, etc. for the VLM).
    
# Reward-shaping config (RL-VLM-F re-implementation)
@dataclass
class RewardModelConfig:
    """Configuration for the CNN-based reward model used during reward shaping."""

    ensemble_size: int = 3
    learning_rate: float = 3e-4
    output_size: int = 1
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3])
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    conv_strides: List[int] = field(default_factory=lambda: [3, 2, 2, 2])
    conv_paddings: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    hidden_sizes: List[int] = field(default_factory=list)
    batch_norm_conv: bool = False
    batch_norm_fc: bool = False
    hidden_activation: str = "ReLU"
    output_activation: str = "Identity" #! Original paper seems to use Tanh, but Identity learns a better reward model (but only makes sense with PBRS to normalize the scale of the shaped reward)
    init_w: float = 1e-3
    hidden_init: str = "fanin"


@dataclass
class RewardShapingConfig:
    """Configuration for learning a reward-shaping model from VLM feedback."""
    enabled: bool = False
    update_interval: int = 5000                     # env steps between updates to the reward model
    max_pairs_per_update: int = 128                 # number of image pairs queried per update
    max_feedback: Optional[int] = 1400              # max total labeled pairs before stopping updates (RL-VLM-F default)
    buffer_sample_size: Optional[int] = None        # number of candidate transitions (already labelled, before filtering for training). If None, use the whole buffer.
    batch_size: int = 64                            # batch size for reward model training
    num_epochs: int = 5                             # number of epochs to train the reward model per update
    accuracy_threshold: float = 0.97                # early stopping epochs if accuracy exceeds this threshold
    alpha: float = 1.0                              # 1.0 -> pure shaped reward, 0.0 -> env reward
    relabel_rollout: bool = True                    # whether to overwrite rollout rewards in-place
    log_preview_pairs: bool = False                 # optionally log sampled pairs to wandb for debugging
    use_pbrs: bool = True                           # use Potential-Based Reward Shaping (not in original RL-VLM-F)
    buffer_capacity: int = 10000                    # capacity of the transition buffer (before sampling pairs)
    dataset_capacity: int = 10000                   # capacity of the reward model training dataset (sampled pairs with VLM label)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

# Agent configuration

@dataclass
class AgentConfig:
    """
    Configuration for the agent architecture.
    This is a base class that can be extended for specific agent types (e.g., MLP, CNN).
    """
    agent_type: AgentType = AgentType.mlp

@dataclass
class AC_MLP_AgentConfig(AgentConfig):
    """
    Configuration for the MLP agent architecture.
    """
    agent_type: AgentType = AgentType.mlp                              # Type of agent architecture (e.g., "mlp", "cnn")
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])  # Hidden layer sizes for the MLP (list can be of any length, and MLP depth will be inferred).
    activation: ActivationType = "Tanh"                                # Activation function for the MLP (e.g., "ReLU", "Tanh", "SiLU")
    actor_std: float = 0.01                                            # Standard deviation for the actor's weights initialization
    critic_std: float = 1.0                                            # Standard deviation for the critic's weights initialization
    ensemble_critic_std: float = 0.01                                  # Standard deviation for ensemble critic heads (small so heads start near-identical)

@dataclass
class AC_CNN_AgentConfig(AgentConfig):
    """
    Configuration for the CNN agent architecture.
    """
    agent_type: AgentType = AgentType.cnn                                  # Type of agent architecture (e.g., "mlp", "cnn")
    residual_activation: ActivationType = "ReLU"                           # Activation function for the residual block of the CNN
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 32]) # Number of output channels for each convolutional sequence layer (list can be of any length, and CNN depth will be inferred).
    conv_head_activation: ActivationType = "ReLU"                          # Activation function for the CNN head (last layer before the actor and critic heads)
    conv_head_out_features: int = 256                                      # Number of output features for the CNN head (flattened output that will be given to the actor and critic heads)
    actor_std: float = 0.01                                                # Standard deviation for the actor's weights initialization
    critic_std: float = 1.0                                                # Standard deviation for the critic's weights initialization
    ensemble_critic_std: float = 0.01                                      # Standard deviation for ensemble critic heads (small so heads start near-identical)
    kernel_size: int = 3                                                   # Kernel size for the convolutional layers (shared across all layers)
    stride: int = 2                                                        # Stride for the convolutional layers (shared across all layers)
    padding: int = 1                                                       # Padding for the convolutional layers (shared across all layers)
    num_critic_heads: int = 1                                              # Number of critic heads (1 = standard single critic, >1 = detached ensemble for uncertainty estimation)

@dataclass
class AC_CNN_Text_AgentConfig(AC_CNN_AgentConfig):
    """
    Configuration for the CNN agent architecture with text embeddings (e.g., for ALFWorld).
    """
    agent_type: AgentType = AgentType.cnn_text                             # Type of agent architecture (e.g., "mlp", "cnn")
    use_text: bool = True                                                  # If True, use text embeddings in the agent architecture
    vocab_size: int = 500                                                  # Vocabulary size for the text tokenizer
    use_memory: bool = True                                                # If True, use an LSTM memory in the agent architecture
    word_embedding_size: int = 32                                          # Size of the word embedding for text observations
    text_embedding_size: int = 128                                         # Size of the text embedding after the RNN

# Training configuration

@dataclass
class TrainingConfig:
    """
    Configuration for a generic RL training algorithm.
    Instantiated with a specific algorithm (e.g., PPO).
    """
    method_name: MethodName = MISSING                        # Method name (e.g., "PPO")
    learning_rate: float = 1e-3                              # Learning rate for the optimizer
    anneal_lr: bool = True                                   # If True, anneal the learning rate for policy and value networks
    gamma: float = 0.99                                      # Discount factor
    bc_coef: float = 1.0                                     # Coefficient for the behaviour cloning loss to imitate the guidance model (if VLM guidance is used)
    bc_schedule: BCoefScheduleType = BCoefScheduleType.linear # Schedule for the behaviour cloning coefficient (if VLM guidance with bc_coef>0 is used)

    eval_frequency : Optional[int] = 5000                    # Frequency of evaluation (in terms of global timesteps).
    eval_episodes: Optional[int] = 100                       # Number of episodes to evaluate the agent (if evaluation_frequency is not None)
    eval_num_envs: int = 10                                  # Number of parallel environments to use during evaluation
    
    checkpoint_frequency: int = 0                            # Frequency of saving model checkpoints in global timesteps (0 disables intermediate checkpoints)
    save_model: bool = False                                 # If True, save the model after training
    
    agent: AgentConfig = field(default_factory=AgentConfig)                          # Agent configuration (e.g., MLP, CNN)
    guidance_model: GuidanceModelConfig = field(default_factory=GuidanceModelConfig) # Guidance model configuration (e.g., VLM guidance)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    
    batch_size: int = MISSING                                # Batch size computed during runtime

@dataclass
class PPOConfig(TrainingConfig):
    """
    Configuration for Proximal Policy Optimization (PPO) algorithm from CleanRL.    
    The algorithm runs for a set number of iterations (computed with total_timesteps // batch_size).
    For each iteration, the agent collects num_steps steps of experience * num_envs environments,
    then updates the policy and value networks for update_epochs epochs with the collected data.
    """
    method_name: MethodName = "ppo"     # Method name
    gae_lambda: float = 0.95            # Lambda for Generalized Advantage Estimation (GAE)
    norm_adv: bool = True               # If True, normalize advantages (zero mean, unit variance)
    clip_coef: float = 0.2              # Clipping coefficient for PPO loss
    clip_vloss: bool = True             # If True, clip the value loss
    ent_coef: float = 0.01              # Entropy coefficient for the loss (mean entropy of the policy)
    vf_coef: float = 0.5                # Coefficient for the value function loss (when summing losses)
    max_grad_norm: float = 0.5          # Maximum gradient norm for clipping
    target_kl: Optional[float] = None   # Target KL divergence for early stopping (if None, no early stopping)
    num_minibatches: int = 4            # Number of minibatches for PPO
    update_epochs: int = 8              # Number of epochs (how many passes through the collected data) for PPO updates
    awbc_filter: str = "exp"             # BC weights on teacher transitions: "exp" (AWBC, exp(A/τ)), "binary" (CRR binary, 1[A>0]) or "none" (plain BC)
    awbc_temperature: float = 0.5        # Temperature τ for the exponential filter (only used when awbc_filter="exp")
    awbc_return_filter: bool = False     # Gate BC on episode return (zero out BC for transitions from failed episodes)
    bc_return_threshold: float = 0.0     # Threshold for return filter (0.0 works for sparse reward envs)
    exclude_value_loss_when_guided: bool = False  # If True, exclude value loss on guided transitions (only BC active when guided, only PPO when unguided)

    agent: AgentConfig = field(default_factory=AgentConfig)  # Agent configuration for AC (PPO)
    minibatch_size: int = 0             # Minibatch size computed during runtime
    num_iterations: int = 0             # Number of iterations computed during runtime (as total_timesteps // batch_size)


@dataclass
class DaggerVLMConfig(TrainingConfig):
    """
    Configuration for the DAgger-VLM imitation baseline.
    The loop collects states under a beta-mixture of VLM teacher and student policy,
    aggregates VLM-labeled state-action pairs, and trains the student with BC only.
    """
    method_name: MethodName = MethodName.dagger_vlm
    buffer_size: int = 10000
    bc_batch_size: int = 128
    bc_updates_per_iter: int = 100
    beta_schedule: DaggerBetaScheduleType = DaggerBetaScheduleType.exponential
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay: float = 0.5
    beta_warmup_iters: int = 10
    max_grad_norm: float = 0.5
    rollout_log_interval: int = 16

    agent: AgentConfig = field(default_factory=AgentConfig)
    num_iterations: int = 0


# Logger configuration

@dataclass
class LoggerConfig:
    """
    Configuration for logging in wandb.
    """
    wandb_mode: WandBLoggerMode = WandBLoggerMode.offline # Whether to track the experiment in WandB ["online", "offline", "disabled"]
    wandb_project_name: str = "sage"                      # Name of the WandB project
    wandb_entity: Optional[str] = None                    # Entity name for WandB (user or team)
    wandb_logging_frequency: int = 48                     # Frequency of logging to WandB in global timesteps (must be a multiple of num_envs)
    capture_video: bool = False                          # Whether to capture video of the agent's performance
    vllm_verbose: bool = False                           # Whether to enable verbose logging for VLLM

# Experiment configuration

@dataclass
class GPUConfig:
    """
    Configuration for GPU devices, specifically where to load each model.
    All GPU IDs are given in terms of indices of CUDA_VISIBLE_DEVICES.
    For example, if CUDA_VISIBLE_DEVICES="3,4", then rl_gpu_id=0 refers to GPU 3 and vlm_gpu_ids=[0, 1] refers to GPUs 3 and 4.
    """
    rl_gpu_id: Optional[int] = None         # GPU ID for the RL agent (defaults to the first visible GPU)
    vlm_gpu_ids: Optional[List[int]] = None # GPU IDs for the VLM guidance model (defaults to all visible GPUs)

@dataclass
class ExperimentConfig:
    """
    Configuration for the entire experiment.
    """
    exp_name: str = MISSING            # Name of the experiment
    seed: int = 0                      # Random seed for reproducibility
    eval_seed: int = 10000             # Random seed for evaluation
    torch_deterministic: bool = True   # Whether to use deterministic mode in PyTorch
    cuda: bool = True                  # CUDA support
    vlm_guidance: bool = True          # If true, loads a guidance model (VLM). This can be used either for reward shaping or for policy guidance (or both) depending on the training config.
    
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    gpu_device_map: GPUConfig = field(default_factory=GPUConfig)

def register_configs() -> None:
    """
    Register the configuration classes with Hydra's ConfigStore.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_experiment", node=ExperimentConfig)
    cs.store(group="logger", name="base_logger", node=LoggerConfig)
    cs.store(group="env", name="base_environment", node=EnvironmentConfig)
    cs.store(group="env/prompt", name="base_prompt", node=PromptConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="training", name="ppo_training", node=PPOConfig)
    cs.store(group="training", name="dagger_vlm_training", node=DaggerVLMConfig)
    cs.store(group="training/agent", name="base_agent", node=AgentConfig)
    cs.store(group="training/agent", name="ac_mlp_agent", node=AC_MLP_AgentConfig)
    cs.store(group="training/agent", name="ac_cnn_agent", node=AC_CNN_AgentConfig)
    cs.store(group="training/agent", name="ac_cnn_text_agent", node=AC_CNN_Text_AgentConfig)
    cs.store(group="training/guidance_model", name="base_guidance_model", node=GuidanceModelConfig)
    cs.store(group="training/reward_shaping", name="base_reward_shaping", node=RewardShapingConfig)

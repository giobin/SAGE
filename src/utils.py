import torch
import random
import numpy as np
import os
import re
import time
from typing import Any

from src.config import MethodName, AgentType, GuidanceModelType
from PIL import Image
from omegaconf import DictConfig, OmegaConf

def extract_hydra_suffix(hydra_experiment_name: str) -> str:
    """
    Given a hydra_experiment_name (full output_dir path),
    extract the suffix string like '2025-09-30_00-18-53_9' or '2025-09-30_00-18-53'.
    """
    # split path into parts
    parts = hydra_experiment_name.strip(os.sep).split(os.sep)
    
    # regex patterns
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # e.g. 2025-09-30
    time_pattern = re.compile(r"^\d{2}-\d{2}-\d{2}$")  # e.g. 00-18-53
    num_pattern = re.compile(r"^\d+$")                 # e.g. 9
    
    suffix_parts = []
    for p in parts:
        if date_pattern.match(p) or time_pattern.match(p) or num_pattern.match(p):
            suffix_parts.append(p)
    
    return "_".join(suffix_parts) if suffix_parts else int(time.time())


def get_exp_name(config: DictConfig):
    parts = []

    # 1. Environment
    parts.append(config.env.env_id)
    parts.append("img" if config.env.image_obs else "sym")

    # 2. Algorithm + agent
    method = config.training.method_name.value  # e.g. "ppo"
    agent = config.training.agent.agent_type.value  # e.g. "cnn", "cnn_text"
    parts.append(f"{method}_{agent}")

    # 3. Critic ensemble
    num_heads = config.training.agent.get("num_critic_heads", 1)
    if num_heads > 1:
        parts.append(f"K{num_heads}")

    # 4. Guidance
    if config.vlm_guidance:
        gm = config.training.guidance_model
        model_tag = f"{gm.model_id}".split("/")[-1]

        if method == MethodName.dagger_vlm.value:
            if config.env.prompt.use_cot_prompt:
                model_tag += "_cot"
            beta_schedule = config.training.beta_schedule
            beta_schedule_str = beta_schedule.value if hasattr(beta_schedule, "value") else str(beta_schedule)
            beta_tag = f"{beta_schedule_str}_b{config.training.beta_start}"
            if beta_schedule_str == "exponential":
                beta_tag += f"_p{config.training.beta_decay}"
            elif beta_schedule_str == "linear":
                beta_tag += f"_warm{config.training.beta_warmup_iters}"
            parts.append(f"{model_tag}_dagger_{beta_tag}_buf{config.training.buffer_size}")
        elif gm.do_policy_guidance:
            # Uncertainty gating
            um = gm.get("uncertainty_method", "entropy")
            um_str = um.value if hasattr(um, "value") else str(um)
            gate_tag = f"{um_str}_thr{gm.guidance_threshold}"
            if gm.get("kappa", 1.0) < 1.0:
                gate_tag += f"_kap{gm.kappa}"

            # Guidance type (action / distribution / full)
            gt = gm.guidance_type
            gt_str = gt.value if hasattr(gt, "value") else str(gt)

            # BC / AWBC
            bc = config.training.bc_coef
            bc_sched = config.training.bc_schedule
            bc_sched_str = bc_sched.value if hasattr(bc_sched, "value") else str(bc_sched)
            awbc_filter = config.training.awbc_filter
            bc_tag = f"bc{bc}_{bc_sched_str}_{awbc_filter}"
            if awbc_filter == "exp":
                bc_tag += f"_T{config.training.awbc_temperature}"
            if config.training.get("awbc_return_filter", False):
                bc_tag += f"_retfilt{config.training.get('bc_return_threshold', 0.0)}"

            if config.env.prompt.use_cot_prompt:
                model_tag += "_cot"

            parts.append(f"{model_tag}_{gt_str}_{gate_tag}_{bc_tag}")
        else:
            parts.append(f"{model_tag}_no_pg")

        # Reward shaping
        if config.training.reward_shaping.enabled:
            rs = config.training.reward_shaping
            parts.append(f"RS_a{rs.alpha}_pbrs{int(rs.use_pbrs)}_upd{rs.update_interval}")

        # Cache
        if gm.cache.use_cache:
            parts.append(f"{gm.cache.cache_type.value}cache")
    else:
        parts.append("no_guidance")

    # 5. Seed
    parts.append(f"s{config.seed}")

    return "_".join(parts)


def set_seed(seed: int, torch_deterministic: bool) -> None:
    """
    Set the random seed for reproducibility across different libraries.
    Args:
        seed (int): The seed value to set.
        torch_deterministic (bool): If True, sets torch.backends.cudnn.deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def get_agent_class(method_name: str, agent_type: str, discrete_env: bool) -> torch.nn.Module:
    """
    Get the agent class based on the method name and agent type.
    Args:
        method_name (str): The name of the method (e.g., "PPO", "DAgger-VLM").
        agent_type (str): The type of agent (e.g., "MLP", "CNN").
        discrete_env (bool): Whether the environment has a discrete action space.
    Returns:
        torch.nn.Module: The corresponding agent class.
    """
    supported_methods = [method.value.lower() for method in MethodName]
    supported_agents = [agent.value.lower() for agent in AgentType]
    if method_name.value.lower() not in supported_methods:
        raise NotImplementedError(f"Method '{method_name.value.lower()}' is not supported. Supported methods: {supported_methods}")
    if agent_type.value.lower() not in supported_agents:
        raise NotImplementedError(f"Agent type '{agent_type.value.lower()}' is not supported. Supported agents: {supported_agents}")
    
    # Actor-critic policy methods
    if method_name.value.lower() in {"ppo", "dagger_vlm"}:
        if not discrete_env:
            raise NotImplementedError("Only discrete action spaces are supported.")
        if agent_type.value.lower() == "mlp":
            from src.agents.ac_mlp import AC_MLP_Agent
            return AC_MLP_Agent
        elif agent_type.value.lower() == "cnn":
            from src.agents.ac_cnn import AC_CNN_Agent
            return AC_CNN_Agent
        elif agent_type.value.lower() == "cnn_text":
            from src.agents.ac_cnn_text import AC_CNN_Text_Agent
            return AC_CNN_Text_Agent
        else:
            raise NotImplementedError(f"No implementation found for agent type '{agent_type.value.lower()}' and method '{method_name.value.lower()}'")
    # This should never trigger as the asserts take care of unsupported methods and agents, but if we forgot to edit something in the config this will catch it.
    else:
        raise NotImplementedError(f"No implementation found for agent type '{agent_type.value.lower()}' and method '{method_name.value.lower()}'")


def get_guidance_model_class(model_type: str, use_vllm: bool):
    """
    Returns the guidance model class based on the model_type string.
    Args:
        model_type (str): The type of the guidance model (e.g., "gemma")
    Returns:
        class: The class corresponding to the guidance model
    Raises:
        NotImplementedError: If the model type is not supported.
    """
    supported_models = [m.value.lower() for m in GuidanceModelType]

    model_type = model_type.lower()
    if model_type not in supported_models:
        raise NotImplementedError(
            f"Guidance model type '{model_type}' is not supported. "
            f"Supported types are: {supported_models}"
        )
    if model_type.lower() == GuidanceModelType.oracle.value.lower():
        from src.guidance_models.oracle_guidance import OracleGuidanceModel
        return OracleGuidanceModel
    elif model_type.lower() == GuidanceModelType.random.value.lower():
        from src.guidance_models.random_guidance import RandomGuidanceModel
        return RandomGuidanceModel
    elif use_vllm:
        assert model_type.lower() == GuidanceModelType.transformers.value.lower(), "vLLM can only be used with Transformers models."
        from src.guidance_models.vllm_model import VllmModel
        return VllmModel
    else:
        # Default to TransformersModel implementation, check inside that class which models are tested
        from src.guidance_models.transformers_model import TransformersModel
        return TransformersModel

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a PyTorch tensor to a PIL Image.
    Args:
        tensor (torch.Tensor): The input tensor, expected to be either in CHW format (C=1 or 3) or HWC format.
    Returns:
        Image.Image: The converted PIL Image.
    """
    if tensor.device.type == 'cuda':
        tensor = tensor.detach().cpu()
    # if there is a batch dimension (ndim = 4), assume batch is the first, squeeze if = 1
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # ensure [0, 255] range for image tensors
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255).byte() 
        
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)  # CHW to HWC
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        pass  # already HWC
    else:
        raise ValueError("Invalid image tensor shape")
    return Image.fromarray(tensor.numpy())


def detach_to_cpu(value: Any) -> Any:
    """Return a detached CPU copy of tensors or numpy arrays for safe storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return np.copy(value)
    if isinstance(value, dict):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(v) for v in value)
    return value

import os
import json
import time
import torch
import numpy as np
import gymnasium as gym

from omegaconf import DictConfig
from typing import Callable, Optional, Tuple, List, Dict
from src.agents.base_agents import AC_Agent
from gymnasium.spaces import Dict as DictSpace
from src.wandb_compat import wandb

def evaluate(
        agent: AC_Agent,
        config: DictConfig, 
        eval_envs: Callable, 
        verbose: bool = False
    ) -> Tuple[list[float], float, float]:
    """
    Evaluate the agent using vectorized environments.
    Args:
        agent (AC_Agent): The agent to evaluate.
        config (DictConfig): Configuration for the environment and evaluation parameters.
        eval_envs (Callable): Function to create the evaluation environment.
        verbose (bool): Whether to print detailed logs during evaluation.
    Returns:
        Tuple[list[float], float, float]: A tuple containing:
            - List of episodic returns from the evaluation episodes.
            - Throughput (steps per second).
            - Total elapsed time for the evaluation.
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    
    # Configuration
    num_eval_episodes = config.training.eval_episodes
    num_envs = min(config.training.eval_num_envs, num_eval_episodes)
    base_seed = config.get("eval_seed", config.get("seed", 0) + 10000)
    agent.eval()
    
    episodic_returns = []
    current_returns = np.zeros(num_envs, dtype=np.float32) 
    episodes_completed = 0
    total_steps_count = 0
    
    # Initial Reset
    initial_seeds = [base_seed + i for i in range(num_envs)]
    obs, _ = eval_envs.reset(seed=initial_seeds)
    
    use_memory = config.training.agent.get("use_memory", False)
    memory = (
        torch.zeros((num_envs, agent.memory_size), device=agent.device)
        if use_memory else None
    )

    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            while episodes_completed < num_eval_episodes:
                obs_tensor = obs_to_tensor(obs, agent.device)
                
                # Greedy action
                if use_memory:
                    _, a_probs, _, _, _, memory = agent.get_action_and_value(
                        obs_tensor, memory=memory
                    )
                else:
                    _, a_probs, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = torch.argmax(a_probs.probs, dim=1).cpu().numpy()

                # Environment Step
                next_obs, reward, terminated, truncated, infos = eval_envs.step(action)
                
                # Update trackers
                total_steps_count += num_envs
                current_returns += reward
                done = terminated | truncated
                
                if np.any(done):
                    for i, is_done in enumerate(done):
                        if is_done:
                            if episodes_completed < num_eval_episodes:
                                final_return = current_returns[i]
                                episodic_returns.append(final_return)
                                episodes_completed += 1
                            
                            current_returns[i] = 0.0
                            if use_memory:
                                memory[i].zero_()

                obs = next_obs
                
    finally:
        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    throughput = total_steps_count / elapsed_time if elapsed_time > 0 else 0

    if episodic_returns:
        mean_return = sum(episodic_returns) / len(episodic_returns)
        
        log_msg = (
            f"Eval Results: Mean Return={mean_return:.2f} | "
            f"Time={elapsed_time:.2f}s | "
            f"Speed={throughput:.0f} steps/s"
        )
        print(log_msg)

    agent.train()
    return episodic_returns, throughput, elapsed_time

def save_checkpoint(model: torch.nn.Module, iteration: Optional[int] = None, subfolder_name: Optional[str] = "checkpoints", pt_name: Optional[str] = "agent") -> str:
    """
    Save the agent's state dictionary to a checkpoint file.
    Args:
        model (torch.nn.Module): The model whose state dictionary is to be saved.
        iteration (Optional[int]): Iteration number for naming the checkpoint file. If None, uses 'agent.pt'.
        subfolder_name (Optional[str]): Subfolder name for organizing checkpoints. Default is "checkpoints".
        pt_name (Optional[str]): Base name for the checkpoint file. Default is "agent".
    Returns:
        str: The path to the saved checkpoint file.
    """
    # adapted from https://docs.cleanrl.dev/advanced/resume-training/
    model_path = f'{pt_name}.pt' if iteration is None else f'{pt_name}_{iteration}.pt'
    checkpoint_path = f"{wandb.run.dir}/{subfolder_name}/{model_path}" if subfolder_name else f"{wandb.run.dir}/{model_path}"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
     
    wandb.save(checkpoint_path, policy="now")

    return checkpoint_path

def append_metrics(row: Dict) -> None:
    """
    Append one row of run metrics to metrics.jsonl in the Hydra output directory (read by scripts/summarize_results.py).
    """
    from hydra.core.hydra_config import HydraConfig
    path = os.path.join(HydraConfig.get().runtime.output_dir, "metrics.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def get_gpu_device_map(device_map_config: DictConfig) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Get the GPU ids (indices into the visible devices) for the RL agent and the VLM.
    Defaults to the first visible GPU for the RL agent and all visible GPUs for the VLM; returns (None, None) without GPUs.
    Note that vLLM always uses all visible GPUs (set CUDA_VISIBLE_DEVICES to restrict them).
    """
    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        return None, None

    if device_map_config.get("rl_gpu_id", None) is not None:
        assert device_map_config.rl_gpu_id in gpu_ids, f"RL GPU ID is {device_map_config.rl_gpu_id}, but there are only {len(gpu_ids)} visible GPUs."
    if device_map_config.get("vlm_gpu_ids", None) is not None:
        assert all(vlm_gpu_id in gpu_ids for vlm_gpu_id in device_map_config.vlm_gpu_ids), \
            f"VLM GPU IDs are {device_map_config.vlm_gpu_ids}, but there are only {len(gpu_ids)} visible GPUs."

    rl_agent_gpu_id = gpu_ids[0] if device_map_config.get("rl_gpu_id", None) is None else device_map_config.rl_gpu_id
    vlm_gpu_ids = gpu_ids if device_map_config.get("vlm_gpu_ids", None) is None else device_map_config.vlm_gpu_ids

    return rl_agent_gpu_id, vlm_gpu_ids

def obs_to_tensor(obs: torch.Tensor | DictSpace, device: torch.device) -> torch.Tensor | Dict:
    """
    Convert observations to tensors and move them to the specified device.
    If the observation is a dictionary, convert each value to a tensor (if possible) and move it to the specified device.
    
    Args:
        obs (torch.Tensor | DictSpace): The observation to convert, which can be a tensor or a dictionary of values (tensors or not).
        device (torch.device): The device to move the tensors to (e.g., 'cuda' or 'cpu').
    Returns:
        torch.Tensor | Dict: The converted observation, either as a tensor or a dictionary of tensors
        
    """
    if isinstance(obs, dict):
        for k, v in obs.items():
                if hasattr(v, "shape") and v.shape is not None:
                    obs[k] = torch.tensor(obs[k], dtype=torch.float32).to(device)
                else:
                    # For text or discrete spaces, store as an object array (list of lists of strings or ints)
                    obs[k] = np.array(obs[k], dtype=object)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
    else:
        return torch.Tensor(obs).to(device) if hasattr(obs, "shape") and obs.shape is not None else obs
    
def flatten_batch_obs(x: torch.Tensor | Dict) -> torch.Tensor | Dict:
    """
    Flatten the batch dimension of the observation (to remvoe the n_envs dimension).
    If the input is a dictionary, flatten each tensor and np array in the dictionary.
    Args:
        x (torch.Tensor | Dict): The observation to flatten, which can be a tensor or a dictionary of tensors.
    Returns:
        torch.Tensor | Dict: The flattened observation, either as a tensor or a dictionary of tensors.
    """ 
    if isinstance(x, dict):
        return {k: flatten_batch_obs(v) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.reshape((-1,) + x.shape[2:])
    elif isinstance(x, np.ndarray):
        return x.reshape((-1,) + x.shape[2:])
    elif isinstance(x, list):
        # If it's a list, we assume it's a list of tensors or np arrays, flatten each element
        return [flatten_batch_obs(item) for item in x]
    else:
        return x

def batch_index(obs: torch.Tensor | Dict, indices: torch.Tensor | List[int]) -> torch.Tensor | Dict:
    """
    Index the observation with the given indices.
    If the observation is a dictionary, index each tensor in the dictionary.
    Args:
        obs (torch.Tensor | Dict): The observation to index, which can be a tensor or a dictionary of tensors.
        indices (torch.Tensor | List[int]): The indices to use for indexing the observation.
    Returns:
        torch.Tensor | Dict: The indexed observation, either as a tensor or a dictionary of tensors
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy() if indices.is_cuda else indices.numpy()
    if isinstance(obs, dict):
        return {k: v[indices] for k, v in obs.items()}
    elif torch.is_tensor(obs):
        return obs[indices]
    else:
        # fallback, e.g. if obs is something else like text or numpy array
        # try indexing if possible, else return as is
        try:
            return obs[indices]
        except Exception:
            return obs

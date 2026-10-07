import os
import torch
import torch.nn as nn
import gymnasium as gym

from torch import Tensor

from abc import ABC, abstractmethod
from omegaconf import DictConfig

class AC_Agent(ABC, nn.Module):
    """
    Abstract base class for actor-critic agents in a vectorized environment.
    An agent encapsulates a specific architecture for a policy and value function estimator. 
    """
    @abstractmethod
    def __init__(self, envs: gym.vector.VectorEnv, agent_config: DictConfig) -> None:
        """
        Initialize the agent with the vectorized environment.
        Args:
            envs (gym.vector.VectorEnv): The vectorized environment the agent will interact with (used for observation and action spaces).
            agent_config (DictConfig): Agent-specific configuration parameters (e.g. hidden layer sizes, activation functions).
        """
        super().__init__()

    @abstractmethod
    def get_value(self, x: Tensor) -> Tensor:
        """
        Compute state‐value estimate.
        Args:
            x (Tensor): Input tensor representing the state.
        Returns:
            Tensor: The estimated value of the state.
        """
        pass

    @abstractmethod
    def get_action_and_value(self, x: Tensor, action: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute action, log_prob, entropy, and state‐value.
        Args:
            x (Tensor): Input tensor representing the state.
            action (Tensor, optional): Action tensor. If None, a new action is sampled from the policy.
        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - action (Tensor): The action to take.
                - probs (Tensor): The action probabilities according to the policy.
                - log_prob (Tensor): The log probability of the action according to the policy.
                - entropy (Tensor): The entropy of the action distribution.
                - value (Tensor): The estimated value of the state.
        """
        pass
    
    def get_value_ensemble(self, x: Tensor, **kwargs) -> Tensor:
        """
        Returns stacked value predictions from all critic heads: shape (batch, K).
        Default implementation returns just the primary critic (K=1).
        Subclasses with ensemble heads should override this.
        """
        return self.get_value(x, **kwargs).unsqueeze(-1)

    def get_value_stats(self, x: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns (v_mean, v_std, v_min) from the critic ensemble.
        With a single critic, v_std is zero and v_min == v_mean.
        """
        values = self.get_value_ensemble(x, **kwargs)  # (batch, K)
        if values.shape[-1] == 1:
            v = values.squeeze(-1)
            return v, torch.zeros_like(v), v
        return values.mean(dim=-1), values.std(dim=-1), values.min(dim=-1).values

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the agent's parameters are located.
        Returns:
            torch.device: The device of the agent's parameters (e.g., 'cpu' or 'cuda').
        """
        return next(self.parameters()).device

    def load_checkpoint(self, checkpoint_path: os.PathLike) -> None:
        """
        Loads model weights from a checkpoint file (.pt)
        Args:
            checkpoint_path (os.PathLike): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint)


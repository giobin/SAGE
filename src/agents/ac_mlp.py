# adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot
from omegaconf import DictConfig
from gymnasium.spaces import Discrete, Box

from src.agents.common import layer_init
from src.agents.base_agents import AC_Agent


class AC_MLP_Agent(AC_Agent):
    def __init__(self, envs, agent_cfg: DictConfig, verbose=False) -> None:
        super().__init__(envs, agent_cfg)
        
        vprint = print if verbose else lambda *args, **kwargs: None
        
        # One-hot encoding for discrete observation space (where the observation is a single integer)
        # Specifically required to work with FrozenLake-v1 and similar toy text environments.
        if isinstance(envs.single_observation_space, Discrete):
            self.obs_shape = envs.single_observation_space.n
            self._one_hot_obs = True
        else:
            self.obs_shape = int(np.prod(envs.single_observation_space.shape))
            self._one_hot_obs = False
        
        n_actions = envs.single_action_space.n
        vprint(f"AC_MLP_Agent: obs_shape={self.obs_shape}, n_actions={n_actions}, one_hot_obs={self._one_hot_obs}")

        hidden_sizes = list(agent_cfg.hidden_sizes)
        actor_std = float(agent_cfg.actor_std)
        critic_std = float(agent_cfg.critic_std)
        
        act = agent_cfg.get("activation")
        Activation = getattr(nn, act.value if act else "Tanh", None)
        assert Activation is not None, f"Activation function '{act}' is not supported. Please use a valid activation function from torch.nn."

        critic_layers = []
        in_dim = self.obs_shape
        for h in hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(in_dim, h)))
            critic_layers.append(Activation())
            vprint(f"AC_MLP_Agent: critic layer {len(critic_layers)}: {in_dim} -> {h}")
            in_dim = h
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=critic_std))
        vprint(f"AC_MLP_Agent: critic layer {len(critic_layers)}: {in_dim} -> 1")
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = []
        in_dim = self.obs_shape
        for h in hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(in_dim, h)))
            vprint(f"AC_MLP_Agent: actor layer {len(actor_layers)}: {in_dim} -> {h}")
            actor_layers.append(Activation())
            in_dim = h
        actor_layers.append(layer_init(nn.Linear(in_dim, n_actions), std=actor_std))
        vprint(f"AC_MLP_Agent: actor layer {len(actor_layers)}: {in_dim} -> {n_actions}")
        self.actor = nn.Sequential(*actor_layers)
        
    def _preprocess(self, x: Tensor) -> Tensor:
        if self._one_hot_obs:
            # x might be shape (batch,) or (batch,1)
            x = x.long().view(-1) # (batch,)
            x = one_hot(x, num_classes=self.obs_shape).float()
        else:
            x = x.view(x.size(0), -1)
        return x

    def get_value(self, x: Tensor) -> Tensor:
        x = self._preprocess(x)
        return self.critic(x)

    def get_action_and_value(self, x: Tensor, action: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self._preprocess(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs, probs.log_prob(action), probs.entropy(), self.critic(x)

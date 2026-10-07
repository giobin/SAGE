# adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Dict

from src.agents.base_agents import AC_Agent
from src.agents.common import layer_init, ConvSequence

class AC_CNN_Agent(AC_Agent):
    def __init__(self, envs, agent_cfg: DictConfig):
        super().__init__(envs, agent_cfg)
        # infer channel-first shape (C, H, W) from various possible obs shapes
        obs_space = envs.single_observation_space
        
        # Support for minigrid environments
        if isinstance(obs_space, Dict):
            obs_space = obs_space["image"] # Text and other observations are processed in downstream agent
        obs_shape = obs_space.shape
        
        if len(obs_shape) == 3:
            # either (C, H, W) or (H, W, C)
            if obs_shape[0] in (1, 3):    # likely channels-first
                c, h, w = obs_shape
            else:                         # assume HWC
                h, w, c = obs_shape
        elif len(obs_shape) == 4:
            # stacked frames, e.g. (N, H, W, C)
            n, h, w, c = obs_shape
            c = n * c
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in agent_cfg.conv_channels:
            conv_seq = ConvSequence(shape, out_channels, 
                                    kernel_size=agent_cfg.kernel_size,
                                    padding=agent_cfg.padding,
                                    stride=agent_cfg.stride,
                                    residual_activation=agent_cfg.get("residual_activation", nn.functional.relu)
                                    )
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        # CNN head
        act = agent_cfg.get("conv_head_activation")
        Activation = getattr(nn, act.value if act else "ReLU", None)
        assert Activation is not None, f"Activation function '{act}' is not supported. Please use a valid activation function from torch.nn."
        conv_seqs += [
            nn.Flatten(),
            Activation(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=agent_cfg.conv_head_out_features),
            Activation(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        # Compute embedding size with a forward pass (used downstream for minigrid)
        with torch.no_grad():
            dummy_input = Tensor(1, *obs_shape)
            self.image_embedding_size = self.network(self._preprocess(dummy_input)).shape[1]
        
        self.build_actor(agent_cfg.conv_head_out_features, envs, agent_cfg)
        self.critic = layer_init(nn.Linear(agent_cfg.conv_head_out_features, 1), std=agent_cfg.critic_std)

        # Detached critic ensemble for uncertainty estimation
        # K includes the primary critic, so we add K-1 extra heads
        # Ensemble heads use small init (std=0.01) so they start near-identical;
        # gradient noise will naturally diversify them during training.
        self.num_critic_heads = agent_cfg.get("num_critic_heads", 1)
        ensemble_std = agent_cfg.get("ensemble_critic_std", 0.01)
        if self.num_critic_heads > 1:
            self.critic_ensemble = nn.ModuleList([
                layer_init(nn.Linear(agent_cfg.conv_head_out_features, 1), std=ensemble_std)
                for _ in range(self.num_critic_heads - 1)
            ])

    def build_actor(self, head_out_features, envs, agent_cfg):
        self.actor = layer_init(
            nn.Linear(head_out_features, envs.single_action_space.n),
            std=agent_cfg.actor_std,
        )
    
    def _preprocess(self, x: Tensor) -> Tensor:
        # Dict safeguard (MiniGrid)
        x = x['image'] if (isinstance(x, dict)) else x

        # If input has 5 dims, assume stacked frames (b, n, h, w, c)
        if x.dim() == 5:
            b, n, h, w, c = x.shape
            # Bring (n, c) next to each other, then merge to channels
            x = x.permute(0, 2, 3, 1, 4).contiguous().view(b, h, w, n * c)  # (b, h, w, n*c)
        # Normalize pixels to [0,1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        # BHWC -> BCHW
        return x.permute(0, 3, 1, 2)

    def get_value(self, x, **kwargs):
        return self.critic(self.network(self._preprocess(x)))

    def get_value_ensemble(self, x, **kwargs):
        """Returns all K value predictions stacked: (batch, K)."""
        hidden = self.network(self._preprocess(x))
        v_main = self.critic(hidden)  # (batch, 1) — backprops into backbone
        if self.num_critic_heads <= 1:
            return v_main
        h_det = hidden.detach()  # CRITICAL: detach so ensemble heads don't affect backbone
        v_others = [head(h_det) for head in self.critic_ensemble]  # each (batch, 1)
        return torch.cat([v_main] + v_others, dim=-1)  # (batch, K)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(self._preprocess(x))
        logits = self.actor(hidden).clamp(-20, 20)
        dist = Categorical(logits=logits.float())
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).to(logits.dtype)
        ent = dist.entropy().to(logits.dtype)
        return action, dist, logp, ent, self.critic(hidden)

# Code adapted from https://github.com/lcswillems/rl-starter-files/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np

from torch.distributions.categorical import Categorical
from src.agents.ac_cnn import AC_CNN_Agent
from omegaconf import DictConfig

class SimpleTokenizer:
    """
    Simple hashing tokenizer that can handle any vocabulary.
    Maps words to a fixed range of indices [1, vocab_size].
    Used instead of a pretrained tokenizer (e.g. tiktoken) to keep the vocab size small.
    """
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.pad_token_idx = 0

    def encode(self, text: str) -> list[int]:
        tokens = text.lower().replace('.', ' ').replace(',', ' ').split()
        indices = []
        for token in tokens:
            hash_digest = hashlib.md5(token.encode('utf-8')).hexdigest()
            # modulo vocab_size (start at 1 to save 0 for padding)
            idx = (int(hash_digest, 16) % (self.vocab_size - 1)) + 1
            indices.append(idx)
        return indices
    
    def pad_sequences(self, sequences, max_length=None):
        if max_length is None:
            max_length = max(len(seq) for seq in sequences) if sequences else 0
        padded_sequences = []
        for seq in sequences:
            seq = seq[:max_length]
            padded_seq = seq + [0] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
        return padded_sequences

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class AC_CNN_Text_Agent(AC_CNN_Agent):
    def __init__(self, envs, agent_cfg: DictConfig):
        super().__init__(envs, agent_cfg)
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space

        # Decide which components are enabled
        self.use_text = agent_cfg.get("use_text", False)
        self.use_memory = agent_cfg.get("use_memory", False)

        # Add direction embedding if needed
        if "direction" in obs_space:
            self.use_direction = True
            self.lstm_input_size = self.image_embedding_size + 4
        else:
            self.use_direction = False
            self.lstm_input_size = self.image_embedding_size
        
        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.lstm_input_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            if not hasattr(obs_space, "spaces") or "mission" not in obs_space.spaces:
                raise ValueError("Mission field missing from observation; text cnn is designed for MiniGrid missions.")
            self.tokenizer = SimpleTokenizer(vocab_size=agent_cfg.get("vocab_size", 500))
            self.word_embedding_size = agent_cfg.word_embedding_size
            self.text_embedding_size = agent_cfg.text_embedding_size
            self.vocab_size = self.tokenizer.vocab_size
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_size, padding_idx=self.tokenizer.pad_token_idx)
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size # Same as image embedding size
        if not self.use_memory and self.use_direction:
            self.embedding_size += 4  # Add direction dims if not absorbed by LSTM
        if self.use_text:
            self.embedding_size += self.text_embedding_size
        
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Detached critic ensemble
        self.num_critic_heads = agent_cfg.get("num_critic_heads", 1)
        if self.num_critic_heads > 1:
            self.critic_ensemble = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embedding_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                for _ in range(self.num_critic_heads - 1)
            ])

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def _get_embed_text(self, text):
        if isinstance(text, torch.Tensor):
            text = text.tolist()
        text_list = [str(t) for t in text] if text.shape != () else [str(text)]
        tokenized_batch = [self.tokenizer.encode(t) for t in text_list]
        padded_batch = self.tokenizer.pad_sequences(tokenized_batch)
        tokenized_text = torch.tensor(padded_batch, dtype=torch.long).to(self.device)
        embedded_text = self.word_embedding(tokenized_text)
        _, hidden = self.text_rnn(embedded_text)
        return hidden[-1]
    
    def _get_embed_direction(self, direction):
        if isinstance(direction, torch.Tensor):
            direction = direction.to(self.device)
        else:
            direction = torch.tensor(direction, dtype=torch.long, device=self.device)

        one_hot = F.one_hot(direction, num_classes=4).float()
        return one_hot

    def _concat_admissible_commands(self, missions, infos):
        if infos is None or "admissible_commands" not in infos:
            return missions

        header = "**Admissible Commands**:"
        if isinstance(missions, np.ndarray):
            missions_list = missions.tolist()
        else:
            missions_list = missions

        if not isinstance(missions_list, list):
            missions_list = [missions_list]

        cmds_per_env = infos.get("admissible_commands")
        if isinstance(cmds_per_env, np.ndarray):
            cmds_per_env = cmds_per_env.tolist()
        if not isinstance(cmds_per_env, list):
            cmds_per_env = [cmds_per_env] * len(missions_list)

        combined = []
        for idx, mission in enumerate(missions_list):
            cmds = cmds_per_env[idx] if idx < len(cmds_per_env) else None
            if cmds is None:
                combined.append(str(mission))
                continue
            if isinstance(cmds, np.ndarray):
                cmds = cmds.tolist()
            if not isinstance(cmds, list):
                cmds = [cmds]
            cmds_text = "\n".join([str(cmd) for cmd in cmds])
            if cmds_text:
                combined.append(f"{mission}\n{header}\n{cmds_text}")
            else:
                combined.append(str(mission))

        return np.array(combined, dtype=object)

    def _get_embedding(self, obs, memory=None, infos=None):
        img = obs["image"] if isinstance(obs, dict) else obs
        x = self._preprocess(img)
        x = self.network(x) # CNN forward pass
        x = x.reshape(x.shape[0], -1)
        
        if self.use_direction:
            embed_direction = self._get_embed_direction(obs["direction"])
            x = torch.cat((x, embed_direction), dim=1)

        if self.use_memory and memory is not None:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            new_memory = torch.cat(hidden, dim=1)
        else:
            embedding = x # img + direction
            new_memory = memory

        if self.use_text:
            missions = obs["mission"]
            missions = self._concat_admissible_commands(missions, infos)
            embed_text = self._get_embed_text(missions)
            embedding = torch.cat((embedding, embed_text), dim=1)

        return embedding, new_memory

    def get_value(self, obs, memory=None, infos=None, **kwargs):
        embedding, _ = self._get_embedding(obs, memory, infos=infos)
        value = self.critic(embedding)
        return value

    def get_value_ensemble(self, obs, memory=None, infos=None, **kwargs):
        """Returns all K value predictions stacked: (batch, K)."""
        embedding, _ = self._get_embedding(obs, memory, infos=infos)
        v_main = self.critic(embedding)  # (batch, 1) — backprops into backbone
        if self.num_critic_heads <= 1:
            return v_main
        e_det = embedding.detach()  # CRITICAL: detach so ensemble heads don't affect backbone
        v_others = [head(e_det) for head in self.critic_ensemble]  # each (batch, 1)
        return torch.cat([v_main] + v_others, dim=-1)  # (batch, K)

    def get_action_and_value(self, obs, memory=None, action=None, infos=None):
        embedding, new_memory = self._get_embedding(obs, memory, infos=infos)
        logits = self.actor(embedding).clamp(-20, 20)
        probs = Categorical(logits=logits.float())
        if action is None:
            action = probs.sample()
        value = self.critic(embedding)
        if self.use_memory:
            return action, probs, probs.log_prob(action).to(logits.dtype), probs.entropy().to(logits.dtype), value, new_memory
        else:
            return action, probs, probs.log_prob(action).to(logits.dtype), probs.entropy().to(logits.dtype), value

import numpy as np
import gymnasium as gym

from gymnasium import spaces
from src.environments.text_descriptions import info_to_text_obs
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, LEFT, DOWN, RIGHT, UP

class VisualObsWrapper(gym.Wrapper):
    """
    Wrapper that returns visual observations from the environment.
    """
    def __init__(self, env, transform=None):
        super(VisualObsWrapper, self).__init__(env)
        self.transform = transform
        dummy_obs, _ = self.reset()
        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def _render_transform(self):
        obs = self.env.render()
        if self.transform is not None:
            obs = self.transform(obs)
        return obs / 255.0  # Normalize to [0, 1] range

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._render_transform()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._render_transform()
        return obs, reward, done, truncated, info
    
class FrozenLakeRandomResetWrapper(gym.Wrapper):
    """
    Wrapper for FrozenLake environments to generate a new random map on reset, controlled by a given initial random seed.
    If no seed is provided, a random one will be generated.
    """
    def __init__(self, env, map_size, seed=None, max_rand=10000, **kwargs):
        super().__init__(env)
        self.max_rand = max_rand
        if seed is None:
            seed = np.random.randint(0, self.max_rand)
            print(f"No seed provided for FrozenLakeRandomResetWrapper, using random seed {seed}")
        self.seed = seed
        assert type(map_size) == int and map_size >= 0, "FrozenLake random map size must be a non-negative integer."
        self.map_size = map_size
        self.random = np.random.RandomState(seed)
        # Save kwargs that original env does not save itself
        self.is_slippery = kwargs.get("is_slippery", False)
        self.success_rate = kwargs.get("success_rate", 1.0 / 3.0)
        self.reward_schedule = kwargs.get("reward_schedule", (1, 0, 0))

    def reset(self, **kwargs):
        # Code from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py
        env_unwrapped = self.env.unwrapped
        env_unwrapped.desc = generate_random_map(size=self.map_size, seed=self.random.randint(0, self.max_rand))
        env_unwrapped.desc = np.asarray(env_unwrapped.desc, dtype="c")
        env_unwrapped.nrow, env_unwrapped.ncol = nrow, ncol = env_unwrapped.desc.shape

        env_unwrapped.initial_state_distrib = np.array(env_unwrapped.desc == b"S").astype("float64").ravel()
        env_unwrapped.initial_state_distrib /= env_unwrapped.initial_state_distrib.sum()
        
        nA = 4
        nS = nrow * ncol
        fail_rate = (1.0 - self.success_rate) / 2.0
        env_unwrapped.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * env_unwrapped.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, env_unwrapped.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, env_unwrapped.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)
        
        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = env_unwrapped.desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = self.reward_schedule[
                b"GHF".index(new_letter if new_letter in b"GHF" else b"F")
            ]
            return new_state, reward, terminated

        for row in range(env_unwrapped.nrow):
            for col in range(env_unwrapped.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = env_unwrapped.P[s][a]
                    letter = env_unwrapped.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (
                                        self.success_rate if b == a else fail_rate,
                                        *update_probability_matrix(row, col, b),
                                    )
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
        
        return self.env.reset(**kwargs)

        
class FrozenLakeGridObsWrapper(gym.Wrapper):
    """
    Wrapper for FrozenLake environments to return the observation as a grid of integers.
    The grid represents the environment state with specific integer values:
    - 0: Start (S)
    - 1: Frozen (F)
    - 2: Hole (H)
    - 3: Goal (G)
    - 4: Agent (represented by max(char_map.values()) + 1)
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.spec.id.startswith("FrozenLake"), "This wrapper is only for FrozenLake environments."
        self.char_map = {
            b'S': 0,  # Start
            b'F': 1,  # Frozen
            b'H': 2,  # Hole
            b'G': 3,  # Goal
        }
        # agent will be represented by max(char_map.values())+1
        self.agent_value = max(self.char_map.values()) + 1

        nrow, ncol = env.unwrapped.desc.shape
        self.observation_space = spaces.Box(
            low=0,
            high=self.agent_value,
            shape=(nrow, ncol),
            dtype=np.int8
        )

    def _make_grid(self, obs):
        desc = self.env.unwrapped.desc
        nrow, ncol = desc.shape
        grid = np.zeros((nrow, ncol), dtype=int)
        for i, row in enumerate(desc):
            for j, ch in enumerate(row):
                grid[i, j] = self.char_map[ch]
        # place agent
        ar, ac = obs // ncol, obs % ncol
        grid[ar, ac] = self.agent_value
        return grid
    
    def _get_current_state(self):
        obs = self.env.unwrapped.s
        return self._make_grid(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._make_grid(obs), info

    def step(self, action, **kwargs):
        obs, reward, done, truncated, info = self.env.step(action, **kwargs)
        return self._make_grid(obs), reward, done, truncated, info
    
    
class GymCardsTextObsWrapper(gym.Wrapper):
    """
    Wrapper for gym_cards environments to convert info to text observations.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.spec.id.startswith("gym_cards"), "This wrapper is only for gym_cards environments."
        self.observation_space = gym.spaces.Text(256)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        text_obs = info_to_text_obs(self.env.spec.id, info)
        return text_obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        text_obs = info_to_text_obs(self.env.spec.id, info)
        return text_obs, reward, done, truncated, info


class GymCardsActionShiftWrapper(gym.ActionWrapper):
    """
    Wrapper for gym_cards environments that shifts actions down by one to emulate 1-indexed inputs.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.action_space.n + 1)
        assert env.spec.id.startswith("gym_cards"), "This wrapper is only for gym_cards environments."

    def action(self, action):
        # Convert to python int to keep downstream gym_cards logic happy.
        return int(action) - 1
    

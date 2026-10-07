from src.environments.gym_cards.envs.ezpoints import EZPointEnv
from src.environments.gym_cards.envs.cardmaze import CardMazeEnv

from gymnasium.envs.registration import register

register(
    id='gym_cards/EZPoints-v0',
    entry_point='src.environments.gym_cards.envs.ezpoints:EZPointEnv',
    max_episode_steps=5,
)

register(
    id='gym_cards/CardMaze-v0',
    entry_point='src.environments.gym_cards.envs.cardmaze:CardMazeEnv',
    max_episode_steps=100, # actually depends on the n_decisions param, so this is just a safety limit
)

from .common import make_env

from gymnasium.envs.registration import register

register(
    id='AlfworldGymEnv-v0',
    entry_point='src.environments.alfworld_env:AlfworldGymEnv',
)

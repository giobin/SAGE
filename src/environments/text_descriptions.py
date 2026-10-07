import torch
import random

from omegaconf import DictConfig
from typing import List


MINIGRID_DIRECTIONS = {
    0: "right",
    1: "down",
    2: "left",
    3: "up"
}

def _get_frozen_lake_grid_obs(obs: torch.Tensor, config: DictConfig) -> str:
    obs_str = ""
    agent_pos = None
    hole_positions = []
    goal_position = None
    for i, row in enumerate(obs):
        for j, cell in enumerate(row):
            if cell == 4:  # Agent's current position
                agent_pos = (i, j)
            elif cell == 2:  # Hole
                hole_positions.append((i, j))
            elif cell == 3:  # Goal
                goal_position = (i, j)
    
    obs_str += "Grid size: {}x{}\n".format(len(obs), len(obs[0]))
    obs_str += f"Current position of the agent: {agent_pos}\n"
    obs_str += f"Hole positions: {hole_positions}\n"
    obs_str += f"Goal position: {goal_position}\n"
    return obs_str

def get_text_obs(obs: torch.Tensor, config: DictConfig) -> List[str]:
    """
    An observation tensor into a text description for the guidance model.
    Automatically detects which environment is being used from the config and formats the observation accordingly.
    
    Args:
        obs (torch.Tensor): The observation tensor (assuming shape [batch_size, ...]).
        config (DictConfig): The configuration object containing environment details.
        
    Returns:
        List[str]: A list of text descriptions for each observation in the batch.
    """
    if config.env.env_id == "FrozenLake-v1":
        if config.env.get("env_args", {}).get("use_grid_obs", False):
            # Grid-based frozen lake observation
            return [ _get_frozen_lake_grid_obs(obs[i], config) for i in range(obs.shape[0]) ]
    else:
        raise NotImplementedError(f"Text description for environment {config.env.env_id} is not implemented.")
    
######################
# Gym Cards Wrappers #
######################

def info_to_text_obs(env_name, info):
    """
    This function directly parse the info from the gym_cards envs
    to customized text observation.

    gym_cards envs: https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards

    please adjust text_obs accordingly as needed
    """
    if env_name == "gym_cards/EZPoints-v0":
        """
        J, Q, and K count as 10, you can choose to increase the difficulty
        only telling the language model J, Q, K, and describe in the prompt that they count as 10
        """
        text_obs = f"Cards: {info['Cards']}. Numbers: {info['Numbers']}."
        current_formula = ''.join(str(element) for element in info['Formula'])
        text_obs = text_obs + f" Current formula: {current_formula}."
    else:
        raise NotImplementedError("Environment not implemented.")
    return text_obs

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    if env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(action_list) - 1))
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.randint(0, len(action_list) - 1))
    ## Please adjust the output dtypes accordingly
    return output_indices

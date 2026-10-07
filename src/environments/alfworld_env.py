import gymnasium as gym
import numpy as np
import yaml
import os
from gymnasium import spaces
from alfworld.agents.environment import get_environment
from omegaconf import DictConfig

class AlfworldGymEnv(gym.Env):
    def __init__(
                self, 
                alfworld_config: DictConfig | dict,
                max_allowed_actions: int = 50, 
                headless: bool = True, 
                reward_type: str = "dense", 
                train_eval: str = 'train'
            ) -> None:
        """
        Args:
            alfworld_config (DictConfig | dict): Configuration for the Alfworld environment.
            max_allowed_actions (int): Maximum number of admissible actions to consider.
            headless (bool): Whether to run in headless mode (no GUI).
            reward_type (str): Type of reward to use ("sparse" or "dense").
            train_eval (str): Whether the environment is for training or evaluation ('train', 'eval_in_distribution', 'eval_out_of_distribution').
        """
        # 1. Headless Setup
        self.headless = headless

        if self.headless:
            print(f"[alfworld_env] DISPLAY in os.environ: {'DISPLAY' in os.environ}, value: {os.environ.get('DISPLAY', 'NOT SET')}")
            if 'DISPLAY' not in os.environ:
                from pyvirtualdisplay import Display
                display = Display(visible=False, size=(400, 400))
                display.start()
                print(f"Started global virtual display: {os.environ['DISPLAY']}")

        # 2. Optional VirtualGL support for AI2-THOR's Unity subprocess.
        # If a launcher saved vglrun's LD_PRELOAD to VGL_PRELOAD (and unset it so vLLM
        # can initialize CUDA), restore it while Unity starts so it gets GPU-accelerated
        # OpenGL, then unset it again.
        vgl = os.environ.get("VGL_PRELOAD", "")
        if vgl:
            os.environ["LD_PRELOAD"] = vgl
            print(f"[alfworld_env] Restored LD_PRELOAD from VGL_PRELOAD for AI2-THOR")

        # 3. Config & Env Setup
        self.config = alfworld_config

        self.config['env']['type'] = 'AlfredThorEnv'
        env_type = self.config['env']['type']
        assert train_eval in ['train', 'eval_in_distribution', 'eval_out_of_distribution'], "train_eval must be one of 'train', 'eval_in_distribution', or 'eval_out_of_distribution'"
        self.env = get_environment(env_type)(self.config, train_eval=train_eval)
        self.env = self.env.init_env(batch_size=1)

        # Unity subprocess is now running and inherited LD_PRELOAD.
        # Unset it so vLLM CUDA init won't be broken by VGL faker libs.
        if vgl:
            os.environ.pop("LD_PRELOAD", None)
            print(f"[alfworld_env] Unset LD_PRELOAD (Unity already running, vLLM needs clean CUDA)")

        # 3. Space Definition
        self.max_actions = max_allowed_actions
        self.action_space = spaces.Discrete(self.max_actions)
        h = alfworld_config["env"]["thor"]["screen_height"]
        w = alfworld_config["env"]["thor"]["screen_width"]
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8),
            "mission": spaces.Text(max_length=1000)
        })

        # 4. Reward Logic State
        self.reward_type = reward_type
        self.last_goal_condition_rate = 0.0
        
        # State variables
        self.admissible_commands = []
        self.goal_text = ""
        
        self.next_oracle_action_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_list, infos = self.env.reset()
        
        # Reset reward tracker
        self.last_goal_condition_rate = 0.0
        
        # We need to extract the goal for the first time
        text_obs = obs_list[0]
        self.goal_text = text_obs.split('\n')[-1]

        observation, info = self._process_obs_and_info(obs_list, infos, is_reset=True)
        return observation, info

    def step(self, action_idx):
        # 1. Action Mapping (No sorting!)
        if action_idx < len(self.admissible_commands):
            cmd = self.admissible_commands[action_idx]
        else:
            cmd = "look"

        # 2. Step Environment
        obs_list, scores, dones, infos = self.env.step([cmd])
        
        done = dones[0]
        won = infos['won'][0]
        current_goal_rate = infos['goal_condition_success_rate'][0]

        # 3. Reward Calculation
        reward = 1.0 if done and won else 0.0
        if self.reward_type == "dense" and not done:
            reward = current_goal_rate - self.last_goal_condition_rate
                
        self.last_goal_condition_rate = current_goal_rate

        # 4. Process Observations & Info
        observation, info = self._process_obs_and_info(obs_list, infos, is_reset=False)
        
        # Add step-specific info
        info["won"] = won
        info["goal_condition_rate"] = current_goal_rate

        return observation, reward, done, False, info
    
    def render(self):
        return self._get_image()

    def _process_obs_and_info(self, obs_list, infos, is_reset=False):
        """
        Consolidated logic for processing raw environment outputs.
        """
        # A. Clean Text Observation
        text_obs = obs_list[0]
        if "-= Welcome to TextWorld, ALFRED! =-" in text_obs:
            text_obs = text_obs.split("-= Welcome to TextWorld, ALFRED! =-")[-1].strip()
        
        # B. Get Admissible Commands (unsorted to preserve oracle index alignment)
        self.admissible_commands = list(infos['admissible_commands'][0])
        
        # C. Extract Oracle Action
        # 'extra.expert_plan' is a list of lists: [['put apple in fridge']]
        expert_plan = infos.get('extra.expert_plan', [[]])[0]
        oracle_action_str = expert_plan[0] if expert_plan else "look"
        
        # Map string back to the index for the agent
        try:
            oracle_action_idx = self.admissible_commands.index(oracle_action_str)
        except ValueError:
            oracle_action_idx = 0 
            
        self.next_oracle_action_idx = oracle_action_idx

        # D. Construct Observation Dict
        observation = {
            "image": self._get_image(),
            "mission": f"**Goal**: {self.goal_text}.\n**Text Observation**: {text_obs}"
        }

        # E. Construct Info Dict
        info = {
            "admissible_commands": self.admissible_commands,
            "oracle_action": oracle_action_str,
            "oracle_action_idx": oracle_action_idx,
            "text_observation": text_obs
        }
        
        return observation, info
    
    def _get_image(self):
        # BGR -> RGB Fix
        im = self.env.get_frames()[0]
        return im[:, :, ::-1]
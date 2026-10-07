import torch
import numpy as np

import src.guidance_models.oracles as oracles

from PIL import Image
from typing import Dict, Optional, List
from omegaconf import DictConfig
from src.guidance_models.base_guidance_model import BaseGuidanceModel

IMPLEMENTED_ORACLES = [
    "FrozenLake",
    "gym_cards/EZPoint",
    "gym_cards/CardMaze",
    "MiniGrid-Fetch",
    "MiniGrid-LavaGap",
    "MiniGrid-GoToDoor",
    "AlfworldGymEnv",
]

class OracleGuidanceModel(BaseGuidanceModel):
    """
    Oracle guidance model based on heuristic rules. Used to test the upper-bound performance of the method.
    """
    
    def __init__(self, model_id: str, env_id: str, guidance_model_config: DictConfig, **kwargs):
        # super initializes cache (if needed, although for oracle this likely makes no difference)
        super().__init__(guidance_model_config=guidance_model_config, verbose=kwargs.get("verbose", False))
        # model_id is not used but kept for the shared interface
        self.model_id = model_id
        self.env_id = env_id # Needed to determine which oracle to use
        self.seed = kwargs.get("seed", None)
        self.rng = np.random.default_rng(self.seed)
        if not any(env_id.startswith(prefix) for prefix in IMPLEMENTED_ORACLES):
            raise NotImplementedError(f"Oracle guidance model not implemented for environment {env_id}. "
                             f"Available oracles: {IMPLEMENTED_ORACLES}")
        
        if "FrozenLake" in self.env_id:
            self.nrows, self.ncols = kwargs.get("grid_size", (None, None))
            if self.nrows is None or self.ncols is None:
                import warnings
                warnings.warn("Grid size not provided for FrozenLake oracle; defaulting to 8x8.")
                self.nrows, self.ncols = 8, 8

    def _generate_no_cache(self, prompts, images=None, **kwargs):
        # Not defined for oracle guidance model, as it does not use a VLM.
        pass

    def _sample_random_action(self, env, fallback_action_space=None):
        action_space = env.action_space if env is not None else fallback_action_space
        if action_space is None:
            raise ValueError("Cannot sample random oracle action: action space is not available.")
        return action_space.sample()

    def _maybe_randomize_actions(self, actions: List[str], *, random_guidance_prob: float, env_list=None, envs=None, action_space=None):
        if random_guidance_prob <= 0.0:
            return actions, 0
        if random_guidance_prob > 1.0:
            raise ValueError(f"random_guidance_prob must be in [0, 1], got {random_guidance_prob}.")

        randomized_actions = list(actions)
        random_action_count = 0
        for i in range(len(randomized_actions)):
            if self.rng.random() < random_guidance_prob:
                env = env_list[i] if env_list is not None else None
                sampled_action = self._sample_random_action(env, fallback_action_space=action_space)
                randomized_actions[i] = str(sampled_action)
                random_action_count += 1
        return randomized_actions, random_action_count
    
    def provide_guidance(self, prompts: torch.Tensor, images: Optional[List[Image.Image]] = None, **kwargs) -> Dict[str, List[str | Image.Image]]:
        """
        Provide guidance based on heuristic rules. Overrides the base method which works by prompting a VLM.
        
        Args:
            config (DictConfig): Configuration for the guidance model (not used in this oracle).
            states (torch.Tensor): The current state of the environment, typically an image or observation.
            **kwargs: Additional arguments, such as 'envs' for FrozenLake.
        """
        verbose = kwargs.pop("verbose", False)
        batch_size = len(prompts)
        random_guidance_prob = float(kwargs.get("random_guidance_prob", 0.0))
        
        envs = kwargs.get("envs")
        env_indices = kwargs.get("env_indices")
        if env_indices is None:
            env_indices = list(range(batch_size))
        if envs is not None and hasattr(envs, "envs"):
            env_list = envs.envs
        else:
            env_list = None
        action_space = kwargs.get("action_space", getattr(envs, "single_action_space", None))

        def _finalize(actions: List[str]) -> Dict[str, List[str | Image.Image]]:
            actions, random_action_count = self._maybe_randomize_actions(
                actions,
                random_guidance_prob=random_guidance_prob,
                env_list=env_list,
                envs=envs,
                action_space=action_space,
            )
            if verbose and random_guidance_prob > 0.0:
                print(f"Oracle guidance randomization enabled: p={random_guidance_prob}")
            return {"responses": actions, "prompts": prompts, "images": images, "random_action_count": random_action_count}

        if self.env_id.startswith("FrozenLake"):
            actions = []
            for i in range(batch_size):
                analysis = oracles.analyze_frozenlake_grid(images[i][0], nrows=self.nrows, ncols=self.ncols)
                action = analysis["next_move"]
                actions.append(f"{action}" if action is not None else "0") # Default to '0' if no action is determined
            return _finalize(actions) # Could return tensor_to_pil for images, but this is likely not necessary
        elif self.env_id.startswith("gym_cards/EZPoint"):
            actions = []
            if env_list is not None:
                assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
                for i in range(batch_size):
                    action = oracles.get_next_action_for_ezpoints(env_list[i].unwrapped.cards_num, env_list[i].unwrapped.formula)
                    actions.append(f"{action}")
            else:
                assert envs is not None, "EZPoints oracle requires the 'envs' argument to be passed."
                try:
                    cards_nums = envs.get_attr("cards_num", indices=env_indices)
                    formulas = envs.get_attr("formula", indices=env_indices)
                except TypeError:
                    cards_nums = [envs.get_attr("cards_num")[i] for i in env_indices]
                    formulas = [envs.get_attr("formula")[i] for i in env_indices]
                for cards_num, formula in zip(cards_nums, formulas):
                    actions.append(f"{oracles.get_next_action_for_ezpoints(cards_num, formula)}")
            return _finalize(actions)
        elif self.env_id.startswith("gym_cards/CardMaze"):
            actions = []
            if env_list is not None:
                assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
                for i in range(batch_size):
                    action = env_list[i].unwrapped._current_correct_action()
                    actions.append(f"{action}")
            else:
                assert envs is not None, "CardMaze oracle requires the 'envs' argument to be passed."
                try:
                    actions = [str(action) for action in envs.call("_current_correct_action", indices=env_indices)]
                except TypeError:
                    actions = [str(action) for action in envs.call("_current_correct_action")]
                    actions = [actions[i] for i in env_indices]
            return _finalize(actions)
        elif self.env_id.startswith("MiniGrid-Fetch"):
            actions = []
            if env_list is None:
                raise NotImplementedError("MiniGrid-Fetch oracle requires SyncVectorEnv (env objects not available in AsyncVectorEnv).")
            assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
            for i in range(batch_size):
                action = oracles.get_fetch_oracle_action(env_list[i])
                actions.append(f"{action}")
            return _finalize(actions)
        elif self.env_id.startswith("MiniGrid-LavaGap"):
            actions = []
            if env_list is None:
                raise NotImplementedError("MiniGrid-LavaGap oracle requires SyncVectorEnv (env objects not available in AsyncVectorEnv).")
            assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
            for i in range(batch_size):
                action = oracles.get_lavagap_oracle_action(env_list[i])
                actions.append(f"{action}")
            return _finalize(actions)
        elif self.env_id.startswith("MiniGrid-GoToDoor"):
            actions = []
            if env_list is None:
                raise NotImplementedError("MiniGrid-GoToDoor oracle requires SyncVectorEnv (env objects not available in AsyncVectorEnv).")
            assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
            for i in range(batch_size):
                action = oracles.get_gotodoor_oracle_action(env_list[i])
                actions.append(f"{action}")
            return _finalize(actions)
        elif self.env_id.startswith("AlfworldGymEnv"):
            actions = []
            if env_list is not None:
                assert len(env_list) == batch_size, f"Number of environments ({len(env_list)}) must match the batch size ({batch_size})."
                for i in range(batch_size):
                    alf_env = env_list[i].unwrapped
                    oracle_action_idx = alf_env.next_oracle_action_idx
                    actions.append(f"{oracle_action_idx}")
            else:
                assert envs is not None, "Alfworld oracle requires the 'envs' argument to be passed."
                try:
                    oracle_action_indices = envs.get_attr("next_oracle_action_idx", indices=env_indices)
                except TypeError:
                    oracle_action_indices = [envs.get_attr("next_oracle_action_idx")[i] for i in env_indices]
                actions = [str(action) for action in oracle_action_indices]
            return _finalize(actions)
        else:
            raise NotImplementedError(f"Oracle guidance model not implemented for environment {self.env_id}. "
                             f"Available oracles: {IMPLEMENTED_ORACLES}")

    def provide_reward_shaping_preferences(
        self,
        difference_prompts,
        preference_prompt_templates,
        *,
        difference_images=None,
        difference_kwargs=None,
        preference_kwargs=None,
    ):
        if "FrozenLake" in self.env_id:
            assert difference_images is not None, "Difference images must be provided for FrozenLake reward shaping preferences."
            analysis_responses = []
            preference_responses = []
            for img1, img2 in difference_images:
                analysis1 = oracles.analyze_frozenlake_grid(img1, nrows=self.nrows, ncols=self.ncols)
                analysis2 = oracles.analyze_frozenlake_grid(img2, nrows=self.nrows, ncols=self.ncols)
                path_length1 = analysis1["path_length"]
                path_length2 = analysis2["path_length"]
                if path_length1 < path_length2:
                    preference = "0"
                elif path_length2 < path_length1:
                    preference = "1"
                else:
                    preference = "-1"
                analysis_responses.append(f"Path length in Image 1: {path_length1}, Path length in Image 2: {path_length2}.")
                preference_responses.append(preference)
            
            return {"analysis_responses": analysis_responses, "preference_responses": preference_responses, "filled_preference_prompts": preference_prompt_templates}
            
        raise NotImplementedError(f"Reward shaping preferences are not available for the oracle guidance model for environment {self.env_id}.")

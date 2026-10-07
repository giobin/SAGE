import torch
from omegaconf import DictConfig
from typing import List, Optional, Any, Dict
from PIL import Image

from src.guidance_models import BaseGuidanceModel


class RandomGuidanceModel(BaseGuidanceModel):
    """
    Random guidance model that provides random actions from the environment's action space, or random preferences for reward shaping.
    Used as a baseline and sanity check.
    """
    def __init__(self, guidance_model_config: DictConfig, verbose: bool = False, **kwargs):
        super().__init__(guidance_model_config, verbose, **kwargs)
        
    def _generate_no_cache(self, prompts, images = None, **kwargs):
        raise NotImplementedError("RandomGuidanceModel does not support _generate_no_cache.")

    def provide_guidance(self, prompts, images = None, **kwargs):
        assert "envs" in kwargs, "envs must be provided to RandomGuidanceModel"
        batch_size = len(prompts)
        # Sample a random action from the action space of each environment for each prompt
        random_actions = []
        envs = kwargs["envs"]
        if hasattr(envs, "envs"):
            for i in range(batch_size):
                env = envs.envs[i]
                random_action = env.action_space.sample()
                random_actions.append(str(random_action))
        else:
            action_space = envs.single_action_space
            for _ in range(batch_size):
                random_actions.append(str(action_space.sample()))
        return {"responses": random_actions, "prompts": prompts, "images": images}
    
    def provide_reward_shaping_preferences(
        self,
        difference_prompts: List[str],
        preference_prompt_templates: List[str],
        *,
        difference_images: Optional[List[List[Image.Image]]] = None,
        difference_kwargs: Optional[Dict[str, Any]] = None,
        preference_kwargs: Optional[Dict[str, Any]] = None,
        max_batch_size: int = 8,
    ) -> List[int]:
        batch_size = len(difference_prompts)
        # Randomly choose preferences (-1 = no preference, 0 = first better, 1 = second better)
        random_preferences = torch.randint(-1, 2, (batch_size,)).tolist()
        random_preferences = [str(pref) for pref in random_preferences]
        analysis_responses = ["Random preference"] * batch_size
        return {
            "analysis": analysis_responses,
            "preference_responses": random_preferences,
            "filled_preference_prompts": preference_prompt_templates
        }

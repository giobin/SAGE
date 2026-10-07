import os
import numpy as np
import gymnasium as gym

from typing import Callable, Optional, Any
from torch import from_numpy
from torchvision.transforms.functional import resize

try:
    import minigrid  # noqa: F401 - imported to register MiniGrid environments
except ImportError:
    minigrid = None

try:
    import src.environments.gym_cards  # noqa: F401 - imported to register gym_cards environments
except ImportError:
    pass


def resize_obs(obs: np.ndarray, h: int = 244, w: int = 244) -> np.ndarray:
    """
    Resize the observation image to fit the model input requirements (from [H, W, C] to [C, H, W]).
    
    Args:
        obs (np.ndarray): The observation image in the format [H, W, C].
        h (int, optional): The target height for resizing. Default is 244.
        w (int, optional): The target width for resizing. Default is 244.
    Returns:
        np.ndarray: The resized observation image in the format [C, H, W].
    """
    assert obs.ndim == 3, "Observation must be a 3D array (H, W, C)"
    assert obs.shape[2] == 3, "Observation must have 3 channels (RGB)"
    
    obs = from_numpy(obs).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    obs = resize(obs, (h, w), antialias=False)
    return obs.numpy()

# Action descriptions for specific environments.
FROZENLAKE_ACTIONS = {
    0: "move left",
    1: "move down",
    2: "move right",
    3: "move up",
}

# For EZPoints the index is shifted compared to the original env,
# so we have index 1 as first move, to add 1. index 2 to add 2, etc.
EZPOINTS_ACTIONS = {
    1: 'add number "1" to the formula',
    2: 'add number "2" to the formula',
    3: 'add number "3" to the formula',
    4: 'add number "4" to the formula',
    5: 'add number "5" to the formula',
    6: 'add number "6" to the formula',
    7: 'add number "7" to the formula',
    8: 'add number "8" to the formula',
    9: 'add number "9" to the formula',
    10: 'add number "10" to the formula',
    11: 'add the operator "+" (addition) to the formula',
    12: 'add the operator "*" (multiplication) to the formula',
    13: 'add the operator "=" (equals) to the formula',
}

MINIGRID_ACTIONS = {
    0: "turn left (in place)",
    1: "turn right (in place)",
    2: "move forward one cell in the direction the agent is facing",
    3: "pick up the object in the cell in the direction the agent is facing",
    4: "drop the object in the cell in the direction the agent is facing",
    5: "toggle the object in the cell in the direction the agent is facing",
    6: "done (end the episode)",
}

def get_action_description(env_id: str, action: int) -> Optional[str]:
    """Return a human-readable action description for known envs, else None."""
    if _is_frozenlake(env_id):
        return FROZENLAKE_ACTIONS.get(action)
    if env_id == "gym_cards/EZPoints-v0":
        return EZPOINTS_ACTIONS.get(action)
    if _is_minigrid(env_id):
        return MINIGRID_ACTIONS.get(action)
    return None

def _is_minigrid(eid: str) -> bool:
    return eid.startswith("MiniGrid") or eid.startswith("BabyAI")

def _is_frozenlake(eid: str) -> bool:
    return eid.startswith("FrozenLake")

def _is_gym_cards(eid: str) -> bool:
    return eid.startswith("gym_cards")

def _is_alfworld(eid: str) -> bool:
    return eid.startswith("Alfworld")

def _gym_make(env_id: str, render_mode: Optional[str], **kwargs: Any) -> gym.Env:
    """Call gym.make, passing render_mode only if not None."""
    return gym.make(env_id, **({**kwargs, "render_mode": render_mode} if render_mode is not None else kwargs))


from typing import Any, Callable, Optional
import numpy as np
import gymnasium as gym

def make_env(
    env_id: str,
    idx: int,
    *,
    capture_video: bool = False,
    video_frequency: int = 500, # in terms of episodes
    use_visual_obs: bool = False,
    stack_size: Optional[int] = None,
    seed: Optional[int] = None,
    gamma: float = 0.99,
    **kwargs: Any,
) -> Callable[[], gym.Env]:
    """
    Compact thunk factory for Gymnasium envs with minimal duplication.

    Families:
      - MiniGrid/BabyAI: RGBImgObsWrapper(+ImgObsWrapper if visual-only), optional ReseedWrapper
      - FrozenLake: optional grid obs or visual obs
      - gym_cards: optional text obs
      - generic: optional VisualObsWrapper

    Video is recorded only for idx==0.
    """

    # Extract custom knobs that shouldn't go to gym.make
    use_grid_obs = bool(kwargs.pop("use_grid_obs", False))
    tile_size    = int(kwargs.pop("tile_size", 50))
    map_size     = int(kwargs.pop("random_map_dimension", 0))

    # Decide render_mode once, per family:
    if _is_gym_cards(env_id):
        render_mode = None
        if capture_video:
            raise ValueError("gym_cards environments do not support video recording.")
    elif _is_alfworld(env_id): # alfworld has no render_mode argument
        render_mode = None
        if not use_visual_obs:
            raise ValueError("Alfworld environments require visual observations.")
    else:
        render_mode = "rgb_array"

    def thunk() -> gym.Env:
        if _is_frozenlake(env_id) and map_size > 0 and not "map_name" in kwargs:
            from gymnasium.envs.toy_text.frozen_lake import generate_random_map
            kwargs["desc"] = generate_random_map(map_size, seed=seed + idx if seed is not None else None)
            
        if not _is_alfworld(env_id):
            os.environ.pop("DISPLAY", None)

        # --- create base env with the chosen render_mode ---
        env = _gym_make(env_id, render_mode, **kwargs)

        # --- seeding (different for each env) ---
        if _is_minigrid(env_id):
            if seed is not None:
                env.reset(seed=seed + idx)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)

        # --- family-specific wrappers ---
        if _is_minigrid(env_id):
            from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
            env = RGBImgObsWrapper(env, tile_size=tile_size)
            if use_visual_obs:
                env = ImgObsWrapper(env)

        elif _is_frozenlake(env_id):
            if map_size > 0:
                from src.environments.wrappers import FrozenLakeRandomResetWrapper
                env = FrozenLakeRandomResetWrapper(env, map_size=map_size, seed=seed + idx if seed is not None else None, **kwargs)
            if not use_visual_obs and use_grid_obs:
                from src.environments.wrappers import FrozenLakeGridObsWrapper
                env = FrozenLakeGridObsWrapper(env)
            elif use_visual_obs:
                from src.environments.wrappers import VisualObsWrapper
                env = VisualObsWrapper(env)

        elif _is_gym_cards(env_id):
            from src.environments.wrappers import GymCardsTextObsWrapper, GymCardsActionShiftWrapper
            if not use_visual_obs:
                env = GymCardsTextObsWrapper(env)
            if env_id == "gym_cards/EZPoints-v0":
                env = GymCardsActionShiftWrapper(env)
            # visual obs: gym_cards is already image-based
        
        elif _is_alfworld(env_id):
            pass  # alfworld returns images by default
        else:
            if use_visual_obs:
                from src.environments.wrappers import VisualObsWrapper
                env = VisualObsWrapper(env)

        # --- shared post-wrappers ---
        if use_visual_obs and stack_size and stack_size > 1:
            from gymnasium.wrappers import FrameStackObservation
            env = FrameStackObservation(env, stack_size=stack_size)

        if capture_video and idx == 0:
            from hydra.core.hydra_config import HydraConfig
            output_dir = HydraConfig.get().runtime.output_dir
            video_dir = os.path.join(output_dir, "videos")
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda ep: ep % video_frequency == 0)

        # Ensure episode stats are available
        if not any(isinstance(env, gym.wrappers.RecordEpisodeStatistics) for env in _iter_wrappers(env)):
            env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def _iter_wrappers(env: gym.Env):
    """Yield env and its wrapper chain (outer to inner)."""
    while True:
        yield env
        if not hasattr(env, "env"):
            break
        env = env.env

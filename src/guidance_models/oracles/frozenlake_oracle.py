import networkx as nx
import numpy as np

from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image

class TileType(Enum):
    AGENT = (99, 171, 63)
    EMPTY = (255, 255, 255)
    HOLE = (61, 202, 242)
    HOLE_CRACKED = (0, 52, 94)
    GOAL = (171, 81, 48)

class MoveAction(Enum):
    LEFT = ((0, -1), 0)
    DOWN = ((1, 0), 1)
    RIGHT = ((0, 1), 2)
    UP = ((-1, 0), 3)

    def __init__(self, delta: Tuple[int, int], action_id: int) -> None:
        self._delta = delta
        self._action_id = action_id

    @property
    def delta(self) -> Tuple[int, int]:
        return self._delta

    @property
    def action_id(self) -> int:
        return self._action_id

DELTA_TO_MOVE = {action.delta: action for action in MoveAction}

def analyze_frozenlake_grid(
    obs: np.ndarray | Image.Image,
    *,
    nrows: int = 8,
    ncols: int = 8,
 ) -> dict[str, Any]:
    """
    Parse a FrozenLake visual observation into a grid representation, identify agent and goal positions,
    and compute the shortest path from the agent to the goal avoiding holes and next optimal move.
    
    Args:
        obs (np.ndarray): The visual observation of the FrozenLake environment (H, W, C).
        nrows (int): Number of rows in the FrozenLake grid.
        ncols (int): Number of columns in the FrozenLake grid.
    
    Returns:
        dict: A dictionary containing:
            - 'graph': The grid graph with holes removed.
            - 'tile_types': A 2D array of TileType enums representing each tile.
            - 'tile_centers': List of (y, x) pixel coordinates for the center of each tile.
            - 'agent_pos': (row, col) of the agent or None if not found.
            - 'goal_pos': (row, col) of the goal or None if not found.
            - 'hole_cracked': Boolean indicating if a cracked hole was found.
            - 'tile_height': Height of each tile in pixels.
            - 'tile_width': Width of each tile in pixels.
            - 'path': List of (row, col) tuples representing the shortest path from agent to goal or None if no path exists.
            - 'path_length': Length of the shortest path or infinity if no path exists.
            - 'next_move': The action ID for the next optimal move towards the goal or None if not applicable.
    """
    if type(obs) is Image.Image:
        obs = np.array(obs)
    if obs.ndim != 3:
        raise ValueError(
            f"Expected observation with 3 dimensions (H, W, C), got shape {obs.shape}."
        )
    height, width, _ = obs.shape
    if height % nrows != 0 or width % ncols != 0:
        raise ValueError(
            f"Observation shape {(height, width)} is not divisible by ({nrows}, {ncols}) tiles."
        )

    tile_height = height // nrows
    tile_width = width // ncols

    tile_types = np.empty((nrows, ncols), dtype=object)
    tile_centers: List[Tuple[int, int]] = []
    agent_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    hole_cracked_found = False

    for row in range(nrows):
        for col in range(ncols):
            center_y = row * tile_height + tile_height // 2
            center_x = col * tile_width + tile_width // 2
            tile_centers.append((center_y, center_x))

            sample_y = max(min(center_y - 12, height - 1), 0)
            pixel = obs[sample_y, center_x]
            if np.issubdtype(pixel.dtype, np.floating) and float(np.max(pixel)) <= 1.0 + 1e-6:
                pixel_tuple = tuple(int(round(channel * 255)) for channel in pixel)
            else:
                pixel_tuple = tuple(int(round(channel)) for channel in pixel)

            try:
                tile_type = TileType(pixel_tuple)
            except ValueError as exc:
                from src.environments.frozenlake_viz import plot_sample_points
                debug_path = "frozenlake_oracle_debug_obs.png"
                img_with_points = plot_sample_points(obs, nrows=nrows, ncols=ncols)
                img_with_points.save(debug_path)
                raise ValueError(
                    f"Unrecognized tile color {pixel_tuple} at grid coordinate ({row}, {col})."
                ) from exc

            tile_types[row, col] = tile_type

            if tile_type is TileType.AGENT:
                if agent_pos is not None:
                    raise ValueError(
                        f"Multiple agent tiles detected at {agent_pos} and ({row}, {col})."
                    )
                agent_pos = (row, col)

            if tile_type is TileType.GOAL:
                if goal_pos is not None:
                    raise ValueError(
                        f"Multiple goal tiles detected at {goal_pos} and ({row}, {col})."
                    )
                goal_pos = (row, col)

            if tile_type is TileType.HOLE_CRACKED:
                hole_cracked_found = True
            elif tile_type is TileType.HOLE or tile_type is TileType.EMPTY:
                cracked_pixel = obs[center_y, center_x]
                if np.issubdtype(cracked_pixel.dtype, np.floating) and float(np.max(cracked_pixel)) <= 1.0 + 1e-6:
                    cracked_pixel_tuple = tuple(int(round(channel * 255)) for channel in cracked_pixel)
                else:
                    cracked_pixel_tuple = tuple(int(round(channel)) for channel in cracked_pixel)
                if cracked_pixel_tuple == TileType.HOLE_CRACKED.value:
                    hole_cracked_found = True

    graph = nx.grid_2d_graph(nrows, ncols)
    hole_nodes = [
        (row, col)
        for row in range(nrows)
        for col in range(ncols)
        if tile_types[row, col] in {TileType.HOLE, TileType.HOLE_CRACKED}
    ]
    graph.remove_nodes_from(hole_nodes)

    if agent_pos is None:
        if hole_cracked_found:
            return {
                "graph": graph,
                "tile_types": tile_types,
                "tile_centers": tile_centers,
                "agent_pos": None,
                "goal_pos": goal_pos,
                "hole_cracked": True,
                "tile_height": tile_height,
                "tile_width": tile_width,
                "path": None,
                "path_length": np.inf,
                "next_move": None,
            }
        
        raise ValueError("No agent tile detected and no cracked hole present.")

    if goal_pos is None:
        return {
            "graph": graph,
            "tile_types": tile_types,
            "tile_centers": tile_centers,
            "agent_pos": agent_pos,
            "goal_pos": None,
            "hole_cracked": hole_cracked_found,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "path": [agent_pos],
            "path_length": 0,
            "next_move": None,
        }

    try:
        path = nx.shortest_path(graph, source=agent_pos, target=goal_pos)
        path_length = len(path) - 1
    except nx.NetworkXNoPath:
        path = None
        path_length = np.inf

    if path is None or len(path) < 2:
        next_move = None
    else:
        (curr_row, curr_col), (next_row, next_col) = path[0], path[1]
        delta = (next_row - curr_row, next_col - curr_col)
        move_action = DELTA_TO_MOVE.get(delta)
        next_move = move_action.action_id if move_action else None

    return {
        "graph": graph,
        "tile_types": tile_types,
        "tile_centers": tile_centers,
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "hole_cracked": hole_cracked_found,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "path": path,
        "path_length": path_length,
        "next_move": next_move,
    }
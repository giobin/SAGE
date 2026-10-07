import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any

def plot_sample_points(obs: np.ndarray | Image.Image, nrows: int = 8, ncols: int = 8) -> Image.Image:
    """
    Plot the sample points on the FrozenLake grid image, used for the oracle guidance model.
    """
    if isinstance(obs, Image.Image):
        obs = np.array(obs)
    
    img_height, img_width, _ = obs.shape
    tile_height = img_height // nrows
    tile_width = img_width // ncols

    tile_centers = []
    for row in range(nrows):
        for col in range(ncols):
            center_x = col * tile_width + tile_width // 2
            center_y = row * tile_height + tile_height // 2
            tile_centers.append((center_y - 12, center_x)) # -12 as the sample point is there to avoid conflicts

    obs_copy = obs.copy()
    pil_img = Image.fromarray((obs_copy * 255).astype(np.uint8))
    for center in tile_centers:
        y, x = center
        coord_str = f"({y // tile_height}, {x // tile_width})"
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        draw.text((x - 10, y - 20), coord_str, fill=(0, 0, 0), font=font)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
    
    return pil_img

def get_detailed_map_img(obs: np.ndarray | Image.Image, analysis: Dict[str, Any]) -> Image.Image:
    """
    Shows computed graph and shortest path from oracle analysis
    """
    
    G = analysis["graph"]
    shortest_path = analysis["path"]
    tile_height = analysis["tile_height"]
    tile_width = analysis["tile_width"]
    
    obs_copy = obs.copy()
    pil_img = Image.fromarray((obs_copy * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    for edge in G.edges():
        (row1, col1), (row2, col2) = edge
        x1 = col1 * tile_width + tile_width // 2
        y1 = row1 * tile_height + tile_height // 2
        x2 = col2 * tile_width + tile_width // 2
        y2 = row2 * tile_height + tile_height // 2
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=2)

    if shortest_path is not None and len(shortest_path) > 1:
        for (row1, col1), (row2, col2) in zip(shortest_path[:-1], shortest_path[1:]):
            x1 = col1 * tile_width + tile_width // 2
            y1 = row1 * tile_height + tile_height // 2
            x2 = col2 * tile_width + tile_width // 2
            y2 = row2 * tile_height + tile_height // 2
            draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=4)
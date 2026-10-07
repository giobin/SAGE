import os
import math
import random
import inflect
from typing import Optional, List, Tuple, Dict
from omegaconf import DictConfig

import numpy as np
import gymnasium as gym

from gymnasium import spaces
from PIL import Image, ImageFont, ImageDraw

FONT_PATH = os.path.join(os.path.dirname(__file__), "font", "DejaVuSans.ttf")

# constants
MAX_RAND_SEED = 1_000_000
SUITS = ["H", "S", "D", "C"]
ALLOWED_RULES = ["number", "suit", "both", "neither", "all"]
ALLOWED_PROGRESS_STYLES = ["text", "symbol", "none"]

def get_image(card_name: str) -> Image.Image:
    """Load a PNG from img/{card_name}.png relative to this source file."""
    path = f"img/{card_name}.png"
    cwd = os.path.dirname(__file__)
    image = Image.open(os.path.join(cwd, path)).convert("RGBA")
    return image

def format_prompt(prompt: str, env_config: DictConfig) -> str:
    """
    Formats the VLM prompt template to fill in environment-specific details, such as the number of cards and the rule.
    
    Args:
        prompt (str): The prompt template containing placeholders ({n_cards}, {n_decisions}, {rule}, {action_space}).
        env_config (DictConfig): The environment configuration containing parameters like n_cards, rule, etc.
    """
    n_cards = env_config.get("n_cards")
    rule = env_config.get("rule")
    n_decisions = env_config.get("n_decisions")

    if rule == "number":
        rule_text = f"Your task is to produce an action corresponding to the card at the top that matches the number or face (J, Q, K, A) of the prompt card, but not its suit."
    elif rule == "suit":
        rule_text = f"Your task is to produce an action corresponding to the card at the top that matches the suit (hearts, diamonds, clubs, spades) of the prompt card, but not its number or face."
    elif rule == "both":
        rule_text = f"Your task is to produce an action corresponding to the card at the top that matches both the number or face (J, Q, K, A) and the suit (hearts, diamonds, clubs, spades) of the prompt card (i.e., the same card)."
    elif rule == "neither":
        rule_text = f"Your task is to produce an action corresponding to the card at the top that matches neither the number or face (J, Q, K, A) nor the suit (hearts, diamonds, clubs, spades) of the prompt card."
    elif rule == "all":
        rule_text = f"""
Furthermore, in the bottom right corner of the image, there will be a small white symbol indicating the current rule for this episode:
- If the symbol is a circle, then your task is to produce an action corresponding to the card at the top that matches the number or face (J, Q, K, A) of the prompt card, but not its suit.
- If the symbol is a square, then your task is to produce an action corresponding to the card at the top that matches the suit (hearts, diamonds, clubs, spades) of the prompt card, but not its number or face.
- If the symbol is a triangle, then your task is to produce an action corresponding to the card at the top that matches both the number or face (J, Q, K, A) and the suit (hearts, diamonds, clubs, spades) of the prompt card (i.e., the same card).
- If the symbol is a star, then your task is to produce an action corresponding to the card at the top that matches neither the number or face (J, Q, K, A) nor the suit (hearts, diamonds, clubs, spades) of the prompt card.
            """
    p = inflect.engine()
    action_space_text = "The action space includes {n_cards} discrete actions:\n".format(n_cards=n_cards)
    for i in range(n_cards):
        action_space_text += f"- {i}: Choose the {p.ordinal(i+1)} card from the left\n"
    action_space_text = action_space_text.strip()

    formatted_prompt = prompt.format(n_cards=n_cards, n_decisions=n_decisions, rule=rule_text, action_space=action_space_text, image="{image}")
    return formatted_prompt

class CardMazeEnv(gym.Env):
    """
    Card matching, sequential-verification environment.

    - Four cards are shown at the top (fixed for the episode).
    - Each timestep a prompt card (bottom) is shown; it equals exactly one of the top cards.
    - Agent actions: Discrete(self.n_cards) selecting one of the top cards (0..self.n_cards-1).
    - The agent must make `n_decisions` correct selections in a row.
    - Reward: 0 for every step except +1 on the final step when all predictions were correct.
    - Any incorrect selection ends the episode immediately with reward 0.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_cards: int = 4,
        n_decisions: int = 5,
        rule: str = "number",  # one of ALLOWED_RULES
        progress_style: str = "text", # one of ALLOWED_PROGRESS_STYLES
        n_confounders: int = 2,  # if use_confounders is True, how many confounders to add (at most)
        card_size: Tuple[int, int] = (80, 112),  # width, height (approx 5:7)
        seed: Optional[int] = None,
        render_mode: str = "rgb_array",
    ):
        """
        Initialize the CardMaze environment.
        
        Args:
            n_cards (int): Number of top cards to display (2 to 51).
            n_decisions (int): Number of correct sequential actions to complete the episode (at least 1).
            rule (str): One of "number", "suit", "both", "neither", "all". Determines the matching rule.
            progress_style (str): One of "text", "symbol", "none". Determines how progress is displayed. (Note that if "none", the environment is non-Markovian/partially observable, as the agent cannot see how many correct selections have been made so far and thus can't anticipate the reward.)
            n_confounders (int): Number of confounder cards to include (0 for easiest). Ignored if rule is "neither".
            card_size (Tuple[int, int]): Width and height of each card image in pixels.
            seed (Optional[int]): Random seed for reproducibility.
            render_mode (str): Must be "rgb_array".
        """
        super().__init__()
        assert render_mode == "rgb_array"
        assert rule.lower() in ALLOWED_RULES, f"rule must be one of {ALLOWED_RULES}"
        assert progress_style.lower() in ALLOWED_PROGRESS_STYLES, f"progress_style must be one of {ALLOWED_PROGRESS_STYLES}"
        assert n_cards >= 2, "n_cards must be at least 2"
        assert n_cards <= 51, "n_cards cannot exceed 51 (52 total cards minus 1 correct)"
        assert n_decisions >= 1, "n_decisions must be at least 1"
        
        self.n_cards = int(n_cards)
        self.n_decisions = int(n_decisions)
        self.rule = rule.lower()
        self.progress_style = progress_style.lower()
        self.current_rule = None 
        self.use_confounders = (n_confounders > 0)
        self.n_confounders = n_confounders
        self.card_width, self.card_height = card_size
        
        if self.rule == "number" and self.n_confounders > 12:
            raise ValueError("For rule='number', n_confounders cannot exceed 12 (only 12 other cards with same suit but different number)")
        if self.rule == "suit" and self.n_confounders > 3:
            raise ValueError("For rule='suit', n_confounders cannot exceed 3 (only 3 other cards with same number but different suit)")
        if self.rule == "both" and self.n_confounders > 15:
            raise ValueError("For rule='both', n_confounders cannot exceed 15 (only 12 other cards with same suit and different number + 3 others with same number and different suit)")
        if self.rule == "neither" and self.n_cards > 16:
            raise ValueError("For rule='neither', n_cards cannot exceed 16 (1 correct + 12 others with same suit and different number + 3 others with same number and different suit)")
        if self.rule == "neither" and self.use_confounders:
            import warnings
            warnings.warn("For rule='neither', confounders are always used regardless of n_confounders, so the parameter is ignored.")
        if self.rule == "all" and self.n_confounders > 3:
            import warnings
            warnings.warn("For rule='all', the maximum number of confounders for each rule will be used (3 for suit, 12 for number, 15 for both, and 15 for neither) if the specified n_confounders exceeds those limits.")
        
        # compute canvas size based on card_size
        horizontal_spacing = self.card_width // 4 
        vertical_padding = self.card_height // 6
        scale_factor_prompt = 1.5
        prompt_height = int(self.card_height * scale_factor_prompt)
        
        self.canvas_width = self.card_width * self.n_cards + horizontal_spacing * (self.n_cards + 1)
        self.canvas_height = self.card_height + prompt_height + vertical_padding * 3 + 16 # extra 16 for text
        
        # cache of loaded and resized images
        self._image_cache: Dict[Tuple[str, Tuple[int, int]], Image.Image] = {}

        # RNGs (match style of your original env)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Gym spaces
        self.action_space = spaces.Discrete(self.n_cards)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.canvas_height, self.canvas_width, 3),
            dtype=np.uint8,
        )

        # Episode state
        self.episode_seed: int = int(self.np_rng.integers(0, MAX_RAND_SEED)) if seed is None else int(seed)
        self.cards: List[str] = []         # top card names like "H7"
        self.sequence: List[str] = []      # prompt card sequence (length n_decisions)
        self.idx: int = 0                  # progress index in sequence
        self.step_count: int = 0
        self.terminated: bool = False

        self._font = ImageFont.truetype(FONT_PATH, 16)

    # ---------------- seed/reset utils ----------------
    def seed(self, seed: Optional[int] = None):
        """Seed both RNGs."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
    def _refresh_top_cards(self):
        """Generate exactly self.n_cards top cards and the prompt for the current rule."""
        assert self.current_rule in ALLOWED_RULES, f"current_rule must be one of {ALLOWED_RULES}"
        rule = self.current_rule  # one of "number","suit","both","neither"
        make_card = lambda suit, num: suit + self._card_num_to_str(int(num))
        top_cards = []

        if rule == "number":
            # Prompt: a suit different from the correct top-card's suit, same number
            prompt_num = self.rng.randint(1, 13)
            prompt_suit = self.rng.choice(SUITS)
            prompt_card = make_card(prompt_suit, prompt_num)

            # Correct: same number, suit != prompt_suit
            correct_suit = self.rng.choice([s for s in SUITS if s != prompt_suit])
            correct_card = make_card(correct_suit, prompt_num)

            if self.use_confounders:
                # Confounder: same suit as prompt, different number
                confound_cards = []
                while len(confound_cards) < min(self.n_confounders, 12):
                    confound_num = self.rng.choice([n for n in range(1, 14) if n != prompt_num])
                    confound_card = make_card(prompt_suit, confound_num)
                    if confound_card not in confound_cards:
                        confound_cards.append(confound_card)
                
                top_cards = [correct_card] + confound_cards
        
        elif rule == "suit":
            # Prompt: chosen suit; ensure prompt and correct have different numbers
            prompt_suit = self.rng.choice(SUITS)
            prompt_num = self.rng.randint(1, 13)
            prompt_card = make_card(prompt_suit, prompt_num)

            # Correct: same suit, different number
            correct_num = self.rng.choice([n for n in range(1, 14) if n != prompt_num])
            correct_card = make_card(prompt_suit, correct_num)

            if self.use_confounders:
                # Confounder: same number as prompt, different suit
                confound_cards = []
                while len(confound_cards) < min(self.n_confounders, 3):
                    confound_suit = self.rng.choice([s for s in SUITS if s != prompt_suit])
                    confound_card = make_card(confound_suit, prompt_num)
                    if confound_card not in confound_cards:
                        confound_cards.append(confound_card)

                top_cards = [correct_card] + confound_cards

        elif rule == "both":
            # Prompt equals correct top-card (same suit and number)
            prompt_suit = self.rng.choice(SUITS)
            prompt_num = self.rng.randint(1, 13)
            prompt_card = make_card(prompt_suit, prompt_num)
            correct_card = prompt_card

            if self.use_confounders:
                confound_cards = []
                while len(confound_cards) < min(self.n_confounders, 15):
                    # Confounders differ in exactly one attribute (suit or number)
                    if self.rng.random() < 0.5:
                        # same suit, different number
                        num = self.rng.choice([n for n in range(1, 14) if n != prompt_num])
                        cand = make_card(prompt_suit, num)
                    else:
                        # same number, different suit
                        suit = self.rng.choice([s for s in SUITS if s != prompt_suit])
                        cand = make_card(suit, prompt_num)

                    if cand not in confound_cards:
                        confound_cards.append(cand)
    
                top_cards = [correct_card] + confound_cards

        elif rule == "neither":
            # Prompt chosen freely
            prompt_suit = self.rng.choice(SUITS)
            prompt_num = self.rng.randint(1, 13)
            prompt_card = make_card(prompt_suit, prompt_num)

            top_cards = []

            # Correct: differs in both suit and number
            while True:
                cand_num = self.rng.randint(1, 13)
                cand_suit = self.rng.choice(SUITS)
                if cand_num != prompt_num and cand_suit != prompt_suit:
                    correct_card = make_card(cand_suit, cand_num)
                    break
            top_cards.append(correct_card)

            # Confounders are always used for neither, otherwise we get repeats: share suit OR number with prompt, but not both
            used = {correct_card}
            while len(top_cards) < self.n_cards:
                if self.rng.random() < 0.5:
                    # share suit, different number
                    num = self.rng.choice([n for n in range(1, 14) if n != prompt_num])
                    cand = make_card(prompt_suit, num)
                else:
                    # share number, different suit
                    suit = self.rng.choice([s for s in SUITS if s != prompt_suit])
                    cand = make_card(suit, prompt_num)

                if cand not in used:
                    top_cards.append(cand)
                    used.add(cand)
        
        else:
            raise ValueError(f"Unknown rule: {rule}")
        
        if not correct_card in top_cards:
            # happens when n_confounders=0
            top_cards = [correct_card] + top_cards
        
        if (not self.use_confounders and rule != "neither") or len(top_cards) < self.n_cards:
            # Easy fillers: no prompt number or suit
            used = {correct_card, *top_cards}
            fillers = []
            while len(used) < self.n_cards:
                n = self.rng.randint(1, 13)
                s = self.rng.choice([s for s in SUITS if s != prompt_suit])
                candidate = make_card(s, n)
                if n == prompt_num or candidate in used:
                    continue
                used.add(candidate)
                fillers.append(candidate)

            top_cards += fillers
        
        # final safety check: all unique and length n_cards
        if len(top_cards) != self.n_cards or len(set(top_cards)) != self.n_cards:
            print(top_cards, len(top_cards), len(set(top_cards)), self.n_cards)
            raise RuntimeError(f"Internal error generating top cards for rule={rule}: {top_cards}")

        self.rng.shuffle(top_cards)
        self.cards = top_cards

        # populate remaining prompts in the sequence
        if not self.sequence or len(self.sequence) != self.n_decisions:
            self.sequence = [prompt_card for _ in range(self.n_decisions)]
        else:
            for i in range(self.idx, self.n_decisions):
                self.sequence[i] = prompt_card

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.episode_seed = int(self.np_rng.integers(0, MAX_RAND_SEED)) if seed is None else int(seed)
        self.seed(self.episode_seed)
        self.sequence = [] # clear previous sequence
        
        # randomize rule if needed
        if self.rule == "all":
            self.current_rule = self.rng.choice([r for r in ALLOWED_RULES if r != "all"])
        else:
            self.current_rule = self.rule

        # initial top cards (even if dynamic, we need first frame)
        self._refresh_top_cards()

        self.idx = 0
        self.step_count = 0
        self.terminated = False

        info = self._get_info(last_correct=False)
        return self._render_obs(), info

    # ---------------- info / helpers ----------------
    def _get_info(self, last_correct: bool) -> Dict:
        correct_card = ""
        correct_index = -1 # must be int for SyncVectorEnv
        if not self.terminated:
            # current prompt card and which top-card index is correct
            prompt_card = self.sequence[self.idx]
            correct_index = self._current_correct_action()
            correct_card = prompt_card
        return {
            "episode_seed": self.episode_seed,
            "current_rule": self.current_rule,    # one of ALLOWED_RULES
            "cards": self.cards,                  # top card names
            "sequence": self.sequence,            # whole prompt sequence
            "correct_card": correct_card,         # current prompt (or None if terminated)
            "correct_action": correct_index,      # index 0..3 for the current correct card, or None
            "progress": (self.idx, self.n_decisions),
            "last_action_correct": last_correct,
        }

    def _current_correct_action(self) -> int:
        """Return the index of the correct top card for the current prompt."""
        if self.idx >= self.n_decisions:
            return -1 # this happens when the episode is already finished
        prompt = self.sequence[self.idx]

        if self.current_rule == "number":
            # match number only
            prompt_number = prompt[1:]
            for i, card in enumerate(self.cards):
                if card[1:] == prompt_number:
                    return i
        elif self.current_rule == "suit":
            # match suit only
            prompt_suit = prompt[0]
            for i, card in enumerate(self.cards):
                if card[0] == prompt_suit:
                    return i
        elif self.current_rule == "both":
            # match both number and suit
            for i, card in enumerate(self.cards):
                if card == prompt:
                    return i
        elif self.current_rule == "neither":
            # match neither number nor suit
            for i, card in enumerate(self.cards):
                if card[0] != prompt[0] and card[1:] != prompt[1:]:
                    return i
        raise RuntimeError(f"Rule {self.current_rule} could not find a matching card for prompt {prompt} in {self.cards}")


    # ---------------- step / termination ----------------
    def step(self, action: int):

        if self.terminated:
            # follow original env behavior: if called after termination, return final state and zero reward
            return self._render_obs(), 0.0, True, False, self._get_info(last_correct=False)

        assert self.action_space.contains(action)
        self.step_count += 1
        correct = self._current_correct_action()
        truncated = False
        reward = 0.0

        if action == correct:
            # correct selection for this step
            self.idx += 1
            self._refresh_top_cards() # new top cards for next step
            # If we've completed the full sequence, give final reward 1
            if self.idx >= self.n_decisions:
                reward = 1.0
                self.terminated = True
            else:
                reward = 0.0
        else:
            # any mistake ends the episode, reward stays 0
            self.terminated = True
            reward = 0.0

        obs = self._render_obs()
        info = self._get_info(last_correct=(action == correct))
        return obs, reward, self.terminated, truncated, info
    
    def _draw_rule_symbol(self, canvas: Image.Image, rule: str):
        """Draw a small symbol in the bottom-right corner indicating the current rule."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        margin = 10
        size = 30
        x0 = self.canvas_width - size - margin
        y0 = self.canvas_height - size - margin
        x1, y1 = x0 + size, y0 + size

        if self.current_rule == "number":  # circle
            draw.ellipse([x0, y0, x0 + size, y0 + size], fill="white")
        elif self.current_rule == "suit":  # square
            draw.rectangle([x0, y0, x0 + size, y0 + size], fill="white")
        elif self.current_rule == "both":  # triangle
            draw.polygon([(x0+size/2, y0), (x0, y0+size), (x0+size, y0+size)], fill="white")
        elif self.current_rule == "neither":  # star
            cx, cy = x0 + size / 2, y0 + size / 2
            r_outer = size / 2
            r_inner = r_outer * 0.5  # inner radius

            points = []
            for i in range(10):
                angle_deg = i * 36 - 90  # start pointing up
                r = r_outer if i % 2 == 0 else r_inner
                angle_rad = math.radians(angle_deg)
                x = cx + r * math.cos(angle_rad)
                y = cy + r * math.sin(angle_rad)
                points.append((x, y))

            draw.polygon(points, fill="white")

    # ---------------- rendering ----------------
    def _get_cached_image(self, card_name: str, size: Tuple[int, int]) -> Image.Image:
        """Return a cached resized image for a card."""
        key = (card_name, size)
        if key not in self._image_cache:
            # load once and resize
            img = get_image(card_name).resize(size, Image.LANCZOS)
            self._image_cache[key] = img
        return self._image_cache[key]
    
    def _render_obs(self) -> np.ndarray:
        """Produce an RGB observation with top n_cards cards and a bottom prompt card."""
        # canvas background (green table)
        canvas = Image.new("RGB", (self.canvas_width, self.canvas_height), "#35654d")

        # spacing and vertical positions
        horizontal_spacing = self.card_width // 4
        vertical_padding = self.card_height // 6
        y_top = vertical_padding

        # paste the top cards
        for i, card_name in enumerate(self.cards[:self.n_cards]):
            x = horizontal_spacing + i * (self.card_width + horizontal_spacing)
            img = self._get_cached_image(card_name, (self.card_width, self.card_height))
            canvas.paste(img, (x, y_top), img)

        # determine current prompt to display (if terminated, show last prompt)
        if self.idx < self.n_decisions:
            prompt_card = self.sequence[self.idx]
        else:
            prompt_card = self.sequence[-1]

        # paste the prompt at the bottom center, bigger than top cards
        scale_factor = 1.5
        prompt_w = int(self.card_width * scale_factor)
        prompt_h = int(self.card_height * scale_factor)
        prompt_img = self._get_cached_image(prompt_card, (prompt_w, prompt_h))
        x_prompt = (self.canvas_width - prompt_w) // 2
        y_prompt = self.canvas_height - prompt_h - vertical_padding - 16
        canvas.paste(prompt_img, (x_prompt, y_prompt), prompt_img)

        # if we are in "all" mode, overlay symbol for the current rule
        if self.rule == "all":
            self._draw_rule_symbol(canvas, self.current_rule)
        
        # progress indicator
        draw = ImageDraw.Draw(canvas)
        if self.progress_style == "text":
            progress_text = f"Progress: {self.idx} / {self.n_decisions}"
            text_x = 10
            text_y = self.canvas_height - 30
            draw.text((text_x, text_y), progress_text, font=self._font, fill="white")
        elif self.progress_style == "symbol":
            radius = 8
            spacing = 8
            total_width = self.n_decisions * (2 * radius) + (self.n_decisions - 1) * spacing
            start_x = (self.canvas_width - total_width) // 2
            y = self.canvas_height - 20
            for i in range(self.n_decisions):
                x = start_x + i * (2 * radius + spacing) + radius
                if i < self.idx:
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="white")
                else:
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="white", width=2)
        
        return np.array(canvas, dtype=np.uint8)

    
    def render(self):
        return self._render_obs()

    # ---------------- utility ----------------
    def _card_num_to_str(self, num: int) -> str:
        face_cards = {1: "A", 10: "T", 11: "J", 12: "Q", 13: "K"}
        return face_cards.get(num, str(num))
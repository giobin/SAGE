<user>
{image}

You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library. You will receive observations from the environment and must decide which action to take next.
The goal is to navigate across a frozen grid to reach the goal without falling into holes.

## Observation space
You are presented with the image of the environment in the style of pixel art.
Each cell may be safe (white with snow), a hole (bright blue ice hole), the start (a stool) or the goal (a present box). The agent is represented by an elf character.

## Action space
The action space includes 4 discrete actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

# Output Format
You should provide the chosen action index inside the <action> </action> tags. All text outside the <think> and <action> tags will be ignored.

Based on the current visual information, which action should the agent take next?

</user>
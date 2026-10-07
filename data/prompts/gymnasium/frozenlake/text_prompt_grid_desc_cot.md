<system>
You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library. You will receive observations from the environment and must decide which action to take next.
You should about the answer step by step inside the <think> tag, and then provide the action inside the <action> tag. All text outside the <think> and <action> tags will be ignored.
</system>

<user>
The goal is to navigate across a frozen grid to reach the goal without falling into holes. 

## Observation space
You will be given the coordinates (x, y) of the agent's current position, the coordinates of the goal, and the coordinates of any holes in the grid.
The coordinates start from (0, 0) at the top-left corner and increase to the right and downwards.

## Action space
The action space includes 4 discrete actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

## Current observation
{observation}

Based on the current observation, which action should the agent take next?  
</user>
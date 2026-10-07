<system>
You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library.
</system>

<user>
The goal is to navigate across a frozen grid to reach the goal without falling into holes. 

## Observation space
The environment is represented as a matrix of integers, with the following mapping:
- 0: Start
- 1: Frozen (safe)
- 2: Hole (unsafe)
- 3: Goal
- 4: Current position of the agent

## Action space
The action space includes 4 discrete actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

## Current observation
{observation}

Based on the current observation, which action should the agent take next?  
Respond only with the number corresponding to the best action.
</user>
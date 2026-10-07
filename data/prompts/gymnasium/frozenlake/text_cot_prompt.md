<system>
You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library. You should think about the answer step by step inside the <think> tag, and then provide the action inside the <action> tag. All text outside the <think> and <action> tags will be ignored.
</system>

<user>
The goal is to navigate across a 4x4 frozen grid to reach the goal without falling into holes. The environment is represented as a discrete state space with 16 possible states (from 0 to 15), where each number indicates the agent's current tile.

The action space includes 4 discrete actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

At each timestep, you are given the current state of the environment.

Current state: {observation}

Based on the current state, which action should the agent take next?  
</user>
<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
Provide the action inside the <action> tag.

## Environment
This environment is a grid with a wall of lava blocking the direct path. There is a gap in the lava wall that the agent must find and navigate through to reach the green goal square. Stepping on lava terminates the episode with zero reward.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares around the border are walls. Orange squares with a zigzag pattern are lava (deadly). The bright green square is the goal.

## Agent and Movement
The agent is the red triangle. The direction the triangle points indicates which way the agent is facing:
- Triangle pointing right (►): agent faces right. "Move forward" moves the agent one cell to the RIGHT.
- Triangle pointing down (▼): agent faces down. "Move forward" moves the agent one cell DOWN.
- Triangle pointing left (◄): agent faces left. "Move forward" moves the agent one cell to the LEFT.
- Triangle pointing up (▲): agent faces up. "Move forward" moves the agent one cell UP.

## Action space
The action space includes 7 discrete actions:
- 0: Turn left (rotate 90° counter-clockwise, staying in the same cell)
- 1: Turn right (rotate 90° clockwise, staying in the same cell)
- 2: Move forward one cell in the direction the agent is facing
- 3: Pick up the object in the cell ahead
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode)

## Mission
{mission}

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

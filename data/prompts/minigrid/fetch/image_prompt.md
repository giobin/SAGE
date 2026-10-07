<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
Provide the action inside the <action> tag.

## Environment
This environment has multiple objects of assorted types and colors. You will receive a mission telling you which object to pick up. Picking up the wrong object terminates the episode with zero reward.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares are walls. Objects (keys, balls) are solid — the agent cannot walk through them. To pick up an object, the agent must be facing it and use action 3.

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
- 3: Pick up the object in the cell ahead (use when facing the target object)
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode)

## Mission
{mission}

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

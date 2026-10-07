<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
Provide the action inside the <action> tag.

## Environment
This environment is a grid room with 4 colored doors placed on the walls. The agent must navigate to the door specified in the mission and stand in front of it (facing it), then use the "done" action (action 6) to complete the episode successfully.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares are walls. Colored rectangles on the walls represent doors.

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
- 6: Done (end the episode — use this when facing the target door)

## Mission
{mission}

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

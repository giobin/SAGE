<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
You should think about the answer step by step inside the <think> tag, reasoning about what you see and how it relates to your mission, then provide the action inside the <action> tag. All text outside the <think> and <action> tags will be ignored.

## Environment
This environment is a grid room with 4 colored doors placed on the walls. The agent must navigate to the door specified in the mission and stand in front of it (facing it), then use the "done" action (action 6) to complete the episode successfully.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares are walls. Colored rectangles on the walls represent doors. Each door has a distinct color (red, blue, green, yellow, purple, or grey).

## Agent and Movement
The agent is the red triangle. The direction the triangle points indicates which way the agent is facing:
- Triangle pointing right (►): agent faces right. "Move forward" moves the agent one cell to the RIGHT.
- Triangle pointing down (▼): agent faces down. "Move forward" moves the agent one cell DOWN.
- Triangle pointing left (◄): agent faces left. "Move forward" moves the agent one cell to the LEFT.
- Triangle pointing up (▲): agent faces up. "Move forward" moves the agent one cell UP.
The agent can only move forward in the direction it is facing. To move in a different direction, the agent must first turn.

## Action space
The action space includes 7 discrete actions:
- 0: Turn left (rotate 90° counter-clockwise, staying in the same cell)
- 1: Turn right (rotate 90° clockwise, staying in the same cell)
- 2: Move forward one cell in the direction the agent is facing
- 3: Pick up the object in the cell ahead
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode — use this when facing the target door)

## Decision procedure
1. Identify the target door color from the mission.
2. Locate the target door on the grid walls.
3. If you are directly facing the target door (one cell away), use action 6 (done).
4. Otherwise, navigate toward the target door: turn to face the right direction, then move forward.
5. Prefer moving forward over turning whenever possible — unnecessary turning wastes steps.

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
go to the blue door

## Observation:
The map is a 5x5 grid room. The agent (red triangle) is in the center, pointing up. The blue door is on the top wall, two cells ahead. There is empty floor between the agent and the blue door.

The agent is facing: up

<think>
The mission is to go to the blue door, which is on the top wall. I am facing up, toward the blue door. The cell directly above me is empty floor — safe to move forward. Moving forward brings me closer to the blue door.
</think>
<action>2</action>

## Example 2
## Mission:
go to the red door

## Observation:
The map is a 5x5 grid room. The agent (red triangle) is directly in front of the red door on the bottom wall, pointing down toward it. The red door is in the cell directly ahead.

The agent is facing: down

<think>
I am directly facing the red door, which is the target door from the mission. The door is in the cell right in front of me. I should use the "done" action to complete the episode.
</think>
<action>6</action>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

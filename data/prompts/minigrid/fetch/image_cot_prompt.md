<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
You should think about the answer step by step inside the <think> tag, reasoning about what you see and how it relates to your mission, then provide the action inside the <action> tag. All text outside the <think> and <action> tags will be ignored.

## Environment
This environment has multiple objects of assorted types and colors. You will receive a mission telling you which object to pick up. Picking up the wrong object terminates the episode with zero reward.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares are walls. Objects (keys, balls) are solid — the agent cannot walk through them. To pick up an object, the agent must be facing it (one cell away) and use action 3.

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
- 3: Pick up the object in the cell ahead (use when facing the target object)
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode)

## Decision procedure
1. Identify the target object (type and color) from the mission.
2. Locate the target object on the grid.
3. If you are directly facing the target object (one cell away), use action 3 (pick up).
4. Otherwise, navigate toward the target object: turn to face the right direction, then move forward.
5. Prefer moving forward over turning whenever possible — unnecessary turning wastes steps.

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
go get a purple key

## Observation:
The map is a 5x5 grid. The agent (red triangle) is in the bottom-right area, pointing up. There is a purple key two cells above. The cell directly above the agent is empty floor.

The agent is facing: up

<think>
The mission is to get the purple key. The purple key is above me. I am facing up, and the cell directly above me is empty floor. Moving forward will bring me closer to the key.
</think>
<action>2</action>

## Example 2
## Mission:
fetch a red ball

## Observation:
The map is a 5x5 grid. The agent (red triangle) is pointing left. The red ball is directly to the left of the agent, in the next cell.

The agent is facing: left

<think>
The red ball is directly in front of me (one cell to the left, and I'm facing left). I should pick it up with action 3.
</think>
<action>3</action>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

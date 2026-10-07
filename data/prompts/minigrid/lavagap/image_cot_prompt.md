<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
You should think about the answer step by step inside the <think> tag, reasoning about what you see and how it relates to your mission, then provide the action inside the <action> tag. All text outside the <think> and <action> tags will be ignored.

## Environment
This environment is a grid with a wall of lava blocking the direct path. There is a gap in the lava wall that the agent must find and navigate through to reach the green goal square. Stepping on lava terminates the episode with zero reward.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares around the border are walls. Orange squares with a zigzag pattern are lava (deadly). The bright green square is the goal.

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
- 6: Done (end the episode)

## Decision procedure
1. Identify what is directly in front of the agent (the cell the triangle is pointing at).
2. If the cell ahead is safe and brings you closer to the goal, move forward (action 2).
3. If the cell ahead is dangerous (lava) or a wall, turn to face a safe direction first.
4. Prefer moving forward over turning whenever possible — unnecessary turning wastes steps.

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
avoid the lava and get to the green goal square

## Observation:
The map shows the agent (red triangle) pointing downward. Below the agent is an empty dark grey cell. The lava wall is to the right.

The agent is facing: down

<think>
The agent faces down. The cell directly below is empty dark grey floor — safe to walk on. Moving forward (down) will bring me closer to the gap in the lava wall. I should move forward.
</think>
<action>2</action>

## Example 2
## Mission:
avoid the lava and get to the green goal square

## Observation:
The map shows the agent (red triangle) pointing right. Directly to the right of the agent is an orange lava cell. The gap in the lava wall is one row below.

The agent is facing: right

<think>
The agent faces right, but the cell directly to the right is lava — stepping there would end the episode. I need to go down to reach the gap. Since I'm facing right, I should turn right (action 1) to face down, then I can move forward through the gap.
</think>
<action>1</action>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next?
</user>

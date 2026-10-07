<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
You should think about the answer step by step inside the <think> tag, reasoning about what you see and how it relates to your mission, then provide the action inside the <action> tag and the probability distribution over all actions inside the <probs> tag. All text outside the <think>, <action> and <probs> tags will be ignored.

## Environment
This environment has multiple objects of assorted types and colors. You will receive a mission telling you which object to pick up. Picking up the wrong object terminates the episode with zero reward. 
The environment is represented as a grid. Black squares represent empty spaces, while grey squares represent walls. The agent is represented by a red triangle, and the direction the agent is facing is indicated by the triangle's point. Objects are solid, the agent cannot pass through them. To pick up an object, the agent must be facing it and use the appropriate action.

## Action space
The action space includes 7 discrete actions:
- 0: Turn left (in place)
- 1: Turn right (in place)
- 2: Move forward one cell in the direction the agent is facing
- 3: Pick up the object in the cell in the direction the agent is facing
- 4: Drop the object in the cell in the direction the agent is facing
- 5: Toggle the object in the cell in the direction the agent is facing
- 6: Done (end the episode)

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
go get a purple key

## Observation:
The map is 3x3 and shows a red triangle in the bottom right cell. There is also a grey ball to the left of the triangle, and a purple key in the top-middle cell.

The agent is facing: right

<think>
To reach the purple key (top-middle), the agent must move up two cells and left one cell.
However, the agent is facing right, and immediately to its right is a wall / outside the grid, so moving forward would fail.
The agent must first reorient. To go toward the key, the agent must eventually face up.
From its current orientation (right), the minimal turn to face up is:
Turn left (Action 0).
</think>
<action>0</action>
<probs>0.95, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0</probs>

## Example 2
## Mission:
fetch a red ball

## Observation:
The map is 3x3 and shows a red triangle in the bottom central cell. There is also a red ball at the middle of the grid. A green key is positioned at the top-right cell.

The agent is facing: left

<think>
To reach the red ball (centre grid), the agent must move up one cell.
However, the agent is facing left, so moving forward would bring him further away from the goal.
The agent must first reorient. To go toward the red ball, the agent must face up.
From its current orientation (left), the minimal turn to face up is to turn right:
action 1.
Other movements are possible but they will eventually require the agent to walk a longer route to the ball.
</think>
<action>1</action>
<probs>0.05, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0</probs>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next? Respond first with your reasoning inside the <think> tag, then provide the action inside the <action> tag and the probability distribution over the actions (ordered as in the list above) inside the <probs> tag.
</user>

<user>
{image}

You are a smart agent acting in a gridworld environment. You must reach the green goal square while avoiding orange lava (stepping on lava terminates the episode with zero reward).
You should think about the answer step by step inside the <think> tag, then provide the action inside the <action> tag and the probability distribution over all actions inside the <probs> tag. All text outside the <think>, <action> and <probs> tags will be ignored.

## Environment
The environment is a grid with a wall of lava blocking the path. There is a gap in the lava wall that the agent must find and navigate through to reach the green goal square.
Dark grey squares are walkable floor. Lighter grey squares are walls. Orange squares are lava (instant death). The green square is the goal. The agent is the red triangle; its facing direction is indicated by the triangle's point.

## Action space
The action space includes 7 discrete actions:
- 0: Turn left (rotate 90 counter-clockwise, staying in the same cell)
- 1: Turn right (rotate 90 clockwise, staying in the same cell)
- 2: Move forward one cell in the direction the agent is facing
- 3: Pick up the object in the cell ahead
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode)

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
reach the green goal

## Observation:
The agent is on the left side of the grid, facing right. There is a gap in the lava wall directly ahead. The green goal is beyond the gap.

The agent is facing: right

<think>
The gap in the lava wall is directly ahead. Moving forward will take me through the gap toward the goal. This is the safest path.
</think>
<action>2</action>
<probs>0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0</probs>

## Example 2
## Mission:
reach the green goal

## Observation:
The agent is facing right. Lava is directly ahead. The gap in the lava wall is one row below.

The agent is facing: right

<think>
Lava is directly ahead. The gap is below me. I need to turn right to face down, then move forward to align with the gap.
</think>
<action>1</action>
<probs>0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1</probs>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next? Respond with your reasoning in <think>, the action in <action>, and the probability distribution over all 7 actions in <probs>.
</user>

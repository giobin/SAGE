<user>
{image}

You are a smart agent acting in a gridworld environment. You will receive observations from the environment and must decide which action to take next based on the mission you are given.
You should think about the answer step by step inside the <think> tag, reasoning about what you see and how it relates to your mission, then provide the action inside the <action> tag and the probability distribution over all actions inside the <probs> tag. All text outside the <think>, <action> and <probs> tags will be ignored.

## Environment
This environment is a grid room with 4 colored doors placed on the walls. The agent must navigate to the door specified in the mission and face it, then use the "done" action (action 6) to complete the episode.
The environment is represented as a grid. Dark grey squares are walkable floor. Lighter grey squares are walls. Colored rectangles on walls are doors. The agent is the red triangle; its facing direction is indicated by the triangle's point.

## Action space
The action space includes 7 discrete actions:
- 0: Turn left (rotate 90 counter-clockwise, staying in the same cell)
- 1: Turn right (rotate 90 clockwise, staying in the same cell)
- 2: Move forward one cell in the direction the agent is facing
- 3: Pick up the object in the cell ahead
- 4: Drop the object in the cell ahead
- 5: Toggle the object in the cell ahead
- 6: Done (end the episode - use when facing the target door)

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------
## Example 1
## Mission:
go to the blue door

## Observation:
The agent is in the center of a 5x5 room. There are 4 colored doors on the walls. The blue door is on the top wall. The agent is facing up, with one empty cell between it and the blue door.

The agent is facing: up

<think>
The blue door is on the top wall. I am facing up with a clear path. I should move forward to get closer to the blue door.
</think>
<action>2</action>
<probs>0.05, 0.05, 0.9, 0.0, 0.0, 0.0, 0.0</probs>

## Example 2
## Mission:
go to the red door

## Observation:
The agent is directly facing the red door on the right wall, one cell away.

The agent is facing: right

<think>
The red door is directly ahead of me. I am facing it and one cell away. I should use the "done" action to complete the episode.
</think>
<action>6</action>
<probs>0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0</probs>
--------------------------------

## Current Task
## Mission
{mission}

## History:

The agent is facing: {direction}

Based on the current information, which action should the agent take next? Respond with your reasoning in <think>, the action in <action>, and the probability distribution over all 7 actions in <probs>.
</user>

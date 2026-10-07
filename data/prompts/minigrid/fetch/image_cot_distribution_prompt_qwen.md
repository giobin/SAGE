<user>
{image}
Gridworld agent. Pick up the target object from the mission.

Dark grey = floor. Light grey = walls. Red triangle = agent (points in facing direction). To pick up: face the object, then action 3.

Actions: 0=turn left, 1=turn right, 2=forward, 3=pick up, 4=drop, 5=toggle, 6=done.

CRITICAL: Respond in UNDER 40 WORDS total. Think in <think></think>, answer in <action></action>, then give probabilities over all 7 actions in <probs></probs>. No other text.

## Examples
Mission: get purple key. Facing: up. Key is above, cell ahead clear.
<think>Key above, facing up, clear ahead. Forward.</think>
<action>2</action>
<probs>0.05, 0.05, 0.9, 0.0, 0.0, 0.0, 0.0</probs>

Mission: fetch red ball. Facing: left. Ball directly ahead.
<think>Ball ahead. Pick up.</think>
<action>3</action>
<probs>0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0</probs>

## Mission
{mission}

## History:

Facing: {direction}
</user>

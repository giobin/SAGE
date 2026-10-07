<user>
{image}
Gridworld agent. Reach the green goal square. Avoid orange lava (instant death).

Dark grey = floor. Light grey = walls. Orange = lava. Green = goal. Red triangle = agent (points in facing direction).

Actions: 0=turn left, 1=turn right, 2=forward, 3=pick up, 4=drop, 5=toggle, 6=done.

CRITICAL: Respond in UNDER 40 WORDS total. Think in <think></think>, answer in <action></action>, then give probabilities over all 7 actions in <probs></probs>. No other text.

## Examples
Mission: reach green goal. Facing: right. Gap in lava directly ahead.
<think>Gap ahead, safe to advance. Forward.</think>
<action>2</action>
<probs>0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0</probs>

Mission: reach green goal. Facing: right. Lava directly ahead, gap one row below.
<think>Lava ahead. Turn right to face gap.</think>
<action>1</action>
<probs>0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1</probs>

## Mission
{mission}

## History:

Facing: {direction}
</user>

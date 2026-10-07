<user>
{image}
Gridworld agent. Reach the green goal square. Avoid orange lava (instant death).

Dark grey = floor. Light grey = walls. Orange = lava. Green = goal. Red triangle = agent (points in facing direction).

Actions: 0=turn left, 1=turn right, 2=forward, 3=pick up, 4=drop, 5=toggle, 6=done.

CRITICAL: Respond in UNDER 30 WORDS total. Think in <think></think>, answer in <action></action>. No other text.

## Examples
Mission: reach green goal. Facing: down. Cell below is safe floor.
<think>Safe floor ahead. Forward.</think>
<action>2</action>

Mission: reach green goal. Facing: right. Lava directly ahead.
<think>Lava ahead. Turn right to face down.</think>
<action>1</action>

## Mission
{mission}

## History:

Facing: {direction}
</user>

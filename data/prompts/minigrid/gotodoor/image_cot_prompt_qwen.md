<user>
{image}
Gridworld agent. Navigate to the target colored door and use action 6 (done) when facing it.

Dark grey = floor. Light grey = walls. Colored rectangles on walls = doors. Red triangle = agent (points in facing direction).

Actions: 0=turn left, 1=turn right, 2=forward, 3=pick up, 4=drop, 5=toggle, 6=done.

CRITICAL: Respond in UNDER 30 WORDS total. Think in <think></think>, answer in <action></action>. No other text.

## Examples
Mission: go to blue door. Facing: up. Blue door on top wall, cell ahead clear.
<think>Blue door ahead, clear path. Forward.</think>
<action>2</action>

Mission: go to red door. Facing: down. Red door directly ahead.
<think>Red door ahead. Done.</think>
<action>6</action>

## Mission
{mission}

## History:

Facing: {direction}
</user>

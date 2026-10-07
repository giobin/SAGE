<user>
{image}
FrozenLake: navigate the elf to the present (goal) without falling into holes.

White = safe ice. Blue = hole (death). Stool = start. Present = goal. Elf = agent.

Actions: 0=left, 1=down, 2=right, 3=up.

CRITICAL: Respond in UNDER 40 WORDS total. Think in <think></think>, answer in <action></action>, then give probabilities over [left, down, right, up] in <probs></probs>. No other text.

## Examples
Elf at row 4 col 7. Hole left and below.
<think>Holes left and below. Move right.</think>
<action>2</action>
<probs>0.0, 0.0, 0.75, 0.25</probs>

Elf at row 3 col 3. Hole right, safe below toward goal.
<think>Hole right. Go down toward goal.</think>
<action>1</action>
<probs>0.1, 0.8, 0.0, 0.1</probs>

## History:

Which action and probability distribution?
</user>

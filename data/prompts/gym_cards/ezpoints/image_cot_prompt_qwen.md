<user>
{image}
You are playing a card game. Build a math formula using poker card numbers that evaluates to 12. Use all card numbers exactly once with + or * operators. "J"/"Q"/"K" = 10, Ace = 1. The game ends when "=" is added.

## Action space (13 actions)
- 1-10: add that number to the formula
- 11: add "+" operator
- 12: add "*" operator
- 13: add "=" operator (submit formula)

## Rules
- Only use numbers shown on your cards
- Formula must equal 12 when "=" is pressed
- Look at the current formula state to decide the next symbol

RESPOND IN UNDER 50 WORDS. Think in <think></think>, answer in <action></action>.

## Examples
Cards: J(=10), 2. Formula: empty.
<think>10+2=12. Start with 2.</think>
<action>2</action>

Cards: 7, 5. Formula: "7".
<think>7+5=12. Add "+".</think>
<action>11</action>

## Current Task
## History:

Which action next?
</user>

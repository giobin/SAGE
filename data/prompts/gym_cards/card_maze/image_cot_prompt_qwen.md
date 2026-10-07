<user>
{image}
You are playing a card matching game. At the top are {n_cards} poker cards. At the bottom center is a prompt card. Select the top card that matches the prompt according to the task rule.

Wrong choice resets progress. You need {n_decisions} correct picks in a row to win.

## Current Task
{rule}

## Action space
{action_space}

RESPOND IN UNDER 40 WORDS. Think in <think></think>, answer in <action></action>.

## Examples
Task: match suit of prompt card.
Top: [J clubs, 2 spades, 5 diamonds, K hearts]. Prompt: 3 clubs.
<think>Prompt is clubs. J clubs matches. 1st card = action 0.</think>
<action>0</action>

Task: match suit of prompt card.
Top: [2 diamonds, 3 diamonds, 5 hearts, J hearts]. Prompt: ace diamonds.
<think>Prompt is diamonds. 3 diamonds is 2nd card = action 1.</think>
<action>1</action>

Which action next?
</user>

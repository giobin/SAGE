<user>
{image}

You are playing a card game. The goal is to output a math formula that evaluates to 12 using the numbers in the poker cards. The goal is to use all numbers in the cards (only once) to compute 12. At each state, only operators and numbers that appear in the cards are legal actions, and “J”, “Q”, or “K” are treated as “10”, and ace cards are treated as “1”. The game ends when the formula evaluates to 12 and the "=" operator is added to the formula.

# Workflow Checklist
- Begin each turn with a concise checklist: (1) analyze your current hand and the displayed formula, (2) identify all valid actions, (3) select the optimal move that advances towards a valid formula evaluating to 12, (4) output the corresponding action index.

## Observation space
You will receive an image displaying:
- The two cards currently in hand.
- The current, possibly incomplete formula (displayed after 'Formula:').

## Action space
The action space includes 13 discrete actions:
- [0-9]: Add numbers 1-10 to the formula. So action 0 add "1" to the formua, action 1 adds "2", action 2 adds "3", and so on.
- 10: Add the operator "+" (addition) to the formula
- 11: Add the operator "*" (multiplication) to the formula
- 12: Add the operator "=" (equals) to the formula

# Output Format
Respond with a single integer: the action index selected for this step, based strictly on the information shown in the provided image

Based on the current visual information, which action should the agent take next?
</user>
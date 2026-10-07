<user>
{image}

You are playing a card game. The goal is to output a math formula that evaluates to 12 using the numbers in the poker cards. The goal is to use all numbers in the cards (only once) to compute 12. At each state, only numbers that appear in the cards and math operators are legal actions. “J”, “Q”, or “K” are treated as “10”, and ace cards are treated as “1”. The game ends when the "=" operator is added to the formula.

# Workflow Checklist
- Begin each turn with a concise checklist: (1) analyze your current hand and the displayed formula, (2) identify all valid actions, (3) select the optimal move that advances towards a valid formula evaluating to 12, (4) output the corresponding action number.

## Observation space
You will receive an image displaying:
- The two cards currently in hand.
- The current, possibly incomplete formula (displayed after 'Formula:').

## Action space
The action space includes 13 discrete actions:
- 1: add number "1" to the formula
- 2: add number "2" to the formula
- 3: add number "3" to the formula
- 4: add number "4" to the formula
- 5: add number "5" to the formula
- 6: add number "6" to the formula
- 7: add number "7" to the formula
- 8: add number "8" to the formula
- 9: add number "9" to the formula
- 10: add number "10" to the formula
- 11: Add the operator "+" (addition) to the formula
- 12: Add the operator "*" (multiplication) to the formula
- 13: Add the operator "=" (equals) to the formula

# Output Format
You should think about the answer step by step inside the <think> </think> tags, then provide the chosen action index within the <action> </action> tags. All text outside the <think> and <action> tags will be ignored.

--------------------------------
# Few-shot Examples
--------------------------------

## Example 1
## Observation:
image: [the image shows two cards in hand: "J of clubs" and "2 of spades"; Formula: "" (empty)]

<think>
The cards in hand are a 2 of spades and a Jack of clubs. Since Jack is treated as 10, the numbers available are 2 and 10. The goal is to create a formula that evaluates to 12. 

Possible actions:
1. Add the number 2 to the formula.
2. Add the number 10 to the formula.
3. Add the addition operator (+).
4. Add the multiplication operator (*).
5. Add the equals operator (=) if a valid formula is formed.

Starting with the numbers:
- 2 and 10 can be combined with addition: 2 + 10 = 12.

The simplest formula is 2 + 10 = 12. Also 10 + 2 = 12 is valid.
Let's go with the first option, so add the number 2 to the formula.
Number 2 corresponds to action 2.
</think>
<action>2</action>

## Example 2
## Observation:
image: [the image shows two cards in hand: "7 of diamonds" and "5 of diamonds"; Formula: "10"]

<think>
Okay, let's see. The cards I have are a 7 of diamonds and a 5 of diamonds. The current formula is "7". I need to use both numbers to make the formula equal 12. 

So far, I have "7". If I add 5 next, that would make it "7 + 5", which equals 12. That works! I should add the 5 to the formula. 
To output add "5" to the formula i need to output the "+" symbol first. The "+" symbol corresponds to action 11.
</think>
<action>11</action>

## Example 3
## Observation:
image: [the image shows two cards in hand: "Q of hearts" and "2 of clubs"; Formula: "10"]

<think>
The cards in hand are the Queen of hearts and the 2 of clubs. The Queen is treated as 10. The current formula is "10". 

Possible actions:
1. Add the number 10 to the formula.
2. Add the number 2 to the formula.
3. Add the addition operator (+).
4. Add the multiplication operator (*).
5. Add the equals operator (=).

Since the current formula is "10", adding the number 2 will give "10 + 2". 

10 + 2 = 12, which is the target value.

So, since the formula is just "10", the next action should be to add the operator "+" to the formula.
The + operator corresponds to action index 11.
</think>
<action>11</action>
--------------------------------

# Current Task
## History:

Based on the current visual information, which action should the agent take next?
</user>

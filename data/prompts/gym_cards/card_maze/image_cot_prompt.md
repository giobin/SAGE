<user>
{image}

You are an expert agent playing a simple game of cards.
You will be presented with {n_cards} different poker cards at the top of the image. At the bottom of the image, you will be shown a larger poker card which acts as your prompt.
You will be given a task, which determines which card you should select from the top cards, based on the card shown in the prompt.
If you make a mistake, the progress will reset to zero and you will have to start over. The game ends when you have made {n_decisions} correct actions in a row.

# Workflow Checklist
- Begin each turn with a concise checklist: (1) analyze the image to identify which card is in the prompt and the top cards, (2) find which card at the top matches the prompt according to the task, (3) identify the index of the matching card, (4) select the action which corresponds to the index of the matching card.

# Output Format
You should think about the answer step by step inside the <think> </think> tags, provide the chosen action index inside the <action> </action> tags. All text outside the <think> and <action> tags will be ignored.

--------------------------------
# Few-shot Examples
--------------------------------

## Example 1
## Task
Your task is to produce an action corresponding to the card at the top that matches the suit (hearts, diamonds, clubs, spades) of the prompt card, but not its number or face.

## Action space
The action space includes 4 discrete actions:
- 0: Choose the 1st card from the left
- 1: Choose the 2nd card from the left
- 2: Choose the 3rd card from the left
- 3: Choose the 4th card from the left

## Observation:
image: [the image shows 4 cards in hand (top of the image): "J of clubs", "2 of spades", "5 of diamonds" and "K of hearts"; The prompt card (image center): "3 of clubs"]

<think>
The top cards are the J of clubs, the 2 of spades, the 5 of diamonds and the K of hearts. The big central card shows a 3 of clubs. 

I have to choose from the top cards the one that matches the suit of the 3 of clubs. There is only one with the same suit, which is the J of clubs.
The J of clubs is the first card from the left, so i choose action 0.

</think>
<action>0</action>

## Example 2
## Task
Your task is to produce an action corresponding to the card at the top that matches the suit (hearts, diamonds, clubs, spades) of the prompt card, but not its number or face.

## Action space
The action space includes 4 discrete actions:
- 0: Choose the 1st card from the left
- 1: Choose the 2nd card from the left
- 2: Choose the 3rd card from the left
- 3: Choose the 4th card from the left

## Observation:
image: [the image shows 4 cards in hand (top of the image): "2 of diamonds", "3 of diamonds", "5 of hearts" and "J of hearts"; The prompt card (image center): "ace of diamonds"]

<think>
The top cards are the 2 of diamonds, 3 of diamonds, 5 of hearts and the J of hearts. The big central card shows an ace of diamonds. 

I have to choose from the top cards the one that matches the suit of the ace. the 2 and 3 of diamonds both are possible choices since they have the suit of diamonds.
I'll pick the 3 of diamonds as final choice, which corresponds to the 2nd card from the left. So the action is 1.

</think>
<action>1</action>
--------------------------------

## Current Task
{rule}

## Action space
{action_space}

Based on the current visual information, which action should the agent take next?
</user>
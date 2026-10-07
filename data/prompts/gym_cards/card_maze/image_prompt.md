<user>
{image}

You are an expert agent playing a simple game of cards.
You will be presented with {n_cards} different poker cards at the top of the image. At the bottom of the image, you will be shown a larger poker card which acts as your prompt.
You will be given a task, which determines which card you should select from the top cards, based on the card shown in the prompt.
If you make a mistake, the progress will reset to zero and you will have to start over. The game ends when you have made {n_decisions} correct actions in a row.

## Task
{rule}

## Action space
{action_space}

# Output Format
Respond with a single integer: the action index selected for this step, based strictly on the information shown in the provided image

Based on the current visual information, which action should the agent take next?
</user>
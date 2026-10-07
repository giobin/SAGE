<user>
{image}

You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library. You will receive observations from the environment and must decide which action to take next. Moreover you are asked to output a distribution of probabilities over the possible actions.
The goal is to navigate across a frozen grid to reach the goal without falling into holes.

## Observation space
You are presented with the image of the environment in the style of pixel art.
Each cell may be safe (white with snow), a hole (bright blue ice hole), the start (a stool) or the goal (a present box). The agent is represented by an elf character.

## Action space
The action space includes 4 discrete actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

# Output Format
You should think about the answer step by step inside the <think> </think> tags, then provide the chosen action index inside the <action> </action> tags and the probability distribution over all four actions (ordered as above) inside the <probs> </probs> tags. All text outside the <think>, <action> and <probs> tags will be ignored.

--------------------------------
# Few-shot Examples (2-shot)
--------------------------------

## Example 1
## Observation:
image: [the image shows the elf in row 4, column 7 (i.e. the coordinates are from 1 to 8). There is a hole left of him and one below.]

<think>
The agent is currently at the center right corner of the grid. The goal is below the agent. The agent needs to avoid the hole and then proceed downwards toward the goal which is located at the bottom right corner. The possible moves to avoid the holes are "move up" and "move right". Moving up is safe but would result in the agent getting further away from the gift. Given all these considerations probably a good option is going up and reevaluate the situation, and a even better option is to go right.
Move right is action 2 and move up is action 3. 
</think>
<action>2</action>
<probs>0.0, 0.0, 0.75, 0.25</probs>

## Example 2
## Observation:
image: [the image shows the elf in row 3, column 3. There is a hole directly to the right, safe tiles straight down toward the bottom row, and the goal is further down to the right.]

<think>
The elf is a couple of tiles above the center of the grid. The cell to the right is a hole, so moving right (action 2) is risky. Moving left would take the agent back toward the start and does not progress toward the goal. Moving up only keeps the elf wandering on the same plateau. The only safe and progress-making move is to step down along the clear column toward the goal. Therefore, action 1 (move down) is preferred, while move 0 and move 3 retain only small exploratory probability mass.
</think>
<action>1</action>
<probs>0.1, 0.8, 0.0, 0.1</probs>
--------------------------------

# Current Task
## History:

Based on the current visual information, which action should the agent take next, and what probability distribution over actions do you assign?

</user>

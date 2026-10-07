<user>
{image}
You are a smart agent acting in the "FrozenLake" environment from the Gymnasium library. You will receive observations from the environment and must decide which action to take next.
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
Respond with ONLY the action index inside <action></action> tags. Nothing else.

--------------------------------
# Few-shot Examples
--------------------------------

## Example 1
## Observation:
image: [the image shows the elf in row 4, column 7. There is a hole left of him and one below.]

<action>2</action>

## Example 2
## Observation:
image: [the image shows the elf in row 3, column 3. There is a hole directly to the right, safe tiles straight down.]

<action>1</action>

## Example 3
## Observation:
image: [the image shows the elf in row 6, column 5. The goal is one tile to the right and one down. There is a hole directly below.]

<action>2</action>
--------------------------------

</user>
from collections import deque
from minigrid.core.constants import DIR_TO_VEC

def get_fetch_oracle_action(env):
    """
    Analyzes the Minigrid FetchEnv state and returns the optimal action index.
    """
    # 1. Parse the mission to find target color and type
    # Mission format: "{syntax} {color} {type}" (e.g., "fetch a red ball")
    env = env.unwrapped
    mission_words = env.mission.split()
    target_color = mission_words[-2]
    target_type = mission_words[-1]

    # 2. Scan the grid for the target and obstacles
    # Access the unwrapped grid to iterate over objects
    grid = env.grid
    width, height = grid.width, grid.height
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    
    target_pos = None
    obstacles = set()
    
    for col in range(width):
        for row in range(height):
            obj = grid.get(col, row)
            if obj:
                if obj.type == target_type and obj.color == target_color:
                    target_pos = (col, row)
                    # The target itself is an obstacle we cannot walk INTO, 
                    # but we must walk UP TO.
                    obstacles.add((col, row))
                elif obj.type in ["wall", "ball", "key"]:
                    # All other objects are obstacles.
                    # Note: In FetchEnv, picking up the wrong object causes failure,
                    # so we treat them as hard walls.
                    obstacles.add((col, row))

    if target_pos is None:
        return "6" # Done/NoOp if target not found (should not happen)

    # 3. BFS Pathfinding
    # State: (x, y, direction)
    # Goal: Any state where the cell IN FRONT is the target_pos
    
    # Queue stores: (x, y, dir, first_action_to_take)
    # We only need the first action to return to the agent.
    queue = deque([(agent_pos[0], agent_pos[1], agent_dir, None)])
    visited = set([(agent_pos[0], agent_pos[1], agent_dir)])
    
    # Standard Minigrid Directions: 0=Right, 1=Down, 2=Left, 3=Up
    # Actions: 0=Left, 1=Right, 2=Forward, 3=Pickup
    
    while queue:
        x, y, d, first_action = queue.popleft()
        
        # Calculate the position in front of the agent
        dx, dy = DIR_TO_VEC[d]
        front_pos = (x + dx, y + dy)
        
        # Check Goal: Are we facing the target?
        if front_pos == target_pos:
            if first_action is None:
                # We are already there and facing it
                return "3" # Action: Pickup
            return str(first_action)

        transitions = []
        
        # Action 0: Turn Left
        transitions.append(((x, y, (d - 1) % 4), 0))
        
        # Action 1: Turn Right
        transitions.append(((x, y, (d + 1) % 4), 1))
        
        # Action 2: Move Forward
        # Only valid if the front cell is in bounds and NOT an obstacle
        if 0 <= front_pos[0] < width and 0 <= front_pos[1] < height:
            if front_pos not in obstacles:
                transitions.append(((front_pos[0], front_pos[1], d), 2))

        for next_state, action_idx in transitions:
            if next_state not in visited:
                visited.add(next_state)
                # If this is the first step, record the action. Otherwise keep the original first action.
                next_first_action = action_idx if first_action is None else first_action
                queue.append((next_state[0], next_state[1], next_state[2], next_first_action))

    return "6" # No path found (Done)


def get_lavagap_oracle_action(env):
    """
    Analyzes a MiniGrid LavaGap environment and returns the optimal action index.
    The agent must navigate around a lava wall (through the gap) to reach the green goal.
    """
    env = env.unwrapped
    grid = env.grid
    width, height = grid.width, grid.height
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir

    target_pos = None
    obstacles = set()

    for col in range(width):
        for row in range(height):
            obj = grid.get(col, row)
            if obj:
                if obj.type == "goal":
                    target_pos = (col, row)
                    # Goal is walkable (can_overlap), do NOT add to obstacles
                elif obj.type in ["wall", "lava"]:
                    obstacles.add((col, row))

    if target_pos is None:
        return "6"  # No goal found

    # BFS: state = (x, y, dir), goal = front_pos == target_pos (then move forward onto it)
    queue = deque([(agent_pos[0], agent_pos[1], agent_dir, None)])
    visited = {(agent_pos[0], agent_pos[1], agent_dir)}

    while queue:
        x, y, d, first_action = queue.popleft()
        dx, dy = DIR_TO_VEC[d]
        front_pos = (x + dx, y + dy)

        # Goal: agent is facing the goal cell -> move forward onto it
        if front_pos == target_pos:
            if first_action is None:
                return "2"  # Already facing goal, move forward
            return str(first_action)

        transitions = [
            ((x, y, (d - 1) % 4), 0),  # turn left
            ((x, y, (d + 1) % 4), 1),  # turn right
        ]
        # Move forward if the front cell is in bounds and not an obstacle
        if 0 <= front_pos[0] < width and 0 <= front_pos[1] < height:
            if front_pos not in obstacles:
                transitions.append(((front_pos[0], front_pos[1], d), 2))

        for next_state, action_idx in transitions:
            if next_state not in visited:
                visited.add(next_state)
                next_first_action = action_idx if first_action is None else first_action
                queue.append((next_state[0], next_state[1], next_state[2], next_first_action))

    return "6"  # No path found

def get_gotodoor_oracle_action(env):
    """
    Analyzes a MiniGrid GoToDoor environment and returns the optimal action index.
    The agent must navigate to face the target door and then use action 6 (done).
    """
    env = env.unwrapped
    # Mission format: "go to the {color} door"
    mission_words = env.mission.split()
    target_color = mission_words[-2]

    grid = env.grid
    width, height = grid.width, grid.height
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir

    target_pos = None
    obstacles = set()

    for col in range(width):
        for row in range(height):
            obj = grid.get(col, row)
            if obj:
                if obj.type == "door" and obj.color == target_color:
                    target_pos = (col, row)
                    obstacles.add((col, row))  # Can't walk through doors
                elif obj.type in ["wall", "door"]:
                    obstacles.add((col, row))

    if target_pos is None:
        return "6"  # Target door not found

    # BFS: state = (x, y, dir), goal = facing the target door
    queue = deque([(agent_pos[0], agent_pos[1], agent_dir, None)])
    visited = {(agent_pos[0], agent_pos[1], agent_dir)}

    while queue:
        x, y, d, first_action = queue.popleft()
        dx, dy = DIR_TO_VEC[d]
        front_pos = (x + dx, y + dy)

        # Goal: agent is facing the target door -> use done action
        if front_pos == target_pos:
            if first_action is None:
                return "6"  # Already facing target door, done
            return str(first_action)

        transitions = [
            ((x, y, (d - 1) % 4), 0),  # turn left
            ((x, y, (d + 1) % 4), 1),  # turn right
        ]
        if 0 <= front_pos[0] < width and 0 <= front_pos[1] < height:
            if front_pos not in obstacles:
                transitions.append(((front_pos[0], front_pos[1], d), 2))

        for next_state, action_idx in transitions:
            if next_state not in visited:
                visited.add(next_state)
                next_first_action = action_idx if first_action is None else first_action
                queue.append((next_state[0], next_state[1], next_state[2], next_first_action))

    return "6"  # No path found

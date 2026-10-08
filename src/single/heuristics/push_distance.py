from collections import deque
from typing import Dict, Tuple
from src.single.core.state import Board

def precompute_push_costs(board: Board) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
    """
    Returns a dict mapping goal_pos -> {box_pos: min_pushes}.
    Computes minimum pushes required to move a box from any valid position to the goal,
    ignoring other boxes but respecting walls and player reachability (pushing geometry).
    """
    costs = {}
    
    for goal in board.goals:
        goal_costs = {goal: 0}
        queue = deque([goal])
        
        while queue:
            curr_pos = queue.popleft()
            cx, cy = curr_pos
            
            # The box was pushed from (cx - dx, cy - dy) to (cx, cy)
            # The player must have been standing at (cx - 2*dx, cy - 2*dy)
            # We iterate over possible push directions that resulted in the box arriving here.
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                prev_box_x = cx - dx
                prev_box_y = cy - dy
                prev_box = (prev_box_x, prev_box_y)
                
                player_x = cx - 2*dx
                player_y = cy - 2*dy
                player_pos = (player_x, player_y)
                
                if (0 <= prev_box_x < board.width and 0 <= prev_box_y < board.height
                        and 0 <= player_x < board.width and 0 <= player_y < board.height
                        and prev_box not in board.walls and player_pos not in board.walls):
                    if prev_box not in goal_costs:
                        goal_costs[prev_box] = goal_costs[curr_pos] + 1
                        queue.append(prev_box)
                        
        costs[goal] = goal_costs
        
    return costs

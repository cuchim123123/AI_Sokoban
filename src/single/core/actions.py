from src.single.core.state import GameState, Board, Action
from typing import List, Tuple

def get_successors(state: GameState, board: Board) -> List[Tuple[Action, GameState]]:
    """Returns a list of (Action, GameState) for all valid moves."""
    successors = []
    
    for action in Action:
        dx, dy = action.value
        new_agent_x = state.agent[0] + dx
        new_agent_y = state.agent[1] + dy
        new_agent = (new_agent_x, new_agent_y)
        
        # Check if hitting a wall
        if new_agent in board.walls:
            continue
            
        # Check if hitting a box
        if new_agent in state.boxes:
            # We are pushing a box
            box_new_x = new_agent_x + dx
            box_new_y = new_agent_y + dy
            new_box = (box_new_x, box_new_y)
            
            # The space behind the box must not be a wall or another box
            if new_box in board.walls or new_box in state.boxes:
                continue
                
            # Create new state with updated box positions
            new_boxes = set(state.boxes)
            new_boxes.remove(new_agent)
            new_boxes.add(new_box)
            successors.append((action, GameState(new_agent, new_boxes)))
        else:
            # Simple move
            successors.append((action, GameState(new_agent, state.boxes)))
            
    return successors

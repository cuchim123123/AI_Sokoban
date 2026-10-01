from src.core.state import GameState, Board

def is_deadlock(state: GameState, board: Board) -> bool:
    """
    Checks for dynamic deadlocks, specifically 2x2 blocks of walls and boxes.
    Static deadlocks (corners, wall lines) are handled by the push_distance heuristic.
    """
    boxes = state.boxes
    walls = board.walls
    goals = board.goals
    
    # Check every box to see if it's part of a 2x2 deadlock
    for (x, y) in boxes:
        if (x, y) in goals:
            continue
            
        # Check all 4 possible 2x2 squares containing this box
        # top-left, top-right, bottom-left, bottom-right
        squares = [
            [(x, y), (x+1, y), (x, y+1), (x+1, y+1)],
            [(x-1, y), (x, y), (x-1, y+1), (x, y+1)],
            [(x, y-1), (x+1, y-1), (x, y), (x+1, y)],
            [(x-1, y-1), (x, y-1), (x-1, y), (x, y)]
        ]
        
        for square in squares:
            is_blocked = True
            for px, py in square:
                if (px, py) not in boxes and (px, py) not in walls:
                    is_blocked = False
                    break
            
            if is_blocked:
                return True
                
    return False

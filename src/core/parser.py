from src.core.state import GameState, Board
from typing import Tuple

def parse_map(file_path: str) -> Tuple[GameState, Board]:
    walls = set()
    goals = set()
    boxes = set()
    agent = None
    
    width = 0
    height = 0
    
    with open(file_path, 'r') as f:
        for y, line in enumerate(f):
            line = line.rstrip('\n')
            if not line:
                continue
            height = max(height, y + 1)
            width = max(width, len(line))
            
            for x, char in enumerate(line):
                if char == '%':
                    walls.add((x, y))
                elif char == 'A':
                    agent = (x, y)
                elif char == 'B':
                    boxes.add((x, y))
                elif char == 'D':
                    goals.add((x, y))
                elif char == 'C':
                    boxes.add((x, y))
                    goals.add((x, y))
                    
    if agent is None:
        raise ValueError("No agent found in map")
        
    board = Board(walls, goals, width, height)
    state = GameState(agent, boxes)
    return state, board

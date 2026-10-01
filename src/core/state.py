from enum import Enum
from typing import Set, Tuple

class Action(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    
    def __str__(self):
        return self.name.capitalize()

class Board:
    """Stores static information about the board to avoid copying it in every state."""
    def __init__(self, walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]], width: int, height: int):
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.width = width
        self.height = height

class GameState:
    """Represents a dynamic state in the Sokoban game."""
    def __init__(self, agent: Tuple[int, int], boxes: Set[Tuple[int, int]]):
        self.agent = agent
        self.boxes = frozenset(boxes)
        
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return self.agent == other.agent and self.boxes == other.boxes
        
    def __hash__(self):
        return hash((self.agent, self.boxes))
        
    def is_goal(self, board: Board) -> bool:
        """Check if all boxes are on goals."""
        return self.boxes == board.goals

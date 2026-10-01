from typing import Optional, List
from src.core.state import GameState, Action

class Node:
    def __init__(self, state: GameState, parent: Optional['Node'], action: Optional[Action], path_cost: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        
    def __lt__(self, other):
        return self.path_cost < other.path_cost

def reconstruct_path(node: Node) -> tuple[List[Action], int]:
    actions = []
    cost = node.path_cost
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    return actions, cost

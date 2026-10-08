import heapq
from typing import Optional, List, Tuple, Dict, Callable
import itertools
from src.single.core.state import GameState, Board, Action
from src.single.core.actions import get_successors
from src.single.search.node import Node, reconstruct_path
from src.single.heuristics.deadlock import is_deadlock
from src.single.search.algorithm import SearchAlgorithm

class AStarSearch(SearchAlgorithm):
    def __init__(self, heuristic: Callable[[GameState], float]):
        self.heuristic = heuristic
        
    def search(self, initial_state: GameState, board: Board) -> Tuple[Optional[List[Action]], int, int, int]:
        """
        Returns (actions, total_cost, generated_states, expanded_states).
        If no solution, actions is None.
        """
        start_node = Node(initial_state, None, None, 0)
        
        if initial_state.is_goal(board):
            return [], 0, 1, 0
            
        frontier = []
        counter = itertools.count()
        h_start = self.heuristic(initial_state)
        heapq.heappush(frontier, (start_node.path_cost + h_start, next(counter), start_node))
        
        explored: Dict[GameState, int] = {initial_state: 0}
        
        expanded_states = 0
        generated_states = 1
        
        while frontier:
            f, _, current_node = heapq.heappop(frontier)
            
            # If we found a cheaper path to this state already, skip it
            if explored.get(current_node.state, float('inf')) < current_node.path_cost:
                continue
                
            if current_node.state.is_goal(board):
                actions, cost = reconstruct_path(current_node)
                return actions, cost, generated_states, expanded_states
                
            expanded_states += 1
            
            for action, next_state in get_successors(current_node.state, board):
                if is_deadlock(next_state, board):
                    continue
                    
                new_cost = current_node.path_cost + 1
                generated_states += 1
                
                if new_cost < explored.get(next_state, float('inf')):
                    explored[next_state] = new_cost
                    child_node = Node(next_state, current_node, action, new_cost)
                    h_val = self.heuristic(next_state)
                    # Deadlock detection based on infinity heuristic
                    if h_val != float('inf'):
                        f_val = new_cost + h_val
                        heapq.heappush(frontier, (f_val, next(counter), child_node))
                    
        return None, 0, generated_states, expanded_states


def astar_search(
    initial_state: GameState,
    board: Board,
    heuristic: Callable[[GameState], float],
) -> Tuple[Optional[List[Action]], int, int, int]:
    return AStarSearch(heuristic).search(initial_state, board)

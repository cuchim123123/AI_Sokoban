from typing import Dict, Tuple, List
from scipy.optimize import linear_sum_assignment
from src.core.state import GameState, Board

class MatchingHeuristic:
    def __init__(self, board: Board, push_costs: Dict[Tuple[int, int], Dict[Tuple[int, int], int]]):
        self.board = board
        self.push_costs = push_costs
        self.goals = list(board.goals)
        
    def __call__(self, state: GameState) -> float:
        boxes = list(state.boxes)
        n = len(boxes)
        if n == 0:
            return 0
            
        cost_matrix = []
        for box in boxes:
            row = []
            for goal in self.goals:
                try:
                    cost = self.push_costs[goal].get(box, float('inf'))
                except TypeError:
                    print(f"CRASH DEBUG:")
                    print(f"goal: {goal}, type: {type(goal)}")
                    print(f"self.push_costs: type {type(self.push_costs)}")
                    val = self.push_costs.get(goal)
                    print(f"self.push_costs[goal]: {val}, type {type(val)}")
                    raise
                row.append(cost)
            cost_matrix.append(row)
            
        # Using scipy's linear_sum_assignment
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))
            return total_cost
        except ValueError:
            # Can happen if matrix is empty, but we handled n=0
            return float('inf')

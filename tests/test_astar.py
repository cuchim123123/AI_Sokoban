import unittest
from src.core.parser import parse_map
from src.search.astar import astar_search
from src.heuristics.push_distance import precompute_push_costs
from src.heuristics.matching import MatchingHeuristic
from src.core.state import Action

class TestAStar(unittest.TestCase):
    def test_astar_solvable(self):
        state, board = parse_map("maps/test_solvable.txt")
        push_costs = precompute_push_costs(board)
        heuristic = MatchingHeuristic(board, push_costs)
        
        actions, cost, generated, expanded = astar_search(state, board, heuristic)
        
        self.assertIsNotNone(actions)
        self.assertEqual(len(actions), 2)
        self.assertEqual(cost, 2)
        self.assertEqual(actions, [Action.EAST, Action.SOUTH])
        
    def test_heuristic_value(self):
        state, board = parse_map("maps/test_solvable.txt")
        push_costs = precompute_push_costs(board)
        heuristic = MatchingHeuristic(board, push_costs)
        
        h_val = heuristic(state)
        # The box is at (2,2), goal at (2,3).
        # We need 1 push South. So minimum pushes is 1.
        self.assertEqual(h_val, 1)

if __name__ == '__main__':
    unittest.main()

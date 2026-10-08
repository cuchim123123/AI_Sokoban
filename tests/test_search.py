import unittest
from src.single.core.parser import parse_map
from src.single.search.ucs import uniform_cost_search
from src.single.core.state import Action

class TestSearch(unittest.TestCase):
    def test_ucs_solvable(self):
        state, board = parse_map("maps/test_solvable.txt")
        actions, cost, generated, expanded = uniform_cost_search(state, board)
        
        self.assertIsNotNone(actions)
        self.assertEqual(len(actions), 2)
        self.assertEqual(cost, 2)
        # Sequence should be East, South, South (wait, A is at (1,1), B at (2,2), D at (2,3)).
        # Let's check coordinates.
        # %%%%% (0 to 4)
        # %A  % A is at (1,1)
        # % B % B is at (2,2)
        # % D % D is at (2,3)
        # %%%%%
        # To push B to D, agent needs to be at (2,1) and push South.
        # From (1,1) -> East to (2,1)
        # From (2,1) -> South to (2,2) (pushes B to (2,3))
        # Wait, (2,3) is D.
        self.assertEqual(actions, [Action.EAST, Action.SOUTH])
        
if __name__ == '__main__':
    unittest.main()

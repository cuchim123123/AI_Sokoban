import unittest
from src.single.core.parser import parse_map
from src.single.core.actions import get_successors, Action

class TestCore(unittest.TestCase):
    def test_parser(self):
        state, board = parse_map("maps/test_map.txt")
        self.assertEqual(board.width, 5)
        self.assertEqual(board.height, 4)
        self.assertEqual(state.agent, (1, 1))
        self.assertIn((3, 1), state.boxes)
        self.assertIn((2, 2), board.goals)
        self.assertFalse(state.is_goal(board))

    def test_actions(self):
        state, board = parse_map("maps/test_map.txt")
        successors = get_successors(state, board)
        
        # Valid moves from (1, 1) in test_map.txt:
        # NORTH: Wall at (1, 0)
        # SOUTH: Empty at (1, 2)
        # EAST: Empty at (2, 1)
        # WEST: Wall at (0, 1)
        
        actions = [a for a, s in successors]
        self.assertIn(Action.SOUTH, actions)
        self.assertIn(Action.EAST, actions)
        self.assertNotIn(Action.NORTH, actions)
        self.assertNotIn(Action.WEST, actions)

if __name__ == '__main__':
    unittest.main()

import unittest

from src.competitive.agent_a import _cache_key
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome


class TestCompetitiveRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        walls = frozenset(
            (x, y)
            for x in range(7)
            for y in range(7)
            if x in (0, 6) or y in (0, 6)
        )
        cls.board = Board(walls, frozenset({(3, 4)}), 7, 7)

    def state(self, agent_a, agent_b, boxes=frozenset()):
        return CompetitiveState(
            agent_a,
            agent_b,
            boxes,
            frozenset(),
            frozenset(),
            0,
        )

    def test_same_destination_uses_remaining_step_priority(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3)),
            Action.EAST,
            Action.WEST,
            self.board,
            11,
        )
        self.assertEqual(out.state.agent_a, (3, 3))
        self.assertEqual(out.state.agent_b, (4, 3))
        self.assertTrue(out.conflict)

        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3)),
            Action.EAST,
            Action.WEST,
            self.board,
            10,
        )
        self.assertEqual(out.state.agent_a, (2, 3))
        self.assertEqual(out.state.agent_b, (3, 3))
        self.assertTrue(out.conflict)

    def test_competing_pushes_fail_for_both(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3), frozenset({(3, 3)})),
            Action.EAST,
            Action.WEST,
            self.board,
            10,
        )
        self.assertEqual(out.state.boxes, frozenset({(3, 3)}))
        self.assertTrue(out.conflict)

    def test_wait_is_a_default_action(self):
        actions = get_valid_actions(
            (2, 2), (4, 2), frozenset(), self.board
        )
        self.assertIn(Action.WAIT, actions)

    def test_entering_occupied_cell_is_blocked(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)),
            Action.EAST,
            Action.WAIT,
            self.board,
            10,
        )
        self.assertEqual(out.state.agent_a, (2, 3))
        self.assertTrue(out.conflict)

    def test_cache_key_includes_step_and_horizon(self):
        state = self.state((2, 2), (4, 2))
        other = CompetitiveState(
            (2, 2), (4, 2), frozenset(), frozenset(), frozenset(), 1
        )
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(other, "A", 10))
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(state, "A", 20))
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(state, "B", 10))


if __name__ == "__main__":
    unittest.main()
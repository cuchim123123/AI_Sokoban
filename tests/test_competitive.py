import unittest
from collections import deque

from src.competitive.agent_a import _cache_key, _robust_successor, best_action
from src.competitive.evaluation import (
    _finish_return_score,
    _joint_interact_scores,
    competitive_heuristic,
)
from src.competitive.parser import parse_competitive_map
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
        self.assertNotEqual(out.state.agent_b, (3, 3))
        self.assertNotEqual(out.state.agent_b, (4, 3))
        self.assertTrue(out.conflict)

        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3)),
            Action.EAST,
            Action.WEST,
            self.board,
            10,
        )
        self.assertNotEqual(out.state.agent_a, (3, 3))
        self.assertNotEqual(out.state.agent_a, (2, 3))
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

    def test_wait_is_explicit_dead_end_fallback(self):
        actions = get_valid_actions(
            (2, 2), (4, 2), frozenset(), self.board
        )
        self.assertNotIn(Action.WAIT, actions)
        self.assertIn(
            Action.WAIT,
            get_valid_actions(
                (2, 2), (4, 2), frozenset(), self.board, include_wait=True
            ),
        )

    def test_entering_occupied_cell_is_blocked(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)),
            Action.EAST,
            Action.NORTH,
            self.board,
            10,
        )
        self.assertEqual(out.state.agent_a, (2, 3))
        self.assertEqual(out.state.agent_b, (3, 2))
        self.assertTrue(out.conflict)

    def test_cache_key_includes_step_and_horizon(self):
        state = self.state((2, 2), (4, 2))
        other = CompetitiveState(
            (2, 2), (4, 2), frozenset(), frozenset(), frozenset(), 1
        )
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(other, "A", 10))
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(state, "A", 20))
        self.assertNotEqual(_cache_key(state, "A", 10), _cache_key(state, "B", 10))

    def test_blocked_push_is_not_repeated(self):
        state = CompetitiveState(
            (3, 5),
            (5, 5),
            frozenset({(4, 5), (8, 5)}),
            frozenset({(4, 5)}),
            frozenset({(8, 5)}),
            8,
        )
        value, _, _ = _robust_successor(
            state,
            Action.WEST,
            self.board,
            "B",
            50,
            {},
        )
        wait_value, _, _ = _robust_successor(
            state,
            Action.WAIT,
            self.board,
            "B",
            50,
            {},
        )
        self.assertLess(value, wait_value)

    def test_recent_position_revisit_is_not_selected_when_wait_is_safe(self):
        state = CompetitiveState(
            (4, 2),
            (14, 2),
            frozenset({(5, 3), (13, 3), (4, 7)}),
            frozenset(),
            frozenset(),
            4,
        )
        action = best_action(
            state,
            self.board,
            40,
            "A",
            deque([(4, 1), (4, 3)], maxlen=4),
            {},
            {},
            time_limit=0.01,
        )
        self.assertNotIn(action, (Action.NORTH, Action.SOUTH))

    def test_reachable_opponent_goal_has_steal_value(self):
        board = Board(
            frozenset(
                (x, y)
                for x in range(7)
                for y in range(7)
                if x in (0, 6) or y in (0, 6)
            ),
            frozenset({(3, 3)}),
            7,
            7,
        )
        state = CompetitiveState(
            (3, 2),
            (5, 3),
            frozenset({(3, 3)}),
            frozenset(),
            frozenset({(3, 3)}),
            0,
        )
        steal, _ = _joint_interact_scores(
            state.agent_a,
            state.agent_b,
            state.boxes_on_goals_b,
            board,
            20,
        )
        self.assertGreater(steal, 1000.0)
        self.assertGreater(competitive_heuristic(state, board, "A", 20), -1000.0)

        losing_state = CompetitiveState(
            (5, 3),
            (3, 2),
            frozenset({(3, 3)}),
            frozenset(),
            frozenset({(3, 3)}),
            0,
        )
        losing_attack, _ = _joint_interact_scores(
            losing_state.agent_a,
            losing_state.agent_b,
            losing_state.boxes_on_goals_b,
            board,
            20,
        )
        self.assertGreater(losing_attack, 0.0)
        self.assertLess(losing_attack, 1000.0)

    def test_agent_does_not_push_its_own_finished_box(self):
        state = CompetitiveState(
            (3, 3),
            (5, 5),
            frozenset({(3, 4)}),
            frozenset({(3, 4)}),
            frozenset(),
            0,
        )
        value, _, _ = _robust_successor(
            state,
            Action.SOUTH,
            self.board,
            "A",
            20,
            {},
        )
        self.assertEqual(value, float("-inf"))

    def test_search_never_selects_own_finished_box_push(self):
        _, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        state = CompetitiveState(
            (7, 8),
            (7, 6),
            frozenset({(5, 5), (8, 8), (13, 5)}),
            frozenset({(5, 5), (8, 8)}),
            frozenset({(13, 5)}),
            16,
        )
        action = best_action(
            state,
            board,
            40,
            "A",
            deque(maxlen=4),
            {},
            {},
            time_limit=0.15,
        )
        self.assertNotEqual(action, Action.EAST)

    def test_tied_agent_captures_adjacent_opponent_goal(self):
        _, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        state = CompetitiveState(
            (7, 6),
            (6, 5),
            frozenset({(8, 8), (5, 5), (13, 5)}),
            frozenset({(8, 8), (5, 5)}),
            frozenset({(13, 5)}),
            18,
        )
        action = best_action(
            state,
            board,
            40,
            "B",
            deque(maxlen=4),
            {},
            {},
            time_limit=0.15,
        )
        self.assertEqual(action, Action.WEST)

    def test_tied_agent_prefers_approaching_opponent_goal(self):
        _, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        state = CompetitiveState(
            (5, 4),
            (13, 4),
            frozenset({(5, 5), (13, 5), (4, 7)}),
            frozenset({(5, 5)}),
            frozenset({(13, 5)}),
            8,
        )
        cache = {}
        west_value = _robust_successor(
            state,
            Action.WEST,
            board,
            "B",
            40,
            cache,
        )[0]
        wait_value = _robust_successor(
            state,
            Action.WAIT,
            board,
            "B",
            40,
            cache,
        )[0]
        self.assertGreater(west_value, wait_value)

    def test_finish_return_mission_has_positive_feasible_value(self):
        _, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        score = _finish_return_score(
            (6, 5),
            frozenset({(5, 5), (13, 5), (4, 8)}),
            frozenset({(5, 5)}),
            board,
            24,
        )
        self.assertGreater(score, 0.0)

    def test_root_action_repositions_behind_remaining_box(self):
        _, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        state = CompetitiveState(
            (3, 7),
            (10, 4),
            frozenset({(4, 8), (5, 5), (13, 5)}),
            frozenset({(5, 5)}),
            frozenset({(13, 5)}),
            11,
        )
        action = best_action(
            state,
            board,
            40,
            "A",
            deque(maxlen=4),
            {},
            {},
            time_limit=0.15,
        )
        self.assertEqual(action, Action.SOUTH)


if __name__ == "__main__":
    unittest.main()
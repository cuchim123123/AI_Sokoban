"""
Behavior tests for the rewritten competitive AI.

These tests pin down the observable contract instead of private helpers:

* the odd/even conflict rule and the diversion rules (transition.py),
* zero-sum evaluation with competitive race assignment (evaluation.py),
* the GBFS agent's hard tactical constraints, time budget and telemetry
  (agent_a.py / agent_b.py),
* a full-game invariant sweep through the real joint transition.

Deterministic: every random sweep is seeded, every search runs with an
explicit (small) time limit.
"""

import random
import time
import unittest
from collections import deque

from src.competitive.agent_a import (
    AgentA,
    LAST_SEARCH,
    best_action,
    history_entry,
)
from src.competitive.agent_b import AgentB
from src.competitive.evaluation import (
    W_LOCKED,
    competitive_heuristic,
    creates_deadlock,
    deadlock_count,
    evaluate,
    projected_score,
)
from src.competitive.parser import parse_competitive_map
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (
    clear_cache,
    conflict_winner,
    get_valid_actions,
    resolve_joint_action_outcome,
)


def square_board(size=7, extra_walls=frozenset(), goals=((3, 4),)):
    walls = frozenset(
        (x, y)
        for x in range(size)
        for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    ) | frozenset(extra_walls)
    return Board(walls, frozenset(goals), size, size)


def swap_labels(state):
    """Reflect a state across the A/B axis (entities and their credits)."""
    return CompetitiveState(
        state.agent_b,
        state.agent_a,
        state.boxes,
        state.boxes_on_goals_b,
        state.boxes_on_goals_a,
        state.step,
    )


def check_state_invariants(test, prev, nxt, board, max_steps):
    """Structural rules that every joint transition must preserve."""
    test.assertEqual(nxt.step, prev.step + 1)
    test.assertLessEqual(nxt.step, max_steps)
    test.assertNotEqual(nxt.agent_a, nxt.agent_b)
    for pos in (nxt.agent_a, nxt.agent_b):
        test.assertNotIn(pos, board.walls)
        test.assertNotIn(pos, nxt.boxes)
    test.assertEqual(len(nxt.boxes), len(prev.boxes))
    for box in nxt.boxes:
        test.assertNotIn(box, board.walls)
    creds = nxt.boxes_on_goals_a | nxt.boxes_on_goals_b
    test.assertTrue(creds <= nxt.boxes)
    test.assertTrue(creds <= board.goals)
    test.assertFalse(nxt.boxes_on_goals_a & nxt.boxes_on_goals_b)


# ---------------------------------------------------------------------------
# Conflict rules: parity priority + diversion
# ---------------------------------------------------------------------------

class TestConflictRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = square_board()

    def state(self, agent_a, agent_b, boxes=frozenset()):
        return CompetitiveState(
            agent_a, agent_b, boxes, frozenset(), frozenset(), 0
        )

    # ── parity ──────────────────────────────────────────────────────────

    def test_conflict_winner_follows_remaining_parity(self):
        # remaining = max_steps - step
        self.assertEqual(conflict_winner(50, 0), "B")   # 50 even
        self.assertEqual(conflict_winner(50, 1), "A")   # 49 odd
        self.assertEqual(conflict_winner(11, 0), "A")
        self.assertEqual(conflict_winner(10, 0), "B")
        self.assertEqual(conflict_winner(1, 0), "A")
        self.assertEqual(conflict_winner(1, 1), "B")

    def test_odd_remaining_gives_agent_a_priority(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3)), Action.EAST, Action.WEST,
            self.board, 11,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_a, (3, 3))     # winner takes the cell
        self.assertNotEqual(out.state.agent_b, (3, 3))  # loser diverted
        self.assertNotEqual(out.state.agent_b, (4, 3))  # and did not stand still

    def test_even_remaining_gives_agent_b_priority(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3)), Action.EAST, Action.WEST,
            self.board, 10,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_b, (3, 3))
        self.assertNotEqual(out.state.agent_a, (3, 3))
        self.assertNotEqual(out.state.agent_a, (2, 3))  # loser moved

    # ── swap (7.2): winner swaps in, loser may not take the vacated cell ─

    def test_swap_odd_winner_takes_cell_loser_cannot_take_vacated_cell(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)), Action.EAST, Action.WEST,
            self.board, 11,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_a, (3, 3))     # A completed the swap
        self.assertNotEqual(out.state.agent_b, (3, 3))  # B did not keep the cell
        self.assertNotEqual(out.state.agent_b, (2, 3))  # B may not enter A's cell
        self.assertNotEqual(out.state.agent_b, (3, 3))

    def test_swap_even_winner_is_agent_b(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)), Action.EAST, Action.WEST,
            self.board, 10,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_b, (2, 3))
        self.assertNotEqual(out.state.agent_a, (3, 3))  # not B's vacated cell
        self.assertNotEqual(out.state.agent_a, (2, 3))

    # ── competing pushes / push destinations ────────────────────────────

    def test_competing_pushes_use_priority_and_yield(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 3), frozenset({(3, 3)})),
            Action.EAST, Action.WEST, self.board, 10,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.boxes, frozenset({(2, 3)}))
        self.assertEqual(out.state.agent_b, (3, 3))     # winner pushed
        self.assertNotEqual(out.state.agent_a, (2, 3))  # loser moved away

    def test_push_destination_conflict_uses_priority_and_yield(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (4, 4), frozenset({(3, 3)})),
            Action.EAST, Action.NORTH, self.board, 11,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.boxes, frozenset({(4, 3)}))
        self.assertEqual(out.state.agent_a, (3, 3))
        self.assertNotEqual(out.state.agent_b, (4, 3))

    # ── occupancy exceptions ────────────────────────────────────────────

    def test_entering_cell_of_moving_opponent_is_not_a_conflict(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)), Action.EAST, Action.NORTH,
            self.board, 10,
        )
        self.assertFalse(out.conflict)
        self.assertEqual(out.state.agent_a, (3, 3))
        self.assertEqual(out.state.agent_b, (3, 2))

    def test_entrant_diverts_when_occupant_cannot_move(self):
        # B wants to move north but its box is pinned against the wall.
        out = resolve_joint_action_outcome(
            self.state((2, 2), (3, 2), frozenset({(3, 1)})),
            Action.EAST, Action.NORTH, self.board, 10,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_b, (3, 2))     # occupant keeps cell
        self.assertNotEqual(out.state.agent_a, (3, 2))  # entrant did not enter
        self.assertNotEqual(out.state.agent_a, (2, 2))  # and did not stand still

    def test_waiting_occupant_keeps_cell_entrant_diverts(self):
        out = resolve_joint_action_outcome(
            self.state((2, 3), (3, 3)), Action.EAST, Action.WAIT,
            self.board, 10,
        )
        self.assertTrue(out.conflict)
        self.assertEqual(out.state.agent_b, (3, 3))     # waiter keeps its cell
        self.assertNotEqual(out.state.agent_a, (3, 3))  # entrant diverted
        self.assertNotEqual(out.state.agent_a, (2, 3))  # entrant moved

    # ── valid actions ───────────────────────────────────────────────────

    def test_wait_is_excluded_unless_explicitly_requested(self):
        actions = get_valid_actions((2, 2), (4, 2), frozenset(), self.board)
        self.assertNotIn(Action.WAIT, actions)
        self.assertTrue(actions)
        self.assertIn(
            Action.WAIT,
            get_valid_actions(
                (2, 2), (4, 2), frozenset(), self.board, include_wait=True
            ),
        )

    def test_blocked_push_is_not_a_valid_action(self):
        # Pushing the box into a wall is impossible.
        actions = get_valid_actions(
            (2, 2), (4, 2), frozenset({(3, 2)}), self.board
        )
        # EAST targets the box at (3, 2); its push destination (4, 2) is free,
        # so EAST stays legal here.
        self.assertIn(Action.EAST, actions)
        wall_board = square_board(extra_walls=((4, 2),))
        actions = get_valid_actions(
            (2, 2), (5, 5), frozenset({(3, 2)}), wall_board
        )
        self.assertNotIn(Action.EAST, actions)

    # ── invariant sweep over real dynamics ──────────────────────────────

    def test_random_joint_actions_preserve_state_invariants(self):
        rng = random.Random(1234)
        _, board = parse_competitive_map("maps/competitive/arena_open.txt")
        start = CompetitiveState(
            (6, 3), (7, 3), frozenset({(4, 4), (8, 4)}),
            frozenset(), frozenset(), 0,
        )
        max_steps = 10_000  # no early exit: play every sampled step
        clear_cache()
        st = start
        conflicts = 0
        for _ in range(400):
            opts_a = get_valid_actions(st.agent_a, st.agent_b, st.boxes, board)
            opts_b = get_valid_actions(st.agent_b, st.agent_a, st.boxes, board)
            action_a = rng.choice(opts_a) if opts_a else Action.WAIT
            action_b = rng.choice(opts_b) if opts_b else Action.WAIT
            out = resolve_joint_action_outcome(
                st, action_a, action_b, board, max_steps
            )
            check_state_invariants(self, st, out.state, board, max_steps)
            if out.conflict:
                conflicts += 1
            st = out.state
        self.assertGreater(conflicts, 0)  # the sweep must exercise conflicts


# ---------------------------------------------------------------------------
# Evaluation: zero-sum, race assignment, steal, deadlock
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = square_board()

    def random_state(self, rng, board, step=0):
        free = [
            (x, y)
            for x in range(board.width)
            for y in range(board.height)
            if (x, y) not in board.walls
        ]
        pa, pb, box = rng.sample(free, 3)
        cred_a, cred_b = frozenset(), frozenset()
        if box in board.goals:
            roll = rng.random()
            if roll < 0.34:
                cred_a = frozenset({box})
            elif roll < 0.67:
                cred_b = frozenset({box})
        return CompetitiveState(pa, pb, frozenset({box}), cred_a, cred_b, step)

    def test_evaluate_is_exactly_zero_sum(self):
        rng = random.Random(7)
        _, board = parse_competitive_map("maps/competitive/arena_open.txt")
        for _ in range(60):
            state = self.random_state(rng, board)
            self.assertAlmostEqual(
                evaluate(state, board, "A", 50),
                -evaluate(state, board, "B", 50),
                places=9,
            )

    def test_evaluate_is_label_symmetric(self):
        rng = random.Random(11)
        _, board = parse_competitive_map("maps/competitive/dense_goals.txt")
        for _ in range(60):
            state = self.random_state(rng, board)
            self.assertAlmostEqual(
                evaluate(state, board, "A", 50),
                evaluate(swap_labels(state), board, "B", 50),
                places=9,
            )

    def test_credited_point_worth_more_than_unclaimed_one(self):
        # A defends the credited box; B is two steps from an approach cell.
        credited = CompetitiveState(
            (3, 3), (5, 5), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        unclaimed = CompetitiveState(
            (3, 3), (5, 5), frozenset({(3, 4)}),
            frozenset(), frozenset(), 0,
        )
        self.assertGreater(
            evaluate(credited, self.board, "A", 30),
            evaluate(unclaimed, self.board, "A", 30),
        )
        self.assertGreater(evaluate(credited, self.board, "A", 30), W_LOCKED * 0.5)

    def test_race_goes_to_the_closer_agent(self):
        # Box one push away from the goal for A, far from B.
        near_a = CompetitiveState(
            (3, 2), (5, 5), frozenset({(3, 3)}),
            frozenset(), frozenset(), 0,
        )
        self.assertGreater(evaluate(near_a, self.board, "A", 30), 0.0)
        near_b = swap_labels(near_a)
        self.assertLess(evaluate(near_b, self.board, "A", 30), 0.0)

    def test_steal_value_grows_as_attacker_approaches(self):
        far = CompetitiveState(
            (1, 1), (5, 5), frozenset({(3, 4)}),
            frozenset(), frozenset({(3, 4)}), 0,
        )
        near = CompetitiveState(
            (3, 3), (5, 5), frozenset({(3, 4)}),
            frozenset(), frozenset({(3, 4)}), 0,
        )
        self.assertGreater(
            evaluate(near, self.board, "A", 30),
            evaluate(far, self.board, "A", 30),
        )

    def test_out_of_steps_freezes_the_score(self):
        state = CompetitiveState(
            (5, 1), (5, 5), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 30,
        )
        self.assertAlmostEqual(
            evaluate(state, self.board, "A", 30), W_LOCKED, places=9
        )
        self.assertAlmostEqual(
            evaluate(state, self.board, "B", 30), -W_LOCKED, places=9
        )

    def test_legacy_helpers_still_exported(self):
        state = CompetitiveState(
            (3, 2), (5, 5), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        self.assertGreater(projected_score(state, self.board, "A", 30), 0.0)
        self.assertGreater(
            competitive_heuristic(state, self.board, "A", 30),
            competitive_heuristic(state, self.board, "B", 30),
        )

    def test_creates_deadlock_detects_corner_but_not_goal(self):
        corner_board = square_board(goals=((5, 5),))
        # Box pushed into the (1, 1) corner: it can never move again.
        self.assertTrue(
            creates_deadlock((1, 3), Action.NORTH, frozenset({(1, 2)}),
                             corner_board)
        )
        goal_board = square_board(goals=((1, 1),))
        # The same push is fine when the corner is a goal.
        self.assertFalse(
            creates_deadlock((1, 3), Action.NORTH, frozenset({(1, 2)}),
                             goal_board)
        )
        floor_board = square_board(goals=((5, 5),))
        # Box pushed into open space stays mobile (pushable from all sides).
        self.assertFalse(
            creates_deadlock((3, 4), Action.NORTH, frozenset({(3, 3)}),
                             floor_board)
        )

    def test_deadlock_count_counts_pinned_corner_boxes(self):
        corner_board = square_board(goals=((5, 5),))
        # The box on the secured goal is excluded via occupied_goals.
        self.assertEqual(
            deadlock_count(
                frozenset({(1, 1), (5, 5)}),
                corner_board,
                frozenset({(5, 5)}),
            ),
            1,
        )


# ---------------------------------------------------------------------------
# Agent behavior: hard constraints, budget, telemetry, compatibility
# ---------------------------------------------------------------------------

class TestAgentBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = square_board()
        cls.arena, cls.arena_board = parse_competitive_map(
            "maps/competitive/arena_open.txt"
        )

    def test_agent_b_shares_agent_a_engine(self):
        self.assertTrue(issubclass(AgentB, AgentA))
        self.assertEqual(AgentA.perspective, "A")
        self.assertEqual(AgentB.perspective, "B")

    def test_legacy_positional_signature(self):
        # debug2.py calls best_action positionally through time_limit.
        action = best_action(
            self.arena, self.arena_board, 50, "A",
            deque(maxlen=6), {}, {}, 0.1,
        )
        self.assertIsInstance(action, Action)

    def test_heuristic_cache_keyword_maps_to_eval_cache(self):
        cache = {}
        action = best_action(
            self.arena, self.arena_board, 50, "A",
            deque(maxlen=6), {}, time_limit=0.1,
            heuristic_cache=cache,
        )
        self.assertIsInstance(action, Action)

    def test_time_budget_is_respected(self):
        started = time.perf_counter()
        best_action(self.arena, self.arena_board, 50, "A",
                    deque(maxlen=6), {}, {}, 0.3)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.3 + 0.45)  # margin for scheduling noise

    def test_telemetry_is_populated(self):
        best_action(self.arena, self.arena_board, 50, "B",
                    deque(maxlen=6), {}, {}, 0.1)
        self.assertEqual(LAST_SEARCH["engine"], "GBFS")
        self.assertEqual(LAST_SEARCH["perspective"], "B")
        self.assertGreaterEqual(LAST_SEARCH["depth"], 1)
        self.assertGreater(LAST_SEARCH["nodes"], 0)
        self.assertGreater(LAST_SEARCH["time"], 0.0)

    def test_agent_never_waits_while_a_move_exists(self):
        for path in (
            "maps/competitive/arena_open.txt",
            "maps/competitive/capacity_lab.txt",
            "maps/competitive/dense_goals.txt",
            "maps/competitive/corridors.txt",
            "maps/competitive/test_race.txt",
        ):
            state, board = parse_competitive_map(path)
            action = best_action(state, board, 50, "A",
                                 deque(maxlen=6), {}, {}, 0.1)
            self.assertNotEqual(
                action, Action.WAIT, msg="WAIT chosen on " + path
            )

    def test_wait_only_when_fully_boxed_in(self):
        board = square_board(extra_walls=((3, 1), (1, 3)), goals=((5, 5),))
        boxes = frozenset({(2, 1), (1, 2)})
        self.assertEqual(
            get_valid_actions((1, 1), (4, 4), boxes, board), []
        )
        state = CompetitiveState(
            (1, 1), (4, 4), boxes, frozenset(), frozenset(), 0
        )
        action = best_action(state, board, 20, "A",
                             deque(maxlen=6), {}, {}, 0.1)
        self.assertEqual(action, Action.WAIT)

    def test_agent_never_pushes_own_credited_box_off_its_goal(self):
        state = CompetitiveState(
            (3, 5), (1, 1), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        action = best_action(state, self.board, 30, "A",
                             deque(maxlen=6), {}, {}, 0.15)
        self.assertNotEqual(action, Action.NORTH)

    def test_agent_never_pushes_box_into_static_deadlock(self):
        state = CompetitiveState(
            (1, 3), (5, 5), frozenset({(1, 2)}),
            frozenset(), frozenset(), 0,
        )
        self.assertTrue(
            creates_deadlock((1, 3), Action.NORTH, state.boxes, self.board)
        )
        action = best_action(state, self.board, 30, "A",
                             deque(maxlen=6), {}, {}, 0.15)
        self.assertNotEqual(action, Action.NORTH)

    def test_agent_takes_the_immediate_scoring_push(self):
        state = CompetitiveState(
            (3, 2), (5, 5), frozenset({(3, 3)}),
            frozenset(), frozenset(), 0,
        )
        action = best_action(state, self.board, 20, "A",
                             deque(maxlen=6), {}, {}, 0.15)
        self.assertEqual(action, Action.SOUTH)

    def test_banned_actions_are_honoured(self):
        first = best_action(self.arena, self.arena_board, 50, "A",
                            deque(maxlen=6), {}, {}, 0.15)
        second = best_action(self.arena, self.arena_board, 50, "A",
                             deque(maxlen=6), {}, {}, 0.15,
                             banned_actions={first})
        self.assertNotEqual(second, first)

    def test_choose_action_records_history_and_last_action(self):
        agent = AgentA(time_limit=0.1)
        action = agent.choose_action(self.arena, self.arena_board, 50)
        self.assertIsInstance(action, Action)
        self.assertEqual(agent._last_action, action)
        self.assertEqual(len(agent._history), 1)
        self.assertEqual(agent._history[0][0], self.arena.agent_a)

        b = AgentB(time_limit=0.1)
        b_action = b.choose_action(self.arena, self.arena_board, 50)
        self.assertIsInstance(b_action, Action)
        self.assertEqual(b._history[0][0], self.arena.agent_b)

    def test_history_entry_tracks_position_and_board(self):
        entry = history_entry(self.arena, "A")
        self.assertEqual(entry[0], self.arena.agent_a)
        self.assertEqual(entry[1], self.arena.boxes)
        self.assertEqual(history_entry(self.arena, "B")[0], self.arena.agent_b)

    def test_full_game_keeps_all_invariants(self):
        st, board = parse_competitive_map("maps/competitive/arena_open.txt")
        agent_a = AgentA(time_limit=0.1)
        agent_b = AgentB(time_limit=0.1)
        max_steps = 30
        clear_cache()
        moves = 0
        conflicts = 0
        while not st.is_terminal(max_steps):
            action_a = agent_a.choose_action(st, board, max_steps)
            action_b = agent_b.choose_action(st, board, max_steps)
            valid_a = get_valid_actions(st.agent_a, st.agent_b, st.boxes, board)
            if valid_a:
                self.assertIn(action_a, valid_a)
            else:
                self.assertEqual(action_a, Action.WAIT)
            valid_b = get_valid_actions(st.agent_b, st.agent_a, st.boxes, board)
            if valid_b:
                self.assertIn(action_b, valid_b)
            else:
                self.assertEqual(action_b, Action.WAIT)
            out = resolve_joint_action_outcome(
                st, action_a, action_b, board, max_steps
            )
            check_state_invariants(self, st, out.state, board, max_steps)
            if out.conflict:
                conflicts += 1
            if (out.state.agent_a, out.state.agent_b) != (st.agent_a, st.agent_b):
                moves += 1
            st = out.state
        self.assertEqual(st.step, max_steps)
        self.assertGreater(moves, 0)


if __name__ == "__main__":
    unittest.main()

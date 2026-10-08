"""
White-box regression tests for search identity, the tactical window, and
the gates around the two hard push filters (evaluation.py's `_min_delivery`
section references this file).

What is pinned here, beyond the behavioral suite in test_competitive.py:

* state identity - the closed set and both caches key on configuration AND
  round (step) and are scoped to their board: the same position at a
  different round has different remaining time and (when the parity flips)
  the OPPOSITE conflict priority, and two boards must never share a cache
  entry;
* the tactical window - a pure-action maximin over complete simultaneous
  rounds, verified against an independent exhaustive reference on a tiny
  board for rounds 2 and 3, plus the capacity_lab ranking that caught the
  wrong-push inversion (the window must put the correct approach first and
  price the wrong push);
* cross-root transposition - a state reached under a second root still
  credits its continuation value to that root (the credit happens before
  the closed-set skip);
* root pricing - a scoring push is free, walking toward the pushing side
  is free, retreats and pushes that drag a box away from its goal pay
  AWAY_PENALTY per damaged step, and returning to a just-visited cell pays
  REVISIT_PENALTY at the root;
* ranking semantics - a transient optimistic descendant peak can never
  override the window's verdict (bucket 1 beats bucket 2), and a timed-out
  window falls back to the one-ply robust values with telemetry reporting
  window=0;
* race + matching - maximum cardinality (a deliverable box is never
  dropped because another box consumed its only goal - the capacity_lab INF
  artifact that swung the conversion edge by the full clamp), independent
  per-box costs for the race winner and the edge, and the ADV_CAP branches;
* the denial gate opens only when the opponent truly wins the delivery race
  within the remaining steps; the own-credited-box rule is independent of it;
* the early exit - a quiet position returns well before the deadline.

Reference semantics (mixed vs pure strategies): the exhaustive reference
below searches PURE actions only. No randomized (mixed) strategies are
modelled anywhere in this engine; the window's ordering (I commit first,
the opponent answers knowing my action, I take their worst case) is the
conservative pure-action maximin of the simultaneous game under the
one-ply-committed interpretation used by the planner, and both sides are
held to exactly that standard here. Terminal leaves are the exact frozen
score; non-terminal leaves are heuristic estimates, not proven bounds.
"""

import time
import unittest
from collections import deque

from src.competitive.agent_a import (
    AWAY_PENALTY,
    LAST_SEARCH,
    REVISIT_PENALTY,
    _Planner,
    _key,
    _rank_structure,
    best_action,
    history_entry,
)
from src.competitive.evaluation import (
    ADV_CAP,
    INF,
    _advantage,
    _match_costs,
    _min_delivery,
    _race,
    denial_justified,
    evaluate,
)
from src.competitive.parser import parse_competitive_map
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (
    clear_cache,
    get_valid_actions,
    resolve_joint_action_outcome,
)

MAX = 50

# Replay that reproduces capacity_lab at the audited decision point
# (step 5): A pushing its own box south, B one cell above its easy box.
CAPACITY_REPLAY = (
    (Action.SOUTH, Action.SOUTH),
    (Action.SOUTH, Action.WEST),
    (Action.EAST, Action.WEST),
    (Action.EAST, Action.SOUTH),
    (Action.NORTH, Action.SOUTH),
)


def square_board(size=7, extra_walls=frozenset(), goals=((3, 4),)):
    walls = frozenset(
        (x, y)
        for x in range(size)
        for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    ) | frozenset(extra_walls)
    return Board(walls, frozenset(goals), size, size)


def capacity_step5():
    state, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
    for action_a, action_b in CAPACITY_REPLAY:
        state = resolve_joint_action_outcome(state, action_a, action_b, board, 60).state
    return state, board


class TestStateIdentityAndCaches(unittest.TestCase):
    """Round and board scoping of identity (the "identity across rounds"
    regression family)."""

    def test_key_distinguishes_the_same_configuration_at_different_rounds(self):
        base = CompetitiveState(
            (3, 3), (5, 5), frozenset({(4, 4)}), frozenset(), frozenset(), 0
        )
        later = CompetitiveState(
            (3, 3), (5, 5), frozenset({(4, 4)}), frozenset(), frozenset(), 3
        )
        self.assertNotEqual(_key(base), _key(later))
        # Same configuration, same round -> identical identity.
        again = CompetitiveState(
            (3, 3), (5, 5), frozenset({(4, 4)}), frozenset(), frozenset(), 0
        )
        self.assertEqual(_key(base), _key(again))
        # ...but a changed box position or credit is also a new identity.
        moved = CompetitiveState(
            (3, 3), (5, 5), frozenset({(4, 5)}), frozenset(), frozenset(), 0
        )
        self.assertNotEqual(_key(base), _key(moved))

    def test_eval_cache_is_board_scoped(self):
        board_a = square_board(goals=((5, 5),))
        board_b = square_board(goals=((1, 5),))
        self.assertNotEqual(board_a.serial, board_b.serial)  # premise
        state = CompetitiveState(
            (1, 3), (5, 1), frozenset({(2, 2)}), frozenset(), frozenset(), 0
        )
        shared = {}
        value_a = evaluate(state, board_a, "A", MAX, cache=shared)
        value_b = evaluate(state, board_b, "A", MAX, cache=shared)
        value_b_fresh = evaluate(state, board_b, "A", MAX, cache={})
        # board_a's entry must not answer for board_b, ...
        self.assertEqual(value_b, value_b_fresh)
        # ...and the two boards genuinely disagree about this position.
        self.assertNotEqual(value_a, value_b)

    def test_transition_cache_is_board_scoped(self):
        # Same state, same joint action, two boards: on board2 the push
        # destination is a wall, so the box must NOT move - unless the
        # memoized transition answers with board1's outcome.
        board1 = square_board(goals=((3, 4),))
        board2 = square_board(goals=((3, 4),), extra_walls=((3, 3),))
        self.assertNotEqual(board1.serial, board2.serial)  # premise
        state = CompetitiveState(
            (1, 3), (5, 1), frozenset({(2, 3)}), frozenset(), frozenset(), 0
        )
        clear_cache()
        out1 = resolve_joint_action_outcome(
            state, Action.EAST, Action.WEST, board1, MAX
        )
        out2 = resolve_joint_action_outcome(
            state, Action.EAST, Action.WEST, board2, MAX
        )
        self.assertIn((3, 3), out1.state.boxes)       # board1: push lands
        self.assertEqual(out2.state.boxes, state.boxes)  # board2: wall blocks
        # and a repeat of board1's call after board2 still answers board1
        out1_again = resolve_joint_action_outcome(
            state, Action.EAST, Action.WEST, board1, MAX
        )
        self.assertIn((3, 3), out1_again.state.boxes)


class TestTacticalWindow(unittest.TestCase):
    """The window is a pure-action maximin over complete simultaneous
    rounds; the planner must agree with an independent exhaustive
    reference to the last bit."""

    def _ref_reply(self, state, board, my_action, rounds_left):
        """Planner `_w_reply` in the test's own words: opponent's worst
        rule-permitted pure reply through the real joint transition."""
        if rounds_left <= 0:
            raise AssertionError("reply with no rounds left")
        replies = (
            get_valid_actions(state.agent_b, state.agent_a, state.boxes, board)
            or [Action.WAIT]
        )
        return min(
            self._ref_value(
                resolve_joint_action_outcome(
                    state, my_action, reply, board, MAX
                ).state,
                board,
                rounds_left - 1,
            )
            for reply in replies
        )

    def _ref_value(self, state, board, rounds_left):
        """Planner `_w_value`: my best pure commit, then their minimum;
        `evaluate` at the leaves."""
        if rounds_left <= 0:
            return evaluate(state, board, "A", MAX)
        actions = (
            get_valid_actions(state.agent_a, state.agent_b, state.boxes, board)
            or [Action.WAIT]
        )
        return max(
            self._ref_reply(state, board, action, rounds_left)
            for action in actions
        )

    def test_window_matches_exhaustive_reference(self):
        # Tiny open position; box (3,3) is 4 pushes from every corner, so
        # within 3 rounds no push can reach a deadlock cell and the
        # planner's forbidden filter stays inert - the reference's raw
        # action lists are exactly the planner's. (Documented premise, so
        # the equality below compares the SEARCH, not filter side effects.)
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (2, 3), (5, 3), frozenset({(3, 3)}), frozenset(), frozenset(), 0
        )
        planner = _Planner(
            state, board, MAX, "A", time.monotonic() + 30, None, None, None
        )
        planner.root_actions = planner._my_actions(state)
        raw_roots = (
            get_valid_actions(state.agent_a, state.agent_b, state.boxes, board)
            or [Action.WAIT]
        )
        self.assertEqual(planner.root_actions, raw_roots)  # filter inert

        for rounds in (2, 3):
            window = planner._tactical_window(rounds, time.monotonic() + 30)
            self.assertEqual(set(window), set(planner.root_actions))
            for root in planner.root_actions:
                reference = self._ref_reply(state, board, root, rounds)
                self.assertAlmostEqual(
                    window[root],
                    reference,
                    places=9,
                    msg=f"root {root.name} disagrees at rounds={rounds}: "
                        f"planner {window[root]} vs reference {reference}",
                )

    def test_capacity_window_ranks_correct_approach_over_wrong_push(self):
        # The audited bug: B one push from its goal. The window (3 rounds)
        # must put the correct approach (EAST: walk, walk, cash in) far
        # ahead of the wrong push (SOUTH: drive the box away).
        state, board = capacity_step5()
        planner = _Planner(
            state, board, 60, "B", time.monotonic() + 30, None, None, None
        )
        planner.root_actions = planner._my_actions(state)
        window = planner._tactical_window(3, time.monotonic() + 30)
        self.assertGreater(
            window[Action.EAST] - window[Action.SOUTH],
            100.0,  # more than one tactical bucket: decisive, not noise
            msg=f"window must rank EAST well above SOUTH: {window}",
        )
        # And the root penalties agree: the wrong push is charged for
        # dragging the box away, the correct approach walk is free.
        child_e, _ = planner._worst_reply(state, Action.EAST)
        child_s, _ = planner._worst_reply(state, Action.SOUTH)
        self.assertEqual(planner._away_penalty(state, child_e), 0.0)
        self.assertGreaterEqual(planner._away_penalty(state, child_s), 75.0)


class TestRootValueAggregation(unittest.TestCase):
    """Backup semantics: cross-root transfer, transient peaks, fallback."""

    def test_cross_root_transposition_credits_both_roots(self):
        # Both of my lines meet the SAME state through distinct branches
        # (opponent MOBILE, not boxed - see the note below):
        #   root EAST  -> (5,1) -> WEST  -> (4,1)  \
        #   root WEST  -> (3,1) -> EAST  -> (4,1)  /  step 2 either way
        # Box (4,2) -> goal (4,4): delivery cost is dist(pos,(4,1)) + 2,
        # so BOTH seed walks (E/W from (4,1)) raise the delivery cost
        # 2 -> 3 and are charged AWAY_PENALTY each - their robust seeds
        # land at 559 while the converged value is 596. Every alternative
        # child of both lines evals strictly below 596 as well (W_ADV
        # prices the longer walks; W_INIT the strike distance).
        #
        # Hence, if the credit is skipped when the key is already in the
        # closed set (the pre-fix behavior), the root that arrives SECOND
        # keeps max(seed, own alternatives) < stored and one of the two
        # assertions below fails - whichever root the heap happens to
        # expand first. (A boxed opponent cannot be used here: eval then
        # collapses to W_LOCKED * score_diff, identical for every cell,
        # and the seed alone would equal the stored value.)
        board = square_board(goals=((4, 4),))
        state = CompetitiveState(
            (4, 1), (2, 5), frozenset({(4, 2)}), frozenset(), frozenset(), 0
        )
        planner = _Planner(
            state, board, MAX, "A", time.monotonic() + 2.0, None, None, None
        )

        # Premises: round 1 gives both roots the SAME worst reply P,
        # round 2 the same worst reply Q, so both lines arrive at one
        # converged state (root-independence of the stored value).
        parent_e, _ = planner._worst_reply(state, Action.EAST)
        parent_w, _ = planner._worst_reply(state, Action.WEST)
        self.assertEqual(parent_e.agent_a, (5, 1))
        self.assertEqual(parent_w.agent_a, (3, 1))
        self.assertEqual(parent_e.agent_b, parent_w.agent_b)
        child_e, value_e = planner._worst_reply(parent_e, Action.WEST)
        child_w, value_w = planner._worst_reply(parent_w, Action.EAST)
        self.assertEqual(child_e, child_w)
        self.assertAlmostEqual(value_e, value_w, places=9)
        converged = child_e

        stored_eval = evaluate(converged, board, "A", MAX)

        # Premise: converged is STRICTLY better than every alternative
        # child of both lines (otherwise the closed-set skip would be
        # invisible: an equal-valued sibling would lift the second
        # root's deep term anyway).
        opp = converged.agent_b
        for alt_pos in ((5, 2), (2, 1), (3, 2)):
            alt = CompetitiveState(
                alt_pos, opp, frozenset({(4, 2)}),
                frozenset(), frozenset(), 2,
            )
            self.assertGreater(
                stored_eval, evaluate(alt, board, "A", MAX),
                msg=f"child {alt_pos} must be strictly worse than (4,1)",
            )
        # Premise: both robust seeds (penalised walks) sit below stored.
        for seed in (parent_e, parent_w):
            robust = (
                evaluate(seed, board, "A", MAX)
                - planner._away_penalty(state, seed)
            )
            self.assertGreater(
                stored_eval, robust,
                msg=f"seed {seed.agent_a} must rank below the convergence",
            )

        planner.run()
        key = _key(converged)
        self.assertIn(key, planner.closed)
        stored = planner.closed[key]
        self.assertAlmostEqual(stored, stored_eval, places=6)
        # BOTH roots received the continuation value of the shared state.
        self.assertGreaterEqual(planner.deep[Action.EAST], stored - 1e-9)
        self.assertGreaterEqual(planner.deep[Action.WEST], stored - 1e-9)

    def test_transient_peak_cannot_override_window(self):
        board = square_board(goals=((5, 5),))
        state = CompetitiveState(
            (3, 3), (5, 1), frozenset({(5, 5)}),
            frozenset({(5, 5)}), frozenset(), 0,
        )
        planner = _Planner(
            state, board, MAX, "A", time.monotonic() + 1, None, None, None
        )
        planner.root_actions = [Action.SOUTH, Action.EAST]
        # South is behind by a whole tactical bucket...
        planner.window = {Action.SOUTH: -100.0, Action.EAST: -350.0}
        planner.window_rounds = 3
        planner.robust = {Action.SOUTH: -100.0, Action.EAST: -50.0}
        planner.penalties = {Action.SOUTH: 0.0, Action.EAST: 0.0}
        planner.strikes = {Action.SOUTH: 1, Action.EAST: 1}
        # ...but the descendant term found a huge optimistic peak for it.
        planner.deep = {Action.SOUTH: -100.0, Action.EAST: 9000.0}
        self.assertEqual(planner._rank_root(), Action.SOUTH)

        # Window timed out: the ranking falls back to the one-ply robust
        # values (EAST ahead there), never to the stale window or peak.
        planner.window = None
        planner.window_rounds = 0
        self.assertEqual(planner._rank_root(), Action.EAST)

    def test_window_timeout_falls_back_to_robust_and_valid_direction(self):
        board = square_board(goals=((5, 5),))
        state = CompetitiveState(
            (3, 3), (5, 1), frozenset({(5, 5)}),
            frozenset({(5, 5)}), frozenset(), 0,
        )
        # Deadline already gone: seeding and/or the window must time out
        # without raising, and the answer must still be a legal direction.
        planner = _Planner(
            state, board, MAX, "A", time.monotonic(), None, None, None
        )
        action = planner.run()
        self.assertIsNone(planner.window)
        self.assertEqual(planner.window_rounds, 0)
        valid = (
            get_valid_actions(state.agent_a, state.agent_b, state.boxes, board)
            or [Action.WAIT]
        )
        self.assertIn(action, valid)

    def test_normal_search_reports_completed_window(self):
        board = square_board(goals=((5, 5),))
        state = CompetitiveState(
            (3, 3), (5, 1), frozenset({(5, 5)}),
            frozenset({(5, 5)}), frozenset(), 0,
        )
        action = best_action(state, board, MAX, "A", deque(maxlen=6), {}, {}, 0.3)
        self.assertGreaterEqual(LAST_SEARCH["window"], 2)
        valid = (
            get_valid_actions(state.agent_a, state.agent_b, state.boxes, board)
            or [Action.WAIT]
        )
        self.assertIn(action, valid)

    def test_structural_stability_ignores_value_creep_inside_buckets(self):
        # The early-exit comparison: refinement inside a bucket is NOT a
        # ranking change (requiring bit-stability made the exit
        # unreachable), while a bucket crossing or a winner flip is.
        winner = (Action.EAST, (-4, -3, 1, -430.0, -410.0, -500.0, 0))
        crept = (Action.EAST, (-4, -3, 1, -430.0, -399.0, -490.0, 0))
        self.assertEqual(_rank_structure(winner), _rank_structure(crept))
        bucket_crossed = (Action.EAST, (-4, -4, 1, -430.0, -410.0, -500.0, 0))
        self.assertNotEqual(_rank_structure(winner), _rank_structure(bucket_crossed))
        winner_changed = (Action.SOUTH, (-4, -3, 1, -430.0, -410.0, -500.0, 0))
        self.assertNotEqual(_rank_structure(winner), _rank_structure(winner_changed))


class TestRootPenalties(unittest.TestCase):
    """_away_penalty pricing: scoring free, approach free, damage charged."""

    def _planner(self, state, board):
        return _Planner(
            state, board, MAX, "A", time.monotonic() + 5, None, None, None
        )

    @staticmethod
    def _opp_reply(state, board):
        return get_valid_actions(
            state.agent_b, state.agent_a, state.boxes, board
        )[0]

    def test_scoring_push_pays_no_penalty(self):
        # Scoring push while a second box stays loose: cashing a point in
        # must never be charged for "losing" the box it just delivered.
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (3, 2), (6, 1), frozenset({(3, 3), (4, 4)}),
            frozenset(), frozenset(), 0,
        )
        planner = self._planner(state, board)
        child = resolve_joint_action_outcome(
            state, Action.SOUTH, self._opp_reply(state, board), board, MAX
        ).state
        self.assertIn((3, 4), child.boxes_on_goals_a)  # premise: it scored
        self.assertEqual(planner._away_penalty(state, child), 0.0)

    def test_approach_walk_is_free_and_retreat_is_charged(self):
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (1, 3), (5, 1), frozenset({(3, 3)}), frozenset(), frozenset(), 0
        )
        planner = self._planner(state, board)
        # EAST closes on the pushing side (3,2): delivery gets cheaper -> free.
        approach = resolve_joint_action_outcome(
            state, Action.EAST, self._opp_reply(state, board), board, MAX
        ).state
        self.assertEqual(approach.agent_a, (2, 3))  # premise: it walked
        self.assertEqual(planner._away_penalty(state, approach), 0.0)
        # SOUTH walks away from the box's only scoring approach: +1 step
        # of delivery cost -> one step of damage charged.
        retreat = resolve_joint_action_outcome(
            state, Action.SOUTH, self._opp_reply(state, board), board, MAX
        ).state
        self.assertEqual(retreat.agent_a, (1, 4))  # premise
        self.assertEqual(planner._away_penalty(state, retreat), AWAY_PENALTY)

    def test_push_that_drags_box_away_is_charged(self):
        # The capacity inversion in miniature: box one correct push from
        # the goal (S from (3,2) cashes it in), but I push it the wrong
        # way - EAST, off the scoring line. The box stays deliverable,
        # just more expensive (3 -> 6 steps of walk+push).
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (2, 3), (6, 1), frozenset({(3, 3)}), frozenset(), frozenset(), 0
        )
        planner = self._planner(state, board)
        child = resolve_joint_action_outcome(
            state, Action.EAST, self._opp_reply(state, board), board, MAX
        ).state
        self.assertIn((4, 3), child.boxes)  # premise: box dragged away
        # Delivery cost 3 -> 6 steps: three damaged steps priced at
        # AWAY_PENALTY each (was: 0, because the strike-based basis
        # moved the target set along with the push).
        self.assertEqual(planner._away_penalty(state, child), 3 * AWAY_PENALTY)

    def test_revisit_penalty_applied_at_root(self):
        board = square_board(goals=((5, 5),))
        state = CompetitiveState(
            (3, 3), (5, 1), frozenset({(5, 5)}),
            frozenset({(5, 5)}), frozenset(), 0,
        )
        probe = self._planner(state, board)
        child_east, _ = probe._worst_reply(state, Action.EAST)

        # Fingerprints: revisiting = same cell with an unchanged board...
        planner = _Planner(
            state, board, MAX, "A", time.monotonic() + 2,
            deque([history_entry(child_east, "A")], maxlen=6), None, None,
        )
        self.assertTrue(planner._is_revisit(child_east))
        self.assertFalse(planner._is_revisit(state))          # different cell
        changed = CompetitiveState(
            child_east.agent_a, child_east.agent_b,
            frozenset({(4, 4)}), child_east.boxes_on_goals_a,
            child_east.boxes_on_goals_b, child_east.step,
        )
        self.assertFalse(planner._is_revisit(changed))        # board changed

        # And run() actually charges it at the root.
        planner.run()
        self.assertGreaterEqual(planner.penalties[Action.EAST], REVISIT_PENALTY)


class TestRaceAndMatching(unittest.TestCase):
    """Race assignment: cardinality-first matching, independent costs."""

    def _capacity_state(self):
        state, board = capacity_step5()
        # The audited endgame position: A's box (4,7) and B's box (14,8)
        # both need goal (8,8) as (14,8)'s ONLY reachable goal; greedy
        # cheapest-first matching left (14,8) unmatched for A -> INF ->
        # "only the opponent can deliver" -> -ADV_CAP of phantom edge.
        state = CompetitiveState(
            (5, 4), (14, 7),
            frozenset({(4, 7), (5, 5), (14, 8)}),
            frozenset({(5, 5)}), frozenset(),
            state.step,
        )
        return state, board

    def test_match_costs_never_drops_a_deliverable_box(self):
        state, board = self._capacity_state()
        free_boxes = state.boxes - state.boxes_on_goals_a - state.boxes_on_goals_b
        free_goals = board.goals - state.boxes_on_goals_a - state.boxes_on_goals_b
        costs = _match_costs(state.agent_a, free_boxes, free_goals, board)
        # Maximum cardinality: BOTH boxes matched, each to a distinct goal.
        self.assertEqual(set(costs), {(4, 7), (14, 8)})
        for box, cost in costs.items():
            self.assertLess(cost, INF, msg=f"{box} unmatched")
        # and the goals are distinct
        self.assertEqual(len(costs), 2)

    def test_race_counts_and_edge_use_independent_costs(self):
        state, board = self._capacity_state()
        remaining = 60 - state.step
        projected_a, projected_b, adv = _race(state, board, remaining)
        # Each side wins exactly the box it can cash cheaper: the counts
        # must not be skewed by a dropped match or a phantom INF.
        self.assertEqual((projected_a, projected_b), (1.0, 1.0))
        # Edge = (-A's cost on (4,7)) + (+B's cost on (14,8)) in cost
        # units: small. The old matched-cost version summed to -35
        # (a -ADV_CAP charged for a box A in fact can deliver).
        self.assertLess(abs(adv), 10.0, msg=f"phantom exclusivity in edge: {adv}")

    def test_advantage_branches(self):
        self.assertEqual(_advantage(2, 4, 3), ADV_CAP)    # only I convert in time
        self.assertEqual(_advantage(INF, 4, 50), -ADV_CAP)  # only they can
        self.assertEqual(_advantage(6, INF, 50), ADV_CAP)   # only I can
        self.assertEqual(_advantage(INF, INF, 50), 0.0)     # nobody converts
        self.assertEqual(_advantage(6, 4, 3), 0.0)          # both out of time
        self.assertEqual(_advantage(4, 7, 50), -4.0)        # A ahead: charge my cost
        self.assertEqual(_advantage(7, 4, 50), 4.0)         # B ahead: gain their cost
        self.assertEqual(_advantage(5, 5, 50), 0.0)         # exact tie: split


class TestPushFilterGates(unittest.TestCase):
    """The gates the evaluation docstring references by file name."""

    def test_denial_only_when_opponent_really_wins_the_race(self):
        board = square_board(goals=((3, 4),))
        # Loose box (2,3): A adjacent-ish (delivery ~4), B across the map
        # (~10) - A wins the race with room to spare.
        state = CompetitiveState(
            (1, 3), (5, 1), frozenset({(2, 3)}), frozenset(), frozenset(), 0
        )
        a_cost = _min_delivery(state, board, (2, 3), "A")
        b_cost = _min_delivery(state, board, (2, 3), "B")
        self.assertLess(a_cost, b_cost)  # premise

        # B wants to kill the box: justified, A already owns this race.
        self.assertTrue(denial_justified(state, board, (2, 3), "B", MAX))
        # A killing its OWN race point is never justified.
        self.assertFalse(denial_justified(state, board, (2, 3), "A", MAX))

        # Out of time: A cannot actually deliver inside the remaining
        # steps, so the denial would destroy a box nobody can cash - the
        # rule stays hard (theirs > remaining -> False).
        theirs = _min_delivery(state, board, (2, 3), "A")
        self.assertGreaterEqual(theirs, 1)
        timed_out = CompetitiveState(
            state.agent_a, state.agent_b, state.boxes,
            state.boxes_on_goals_a, state.boxes_on_goals_b,
            MAX - (theirs - 1),   # remaining = theirs - 1
        )
        self.assertFalse(
            denial_justified(timed_out, board, (2, 3), "B", MAX)
        )

    def test_denial_refused_when_nobody_can_deliver(self):
        # The only goal is occupied by A's credited box: free goals are
        # empty, both sides read INF, denial is pointless -> False.
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (1, 3), (5, 1), frozenset({(3, 4), (2, 3)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        free_goals = board.goals - state.boxes_on_goals_a - state.boxes_on_goals_b
        self.assertEqual(free_goals, frozenset())  # premise
        self.assertFalse(denial_justified(state, board, (2, 3), "B", MAX))

    def test_own_credited_box_rule_is_independent_of_denial(self):
        # Pushing my own credited box off its goal stays forbidden no
        # matter what any race elsewhere says - the denial gate must not
        # unlock it (the constructed counterexample was refuted; see the
        # audit note at the bottom of evaluation.py).
        board = square_board(goals=((3, 4),))
        state = CompetitiveState(
            (2, 4), (5, 1), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        planner = _Planner(
            state, board, MAX, "A", time.monotonic() + 1, None, None, None
        )
        # EAST pushes the credited box (3,4) -> (4,4), off its goal.
        self.assertTrue(
            planner._forbidden(state, Action.EAST, state.boxes_on_goals_a)
        )
        # Pushing it back ONTO a goal is not throwing the point away.
        state2 = CompetitiveState(
            (4, 4), (5, 1), frozenset({(3, 4)}),
            frozenset({(3, 4)}), frozenset(), 0,
        )
        # (WEST would push it (3,4) -> (2,4), off goal: still forbidden.)
        planner2 = _Planner(
            state2, board, MAX, "A", time.monotonic() + 1, None, None, None
        )
        self.assertTrue(
            planner2._forbidden(state2, Action.WEST, state2.boxes_on_goals_a)
        )


class TestEarlyExit(unittest.TestCase):
    """Quiet positions return before the deadline; deadlines stay hard."""

    def test_quiet_position_exits_before_deadline(self):
        state, board = parse_competitive_map("maps/competitive/capacity_lab.txt")
        # Fresh process state, cold-ish caches: even so, the structural
        # stability check must fire long before the 2.95s deadline.
        best_action(state, board, 60, "A", deque(maxlen=6), {}, {}, 3.0)
        self.assertGreater(LAST_SEARCH["time"], 0.0)
        self.assertLess(
            LAST_SEARCH["time"], 2.0,
            msg="early exit did not fire on a quiet opening position "
                f"(time={LAST_SEARCH['time']:.3f}s, deadline=2.95s)",
        )
        self.assertGreaterEqual(LAST_SEARCH["window"], 2)

    def test_time_budget_is_still_a_hard_cap(self):
        # The other direction: an active search must never EXCEED the
        # limit (monotonic deadline; early exit only under-runs it).
        state, board = parse_competitive_map("maps/competitive/dense_goals.txt")
        started = time.monotonic()
        best_action(state, board, 60, "A", deque(maxlen=6), {}, {}, 0.4)
        wall = time.monotonic() - started
        self.assertLessEqual(wall, 0.4 + 0.15)  # limit + scheduling slack


if __name__ == "__main__":
    unittest.main()

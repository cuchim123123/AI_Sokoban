"""Independent exhaustive backups, deadline atomicity and rule regressions.

Replaces tests tied to removed GBFS buckets, peak transfer, hard filters and
forced-event extensions. Rule and state-identity coverage remains intact.
"""
import time
import gc
import unittest
from functools import lru_cache
from unittest.mock import patch
from src.competitive.agent_a import (_Planner, _Timeout, _key, best_action, history_entry,
                                     LAST_SEARCH, AgentA, round_preferences)
from src.competitive.evaluation import evaluate, evaluation_components
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (get_valid_actions, resolve_joint_action_outcome,
                                       clear_cache)
from tests.test_competitive import square_board, check_state_invariants

MAX = 50


def reference_roots(state, board, limit, who, depth):
    """No pruning, planner helpers or transposition bounds; full joint matrix."""
    def actions(s, side):
        p, o = (s.agent_a, s.agent_b) if side == "A" else (s.agent_b, s.agent_a)
        return get_valid_actions(p, o, s.boxes, board) or [Action.WAIT]

    def child(s, a, b):
        aa, ab = (a, b) if who == "A" else (b, a)
        return resolve_joint_action_outcome(
            s, aa, ab, board, limit,
            round_preferences(s, board, limit, "A", aa),
            round_preferences(s, board, limit, "B", ab)).state

    @lru_cache(None)
    def value(s, d):
        if d == 0 or s.is_terminal(limit):
            return evaluate(s, board, who, limit)
        return max(min(value(child(s, a, b), d-1)
                       for b in actions(s, "B" if who == "A" else "A"))
                   for a in actions(s, who))

    return {a: min(value(child(state, a, b), depth-1)
                   for b in actions(state, "B" if who == "A" else "A"))
            for a in actions(state, who)}


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



class TestCompletedSearch(unittest.TestCase):
    def planner(self, state, board, limit, who="A"):
        return _Planner(state, board, limit, who, time.monotonic()+30)

    def test_pruned_search_matches_full_joint_matrix_both_parities_and_sides(self):
        board = square_board(size=5, goals=((3, 3),))
        for step in (0, 1):
            for who in ("A", "B"):
                state = CompetitiveState((1,2),(3,2),{(2,2)},(),(),step)
                expected = reference_roots(state, board, 4, who, 3)
                p = self.planner(state, board, 4, who)
                roots = list(expected)
                actual = p.root_iteration(roots,3)
                for a in roots:
                    self.assertAlmostEqual(actual[a],expected[a])
                # Cached bounds and reversed traversal cannot change values.
                for a in roots:
                    p.row_value(state,a,3,-0.2,0.2)
                reversed_values = p.root_iteration(list(reversed(roots)),3)
                self.assertEqual(actual,reversed_values)

    def test_terminal_stops_expansion_and_uses_actual_score(self):
        b = square_board()
        s = CompetitiveState((1,1),(5,5),{(3,4)},{(3,4)},(),2)
        p = self.planner(s,b,2)
        with patch.object(p, "children", side_effect=AssertionError("expanded terminal")):
            self.assertEqual(p.value(s,10),1.0)
        self.assertEqual(evaluate(s,b,"B",2),-1.0)

    def test_partial_iteration_never_replaces_completed_results(self):
        b = square_board()
        s = CompetitiveState((3,2),(5,5),{(3,3)},(),(),0)
        p = self.planner(s,b,20)
        original = p.root_iteration
        completed = {}
        def interrupted(roots, depth):
            if depth == 1:
                completed.update(original(roots,depth))
                return completed.copy()
            original(roots[:1],depth)  # real work on only one candidate
            raise _Timeout()
        with patch.object(p,"root_iteration",side_effect=interrupted):
            chosen = p.run()
        self.assertEqual(p.completed_depth,1)
        self.assertEqual(p.root_values,completed)
        self.assertEqual(completed[chosen],max(completed.values()))

    def test_no_completed_iteration_returns_legal_direction(self):
        b = square_board()
        s = CompetitiveState((3,2),(5,5),{(3,3)},(),(),0)
        a = best_action(s,b,20,time_limit=0)
        self.assertIn(a,get_valid_actions(s.agent_a,s.agent_b,s.boxes,b))
        self.assertEqual(LAST_SEARCH["depth"],0)
        self.assertEqual(LAST_SEARCH["roots"],{})

    def test_deadline_gc_guard_restores_previous_setting_on_error(self):
        b=square_board()
        s=CompetitiveState((3,2),(5,5),{(3,3)},(),(),0)
        original=gc.isenabled()
        try:
            for enabled in (True,False):
                gc.enable() if enabled else gc.disable()
                with patch.object(_Planner,"run",side_effect=ValueError("test")):
                    with self.assertRaises(ValueError):
                        best_action(s,b,20,time_limit=.01)
                self.assertEqual(gc.isenabled(),enabled)
        finally:
            gc.enable() if original else gc.disable()

    def test_live_and_search_use_identical_preferences_for_both_players(self):
        b = square_board()
        s = CompetitiveState((2,3),(4,3),{(3,3)},(),(),0)
        p = self.planner(s,b,9)
        for a in p.actions(s,"A"):
            for _,reply,out in p.children(s,a):
                live = resolve_joint_action_outcome(s,a,reply,b,9,
                    round_preferences(s,b,9,"A",a),
                    round_preferences(s,b,9,"B",reply))
                self.assertEqual(out,live)

    def test_legal_sacrifices_are_not_filtered(self):
        b = square_board()
        s = CompetitiveState((3,5),(1,1),{(3,4)},{(3,4)},(),0)
        self.assertIn(Action.NORTH,self.planner(s,b,30).actions(s,"A"))
        s = CompetitiveState((1,3),(5,5),{(1,2)},(),(),0)
        self.assertIn(Action.NORTH,self.planner(s,b,30).actions(s,"A"))

    def test_contested_scoring_is_backed_by_exhaustive_continuations(self):
        # Former "safe immediate score" test. B can approach via (4,5),
        # (3,5), then push NORTH to strip A's goal on round three.
        b = square_board()
        s = CompetitiveState((3,2),(5,5),{(3,3)},(),(),0)
        expected = reference_roots(s,b,20,"A",5)
        planner = self.planner(s,b,20)
        shallow = planner.root_iteration(list(expected),1)
        self.assertEqual(max(shallow, key=shallow.get), Action.SOUTH)
        actual = planner.root_iteration(list(expected),5)
        self.assertEqual(actual,expected)
        self.assertLess(expected[Action.SOUTH],max(expected.values()))
        for aa,bb in ((Action.SOUTH,Action.WEST),
                      (Action.EAST,Action.WEST),(Action.EAST,Action.NORTH)):
            s=resolve_joint_action_outcome(s,aa,bb,b,20,
                round_preferences(s,b,20,"A",aa),
                round_preferences(s,b,20,"B",bb)).state
        self.assertEqual(s.score_a(),0)
        self.assertIn((3,3),s.boxes)

    def test_repetition_only_breaks_equal_value_ties(self):
        b=square_board()
        s=CompetitiveState((3,3),(5,5),(),(),(),0)
        previous=CompetitiveState((3,2),(5,5),(),(),(),0)
        a=best_action(s,b,1,recent_positions=[history_entry(previous,"A")],time_limit=.1)
        self.assertNotEqual(a,Action.NORTH)
        self.assertEqual({v["value"] for v in LAST_SEARCH["roots"].values()},{0.0})

    def test_safe_score_both_perspectives_at_round_limit(self):
        b=square_board(goals=((5,5),))
        for who in ("A","B"):
            a,c=((5,3),(1,1)) if who=="A" else ((1,1),(5,3))
            s=CompetitiveState(a,c,{(5,4)},(),(),0)
            self.assertEqual(best_action(s,b,1,who,time_limit=.1),Action.SOUTH)

    def test_preference_submission_is_cached_inside_decision(self):
        b=square_board()
        s=CompetitiveState((3,2),(5,5),{(3,3)},(),(),0)
        agent=AgentA(.05)
        a=agent.choose_action(s,b,10)
        with patch("src.competitive.agent_a.round_preferences",side_effect=AssertionError):
            self.assertEqual(agent.preference_list(s,b,10,a)[0],a)


class TestScoringAndEvaluation(unittest.TestCase):
    def test_credit_removal_and_goal_to_goal_transfer(self):
        b=square_board(goals=((3,3),(3,4)))
        s=CompetitiveState((1,1),(3,2),{(3,3)},{(3,3)},(),0)
        out=resolve_joint_action_outcome(s,Action.EAST,Action.SOUTH,b,4)
        self.assertEqual((out.state.score_a(),out.state.score_b()),(0,1))
        self.assertEqual(out.state.boxes_on_goals_b,frozenset({(3,4)}))
        out=resolve_joint_action_outcome(out.state,Action.WEST,Action.SOUTH,b,4)
        self.assertEqual((out.state.score_a(),out.state.score_b()),(0,0))

    def test_filled_goals_do_not_end_game(self):
        b=square_board()
        s=CompetitiveState((1,1),(5,5),{(3,4)},{(3,4)},(),0)
        self.assertFalse(s.is_terminal(10))

    def test_bounds_are_physical_not_just_wall_membership(self):
        b=Board(frozenset(),frozenset({(1,1)}),3,3)
        self.assertNotIn(Action.WEST,get_valid_actions((0,0),(2,2),frozenset(),b))
        self.assertNotIn(Action.NORTH,get_valid_actions((0,0),(2,2),frozenset(),b))

    def test_opportunities_do_not_sum_independent_deliveries(self):
        b=square_board(goals=((2,4),(4,4)))
        s=CompetitiveState((2,2),(5,1),{(2,3),(4,3)},(),(),0)
        terms=evaluation_components(s,b,20)
        self.assertLessEqual(abs(terms["opportunity"]),.65)
        self.assertEqual(terms["credit"],0)

    def test_uncredited_goal_box_does_not_count_as_zero_step_delivery(self):
        b=square_board(goals=((1,1),))
        s=CompetitiveState((1,2),(5,5),{(1,1)},(),(),0)
        self.assertEqual(evaluate(s,b,"A",1),0)

    def test_blocked_round_consumes_time_without_overlap(self):
        b=Board(frozenset({(x,y) for x in range(4) for y in range(3)
                           if (x,y) not in ((1,1),(2,1))}),frozenset(),4,3)
        s=CompetitiveState((1,1),(2,1),(),(),(),0)
        out=resolve_joint_action_outcome(s,Action.EAST,Action.WEST,b,3)
        self.assertEqual((out.state.agent_a,out.state.agent_b),((1,1),(2,1)))
        self.assertEqual(out.state.step,1)
        self.assertEqual(out.resolved_action_a,Action.WAIT)
        self.assertEqual(out.resolved_action_b,Action.WAIT)


if __name__ == "__main__":
    unittest.main()

"""Agent B yields only after two observed, complete position cycles."""
import unittest
from unittest.mock import patch

from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB
from src.competitive.state import Action, CompetitiveState
from src.competitive.transition import get_valid_actions
from tests.test_competitive import square_board


class TestAgentBLoopBreaker(unittest.TestCase):
    def setUp(self):
        self.board = square_board()
        self.agent = AgentB(time_limit=.02)
        self.calls = []

        def search(agent, state, board, limit, banned_actions=None):
            self.calls.append(set(banned_actions or ()))
            legal = get_valid_actions(state.agent_b, state.agent_a, state.boxes, board)
            legal = [a for a in legal if a not in (banned_actions or ())] or legal
            agent.last_search = {}
            return next((a for a in (Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH)
                         if a in legal), Action.WAIT)

        self.search = patch.object(AgentA, "choose_action", search)
        self.search.start()
        self.addCleanup(self.search.stop)

    def state(self, step, b=None, boxes=(), credited=()):
        return CompetitiveState((4,4) if step % 2 == 0 else (4,3),
                                b or ((2,2) if step % 2 == 0 else (3,2)),
                                boxes, (), credited, step)

    def test_two_complete_loops_then_only_b_excludes_loop_moves(self):
        for step in range(4):
            self.agent.choose_action(self.state(step), self.board, 30)
            self.assertEqual(self.calls[-1],set())
        action = self.agent.choose_action(self.state(4),self.board,30)
        self.assertNotEqual(action,Action.EAST)
        self.assertIn(Action.EAST,self.calls[-1])
        self.assertTrue(self.agent.last_search["loop_breaker"]["active"])
        self.assertEqual(self.agent.last_search["loop_breaker"]["period"],2)
        # Agent A has no detector or yielding override.
        self.assertFalse(hasattr(AgentA(), "_observe_cycle"))
        # One search per decision, no second planner run for yielding.
        self.assertEqual(len(self.calls),5)

    def test_longer_cycles_are_detected(self):
        cells=[(2,2),(3,2),(3,3),(2,3)]
        for step in range(9):
            self.agent.choose_action(self.state(step,cells[step%4]),self.board,30)
            self.assertEqual(bool(self.calls[-1]),step==8)
        self.assertEqual(self.agent.last_search["loop_breaker"]["period"],4)

    def test_repeated_queries_in_one_round_do_not_count_as_loops(self):
        for _ in range(5):
            self.agent.choose_action(self.state(0),self.board,30)
            self.assertEqual(self.calls[-1],set())

    def test_box_or_credit_progress_does_not_trigger(self):
        for step in range(5):
            state=self.state(step,boxes={(3,4)},credited={(3,4)} if step==4 else ())
            self.agent.choose_action(state,self.board,30)
        self.assertEqual(self.calls[-1],set())
        self.assertEqual(self.agent.last_search["loop_breaker"]["period"],0)

    def test_changed_box_position_is_not_a_repeated_configuration(self):
        for step in range(5):
            box=(3,5) if step==4 else (3,4)
            self.agent.choose_action(self.state(step,boxes={box}),self.board,30)
        self.assertEqual(self.calls[-1],set())

    def test_normal_search_resumes_after_leaving_cycle(self):
        for step in range(5):
            self.agent.choose_action(self.state(step),self.board,30)
        self.assertTrue(self.calls[-1])
        self.agent.choose_action(self.state(5,b=(1,2)),self.board,30)
        self.assertEqual(self.calls[-1],set())

    def test_rewind_new_board_or_skipped_round_resets_evidence(self):
        for reset in ("rewind","board","gap","limit"):
            self.agent=AgentB(.02)
            for step in range(4):
                self.agent.choose_action(self.state(step),self.board,30)
            step=0 if reset=="rewind" else 6 if reset=="gap" else 4
            board=square_board() if reset=="board" else self.board
            self.agent.choose_action(self.state(step),board,31 if reset=="limit" else 30)
            self.assertEqual(self.calls[-1],set(),reset)

    def test_no_legal_exit_preserves_legal_move(self):
        board=square_board(extra_walls={(1,2),(2,1),(2,3)})
        for step in range(5):
            action=self.agent.choose_action(self.state(step),board,30)
        self.assertEqual(action,Action.EAST)
        self.assertFalse(self.agent.last_search["loop_breaker"]["active"])

    def test_external_bans_are_preserved(self):
        for step in range(5):
            self.agent.choose_action(self.state(step),self.board,30,{Action.NORTH})
        self.assertIn(Action.NORTH,self.calls[-1])
        self.assertIn(Action.EAST,self.calls[-1])

    def test_two_conflict_stalls_try_another_submission(self):
        for step in range(3):
            state=CompetitiveState((4,4),(2,2),(),(),(),step)
            action=self.agent.choose_action(state,self.board,30)
        self.assertNotEqual(action,Action.EAST)
        self.assertEqual(self.agent.last_search["loop_breaker"]["period"],1)

    def test_real_search_caches_yielding_primary_for_live_execution(self):
        self.search.stop()
        for step in range(4):
            self.agent.choose_action(self.state(step),self.board,30)
        state=self.state(4)
        action=self.agent.choose_action(state,self.board,30)
        prefs=self.agent.preference_list(state,self.board,30,action)
        self.assertEqual(prefs[0],action)
        self.assertEqual(self.agent._last_action,action)
        self.assertEqual(set(prefs),set(get_valid_actions(
            state.agent_b,state.agent_a,state.boxes,self.board)))
        self.assertNotIn(action.name,self.agent.last_search["loop_breaker"]["excluded"])


if __name__ == "__main__":
    unittest.main()

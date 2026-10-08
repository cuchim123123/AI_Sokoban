"""Input integration without opening a window (requires optional Pygame)."""
import unittest
from unittest.mock import patch

try:
    import pygame
    from src.competitive.gui.app import CompetitiveApp
except ImportError:
    pygame = None

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.parser import parse_competitive_map
from src.competitive.transition import resolve_joint_action_outcome
from tests.test_competitive import square_board


@unittest.skipIf(pygame is None, "Pygame is not installed in this Python")
class TestHumanInput(unittest.TestCase):
    def setUp(self):
        self.app = CompetitiveApp.__new__(CompetitiveApp)
        self.app.in_menu = False
        self.app.running = True
        self.app.computing = True
        self.app.finished = False
        self.app.anim_t = 1
        self.app.agent_a = self.app.agent_b = None
        self.app.pending_human_a = self.app.pending_human_b = None
        self.app.board = square_board()
        self.app.state = CompetitiveState((1,1),(5,5),(),(),(),0)
        self.app.max_steps = 10

    def key(self, key):
        with patch.object(pygame.event, "get", return_value=[
                pygame.event.Event(pygame.KEYDOWN,key=key)]):
            self.app._handle_events()

    def test_shift_cannot_submit_wait(self):
        self.key(pygame.K_LSHIFT)
        self.key(pygame.K_RSHIFT)
        self.assertIsNone(self.app.pending_human_a)
        self.assertIsNone(self.app.pending_human_b)

    def test_illegal_direction_is_rejected_and_legal_direction_accepted(self):
        self.key(pygame.K_w)
        self.assertIsNone(self.app.pending_human_a)
        self.key(pygame.K_s)
        self.assertEqual(self.app.pending_human_a,Action.SOUTH)

    def test_boxed_in_humans_do_not_wait_for_impossible_key_submission(self):
        self.app.board = Board(frozenset((x,y) for x in range(5) for y in range(3)
                                        if (x,y) not in ((1,1),(3,1))),frozenset(),5,3)
        self.app.state = CompetitiveState((1,1),(3,1),(),(),(),0)
        self.app._compute_step()
        outcome = self.app.pending_out[3]
        self.assertEqual(outcome.state.step,1)
        self.assertEqual(outcome.resolved_action_a,Action.WAIT)
        self.assertEqual(outcome.resolved_action_b,Action.WAIT)


@unittest.skipIf(pygame is None, "Pygame is not installed in this Python")
class TestBoxAnimation(unittest.TestCase):
    def setUp(self):
        initial, board = parse_competitive_map("maps/competitive/main.txt")
        # Two west pushes with equally near crossed destinations: the old
        # greedy matching made both boxes meet halfway on a diagonal.
        boxes = initial.boxes - {(4, 3)} | {(4, 1)}
        self.before = CompetitiveState((4, 2), (5, 1), boxes, (), (), 21)
        self.outcome = resolve_joint_action_outcome(
            self.before, Action.WEST, Action.WEST, board, 50)
        self.app = CompetitiveApp.__new__(CompetitiveApp)
        self.app.prev_state = self.before
        self.app.state = self.outcome.state

    def test_simultaneous_west_pushes_keep_their_actual_box_paths(self):
        self.assertFalse(self.outcome.conflict)
        self.assertEqual(self.outcome.resolved_action_a, Action.WEST)
        self.assertEqual(self.outcome.resolved_action_b, Action.WEST)
        for t in (0, 0.25, 0.5, 1):
            with self.subTest(t=t):
                visuals = dict((dest, pos) for pos, dest
                               in self.app._get_box_visual_positions(t))
                self.assertEqual(visuals[(2, 2)], (3 - t, 2))
                self.assertEqual(visuals[(3, 1)], (4 - t, 1))
                for box in self.before.boxes & self.outcome.state.boxes:
                    self.assertEqual(visuals[box], box)
                self.assertEqual(len(visuals), len(self.before.boxes))

    def test_rewind_reverses_both_push_paths(self):
        self.app.prev_state, self.app.state = self.app.state, self.app.prev_state
        for t in (0, 0.5, 1):
            with self.subTest(t=t):
                visuals = dict((dest, pos) for pos, dest
                               in self.app._get_box_visual_positions(t))
                self.assertEqual(visuals[(3, 2)], (2 + t, 2))
                self.assertEqual(visuals[(4, 1)], (3 + t, 1))

    def test_identical_states_leave_all_boxes_stationary(self):
        self.app.state = self.app.prev_state
        self.assertEqual(dict((dest, pos) for pos, dest
                              in self.app._get_box_visual_positions(0.5)),
                         {box: box for box in self.before.boxes})

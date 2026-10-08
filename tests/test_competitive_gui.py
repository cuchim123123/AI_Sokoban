"""Input integration without opening a window (requires optional Pygame)."""
import unittest
from unittest.mock import patch

try:
    import pygame
    from src.competitive.gui.app import CompetitiveApp
except ImportError:
    pygame = None

from src.competitive.state import Action, Board, CompetitiveState
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

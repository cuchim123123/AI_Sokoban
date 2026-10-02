"""
Agent B controller — Greedy Best-First Search (GBFS).

Separate source file as required by Requirement 8, so different student
groups can swap in their own implementations without touching the game engine.
"""
from collections import deque
from typing import Deque, Tuple

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.agent_a import _gbfs_best_action


class AgentB:
    """
    Agent B controller.  Mirrors AgentA's GBFS but from B's perspective.
    Maintains its own position history for loop detection.
    """

    def __init__(self):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=4)

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
    ) -> Action:
        action = _gbfs_best_action(
            state, board, max_steps,
            perspective='B',
            opponent_perspective='A',
            recent_positions=self._history,
        )
        self._history.append(state.agent_b)
        return action

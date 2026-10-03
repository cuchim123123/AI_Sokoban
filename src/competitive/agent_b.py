"""
Agent B controller.
Delegates to the shared `best_action` search function in agent_a.py.
"""
from collections import deque
from typing import Deque, Tuple, Dict
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.agent_a import best_action


class AgentB:
    """
    Agent B controller. Uses a global Transposition Table and Heuristic Cache
    to retain knowledge across turns.
    """

    def __init__(self):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=4)
        self.tt: Dict[int, Tuple[int, float, Action]] = {}
        self.heuristic_cache: Dict[int, float] = {}

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions: list = None
    ) -> Action:
        
        if len(self.heuristic_cache) > 500000:
            self.heuristic_cache.clear()

        action = best_action(
            state, board, max_steps,
            perspective="B",
            recent_positions=self._history,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=banned_actions
        )

        dx, dy = action.value
        expected = (state.agent_b[0] + dx, state.agent_b[1] + dy)
        self._history.append(state.agent_b)

        return action

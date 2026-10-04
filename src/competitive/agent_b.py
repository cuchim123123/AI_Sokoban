"""
Agent B controller.
Delegates to the shared `best_action` search function in agent_a.py.
"""
from collections import deque
from typing import Deque, Tuple, Dict, Optional
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.agent_a import best_action


class AgentB:
    """
    Agent B controller. Uses a global Transposition Table and Heuristic Cache
    to retain knowledge across turns.
    """

    def __init__(self):
        self._history: Deque = deque(maxlen=4)
        self.tt: Dict[int, Tuple[int, float, Action]] = {}
        self.heuristic_cache: Dict[int, float] = {}
        self.action_cache: Dict[tuple, List[Action]] = {}
        self.tactical_cache: Dict[tuple, object] = {}
        self.result_cache: Dict[tuple, tuple] = {}
        self._last_action: Optional[Action] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions: list = None
    ) -> Action:

        if len(self.heuristic_cache) > 500000:
            self.heuristic_cache.clear()
        if len(self.action_cache) > 500000:
            self.action_cache.clear()
        if len(self.tactical_cache) > 500000:
            self.tactical_cache.clear()
        if len(self.result_cache) > 500000:
            self.result_cache.clear()

        auto_banned = list(banned_actions) if banned_actions else []

        action = best_action(
            state, board, max_steps,
            perspective="B",
            recent_positions=self._history,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=auto_banned,
            action_cache=self.action_cache,
            tactical_cache=self.tactical_cache,
            result_cache=self.result_cache,
        )

        self._history.append((state.agent_b, state.board_hash))
        
        self._last_pos = state.agent_b
        self._last_action = action

        return action

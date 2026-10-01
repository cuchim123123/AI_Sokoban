from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from src.core.state import GameState, Board, Action

class SearchAlgorithm(ABC):
    @abstractmethod
    def search(self, initial_state: GameState, board: Board) -> Tuple[Optional[List[Action]], int, int, int]:
        """
        Executes the search algorithm.
        Returns: (actions, total_cost, generated_states, expanded_states)
        If no solution is found, actions should be None.
        """
        pass

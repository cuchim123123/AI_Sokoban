"""
Agent B controller — Greedy Best-First Search (GBFS), depth-1 (pure greedy).

Separate source file as required by Requirement 8, so different student
groups can swap in their own implementations without touching the game engine.

This implementation uses PURE DEPTH-1 GREEDY:
  - For each valid action, simulate one step (predicting opponent plays best response)
  - Pick the action with the highest immediate evaluation score.
  - No deep search tree — faster per step, but less strategic than Agent A.

This intentional asymmetry between A (deep GBFS) and B (pure greedy) produces
clear, demonstrable differences in play quality for the assignment presentation.
"""
from collections import deque
from typing import Deque, List, Tuple

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import resolve_joint_action, get_valid_actions
from src.competitive.evaluation import competitive_heuristic


class AgentB:
    """
    Agent B — pure depth-1 greedy GBFS (no deep search tree).
    Picks the immediately best action using a single-level look-ahead.
    """

    def __init__(self):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=3)

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
    ) -> Action:
        my_pos  = state.agent_b
        opp_pos = state.agent_a

        my_actions = get_valid_actions(my_pos, opp_pos, state.boxes, board)

        best_action = my_actions[0] if my_actions else Action.WAIT
        best_h      = -float('inf')

        # Predict opponent's greedy best action (depth-1 opponent model)
        opp_actions = get_valid_actions(opp_pos, my_pos, state.boxes, board)
        opp_best    = Action.WAIT
        opp_best_h  = -float('inf')
        for opp_act in opp_actions:
            ns = resolve_joint_action(state, opp_act, Action.WAIT, board)
            h  = competitive_heuristic(ns, board, 'A', max_steps)
            if h > opp_best_h:
                opp_best_h = h
                opp_best   = opp_act

        # Evaluate each of B's actions against opponent's predicted move
        LOOP_PENALTY = 25
        for act in my_actions:
            ns = resolve_joint_action(state, opp_best, act, board)
            h  = competitive_heuristic(ns, board, 'B', max_steps)
            # Penalise revisiting recent cells
            if ns.agent_b in self._history:
                h -= LOOP_PENALTY
            if h > best_h:
                best_h      = h
                best_action = act

        self._history.append(state.agent_b)
        return best_action

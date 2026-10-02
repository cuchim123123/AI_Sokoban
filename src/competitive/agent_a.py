"""
Agent A controller — time-bounded Greedy Best-First Search (GBFS).

Design decisions to avoid looping / staying still:
  - WAIT is only ever returned as a last resort (no other action available).
  - Recent position history (last 4 cells) penalises revisiting the same cell.
  - Opponent is modelled via a 1-step greedy look-ahead (not assumed to WAIT).
  - GBFS explores states ordered by -h(s); the best first action on the
    deepest explored path is returned before the deadline.
  - A tie-breaking counter prevents heap-order ambiguity.
"""
import heapq
import time
import itertools
from typing import Optional, Tuple, Deque
from collections import deque

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import resolve_joint_action, get_valid_actions
from src.competitive.evaluation import competitive_heuristic

_TIME_LIMIT = 0.90   # seconds — leaves headroom for GUI
_LOOP_PENALTY = 30   # heuristic penalty for immediately revisiting a cell


def _predict_opponent_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    opp_perspective: str,
    my_action: Action,
    my_perspective: str,
) -> Action:
    """
    Predict what the opponent will greedily do given that we play `my_action`.
    Evaluates all opponent actions against the joint state to pick the best.
    """
    # 'opp_perspective' is the opponent's label ('A' or 'B')
    if opp_perspective == 'B':
        opp_pos = state.agent_b
        my_pos  = state.agent_a
    else:
        opp_pos = state.agent_a
        my_pos  = state.agent_b

    opp_acts = get_valid_actions(opp_pos, my_pos, state.boxes, board)

    best_act = opp_acts[0] if opp_acts else Action.WAIT
    best_h   = -float('inf')

    for opp_act in opp_acts:
        # Simulate joint move: my action + opponent's candidate action
        if my_perspective == 'A':
            ns = resolve_joint_action(state, my_action, opp_act, board)
        else:
            ns = resolve_joint_action(state, opp_act, my_action, board)
        h = competitive_heuristic(ns, board, opp_perspective, max_steps)
        if h > best_h:
            best_h   = h
            best_act = opp_act

    return best_act


def _gbfs_best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str,
    opponent_perspective: str,
    recent_positions: Deque,
    time_limit: float = _TIME_LIMIT,
) -> Action:
    """
    Time-bounded GBFS returning the single best action for the given agent.
    `recent_positions` is a deque of the last few (x,y) cells the agent
    actually occupied — used to penalise revisiting and break loops.
    """
    deadline = time.time() + time_limit

    my_pos  = state.agent_a if perspective == 'A' else state.agent_b
    opp_pos = state.agent_b if perspective == 'A' else state.agent_a

    my_actions = get_valid_actions(my_pos, opp_pos, state.boxes, board)
    # include_wait=True already handled inside get_valid_actions when empty

    if not my_actions:
        return Action.WAIT

    # ── Depth-1 greedy pick (guaranteed before deadline) ─────────────────────
    best_action = my_actions[0]
    best_h      = -float('inf')

    for act in my_actions:
        # Predict opponent response to this action
        opp_act = _predict_opponent_action(
            state, board, max_steps, opponent_perspective, act, perspective
        )
        if perspective == 'A':
            ns = resolve_joint_action(state, act, opp_act, board)
        else:
            ns = resolve_joint_action(state, opp_act, act, board)

        h = competitive_heuristic(ns, board, perspective, max_steps)

        # Penalise immediately returning to a recently visited cell
        new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
        if new_my_pos in recent_positions:
            h -= _LOOP_PENALTY

        if h > best_h:
            best_h   = h
            best_action = act

    if state.is_terminal(max_steps):
        return best_action

    # ── GBFS deeper search ────────────────────────────────────────────────────
    # Frontier: (-h, tie_counter, state, first_action_taken)
    counter  = itertools.count()
    frontier: list = []
    visited  = set()

    # Seed frontier with depth-1 successors
    for act in my_actions:
        if time.time() >= deadline:
            break
        opp_act = _predict_opponent_action(
            state, board, max_steps, opponent_perspective, act, perspective
        )
        if perspective == 'A':
            ns = resolve_joint_action(state, act, opp_act, board)
        else:
            ns = resolve_joint_action(state, opp_act, act, board)

        h = competitive_heuristic(ns, board, perspective, max_steps)
        new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
        if new_my_pos in recent_positions:
            h -= _LOOP_PENALTY

        heapq.heappush(frontier, (-h, next(counter), ns, act))
        visited.add(ns)

    while frontier and time.time() < deadline:
        neg_h, _, current, first_action = heapq.heappop(frontier)
        h_val = -neg_h

        if h_val > best_h:
            best_h      = h_val
            best_action = first_action

        if current.is_terminal(max_steps):
            continue

        curr_my_pos  = current.agent_a if perspective == 'A' else current.agent_b
        curr_opp_pos = current.agent_b if perspective == 'A' else current.agent_a
        curr_acts    = get_valid_actions(curr_my_pos, curr_opp_pos, current.boxes, board)

        for act in curr_acts:
            if time.time() >= deadline:
                break
            opp_act = _predict_opponent_action(
                current, board, max_steps, opponent_perspective, act, perspective
            )
            if perspective == 'A':
                ns = resolve_joint_action(current, act, opp_act, board)
            else:
                ns = resolve_joint_action(current, opp_act, act, board)

            if ns in visited:
                continue
            visited.add(ns)

            h = competitive_heuristic(ns, board, perspective, max_steps)
            new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
            if new_my_pos in recent_positions:
                h -= _LOOP_PENALTY

            heapq.heappush(frontier, (-h, next(counter), ns, first_action))

    return best_action


class AgentA:
    """
    Agent A controller.  Maintains its own position history to detect and
    escape loops.  Called once per game step via `choose_action()`.
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
            perspective='A',
            opponent_perspective='B',
            recent_positions=self._history,
        )
        # Record where we expect to move (optimistically — transition may block us,
        # but it still breaks the pattern)
        dx, dy = action.value
        expected = (state.agent_a[0] + dx, state.agent_a[1] + dy)
        self._history.append(state.agent_a)
        return action

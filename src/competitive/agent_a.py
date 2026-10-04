"""
Agent controller — Greedy Best-First Search (GBFS).
"""
import time
from collections import deque
from typing import Optional, Tuple, Deque, Dict, List

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome
from src.competitive.evaluation import competitive_heuristic

# ── Search Constants ──────────────────────────────────────────────────────────
TIME_LIMIT = 0.90
SEARCH_DEPTH = 6
BEAM_WIDTH = 24
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(
    state: CompetitiveState,
    perspective: str,
    max_steps: int,
) -> tuple:
    return (state._hash, perspective, max_steps)


def _value(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    max_steps: int,
    heuristic_cache: Dict[tuple, float],
) -> float:
    key = _cache_key(state, perspective, max_steps)
    if key not in heuristic_cache:
        heuristic_cache[key] = competitive_heuristic(
            state, board, perspective, max_steps
        )
    return heuristic_cache[key]


def _joint_action(state: CompetitiveState, perspective: str, own_action: Action, opp_action: Action):
    if perspective == 'A':
        return own_action, opp_action
    return opp_action, own_action


def _pushes_own_finished_box(
    state: CompetitiveState,
    action: Action,
    perspective: str,
) -> bool:
    if action == Action.WAIT:
        return False

    position = state.agent_a if perspective == 'A' else state.agent_b
    own_goals = (
        state.boxes_on_goals_a
        if perspective == 'A'
        else state.boxes_on_goals_b
    )
    destination = (
        position[0] + action.value[0],
        position[1] + action.value[1],
    )
    return destination in own_goals


def _robust_successor(
    curr: CompetitiveState,
    own_action: Action,
    board: Board,
    perspective: str,
    max_steps: int,
    heuristic_cache: Dict[tuple, float],
):
    """Return the worst legal opponent response to one own action."""
    my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
    op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
    opponent_actions = get_valid_actions(op_pos, my_pos, curr.boxes, board)
    candidates = []
    pushes_own_finished_box = _pushes_own_finished_box(
        curr, own_action, perspective
    )
    for opponent_action in opponent_actions:
        action_a, action_b = _joint_action(
            curr, perspective, own_action, opponent_action
        )
        outcome = resolve_joint_action_outcome(
            curr, action_a, action_b, board, max_steps
        )
        next_state = outcome.state
        own_position_before = my_pos
        own_position_after = (
            next_state.agent_a if perspective == 'A' else next_state.agent_b
        )
        effective_value = _value(
            next_state, board, perspective, max_steps, heuristic_cache
        )
        if pushes_own_finished_box:
            # The opponent may steal a credited box, but an agent should not
            # voluntarily destroy its own score while pursuing another route.
            effective_value = float('-inf')
        if own_action != Action.WAIT and own_position_after == own_position_before:
            # A rejected move or push is not a useful response to an adversarial
            # opponent. Treat any response that blocks our action as unsafe.
            effective_value = float('-inf')
        candidates.append((
            effective_value,
            opponent_action,
            next_state,
        ))
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1].value))


def best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str,
    recent_positions: Deque[Tuple[int, int]],
    tt: Dict[int, Tuple[int, float, Action]],
    heuristic_cache: Dict[tuple, float],
    time_limit: float = TIME_LIMIT,
    banned_actions: List[Action] = None
) -> Action:
    deadline = time.time() + time_limit
    
    visited = set()
    visited.add(state._hash)
    
    my_pos = state.agent_a if perspective == 'A' else state.agent_b
    op_pos = state.agent_b if perspective == 'A' else state.agent_a
    
    root_acts = get_valid_actions(my_pos, op_pos, state.boxes, board)
    if banned_actions:
        root_acts = [a for a in root_acts if a not in banned_actions]
    if not root_acts:
        root_acts = [Action.NORTH]
    root_action_best_val = {act: -float('inf') for act in root_acts}
    root_action_revisits = {act: False for act in root_acts}

    # Seed one frontier per root action. Keeping roots separate prevents a
    # temporarily unattractive but strategically necessary route from being
    # starved by a better-looking WAIT branch.
    frontiers = {act: [] for act in root_acts}
    for act in root_acts:
        val, _, ns = _robust_successor(
            state, act, board, perspective, max_steps, heuristic_cache
        )

        new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
        root_action_revisits[act] = (
            new_my_pos != my_pos and new_my_pos in recent_positions
        )
        
        if val > root_action_best_val[act]:
            root_action_best_val[act] = val
        frontiers[act].append((val, ns))
        visited.add(ns._hash)

    nodes_expanded = 0
    for _ in range(1, SEARCH_DEPTH):
        if time.time() >= deadline:
            break

        next_frontiers = {act: [] for act in root_acts}
        for first_act in root_acts:
            candidates = []
            for _, curr in frontiers[first_act]:
                if curr.is_terminal(max_steps):
                    continue

                nodes_expanded += 1
                curr_my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
                curr_op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
                for next_action in get_valid_actions(
                    curr_my_pos, curr_op_pos, curr.boxes, board
                ):
                    n_val, _, ns = _robust_successor(
                        curr,
                        next_action,
                        board,
                        perspective,
                        max_steps,
                        heuristic_cache,
                    )
                    if ns._hash in visited:
                        continue
                    visited.add(ns._hash)
                    candidates.append((n_val, ns))
                    if n_val > root_action_best_val[first_act]:
                        root_action_best_val[first_act] = n_val

            candidates.sort(key=lambda candidate: candidate[0], reverse=True)
            next_frontiers[first_act] = candidates[:BEAM_WIDTH]

        frontiers = next_frontiers
        if not any(frontiers.values()):
            break

    non_revisiting_actions = [
        act for act in root_acts if not root_action_revisits[act]
    ]
    eligible_actions = non_revisiting_actions or root_acts
    return max(eligible_actions, key=root_action_best_val.get)


class AgentA:
    """
    Agent A controller (Unified).
    """
    def __init__(self):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=4)
        self.tt: Dict[int, Tuple[int, float, Action]] = {}
        self.heuristic_cache: Dict[int, float] = {}
        self._last_action: Optional[Action] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions: List[Action] = None
    ) -> Action:
        
        if len(self.heuristic_cache) > 500000:
            self.heuristic_cache.clear()
            
        auto_banned = list(banned_actions) if banned_actions else []
            
        action = best_action(
            state, board, max_steps,
            perspective='A',
            recent_positions=self._history,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=auto_banned
        )
        
        self._history.append(state.agent_a)
        self._last_pos = state.agent_a
        self._last_action = action
        
        return action

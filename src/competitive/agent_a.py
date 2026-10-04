"""
Agent controller — Greedy Best-First Search (GBFS).
"""
import time
from collections import deque
from typing import Optional, Tuple, Deque, Dict, List

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome
from src.competitive.evaluation import competitive_heuristic, W_SCORE

# ── Search Constants ──────────────────────────────────────────────────────────
TIME_LIMIT = 0.90
SEARCH_DEPTH = 6
BEAM_WIDTH = 24
BLOCKED_ACTION_PENALTY = 250.0
TACTICAL_PROGRESS_WEIGHT = 20.0
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


def _pushes_opponent_finished_box(state, action, perspective):
    if action == Action.WAIT:
        return False
    position = state.agent_a if perspective == 'A' else state.agent_b
    opponent_goals = (
        state.boxes_on_goals_b
        if perspective == 'A'
        else state.boxes_on_goals_a
    )
    destination = (
        position[0] + action.value[0],
        position[1] + action.value[1],
    )
    return destination in opponent_goals


def _push_approach_distance(pos, box, board):
    distances = []
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        approach = (box[0] - dx, box[1] - dy)
        push_to = (box[0] + dx, box[1] + dy)
        if approach in board.walls or push_to in board.walls:
            continue
        distances.append(board.dist(pos, approach))
    return min(distances, default=9999)


def _tactical_target(state, board, perspective, max_steps):
    """Choose one concrete box objective for directional action guidance."""
    own_score = state.score_a() if perspective == 'A' else state.score_b()
    opponent_score = state.score_b() if perspective == 'A' else state.score_a()
    own_pos = state.agent_a if perspective == 'A' else state.agent_b
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b

    # When behind or when no neutral box remains, pursue an opponent goal.
    opponent_goals = (
        state.boxes_on_goals_b
        if perspective == 'A'
        else state.boxes_on_goals_a
    )
    if opponent_goals and (opponent_score > own_score or not (state.boxes - occupied)):
        target = min(
            opponent_goals,
            key=lambda box: _push_approach_distance(own_pos, box, board),
        )
        return ('steal', target, None)

    free_goals = board.goals - occupied
    candidates = []
    for box in state.boxes - occupied:
        for goal in free_goals:
            cost = board.exact_steps(box, own_pos, goal)
            if cost < max_steps - state.step:
                candidates.append((cost, box, goal))

    if candidates:
        _, box, goal = min(candidates)
        return ('finish', box, goal)

    if opponent_goals:
        target = min(
            opponent_goals,
            key=lambda box: _push_approach_distance(own_pos, box, board),
        )
        return ('steal', target, None)
    return None


def _tactical_cost(state, board, perspective, target):
    if target is None:
        return 9999
    kind, box, goal = target
    pos = state.agent_a if perspective == 'A' else state.agent_b
    if box not in state.boxes:
        return 9999
    if kind == 'steal':
        return _push_approach_distance(pos, box, board)
    return board.exact_steps(box, pos, goal)


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
    target = _tactical_target(curr, board, perspective, max_steps)
    target_cost_before = _tactical_cost(curr, board, perspective, target)
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
        target_cost_after = _tactical_cost(
            next_state, board, perspective, target
        )
        if target_cost_before < 9999 and target_cost_after < 9999:
            effective_value += TACTICAL_PROGRESS_WEIGHT * (
                target_cost_before - target_cost_after
            )

        opponent_credits_before = (
            curr.boxes_on_goals_b
            if perspective == 'A'
            else curr.boxes_on_goals_a
        )
        opponent_credits_after = (
            next_state.boxes_on_goals_b
            if perspective == 'A'
            else next_state.boxes_on_goals_a
        )
        stolen_count = len(opponent_credits_before - opponent_credits_after)
        if stolen_count:
            # Removing an opponent point is strategically valuable even
            # though the transition awards the box only after re-scoring it.
            effective_value += W_SCORE * stolen_count
        if pushes_own_finished_box:
            # The opponent may steal a credited box, but an agent should not
            # voluntarily destroy its own score while pursuing another route.
            effective_value = float('-inf')
        if own_action != Action.WAIT and own_position_after == own_position_before:
            # A blocking response is a bad outcome, but not proof that the
            # strategic route is impossible. Keeping it finite lets an agent
            # contest a defended box instead of freezing on WAIT forever.
            effective_value -= BLOCKED_ACTION_PENALTY
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
    root_acts = [
        action for action in root_acts
        if not _pushes_own_finished_box(state, action, perspective)
    ]
    if banned_actions:
        root_acts = [a for a in root_acts if a not in banned_actions]
    if not root_acts:
        root_acts = [Action.NORTH]
    root_action_best_val = {act: -float('inf') for act in root_acts}
    root_action_revisits = {act: False for act in root_acts}
    root_action_progress = {act: 0 for act in root_acts}
    own_score = state.score_a() if perspective == 'A' else state.score_b()
    opponent_score = state.score_b() if perspective == 'A' else state.score_a()

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
        target = _tactical_target(state, board, perspective, max_steps)
        before_cost = _tactical_cost(state, board, perspective, target)
        after_cost = _tactical_cost(ns, board, perspective, target)
        if before_cost < 9999 and after_cost < 9999:
            root_action_progress[act] = before_cost - after_cost
        if (
            opponent_score >= own_score
            and _pushes_opponent_finished_box(state, act, perspective)
        ):
            # A legal immediate capture is the clearest possible response to
            # a tied or losing scoreboard. Prefer taking the point off the
            # opponent now; the following search step plans the re-score.
            root_action_progress[act] = max(
                root_action_progress[act], 10000
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
                    if _pushes_own_finished_box(curr, next_action, perspective):
                        continue
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
    best_progress = max(root_action_progress[act] for act in eligible_actions)
    progressing_actions = [
        act for act in eligible_actions
        if root_action_progress[act] == best_progress
    ]
    return max(progressing_actions, key=root_action_best_val.get)


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

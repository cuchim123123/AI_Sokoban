"""
Agent controller — Greedy Best-First Search (GBFS).
"""
import time
from collections import deque
from typing import Optional, Tuple, Deque, Dict, List
from scipy.optimize import linear_sum_assignment

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome
from src.competitive.evaluation import (
    _deadlock_count,
    competitive_heuristic,
    W_SCORE,
)

# ── Search Constants ──────────────────────────────────────────────────────────
TIME_LIMIT = 0.90
SEARCH_DEPTH = 6
MAX_SEARCH_DEPTH = 3
BEAM_WIDTH = 32
BLOCKED_ACTION_PENALTY = 250.0
TACTICAL_PROGRESS_WEIGHT = 20.0
IMMEDIATE_FINISH_PRIORITY = 9000
IMMEDIATE_STEAL_PRIORITY = 10000
GUARD_DRIFT_PENALTY = 100.0
STEAL_COST_MARGIN = 0
ENDGAME_FINISH_STEPS = 20
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


def _same_recent_position_and_board(
    state: CompetitiveState,
    recent_positions: Deque,
    perspective: str,
) -> bool:
    """Detect a true revisit without rejecting progress on a changed board."""
    position = state.agent_a if perspective == 'A' else state.agent_b
    for entry in recent_positions:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], tuple)
            and isinstance(entry[1], int)
        ):
            if entry[0] == position and entry[1] == state.board_hash:
                return True
        elif entry == position:
            # Keep compatibility with callers that provide position-only history.
            return True
    return False


def _reachable_region(
    start: Tuple[int, int],
    board: Board,
    boxes,
) -> set:
    if start not in board.floor_cells:
        return set()
    region = {start}
    pending = deque([start])
    while pending:
        position = pending.popleft()
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            next_position = (position[0] + dx, position[1] + dy)
            if (
                next_position not in board.floor_cells
                or next_position in boxes
            ):
                continue
            if next_position not in region:
                region.add(next_position)
                pending.append(next_position)
    return region


def _agents_are_independent(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    horizon: int = SEARCH_DEPTH,
) -> bool:
    """Use static opponent response when no interaction fits the search horizon."""
    own_position = state.agent_a if perspective == 'A' else state.agent_b
    opponent_position = state.agent_b if perspective == 'A' else state.agent_a
    if (
        own_position not in board.floor_cells
        or opponent_position not in board.floor_cells
    ):
        return False
    if board.dist(own_position, opponent_position) <= horizon:
        return False

    for box in state.boxes:
        own_push_distance = 9999
        opponent_push_distance = 9999
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            approach = (box[0] - dx, box[1] - dy)
            destination = (box[0] + dx, box[1] + dy)
            if (
                destination not in board.walls
                and destination not in state.boxes
            ):
                own_push_distance = min(
                    own_push_distance,
                    board.dist(own_position, approach),
                )
                opponent_push_distance = min(
                    opponent_push_distance,
                    board.dist(opponent_position, approach),
                )
        if (
            own_push_distance <= horizon
            and opponent_push_distance <= horizon
        ):
            return False
        if (
            own_push_distance <= horizon
            and board.dist(own_position, opponent_position) <= horizon + 2
        ):
            return False
        if (
            opponent_push_distance <= horizon
            and board.dist(own_position, opponent_position) <= horizon + 2
        ):
            return False
    return True


def _assignment_capacity(
    boxes,
    goals,
    board: Board,
) -> int:
    """Count how many remaining boxes can still receive distinct goals."""
    if not boxes:
        return 0
    if not goals:
        return 0
    box_list = list(boxes)
    goal_list = list(goals)
    costs = [
        [board.push_dist(box, goal) for goal in goal_list]
        for box in box_list
    ]
    rows, columns = linear_sum_assignment(costs)
    return sum(
        costs[row][column] < 9999
        for row, column in zip(rows, columns)
    )


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


def _steal_preserves_goal_access(
    state: CompetitiveState,
    action: Action,
    board: Board,
    perspective: str,
) -> bool:
    if not _pushes_opponent_finished_box(state, action, perspective):
        return False
    position = state.agent_a if perspective == 'A' else state.agent_b
    box = (
        position[0] + action.value[0],
        position[1] + action.value[1],
    )
    pushed_to = (
        box[0] + action.value[0],
        box[1] + action.value[1],
    )
    return any(
        board.push_dist(pushed_to, goal) < 9999
        for goal in board.goals
    )


def _finishes_neutral_box(
    state: CompetitiveState,
    action: Action,
    board: Board,
    perspective: str,
) -> bool:
    if action == Action.WAIT:
        return False
    position = state.agent_a if perspective == 'A' else state.agent_b
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    box = (
        position[0] + action.value[0],
        position[1] + action.value[1],
    )
    destination = (
        box[0] + action.value[0],
        box[1] + action.value[1],
    )
    return (
        box in state.boxes
        and box not in occupied
        and destination in board.goals
        and destination not in occupied
    )


def _is_endgame_finish(
    state: CompetitiveState,
    action: Action,
    board: Board,
    perspective: str,
    max_steps: int,
) -> bool:
    return (
        max_steps - state.step <= ENDGAME_FINISH_STEPS
        and _finishes_neutral_box(state, action, board, perspective)
    )


def _creates_deadlock(
    state: CompetitiveState,
    action: Action,
    board: Board,
    perspective: str,
    max_steps: int,
) -> bool:
    """Reject a push that increases the currently detectable deadlock count."""
    if action == Action.WAIT:
        return False
    position = state.agent_a if perspective == 'A' else state.agent_b
    destination = (
        position[0] + action.value[0],
        position[1] + action.value[1],
    )
    if destination not in state.boxes:
        return False

    action_a, action_b = _joint_action(
        state, perspective, action, Action.WAIT
    )
    next_state = resolve_joint_action_outcome(
        state, action_a, action_b, board, max_steps
    ).state
    pushed_to = (
        destination[0] + action.value[0],
        destination[1] + action.value[1],
    )
    if pushed_to not in board.goals:
        blocked_x = (
            (pushed_to[0] - 1, pushed_to[1]) in board.walls
            or (pushed_to[0] + 1, pushed_to[1]) in board.walls
        )
        blocked_y = (
            (pushed_to[0], pushed_to[1] - 1) in board.walls
            or (pushed_to[0], pushed_to[1] + 1) in board.walls
        )
        if blocked_x and blocked_y:
            return True
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    next_occupied = (
        next_state.boxes_on_goals_a | next_state.boxes_on_goals_b
    )
    return _deadlock_count(
        next_state.boxes, board, next_occupied
    ) > _deadlock_count(state.boxes, board, occupied)


def _push_approach_distance(pos, box, board):
    distances = []
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        approach = (box[0] - dx, box[1] - dy)
        push_to = (box[0] + dx, box[1] + dy)
        if approach in board.walls or push_to in board.walls:
            continue
        distances.append(board.dist(pos, approach))
    return min(distances, default=9999)


def _state_steal_cost(state, box, board, perspective, max_steps):
    own_pos = state.agent_a if perspective == 'A' else state.agent_b
    opponent_pos = state.agent_b if perspective == 'A' else state.agent_a
    region = _reachable_region(
        own_pos,
        board,
        state.boxes | {opponent_pos},
    )
    distances = []
    priority = 'A' if (max_steps - state.step) % 2 else 'B'
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        approach = (box[0] - dx, box[1] - dy)
        push_to = (box[0] + dx, box[1] + dy)
        if approach not in region:
            continue
        if push_to in board.walls or push_to in state.boxes:
            continue
        if (
            priority != perspective
            and board.dist(opponent_pos, push_to) <= 1
        ):
            continue
        distances.append(board.dist(own_pos, approach))
    return min(distances, default=9999)


def _tactical_target(state, board, perspective, max_steps):
    """Choose one concrete box objective for directional action guidance."""
    own_score = state.score_a() if perspective == 'A' else state.score_b()
    opponent_score = state.score_b() if perspective == 'A' else state.score_a()
    own_pos = state.agent_a if perspective == 'A' else state.agent_b
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b

    # A nearby opponent goal is a hot-zone opportunity even when scores are tied.
    opponent_goals = (
        state.boxes_on_goals_b
        if perspective == 'A'
        else state.boxes_on_goals_a
    )
    steal_target = None
    steal_cost = 9999
    if opponent_goals:
        steal_target = min(
            opponent_goals,
            key=lambda box: _state_steal_cost(
                state, box, board, perspective, max_steps
            ),
        )
        steal_cost = _state_steal_cost(
            state, steal_target, board, perspective, max_steps
        )

    free_goals = board.goals - occupied
    candidates = []
    for box in state.boxes - occupied:
        for goal in free_goals:
            cost = board.exact_steps(box, own_pos, goal)
            remaining_boxes = (state.boxes - occupied) - {box}
            remaining_goals = free_goals - {goal}
            if cost < max_steps - state.step:
                candidates.append((
                    _assignment_capacity(
                        remaining_boxes, remaining_goals, board
                    ),
                    cost,
                    box,
                    goal,
                ))

    if candidates:
        _, best_neutral_cost, box, goal = min(
            candidates,
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2], candidate[3]),
        )
        if (
            steal_target is not None
            and steal_cost < 9999
            and (
                steal_cost <= best_neutral_cost + STEAL_COST_MARGIN
                and opponent_score >= own_score
            )
        ):
            return ('steal', steal_target, None)
        return ('finish', box, goal)

    if steal_target is not None:
        return ('steal', steal_target, None)
    own_goals = (
        state.boxes_on_goals_a
        if perspective == 'A'
        else state.boxes_on_goals_b
    )
    if own_goals and not opponent_goals:
        opponent_pos = state.agent_b if perspective == 'A' else state.agent_a
        target = min(
            own_goals,
            key=lambda box: board.dist(own_pos, box),
        )
        guard_cells = []
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            cell = (target[0] + dx, target[1] + dy)
            if cell not in board.floor_cells or cell in state.boxes:
                continue
            guard_cells.append(cell)
        if guard_cells:
            guard_cell = min(
                guard_cells,
                key=lambda cell: (
                    board.dist(own_pos, cell) - board.dist(opponent_pos, cell),
                    board.dist(own_pos, cell),
                ),
            )
            return ('guard', target, guard_cell)
        return ('guard', target, None)
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
    if kind == 'guard':
        return board.dist(pos, goal) if goal is not None else board.dist(pos, box)
    return board.exact_steps(box, pos, goal)


def _opponent_actions(
    op_pos,
    my_pos,
    boxes,
    board: Board,
    action_cache: Optional[Dict[tuple, List[Action]]] = None,
    include_wait: bool = True,
) -> List[Action]:
    if action_cache is not None:
        cache_key = (op_pos, my_pos, boxes)
        if cache_key in action_cache:
            return action_cache[cache_key]
    actions = get_valid_actions(
        op_pos, my_pos, boxes, board, include_wait=include_wait
    )
    if action_cache is not None:
        action_cache[cache_key] = actions
    return actions


def _robust_successor(
    curr: CompetitiveState,
    own_action: Action,
    board: Board,
    perspective: str,
    max_steps: int,
    heuristic_cache: Dict[tuple, float],
    independence_cache: Optional[Dict[tuple, bool]] = None,
    action_cache: Optional[Dict[tuple, List[Action]]] = None,
    tactical_cache: Optional[Dict[tuple, object]] = None,
    result_cache: Optional[Dict[tuple, tuple]] = None,
):
    """Return the worst legal opponent response to one own action."""
    result_key = (curr._hash, own_action, perspective, max_steps)
    if result_cache is not None and result_key in result_cache:
        return result_cache[result_key]
    my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
    op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
    independence_key = (curr._hash, perspective)
    if independence_cache is not None and independence_key in independence_cache:
        independent = independence_cache[independence_key]
    else:
        independent = _agents_are_independent(
            curr, board, perspective, SEARCH_DEPTH
        )
        if independence_cache is not None:
            independence_cache[independence_key] = independent
    if independent:
        opponent_actions = [Action.WAIT]
    else:
        opponent_actions = _opponent_actions(
            op_pos, my_pos, curr.boxes, board, action_cache, include_wait=True
        )
    if not opponent_actions:
        opponent_actions = [Action.WAIT]
    candidates = []
    target_key = (curr._hash, perspective, max_steps)
    if tactical_cache is not None and target_key in tactical_cache:
        target = tactical_cache[target_key]
    else:
        target = _tactical_target(curr, board, perspective, max_steps)
        if tactical_cache is not None:
            tactical_cache[target_key] = target
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
        if target is not None and target[0] == 'guard':
            if target_cost_after > target_cost_before:
                effective_value -= GUARD_DRIFT_PENALTY * (
                    target_cost_after - target_cost_before
                )
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
    result = min(candidates, key=lambda candidate: (candidate[0], candidate[1].value))
    if result_cache is not None:
        result_cache[result_key] = result
    return result


def best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str,
    recent_positions: Deque[Tuple[int, int]],
    tt: Dict[int, Tuple[int, float, Action]],
    heuristic_cache: Dict[tuple, float],
    time_limit: float = TIME_LIMIT,
    banned_actions: List[Action] = None,
    action_cache: Optional[Dict[tuple, List[Action]]] = None,
    tactical_cache: Optional[Dict[tuple, object]] = None,
    result_cache: Optional[Dict[tuple, tuple]] = None,
) -> Action:
    deadline = time.time() + time_limit

    visited = set()
    visited.add(state._hash)

    my_pos = state.agent_a if perspective == 'A' else state.agent_b
    op_pos = state.agent_b if perspective == 'A' else state.agent_a

    if action_cache is None:
        action_cache: Dict[tuple, List[Action]] = {}
    if tactical_cache is None:
        tactical_cache: Dict[tuple, object] = {}
    if result_cache is None:
        result_cache: Dict[tuple, tuple] = {}

    root_acts = get_valid_actions(my_pos, op_pos, state.boxes, board)
    root_acts = [
        action for action in root_acts
        if not _pushes_own_finished_box(state, action, perspective)
        and (
            (
                _pushes_opponent_finished_box(state, action, perspective)
                and _steal_preserves_goal_access(
                    state, action, board, perspective
                )
            )
            or (
                not _pushes_opponent_finished_box(
                    state, action, perspective
                )
                and (
                    _is_endgame_finish(
                        state, action, board, perspective, max_steps
                    )
                    or not _creates_deadlock(
                        state, action, board, perspective, max_steps
                    )
                )
            )
        )
    ]
    if banned_actions:
        root_acts = [a for a in root_acts if a not in banned_actions]
    if not root_acts:
        root_acts = _opponent_actions(my_pos, op_pos, state.boxes, board, action_cache, include_wait=True)
    root_action_best_val = {act: -float('inf') for act in root_acts}
    root_action_initial_val = {act: -float('inf') for act in root_acts}
    root_action_revisits = {act: False for act in root_acts}
    root_action_progress = {act: 0 for act in root_acts}
    own_score = state.score_a() if perspective == 'A' else state.score_b()
    opponent_score = state.score_b() if perspective == 'A' else state.score_a()

    best_choice = None
    best_choice_value = -float('inf')
    independence_cache: Dict[tuple, bool] = {}

    root_target = _tactical_target(state, board, perspective, max_steps)
    root_before_cost = _tactical_cost(state, board, perspective, root_target)

    independent = _agents_are_independent(
        state, board, perspective, SEARCH_DEPTH
    )
    independence_cache[(state._hash, perspective)] = independent

    if independent and root_target is not None:
        best_act = None
        best_val = -float('inf')
        for act in root_acts:
            val, _, ns = _robust_successor(
                state,
                act,
                board,
                perspective,
                max_steps,
                heuristic_cache,
                independence_cache,
                action_cache,
                tactical_cache,
                result_cache,
            )
            after_cost = _tactical_cost(ns, board, perspective, root_target)
            tactical_bonus = 0.0
            if root_before_cost < 9999 and after_cost < 9999:
                tactical_bonus = TACTICAL_PROGRESS_WEIGHT * (
                    root_before_cost - after_cost
                )
            total_val = val + tactical_bonus
            if total_val > best_val:
                best_val = total_val
                best_act = act
        if best_act is not None:
            return best_act

    for depth_limit in range(1, MAX_SEARCH_DEPTH + 1):
        if time.time() >= deadline:
            break

        depth_completed = True

        visited = {state._hash}
        root_action_best_val = {act: -float('inf') for act in root_acts}
        root_action_revisits = {act: False for act in root_acts}
        root_action_progress = {act: 0 for act in root_acts}
        root_action_initial_val = {act: -float('inf') for act in root_acts}

        frontiers = {act: [] for act in root_acts}
        for act in root_acts:
            val, _, ns = _robust_successor(
                state,
                act,
                board,
                perspective,
                max_steps,
                heuristic_cache,
                independence_cache,
                action_cache,
                tactical_cache,
                result_cache,
            )
            root_action_initial_val[act] = val
            root_action_best_val[act] = max(root_action_best_val[act], val)
            best_choice = act if best_choice is None else best_choice
            frontiers[act].append((val, ns))
            visited.add(ns._hash)

            new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
            root_action_revisits[act] = (
                new_my_pos != my_pos
                and _same_recent_position_and_board(
                    ns,
                    recent_positions,
                    perspective,
                )
            )
            after_cost = _tactical_cost(ns, board, perspective, root_target)
            if root_before_cost < 9999 and after_cost < 9999:
                root_action_progress[act] = root_before_cost - after_cost
            if (
                opponent_score >= own_score
                and _steal_preserves_goal_access(
                    state, act, board, perspective
                )
            ):
                root_action_progress[act] = max(
                    root_action_progress[act], IMMEDIATE_STEAL_PRIORITY
                )
            if _finishes_neutral_box(state, act, board, perspective):
                root_action_progress[act] = max(
                    root_action_progress[act], IMMEDIATE_FINISH_PRIORITY
                )

        for _ in range(1, depth_limit):
            if time.time() >= deadline:
                depth_completed = False
                break

            next_frontiers = {act: [] for act in root_acts}
            for first_act in root_acts:
                candidates = []
                for _, curr in frontiers[first_act]:
                    if curr.is_terminal(max_steps):
                        continue

                    curr_my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
                    curr_op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
                    for next_action in _opponent_actions(
                        curr_my_pos, curr_op_pos, curr.boxes, board, action_cache, include_wait=False
                    ):
                        if _pushes_own_finished_box(curr, next_action, perspective):
                            continue
                        if (
                            (
                                not _pushes_opponent_finished_box(
                                    curr, next_action, perspective
                                )
                                and not _is_endgame_finish(
                                    curr, next_action, board, perspective, max_steps
                                )
                                and _creates_deadlock(
                                    curr, next_action, board, perspective, max_steps
                                )
                            )
                            or (
                                _pushes_opponent_finished_box(
                                    curr, next_action, perspective
                                )
                                and not _steal_preserves_goal_access(
                                    curr, next_action, board, perspective
                                )
                            )
                        ):
                            continue
                        n_val, _, ns = _robust_successor(
                            curr,
                            next_action,
                            board,
                            perspective,
                            max_steps,
                            heuristic_cache,
                            independence_cache,
                            action_cache,
                            tactical_cache,
                            result_cache,
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

        if time.time() >= deadline:
            depth_completed = False
        if not depth_completed:
            break

        best_overall_progress = max(root_action_progress.values())
        if best_overall_progress > 0:
            eligible_actions = [
                act for act in root_acts
                if root_action_progress[act] == best_overall_progress
            ]
        else:
            non_revisiting_actions = [
                act for act in root_acts if not root_action_revisits[act]
            ]
            eligible_actions = non_revisiting_actions or root_acts

        if not eligible_actions:
            eligible_actions = root_acts

        best_progress = max(root_action_progress[act] for act in eligible_actions)
        progressing_actions = [
            act for act in eligible_actions
            if root_action_progress[act] == best_progress
        ]
        candidate = max(
            progressing_actions,
            key=lambda act: (
                root_action_initial_val[act],
                root_action_best_val[act],
            ),
        )
        candidate_value = root_action_best_val[candidate]
        best_choice = candidate
        best_choice_value = candidate_value

    if best_choice is not None:
        return best_choice

    # Conservative fallback: the initial immediate root values are already
    # filtered and protected against self-damaging moves.
    for act in root_acts:
        val, _, _ = _robust_successor(
            state, act, board, perspective, max_steps, heuristic_cache, action_cache=action_cache, tactical_cache=tactical_cache, result_cache=result_cache
        )
        if val > best_choice_value:
            best_choice = act
            best_choice_value = val
    return best_choice if best_choice is not None else root_acts[0]


class AgentA:
    """
    Agent A controller (Unified).
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
        banned_actions: List[Action] = None
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
            perspective='A',
            recent_positions=self._history,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=auto_banned,
            action_cache=self.action_cache,
            tactical_cache=self.tactical_cache,
            result_cache=self.result_cache,
        )
        
        self._history.append((state.agent_a, state.board_hash))
        self._last_pos = state.agent_a
        self._last_action = action
        
        return action

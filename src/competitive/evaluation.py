"""Score-unit evaluation for simultaneous competitive Sokoban.

Only the best next delivery opportunity per side is estimated: independent
single-box costs are NOT added up as a fictitious multi-delivery schedule.
Distances respect walls and current box blockage up to the first push; later
pushes are relaxed. Real conflicts, removal and ownership transfer belong to
search. The bounded approach term is guidance, never proof of a future steal.
"""
from collections import deque
from functools import lru_cache
from typing import FrozenSet, Optional, Tuple
from src.competitive.state import Action, Board, CompetitiveState

INF = 9999
W_LOCKED = 1.0
Pos = Tuple[int, int]
_DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))
_eval_cache = {}
_CACHE_LIMIT = 40000


def clear_cache():
    _eval_cache.clear()
    walking_distances.cache_clear()
    _delivery_min.cache_clear()


@lru_cache(maxsize=8192)
def walking_distances(pos, boxes, board):
    """Current boxes block walking; a moving opponent is not a static wall."""
    distances = {pos: 0}
    queue = deque([pos])
    while queue:
        cell = queue.popleft()
        distance = distances[cell] + 1
        for nxt in board.neighbors.get(cell, ()):
            if nxt not in boxes and nxt not in distances:
                distances[nxt] = distance
                queue.append(nxt)
    return distances


def delivery_cost(box, goal, walks, boxes, board):
    """Reach a real first push, then use the relaxed single-box table.

    An uncredited box already on a goal must leave and be delivered again;
    simply standing beside it never awards a point.
    """
    best = INF
    for dx, dy in _DIRS:
        approach = (box[0] - dx, box[1] - dy)
        dest = (box[0] + dx, box[1] + dy)
        if approach not in walks or dest not in board.floor_cells or dest in boxes:
            continue
        tail = 0 if dest == goal else board.exact_steps(dest, box, goal)
        best = min(best, walks[approach] + 1 + tail)
    return best


@lru_cache(maxsize=16384)
def _delivery_min(pos, box, goals, boxes, board):
    walks = walking_distances(pos, boxes, board)
    return min((delivery_cost(box, g, walks, boxes, board) for g in goals), default=INF)


def strike_distance(pos, box, boxes, board):
    walks = walking_distances(pos, boxes, board)
    return min((walks.get((box[0]-dx, box[1]-dy), INF)
                for dx, dy in _DIRS
                if (box[0]+dx, box[1]+dy) in board.floor_cells
                and (box[0]+dx, box[1]+dy) not in boxes), default=INF)


def evaluation_components(state, board, max_steps):
    """A-perspective components, in points; terminal values are exact."""
    credit = state.score_a() - state.score_b()
    remaining = max_steps - state.step
    if remaining <= 0:
        return dict(credit=float(credit), opportunity=0.0, pressure=0.0)
    wa = walking_distances(state.agent_a, state.boxes, board)
    wb = walking_distances(state.agent_b, state.boxes, board)
    credited = state.boxes_on_goals_a | state.boxes_on_goals_b
    goals = board.goals - credited
    best_a = best_b = 0.0
    # Each box competes for the same available goals on both sides. Taking
    # only one next opportunity per side avoids summing incompatible plans.
    for box in sorted(state.boxes - credited):
        ca = _delivery_min(state.agent_a, box, goals, state.boxes, board)
        cb = _delivery_min(state.agent_b, box, goals, state.boxes, board)
        va = 0.65 / (1 + 0.12 * ca) if ca <= remaining else 0.0
        vb = 0.65 / (1 + 0.12 * cb) if cb <= remaining else 0.0
        # A nearby rival discounts the opportunity, never changes legality.
        if ca <= remaining and cb <= remaining:
            va *= 0.5 + 0.5 * max(-1.0, min(1.0, (cb-ca)/4))
            vb *= 0.5 + 0.5 * max(-1.0, min(1.0, (ca-cb)/4))
        best_a = max(best_a, va)
        best_b = max(best_b, vb)

    def pressure(walks, targets):
        best = 0.0
        for box in targets:
            for dx, dy in _DIRS:
                dest = (box[0]+dx, box[1]+dy)
                d = walks.get((box[0]-dx, box[1]-dy), INF) + 1
                if dest in board.floor_cells and dest not in state.boxes and d <= remaining:
                    best = max(best, 0.10 / d)
        return best

    return dict(credit=float(credit), opportunity=best_a-best_b,
                pressure=pressure(wa, state.boxes_on_goals_b)
                         - pressure(wb, state.boxes_on_goals_a))


def evaluate(state, board, perspective, max_steps, cache=None):
    store = _eval_cache if cache is None else cache
    key = (board.serial, state, max_steps)
    if key not in store:
        value = sum(evaluation_components(state, board, max_steps).values())
        if len(store) >= _CACHE_LIMIT:
            store.clear()
        store[key] = value
    raw = store[key]
    return raw if perspective == "A" else -raw


competitive_heuristic = evaluate
projected_score = evaluate  # historical API; an estimate, not a delivery count


def nearest_objective(state, board, perspective):
    pos = state.agent_a if perspective == "A" else state.agent_b
    own = state.boxes_on_goals_a if perspective == "A" else state.boxes_on_goals_b
    return min(sorted(state.boxes - own), key=lambda b: board.dist(pos, b), default=None)


def has_legal_push(
    box: Pos,
    boxes: FrozenSet[Pos],
    board: Board,
) -> bool:
    """Whether `box` has any locally valid push direction."""
    for dx, dy in _DIRS:
        approach = (box[0] - dx, box[1] - dy)
        destination = (box[0] + dx, box[1] + dy)
        if approach not in board.floor_cells or approach in boxes:
            continue
        if destination not in board.floor_cells or destination in boxes:
            continue
        return True
    return False


def deadlock_count(
    boxes: FrozenSet[Pos],
    board: Board,
    occupied_goals: FrozenSet[Pos] = frozenset(),
) -> int:
    """
    Number of boxes that can never reach a free goal again:
    frozen in place locally, statically unreachable, or locked into a 2-box
    wall pair. Boxes already sitting on a goal are never counted.
    """
    available_goals = board.goals - occupied_goals
    count = 0

    for box in boxes:
        if box in occupied_goals:
            continue

        if not has_legal_push(box, boxes, board):
            count += 1
            continue

        if available_goals and all(
            board.push_dist(box, goal) >= INF for goal in available_goals
        ):
            count += 1
            continue
        if not available_goals:
            continue

        bx, by = box
        if box in board.goals:
            continue

        # Two adjacent boxes jammed against a continuous wall.
        if (bx + 1, by) in boxes and (bx + 1, by) not in board.goals:
            if ((bx, by + 1) in board.walls and (bx + 1, by + 1) in board.walls) or (
                (bx, by - 1) in board.walls and (bx + 1, by - 1) in board.walls
            ):
                count += 1
                continue
        if (bx, by + 1) in boxes and (bx, by + 1) not in board.goals:
            if ((bx + 1, by) in board.walls and (bx + 1, by + 1) in board.walls) or (
                (bx - 1, by) in board.walls and (bx - 1, by + 1) in board.walls
            ):
                count += 1
                continue

    return count


def creates_deadlock(pos: Pos, action: Action, boxes, board: Board) -> bool:
    """True when an agent at `pos` performing `action` pushes a box into a
    position it can never leave."""
    if action is Action.WAIT:
        return False
    dest = (pos[0] + action.value[0], pos[1] + action.value[1])
    if dest not in boxes:
        return False
    new_cell = (dest[0] + action.value[0], dest[1] + action.value[1])
    if new_cell in board.goals:
        return False

    new_boxes = (boxes - {dest}) | {new_cell}
    if not has_legal_push(new_cell, new_boxes, board):
        return True
    return all(board.push_dist(new_cell, goal) >= INF for goal in board.goals)

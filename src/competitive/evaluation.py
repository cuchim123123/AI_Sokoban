"""
Evaluation for the competitive 2-agent Sokoban.

Perspective / zero-sum guarantee
--------------------------------
`_value_a` computes every term as "agent A minus agent B". `evaluate` returns
that number for perspective "A" and its negation for "B", so

    evaluate(state, board, "B", ...) == -evaluate(state, board, "A", ...)

holds exactly. That makes the maximin search in `agent_a` a correct security
strategy for a zero-sum game.

Terms (each measures exactly one thing, no double counting)
-----------------------------------------------------------
1. projected score  - competitive race assignment: every unclaimed box is
                      assigned to the agent who can actually deliver it sooner
                      (per-agent greedy matching over the precomputed exact
                      walk+push costs). A tied race splits the box. Credited
                      boxes are worth a full locked point; a won race is worth
                      a discounted projected point, so cashing a box in on a
                      goal is always an improvement over merely chasing it -
                      the agent must not defer deliveries forever. This is
                      the term that tells the search WHO IS WINNING each box.
2. race gradient    - per box, how much cheaper the delivery is for A than
                      for B (clamped), so best-first search has a smooth
                      climbing signal toward the boxes it is winning and
                      toward opponent boxes it can overtake.
3. steal races      - advantage in reaching an opponent-credited box first,
                      measured as attacker distance vs defender distance to
                      the push approach cells. Credited boxes only; unclaimed
                      boxes are covered by the race assignment above.

Caching
-------
`evaluate` memoizes on (positions, boxes, credits, remaining steps) in a
module level cache shared by the search, by the root move ordering and by the
transition function's conflict-diversion choice. The cache is the reason the
agent can think deep inside the time budget.
"""

from typing import Dict, FrozenSet, Optional, Tuple

from src.competitive.state import Action, Board, CompetitiveState


INF = 9999

W_LOCKED = 1000.0      # one CREDITED point (already on a goal)
W_PROJECTED = 600.0    # one projected point (won the cost race, not cashed in)
W_STEAL = 1200.0       # winning the approach race to an opponent's box
W_ADV = 6.0            # weight of the per-box delivery-cost advantage
ADV_CAP = 25.0         # advantage clamp per box (stays well below 1 point)

PROGRESS_BASE = 200.0  # progress gradient saturates past this many steps
STEAL_RANGE = 40.0     # approach distance over which a steal stays attractive

_CACHE_LIMIT = 400_000
_eval_cache: Dict[tuple, float] = {}

_DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))

# Set membership helpers reused by several functions (frozensets hash fast).
Pos = Tuple[int, int]


def clear_cache() -> None:
    """Drop the shared evaluation cache (used by tests)."""
    _eval_cache.clear()


# ── Term 1 + 2: projected score and push progress ────────────────────────────

def _agent_plan(
    pos: Pos,
    free_boxes,
    free_goals,
    board: Board,
    remaining: int,
) -> Tuple[int, float]:
    """
    Greedy box -> goal matching for one agent.

    Returns (finishable, progress):
      finishable - number of distinct boxes that can be delivered to distinct
                   free goals within `remaining` steps,
      progress   - sum of (PROGRESS_BASE - exact cost) over the matched pairs,
                   the gradient that points at the boxes worth chasing.
    """
    if not free_boxes or not free_goals:
        return 0, 0.0

    px, py = pos
    pairs = []
    for box in free_boxes:
        bx, by = box
        for goal in free_goals:
            cost = board.exact_step_costs.get(goal, {}).get((bx, by, px, py), INF)
            if cost < INF:
                pairs.append((cost, box, goal))
    if not pairs:
        return 0, 0.0

    # Cheapest deliveries first.
    pairs.sort()

    used_boxes = set()
    used_goals = set()
    finishable = 0
    progress = 0.0
    for cost, box, goal in pairs:
        if box in used_boxes or goal in used_goals:
            continue
        used_boxes.add(box)
        used_goals.add(goal)
        if cost <= remaining:
            finishable += 1
        if cost < PROGRESS_BASE:
            progress += PROGRESS_BASE - cost
    return finishable, progress


# ── Term 3: steal races ──────────────────────────────────────────────────────

def _steal_potential(
    attacker: Pos,
    defender: Pos,
    target_boxes,
    boxes,
    board: Board,
    remaining: int,
) -> float:
    """
    Attacker's advantage in reaching `target_boxes` (credited to the defender).

    A cell scores 1.0 when the attacker reaches a push approach cell before
    the defender and still has time to push; otherwise only a small gradient
    keeps the pursuit visible.
    """
    if not target_boxes:
        return 0.0

    total = 0.0
    for box in target_boxes:
        bx, by = box
        best_att = INF
        best_def = INF
        for dx, dy in _DIRS:
            approach = (bx - dx, by - dy)
            push_to = (bx + dx, by + dy)
            if approach in board.walls or push_to in board.walls:
                continue
            if approach in boxes or push_to in boxes:
                continue
            da = board.dist(attacker, approach)
            dd = board.dist(defender, approach)
            if da < best_att:
                best_att = da
            if dd < best_def:
                best_def = dd

        # Standing on the box body-blocks every push direction.
        dd_box = board.dist(defender, box)
        if dd_box < best_def:
            best_def = dd_box

        if best_att >= INF:
            continue

        gradient = max(0.0, STEAL_RANGE - best_att) / STEAL_RANGE
        if best_att < best_def and best_att + 1 <= remaining:
            total += 1.0 + 0.5 * gradient
        else:
            total += 0.3 * gradient
    return total


# ── Term 1 + 2: competitive race assignment and cost gradient ─────────────────

def _match_costs(
    pos: Pos,
    free_boxes,
    free_goals,
    board: Board,
) -> Dict[Pos, int]:
    """
    Cheapest-first box->goal matching for one agent.

    Returns {box: exact walk+push cost} for every box the agent can deliver
    to some distinct free goal; unmatched (unreachable) boxes are absent.
    """
    if not free_boxes or not free_goals:
        return {}
    px, py = pos
    pairs = []
    for box in free_boxes:
        bx, by = box
        for goal in free_goals:
            cost = board.exact_step_costs.get(goal, {}).get((bx, by, px, py), INF)
            if cost < INF:
                pairs.append((cost, box, goal))
    pairs.sort()

    used_boxes = set()
    used_goals = set()
    costs: Dict[Pos, int] = {}
    for cost, box, goal in pairs:
        if box in used_boxes or goal in used_goals:
            continue
        used_boxes.add(box)
        used_goals.add(goal)
        costs[box] = cost
    return costs


def _advantage(x: int, y: int, cap: float = ADV_CAP) -> float:
    """Cost advantage of an agent priced at x against one priced at y."""
    if x >= INF and y >= INF:
        return 0.0
    if y >= INF:        # only I can deliver this box at all
        return cap
    if x >= INF:        # only the opponent can
        return -cap
    return max(-cap, min(cap, y - x))


def _race(
    state: CompetitiveState,
    board: Board,
    remaining: int,
) -> Tuple[float, float, float]:
    """
    Competitive projection over the unclaimed boxes.

    Returns (projected_a, projected_b, adv_a - adv_b):
      projected_* - boxes each agent is expected to deliver (credited boxes
                    are counted by the caller), winner of each cost race
                    takes the box, a dead tie splits it in half,
      adv         - summed clamped cost advantage, A positive / B negative.
    """
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    free_boxes = state.boxes - occupied
    if not free_boxes:
        return 0.0, 0.0, 0.0
    free_goals = board.goals - occupied

    costs_a = _match_costs(state.agent_a, free_boxes, free_goals, board)
    costs_b = _match_costs(state.agent_b, free_boxes, free_goals, board)

    projected_a = 0.0
    projected_b = 0.0
    adv = 0.0
    for box in free_boxes:
        x = costs_a.get(box, INF)
        y = costs_b.get(box, INF)
        adv += _advantage(x, y)
        if x < y:
            if x <= remaining:
                projected_a += 1.0
        elif y < x:
            if y <= remaining:
                projected_b += 1.0
        elif x < INF:  # exact tie between reachable deliveries
            if x <= remaining:
                projected_a += 0.5
                projected_b += 0.5
    return projected_a, projected_b, adv



# ── Raw value from agent A's perspective ─────────────────────────────────────

def _value_a(state: CompetitiveState, board: Board, max_steps: int) -> float:
    remaining = max_steps - state.step
    if remaining < 0:
        remaining = 0

    cred_a = state.boxes_on_goals_a
    cred_b = state.boxes_on_goals_b

    if remaining == 0:
        # Out of steps: the score is frozen, nothing can move any more.
        return W_LOCKED * (len(cred_a) - len(cred_b))

    projected_a, projected_b, adv = _race(state, board, remaining)
    value = W_LOCKED * (len(cred_a) - len(cred_b))
    value += W_PROJECTED * (projected_a - projected_b)
    value += W_ADV * adv

    steal_a = _steal_potential(
        state.agent_a, state.agent_b, cred_b, state.boxes, board, remaining
    )
    steal_b = _steal_potential(
        state.agent_b, state.agent_a, cred_a, state.boxes, board, remaining
    )
    value += W_STEAL * (steal_a - steal_b)

    return value


# ── Public evaluation API ────────────────────────────────────────────────────

def evaluate(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    max_steps: int,
    cache: Optional[Dict[tuple, float]] = None,
) -> float:
    """
    Cached evaluation from `perspective` ("A" or "B"). Higher is better.
    The cached value is stored once (from A's perspective) and mirrored, so
    the two perspectives stay exactly zero-sum.
    """
    store = _eval_cache if cache is None else cache
    key = (
        state.agent_a,
        state.agent_b,
        state.boxes,
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
        max_steps - state.step,
    )
    raw = store.get(key)
    if raw is None:
        raw = _value_a(state, board, max_steps)
        if len(store) >= _CACHE_LIMIT:
            store.clear()
        store[key] = raw
    return raw if perspective == "A" else -raw


def competitive_heuristic(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    max_steps: int,
) -> float:
    """Legacy name for `evaluate`."""
    return evaluate(state, board, perspective, max_steps)


def projected_score(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    max_steps: int,
) -> float:
    """Projected final score (credited + realistically finishable boxes)."""
    remaining = max(max_steps - state.step, 0)
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    free_boxes = state.boxes - occupied
    free_goals = board.goals - occupied

    finish_a, _ = _agent_plan(state.agent_a, free_boxes, free_goals, board, remaining)
    finish_b, _ = _agent_plan(state.agent_b, free_boxes, free_goals, board, remaining)
    raw = (W_LOCKED * (len(state.boxes_on_goals_a) - len(state.boxes_on_goals_b))
           + W_PROJECTED * (finish_a - finish_b))
    return raw if perspective == "A" else -raw


def nearest_objective(
    state: CompetitiveState,
    board: Board,
    perspective: str,
) -> Optional[Pos]:
    """Cheap cell an agent is currently heading for (used for move ordering)."""
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    if perspective == "A":
        pos = state.agent_a
        targets = state.boxes - occupied or state.boxes_on_goals_b
    else:
        pos = state.agent_b
        targets = state.boxes - occupied or state.boxes_on_goals_a
    if not targets:
        return None

    best = None
    best_dist = INF
    for target in targets:
        d = board.dist(pos, target)
        if d < best_dist:
            best_dist = d
            best = target
    return best


# ── Deadlock helpers (used by the search's action filters) ───────────────────

def has_legal_push(
    box: Pos,
    boxes: FrozenSet[Pos],
    board: Board,
) -> bool:
    """Whether `box` has any locally valid push direction."""
    for dx, dy in _DIRS:
        approach = (box[0] - dx, box[1] - dy)
        destination = (box[0] + dx, box[1] + dy)
        if approach in board.walls or approach in boxes:
            continue
        if destination in board.walls or destination in boxes:
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

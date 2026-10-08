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
1. projected score  - competitive race assignment: every unclaimed box goes
                      to the agent who can actually deliver it sooner
                      (independent per-box exact walk+push cost to the
                      cheapest free goal), a tied race splits the box; the
                      count is then capped by goal contention (a maximum
                      cardinality matching over the won boxes and the
                      distinct free goals) and by the remaining steps. A
                      box whose only goal was already claimed by another
                      of my won boxes counts once, never twice - and a box
                      is never dropped from the race just because a
                      greedy assignment consumed its goal (dropping it
                      read as "undeliverable" and flipped the edge term
                      below by the full clamp). Credited boxes are worth
                      a full locked point; a won race is worth a
                      discounted projected point, so cashing a box in on
                      a goal is always an improvement over merely chasing
                      it - the agent must not defer deliveries forever.
                      This is the term that tells the search WHO IS
                      WINNING each box.
2. conversion edge   - per box, priced ONLY for the side that is actually
                      ahead on it (the projected winner): what I pay to
                      cash a box I am winning, what the opponent pays to
                      cash a box THEY are winning (both clamped). Under
                      the real objective - score at the end of the
                      allotted rounds - my efficiency on my won box and
                      their efficiency on theirs are the margins that
                      decide the point. Distances to boxes neither side
                      can wrestle away are NOT priced here: including
                      them let cross-box noise swamp the signal, so an
                      agent could score a push that drags its box AWAY
                      from the goal above the correct approach. The edge
                      reads the independent per-box costs (same source as
                      the race winner), so a matching assignment can
                      never fake exclusivity. When only one side can
                      deliver within the remaining steps the
                      edge is the full clamp; when neither can it is zero.
                      The smooth walk-toward-contested-boxes gradient
                      stays in the initiative term below.
3. steal races      - advantage in reaching an opponent-credited box first,
                      measured as attacker distance vs defender distance to
                      the push approach cells. Credited boxes only; unclaimed
                      boxes are covered by the race assignment above.
4. initiative       - over the loose (free, off-goal) boxes: how much closer
                      I am to a push approach than the opponent. The race
                      assignment is all-or-nothing, so once a race looks
                      lost the raw assignment gives the search no reason to
                      keep walking; this term adds a smooth per-step gradient
                      that points at the contested boxes and breaks ties
                      between otherwise equal moves.

Caching
-------
`evaluate` memoizes on (board identity, positions, boxes, credits, remaining
steps) in a module level cache shared by the search, by the root move ordering
and by the transition function's conflict-diversion choice. The cache is the
reason the agent can think deep inside the time budget.
"""

from typing import Dict, FrozenSet, Optional, Tuple

from src.competitive.state import Action, Board, CompetitiveState


INF = 9999

W_LOCKED = 1000.0      # one CREDITED point (already on a goal)
W_PROJECTED = 600.0    # one projected point (won the cost race, not cashed in)
W_STEAL = 1200.0       # winning the approach race to an opponent's box
W_ADV = 12.0           # weight of the per-box winner-restricted conversion
                       # edge (one step of my delivery cost on my won box
                       # must out-weigh a step of strike-noise: 12 > W_INIT
                       # so the wrong push can never out-score the approach)
W_INIT = 10.0          # per-step strike-distance initiative on loose boxes
INIT_CAP = 3.0         # initiative clamp per box: cashing a box in must stay
                       # strictly better than chasing it (W_LOCKED - W_PROJECTED
                       # - W_STEAL*0.2 - W_INIT*INIT_CAP > 0)
ADV_CAP = 25.0         # per-box edge clamp: W_ADV * ADV_CAP = 300 stays
                       # below one projected point (600), so no single box's
                       # efficiency margin can out-shout the race counts

PROGRESS_BASE = 200.0  # progress gradient saturates past this many steps
STEAL_RANGE = 40.0     # approach distance over which a steal stays attractive
STEAL_WIN_BASE = 0.35  # race-won threat on one credited box: base contribution
STEAL_WIN_GRAD = 0.15  # ... plus gradient part. Max (0.5 * W_STEAL = 600) must
                       # stay BELOW W_LOCKED (1000): a strip preview may cost
                       # part of the point it threatens, never the whole point.
                       # The locked term charges the rest exactly once, when
                       # the strip actually happens - so holding a cred under
                       # attack always stays net-positive and scoring the next
                       # box can never evaluate as a net loss.

# Experiment only (see docs/ai_search_improvement_plan.md): a mild temporal
# adjustment for SPECULATIVE (projected, not yet credited) points. 0.0 =
# undiscounted baseline = the actual objective. Credit and terminal values
# are never discounted, so proven wins are never scaled below losses.
DISCOUNT_ALPHA = 0.0

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

    A cell scores STEAL_WIN_BASE..STEAL_WIN_BASE+STEAL_WIN_GRAD when the
    attacker reaches a push approach cell before the defender and still has
    time to push (bounded so the threat never outweighs the point it
    previews - see the constant); otherwise only a small gradient keeps the
    pursuit visible.
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
            total += STEAL_WIN_BASE + STEAL_WIN_GRAD * gradient
        else:
            total += 0.2 * gradient
    return total


def strike_distance(pos: Pos, box: Pos, boxes, board: Board) -> int:
    """Cheapest walk from `pos` to any valid push approach cell of `box`."""
    bx, by = box
    best = INF
    for dx, dy in _DIRS:
        approach = (bx - dx, by - dy)
        push_to = (bx + dx, by + dy)
        if approach in board.walls or push_to in board.walls:
            continue
        if approach in boxes or push_to in boxes:
            continue
        d = board.dist(pos, approach)
        if d < best:
            best = d
    return best


def _initiative(state: CompetitiveState, board: Board) -> float:
    """
    Offense gap over the loose boxes (free and not sitting on a goal):
    the opponent's strike distance minus mine, summed per box.

    Positive means I am closer to the contested boxes than the opponent
    is, giving the search a continuous reason to keep walking toward a
    fight even when the race assignment already counts the box against it.
    A box nobody can approach from anywhere is skipped.
    """
    free = state.boxes - (state.boxes_on_goals_a | state.boxes_on_goals_b)
    if not free:
        return 0.0
    total = 0.0
    for box in free:
        if box in board.goals:
            continue                    # loose box already on its goal: no race
        mine = strike_distance(state.agent_a, box, state.boxes, board)
        theirs = strike_distance(state.agent_b, box, state.boxes, board)
        if mine >= INF or theirs >= INF:
            continue
        total += max(-INIT_CAP, min(INIT_CAP, theirs - mine))
    return total


# ── Term 1 + 2: competitive race assignment and cost gradient ─────────────────

def _match_costs(
    pos: Pos,
    free_boxes,
    free_goals,
    board: Board,
) -> Dict[Pos, int]:
    """
    Maximum-cardinality box->goal matching for one agent, cheapest edges
    preferred (Kuhn augmenting paths over cost-sorted edges).

    Returns {box: exact walk+push cost} for every box the agent can deliver
    to some distinct free goal; unmatched (undeliverable) boxes are absent.

    Cardinality first, cost second: the old cheapest-first scan let an
    early box consume the only goal another box could reach and then
    silently DROPPED that box (capacity_lab: boxes (4,7) and (14,8) both
    need goal (8,8) as (14,8)'s only reachable goal - (4,7) grabbed it at
    cost 10 and (14,8) came back unmatched, i.e. undeliverable, which the
    race then read as "only the opponent can cash this" and charged the
    full clamp). Augmenting paths repair exactly that: the displaced box
    re-homes to its next goal. Within a box's search, goals are tried
    cheapest-first, so cheap assignments survive whenever they can.
    """
    if not free_boxes or not free_goals:
        return {}
    px, py = pos
    edges_by_box: Dict[Pos, list] = {}
    for box in free_boxes:
        bx, by = box
        for goal in free_goals:
            cost = board.exact_step_costs.get(goal, {}).get((bx, by, px, py), INF)
            if cost < INF:
                edges_by_box.setdefault(box, []).append((cost, goal))
    if not edges_by_box:
        return {}

    owner: Dict[Pos, Pos] = {}          # goal -> box
    matched: Dict[Pos, Tuple[Pos, int]] = {}   # box -> (goal, cost)

    def augment(box: Pos, seen: set) -> bool:
        for _cost, goal in sorted(edges_by_box[box]):
            if goal in seen:
                continue
            seen.add(goal)
            prev = owner.get(goal)
            if prev is None or augment(prev, seen):
                owner[goal] = box
                matched[box] = (goal, _cost)
                return True
        return False

    # Boxes with the cheapest option first: the constrained boxes claim
    # their goals before the flexible ones, and augmenting repairs any
    # over-claim either way (max cardinality is order-independent).
    order = sorted(
        edges_by_box,
        key=lambda b: min(cost for cost, _g in edges_by_box[b]),
    )
    for box in order:
        augment(box, set())
    return {box: cost for box, (_goal, cost) in matched.items()}


def _advantage(x: int, y: int, remaining: int) -> float:
    """
    A-perspective conversion edge on one free box, in cost units.

    Winner-restricted pricing (see the terms list): only the side that is
    ahead on THIS box is charged for its own cost, so one agent's distances
    to boxes it will never win cannot drown out the margin that decides
    the box it is actually trying to finish.

      * only A can deliver within `remaining`  -> +ADV_CAP
      * only B can deliver within `remaining`  -> -ADV_CAP
      * neither can                            -> 0  (the projected counts
        already score this zero for both sides)
      * both can: -x if A is ahead (A pays for its own speed),
                  +y if B is ahead (A gains from B's cost),
                  0 on an exact tie (the counts split it).

    The result is clamped to +/- ADV_CAP per box.
    """
    x_ok = x <= remaining
    y_ok = y <= remaining
    if x_ok and not y_ok:
        return ADV_CAP           # only I can cash this one in time
    if y_ok and not x_ok:
        return -ADV_CAP          # only the opponent can
    if not x_ok and not y_ok:
        return 0.0               # nobody converts in time: neutral
    if x < y:
        edge = -float(x)         # I'm ahead: charged for my own cost
    elif y < x:
        edge = float(y)          # they're ahead: I gain from their cost
    else:
        return 0.0               # exact tie: the counts split it
    return max(-ADV_CAP, min(ADV_CAP, edge))


def _race(
    state: CompetitiveState,
    board: Board,
    remaining: int,
) -> Tuple[float, float, float]:
    """
    Competitive projection over the unclaimed boxes.

    Returns (projected_a, projected_b, adv_a - adv_b):
      projected_* - boxes each agent is expected to deliver (credited boxes
                    are counted by the caller). WHO is ahead on a box comes
                    from independent per-box delivery costs (cheapest free
                    goal, no contention): that is the real race. The count
                    itself is then capped by goal contention - a maximum
                    cardinality matching over the won boxes and the distinct
                    free goals, each counted only if the ASSIGNED goal is
                    still reachable within the remaining steps - so two won
                    boxes that need the same single goal count once, not
                    twice. A dead tie splits the box in half.
      adv         - summed winner-restricted conversion edge (see
                    `_advantage`), A positive / B negative, computed from
                    the same independent costs: the race winner's own
                    efficiency, uncontaminated by which goal the matching
                    happened to assign.
    """
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    free_boxes = state.boxes - occupied
    if not free_boxes:
        return 0.0, 0.0, 0.0
    free_goals = board.goals - occupied

    # Independent per-box delivery cost per side: cheapest of the free
    # goals, contention ignored (contention belongs to the counts below).
    min_a: Dict[Pos, int] = {}
    min_b: Dict[Pos, int] = {}
    ax, ay = state.agent_a
    bx, by = state.agent_b
    for box in free_boxes:
        ox, oy = box
        best_a = INF
        best_b = INF
        for goal in free_goals:
            gx, gy = goal
            cost = board.exact_step_costs.get(goal, {}).get(
                (ox, oy, ax, ay), INF)
            if cost < best_a:
                best_a = cost
            cost = board.exact_step_costs.get(goal, {}).get(
                (ox, oy, bx, by), INF)
            if cost < best_b:
                best_b = cost
        min_a[box] = best_a
        min_b[box] = best_b

    projected_a = 0.0
    projected_b = 0.0
    adv = 0.0
    won_a = []
    won_b = []
    for box in free_boxes:
        x = min_a[box]
        y = min_b[box]
        adv += _advantage(x, y, remaining)
        if x < y:
            won_a.append(box)
        elif y < x:
            won_b.append(box)
        elif x < INF:  # exact tie between reachable deliveries
            if x <= remaining:
                projected_a += 0.5
                projected_b += 0.5

    if won_a:
        for cost in _match_costs(
            state.agent_a, won_a, free_goals, board
        ).values():
            if cost <= remaining:
                projected_a += 1.0
    if won_b:
        for cost in _match_costs(
            state.agent_b, won_b, free_goals, board
        ).values():
            if cost <= remaining:
                projected_b += 1.0
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
    if DISCOUNT_ALPHA:
        # Speculative points only, scaled by how much time is left to
        # convert them (never the credited/terminal terms above).
        factor = 1.0 - DISCOUNT_ALPHA * (
            1.0 - remaining / max(max_steps, 1)
        )
    else:
        factor = 1.0
    value += factor * W_PROJECTED * (projected_a - projected_b)
    value += W_ADV * adv
    value += W_INIT * _initiative(state, board)

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
        board.serial,               # board identity: values are wall/goal dependent
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


# ── Gates for the two hard push filters (see tests/test_search_identity.py) ──

def _min_delivery(state: CompetitiveState, board: Board, box: Pos, who: str) -> int:
    """Cheapest walk+push delivery cost for `who` to move `box` to any goal."""
    pos = state.agent_a if who == "A" else state.agent_b
    best = INF
    for goal in board.goals:
        cost = board.exact_step_costs.get(goal, {}).get(
            (box[0], box[1], pos[0], pos[1]), INF
        )
        if cost < best:
            best = cost
    return best


def denial_justified(
    state: CompetitiveState, board: Board, box: Pos, who: str, max_steps: int
) -> bool:
    """
    Whether killing `box` with a deadlock push is a rational denial.

    This is strategic pruning, not a game rule - the engine allows any
    push. Killing a box removes it from play for BOTH players, so it is
    only correct when the opponent is already winning the delivery race
    for it: the box trends to +1 for them and denial converts that into
    zero (preserving a lead, or turning a loss into a draw when it is the
    last box). When the race is mine, or nobody can deliver the box at
    all, the push only throws away a point I would otherwise score, so
    the deadlock filter stays in force.
    """
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b
    remaining = max(max_steps - state.step, 0)
    if box in occupied:
        mine = _min_delivery(state, board, box, who)
        theirs = _min_delivery(state, board, box, "B" if who == "A" else "A")
    else:
        # Loose box: the race is per-box and needs no matching - goal
        # contention belongs to the eval's count term, not to "can the
        # opponent out-deliver me here". Matched costs used to be read
        # here too, and a matching that dropped this box (its only goal
        # consumed by another box) returned INF, which blocked a
        # perfectly justified denial as "they cannot deliver it either".
        free_goals = board.goals - occupied
        bx, by = box
        mine = INF
        theirs = INF
        my_pos = state.agent_a if who == "A" else state.agent_b
        their_pos = state.agent_b if who == "A" else state.agent_a
        for goal in free_goals:
            gx, gy = goal
            cost = board.exact_step_costs.get(goal, {}).get(
                (bx, by, my_pos[0], my_pos[1]), INF
            )
            if cost < mine:
                mine = cost
            cost = board.exact_step_costs.get(goal, {}).get(
                (bx, by, their_pos[0], their_pos[1]), INF
            )
            if cost < theirs:
                theirs = cost
    if theirs >= INF:
        return False              # they cannot deliver it either: pointless
    return theirs < mine and theirs <= remaining


# Audit note (own credited box off its goal, the OTHER hard push filter):
# a counterexample "loose box needs exactly my cell while I re-seat
# elsewhere" was constructed and REFUTED: under static push-reachability
# the needy box can chain through the vacated cell to any goal the
# vacating box can reach (the agent cell needed for the g1 -> new_cell
# push is reachable - the pushing agent literally stands there), so
# "needy elsewhere" and "I can re-seat" are unsatisfiable together.
# Dynamic escapes depend on opponent interference and are EV-negative
# (a stripped box mid-plan loses the point outright), so the rule stays
# hard - see tests/test_search_identity.py.

"""Shared deterministic simultaneous transition, including forced blockage.

Odd remaining rounds favor A; even favor B. A conflict loser tries its fixed
ranked alternatives. If a stationary loser blocks the winner, the winner
tries a compatible alternative too (preserved existing gameplay). A side
with no compatible move is forced to stay. Every resolved round consumes time.
WAIT is an internal/legacy sentinel, not a selectable player direction.
"""

from typing import Dict, NamedTuple, Optional, Tuple

from src.competitive.state import Action, Board, CompetitiveState


class Outcome(NamedTuple):
    """Result of one joint transition."""
    state: CompetitiveState
    conflict: bool
    resolved_action_a: Optional[Action]
    resolved_action_b: Optional[Action]


_Outcome = Outcome  # backwards compatible alias

Pos = Tuple[int, int]

_DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))
_YIELD_ORDER = (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)

_CACHE_LIMIT = 40_000
_transition_cache: Dict[tuple, Outcome] = {}


class _Intent(NamedTuple):
    """A committed move: where the agent ends up, and where its box lands."""
    dest: Pos
    push_dest: Optional[Pos]


# ── Small helpers ────────────────────────────────────────────────────────────

def _step(pos: Pos, action: Action) -> Pos:
    dx, dy = action.value
    return (pos[0] + dx, pos[1] + dy)


def _other(who: str) -> str:
    return "B" if who == "A" else "A"


def conflict_winner(max_steps: int, step: int) -> str:
    """Odd remaining steps -> A, even remaining steps -> B."""
    return "A" if (max_steps - step) % 2 else "B"


def _intent(pos: Pos, action: Optional[Action], boxes, board: Board) -> Optional[_Intent]:
    """Physical intent of an action at `pos`, or None if it cannot happen."""
    if action is None or action is Action.WAIT:
        return None
    dest = _step(pos, action)
    if dest not in board.floor_cells:
        return None
    if dest in boxes:
        push_dest = _step(dest, action)
        if push_dest not in board.floor_cells or push_dest in boxes:
            return None
        return _Intent(dest, push_dest)
    return _Intent(dest, None)


def _enters(intent: _Intent, cell: Pos) -> bool:
    """True when the intent ends (agent or box) on `cell`."""
    if intent.dest == cell:
        return True
    return intent.push_dest is not None and intent.push_dest == cell


def _conflict(one: _Intent, other: _Intent, one_pos: Pos, other_pos: Pos) -> bool:
    """True when two intents cannot both be executed (rules 7.1 - 7.5)."""
    if one.dest == other.dest:                                   # 7.1 / 7.3
        return True
    if one.dest == other_pos and other.dest == one_pos:           # 7.2 swap
        return True
    if one.push_dest is not None and one.push_dest == other.dest: # 7.5
        return True
    if other.push_dest is not None and other.push_dest == one.dest:
        return True
    if one.push_dest is not None and other.push_dest is not None: # 7.4
        if one.push_dest == other.push_dest:
            return True
    return False


# ── Commit ───────────────────────────────────────────────────────────────────

def _commit(
    state: CompetitiveState,
    action_a: Optional[Action],
    action_b: Optional[Action],
    board: Board,
    max_steps: int,
) -> CompetitiveState:
    """
    Apply two compatible (already conflict-free) actions. `None` means the
    agent does not move. Callers guarantee the pair passes `_conflict`.
    """
    boxes = state.boxes
    moves = {"A": state.agent_a, "B": state.agent_b}
    intents = {}
    for who, action in (("A", action_a), ("B", action_b)):
        intents[who] = (
            _intent(moves[who], action, boxes, board)
            if action is not None and action is not Action.WAIT
            else None
        )

    pushes = []
    for who in ("A", "B"):
        intent = intents[who]
        if intent is None:
            continue
        if intent.push_dest is not None:
            boxes = boxes - {intent.dest} | {intent.push_dest}
            pushes.append((intent.dest, intent.push_dest, who))
        moves[who] = intent.dest

    cred_a = set(state.boxes_on_goals_a)
    cred_b = set(state.boxes_on_goals_b)
    for src, dst, who in pushes:
        cred_a.discard(src)
        cred_b.discard(src)
        if dst in board.goals:
            if who == "A":
                cred_a.add(dst)
                cred_b.discard(dst)
            else:
                cred_b.add(dst)
                cred_a.discard(dst)

    final_boxes = frozenset(boxes)
    cred_a = frozenset(p for p in cred_a if p in final_boxes and p in board.goals)
    cred_b = frozenset(p for p in cred_b if p in final_boxes and p in board.goals)

    return CompetitiveState(
        agent_a=moves["A"],
        agent_b=moves["B"],
        boxes=final_boxes,
        boxes_on_goals_a=frozenset(cred_a),
        boxes_on_goals_b=frozenset(cred_b),
        step=state.step + 1,
    )


# ── Diversion of the conflict loser ──────────────────────────────────────────

def _pick_diversion(
    state: CompetitiveState,
    who: str,
    other_action: Optional[Action],
    board: Board,
    max_steps: int,
    prefs: Optional[Tuple[Action, ...]] = None,
) -> Optional[Action]:
    """First compatible legal fallback in a fixed starting-state ranking.

    No planner/evaluator is called during conflict resolution. Legacy callers
    use the same cheap default policy as agents. An explicit empty list means
    no alternatives; forced immobility is represented by None internally.
    """
    my_pos = state.agent_a if who == "A" else state.agent_b
    other_pos = state.agent_b if who == "A" else state.agent_a
    boxes = state.boxes

    theirs = (
        _intent(other_pos, other_action, boxes, board)
        if other_action is not None
        else None
    )

    if prefs is None:
        from src.competitive.preferences import round_preferences
        prefs = round_preferences(state, board, max_steps, who)
    for action in prefs:
        mine = _intent(my_pos, action, boxes, board)
        if mine is None:
            continue
        if theirs is None:
            if _enters(mine, other_pos):
                continue
        elif _conflict(mine, theirs, my_pos, other_pos):
            continue
        return action
    return None


def _diversion_order(divert, max_steps: int, step: int):
    """Priority winner settles first, so the loser diverts around it."""
    winner = conflict_winner(max_steps, step)
    return [who for who in divert if who == winner] + [
        who for who in divert if who != winner
    ]


# ── Core transition ──────────────────────────────────────────────────────────

def _resolve(
    state: CompetitiveState,
    action_a: Action,
    action_b: Action,
    board: Board,
    max_steps: int,
    prefs_a: Optional[Tuple[Action, ...]] = None,
    prefs_b: Optional[Tuple[Action, ...]] = None,
) -> Outcome:
    pos = {"A": state.agent_a, "B": state.agent_b}
    acts = {"A": action_a, "B": action_b}
    boxes = state.boxes

    intent = {
        "A": _intent(pos["A"], action_a, boxes, board),
        "B": _intent(pos["B"], action_b, boxes, board),
    }
    divert = []
    conflict = False

    # (1) incompatible intents -> parity winner keeps its move
    if intent["A"] is not None and intent["B"] is not None:
        if _conflict(intent["A"], intent["B"], pos["A"], pos["B"]):
            conflict = True
            loser = _other(conflict_winner(max_steps, state.step))
            intent[loser] = None
            divert.append(loser)

    # (2) entering a cell whose occupant does not move -> entrant diverts
    for who in ("A", "B"):
        if intent[who] is None:
            continue
        other = _other(who)
        if intent[other] is None and other not in divert:
            if _enters(intent[who], pos[other]):
                conflict = True
                intent[who] = None
                divert.append(who)

    # final action of each agent; None = it does not move
    final = {
        who: (acts[who] if intent[who] is not None else None)
        for who in ("A", "B")
    }

    # (3) the conflict loser(s) move somewhere else instead of standing still
    #     (Rule 3: each loser diverts along ITS OWN preference list; the
    #     winner's action is fixed first, never overturned by a fallback)
    for who in _diversion_order(divert, max_steps, state.step):
        final[who] = _pick_diversion(
            state, who, final[_other(who)], board, max_steps,
            prefs_a if who == "A" else prefs_b,
        )
        conflict = True

    # (4) an agent with no alternative stays put, which can invalidate the
    #     other agent's entry into that cell
    for who in ("A", "B"):
        if intent[who] is None:
            continue
        other = _other(who)
        if final[other] is None and _enters(intent[who], pos[other]):
            conflict = True
            intent[who] = None
            final[who] = _pick_diversion(
                state, who, final[other], board, max_steps,
                prefs_a if who == "A" else prefs_b,
            )

    next_state = _commit(state, final["A"], final["B"], board, max_steps)
    return Outcome(
        state=next_state,
        conflict=conflict,
        resolved_action_a=final["A"] if final["A"] is not None else Action.WAIT,
        resolved_action_b=final["B"] if final["B"] is not None else Action.WAIT,
    )


def resolve_joint_action_outcome(
    state: CompetitiveState,
    action_a: Action,
    action_b: Action,
    board: Board,
    max_steps: int,
    prefs_a: Optional[Tuple[Action, ...]] = None,
    prefs_b: Optional[Tuple[Action, ...]] = None,
) -> Outcome:
    """Deterministic joint transition (memoized). Returns `Outcome`.

    `prefs_a`/`prefs_b` are this round's Rule 3 preference lists (see the
    module docstring): the conflict loser of that side falls back along its
    own list, while a side without a list uses the starting-state ranked default.
    The lists are part of the cache key (None and an empty list are distinct
    inputs - the first means "no list", the second "no legal direction").
    """
    pa = tuple(prefs_a) if prefs_a is not None else None
    pb = tuple(prefs_b) if prefs_b is not None else None
    key = (
        board.serial,        # diversion choice depends on walls/goals
        max_steps,           # ...and the cached state carries an absolute
        state.agent_a,       # step, so (remaining, max_steps) must both match
        state.agent_b,
        state.boxes,
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
        max_steps - state.step,
        action_a,
        action_b,
        pa,                  # Rule 3: fallback order shapes the diversion
        pb,
    )
    outcome = _transition_cache.get(key)
    if outcome is None:
        outcome = _resolve(state, action_a, action_b, board, max_steps, pa, pb)
        if len(_transition_cache) >= _CACHE_LIMIT:
            _transition_cache.clear()
        _transition_cache[key] = outcome
    return outcome


def resolve_joint_action(
    state: CompetitiveState,
    action_a: Action,
    action_b: Action,
    board: Board,
) -> CompetitiveState:
    """Legacy wrapper: next state only."""
    return resolve_joint_action_outcome(state, action_a, action_b, board, 1000).state


def clear_cache() -> None:
    """Drop the joint-transition memo (used by tests)."""
    _transition_cache.clear()


# ── Valid action generator ───────────────────────────────────────────────────

def get_valid_actions(
    pos: Pos,
    other_pos: Pos,
    boxes,
    board: Board,
    include_wait: bool = False,
):
    """
    Physically valid actions for an agent at `pos`.

    WAIT is excluded by default: every agent must move every turn. Occupancy
    rules involving `other_pos` are handled by the joint transition, so the
    opponent's cell is deliberately not excluded here.
    """
    valid = []
    for action in _YIELD_ORDER:
        dest = _step(pos, action)
        if dest not in board.floor_cells:
            continue
        if dest in boxes:
            push_dest = _step(dest, action)
            if push_dest not in board.floor_cells or push_dest in boxes:
                continue
        valid.append(action)

    if include_wait:
        valid.append(Action.WAIT)
    return valid

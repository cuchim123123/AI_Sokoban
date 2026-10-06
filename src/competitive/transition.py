"""
Deterministic joint transition for the 2-agent competitive Sokoban.

The GUI and the search both call `resolve_joint_action_outcome`, so the AI can
never disagree with the rules.

Conflict rules
--------------
7.1  both agents target the same destination cell
7.2  the agents swap cells (A -> B's cell while B -> A's cell)
7.3  both agents push the same box
7.4  two pushes would drop a box on the same cell
7.5  a push would drop a box on the cell the other agent ends up in
7.6  an agent enters (or pushes into) a cell the other agent does not vacate

Odd/even advantage
------------------
    remaining = max_steps - state.step
    remaining odd  -> agent A wins every conflict
    remaining even -> agent B wins every conflict

The winner's intent is executed. The loser never stands still: its action is
replaced by its best legal alternative (highest shared evaluation, tie-broken
by a fixed direction order), chosen so that it cannot immediately trigger a
second conflict. Only when an agent is physically boxed in with no legal
alternative does it stay in place.

Exception to "the winner takes the cell": if the other agent does not move at
all (WAIT or an illegal action), it keeps the cell it already occupies and the
agent trying to enter diverts instead - the occupant is not competing for a
contested target, and the entrant still has to move.

Speed
-----
Complete joint transitions are memoized on (positions, boxes, credits,
remaining steps, both actions). Iterative deepening re-expands the same nodes
every round and the search visits many transpositions, so the cache is a large
part of why the agent reaches depth inside its time budget.
"""

from typing import Dict, NamedTuple, Optional, Tuple

from src.competitive.evaluation import evaluate
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
    if dest in board.walls:
        return None
    if dest in boxes:
        push_dest = _step(dest, action)
        if push_dest in board.walls or push_dest in boxes:
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
) -> Optional[Action]:
    """
    Best legal alternative for `who`, given what the other agent ends up doing
    (`None` = it does not move). Returns None only when no alternative exists.
    """
    my_pos = state.agent_a if who == "A" else state.agent_b
    other_pos = state.agent_b if who == "A" else state.agent_a
    boxes = state.boxes

    theirs = (
        _intent(other_pos, other_action, boxes, board)
        if other_action is not None
        else None
    )

    best_action: Optional[Action] = None
    best_value = None
    for action in _YIELD_ORDER:
        mine = _intent(my_pos, action, boxes, board)
        if mine is None:
            continue
        if theirs is None:
            if _enters(mine, other_pos):
                continue
        elif _conflict(mine, theirs, my_pos, other_pos):
            continue

        action_a = action if who == "A" else other_action
        action_b = other_action if who == "A" else action
        candidate = _commit(state, action_a, action_b, board, max_steps)
        value = evaluate(candidate, board, who, max_steps)
        if best_value is None or value > best_value:
            best_value = value
            best_action = action

    return best_action


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
    for who in _diversion_order(divert, max_steps, state.step):
        final[who] = _pick_diversion(state, who, final[_other(who)], board, max_steps)
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
            final[who] = _pick_diversion(state, who, final[other], board, max_steps)

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
) -> Outcome:
    """Deterministic joint transition (memoized). Returns `Outcome`."""
    key = (
        state.agent_a,
        state.agent_b,
        state.boxes,
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
        max_steps - state.step,
        action_a,
        action_b,
    )
    outcome = _transition_cache.get(key)
    if outcome is None:
        outcome = _resolve(state, action_a, action_b, board, max_steps)
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
        if dest in board.walls:
            continue
        if dest in boxes:
            push_dest = _step(dest, action)
            if push_dest in board.walls or push_dest in boxes:
                continue
        valid.append(action)

    if include_wait:
        valid.append(Action.WAIT)
    return valid

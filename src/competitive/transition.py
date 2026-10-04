"""
resolve_joint_action — single deterministic transition function.
The search simulation and the GUI BOTH call this function (same rules).

All conflict rules from the design document:
    7.1  Both agents target the same destination cell       → priority winner moves; loser yields
    7.2  Agents try to swap positions                       → both fail
    7.3  Both push the same box in opposite directions      → both fail
    7.4  Both push the same box in perpendicular dirs       → both fail
    7.5  An agent pushes a box into the other agent         → that push fails
    +    Moving into the other agent's current cell         → blocked
         (agent cannot pass through / enter occupied cell)
"""
from typing import Tuple, Optional, List
from src.competitive.state import Action, Board, CompetitiveState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step(pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
    dx, dy = action.value
    return (pos[0] + dx, pos[1] + dy)


# ── Core transition ───────────────────────────────────────────────────────────

def resolve_joint_action_outcome(
    state: CompetitiveState,
    action_a: Action,
    action_b: Action,
    board: Board,
    max_steps: int,
    _resolve_yield: bool = True,
):
    """
    Deterministic joint transition.  Returns next CompetitiveState (step+1).
    All conflict rules produce no-ops for the conflicting agents.
    """
    pos_a = state.agent_a
    pos_b = state.agent_b
    boxes = state.boxes   # frozenset — we never mutate it directly

    # ── 1. Compute intended destinations ─────────────────────────────────────
    dest_a = _step(pos_a, action_a) if action_a != Action.WAIT else pos_a
    dest_b = _step(pos_b, action_b) if action_b != Action.WAIT else pos_b

    # ── 2. Determine push targets ─────────────────────────────────────────────
    push_a_box  = dest_a if (action_a != Action.WAIT and dest_a in boxes) else None
    push_a_dest = _step(dest_a, action_a) if push_a_box is not None else None
    push_b_box  = dest_b if (action_b != Action.WAIT and dest_b in boxes) else None
    push_b_dest = _step(dest_b, action_b) if push_b_box is not None else None

    # ── 3. Individual physical validity ───────────────────────────────────────
    def _physically_valid(
        action, dest, push_box, push_dest_cell, other_pos
    ) -> bool:
        if action == Action.WAIT:
            return True
        # Destination must be a floor cell
        if dest in board.walls:
            return False
        if push_box is None:
            pass # Moved to inter-agent rules
        else:
            # Push: box destination must be free of walls and other boxes
            if push_dest_cell in board.walls:
                return False
            if push_dest_cell in boxes:
                return False
        return True

    valid_a = _physically_valid(action_a, dest_a, push_a_box, push_a_dest, pos_b)
    valid_b = _physically_valid(action_b, dest_b, push_b_box, push_b_dest, pos_a)

    # ── 4. Inter-agent conflict rules ─────────────────────────────────────────

    conflict_occurred = False

    def _fail_both() -> None:
        nonlocal valid_a, valid_b, conflict_occurred
        valid_a = False
        valid_b = False
        conflict_occurred = True

    # Rules 7.3 / 7.4 — both push the same box in any directions.
    # This takes precedence over same-destination handling because both
    # agents' destinations are the shared box cell.
    if valid_a and valid_b and push_a_box is not None and push_b_box is not None:
        if push_a_box == push_b_box:
            _fail_both()

    # Rule 7.1 — both target the same destination cell. The priority winner
    # occupies it; the yielding action is replaced by a deterministic best
    # legal alternative rather than becoming a silent no-op.
    same_destination_conflict = False
    if valid_a and valid_b and dest_a == dest_b:
        if action_a != Action.WAIT or action_b != Action.WAIT:
            if _resolve_yield:
                if (max_steps - state.step) % 2:
                    yielding_perspective = 'B'
                    winning_action = action_a
                else:
                    yielding_perspective = 'A'
                    winning_action = action_b

                alternative = _best_yield_action(
                    state,
                    winning_action,
                    yielding_perspective,
                    board,
                    max_steps,
                )
                if alternative is not None:
                    if yielding_perspective == 'A':
                        action_a, action_b = alternative, winning_action
                    else:
                        action_a, action_b = winning_action, alternative
                    resolved = resolve_joint_action_outcome(
                        state,
                        action_a,
                        action_b,
                        board,
                        max_steps,
                        _resolve_yield=False,
                    )
                    resolved.conflict = True
                    return resolved

            same_destination_conflict = True
            conflict_occurred = True
            if (max_steps - state.step) % 2:
                valid_b = False
            else:
                valid_a = False

    # Rule 7.2 — agents try to swap positions.
    if valid_a and valid_b and dest_a == pos_b and dest_b == pos_a:
        _fail_both()

    # Rule 7.5 — a push into the other agent fails, while the other move may
    # still commit if it is otherwise valid.
    if valid_a and valid_b:
        if push_a_dest is not None and push_a_dest == dest_b:
            valid_a = False
            conflict_occurred = True
        if push_b_dest is not None and push_b_dest == dest_a:
            valid_b = False
            conflict_occurred = True

    # An agent cannot enter the other agent's current cell, even when the
    # other agent is moving away during this joint step.
    if valid_a and not same_destination_conflict and action_a != Action.WAIT and dest_a == pos_b:
        valid_a = False
        conflict_occurred = True
    if valid_b and not same_destination_conflict and action_b != Action.WAIT and dest_b == pos_a:
        valid_b = False
        conflict_occurred = True

    # ── 5. Commit valid moves ─────────────────────────────────────────────────
    new_boxes: set = set(boxes)
    new_pos_a = pos_a
    new_pos_b = pos_b
    committed_push_a_box  = None
    committed_push_a_dest = None
    committed_push_b_box  = None
    committed_push_b_dest = None

    if valid_a and action_a != Action.WAIT:
        if push_a_box is not None:
            new_boxes.discard(push_a_box)
            new_boxes.add(push_a_dest)
            committed_push_a_box  = push_a_box
            committed_push_a_dest = push_a_dest
        new_pos_a = dest_a

    if valid_b and action_b != Action.WAIT:
        if push_b_box is not None:
            new_boxes.discard(push_b_box)
            new_boxes.add(push_b_dest)
            committed_push_b_box  = push_b_box
            committed_push_b_dest = push_b_dest
        new_pos_b = dest_b

    # ── 6. Update completion credit ───────────────────────────────────────────
    new_boxes_fs = frozenset(new_boxes)
    new_bga, new_bgb = _update_credit(
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
        boxes,
        new_boxes_fs,
        board,
        committed_push_a_box, committed_push_a_dest,
        committed_push_b_box, committed_push_b_dest,
    )

    ns = CompetitiveState(
        agent_a=new_pos_a,
        agent_b=new_pos_b,
        boxes=new_boxes_fs,
        boxes_on_goals_a=new_bga,
        boxes_on_goals_b=new_bgb,
        step=state.step + 1,
    )
    
    class Outcome:
        pass
    out = Outcome()
    out.state = ns
    out.conflict = conflict_occurred
    return out

def resolve_joint_action(
    state: CompetitiveState,
    action_a: Action,
    action_b: Action,
    board: Board,
) -> CompetitiveState:
    return resolve_joint_action_outcome(state, action_a, action_b, board, 1000).state


def _best_yield_action(
    state: CompetitiveState,
    winning_action: Action,
    yielding_perspective: str,
    board: Board,
    max_steps: int,
):
    """Choose the strongest non-WAIT response for a yielding agent."""
    from src.competitive.evaluation import competitive_heuristic

    yielding_pos = state.agent_a if yielding_perspective == 'A' else state.agent_b
    winning_pos = state.agent_b if yielding_perspective == 'A' else state.agent_a
    actions = get_valid_actions(
        yielding_pos,
        winning_pos,
        state.boxes,
        board,
        include_wait=False,
    )
    ranked = []
    for action in actions:
        action_a = action if yielding_perspective == 'A' else winning_action
        action_b = winning_action if yielding_perspective == 'A' else action
        next_state = resolve_joint_action_outcome(
            state,
            action_a,
            action_b,
            board,
            max_steps,
            _resolve_yield=False,
        ).state
        next_pos = next_state.agent_a if yielding_perspective == 'A' else next_state.agent_b
        if next_pos == yielding_pos:
            continue
        ranked.append((
            competitive_heuristic(
                next_state,
                board,
                yielding_perspective,
                max_steps,
            ),
            action,
        ))

    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1].value))[1]


# ── Credit bookkeeping ────────────────────────────────────────────────────────

def _update_credit(
    old_bga, old_bgb,
    old_boxes, new_boxes,
    board,
    push_a_box, push_a_dest,
    push_b_box, push_b_dest,
):
    bga = set(old_bga)
    bgb = set(old_bgb)

    # Strip credit for boxes that were pushed OFF their goal
    for old_pos in filter(None, [push_a_box, push_b_box]):
        if old_pos in board.goals:
            bga.discard(old_pos)
            bgb.discard(old_pos)

    # Grant credit to the pushing agent when a box lands on a goal
    if push_a_dest is not None and push_a_dest in board.goals:
        bgb.discard(push_a_dest)
        bga.add(push_a_dest)

    if push_b_dest is not None and push_b_dest in board.goals:
        bga.discard(push_b_dest)
        bgb.add(push_b_dest)

    # Sanity guard: only keep credit for boxes actually on goals right now
    bga = frozenset(p for p in bga if p in new_boxes and p in board.goals)
    bgb = frozenset(p for p in bgb if p in new_boxes and p in board.goals)
    return bga, bgb


# ── Valid action generator ────────────────────────────────────────────────────

def get_valid_actions(
    pos: Tuple[int, int],
    other_pos: Tuple[int, int],
    boxes,
    board: Board,
    include_wait: bool = False,
) -> List[Action]:
    """
    Return physically valid actions for an agent at `pos`.

    WAIT is excluded by default: every normal agent turn must make a move.
    Pass include_wait=True only for an unavoidable dead-end fallback.
    """
    valid: List[Action] = []
    for action in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
        dest = _step(pos, action)
        if dest in board.walls:
            continue
        # We do NOT exclude dest == other_pos here because the opponent might move out of the tile.
        # Collision resolution is handled by resolve_joint_action.
        if dest in boxes:
            push_dest = _step(dest, action)
            if push_dest in board.walls or push_dest in boxes:
                continue
            # Note: We also do not forbid push_dest == other_pos here, for the same reason.
        valid.append(action)

    if include_wait:
        valid.append(Action.WAIT)

    return valid

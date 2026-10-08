"""Cheap, deterministic fallback policy, shared by live play and search.

Rank from the starting state, without pretending the opponent has committed
to WAIT. Only local push/approach features are used; no planner recursion.
"""
from src.competitive.state import Action


def round_preferences(state, board, max_steps, me, primary=None, eval_fn=None):
    from src.competitive.transition import get_valid_actions
    from src.competitive.evaluation import creates_deadlock

    pos = state.agent_a if me == "A" else state.agent_b
    other = state.agent_b if me == "A" else state.agent_a
    own = state.boxes_on_goals_a if me == "A" else state.boxes_on_goals_b
    theirs = state.boxes_on_goals_b if me == "A" else state.boxes_on_goals_a
    legal = get_valid_actions(pos, other, state.boxes, board)
    goals = board.goals - own - theirs

    def rank(action):
        dx, dy = action.value
        dest = (pos[0] + dx, pos[1] + dy)
        boxes = state.boxes
        gain = 0
        if dest in boxes:
            end = (dest[0] + dx, dest[1] + dy)
            gain = int(end in board.goals) - int(dest in own) + int(dest in theirs)
            boxes = boxes - {dest} | {end}
        targets = boxes - own - theirs
        cost = min((board.exact_steps(b, dest, g)
                    for b in targets for g in goals), default=9999)
        attack = min((board.dist(dest, b) for b in theirs), default=9999)
        return (-gain, creates_deadlock(pos, action, state.boxes, board), cost, attack)

    ranked = sorted(legal, key=rank)
    if primary in ranked:
        ranked.remove(primary)
        ranked.insert(0, primary)
    return tuple(ranked)

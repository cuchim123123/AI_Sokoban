from typing import FrozenSet, Tuple
from src.competitive.state import CompetitiveState, Board

# ── Heuristic Weights ─────────────────────────────────────────────────────────
# These constants define the relative importance of different strategic goals.
# Tweak these to change the AI's behavior.

W_SCORE      = 1000.0   # Reward for each point (box on goal)
W_CHAIN      = 150.0    # Reward for being in position to push an unplaced box to a goal
W_STEAL      = 120.0    # Reward for being in position to steal an opponent's box
W_GUARD      = 80.0     # Reward for guarding your own scored boxes
W_OFF_GOAL   = 10.0     # Penalty for pushing boxes away from goals (reduces scatter)
W_DEAD       = 10000.0  # Massive penalty for causing a deadlock (permanently stuck box)
W_MOBILITY   = 2.0      # Small reward for having more movement options (prevents getting trapped)

_UNREACHABLE = 9999     # Distance constant for unreachable states
# ─────────────────────────────────────────────────────────────────────────────


def _deadlock_count(boxes: FrozenSet[Tuple[int, int]], board: Board) -> int:
    """
    Count the number of boxes that are permanently deadlocked.
    A box is deadlocked if it cannot be pushed to ANY goal on the board.
    This safely ignores boxes that are ALREADY on goals (since push_dist to its own cell is 0).
    """
    count = 0
    for box in boxes:
        # If the box is unreachable to ALL goals via pushing, it's deadlocked.
        if all(board.push_dist(box, g) >= _UNREACHABLE for g in board.goals):
            count += 1
    return count


def _push_chain_score(
    agent_pos: Tuple[int, int],
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]],
) -> float:
    """
    Reward for being well-positioned to push an unplaced box to a free goal.
    Uses proper `push_dist` to ensure we don't try to push unpushable boxes.
    """
    free_goals = board.goals - occupied_goals
    unplaced = boxes - occupied_goals

    if not free_goals or not unplaced:
        return 0.0

    best_scores = []
    for box in unplaced:
        best_for_box = -1.0
        for goal in free_goals:
            # How many actual PUSHES required to get the box to this goal?
            pushes = board.push_dist(box, goal)
            if pushes >= _UNREACHABLE:
                continue
            
            # Find which direction we should push the box FIRST to optimally reach the goal.
            # A good heuristic is to look at adjacent cells and see which one strictly reduces push_dist.
            best_approach_dist = _UNREACHABLE
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                push_dest = (box[0] + dx, box[1] + dy)
                if board.push_dist(push_dest, goal) < pushes:
                    # To push the box to push_dest, agent must stand at (box[0]-dx, box[1]-dy)
                    approach = (box[0] - dx, box[1] - dy)
                    if approach not in board.walls:
                        d = board.dist(agent_pos, approach)
                        if d < best_approach_dist:
                            best_approach_dist = d

            if best_approach_dist < _UNREACHABLE:
                # Score combines box proximity to goal, and agent proximity to the CORRECT push side
                score = 1.0 / (1 + pushes) + 0.8 / (1 + best_approach_dist)
                if score > best_for_box:
                    best_for_box = score

        if best_for_box > 0:
            best_scores.append(best_for_box)

    best_scores.sort(reverse=True)
    return sum(best_scores[:2])  # Consider top 2 opportunities


def _interact_score(
    agent_pos: Tuple[int, int],
    target_boxes: FrozenSet[Tuple[int, int]],
    board: Board,
) -> float:
    """
    General score for being close to a specific set of boxes (used for stealing or guarding).
    Rewards the agent for being on a valid push-approach cell next to the target box.
    """
    if not target_boxes:
        return 0.0

    best = 0.0
    for box in target_boxes:
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            approach = (box[0] - dx, box[1] - dy)
            push_to = (box[0] + dx, box[1] + dy)
            if approach not in board.walls and push_to not in board.walls:
                d = board.dist(agent_pos, approach)
                if d < _UNREACHABLE:
                    score = 1.0 / (1 + d)
                    if score > best:
                        best = score
    return best


def _off_goal_penalty(
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]],
) -> float:
    """Discourages scattering boxes away from goals into corners."""
    unplaced = boxes - occupied_goals
    if not unplaced:
        return 0.0
    
    total = 0.0
    for box in unplaced:
        # Use push_dist if possible, fallback to walking dist
        min_d = min(board.push_dist(box, g) for g in board.goals)
        if min_d >= _UNREACHABLE:
            min_d = min(board.dist(box, g) for g in board.goals)
        total += min_d
    return total


def _mobility(
    pos: Tuple[int, int],
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
) -> int:
    """Count immediately available non-stuck moves from pos."""
    count = 0
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        dest = (pos[0] + dx, pos[1] + dy)
        if dest in board.walls:
            continue
        if dest in boxes:
            push = (dest[0] + dx, dest[1] + dy)
            if push in board.walls or push in boxes:
                continue
        count += 1
    return count


def competitive_heuristic(
    state: CompetitiveState,
    board: Board,
    perspective: str,
    max_steps: int,
) -> float:
    """
    Returns h(state) from the perspective of the given agent.
    Higher = better for that agent.
    """
    score_a = state.score_a()
    score_b = state.score_b()
    occupied = state.boxes_on_goals_a | state.boxes_on_goals_b

    # Time weight: securing a score earlier is worth more, up to a 50% bonus.
    remaining = max_steps - state.step
    time_weight = 1.0 + 0.5 * (remaining / max(max_steps, 1))

    # Push-chain: positioning to score neutral boxes
    chain_a = _push_chain_score(state.agent_a, state.boxes, board, occupied)
    chain_b = _push_chain_score(state.agent_b, state.boxes, board, occupied)

    # Steal: positioning to knock opponent's box off a goal
    steal_a = _interact_score(state.agent_a, state.boxes_on_goals_b, board)
    steal_b = _interact_score(state.agent_b, state.boxes_on_goals_a, board)
    
    # Guard: positioning to defend own boxes on goals from being stolen
    guard_a = _interact_score(state.agent_a, state.boxes_on_goals_a, board)
    guard_b = _interact_score(state.agent_b, state.boxes_on_goals_b, board)

    off_goal = _off_goal_penalty(state.boxes, board, occupied)
    dead = _deadlock_count(state.boxes, board)

    mob_a = _mobility(state.agent_a, state.boxes, board)
    mob_b = _mobility(state.agent_b, state.boxes, board)

    if perspective == 'A':
        own_score, opp_score = score_a, score_b
        own_chain, opp_chain = chain_a, chain_b
        own_steal, opp_steal = steal_a, steal_b
        own_guard, opp_guard = guard_a, guard_b
        own_mob, opp_mob     = mob_a, mob_b
    else:
        own_score, opp_score = score_b, score_a
        own_chain, opp_chain = chain_b, chain_a
        own_steal, opp_steal = steal_b, steal_a
        own_guard, opp_guard = guard_b, guard_a
        own_mob, opp_mob     = mob_b, mob_a

    return (
        W_SCORE * time_weight * (own_score - opp_score)
        + W_CHAIN * (own_chain - opp_chain)
        + W_STEAL * own_steal - (W_STEAL * opp_steal)
        + W_GUARD * own_guard - (W_GUARD * opp_guard)
        + W_MOBILITY * (own_mob - opp_mob)
        - W_OFF_GOAL * off_goal
        - W_DEAD * dead
        - 0.1 * state.step
    )

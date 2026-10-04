from typing import FrozenSet, Tuple
from src.competitive.state import CompetitiveState, Board

# ── Heuristic Weights ─────────────────────────────────────────────────────────
# These constants define the relative importance of different strategic goals.
# Tweak these to change the AI's behavior.

W_SCORE      = 1000.0   # Reward for each point (box on goal)
W_CHAIN      = 200.0    # Reward for being in position to push an unplaced box to a goal
W_STEAL      = 80.0     # Reward for stealing opponent's scored box (reduced — feasibility-gated)
W_GUARD      = 100.0    # Reward for guarding own scored boxes (raised — losing a point = -1000)
W_OFF_GOAL   = 15.0     # Penalty for pushing boxes far from goals (discourages scatter)
W_DEAD       = 5000.0   # Large penalty for deadlocking a box
W_MOBILITY   = 5.0      # Reward for movement options (important in corridors)
W_PARITY     = 15.0     # Bonus for acting on a priority step (own parity advantage)

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


def _joint_push_chain_score(
    pos_a: Tuple[int, int],
    pos_b: Tuple[int, int],
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]],
    remaining_steps: int,
) -> Tuple[float, float]:
    """
    Evaluates scoring potential for every unplaced box.

    Key improvements over previous version:
    - Uses TOTAL COST = walk_dist(agent, approach) + push_dist(box, goal).
      Each agent independently picks the best goal for them, not just the
      goal with fewest pushes.
    - Feasibility gate: if total_cost > remaining_steps, the agent gets
      scaled-down credit (proportional to how far out of reach it is).
    - Parity tie-breaking: when both agents have equal total cost to the
      same contested box, one of them wins deterministically based on
      the parity rule at the step of arrival.
    """
    free_goals = board.goals - occupied_goals
    unplaced = boxes - occupied_goals

    if not free_goals or not unplaced:
        return 0.0, 0.0

    scores_a: list = []
    scores_b: list = []

    for box in unplaced:
        # Find best (goal, approach) for A: minimise own total cost
        best_cost_a = _UNREACHABLE
        best_cost_b = _UNREACHABLE

        for goal in free_goals:
            # We use the precomputed EXACT step count (walking + pushing + maneuvering)
            cost_a = board.exact_steps(box, pos_a, goal)
            cost_b = board.exact_steps(box, pos_b, goal)
            
            if cost_a < best_cost_a:
                best_cost_a = cost_a
            if cost_b < best_cost_b:
                best_cost_b = cost_b

        if best_cost_a >= _UNREACHABLE and best_cost_b >= _UNREACHABLE:
            continue

        # Feasibility scaling: agent gets less credit if they can't finish in time
        def feasible_score(cost: int) -> float:
            if cost >= _UNREACHABLE:
                return 0.0
            base = 1.0 / (1 + cost)
            if cost <= remaining_steps:
                return base  # fully achievable
            # Partially feasible: scale down proportionally
            return base * (remaining_steps / max(cost, 1))

        score_a = feasible_score(best_cost_a)
        score_b = feasible_score(best_cost_b)

        if best_cost_a < best_cost_b:
            # A clearly wins this race
            scores_a.append(score_a)
        elif best_cost_b < best_cost_a:
            # B clearly wins this race
            scores_b.append(score_b)
        else:
            # True tie in total cost — use parity at the step of arrival
            # arrival_step relative to remaining: who wins the conflict then?
            # remaining_steps % 2 != 0  => A has priority NOW
            # Each step we take, parity flips. After `best_cost_a` steps:
            # remaining at conflict = remaining_steps - best_cost_a
            remaining_at_conflict = remaining_steps - best_cost_a
            a_wins_tie = (remaining_at_conflict % 2 != 0)
            if a_wins_tie:
                scores_a.append(score_a)
            else:
                scores_b.append(score_b)

    scores_a.sort(reverse=True)
    scores_b.sort(reverse=True)
    return sum(scores_a[:2]), sum(scores_b[:2])


def _joint_interact_scores(
    pos_attacker: Tuple[int, int],
    pos_defender: Tuple[int, int],
    target_boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    remaining_steps: int,
) -> Tuple[float, float]:
    """
    Evaluates steal vs guard race for scored boxes.

    Key improvements:
    - Feasibility gate: steal/guard credit scales to zero if impossible in time.
    - Body-blocking: defender standing ON the box cell is a valid full block.
    - Parity tie-breaking: equal distances resolved by priority rule.
    Returns (steal_score, guard_score) from attacker/defender perspectives.
    """
    if not target_boxes:
        return 0.0, 0.0

    steal_total = 0.0
    guard_total = 0.0

    for box in target_boxes:
        # Attacker needs to reach any valid push-approach cell
        best_attack = _UNREACHABLE
        # Defender can block by reaching any push-approach cell OR the box cell itself
        best_defend = _UNREACHABLE

        # Defender body-block: stand on the box's own cell (blocks all 4 push directions)
        d_on_box = board.dist(pos_defender, box)
        if d_on_box < best_defend:
            best_defend = d_on_box

        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            approach = (box[0] - dx, box[1] - dy)
            push_to  = (box[0] + dx, box[1] + dy)
            if approach in board.walls or push_to in board.walls:
                continue
            da = board.dist(pos_attacker, approach)
            dd = board.dist(pos_defender, approach)
            if da < best_attack:
                best_attack = da
            if dd < best_defend:
                best_defend = dd

        def feasible(cost: int) -> float:
            if cost >= _UNREACHABLE:
                return 0.0
            base = 1.0 / (1 + cost)
            if cost <= remaining_steps:
                return base
            return base * (remaining_steps / max(cost, 1))

        if best_attack < best_defend:
            steal_total += feasible(best_attack)
        elif best_defend < best_attack:
            guard_total += feasible(best_defend)
        else:
            # Tie — resolve by parity at arrival
            remaining_at_conflict = remaining_steps - best_attack
            a_wins_tie = (remaining_at_conflict % 2 != 0)
            # Note: attacker here could be A or B — we resolve generically
            # The attacker gets it if remaining_at_conflict is odd (A priority)
            # but we don't know which agent is "A" here.
            # Safe: split credit equally when tied (parity resolve happens in transition)
            steal_total += feasible(best_attack) * 0.5
            guard_total += feasible(best_defend) * 0.5

    return steal_total, guard_total


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
    remaining = max_steps - state.step

    # Time weight: securing a score earlier is worth more, up to a 50% bonus.
    time_weight = 1.0 + 0.5 * (remaining / max(max_steps, 1))

    # Push-chain: positioning to score neutral boxes (with race + horizon awareness)
    chain_a, chain_b = _joint_push_chain_score(
        state.agent_a, state.agent_b, state.boxes, board, occupied, remaining
    )

    # Steal vs Guard races (feasibility-gated by remaining steps)
    steal_a, guard_b = _joint_interact_scores(
        state.agent_a, state.agent_b, state.boxes_on_goals_b, board, remaining
    )
    steal_b, guard_a = _joint_interact_scores(
        state.agent_b, state.agent_a, state.boxes_on_goals_a, board, remaining
    )

    mob_a = _mobility(state.agent_a, state.boxes, board)
    mob_b = _mobility(state.agent_b, state.boxes, board)

    # Shared board penalties (applied symmetrically — reduce total resource pool damage)
    deadlocks  = _deadlock_count(state.boxes, board)
    off_goal   = _off_goal_penalty(state.boxes, board, occupied)

    # Parity bonus: on my priority step, any conflict resolves in my favour.
    # This gives a small nudge to be aggressive when priority is mine.
    a_has_priority = (remaining % 2 != 0)
    if perspective == 'A':
        parity_bonus = W_PARITY if a_has_priority else -W_PARITY
    else:
        parity_bonus = W_PARITY if not a_has_priority else -W_PARITY

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
        W_SCORE    * time_weight * (own_score - opp_score)
        + W_CHAIN  * (own_chain - opp_chain)
        + W_STEAL  * own_steal  - W_STEAL * opp_steal
        + W_GUARD  * own_guard  - W_GUARD * opp_guard
        + W_MOBILITY * (own_mob - opp_mob)
        + parity_bonus
        - W_DEAD     * deadlocks
        - W_OFF_GOAL * off_goal
    )

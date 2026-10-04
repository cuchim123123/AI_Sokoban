from typing import FrozenSet, Tuple
from scipy.optimize import linear_sum_assignment

from src.competitive.state import CompetitiveState, Board


# ── Heuristic Weights ─────────────────────────────────────────────────────────
# These constants define the relative importance of different strategic goals.
# Tweak these to change the AI's behavior.


W_SCORE      = 1000.0   # Reward for each point (box on goal)
W_CHAIN      = 2.0      # Linear multiplier per step closer. Max = 2.0 * 200 = 400
W_STEAL     = 3.0      # Makes approach progress visible beside chain/guard terms
W_GUARD      = 1.5      # Reward for maintaining access to credited boxes
W_OFF_GOAL   = 15.0     # Penalty for pushing boxes far from goals
W_DEAD       = 5000.0   # Large penalty for deadlocking a box
W_MOBILITY   = 0.0      # Disabled — fights exact_step gradient


_UNREACHABLE = 9999     # Distance constant for unreachable states


# ─────────────────────────────────────────────────────────────────────────────


def _deadlock_count(
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]] = frozenset(),
) -> int:
    """
    Count the number of boxes that are permanently deadlocked.

    A box is deadlocked if it cannot be pushed to ANY goal on the board.

    This safely ignores boxes that are ALREADY on goals (since push_dist to its own cell is 0).
    """
    count = 0

    available_goals = board.goals - occupied_goals

    for box in boxes:
        if box in occupied_goals:
            continue

        # 1. Static deadlock (precomputed unreachable)
        if all(
            board.push_dist(box, g) >= _UNREACHABLE
            for g in available_goals
        ):
            count += 1
            continue

        # 2. Dynamic 2-box deadlock (adjacent boxes on a wall)
        bx, by = box
        if box in board.goals:
            continue

        # Horizontal adjacency against vertical walls
        if (bx + 1, by) in boxes and (bx + 1, by) not in board.goals:
            # Check if there is a continuous wall above OR below both boxes
            if ((bx, by + 1) in board.walls and (bx + 1, by + 1) in board.walls) or \
               ((bx, by - 1) in board.walls and (bx + 1, by - 1) in board.walls):
                count += 1
                continue

        # Vertical adjacency against horizontal walls
        if (bx, by + 1) in boxes and (bx, by + 1) not in board.goals:
            # Check if there is a continuous wall left OR right of both boxes
            if ((bx + 1, by) in board.walls and (bx + 1, by + 1) in board.walls) or \
               ((bx - 1, by) in board.walls and (bx - 1, by + 1) in board.walls):
                count += 1
                continue

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

    def assignment_costs(pos):
        box_list = list(unplaced)
        goal_list = list(free_goals)
        costs = [
            [board.exact_steps(box, pos, goal) for goal in goal_list]
            for box in box_list
        ]
        row_ind, col_ind = linear_sum_assignment(costs)
        return {
            box_list[row]: costs[row][col]
            for row, col in zip(row_ind, col_ind)
            if costs[row][col] < _UNREACHABLE
        }

    assigned_a = assignment_costs(pos_a)
    assigned_b = assignment_costs(pos_b)
    scores_a: list = []
    scores_b: list = []

    for box in unplaced:
        best_cost_a = assigned_a.get(box, _UNREACHABLE)
        best_cost_b = assigned_b.get(box, _UNREACHABLE)

        if best_cost_a >= _UNREACHABLE and best_cost_b >= _UNREACHABLE:
            continue

        # Feasibility scaling: agent gets less credit if they can't finish in time
        def feasible_score(cost: int) -> float:
            if cost >= _UNREACHABLE:
                return 0.0

            # Linear gradient ensures constant reward per step taken
            base = float(max(0, 200 - cost))

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

            base = float(max(0, 200 - cost))

            if cost <= remaining_steps:
                return base

            return base * (remaining_steps / max(cost, 1))

        if best_attack < best_defend:
            steal_total += feasible(best_attack)

            # Horizon fix: If attacker wins the race, defender WILL lose the box.
            # Price the 1000 point score swing immediately to prevent mirage nodes.
            steal_total += W_SCORE

        elif best_defend < best_attack:
            # The defender currently wins the race, but the attacker still
            # needs a gradient toward the box. Otherwise every losing attack
            # looks identical and the trailing agent never pursues it.
            steal_total += feasible(best_attack)
            guard_total += feasible(best_defend)

        else:
            # Tie — resolve by parity at arrival
            remaining_at_conflict = remaining_steps - best_attack
            a_wins_tie = (remaining_at_conflict % 2 != 0)

            # Split the expected value of the box loss
            steal_total += feasible(best_attack) * 0.5 + (W_SCORE * 0.5)
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

    available_goals = board.goals - occupied_goals
    if not available_goals:
        return float(_UNREACHABLE * len(unplaced))

    for box in unplaced:
        # Use push_dist if possible, fallback to walking dist
        min_d = min(board.push_dist(box, g) for g in available_goals)

        if min_d >= _UNREACHABLE:
            min_d = min(board.dist(box, g) for g in available_goals)

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

    # Push-chain: globally matched legal walk-plus-push assignments.
    chain_a, chain_b = _joint_push_chain_score(
        state.agent_a,
        state.agent_b,
        state.boxes,
        board,
        occupied,
        remaining,
    )

    # Score the race for opponent boxes as well as defending own credited
    # boxes. Without this term, stealing is legal in the transition but has
    # zero value in the search evaluation.
    steal_a, guard_b = _joint_interact_scores(
        state.agent_a,
        state.agent_b,
        state.boxes_on_goals_b,
        board,
        remaining,
    )
    steal_b, guard_a = _joint_interact_scores(
        state.agent_b,
        state.agent_a,
        state.boxes_on_goals_a,
        board,
        remaining,
    )

    mob_a = _mobility(state.agent_a, state.boxes, board)
    mob_b = _mobility(state.agent_b, state.boxes, board)

    # Shared board penalties (applied symmetrically — reduce total resource pool damage)
    deadlocks = _deadlock_count(state.boxes, board, occupied)
    off_goal = _off_goal_penalty(state.boxes, board, occupied)

    if perspective == 'A':
        own_score, opp_score = score_a, score_b
        own_chain, opp_chain = chain_a, chain_b
        own_steal, opp_steal = steal_a, steal_b
        own_guard, opp_guard = guard_a, guard_b
        own_mob, opp_mob = mob_a, mob_b
    else:
        own_score, opp_score = score_b, score_a
        own_chain, opp_chain = chain_b, chain_a
        own_steal, opp_steal = steal_b, steal_a
        own_guard, opp_guard = guard_b, guard_a
        own_mob, opp_mob = mob_b, mob_a

    return (
        W_SCORE * (own_score - opp_score)
        + W_CHAIN * (own_chain - opp_chain)
        + W_STEAL * (own_steal - opp_steal)
        + W_GUARD * (own_guard - opp_guard)
        + W_MOBILITY * (own_mob - opp_mob)
        - W_DEAD * deadlocks
        - W_OFF_GOAL * off_goal
    )
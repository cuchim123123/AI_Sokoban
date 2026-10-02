"""
Competitive evaluation heuristic for GBFS.

h(s) from Agent A's perspective:
    W_SCORE  * time_weight * (score_A - score_B)
  + W_PUSH   * (score_A - score_B)           [stacking bonus]
  + W_CHAIN  * (chain_A - chain_B)           [agent→box→goal chain proximity]
  + W_STEAL  * steal_opportunity_A
  - W_DEFEND * steal_opportunity_B
  - W_DEAD   * deadlock_count
  + W_MOB    * (mobility_A - mobility_B)

The key is W_CHAIN uses the full push chain:
    chain = 1/(1 + dist(agent, push_pos)) + 1/(1 + dist(box, goal))
where push_pos is the cell BEHIND the box relative to the goal.
"""
from typing import Tuple, FrozenSet, Optional
from src.competitive.state import CompetitiveState, Board

W_SCORE    = 200
W_PUSH     = 60    # very strong reward per box on goal
W_CHAIN    = 30    # reward for agent-box-goal chain proximity
W_STEAL    = 5     # small steal incentive (steal only when worthwhile)
W_DEFEND   = 4
W_OFF_GOAL = 8     # penalty per unplaced box that is far from all goals
W_MOB      = 1
W_DEAD     = 60





def _is_corner_deadlock(box: Tuple[int, int], board: Board) -> bool:
    if box in board.goals:
        return False
    bx, by = box
    wall_h = (bx - 1, by) in board.walls or (bx + 1, by) in board.walls
    wall_v = (bx, by - 1) in board.walls or (bx, by + 1) in board.walls
    return wall_h and wall_v


def _push_chain_score(
    agent_pos: Tuple[int, int],
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]],
) -> float:
    """
    Directional push-chain score.

    For each (box, goal) pair compute:
        push_dir     = unit vector from box toward goal (clamped to 4 directions)
        approach_pos = box - push_dir  (cell agent must stand on to push)
        score = 1/(1 + dist(box, goal)) + 0.8/(1 + dist(agent, approach_pos))

    Using approach_pos instead of raw box position means the heuristic
    rewards being on the CORRECT SIDE of the box, eliminating wasted
    repositioning moves.
    """
    free_goals = board.goals - occupied_goals
    unplaced   = boxes - occupied_goals

    if not free_goals or not unplaced:
        return 0.0

    pairs = []
    for box in unplaced:
        best_score = -1.0
        for goal in free_goals:
            box_to_goal_d = board.dist(box, goal)

            # Determine the dominant push direction (horizontal or vertical)
            dx = goal[0] - box[0]
            dy = goal[1] - box[1]

            if abs(dx) >= abs(dy):
                # Horizontal push is primary
                push_dir = (1 if dx > 0 else -1, 0)
            else:
                # Vertical push is primary
                push_dir = (0, 1 if dy > 0 else -1)

            # Cell the agent must occupy to perform this push
            approach = (box[0] - push_dir[0], box[1] - push_dir[1])

            # If approach is a wall, try the other axis
            if approach in board.walls:
                if abs(dx) >= abs(dy):
                    push_dir = (0, 1 if dy > 0 else (-1 if dy < 0 else 1))
                else:
                    push_dir = (1 if dx > 0 else (-1 if dx < 0 else 1), 0)
                approach = (box[0] - push_dir[0], box[1] - push_dir[1])

            agent_to_approach = board.dist(agent_pos, approach)
            score = 1.0 / (1 + box_to_goal_d) + 0.8 / (1 + agent_to_approach)

            if score > best_score:
                best_score = score

        if best_score > 0:
            pairs.append(best_score)

    pairs.sort(reverse=True)
    return sum(pairs[:2])


def _off_goal_penalty(
    boxes: FrozenSet[Tuple[int, int]],
    board: Board,
    occupied_goals: FrozenSet[Tuple[int, int]],
) -> float:
    """
    For each unplaced box, compute its distance to the nearest goal.
    Return a penalty proportional to that distance — discourages pushing
    boxes away from goals into useless corners.
    """
    unplaced = boxes - occupied_goals
    if not unplaced:
        return 0.0
    total = 0.0
    for box in unplaced:
        min_d = min(board.dist(box, g) for g in board.goals)
        total += min_d
    return total



def _steal_score(
    agent_pos: Tuple[int, int],
    opponent_completed: FrozenSet[Tuple[int, int]],
    board: Board,
    free_goals: FrozenSet[Tuple[int, int]],
) -> float:
    """
    Directional steal score.

    For each opponent completed box:
      - If there is a free goal, find the best push direction that moves the
        box TOWARD that free goal, and reward agent proximity to the correct
        approach position (same logic as _push_chain_score).
      - If no free goal exists, fall back to rewarding any pushable direction
        (at least get it off the goal so our score increases).

    This prevents the agent from approaching from the wrong side and pushing
    the stolen box into a deadlock wall.
    """
    if not opponent_completed:
        return 0.0

    best = 0.0
    for box in opponent_completed:
        if free_goals:
            # Find push direction that moves box closest to a free goal
            box_best = -1.0
            for goal in free_goals:
                dx = goal[0] - box[0]
                dy = goal[1] - box[1]
                if abs(dx) >= abs(dy):
                    push_dir = (1 if dx > 0 else -1, 0)
                else:
                    push_dir = (0, 1 if dy > 0 else -1)

                approach = (box[0] - push_dir[0], box[1] - push_dir[1])
                push_to  = (box[0] + push_dir[0], box[1] + push_dir[1])

                # Approach must be floor, push destination must be floor
                if approach in board.walls or push_to in board.walls:
                    # Try opposite axis
                    if abs(dx) >= abs(dy):
                        push_dir = (0, 1 if dy > 0 else (-1 if dy < 0 else 1))
                    else:
                        push_dir = (1 if dx > 0 else (-1 if dx < 0 else 1), 0)
                    approach = (box[0] - push_dir[0], box[1] - push_dir[1])
                    push_to  = (box[0] + push_dir[0], box[1] + push_dir[1])
                    if approach in board.walls or push_to in board.walls:
                        continue

                d = board.dist(agent_pos, approach)
                # Reward both proximity to approach AND box proximity to goal
                box_to_goal = board.dist(box, goal)
                score = 0.7 / (1 + d) + 0.3 / (1 + box_to_goal)
                box_best = max(box_best, score)

            if box_best > 0:
                best = max(best, box_best)
        else:
            # No free goal — just get the box off the goal (any pushable direction)
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                approach = (box[0] - dx, box[1] - dy)
                push_to  = (box[0] + dx, box[1] + dy)
                if approach not in board.walls and push_to not in board.walls:
                    d = board.dist(agent_pos, approach)
                    best = max(best, 0.5 / (1 + d))
                    break

    return best



def _mobility(
    pos: Tuple[int, int],
    other_pos: Tuple[int, int],
    boxes: FrozenSet,
    board: Board,
) -> int:
    count = 0
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        dest = (pos[0] + dx, pos[1] + dy)
        if dest in board.walls or dest == other_pos:
            continue
        if dest in boxes:
            pd = (dest[0] + dx, dest[1] + dy)
            if pd in board.walls or pd in boxes or pd == other_pos:
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
    occupied_goals = state.boxes_on_goals_a | state.boxes_on_goals_b

    remaining = max_steps - state.step
    time_weight = 1.0 + 0.5 * (remaining / max(max_steps, 1))

    # Push-chain proximity (agent → box → goal)
    chain_a = _push_chain_score(state.agent_a, state.boxes, board, occupied_goals)
    chain_b = _push_chain_score(state.agent_b, state.boxes, board, occupied_goals)

    # Steal opportunity (directional — approach from correct side toward a free goal)
    free_goals = board.goals - occupied_goals
    steal_a = _steal_score(state.agent_a, state.boxes_on_goals_b, board, free_goals)
    steal_b = _steal_score(state.agent_b, state.boxes_on_goals_a, board, free_goals)

    # Off-goal scatter penalty (same for both — applied symmetrically)
    off_goal = _off_goal_penalty(state.boxes, board, occupied_goals)

    # Deadlock
    dead = sum(1 for box in state.boxes if _is_corner_deadlock(box, board))

    # Mobility
    mob_a = _mobility(state.agent_a, state.agent_b, state.boxes, board)
    mob_b = _mobility(state.agent_b, state.agent_a, state.boxes, board)

    h_a = (
        W_SCORE    * time_weight * (score_a - score_b)
        + W_PUSH   * (score_a - score_b)
        + W_CHAIN  * (chain_a - chain_b)
        + W_STEAL  * steal_a
        - W_DEFEND * steal_b
        - W_OFF_GOAL * off_goal     # discourages pushing boxes into useless spots
        - W_DEAD   * dead
        + W_MOB    * (mob_a - mob_b)
    )

    return h_a if perspective == 'A' else -h_a

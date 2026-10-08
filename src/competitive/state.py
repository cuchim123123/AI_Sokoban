from enum import Enum
from typing import FrozenSet, Tuple, Optional


class Action(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    WAIT  = (0, 0)   # no-op, used as safe fallback

    def __str__(self):
        return self.name.capitalize()


class Board:
    """Static board information — walls, goals, dimensions. Created once and shared."""
    __slots__ = ("walls", "goals", "width", "height", "floor_cells", "distances",
                 "push_costs", "exact_step_costs", "serial", "neighbors")

    # Monotone board identity. The evaluation and transition memo caches are
    # keyed by positions/boxes/credits, which are only meaningful together
    # with the board they were computed on (different walls/goals give
    # different distances, diversion choices and deadlock verdicts). The
    # serial distinguishes boards even when they are recycled at the same
    # memory address, so a stale cache entry can never be served.
    _serial_counter = 0

    def __init__(
        self,
        walls: FrozenSet[Tuple[int, int]],
        goals: FrozenSet[Tuple[int, int]],
        width: int,
        height: int,
    ):
        Board._serial_counter += 1
        self.serial = Board._serial_counter
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.width = width
        self.height = height
        # Pre-compute all non-wall cells for fast membership tests
        self.floor_cells: FrozenSet[Tuple[int, int]] = frozenset(
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.walls
        )
        self.neighbors = {
            (x, y): tuple((x + dx, y + dy)
                          for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0))
                          if (x + dx, y + dy) in self.floor_cells)
            for x, y in self.floor_cells
        }
        
        # Precompute all-pairs shortest path (BFS) for legal distances
        from collections import deque
        self.distances = {}
        for start in self.floor_cells:
            distances = {start: 0}
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if nxt in self.floor_cells and nxt not in distances:
                        distances[nxt] = distances[curr] + 1
                        queue.append(nxt)
            self.distances[start] = distances

        # Precompute push distances
        from src.single.heuristics.push_distance import precompute_push_costs
        self.push_costs = precompute_push_costs(self)

        # Precompute EXACT minimum steps (walking + pushing) for a player to deliver a box
        self.exact_step_costs = self._precompute_exact_steps()

    def _precompute_exact_steps(self):
        costs = {}
        for goal in self.goals:
            # state: (bx, by, px, py)
            goal_costs = {}
            from collections import deque
            queue = deque()
            
            # Initialize targets: box at goal, player anywhere adjacent
            for dx, dy in ((0,1), (0,-1), (1,0), (-1,0)):
                px, py = goal[0] + dx, goal[1] + dy
                if (px, py) in self.floor_cells:
                    state = (goal[0], goal[1], px, py)
                    goal_costs[state] = 0
                    queue.append(state)
                    
            while queue:
                bx, by, px, py = queue.popleft()
                curr_cost = goal_costs[(bx, by, px, py)]
                
                # 1. Reverse a walk: player walked from (nx, ny) to (px, py)
                for dx, dy in ((0,1), (0,-1), (1,0), (-1,0)):
                    nx, ny = px + dx, py + dy
                    if (nx, ny) in self.floor_cells and (nx, ny) != (bx, by):
                        nstate = (bx, by, nx, ny)
                        if nstate not in goal_costs:
                            goal_costs[nstate] = curr_cost + 1
                            queue.append(nstate)
                            
                # 2. Reverse a push: player pushed box from (px, py) to (bx, by)
                if abs(bx - px) + abs(by - py) == 1:
                    dx = bx - px
                    dy = by - py
                    prev_bx, prev_by = px, py
                    prev_px, prev_py = px - dx, py - dy
                    if (prev_px, prev_py) in self.floor_cells:
                        nstate = (prev_bx, prev_by, prev_px, prev_py)
                        if nstate not in goal_costs:
                            goal_costs[nstate] = curr_cost + 1
                            queue.append(nstate)
                            
            costs[goal] = goal_costs
        return costs

    def exact_steps(self, box: Tuple[int, int], player: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Returns the exact min steps (walk + push) to deliver box to goal, or 9999."""
        return self.exact_step_costs.get(goal, {}).get((box[0], box[1], player[0], player[1]), 9999)

    def dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Returns the legal shortest path distance between a and b, or 9999 if unreachable."""
        return self.distances.get(a, {}).get(b, 9999)

    def push_dist(self, box: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Returns min pushes to move a box to the goal, or 9999 if unreachable."""
        return self.push_costs.get(goal, {}).get(box, 9999)

class CompetitiveState:
    """
    Full competitive game state.

    Boxes are SHARED — no permanent ownership.
    `boxes_on_goals_a` and `boxes_on_goals_b` track *temporary* credit:
      - A box currently sitting on a goal is credited to whoever placed it last.
      - Pushing it off the goal strips the credit immediately.
      - There is no overlap: a goal cell is credited to at most one agent.
    """
    __slots__ = (
        "agent_a", "agent_b",
        "boxes",
        "boxes_on_goals_a", "boxes_on_goals_b",
        "step",
        "_hash",
    )

    def __init__(
        self,
        agent_a: Tuple[int, int],
        agent_b: Tuple[int, int],
        boxes: FrozenSet[Tuple[int, int]],
        boxes_on_goals_a: FrozenSet[Tuple[int, int]],
        boxes_on_goals_b: FrozenSet[Tuple[int, int]],
        step: int,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.boxes = frozenset(boxes)
        self.boxes_on_goals_a = frozenset(boxes_on_goals_a)
        self.boxes_on_goals_b = frozenset(boxes_on_goals_b)
        self.step = step
        # Pre-compute hash for O(1) repeated lookups (used heavily by search)
        self._hash = hash((
            agent_a, agent_b,
            self.boxes,
            self.boxes_on_goals_a,
            self.boxes_on_goals_b,
            step,
        ))

    # ------------------------------------------------------------------
    # Identity / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompetitiveState):
            return False
        return (
            self.agent_a == other.agent_a
            and self.agent_b == other.agent_b
            and self.boxes == other.boxes
            and self.boxes_on_goals_a == other.boxes_on_goals_a
            and self.boxes_on_goals_b == other.boxes_on_goals_b
            and self.step == other.step
        )

    def __hash__(self) -> int:
        return self._hash

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def score_a(self) -> int:
        return len(self.boxes_on_goals_a)

    def score_b(self) -> int:
        return len(self.boxes_on_goals_b)

    def is_terminal(self, max_steps: int) -> bool:
        return self.step >= max_steps

    def __repr__(self) -> str:
        return (
            f"CompetitiveState(step={self.step}, "
            f"A={self.agent_a} score={self.score_a()}, "
            f"B={self.agent_b} score={self.score_b()}, "
            f"boxes={len(self.boxes)})"
        )

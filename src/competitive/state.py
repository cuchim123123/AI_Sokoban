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
    __slots__ = ("walls", "goals", "width", "height", "floor_cells", "distances")

    def __init__(
        self,
        walls: FrozenSet[Tuple[int, int]],
        goals: FrozenSet[Tuple[int, int]],
        width: int,
        height: int,
    ):
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

    def dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Returns the legal shortest path distance between a and b, or 9999 if unreachable."""
        return self.distances.get(a, {}).get(b, 9999)

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
        # Pre-compute hash for O(1) repeated lookups (used heavily by GBFS visited set)
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

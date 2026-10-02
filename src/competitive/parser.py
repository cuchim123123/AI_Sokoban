"""
Competitive map parser.

Map format (superset of single-agent format):
    %   wall
    .   empty floor
    D   goal
    B   box (initially uncredited to either agent)
    A   Agent A starting position
    C   Agent B starting position   (C = "Challenger")
    space = floor (same as '.')

The parser returns (CompetitiveState, Board).
"""
from typing import Tuple
from src.competitive.state import Board, CompetitiveState


def parse_competitive_map(path: str) -> Tuple[CompetitiveState, Board]:
    with open(path, "r") as f:
        lines = f.read().splitlines()

    walls = set()
    goals = set()
    boxes = set()
    agent_a = None
    agent_b = None

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == '%':
                walls.add((x, y))
            elif ch == 'D':
                goals.add((x, y))
            elif ch == 'B':
                boxes.add((x, y))
            elif ch == 'A':
                agent_a = (x, y)
            elif ch == 'C':
                agent_b = (x, y)
            # ' ' and '.' are treated as floor — nothing to do

    if agent_a is None:
        raise ValueError(f"Map '{path}' has no Agent A start position (A).")
    if agent_b is None:
        raise ValueError(f"Map '{path}' has no Agent B start position (C).")
    if not goals:
        raise ValueError(f"Map '{path}' has no goal positions (D).")

    board = Board(
        walls=frozenset(walls),
        goals=frozenset(goals),
        width=width,
        height=height,
    )

    initial_state = CompetitiveState(
        agent_a=agent_a,
        agent_b=agent_b,
        boxes=frozenset(boxes),
        boxes_on_goals_a=frozenset(),
        boxes_on_goals_b=frozenset(),
        step=0,
    )

    return initial_state, board

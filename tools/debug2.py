"""Inspect a reproducible dense-goals decision without patching the planner."""
import json
import os
import sys

# Allow running from anywhere: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.competitive.parser import parse_competitive_map
from src.competitive.agent_b import AgentB
from src.competitive.state import CompetitiveState


def main():
    _, board = parse_competitive_map("maps/competitive/dense_goals.txt")
    state = CompetitiveState((4, 2), (10, 3),
                             {(4, 4), (7, 4), (5, 2), (10, 4), (8, 2)},
                             (), (), 2)
    agent = AgentB()
    action = agent.choose_action(state, board, 40)
    print(f"Agent B chose: {action}")
    print(json.dumps(agent.last_search, indent=2))


if __name__ == "__main__":
    main()

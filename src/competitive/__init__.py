"""Two-agent simultaneous-move Sokoban: domain, agents, joint engine, GUI."""

from src.competitive.agent_a import AgentA, best_action
from src.competitive.agent_b import AgentB
from src.competitive.parser import parse_competitive_map
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (
    conflict_winner,
    resolve_joint_action_outcome,
)

__all__ = [
    "Action", "Board", "CompetitiveState",
    "parse_competitive_map",
    "AgentA", "AgentB", "best_action",
    "resolve_joint_action_outcome", "conflict_winner",
]

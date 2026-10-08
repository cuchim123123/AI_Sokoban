"""Single-agent heuristics: push costs, Hungarian matching, deadlock detection."""

from src.single.heuristics.deadlock import is_deadlock
from src.single.heuristics.matching import MatchingHeuristic
from src.single.heuristics.push_distance import precompute_push_costs

__all__ = ["precompute_push_costs", "MatchingHeuristic", "is_deadlock"]

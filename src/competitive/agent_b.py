"""
Agent B controller.

Identical search to Agent A (`best_action` in agent_a.py), opposite
perspective: everything in the evaluation is mirrored, so both agents share
one implementation and one set of caches.
"""

from src.competitive.agent_a import AgentA


class AgentB(AgentA):
    """Agent B controller."""

    perspective = "B"

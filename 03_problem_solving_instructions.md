# Instructions for an AI Agent Analyzing This Assignment

## Current scope

Focus ONLY on the original single-agent Sokoban requirements 1--5.

Do not analyze or design the competitive two-agent version unless
explicitly requested later.

## Source-grounded facts

Use the supplied assignment as the authoritative specification.

Known facts: - One agent in the original problem. - Boxes are moved to
designated positions. - Input is a layout-file path. - Output is
directional actions and total cost. - UCS and A\* are required. -
Manhattan and Euclidean heuristics are forbidden. - Time/space
comparison is required. - Heuristic admissibility and consistency must
be discussed and experimentally verified. - Pygame GUI is required. -
OOP structure is required.

## Reasoning priorities

When proposing a heuristic:

1.  Start from the exact state and cost model.
2.  Determine what one search edge/action costs.
3.  Do not use Manhattan or Euclidean distance.
4.  Use actual Sokoban constraints: walls, pushing direction, player
    reachability and box-goal feasibility.
5.  Separate box-push cost from player-walking cost.
6.  Analyze admissibility formally.
7.  Analyze consistency formally.
8.  Only then discuss implementation and optimization.

## Do not make unsupported assumptions

In particular, do not automatically assume: - a push has cost 1 if the
assignment's cost model counts every player action - a box-goal distance
is admissible merely because it is based on BFS - a deadlock penalty
preserves admissibility - a matching heuristic is consistent without
proof - competitive-game rules apply to the single-agent problem

## Desired output from future analysis

When asked to propose a heuristic, provide: 1. exact mathematical
definition 2. state/input required 3. algorithm for calculating it 4.
why it is a lower bound, if claimed admissible 5. consistency argument,
if claimed consistent 6. counterexamples if a property fails 7.
computational complexity 8. experimental method to verify the property
9. integration into A\*

Avoid buzzwords. The objective is a technically defensible heuristic
suitable for Requirement 4.

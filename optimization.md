# Massive AI Search Optimization Plan

This document outlines a massive optimization plan for the `competitive` AI agents, drawing on classical game-tree search concepts (Transposition Tables, Alpha-Beta Pruning, Action Ordering) tailored specifically to the Sokoban architecture.

## 1. Global Transposition Table (TT)

Currently, the search cache (`cache = {}`) is local to a single turn. It is thrown away after the turn ends, meaning the AI re-evaluates identical states from scratch on the very next turn.

**Implementation Plan:**
- Move the Transposition Table to a class-level or instance-level dictionary (`self.tt = {}`) on the Agent.
- Limit memory usage by utilizing an `OrderedDict` (LRU Cache) or periodically clearing it if it exceeds a certain bound (e.g., 500,000 entries).
- **Structure**: `self.tt[state] = { 'depth': depth, 'value': value, 'best_act': act, 'type': 'exact' | 'lower' | 'upper' }`.
- When encountering a state in search, if `state in self.tt` and `self.tt[state]['depth'] >= current_depth`, return the cached value directly.

## 2. Principal Variation & Action Ordering

Action ordering is the single biggest multiplier for Alpha-Beta pruning efficiency. Right now, `_valid_actions` returns actions in an arbitrary static order.

**Implementation Plan:**
- **PV (Principal Variation) Re-use:** When Iterative Deepening searches depth `d`, it should query the TT for the best action found at depth `d-1`. It must explore this PV action *first*.
- **Heuristic Ordering:** For the remaining actions, pre-evaluate them using the fast `competitive_heuristic` (or simply their proximity to goals) and sort them from best to worst before expanding their full subtrees.
- *Why?* If you find a massive score increase on the very first branch, the pruning threshold (Alpha) jumps up immediately, pruning the rest of the useless branches instantly.

## 3. True Alpha-Beta Pruning (for Maximin)

Currently, the `maximin` AI only looks 1-ply deep for the opponent's response (`Action.WAIT` or a 1-step heuristic response). To be a true adversarial AI, it should use proper Minimax with Alpha-Beta pruning.

**Implementation Plan:**
- Rewrite the search to alternate turns: `Max Node (A's turn) -> Min Node (B's turn) -> Max Node (A's turn)`.
- Pass `alpha` (minimum guaranteed score for A) and `beta` (maximum guaranteed score for B).
- **Pruning Logic**: If the evaluation of a branch is `>= beta` at a Max node, prune it (the opponent would never let you reach this state anyway).
- For the `aggressive` AI (single-agent search), Alpha-Beta reduces to Branch & Bound. If we can establish a strict upper-bound on future scores, we can prune max-nodes that fall below `alpha`.

## 4. Evaluation Caching

The `competitive_heuristic` is expensive (it runs `_chain_score`, calculates distances, and counts deadlocks).
- Create a dedicated `heuristic_cache` (independent of the TT).
- Since heuristic values don't depend on `depth`, `heuristic_cache[state]` is universally valid and never expires until the game ends.
- *Note:* Push-distance precomputation (`Board.push_dist`) is already efficiently implemented and cached in the `Board` initialization, fulfilling the highest-value optimization from the reference material.

## Summary of Impact
With a Global TT and PV Action Ordering, the branching factor of the search tree drops dramatically. An AI that currently reaches Depth 12 in 0.95s might effortlessly reach Depth 16-20 in the same time limit, resulting in vastly superior strategic play.

  # Sokoban Single-Agent Implementation Plan

## 1. Requirements & Scope
- **Problem**: Original single-agent Sokoban.
- **Input**: Map text file (`%` wall, `A` agent, `B` box, `D` goal, `C` box on goal, ` ` empty).
- **Output**: Sequence of player actions (`North`, `East`, `West`, `South`) and total cost.
- **Cost Model**: Each player movement (step) has a cost of 1.
- **Algorithms**: Uniform Cost Search (UCS) and A*.
- **Heuristic**: No Manhattan/Euclidean distance allowed. Must be empirically verified for admissibility and consistency.
- **Experiments**: Time and space complexity comparison between UCS and A*. Heuristic property verification.
- **GUI**: Pygame interface to visualize algorithms, step forward/backward.

## 2. Architecture & Modules
- `src/core/state.py`: Represents the Sokoban state (Agent position, Box positions) immutably for hashing.
- `src/core/parser.py`: Parses the map file and creates the initial state.
- `src/core/actions.py`: Validates and applies actions, yielding successors.
- `src/search/node.py`: Search node (state, parent, action, path_cost).
- `src/search/ucs.py`: UCS algorithm.
- `src/search/astar.py`: A* algorithm.
- `src/heuristics/deadlock.py`: Static deadlock detection (corners, wall lines, 2x2 blocks).
- `src/heuristics/push_distance.py`: Calculates minimum legal push distance using reverse BFS from goals.
- `src/heuristics/matching.py`: Bipartite matching (Hungarian algorithm) for boxes to goals.
- `src/experiments/benchmark.py`: Compare UCS vs A* (time, space, expanded nodes).
- `src/experiments/verify_heuristic.py`: Check admissibility (`h(s) <= h*(s)`) and consistency (`h(s) <= 1 + h(s')`).
- `src/gui/app.py`: Pygame application.

## 3. Heuristic Design: Legal Push Assignment
**Definition**: For each box `b` and goal `d`, let `PushCost(b, d)` be the minimum number of legal pushes required to move a box from `b` to `d` ignoring other boxes but respecting walls. We construct a bipartite graph between current box positions and goal positions, and find the minimum weight perfect matching using the Hungarian algorithm. If a box is deadlocked (cannot reach any goal), the heuristic value is infinity.

**Admissibility**: Every push action requires exactly one player movement. A sequence of player movements that solves the level must include at least as many pushes as the optimal assignment of boxes to goals. Therefore, the total number of pushes in a solution is `>=` the minimum push matching cost. Since each push corresponds to a player move, total player moves `>=` total pushes `>= h(s)`. Thus `h(s)` is an admissible lower bound.

**Consistency**: A single player action can move a box by at most 1 cell (1 push). Therefore, `h(s)` can decrease by at most 1 after a single action. `h(s) - h(s') <= 1 == cost(s, a, s')`. Thus, the heuristic is consistent.

## 4. Execution Workflow (Iterative)
1. **Core Domain**: Implement parser, state, actions, and goal test. Write unit tests.
2. **UCS**: Implement and test UCS on simple maps.
3. **Heuristics & Deadlocks**: Implement push cost BFS, deadlock detection, Hungarian matching.
4. **A***: Implement A*, test correctness.
5. **Experiments**: Generate exact costs with UCS on small maps, check `h(s) <= h*(s)` and `h(s) <= 1 + h(s')`. Benchmark performance against UCS.
6. **GUI**: Build Pygame renderer.
7. **Documentation**: Write final reports.

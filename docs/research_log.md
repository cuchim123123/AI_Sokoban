# Research & Decision Log

## Decision 1: State Representation
- **Problem**: How to represent the state to allow fast hashing and equality checks for the `explored` set?
- **Decision**: Separate static elements (walls, goals, dimensions) into a `Board` class and dynamic elements (agent, boxes) into a `GameState` class. Use `frozenset` for boxes and tuples for the agent position.
- **Reason**: `frozenset` is immutable and hashable, making the state object safely usable as a dictionary key for duplicate detection.
- **Consequences**: Extremely fast state deduplication. Requires recreating the box set on every push, but the overhead is negligible for typical Sokoban instances.

## Decision 2: Search Node Cost
- **Problem**: Should the path cost represent player actions or box pushes?
- **Decision**: The cost model strictly tracks player actions (North, South, East, West), each costing 1.
- **Reason**: The assignment output format is player directional actions, not box moves.
- **Consequences**: A heuristic based solely on box pushes acts as a lower bound, requiring careful mathematical justification.

## Decision 3: Heuristic Selection
- **Problem**: Finding an admissible and consistent heuristic without using Manhattan/Euclidean distance.
- **Decision**: **Minimum Legal Push Distance Matching**. Use a reverse BFS from each goal to compute the minimum number of valid pushes required to pull a box from any cell to the goal, ignoring other boxes but respecting walls and player reachability. Use the Hungarian algorithm (bipartite matching) to optimally pair boxes to goals.
- **Reason**: Ensures admissibility because each push requires at least one player movement. Accounts for actual board geometry (walls).
- **Consequences**: Excellent lower bound. Automatically assigns `infinity` to static deadlocks (corners, unresolvable wall lines) because the reverse BFS cannot reach them.

## Decision 4: Admissibility and Consistency Proofs
- **Problem**: Verifying the heuristic properties as requested by Requirement 4.
- **Decision**: 
  - *Admissibility*: Total player moves >= Total pushes >= Matching Cost. `h(s) <= h*(s)`.
  - *Consistency*: A single player action can perform at most 1 box push. Thus, the minimum total pushes can decrease by at most 1. `h(s) - h(s') <= 1 = cost(s, a, s')`.
- **Reason**: A rigorous mathematical foundation guarantees A* optimality. We built an empirical verification script that performs exhaustive local checks to confirm this holds in practice.

## Decision 5: Deadlock Handling
- **Problem**: Pruning the search space from impossible states.
- **Decision**: Rely on the heuristic for static deadlocks (it returns `infinity`), and implement a dedicated 2x2 block detector for dynamic deadlocks during successor generation.
- **Reason**: The reverse push BFS natively handles corners and wall-lines perfectly. The 2x2 detector covers cases where boxes block each other, which the heuristic ignores.
- **Consequences**: Search space is drastically reduced without compromising admissibility.

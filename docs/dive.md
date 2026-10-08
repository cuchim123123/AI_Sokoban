

## 1. Project overview and presentation script

This Python project demonstrates two different kinds of artificial intelligence in the same Sokoban environment:

1. **Puzzle mode:** one agent finds a shortest sequence of walking and pushing actions using Uniform Cost Search (UCS) or A*. A* uses minimum legal push-distance matching and deadlock detection.
2. **Competitive mode:** two agents choose actions from the same board state and the engine resolves those actions together. A time-bounded, iterative-deepening maximin planner estimates the safest action against every legal opponent response. Points represent current credit for boxes on goals and can be lost or transferred.

**Suggested opening:** “My project compares optimal path search with adversarial decision-making. In puzzle mode, the objective is to solve Sokoban using the fewest actions. In competitive mode, the objective is to finish a limited number of simultaneous rounds with more credited goal boxes than the opponent. I separate the board model, transition rules, AI, interface, and experiments so that the planner simulates the same rules used by the running game.”

**Main contributions to explain:** geometry-aware heuristic design; state-space pruning; deterministic simultaneous conflict resolution; bounded adversarial search; reversible ownership scoring; a B-only repetition escape policy; and a Pygame interface that exposes decisions and supports replay.

The AI is rule-based search. It does not use neural networks, reinforcement learning, training data, or online learning. Heuristic weights are manually chosen guidance parameters, not learned probabilities.

## 2. Architecture and execution flow

```text
main.py
  -> shared/launcher.App
       -> single/gui/SinglePlayerApp
            -> parser -> GameState + Board
            -> UCS or A* -> action sequence -> replay history -> renderer
       -> competitive/gui/CompetitiveApp
            -> parser -> CompetitiveState + Board preprocessing
            -> AgentA / AgentB -> actions + ranked fallback lists
            -> transition.resolve_joint_action_outcome
            -> new state + executed actions + conflict metadata -> renderer

shared/common.py: widgets, sprites, interpolation, colors, previews
experiments/: measurements and heuristic checks
tests/: automated behavior and invariant checks
```

The game model contains grid coordinates, not screen pixels. Renderers convert grid positions to pixels and interpolate between previous and current states. Cosmetic randomness affects dust particles; it does not choose AI actions or change scores.

Both GUI applications assemble four concern-based mixins: setup, events, game lifecycle, and rendering. The application class owns shared fields and the main event loop. Background threads keep the interface responsive while search runs. Competitive decisions are computed sequentially in the worker, but both submissions use the same starting state: physical execution is simultaneous, not A moving before B.

## 3. Game model, legal actions, and scoring

### 3.1 Coordinates and maps

Coordinates are `(x, y)`: east increases x; south increases y. Each direction has a unit displacement. Walls and goals belong to the static board; player positions, boxes, credit, and round number belong to the changing state.

| Character | Puzzle parser | Competitive parser |
|---|---|---|
| `%` | Wall | Wall |
| `A` | Single agent | Agent A |
| `E` | No special entity | Agent B |
| `B` | Box | Shared box, initially uncredited |
| `D` | Goal | Goal |
| `C` | Box on goal | No special entity |
| `.` / space | Empty floor | Empty floor |

Unknown characters are effectively ignored as empty cells. Competitive parsing requires A, E, and at least one goal; single parsing requires an agent. Neither parser fully validates map shape, entity counts, or duplicate start markers. Missing cells in ragged competitive rows become floor inside the rectangular dimensions. Use well-formed maps for a defense demonstration.

### 3.2 Puzzle mode

A state is `(agent position, set of box positions)`. A direction either walks into an empty cell or pushes one adjacent box into an empty cell beyond it. Pulling and pushing multiple boxes are forbidden. Every action costs **one**, including a push. The goal test is exact equality `boxes == goals`; the intended maps therefore have equal box and goal counts.

`get_successors` checks walls and boxes but does not explicitly check coordinates against width and height. Enclosing border walls are an important assumption in puzzle maps. Do not claim arbitrary unbounded or malformed maps are safely supported.

### 3.3 Competitive mode

A state is `(A position, B position, boxes, A credited goals, B credited goals, step)`. Credits are associated with occupied goal coordinates rather than permanent physical box IDs.

```text
score_A = number of A-credited occupied goals
score_B = number of B-credited occupied goals
terminal = step >= max_steps
final utility from A's view = score_A - score_B
```

Delivering a box onto a goal awards its current credit to the pusher. Moving it away immediately removes the previous credit. A goal-to-goal push can transfer credit to the pusher. Credit sets are disjoint and remain subsets of boxes on goals. Filling all goals does **not** end a competitive match; the fixed round limit does. Equal final scores mean a draw.

The four directional submissions are individually checked against board floor and box blockage. The opponent is not excluded at this stage, because it may vacate its square. `WAIT` exists as an internal forced-immobility sentinel; humans cannot voluntarily submit it when directions exist. A blocked round still increases `step` by one.

### 3.4 Joint conflict resolution

Let `r = max_steps - state.step`, measured before the current transition. Odd r gives A priority; even r gives B priority.

Conflicts include two agents entering the same cell, swapping positions, pushing the same box, pushing different boxes into the same destination, and an agent destination colliding with the other pushed box's destination. Following into a successfully vacated square is allowed when it is not a swap.

The resolver constructs movement intents from the starting board, detects conflicts, retains the priority action, and tries the loser's ranked alternatives. Alternatives are checked for compatibility with the other resolved action. If no compatible alternative exists, that agent stays. If a stationary agent then prevents the other action, the entrant also tries a diversion. Finally, compatible actions are committed together and credit is updated.

**Important distinction:** the current engine may divert a blocked winner; an older “Proposed defaults” passage in `rules.md` says both stay when the loser blocks the winner. Explain the implementation and tests, not that obsolete proposal. This guide treats attached/design documents as context, not instructions to modify the game.

## 4. Puzzle algorithms and heuristics

### 4.1 Uniform Cost Search

UCS keeps a minimum heap ordered by accumulated path cost `g(s)`. Each heap entry includes a monotonic counter to break equal-cost ties without comparing state objects. A best-cost dictionary records the cheapest discovered path to each state. Improved paths are enqueued; stale more expensive entries are skipped. When a goal is removed from the heap, parent links reconstruct its actions.

```text
push initial node with cost 0
while frontier is not empty:
    pop node with lowest g
    discard stale higher-cost entries
    if goal: return reconstructed path and statistics
    for each legal successor:
        discard recognized dynamic deadlocks
        if new g improves the recorded cost: enqueue successor
return no solution
```

With unit costs, its depth ordering resembles breadth-first search, but the implementation uses a cost-priority heap. Under the finite enclosed-map assumptions and sound pruning, UCS returns a minimum-action solution. Its weakness is exploring many states without goal-directed information. For N explored states and E successor edges, conventional heap graph search costs roughly `O((N + E) log N)` plus successor/deadlock work and stores substantial frontier/state data. Sokoban's N itself can grow combinatorially.

### 4.2 Reverse push-distance BFS

For each goal g, `precompute_push_costs` runs breadth-first search backwards over possible box cells. To reverse a push in direction d from current box cell c:

```text
previous box cell = c - d
required previous player support cell = c - 2d
```

Both cells must be in bounds and not walls. A reverse edge costs one push. The resulting table `D(b, g)` is the minimum number of pushes in this relaxed wall-only box graph. Missing entries mean unreachable. This is stronger than Manhattan distance because walls and support geometry matter.

The relaxation ignores other boxes and whether the actual player can navigate between all support positions. It is therefore not an exact multi-box delivery solver. With F floor cells and G goals, preprocessing is approximately `O(GF)` time and storage on a four-neighbor grid.

### 4.3 Minimum push-distance matching

With boxes `b_1 ... b_n` and goals `g_1 ... g_n`, form the matrix `C[i,j] = D(b_i,g_j)`. The heuristic is:

```text
h(s) = minimum over one-to-one assignments pi of sum_i C[i, pi(i)]
```

`MatchingHeuristic` uses NumPy for the cost matrix and SciPy's `linear_sum_assignment` for optimal assignment. The assignment idea is often taught as Hungarian matching; the project calls a library solver and does not implement a handwritten Hungarian algorithm. Infinite costs represent impossible static deliveries; infeasible assignment raises `ValueError`, which the heuristic converts to infinity. With no boxes it returns zero.

**Why one-to-one assignment?** Summing each box's independently nearest goal can assign several boxes to the same goal. Matching enforces distinct target goals and provides a stronger lower bound. For example, a cost matrix `[[1, 4], [2, 8]]` has independent minima 1+2=3, but the cheapest valid assignment costs 4+2=6.

**Admissibility:** every real solution induces a box-to-goal assignment. Its pushes must cost at least the corresponding relaxed push distances. Minimizing over assignments cannot increase that lower bound. Since every real push is also an action and walking adds further actions, `h(s) <= true minimum remaining action cost`.

**Consistency:** a walk leaves box positions and h unchanged. For a legal push from b to b', the reverse graph contains its edge, so `D(b,g) <= 1 + D(b',g)`. Apply the optimal successor assignment to the predecessor: only the moved box can add at most one. Therefore `h(s) <= 1 + h(s')`. These arguments assume the intended equal-count puzzle formulation and valid map geometry. Empirical verification is supporting evidence, not a universal proof.

Matrix construction is `O(nG)` lookups. Assignment is a polynomial-time optimization; square assignment is commonly discussed with cubic-scale complexity. The practical benefit must be weighed against evaluating this matrix repeatedly.

### 4.4 A* search

A* prioritizes `f(s) = g(s) + h(s)`. It otherwise shares the UCS pattern: parent-linked nodes, best discovered g, improved-state reopening, stale-entry checks, and dynamic deadlock pruning. Children with infinite h are not put on the frontier. The goal is checked when popped, not merely when generated.

Under the preceding admissibility/finite-map assumptions, A* returns a minimum-action solution. It often expands fewer states than UCS, but this is an empirical expectation, not a guarantee of lower runtime on every map. Heuristic computation has a cost, and ties can still create large frontiers.

### 4.5 Deadlock detection

**Static information:** an off-goal box that cannot reach any goal in the reverse push graph has infinite delivery distance. More generally, matching can detect that no finite complete box-to-goal assignment exists. A* rejects these infinity-valued states. UCS does not use this static heuristic filter.

**Dynamic 2x2 filter:** `is_deadlock` examines four 2x2 squares around each off-goal box. If every cell of one square is a wall or a box, the involved off-goal box cannot be released by legal pushes. Both searches discard these successor states. An already completed all-goal block is not rejected merely for being filled.

These checks do not identify every Sokoban deadlock. They are useful limited patterns, not a complete solvability test. Puzzle pruning removes states proved useless under the model; competitive preferences deliberately retain physically legal sacrifice pushes.

### 4.6 Search outputs and measurement

Both search objects return `(actions, total_cost, generated_states, expanded_states)`. `actions=None` means failure; an empty list is a valid already-solved result. Failure currently reports cost zero, so zero cost alone is not evidence of success.

Generated starts at one and increments for successors after dynamic deadlock rejection but before duplicate rejection; it is not a count of unique states. Expanded counts popped, nonterminal, nonstale nodes. The benchmark's A* timer excludes push-table/matching construction, whereas the single GUI's solve timer includes it. State this difference when comparing measured runtimes.

## 5. Competitive algorithms and heuristics

### 5.1 Board preprocessing

The competitive Board builds floor cells and neighbors, then three geometry tables:

| Table | Algorithm | What it measures | What it ignores |
|---|---|---|---|
| `distances` | BFS from every floor cell | Shortest wall-respecting walk distance | Boxes and other agent |
| `push_costs` | Shared reverse push BFS | Relaxed minimum pushes from box cell to goal | Other boxes and full player connectivity |
| `exact_step_costs` | Reverse BFS on `(box, player)` pairs for each goal | Minimum walking + pushing actions in a single-box model | Other boxes and other agent |

The pair-state BFS starts with a box on the goal and an adjacent player. It reverses either a walk or a push; each costs one. “Exact” means exact for this **single-box relaxation**, not for the live competitive board. Its state count can be `O(F²)` per goal, making preprocessing approximately `O(GF²)` time and storage in the worst case. All-pairs walking BFS is approximately `O(F²)` on the grid. These costs are paid when parsing a board, outside individual AI search decisions; even menu previews currently construct this Board.

### 5.2 Simultaneous pure-action maximin

At each searched state the planner considers each own legal action a, and every opponent legal response b, then calls the actual joint transition T with both sides' primary-pinned fallback lists.

```text
V_0(s) = evaluation(s)
V_d(s) = max_a min_b V_(d-1)(T(s, a, b, preferences_A, preferences_B))
```

Terminal states use actual final score difference regardless of remaining search depth. For B, evaluation is negated and A becomes the minimizing opponent. A depth of d means d simultaneous rounds, not d alternating individual player turns.

Maximin chooses the action with the best worst-case response. This is a conservative pure-action security strategy. The ordering “max then min” expresses evaluation of possible joint outcomes; the live opponent does not see the selected action before choosing. The planner is not a mixed-strategy Nash equilibrium solver and does not predict the rival's personality or exact future controller behavior.

With up to four directions on each side, there can be up to 16 joint outcomes per searched round. Naive depth-d enumeration is exponential, approximately `O(16^d)` before pruning, transpositions, forced actions, and terminal cutoffs. This motivates bounded search rather than full game-tree solution.

### 5.3 Iterative deepening and deadline behavior

The default controller budget is one second per decision. Search completes depths 1, 2, 3, and so on until the round limit or a monotonic-clock deadline. It reserves up to 0.04 seconds, or ten percent of a smaller budget, for returning diagnostics and the submission.

**Fair comparison:** every root action must finish evaluation at the same depth before that iteration replaces the previous result. An interrupted depth is discarded as a decision set. A partially explored promising branch cannot replace a completed root comparison. If no depth completes, a legal preference-ordered direction is returned; WAIT is returned for terminal or fully immobile states.

This is a best-effort timing budget. Checks occur between operations; Python, operating-system scheduling, preprocessing, and diagnostic work prevent a strict hard-real-time guarantee. `_bounded_allocations` temporarily defers cyclic garbage collection across nested/concurrent calls and restores its previous state in `finally`; ordinary reference counting continues.

### 5.4 Alpha-beta bounds, transpositions, and ordering

`row_value` minimizes across replies and stops when a row cannot improve the maximizing bound. `value` maximizes across rows and stops when its alpha reaches beta. The transposition table stores `(state, depth)` entries with **exact**, **lower**, or **upper** flags; cutoff bounds must not be misread as exact evaluations.

Each root starts with unrestricted alpha/beta so its completed logged value is exact for that horizon and heuristic model. Previous principal choices are searched first; opponent replies are ordered by low immediate evaluation to expose threats early. Ordering changes efficiency without removing legal replies.

Search-local caches hold preferences, joint-outcome rows, values, and principal choices. Evaluation/transition memoization also uses board identity and round information. Step must be part of search state identity: the same physical board with different remaining time has different terminal distance and conflict priority.

### 5.5 Evaluation function, with every active weight

For A, nonterminal evaluation is:

```text
E_A(s) = (score_A - score_B) + (best opportunity_A - best opportunity_B)
                               + (pressure_A - pressure_B)
E_B(s) = -E_A(s)
```

**Current credit:** one credited box contributes one point. Credited boxes are excluded from unclaimed delivery opportunities; occupied credited goals are excluded from available targets. At the round limit, both guidance terms become zero and the value is exactly the score difference.

**Current walking BFS:** boxes block walking; the opponent does not act as a permanent wall. This finds real currently reachable approach cells for the first push.

**Delivery estimate:** for each uncredited box and available goal, enumerate legal first pushes. Cost is walking to its support cell + one push + the relaxed pair-table cost from the new box/player positions. The first push respects current boxes, but later movement ignores them. An uncredited box already on a goal must actually leave and be delivered again; standing beside it does not award a point.

For estimated cost c within r remaining rounds:

```text
base opportunity(c) = 0.65 / (1 + 0.12c)
otherwise = 0
```

When both agents can reach the same box opportunity within the limit, A's value is multiplied by `0.5 + 0.5 * clamp((c_B - c_A)/4, -1, 1)` and B's by the symmetric expression. Equal costs produce a half discount; a four-action advantage reaches the factor's extreme. This encodes a race preference, not a calibrated success probability.

Only the **largest single next-delivery opportunity** is retained for each side. Independent relaxed costs are not added as if the player could deliver multiple boxes at once. This avoids overstating incompatible schedules but also ignores longer-term multi-delivery coordination.

**Pressure:** among feasible pushes of opponent-credited boxes, let d be the currently reachable support walk distance plus one. If d fits in remaining rounds, the guidance candidate is `0.10/d`; use only the best candidate. It guides approaching a possible credit removal, not proof that the opponent allows it. Search handles actual execution and transfer.

Opportunity is bounded by 0.65 per side and pressure by 0.10, so their combined differential lies within approximately ±0.75. This keeps credit important, but does not prove that every comparison between distinct states strictly prefers a one-point lead: guidance values can differ across both states. The constants are engineering choices and need sensitivity experiments before claiming they are optimal.

### 5.6 Ranked fallback preferences

`round_preferences` sorts all physically legal directions using this lexicographic key:

1. Larger immediate net credit gain first: destination is a goal, minus removal of own credit, plus removal of opponent credit.
2. Prefer pushes not flagged by `creates_deadlock`.
3. Lower relaxed remaining delivery cost from the moved position.
4. Lower wall-only distance toward an opponent-credited box.

Ties keep deterministic north, south, east, west generation order. The submitted primary action is placed first. Legal sacrifice/deadlock pushes are retained, not banned by the transition rules. The same shared policy is used in search and live fallback execution; cached submissions avoid computing a different order after the deadline.

`has_legal_push` checks local support/destination geometry, not actual player reachability. `creates_deadlock` flags a prospective off-goal push with no locally possible future push or no reverse-push route to any goal. `deadlock_count` additionally checks static reachability and adjacent box/wall patterns, but it is a diagnostic helper and **is not added to the active evaluation or used to prune the competitive game tree**. The legacy `W_LOCKED` constant is not an active evaluation weight.

### 5.7 Equal-value repetition preference

After completed root values tie, the planner favors the better immediate worst-case value, then fewer replies returning to recent own-position/box/credit fingerprints. AgentA keeps six recent entries. This secondary criterion never overrides a strictly higher completed maximin value. It is distinct from B's active yielding mechanism.

### 5.8 B's two-loop breaker

AgentB inherits AgentA's engine and uses perspective B. It also observes full physical configurations: both positions, all boxes, and both credit sets, excluding step because looping still consumes time.

For candidate period p, it requires two consecutive complete cycles. Examples:

```text
period 1: X, X, X             -> two stalled transitions
period 2: X, Y, X, Y, X       -> two complete two-round cycles
```

Repeated queries in one round are not new observations. New board/round limit, rewind, skipped rounds, or changed same-round state resets evidence. It compares repeating sequences, not only whether B happens to revisit one square.

After detection, B prefers root alternatives that differ from the previous cycle submission, leave the cycle's B-position cells, and do not enter A's current cell. If none exists, it tries any different legal submission. It temporarily excludes cycle-preserving alternatives and asks the **same ordinary planner** to choose once from the remaining roots. The evaluator, transition engine, and deep search are unchanged. The restriction is for that decision; it may accept a strategically worse move or sacrifice to yield.

This is an intentional controller asymmetry. It is not a final-score tiebreaker: equal final scores still draw. It does not guarantee escape if no alternative exists or the other action diverts B back. Future hypothetical nodes use the ordinary planner, not a simulated stateful future loop-breaker history. “Active” telemetry means an exclusion was applied, not that escape was proven.

The renderer displays an amber **“B is breaking the loop”** pill below B for active rounds. Conflict priority is shown separately above the priority winner. These are presentation metadata and do not affect legality or scoring. A conflict label identifies parity priority, not proof that the winner's original move executed in every stationary-blockage case.

## 6. Object-oriented design and limitations

| Design feature | Evidence in this project | Defense explanation |
|---|---|---|
| Abstraction | `SearchAlgorithm` abstract base class | Both puzzle search implementations provide the same search contract. |
| Polymorphism | UCS/A* `.search`; A/B `.choose_action` | Callers use a shared behavioral interface. |
| Inheritance | `AgentB(AgentA)` | Reuses normal adversarial search, adding B's perspective and repetition policy. |
| Encapsulation | Board, state, node, planner, widgets | Groups related data and operations; private naming is a convention. |
| Separation of concerns | Domain/search/heuristics/UI/experiments | Core rules can be tested without the graphical loop. |
| Shared components | `shared/common.py` | Reuses widgets, assets, colors, and interpolation across modes. |
| Functional rule boundary | Joint transition returns a new state | Search and live play use the same deterministic model. |

The structure is reasonable for a university prototype, but it is not fully encapsulated or universally immutable. Frozen box/goal sets protect set contents; state attributes remain writable. CompetitiveState caches its hash, so mutating its attributes after using it as a dictionary key is unsafe. Mixins depend on many host fields instead of explicit interfaces; GUI code writes an agent's private `_last_action`; module-level telemetry and caches make concurrency reasoning harder.

Priority future improvements, **not implemented by this document**:

1. Capture immutable worker inputs and use a game-generation token in competitive mode. Its expected-step check can be insufficient after restart to the same step. Puzzle mode already has a solve-result token, but stronger immutable input capture would improve both.
2. Enforce immutable state objects and return a public decision/result object containing action, preferences, and telemetry.
3. Replace tightly coupled GUI mixins with composed menu, controller, and renderer objects if the application grows.
4. Put shared geometry preprocessing in a neutral domain utility rather than competitive Board importing a single-mode heuristic module.
5. Resolve maps/assets relative to the installation and add map validation/bounds checks.
6. Improve competitive experiments to instantiate real AgentB when testing the full game. Current comparison matches use AgentA with perspective B, so they measure the shared engine without B's yielding extension.

Do not claim the project guarantees no loops, solves every Sokoban deadlock efficiently, finds an optimal competitive policy within one second, or implements a mixed equilibrium. These are explicit boundaries of the current design.

## 7. Validation and defense demonstration

On 9 October 2026, the current automated suite was rerun using Python 3.11.9 and Pygame 2.6.1: **82 tests passed**. Tests cover parsing, puzzle actions/search, joint conflicts, scoring transfer, zero-sum evaluation, board-scoped caches, exact comparison against small exhaustive maximin trees, completed-depth deadline behavior, forced immobility, B's two-cycle trigger, and human input validation. This establishes tested behavior, not universal correctness or a measured tournament win rate.

Run from the project root with the same Python installation that has the dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 main.py gui maps/test_solvable.txt
py -3.11 main.py competitive maps/competitive/arena_open.txt 50
py -3.11 -B -m unittest discover -s tests
py -3.11 main.py verify maps/test_solvable.txt
py -3.11 -m src.experiments.competitive_benchmark --maps test_race --budget 1 --rounds 20
```

The final benchmark command writes decision diagnostics; matches require an external compatible baseline and `--matches`. No baseline snapshot is bundled. Old match claims should not be presented as measurements of this snapshot. Puzzle benchmark writes `results/benchmark_results.csv`; heuristic verification samples at most 200 reachable states and writes `results/verification_results.txt`. Competitive reporting defaults to `results/competitive_comparison.json`.

**Suggested six-minute demonstration:**

1. Explain both objectives and show the launcher (30 seconds).
2. Solve a small puzzle with UCS and A*, compare cost and expanded/generated counts (90 seconds).
3. Explain reverse-push matching with the small matrix in Section 4 (60 seconds).
4. Run competitive play, show current credits and a parity conflict, and explain joint action resolution (90 seconds).
5. Explain B's two-loop trigger and the amber notice; use a verified repeatable scenario if demonstrating it live rather than promising a loop on every map (30 seconds).
6. Show the passing tests and discuss one limitation and future improvement (60 seconds).

**Current controls:** launcher 1/2 selects modes; Enter defaults to puzzle. In setup use map/algorithm/controller buttons and Enter to start. Space pauses/resumes; comma rewinds; period advances through stored history; M/Escape returns to menu; R replays the puzzle solution or restarts competitive play. Human A uses WASD and human B uses arrows. Earlier README puzzle controls mentioning 1/2 algorithms or left/right replay are outdated relative to the event handlers.

## 8. Questions likely to be asked

**Why not use Manhattan distance?** It ignores walls and push-support geometry. Reverse push BFS provides a stronger geometry-aware lower bound while still relaxing other boxes and walking.

**Why matching?** Each goal must receive a distinct box. Minimum one-to-one assignment avoids counting the same nearest goal for multiple boxes.

**What does optimal mean?** Puzzle optimality minimizes all walking and pushing actions, not pushes alone. Competitive search is exact only for a completed limited-depth pure-action maximin tree under its heuristic and fallback model; it is not a solved whole-game policy.

**Why simultaneous maximin rather than ordinary alternating minimax?** Both agents choose from the same starting state. Applying A before generating B's physical action would simulate different game rules. Max/min here aggregates a matrix of jointly executed action pairs.

**Why include the step in the state key?** Remaining rounds change priority, opportunity feasibility, and terminal conditions even when physical positions repeat.

**Does the AI recognize a completed box?** Yes. Credit contributes actual score; credited boxes and goals are excluded from unclaimed delivery estimates. They remain movable, so protecting or stealing them is a strategic consideration rather than an absolute prohibition.

**Does it move randomly?** The planner and fallback ordering are deterministic for fixed evaluated results and ordering. Timing can change completed depth and therefore the chosen action. Cosmetic particles use randomness. Limited horizons can still produce unproductive movement; equal-value history preferences and B's yielding policy reduce some repetition.

**Why only B yields?** It deliberately gives one controller responsibility for escaping repeated configurations. It is an asymmetric design choice, so report it when discussing fairness; it does not change conflict parity rules.

**Why not prune competitive deadlock pushes?** The game rewards current credit and competition rather than requiring every box to be solved. A destructive push can be a legal sacrifice or deny an opponent's point. Ordering can discourage it without changing the game's action set.

**Is the competitive evaluator admissible?** That is not the relevant A* guarantee here. It is a bounded strategic estimate in score units. Terminal values are exact; nonterminal values are not proven outcome bounds.

**What is the biggest performance cost?** Exponential joint search and repeated evaluation during play; pair-state geometry preprocessing can also be expensive for larger boards. Caching, ordering, alpha-beta bounds, and deadline control address different parts of this cost.

**How did you validate search pruning?** Small exhaustive joint-tree reference calculations are compared with the pruned planner across perspectives/parities and selected horizons. Deadline tests ensure unfinished iterations never replace a completed comparison.

**What would you refactor first?** Competitive worker lifetime handling and enforced immutable state identity, because they protect correctness. Larger OOP/UI composition changes are secondary maintainability work.

## 9. Complete Python file and symbol reference

The catalogue below covers every current Python file, class, method, named function, and nested named function. Line numbers refer to this reviewed snapshot and will move after later edits. Imported symbols are described in their defining modules. Anonymous lambdas are callback glue: menu buttons capture map choices or select algorithms; sort keys order search rows; the optional baseline adapter supplies a missing hook. They are not additional AI algorithms.

### `main.py`

Command-line entry and application/experiment dispatch. Its __main__ block handles help, dependencies, map/round/controller arguments, and the selected command.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `print_help` | 7 | Print command syntax, examples, and available launch/experiment commands. |
| `check_dependencies` | 25 | Import pygame, SciPy, and NumPy; explain installation and exit on a missing dependency. |

### `src/competitive/__init__.py`

Package initialization and namespace boundary. Re-exports: `Action`, `Board`, `CompetitiveState`, `parse_competitive_map`, `AgentA`, `AgentB`, `best_action`, `resolve_joint_action_outcome`, `conflict_winner`.

No locally defined classes or named functions.

### `src/competitive/agent_a.py`

Shared simultaneous search engine and normal controller. TIME_LIMIT is 1.0; LAST_SEARCH publishes diagnostics; temporary GC bookkeeping controls decision allocation pauses.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `_bounded_allocations` | 26 | Decorator deferring cyclic GC during decisions, with shared nesting/concurrency bookkeeping. |
| `_bounded_allocations.call` | 34 | Nested wrapper acquiring GC bookkeeping locks, invoking the wrapped function, and restoring GC in finally. |
| `_Timeout` | 51 | Internal exception used to abandon an unfinished iterative-deepening iteration. |
| `_key` | 55 | Return a tuple including both positions, boxes, credits, and step for precise competitive identity. |
| `history_entry` | 60 | Return own position, boxes, and credits without step for equal-value recent-position preference. |
| `_Planner` | 65 | One decision's maximin engine, deadline, caches, root comparisons, and telemetry. |
| `_Planner.__init__` | 66 | Capture board/state/limit/perspective/deadline, recent history and root bans; initialize per-decision caches and counters. |
| `_Planner.check` | 83 | Raise _Timeout when the monotonic deadline is reached. |
| `_Planner.score` | 87 | Evaluate a state with deadline checks before and after computation. |
| `_Planner.preferences` | 93 | Cache starting-state fallback rankings per side, optionally pinning that side's primary action first. |
| `_Planner.actions` | 103 | Return ranked legal directions; return the WAIT sentinel only when none exists. |
| `_Planner.children` | 106 | Resolve every opponent reply jointly with both pinned preference lists; cache and order outcomes by immediate score. |
| `_Planner.row_value` | 128 | Compute the minimum continuation value across replies, stopping at an alpha cutoff. |
| `_Planner.value` | 138 | Compute recursive maximizing values with terminal/depth evaluation, transposition bounds, and alpha-beta pruning. |
| `_Planner.root_iteration` | 174 | Evaluate every root at one common depth with independent full alpha/beta windows; return only if all finish. |
| `_Planner.tie_key` | 185 | Prefer greater immediate worst-case score, then fewer recent-configuration replies when backed-up values tie. |
| `_Planner.run` | 192 | Filter roots by permitted bans, keep a legal fallback, iterate completed depths, discard timeout's partial results. |
| `_Planner.diagnostics` | 217 | Describe cached root/reply values, conflicts, execution, evaluation components, and preferences without new search. |
| `_Planner._logged_preferences` | 233 | Produce primary-pinned names from cached rankings for diagnostics. |
| `best_action` | 241 | Public search entry: derive deadline/reserve, run planner, publish LAST_SEARCH, return a direction; retains legacy arguments. |
| `AgentA` | 264 | Stateful controller wrapping the shared planner, recent history, evaluation cache, and cached live submission. |
| `AgentA.__init__` | 267 | Set time budget and initialize short history, compatibility TT field, cache, latest action/telemetry, and submission. |
| `AgentA.choose_action` | 277 | Rank preferences within budget, run maximin, cache the primary-pinned submission, and record history/telemetry. |
| `AgentA.preference_list` | 294 | Return matching cached submission order or compute the shared starting-state ranking. |

### `src/competitive/agent_b.py`

B perspective plus stateful two-cycle yielding; no separate base search algorithm.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `AgentB` | 13 | AgentA specialization with B perspective and an active yielding policy after two repeated cycles. |
| `AgentB.__init__` | 18 | Initialize the base controller and full configuration/action observation history. |
| `AgentB._observe_cycle` | 25 | Record consecutive configurations, reset discontinuities, and detect two complete repetitions of the shortest matching period. |
| `AgentB.choose_action` | 54 | Detect loops, derive one-round alternative-root restrictions, call normal base search once, and publish loop metadata. |
| `AgentB.choose_action.destination` | 64 | Nested helper returning B's intended destination for a candidate direction. |

### `src/competitive/evaluation.py`

Score-unit leaf guidance and diagnostic push/deadlock helpers. LRU caches and raw value memoization reduce repeated work.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `clear_cache` | 22 | Clear raw evaluation and both LRU walking/delivery caches. |
| `walking_distances` | 29 | LRU-cached BFS from the player, blocking current boxes but allowing the moving opponent's cell. |
| `delivery_cost` | 43 | Enumerate reachable legal first pushes, then add the relaxed single-box delivery tail; never award idle delivery credit. |
| `_delivery_min` | 61 | LRU-cached minimum delivery estimate for one box over the supplied available goals. |
| `evaluation_components` | 66 | Calculate A-perspective actual credit, best delivery differential, and best removal-pressure differential. |
| `evaluation_components.pressure` | 91 | Nested helper estimating the best reachable push pressure against opponent-credited boxes within remaining rounds. |
| `evaluate` | 106 | Cache the sum of components by board serial/state/limit; negate the raw A value for B. |
| `has_legal_push` | 118 | Check whether any local support/destination pair permits a box push, ignoring full player reachability. |
| `deadlock_count` | 135 | Diagnostic count using local immobility, free-goal reverse reachability, and adjacent wall-pair patterns; not active leaf evaluation. |
| `creates_deadlock` | 185 | Flag prospective off-goal pushes with no locally possible follow-up or no static goal route; used for preference ordering. |

### `src/competitive/gui/__init__.py`

Package initialization and namespace boundary. Contains package documentation; defines no local classes or functions.

No locally defined classes or named functions.

### `src/competitive/gui/app.py`

Competitive GUI construction and 60 FPS main-loop scheduling. Uses a daemon worker for round decisions.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `CompetitiveApp` | 46 | Competitive GUI host combining setup, round computation, events, and rendering mixins. |
| `CompetitiveApp.__init__` | 47 | Initialize controller settings, display/assets/maps, human input, animation, history metadata, and worker fields. |
| `CompetitiveApp.run` | 119 | Run events/animation/drawing and dispatch background round computation when running and ready. |

### `src/competitive/gui/events.py`

Competitive menu and play input, human direction validation, pause/restart/history controls.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `EventsMixin` | 18 | Pygame input routing for its containing mode; updates application-owned fields and invokes menu/game callbacks. |
| `EventsMixin._handle_events` | 21 | Route menu mouse/keyboard input and play controls; competitive version also validates WASD/arrows and forced immobility. |

### `src/competitive/gui/game.py`

Competitive worker, joint result publication/application, current metrics/history, and presentation metadata.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `GameMixin` | 21 | Game lifecycle/worker result/replay operations for its containing mode; expects application-owned session fields. |
| `GameMixin._compute_step` | 24 | Obtain both AI/human submissions from the current state, obtain fallback lists, resolve one joint round, and publish result/metadata. |
| `GameMixin._apply_computed_step` | 87 | Reject mismatched step results, apply execution/state, update metrics/history/conflict/loop metadata and cosmetics, mark terminal play. |

### `src/competitive/gui/render.py`

Competitive visual composition including animated board, score HUD, conflict priority, B yielding, and final result.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `RenderMixin` | 33 | Mode-specific board, actors, effects, metrics, menus, and overlay rendering. |
| `RenderMixin._draw_particles` | 36 | Render and advance dust particle position/lifetime, removing expired particles. |
| `RenderMixin._draw_floating_texts` | 47 | Render and fade upward-moving goal-award effects, removing expired texts. |
| `RenderMixin._draw` | 57 | Select menu or board rendering, compose effects/HUD/overlays, and update the display. |
| `RenderMixin._draw_menu` | 87 | Fill cyan and render map preview, choices/settings, contrasting panels, and menu buttons. |
| `RenderMixin._get_box_visual_positions` | 127 | Match previous/current competitive box cells to interpolate moved boxes during simultaneous rounds. |
| `RenderMixin._draw_board` | 158 | Render floors/walls/goals, interpolated boxes and actors; competitive version also draws conflict and amber loop-yield indicators. |
| `RenderMixin._draw_agent_anim` | 249 | Select and tint A/B animation frames, interpolate movement, draw shadows and labels, use placeholders if needed. |
| `RenderMixin._draw_ui` | 300 | Render mode metrics/help: puzzle cost/search counts/replay/goals, or competitive scores/time/submissions/conflict priority. |
| `RenderMixin._draw_result_overlay` | 345 | Show puzzle solved/failure metrics or competitive winner/draw and final scores, with return-to-menu guidance. |

### `src/competitive/gui/setup.py`

Competitive menu buttons, map/controller/limit settings, navigation, and new-game initialization.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `SetupMixin` | 23 | Menu construction and choice callbacks for its containing game mode; expects fields on the host application. |
| `SetupMixin._setup_menu_buttons` | 26 | Build map choices, mode-specific algorithm/controller settings, and start/back buttons. |
| `SetupMixin._back` | 53 | Return from competitive setup to launcher when embedded; otherwise quit the standalone game. |
| `SetupMixin._to_menu` | 61 | Return competitive play to its setup screen and resize the display. |
| `SetupMixin._select_map` | 66 | Store the chosen map; puzzle mode also refreshes selected controls. |
| `SetupMixin._toggle_ai_a` | 69 | Toggle A between human and AI and update the corresponding button text. |
| `SetupMixin._toggle_ai_b` | 73 | Toggle B between human and AI and update the corresponding button text. |
| `SetupMixin._inc_steps` | 77 | Increase the competitive round limit by five. |
| `SetupMixin._dec_steps` | 80 | Decrease the competitive round limit by five, never below five. |
| `SetupMixin._start_game` | 83 | Create configured A/B controllers, parse the competitive map, reset session/history/effects, and resize the board display. |

### `src/competitive/parser.py`

Competitive text-map input and initial uncredited state construction.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `parse_competitive_map` | 19 | Read A/E starts, walls, boxes, and goals; validate required starts/goals and construct initially empty credit sets. |

### `src/competitive/preferences.py`

Shared deterministic fallback ordering for both live play and searched transitions.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `round_preferences` | 9 | Deterministically rank all legal directions by credit gain, deadlock flag, delivery cost, and attacking distance; pin primary. |
| `round_preferences.rank` | 20 | Nested sort-key helper calculating immediate credit swing and relaxed objectives for a candidate move. |

### `src/competitive/state.py`

Competitive domain types and static distance preprocessing; state identity includes round and credit.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `Action` | 5 | Enum of movement vectors; competitive mode additionally defines WAIT for forced immobility. |
| `Action.__str__` | 12 | Return a human-readable capitalized direction name. |
| `Board` | 16 | Static geometry shared by changing states; competitive Board also owns distance/preprocessing tables and a unique serial. |
| `Board.__init__` | 29 | Store frozen wall/goal sets and dimensions; competitive constructor additionally builds floor/neighbors and all distance tables. |
| `Board._precompute_exact_steps` | 78 | Reverse BFS over box/player pairs per goal, reversing walks and pushes in the single-box model. |
| `Board.exact_steps` | 122 | Look up relaxed minimum walk-plus-push delivery cost, returning 9999 when absent. |
| `Board.dist` | 126 | Look up wall-only shortest walking distance between two cells, returning 9999 when absent. |
| `Board.push_dist` | 130 | Look up relaxed minimum pushes from a box cell to a goal, returning 9999 when absent. |
| `CompetitiveState` | 134 | Full round configuration including both players, shared boxes, both credit sets, and step. |
| `CompetitiveState.__init__` | 152 | Normalize sets and cache the hash of all identity fields. |
| `CompetitiveState.__eq__` | 180 | Compare positions, boxes, credits, and step; different rounds are different search states. |
| `CompetitiveState.__hash__` | 192 | Return the cached identity hash; assumes attributes are never mutated after construction. |
| `CompetitiveState.score_a` | 199 | Return the size of A's credited goal set. |
| `CompetitiveState.score_b` | 202 | Return the size of B's credited goal set. |
| `CompetitiveState.is_terminal` | 205 | Check whether the fixed round limit has been reached. |
| `CompetitiveState.__repr__` | 208 | Build a compact debug representation with positions, scores, step, and box count. |

### `src/competitive/transition.py`

Authoritative deterministic simultaneous rule engine and transition cache. No Pygame dependency.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `Outcome` | 15 | NamedTuple with next state, conflict flag, and both executed actions; _Outcome is an alias, not another implementation. |
| `_Intent` | 34 | Internal NamedTuple recording the agent destination and optional pushed-box destination. |
| `_step` | 42 | Add an Action displacement to a position. |
| `_other` | 47 | Return the opposite agent label. |
| `conflict_winner` | 51 | Return A for odd remaining rounds and B for even remaining rounds. |
| `_intent` | 56 | Construct a physically legal directional intent; return None for blocked directions or WAIT. |
| `_enters` | 71 | Check whether the agent or its pushed box ends on a particular cell. |
| `_conflict` | 78 | Recognize incompatible agent/box destinations and position swaps. |
| `_commit` | 96 | Apply compatible intents, remove old goal credit, award successful new deliveries, and construct step+1 state. |
| `_pick_diversion` | 156 | Try the side's fixed ranked legal alternatives against the other final action; return None if none fits. |
| `_diversion_order` | 196 | Order diversion handling by current parity priority. |
| `_resolve` | 206 | Perform intent conflict detection, fallback selection, stationary-blockage handling, and joint commitment. |
| `resolve_joint_action_outcome` | 284 | Memoize deterministic resolution including board, time, state, submissions, and both fallback lists in the key. |
| `clear_cache` | 326 | Clear memoized joint-transition outcomes. |
| `get_valid_actions` | 333 | List physically possible directions using floor bounds and box clearance; opponent occupancy is resolved jointly. |

### `src/experiments/__init__.py`

Package initialization and namespace boundary. Re-exports: `run_benchmarks`, `verify_properties`.

No locally defined classes or named functions.

### `src/experiments/benchmark.py`

UCS/A* measurements and CSV export; __main__ runs the small solvable map.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `run_benchmarks` | 10 | Run UCS and A* on each map; check equal reported cost and write CSV timing/count/cost results. |

### `src/experiments/competitive_benchmark.py`

Decision diagnostics and optional external-baseline matches. __main__ defines argparse options; report uses AgentA for either perspective rather than AgentB.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `load_baseline` | 21 | Temporarily import a supplied historical engine snapshot, repair its missing preference hook in memory, then restore module bindings. |
| `run` | 43 | Competitive experiment entry: measure decisions and optional seat-swapped baseline matches, assert invariants, write JSON diagnostics. |

### `src/experiments/verify_heuristic.py`

Sampled admissibility/consistency experiment using UCS as a reference; __main__ chooses the supplied/default map.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `verify_properties` | 11 | BFS-sample up to 200 states, compare h against UCS solution costs and one-step consistency, and write results. |

### `src/shared/__init__.py`

Package initialization and namespace boundary. Contains package documentation; defines no local classes or functions.

No locally defined classes or named functions.

### `src/shared/common.py`

Shared constants (64-pixel tiles, 120-pixel HUD), cyan menu styles, widgets, sprite animation, interpolation, and map previews.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `load_placeholder_image` | 35 | Load/scale/optionally tint a texture; return a colored labeled fallback surface on failure. |
| `lerp` | 55 | Calculate scalar linear interpolation a+(b-a)t. |
| `lerp_pos` | 59 | Apply lerp independently to the x and y coordinates. |
| `make_fonts` | 63 | Construct the shared font sizes, using Segoe UI or Arial fallback. |
| `draw_glass_panel` | 75 | Draw a translucent rounded panel and border on a temporary alpha surface. |
| `draw_glow` | 82 | Draw an enlarged translucent rounded glow behind a widget. |
| `draw_menu_panel` | 89 | Draw an opaque dark panel with border, contrasting with the solid cyan menu background. |
| `Button` | 95 | Reusable rectangle/text/callback widget with selected and hover states. |
| `Button.__init__` | 96 | Create button bounds, labels, callback, style fields, and font. |
| `Button.draw` | 106 | Render selected/hover/default fill, border, and centered text. |
| `Button.handle_event` | 119 | Update hover on mouse movement and invoke the callback for a left click while hovered. |
| `Animator` | 127 | Shared cardinal-direction Businessman sprite loader and animation-frame selector. |
| `Animator.__init__` | 130 | Read PNG/JSON frame pairs, union-crop visible bounds, scale to tile height, and index animations by state/direction. |
| `Animator.get_frame` | 189 | Select a movement or time-driven frame; fall back to south/missing-frame status and clamp finish animations. |
| `_preview_data` | 218 | Parse the selected mode's map and return preview walls/goals/boxes/colored agents; competitive parsing also preprocesses Board. |
| `draw_map_preview` | 237 | Draw a scaled miniature board and filename; show an unavailable message when parsing fails. |

### `src/shared/launcher.py`

Unified mode selection and dispatch. Keeps Pygame alive while child modes return to the launcher.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `ModeCard` | 29 | Large launcher choice widget combining mode title, explanatory lines, accent, and callback. |
| `ModeCard.__init__` | 32 | Store rectangle, mode/title/descriptions, accent, and selection callback. |
| `ModeCard.draw` | 41 | Render the mode card and its text with hover styling. |
| `ModeCard.handle_event` | 75 | Track hover and trigger mode selection for a card click. |
| `App` | 83 | Unified launcher owning the mode menu and lazily created game applications. |
| `App.__init__` | 86 | Initialize Pygame display/fonts and construct both mode cards; store a requested puzzle-map preselection. |
| `App._select` | 124 | Record the requested mode so the launcher menu loop can exit. |
| `App.run` | 127 | Alternate mode selection and game dispatch until exit; quit Pygame on completion. |
| `App._run_menu` | 140 | Run menu input/drawing at 60 FPS, supporting mouse and 1/2/Enter/escape shortcuts. |
| `App._draw_menu` | 168 | Fill solid cyan and draw title, mode cards, divider, and help text. |
| `App._run_single` | 198 | Create/reuse SinglePlayerApp and run puzzle mode until it returns to the launcher. |
| `App._run_competitive` | 205 | Create/reuse CompetitiveApp with launcher navigation enabled and run competitive mode. |

### `src/single/__init__.py`

Package initialization and namespace boundary. Contains package documentation; defines no local classes or functions.

No locally defined classes or named functions.

### `src/single/core/__init__.py`

Package initialization and namespace boundary. Re-exports: `Action`, `Board`, `GameState`, `parse_map`, `get_successors`.

No locally defined classes or named functions.

### `src/single/core/actions.py`

Puzzle legal successor generation; depends on enclosing walls rather than explicit dimension bounds.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `get_successors` | 4 | Generate the four legal walk/push successor pairs for a puzzle state, checking walls and boxes. |

### `src/single/core/parser.py`

Puzzle text-map input, including C for a box initially on a goal.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `parse_map` | 4 | Read puzzle symbols into GameState and Board; require an agent, track map dimensions. |

### `src/single/core/state.py`

Puzzle action/board/state domain types; no rendering or solver dependencies.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `Action` | 4 | Enum of movement vectors; competitive mode additionally defines WAIT for forced immobility. |
| `Action.__str__` | 10 | Return a human-readable capitalized direction name. |
| `Board` | 13 | Static geometry shared by changing states; competitive Board also owns distance/preprocessing tables and a unique serial. |
| `Board.__init__` | 15 | Store frozen wall/goal sets and dimensions; competitive constructor additionally builds floor/neighbors and all distance tables. |
| `GameState` | 21 | Puzzle configuration holding one agent and a frozen set of boxes. |
| `GameState.__init__` | 23 | Store agent coordinates and normalize box positions to a frozenset. |
| `GameState.__eq__` | 27 | Compare agent and boxes; board geometry is supplied separately by the search. |
| `GameState.__hash__` | 32 | Hash the agent/boxes tuple for state-set and best-cost lookups. |
| `GameState.is_goal` | 35 | Check exact equality between current box positions and board goals. |

### `src/single/gui/__init__.py`

Package initialization and namespace boundary. Contains package documentation; defines no local classes or functions.

No locally defined classes or named functions.

### `src/single/gui/events.py`

Puzzle setup input and solution replay/pause/scrub controls.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `EventsMixin` | 14 | Pygame input routing for its containing mode; updates application-owned fields and invokes menu/game callbacks. |
| `EventsMixin._handle_events` | 17 | Route menu mouse/keyboard input and play controls; competitive version also validates WASD/arrows and forced immobility. |

### `src/single/gui/game.py`

Puzzle solve worker, result handling, replay state construction, lifecycle, and cosmetic effects.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `GameMixin` | 25 | Game lifecycle/worker result/replay operations for its containing mode; expects application-owned session fields. |
| `GameMixin._start_game` | 30 | Parse puzzle state, reset replay/metrics/effects, resize display, and start a solve worker tagged with a new token. |
| `GameMixin._compute_solution` | 66 | Construct the selected solver/heuristic, solve in the worker, and publish results only for the current solve token. |
| `GameMixin._apply_solution` | 80 | Store metrics, handle failure/already-solved state, and reconstruct every state of the found action sequence for replay. |
| `GameMixin._advance` | 109 | Move forward in puzzle replay, start animation/effects, and mark completion when boxes match goals. |
| `GameMixin._rewind` | 122 | Move one puzzle replay step backwards and hide the final overlay. |
| `GameMixin._spawn_step_effects` | 133 | Create cosmetic dust on moved boxes and +1/shake effects when occupied-goal count increases. |
| `GameMixin._to_menu` | 159 | Return puzzle play to setup, pause replay, and restore the menu display dimensions. |

### `src/single/gui/render.py`

Puzzle menu/board/actor rendering, search statistics, solve waiting view, and completion result.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `RenderMixin` | 26 | Mode-specific board, actors, effects, metrics, menus, and overlay rendering. |
| `RenderMixin._draw` | 31 | Select menu or board rendering, compose effects/HUD/overlays, and update the display. |
| `RenderMixin._draw_menu` | 60 | Fill cyan and render map preview, choices/settings, contrasting panels, and menu buttons. |
| `RenderMixin._board_offsets` | 96 | Compute pixel offsets centering the puzzle board above its HUD. |
| `RenderMixin._draw_board` | 100 | Render floors/walls/goals, interpolated boxes and actors; competitive version also draws conflict and amber loop-yield indicators. |
| `RenderMixin._box_visual_positions` | 124 | Match previous/current puzzle box cells to interpolate a pushed box's motion. |
| `RenderMixin._draw_agent` | 150 | Select puzzle running/pushing/idle/finish frames and draw the interpolated player with texture fallback. |
| `RenderMixin._draw_particles` | 184 | Render and advance dust particle position/lifetime, removing expired particles. |
| `RenderMixin._draw_floating_texts` | 195 | Render and fade upward-moving goal-award effects, removing expired texts. |
| `RenderMixin._draw_ui` | 205 | Render mode metrics/help: puzzle cost/search counts/replay/goals, or competitive scores/time/submissions/conflict priority. |
| `RenderMixin._draw_solving_overlay` | 248 | Show an animated puzzle-search waiting panel while the background solver runs. |
| `RenderMixin._draw_result_overlay` | 263 | Show puzzle solved/failure metrics or competitive winner/draw and final scores, with return-to-menu guidance. |

### `src/single/gui/setup.py`

Puzzle map/algorithm menu and launcher navigation.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `SetupMixin` | 17 | Menu construction and choice callbacks for its containing game mode; expects fields on the host application. |
| `SetupMixin._setup_menu_buttons` | 22 | Build map choices, mode-specific algorithm/controller settings, and start/back buttons. |
| `SetupMixin._refresh_menu_selection` | 46 | Synchronize selected map/algorithm appearance with current puzzle choices. |
| `SetupMixin._select_map` | 53 | Store the chosen map; puzzle mode also refreshes selected controls. |
| `SetupMixin._set_algorithm` | 57 | Store UCS or A* and refresh puzzle menu selection. |
| `SetupMixin._back_to_launcher` | 61 | Set the puzzle application's exit flag so its parent launcher resumes. |

### `src/single/gui/single.py`

Puzzle GUI construction, map normalization, solve-result token fields, and main loop.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `SinglePlayerApp` | 40 | Puzzle GUI host combining setup, lifecycle, events, and rendering mixins. |
| `SinglePlayerApp.__init__` | 41 | Initialize display, map/algorithm choices, assets, solve-token state, replay fields, and menu widgets. |
| `SinglePlayerApp._resolve_map` | 98 | Normalize a requested path and match it to the discovered puzzle-map list. |
| `SinglePlayerApp.run` | 110 | Handle input, await a worker result, animate/replay the solution, and draw until returning to launcher. |

### `src/single/heuristics/__init__.py`

Package initialization and namespace boundary. Re-exports: `precompute_push_costs`, `MatchingHeuristic`, `is_deadlock`.

No locally defined classes or named functions.

### `src/single/heuristics/deadlock.py`

Dynamic 2x2 wall/box deadlock pattern used by both puzzle solvers.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `is_deadlock` | 3 | Detect a full wall/box 2x2 square containing an off-goal box; used to prune puzzle successors. |

### `src/single/heuristics/matching.py`

NumPy cost-matrix construction and SciPy minimum assignment heuristic.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `MatchingHeuristic` | 5 | Callable object calculating a minimum one-to-one box/goal push-distance assignment. |
| `MatchingHeuristic.__init__` | 6 | Store board, precomputed distances, and the ordered goal list. |
| `MatchingHeuristic.__call__` | 11 | Build the cost matrix, call SciPy linear_sum_assignment, sum selected costs; return infinity if infeasible. |

### `src/single/heuristics/push_distance.py`

Wall/support-aware reverse push graph preprocessing, also reused by competitive Board.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `precompute_push_costs` | 5 | Build each goal's reverse-BFS push distances with wall and player-support geometry. |

### `src/single/search/__init__.py`

Package initialization and namespace boundary. Re-exports: `SearchAlgorithm`, `Node`, `reconstruct_path`, `AStarSearch`, `astar_search`, `UniformCostSearch`, `uniform_cost_search`.

No locally defined classes or named functions.

### `src/single/search/algorithm.py`

Abstract puzzle search interface defining the common result contract.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `SearchAlgorithm` | 5 | Abstract interface for puzzle solvers with a common result tuple. |
| `SearchAlgorithm.search` | 7 | Declare the required search contract: actions, cost, generated, expanded; subclasses implement it. |

### `src/single/search/astar.py`

Heuristic graph search ordered by g+h, with reopening and static/dynamic deadlock rejection.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `AStarSearch` | 10 | Concrete SearchAlgorithm using a supplied heuristic and heap priority g+h. |
| `AStarSearch.__init__` | 11 | Store the heuristic callable so search is independent of its implementation. |
| `AStarSearch.search` | 14 | Run A* with improved-cost reopening, deadlock/infinity rejection, and the standard result tuple. |
| `astar_search` | 66 | Convenience wrapper constructing AStarSearch with the given heuristic and calling search. |

### `src/single/search/node.py`

Parent-linked search nodes and final action-path reconstruction.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `Node` | 4 | Parent-linked search record containing state, entering action, and accumulated cost. |
| `Node.__init__` | 5 | Store state, parent, action, and path cost for reconstruction. |
| `Node.__lt__` | 11 | Compare nodes by path cost; current heaps also use a numeric tie counter. |
| `reconstruct_path` | 14 | Follow parents backwards, reverse the collected actions, and return them with the final cost. |

### `src/single/search/ucs.py`

Uniform-cost graph search using best g values, a priority heap, and dynamic deadlock pruning.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `UniformCostSearch` | 10 | Concrete SearchAlgorithm that orders its heap by g, the accumulated action cost. |
| `UniformCostSearch.search` | 11 | Run UCS with best-cost duplicate handling, dynamic deadlock pruning, goal detection, and statistics. |
| `uniform_cost_search` | 57 | Compatibility convenience wrapper constructing UniformCostSearch and calling search. |

### `tests/test_agent_b_loop_breaker.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `TestAgentBLoopBreaker` | 12 | unittest.TestCase grouping agentbloopbreaker regression checks; methods below contain assertions. |
| `TestAgentBLoopBreaker.setUp` | 13 | Prepare fresh test fixtures, application stubs, or patches before each test. |
| `TestAgentBLoopBreaker.setUp.search` | 18 | Nested mocked best_action helper recording root bans and choosing a legal remaining action for loop-policy isolation. |
| `TestAgentBLoopBreaker.state` | 30 | Construct a test configuration with requested positions/boxes/credits/round. |
| `TestAgentBLoopBreaker.test_two_complete_loops_then_only_b_excludes_loop_moves` | 35 | Assert that two complete loops then only b excludes loop moves. |
| `TestAgentBLoopBreaker.test_longer_cycles_are_detected` | 49 | Assert that longer cycles are detected. |
| `TestAgentBLoopBreaker.test_repeated_queries_in_one_round_do_not_count_as_loops` | 56 | Assert that repeated queries in one round do not count as loops. |
| `TestAgentBLoopBreaker.test_box_or_credit_progress_does_not_trigger` | 61 | Assert that box or credit progress does not trigger. |
| `TestAgentBLoopBreaker.test_changed_box_position_is_not_a_repeated_configuration` | 68 | Assert that changed box position is not a repeated configuration. |
| `TestAgentBLoopBreaker.test_normal_search_resumes_after_leaving_cycle` | 74 | Assert that normal search resumes after leaving cycle. |
| `TestAgentBLoopBreaker.test_rewind_new_board_or_skipped_round_resets_evidence` | 81 | Assert that rewind new board or skipped round resets evidence. |
| `TestAgentBLoopBreaker.test_no_legal_exit_preserves_legal_move` | 91 | Assert that no legal exit preserves legal move. |
| `TestAgentBLoopBreaker.test_external_bans_are_preserved` | 98 | Assert that external bans are preserved. |
| `TestAgentBLoopBreaker.test_two_conflict_stalls_try_another_submission` | 104 | Assert that two conflict stalls try another submission. |
| `TestAgentBLoopBreaker.test_real_search_caches_yielding_primary_for_live_execution` | 111 | Assert that real search caches yielding primary for live execution. |

### `tests/test_astar.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `TestAStar` | 8 | unittest.TestCase grouping astar regression checks; methods below contain assertions. |
| `TestAStar.test_astar_solvable` | 9 | Assert that astar solvable. |
| `TestAStar.test_heuristic_value` | 21 | Assert that heuristic value. |

### `tests/test_competitive.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `square_board` | 45 | Construct a small enclosed competitive fixture board. |
| `swap_labels` | 55 | Exchange A/B positions and credits to test perspective symmetry. |
| `check_state_invariants` | 67 | Assert floor membership, nonoverlapping players/boxes, box conservation, disjoint valid credits, and round advancement. |
| `TestConflictRules` | 88 | unittest.TestCase grouping conflictrules regression checks; methods below contain assertions. |
| `TestConflictRules.setUpClass` | 90 | Prepare shared fixture geometry/controller data once for the test class. |
| `TestConflictRules.state` | 93 | Construct a test configuration with requested positions/boxes/credits/round. |
| `TestConflictRules.test_conflict_winner_follows_remaining_parity` | 100 | Assert that conflict winner follows remaining parity. |
| `TestConflictRules.test_odd_remaining_gives_agent_a_priority` | 109 | Assert that odd remaining gives agent a priority. |
| `TestConflictRules.test_even_remaining_gives_agent_b_priority` | 119 | Assert that even remaining gives agent b priority. |
| `TestConflictRules.test_swap_odd_winner_takes_cell_loser_cannot_take_vacated_cell` | 131 | Assert that swap odd winner takes cell loser cannot take vacated cell. |
| `TestConflictRules.test_swap_even_winner_is_agent_b` | 142 | Assert that swap even winner is agent b. |
| `TestConflictRules.test_competing_pushes_use_priority_and_yield` | 154 | Assert that competing pushes use priority and yield. |
| `TestConflictRules.test_push_destination_conflict_uses_priority_and_yield` | 164 | Assert that push destination conflict uses priority and yield. |
| `TestConflictRules.test_entering_cell_of_moving_opponent_is_not_a_conflict` | 176 | Assert that entering cell of moving opponent is not a conflict. |
| `TestConflictRules.test_entrant_diverts_when_occupant_cannot_move` | 185 | Assert that entrant diverts when occupant cannot move. |
| `TestConflictRules.test_waiting_occupant_keeps_cell_entrant_diverts` | 196 | Assert that waiting occupant keeps cell entrant diverts. |
| `TestConflictRules.test_wait_is_excluded_unless_explicitly_requested` | 208 | Assert that wait is excluded unless explicitly requested. |
| `TestConflictRules.test_blocked_push_is_not_a_valid_action` | 219 | Assert that blocked push is not a valid action. |
| `TestConflictRules.test_preference_list_orders_the_loser_diversion` | 235 | Assert that preference list orders the loser diversion. |
| `TestConflictRules.test_preference_fallback_rechecks_occupancy_and_boxes` | 258 | Assert that preference fallback rechecks occupancy and boxes. |
| `TestConflictRules.test_engine_never_declines_a_preferred_deadlock_push` | 275 | Assert that engine never declines a preferred deadlock push. |
| `TestConflictRules.test_loser_with_no_legal_alternative_stays_put` | 294 | Assert that loser with no legal alternative stays put. |
| `TestConflictRules.test_random_joint_actions_preserve_state_invariants` | 314 | Assert that random joint actions preserve state invariants. |
| `TestRule3PreferenceList` | 344 | unittest.TestCase grouping rule3preferencelist regression checks; methods below contain assertions. |
| `TestRule3PreferenceList.test_list_is_deterministic_legal_and_primary_pinned` | 345 | Assert that list is deterministic legal and primary pinned. |
| `TestRule3PreferenceList.test_list_is_label_symmetric` | 368 | Assert that list is label symmetric. |
| `TestRule3PreferenceList.test_fully_blocked_agent_submits_empty_list` | 381 | Assert that fully blocked agent submits empty list. |
| `TestEvaluation` | 397 | unittest.TestCase grouping evaluation regression checks; methods below contain assertions. |
| `TestEvaluation.setUpClass` | 399 | Prepare shared fixture geometry/controller data once for the test class. |
| `TestEvaluation.random_state` | 402 | Construct randomized valid competitive states for evaluation property tests. |
| `TestEvaluation.test_evaluate_is_exactly_zero_sum` | 419 | Assert that evaluate is exactly zero sum. |
| `TestEvaluation.test_evaluate_is_label_symmetric` | 430 | Assert that evaluate is label symmetric. |
| `TestEvaluation.test_credited_point_worth_more_than_unclaimed_one` | 441 | Assert that credited point worth more than unclaimed one. |
| `TestEvaluation.test_race_goes_to_the_closer_agent` | 457 | Assert that race goes to the closer agent. |
| `TestEvaluation.test_steal_value_grows_as_attacker_approaches` | 467 | Assert that steal value grows as attacker approaches. |
| `TestEvaluation.test_out_of_steps_freezes_the_score` | 481 | Assert that out of steps freezes the score. |
| `TestEvaluation.test_evaluation_scores_reflect_credit_and_perspective` | 493 | Assert that evaluation scores reflect credit and perspective. |
| `TestEvaluation.test_creates_deadlock_detects_corner_but_not_goal` | 504 | Assert that creates deadlock detects corner but not goal. |
| `TestEvaluation.test_deadlock_count_counts_pinned_corner_boxes` | 524 | Assert that deadlock count counts pinned corner boxes. |
| `TestAgentBehavior` | 541 | unittest.TestCase grouping agentbehavior regression checks; methods below contain assertions. |
| `TestAgentBehavior.setUpClass` | 543 | Prepare shared fixture geometry/controller data once for the test class. |
| `TestAgentBehavior.test_agent_b_shares_agent_a_engine` | 549 | Assert that agent b shares agent a engine. |
| `TestAgentBehavior.test_legacy_positional_signature` | 554 | Assert that legacy positional signature. |
| `TestAgentBehavior.test_heuristic_cache_keyword_maps_to_eval_cache` | 562 | Assert that heuristic cache keyword maps to eval cache. |
| `TestAgentBehavior.test_time_budget_is_respected` | 571 | Assert that time budget is respected. |
| `TestAgentBehavior.test_telemetry_is_populated` | 578 | Assert that telemetry is populated. |
| `TestAgentBehavior.test_agent_never_waits_while_a_move_exists` | 587 | Assert that agent never waits while a move exists. |
| `TestAgentBehavior.test_wait_only_when_fully_boxed_in` | 602 | Assert that wait only when fully boxed in. |
| `TestAgentBehavior.test_agent_preserves_unthreatened_credited_box` | 615 | Assert that agent preserves unthreatened credited box. |
| `TestAgentBehavior.test_agent_avoids_pointless_deadlock_when_values_tie` | 624 | Assert that agent avoids pointless deadlock when values tie. |
| `TestAgentBehavior.test_agent_takes_the_immediate_scoring_push` | 636 | Assert that agent takes the immediate scoring push. |
| `TestAgentBehavior.test_banned_actions_are_honoured` | 649 | Assert that banned actions are honoured. |
| `TestAgentBehavior.test_choose_action_records_history_and_last_action` | 657 | Assert that choose action records history and last action. |
| `TestAgentBehavior.test_history_entry_tracks_position_and_board` | 670 | Assert that history entry tracks position and board. |
| `TestAgentBehavior.test_full_game_keeps_all_invariants` | 676 | Assert that full game keeps all invariants. |

### `tests/test_competitive_gui.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `TestHumanInput` | 16 | unittest.TestCase grouping humaninput regression checks; methods below contain assertions. |
| `TestHumanInput.setUp` | 17 | Prepare fresh test fixtures, application stubs, or patches before each test. |
| `TestHumanInput.key` | 30 | Build and deliver a Pygame keyboard event to the GUI input handler. |
| `TestHumanInput.test_shift_cannot_submit_wait` | 35 | Assert that shift cannot submit wait. |
| `TestHumanInput.test_illegal_direction_is_rejected_and_legal_direction_accepted` | 41 | Assert that illegal direction is rejected and legal direction accepted. |
| `TestHumanInput.test_boxed_in_humans_do_not_wait_for_impossible_key_submission` | 47 | Assert that boxed in humans do not wait for impossible key submission. |

### `tests/test_core.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `TestCore` | 5 | unittest.TestCase grouping core regression checks; methods below contain assertions. |
| `TestCore.test_parser` | 6 | Assert that parser. |
| `TestCore.test_actions` | 15 | Assert that actions. |

### `tests/test_search.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `TestSearch` | 6 | unittest.TestCase grouping search regression checks; methods below contain assertions. |
| `TestSearch.test_ucs_solvable` | 7 | Assert that ucs solvable. |

### `tests/test_search_identity.py`

Automated regression tests. Named test methods below describe the property or scenario asserted; helpers build fixtures or independent reference calculations.

| Symbol (nested scope included) | Line | Responsibility |
|---|---:|---|
| `reference_roots` | 22 | Compute exhaustive unpruned finite-depth root values as an independent reference for planner comparisons. |
| `reference_roots.actions` | 24 | Nested reference helper listing legal directions, with forced WAIT if no direction exists. |
| `reference_roots.child` | 28 | Nested reference helper resolving one joint pair with both primary-pinned fallback lists. |
| `reference_roots.value` | 36 | Nested reference maximin recursion used to validate pruned completed-depth values. |
| `TestStateIdentityAndCaches` | 48 | unittest.TestCase grouping stateidentity and caches regression checks; methods below contain assertions. |
| `TestStateIdentityAndCaches.test_key_distinguishes_the_same_configuration_at_different_rounds` | 52 | Assert that key distinguishes the same configuration at different rounds. |
| `TestStateIdentityAndCaches.test_eval_cache_is_board_scoped` | 71 | Assert that eval cache is board scoped. |
| `TestStateIdentityAndCaches.test_transition_cache_is_board_scoped` | 87 | Assert that transition cache is board scoped. |
| `TestCompletedSearch` | 114 | unittest.TestCase grouping completedsearch regression checks; methods below contain assertions. |
| `TestCompletedSearch.planner` | 115 | Construct a test planner with controlled state, perspective, round limit, and generous deadline. |
| `TestCompletedSearch.test_pruned_search_matches_full_joint_matrix_both_parities_and_sides` | 118 | Assert that pruned search matches full joint matrix both parities and sides. |
| `TestCompletedSearch.test_terminal_stops_expansion_and_uses_actual_score` | 135 | Assert that terminal stops expansion and uses actual score. |
| `TestCompletedSearch.test_partial_iteration_never_replaces_completed_results` | 143 | Assert that partial iteration never replaces completed results. |
| `TestCompletedSearch.test_partial_iteration_never_replaces_completed_results.interrupted` | 149 | Nested injected interruption that allows a completed shallow iteration and aborts the next, exercising atomic result retention. |
| `TestCompletedSearch.test_no_completed_iteration_returns_legal_direction` | 161 | Assert that no completed iteration returns legal direction. |
| `TestCompletedSearch.test_deadline_gc_guard_restores_previous_setting_on_error` | 169 | Assert that deadline gc guard restores previous setting on error. |
| `TestCompletedSearch.test_live_and_search_use_identical_preferences_for_both_players` | 183 | Assert that live and search use identical preferences for both players. |
| `TestCompletedSearch.test_legal_sacrifices_are_not_filtered` | 194 | Assert that legal sacrifices are not filtered. |
| `TestCompletedSearch.test_contested_scoring_is_backed_by_exhaustive_continuations` | 201 | Assert that contested scoring is backed by exhaustive continuations. |
| `TestCompletedSearch.test_repetition_only_breaks_equal_value_ties` | 221 | Assert that repetition only breaks equal value ties. |
| `TestCompletedSearch.test_safe_score_both_perspectives_at_round_limit` | 229 | Assert that safe score both perspectives at round limit. |
| `TestCompletedSearch.test_preference_submission_is_cached_inside_decision` | 236 | Assert that preference submission is cached inside decision. |
| `TestScoringAndEvaluation` | 245 | unittest.TestCase grouping scoring and evaluation regression checks; methods below contain assertions. |
| `TestScoringAndEvaluation.test_credit_removal_and_goal_to_goal_transfer` | 246 | Assert that credit removal and goal to goal transfer. |
| `TestScoringAndEvaluation.test_filled_goals_do_not_end_game` | 255 | Assert that filled goals do not end game. |
| `TestScoringAndEvaluation.test_bounds_are_physical_not_just_wall_membership` | 260 | Assert that bounds are physical not just wall membership. |
| `TestScoringAndEvaluation.test_opportunities_do_not_sum_independent_deliveries` | 265 | Assert that opportunities do not sum independent deliveries. |
| `TestScoringAndEvaluation.test_uncredited_goal_box_does_not_count_as_zero_step_delivery` | 272 | Assert that uncredited goal box does not count as zero step delivery. |
| `TestScoringAndEvaluation.test_blocked_round_consumes_time_without_overlap` | 277 | Assert that blocked round consumes time without overlap. |


**Catalogue coverage:** 49 Python files; 309 class/function definitions, including methods and nested named functions.

## 10. Non-Python files, maps, and assets

### 10.1 Project and documentation files

| File | Purpose and defense relevance |
|---|---|
| `.gitignore` | Excludes generated Python/test caches, macOS metadata, and three assignment-context filenames from version control. Does not implement game behavior. |
| `requirements.txt` | Declares pygame for interface/animation, SciPy for assignment optimization, and NumPy for matrix construction. Versions are currently unpinned, so reproduce the demonstration with a known environment. |
| `README.md` | Installation, launch, and experiment overview. Some older statements about immutable states and puzzle shortcuts are broader/outdated; use this guide's code-based qualifications. |
| `docs/project_analysis.md` | Prior structure/OOP review, cleanup record, and proposed future refactors. It is analysis rather than runtime configuration. |
| `docs/university_defense.md` | This presentation guide, algorithm explanations, complete symbol catalogue, evidence, and defense questions. |
| `src/competitive/rules.md` | Human-readable rule/design notes. Its final section contains proposals, including an outdated stationary-blockage default; the transition implementation is authoritative for current behavior. |

### 10.2 Every bundled map

Maps are data files, not classes or algorithms. Menus discover `.txt` files dynamically; changing a map changes the problem instance. Counts below include puzzle `C` as both one box and one goal.

| File | Boxes / goals | Role and characteristic |
|---|---:|---|
| `maps/test_solvable.txt` | 1 / 1 | Small solvable parser/search/heuristic fixture and safe first demonstration. |
| `maps/test_map.txt` | 1 / 1 | Small core parser/action fixture; box starts against the upper-right enclosing walls, useful for discussing deadlocks. |
| `maps/benchmark_1.txt` | 2 / 2 | Puzzle comparison with an internal wall barrier and two deliveries. |
| `maps/benchmark_2.txt` | 7 / 7 | Larger irregular puzzle with an initially completed box; used for algorithm benchmarking. |
| `maps/user_map.txt` | 7 / 7 | User puzzle variant of the larger layout, with an initially completed box. |
| `maps/competitive/arena_open.txt` | 2 / 2 | Open symmetric-style arena for easy visual comparison and the default competitive launch. |
| `maps/competitive/capacity_lab.txt` | 3 / 3 | Larger separated areas and narrow connections, stressing navigation and board preprocessing. |
| `maps/competitive/corridors.txt` | 3 / 3 | Obstacle/corridor layout with competing box approaches. |
| `maps/competitive/main.txt` | 6 / 6 | Dense irregular competitive layout used by current tests/experiments. |
| `maps/competitive/test_race.txt` | 1 / 1 | Narrow-lane race fixture with a shared box and goal. |

The previously deleted dense-goals map is not part of this snapshot. No algorithm depends on it.

### 10.3 Every sprite pair

Each table entry denotes **two actual files**, one `.png` sprite sheet and its matching `.json` frame-coordinate data, located under `src/assets/`. Both are loaded by `Animator.__init__`. JSON describes frame rectangles; it does not contain strategy instructions. Direction indices are 1=south, 3=west, 5=north, 7=east. Other direction assets were removed because motion is cardinal.

| PNG + JSON filename stem | Animation role |
|---|---|
| `Running/Businessman_Running_dir1` | South walking/running |
| `Running/Businessman_Running_dir3` | West walking/running |
| `Running/Businessman_Running_dir5` | North walking/running |
| `Running/Businessman_Running_dir7` | East walking/running |
| `Push/Businessman_Push_dir1` | South box push |
| `Push/Businessman_Push_dir3` | West box push |
| `Push/Businessman_Push_dir5` | North box push |
| `Push/Businessman_Push_dir7` | East box push |
| `Idle/Businessman_Idle_dir1` | South idle stance |
| `Idle/Businessman_Idle_dir3` | West idle stance |
| `Idle/Businessman_Idle_dir5` | North idle stance |
| `Idle/Businessman_Idle_dir7` | East idle stance |
| `Die/Businessman_Die_dir1` | South defeat animation |
| `Die/Businessman_Die_dir3` | West defeat animation |
| `Die/Businessman_Die_dir5` | North defeat animation |
| `Die/Businessman_Die_dir7` | East defeat animation |
| `Uppercut/Businessman_Uppercut_dir1` | South victory animation |
| `Uppercut/Businessman_Uppercut_dir3` | West victory animation |
| `Uppercut/Businessman_Uppercut_dir5` | North victory animation |
| `Uppercut/Businessman_Uppercut_dir7` | East victory animation |

### 10.4 Every board texture and license

| File under `src/assets/soko/` | Runtime purpose |
|---|---|
| `PNG/Retina/Blocks/block_03.png` | Wall tile in both modes. |
| `PNG/Retina/Ground/ground_06.png` | Empty-floor tile in both modes. |
| `PNG/Retina/Environment/environment_02.png` | Goal marker in both modes. |
| `PNG/Retina/Crates/crate_02.png` | Ordinary uncredited/off-goal box. |
| `PNG/Retina/Crates/crate_03.png` | Completed puzzle box and A credit/color presentation. |
| `PNG/Retina/Crates/crate_04.png` | B credit/color presentation. |
| `License.txt` | Kenney asset license/attribution information; retain it with those assets. It does not establish the license of the separate Businessman sprites. |

There are 26 current PNG files and 20 animation JSON files. The menus use a solid cyan fill, so no menu background image is required. Competitive constructor references optional `assets/agent_a.png` and `assets/agent_b.png`, which are not bundled; the placeholder loader intentionally supplies colored labels and normal sprite animation supplies the visible actors.

Generated `__pycache__`/`.pyc` files are interpreter caches with no authored functions to defend. `.git` contains repository history and metadata rather than game code. `results/` outputs are created by experiment commands and are not required to run the game; no historical output is treated here as current measured evidence.

## 11. A small measured example and quick revision notes

The following counts were rerun on `maps/test_solvable.txt` for this guide on 9 October 2026:

| Solver | Solution | Cost | Generated | Expanded |
|---|---|---:|---:|---:|
| UCS | EAST, SOUTH | 2 | 14 | 5 |
| A* | EAST, SOUTH | 2 | 9 | 3 |

Initial matching heuristic is 1: one push is needed, while the optimal actual path also requires one walk. Both algorithms find the same optimal two-action solution; A* expands fewer nodes in this example. This small example illustrates the heuristic mechanism, not a general speedup claim.

For a competitive action matrix with A-perspective continuation values:

| A submission | B reply 1 | B reply 2 | Row minimum |
|---|---:|---:|---:|
| Action 1 | 1.0 | -0.2 | -0.2 |
| Action 2 | 0.3 | 0.1 | 0.1 |

Pure maximin selects Action 2 because 0.1 is the larger row minimum, even though Action 1 has the most optimistic individual outcome. The table is an explanatory example, not a recorded game position.

Before the presentation, remember these distinctions:

- Puzzle g counts **all actions**; matching h counts **relaxed pushes**.
- A*'s lower bound is a mathematical property; competitive opportunity/pressure are manually weighted strategic estimates.
- “Exact steps” is exact in a **single-box model**; live conflicts and other boxes are handled by transitions/search.
- A completed competitive depth compares **all allowed roots at the same horizon**.
- Finished competitive boxes have **reversible credit**, not permanent ownership.
- B yielding changes **one controller's permitted root choices**, not the transition rules or final draw rule.
- The current architecture uses useful OOP and pure functions together; more classes alone would not improve it.

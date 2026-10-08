# Project Analysis — Sokoban (Single-Agent & Competitive)

Deep-dive documentation of the repository: (1) what every file is for,
(2) the heuristic used in each version, (3) the algorithms used and how
they work. Verified against the current source tree (rewrite documented
in `docs/competitive_rewrite.md`; the older GBFS-era design survives
only in `docs/ai_search_improvement_plan.md` and
`docs/audit_findings_report.md`).

---

## 1. What each file is for

### Repository root

| File | Purpose |
|---|---|
| `main.py` | CLI entry point. Checks dependencies (pygame/scipy/numpy) and dispatches: `gui [map]` (launcher menu), `competitive [map] [steps] [ai_a] [ai_b]` (direct competitive launch), `benchmark` (UCS vs A\*), `verify [map]` (heuristic property checks). |
| `README.md` | Project overview: single-agent features (UCS, A\*, matching heuristic), install/run/test instructions, competitive-mode pointer. |
| `requirements.txt` | Python dependencies (pygame, scipy, numpy). |
| `optimization.md` | Historical "AI optimization plan" for the *old* competitive agent (global TT, PV ordering, alpha-beta, eval cache). README states it no longer describes the current AI. |
| `sim.py` | One-off headless harness: runs ~40 AI-vs-AI rounds on `arena_open.txt` through the real joint engine, printing submitted/executed actions, horizons and conflicts. Console output kept in `sim_out.txt` / `sim_output.txt`. |
| `debug2.py` | One-off diagnostic: hard-codes a dense-goals state, runs `AgentB`, dumps `last_search` telemetry as JSON. |
| `patch_juice.py`, `patch_rewind.py` | Historical one-shot text patchers that used to rewrite the competitive GUI (juice: blur/particles/shake; rewind: key-repeat + history scrubbing). Not runtime code — do not re-run. |
| `.gitignore` | Excludes `__pycache__`, `*.pyc`, assignment-context markdown. |

### `src/core/` — single-agent domain model

| File | Purpose |
|---|---|
| `state.py` | `Action` enum (NORTH/SOUTH/EAST/WEST), static `Board` (walls/goals, kept out of states), immutable `GameState` (`agent` tuple + `boxes` frozenset). State identity = `hash((agent, boxes))`; goal test = exact set equality `boxes == goals`. |
| `parser.py` | ASCII map parser for single-agent maps: `%` wall, `A` agent, `B` box, `D` goal, `C` box-on-goal, space floor. Returns `(GameState, Board)`. |
| `actions.py` | `get_successors(state, board)` — the only successor generator: four directions, walk or push (box destination must be free floor). No costs attached; no pruning here. |

### `src/search/` — single-agent search algorithms

| File | Purpose |
|---|---|
| `algorithm.py` | Abstract base `SearchAlgorithm` fixing the return contract: `(actions, total_cost, generated, expanded)`, `actions=None` if unsolvable. |
| `node.py` | `Node(state, parent, action, path_cost)` + `reconstruct_path()` (parent-walk back to the action list). |
| `ucs.py` | Uniform-Cost Search (graph search, unit cost). |
| `astar.py` | A\* search with an injected heuristic callable; also the module-level `astar_search()` wrapper. |

### `src/heuristics/` — single-agent heuristic components

| File | Purpose |
|---|---|
| `push_distance.py` | `precompute_push_costs(board)`: per-goal reverse BFS over box positions using the 2-cell push geometry → `goal → {box_cell: min pushes}`. This is where static deadlocks (corners/wall lines) become implicit `inf`. |
| `matching.py` | `MatchingHeuristic`: builds a box×goal cost matrix from those tables and solves the assignment problem with `scipy.optimize.linear_sum_assignment` (Hungarian/JV) — the "Minimum Legal Push Distance Matching" heuristic. |
| `deadlock.py` | `is_deadlock(state, board)`: explicit **dynamic** deadlock test — a non-goal box inside any 2×2 square fully occupied by boxes/walls. |

### `src/experiments/` — measurement harnesses

| File | Purpose |
|---|---|
| `benchmark.py` | Runs UCS then A\* on the benchmark maps, **asserts equal solution cost**, appends `results/benchmark_results.csv` (cost, length, generated, expanded, time). |
| `verify_heuristic.py` | Samples up to 200 states per map and empirically checks **admissibility** (`h(s) ≤ true cost` via UCS) and **consistency** (`h(s) ≤ 1 + h(s′)`), writing `results/verification_results.txt`. |
| `competitive_benchmark.py` | Competitive harness: per-map/per-perspective **decision traces** (chosen action, completed depth, nodes, root values, per-reply diagnostics) and optional **seat-swapped matches** against an on-disk baseline snapshot of `evaluation/transition/agent_a`; emits JSON to `results/`. |

### `src/gui/` — presentation layer (shared + single-player)

| File | Purpose |
|---|---|
| `app.py` | Unified launcher: main menu with two mode cards (1-Player Puzzle / 2-Players Competitive); dispatches to `SinglePlayerApp` or `CompetitiveApp` and returns to the menu. |
| `single.py` | Single-player game screen: map-select setup (algorithm toggle A\*/UCS, live preview), background solve thread with token guard, animated auto-replay of the solution, HUD (steps, goals, cost/generated/expanded/ms), scrubbing controls. |
| `common.py` | Shared visual toolkit both modes use: colors/TILE, fonts, image loading with fallbacks, gradient/blur helpers, glass panels, `Button`, the `Animator` (Businessman sprite sheets), and map-preview rendering. |

### `src/competitive/` — the two-player game

| File | Purpose |
|---|---|
| `rules.md` | The rules spec given to the AI: simultaneous rounds, four directions only (no selectable WAIT), parity conflict priority, Rule-3 ranked fallbacks, temporary-credit scoring, proposed defaults. |
| `state.py` | Competitive data model: `Action` (incl. `WAIT` sentinel), shared `Board` with precomputed tables (all-pairs walk `distances`, `push_costs`, `exact_step_costs` — exact walk+push step BFS per goal over `(box, player)` states) and a monotone `serial` used to scope caches; `CompetitiveState` (both agents, shared boxes, credit sets `boxes_on_goals_a/b`, `step`), with hash-with-step and hash-without-step variants. |
| `parser.py` | Competitive map parser: `%` wall, ` `/`.` floor, `D` goal, `B` box (starts uncredited), `A` agent-A start, `C` agent-B start. Requires A, C and ≥1 goal. |
| `transition.py` | The **shared deterministic joint engine**: intent/conflict detection, parity priority, Rule-3 loser diversion, credit bookkeeping, `Outcome`, memoized `resolve_joint_action_outcome` (40 000-entry cache), `get_valid_actions`. Used identically by live play, the AI's search, the GUI and tests. |
| `preferences.py` | `round_preferences(...)`: the cheap, deterministic, non-recursive Rule-3 fallback ranking shared by live play, the engine's diversions and the search (so all three simulate identical fallbacks). |
| `evaluation.py` | Score-unit, zero-sum evaluation (`evaluate`, `evaluation_components`), delivery-cost BFS/exact-step helpers, deadlock helpers (`deadlock_count`, `creates_deadlock`), module + injected caches. |
| `agent_a.py` | The entire decision pipeline: `best_action()` → `_Planner` (time-bounded iterative-deepening alpha-beta maximin over simultaneous rounds) and the `AgentA` controller (budget, history, preference submission, telemetry). |
| `agent_b.py` | `class AgentB(AgentA)` with only `perspective = "B"` — same engine from the other seat; evaluation mirrored, parity still uses real player labels. |
| `rules.md` | (see above) |
| `gui/__init__.py`, `gui/app.py` | `CompetitiveApp`: setup screen (map list, AI/human seat toggles, steps ±5), main loop with background compute threads, history scrubbing, human input (WASD / arrows), HUD (scores, **Steps Left**, last round's actions + ms, **conflict-priority readout**), floating "X WINS CONFLICT" badge, box color = neutral off goal / done on goal, result overlay. |

### `tests/`

| File | Coverage (approx. count) |
|---|---|
| `test_core.py` | (2) Map parsing and successor legality. |
| `test_search.py` | (1) UCS solves the fixture at cost 2. |
| `test_astar.py` | (2) A\* matches UCS optimum; heuristic tightness on the fixture. |
| `test_competitive.py` | (~43) Engine contract: parity winner, diversion/fallbacks, WAIT rules, random-joint-action invariants, Rule-3 list properties (deterministic, primary-pinned, label-symmetric), evaluation (exact zero-sum, gradients, terminal freeze), deadlocks, agent behavior (budget respected, never waits needlessly, banned actions, full-game sweep). |
| `test_competitive_gui.py` | (3, skip w/o pygame) Windowless GUI input rules (WAIT gating, illegal-move rejection, boxed-in auto-WAIT). |
| `test_search_identity.py` | (~20) Planner vs independent exhaustive joint-matrix reference across perspectives/parities/horizons, cache identity, completed-iteration atomicity, deadline/GC guard, live-vs-search preference parity, repetition tie-breaks, scoring edge cases. |

### Other directories

- **`docs/`** — `implementation_plan.md` + `research_log.md` (original single-agent plan & design log); `ai_search_improvement_plan.md` + `audit_findings_report.md` (audit/phases of the *previous* GBFS-era competitive AI — historical); `competitive_rewrite.md` (**current** authoritative competitive-AI report).
- **`maps/`** — single-agent maps (`benchmark_*.txt`, `test_*.txt`, `user_map.txt`); `maps/competitive/*.txt` — arena maps (5 fixtures).
- **`results/`** — `benchmark_results.csv` (UCS vs A\*), `verification_results.txt` (0 admissibility/consistency violations), `competitive_comparison.json` + `competitive_matches*.json` (decision/match traces: depth, nodes, root values, per-turn submissions/executions, scores).
- **`src/assets/`** — art (Kenney Sokoban tiles, Businessman sprite sheets + frame JSON, sci-fi pack). No logic.

---

## 2. Heuristic used in each version

### 2.1 Single-agent version — *Minimum Legal Push Distance Matching*

Three cooperating components (no weighted sum — the pieces plug into
different stages of A\*):

**a) Legal push distances — `precompute_push_costs(board)`**
For every goal, a reverse BFS over *box positions*. A box arriving at
`curr` from direction `(d)` came from `curr − d`, with the player having
stood at `curr − 2d` (standard push geometry); if both cells are in
bounds and off-wall, `cost(prev) = cost(curr) + 1`. Result:
`goal → {box_cell: min pushes}`. Cells never reached are simply absent
→ treated as `inf`.
*This is also the static-deadlock detector:* a box in a corner or
against a wall line has no legal reverse-push predecessor, so its cost
is `inf`. There is no separate corner/wall-line rule in the code.

**b) Optimal assignment — `MatchingHeuristic.__call__`**
Cost matrix `M[box][goal] = push_costs[goal].get(box, inf)`, then
`scipy.optimize.linear_sum_assignment` (Hungarian/Jonker–Volgenant)
solves the **minimum-weight matching** of boxes to goals; `h` = total
matched cost. Boxes already on a goal cost 0. An infeasible row (a box
that can reach nothing) makes the whole state `inf` (scipy raises on
non-finite entries → caught → `inf`).
Why *admissible*: every push costs ≥1 step in the search (unit step
cost), the BFS ignores other boxes/player detours (a relaxation, so it
underestimates), and the matching is the minimum over pairings. Why
*consistent*: one action performs ≤1 push, and BFS distances obey the
triangle inequality, so `h(s) − h(s′) ≤ 1 = c`. Empirically verified by
`verify_heuristic.py` (0 violations across sampled states).

**c) Dynamic deadlock — `is_deadlock(state, board)`**
A non-goal box that shares any 2×2 square entirely filled by
boxes/walls is pruned immediately (its standing cell for every push is
occupied). Used as a successor filter in *both* UCS and A\*.

**How they combine in A\*:** `f = g + h`; successors are rejected if
`is_deadlock(s′)` **or** `h(s′) == inf` (the "infinite heuristic =
deadlock" prune). UCS uses only the 2×2 prune (it must stay
uninformed).

### 2.2 Competitive version — two heuristic layers

**Layer 1 — Board precomputations (`competitive/state.py`)**
All computed once per board and shared by everything:
- `distances`: all-pairs walk BFS (walls + boxes block; opponent ignored);
- `push_costs`: single-agent reverse-push tables (reused);
- `exact_step_costs`: exact per-goal BFS over `(box_cell, player_cell)`
  states → `exact_steps(box, player, goal)` = true minimum walk+push
  steps for a single box (used where precision matters after the first
  push).

**Layer 2a — the evaluation heuristic (`evaluation.py`)**
`evaluate(state, board, perspective, max_steps)` = sum of three
A-perspective terms in *points*, sign-flipped for B (exactly zero-sum):

1. **`credit`** = `score_a − score_b` (each locked box = `W_LOCKED = 1.0`
   point). At/after the round limit this is the *entire* value —
   terminal values are exact, other terms zeroed.
2. **`opportunity`** = best *single* next delivery per side (never a sum
   of independent plans): for each uncredited box, delivery cost =
   walk to a real first push + 1 + `exact_steps` tail, minimized over
   free goals. Value `v = 0.65 / (1 + 0.12·cost)` if deliverable within
   the remaining rounds, else 0; then a **race discount** if both sides
   can make it: `v *= 0.5 + 0.5·clamp((rival−mine)/4, −1, 1)`. Keep the
   max per side; `opportunity = best_a − best_b` (bounded by ±0.65).
3. **`pressure`** = bounded steal-threat: for each opponent-credited
   box and push direction, `0.10 / (walk_to_approach + 1)` if the push
   is physically available within the remaining rounds; max per side,
   differenced. Guidance toward contested boxes, not a proof of steal.

Caching: module `_eval_cache` or caller dict keyed
`(board.serial, state, max_steps)` (limit 40 000), plus `lru_cache` on
walking BFS (8192) and delivery-min (16384).

**Layer 2b — the action-ranking heuristic (`preferences.py`)**
`round_preferences` orders the legal directions for Rule-3 fallbacks
(identical list used by live play, engine diversion, and search) with
the ascending key:

```
(-gain, creates_deadlock(pos, action, boxes, board), cost, attack)
```

- `gain` = +1 push onto a goal, −1 strip my own credited box, +1 push
  the opponent's credited box off (score now, don't self-harm, do steal);
- `creates_deadlock` = the push leaves a box it can never leave (irreversible pushes sort last);
- `cost` = min `exact_steps` from the post-push cell to any free goal;
- `attack` = walk distance from the post-push cell to the nearest
  opponent-credited box (steal proximity tiebreak).

No planner recursion — deliberately cheap and deterministic so search
and engine agree exactly.

---

## 3. Algorithms used

### 3.1 Single-agent — Uniform-Cost Search and A\*

Both are **graph searches over `(agent, boxes)` with unit step cost
(walk or push = 1)**, sharing identical plumbing:

- **Open list:** binary heap of `(key, tie_counter, node)` — `g` for
  UCS, `f = g + h` for A\*; the counter makes tie-breaking FIFO.
- **Duplicate handling:** `explored: {state → best g}` updated at
  *generation*; stale pops (stored `g` < node's `g`) are skipped, so a
  strictly cheaper path may re-open a state.
- **Goal test on pop** (`boxes == goals`) → optimal under admissible +
  consistent `h`.
- **Successors:** `get_successors` (walk or push, walls/box collision
  checked).
- **Pruning:** UCS rejects 2×2 deadlocks; A\* rejects 2×2 deadlocks
  **and** any state with `h(s) == inf` (box statically unable to reach
  any goal).
- **Failure:** heap exhaustion → `actions = None`.

A\* is therefore optimal-but-faster (benchmark: identical cost,
~4.6× fewer generated states on `benchmark_2`), UCS is the reference
used to verify the heuristic.

### 3.2 Competitive engine — deterministic simultaneous round resolution (`transition.py`)

Every round (one per step, always consumes a step):
1. **Intent:** each side's action → physical intent (destination must be
   floor; a push needs a free destination). `None`/`WAIT` = no intent
   (WAIT is a forced-immobility sentinel, never a selectable action).
2. **Parity conflict:** `conflict_winner = "A" if (max_steps − step) % 2
   else "B"` (odd remaining rounds → A). Fires on the rule's conflict
   cases: same destination, swap, both pushing the same box, entering a
   push destination / two pushes to one destination. The **loser's
   intent is voided**.
3. **Entrant conflict:** walking into a non-moving occupant's cell →
   the entrant yields (no parity involved).
4. **Rule-3 diversion:** each loser moves along *its own* fixed
   preference list, parity winner settling first so the loser diverts
   around it; a fallback can never overturn the winner's action. No
   alternative → forced stay (`WAIT` internally).
5. **Re-check:** a stationary loser invalidating the winner's entry
   makes the winner divert too; still impossible → both stay.
6. **Commit:** both actions applied together; credit bookkeeping strips
   credit when a credited box leaves a goal and re-credits to the pusher
   (goal→goal transfers). Credits kept only where a box sits on a goal.

`resolve_joint_action_outcome` memoizes the whole resolution
(40 000-entry cache keyed by board, state incl. credits and remaining
rounds, both actions, and both preference lists).

### 3.3 Competitive AI (Agent A **and** Agent B) — time-bounded
iterative-deepening alpha-beta **maximin over simultaneous rounds**
(`agent_a.py`)

One implementation serves both seats (`AgentB(AgentA)` with opposite
perspective; evaluation sign-flipped, parity untouched).

**Tree structure.** A *node* is a full game state; one ply = one
simultaneous round:
- **MAX node** (`value()`): my legal actions (from the shared
  preference list, PV move moved to front) → pick the highest security
  value.
- **MIN row** (`row_value()`): for a fixed my-action, the opponent's
  *every* reply is enumerated (from *its* preference list) and the
  **minimum** (worst-for-me) child value is taken — a conservative
  maximin opponent model: the opponent is assumed to play its worst
  case for us, following the same fallback policy as the engine.
- Each (my action × reply) pair is resolved by the **real engine**
  (`resolve_joint_action_outcome` with both sides' preference lists),
  so search and live play simulate identical conflicts/fallbacks.

**Pruning & tables.** Standard alpha-beta (`alpha ≥ beta` cutoffs in
both halves, rows sorted ascending by child value for better pruning
while keeping *all* replies), plus a transposition table keyed
`(state, depth)` storing `(value, flag ∈ exact/lower/upper)` with proper
bound adjustments. Principal-variation move ordering per state.
Depth `d` = `d` rounds ahead; leaves (`depth == 0` or terminal) are
evaluated with `evaluate` (terminal = exact credit difference).

**Iterative deepening & time management.**
- Deadline = start + `budget − min(0.04, 10% of budget)` (reserve for
  telemetry/submission); a monotonic-clock `check()` raises `_Timeout`
  inside every operation.
- `for depth in 1, 2, 3, …` until the deadline; **only fully completed
  iterations publish root values** (atomic wholesale replacement — a
  partial deeper search never clobbers the previous results), and the
  returned action is always from the **last completed depth** (or the
  preference-first legal root if none completed). `TIME_LIMIT = 1.0 s`
  default per decision.

**Root selection.** `best = argmax (value, tie_key)` where
`tie_key = (immediate worst child value, −repetition count)` — exact
maximin value first, then the better immediate worst case, then
avoiding a position seen in the last 6 own states. Completed roots are
re-sorted by the same key to seed the next depth's ordering.

**Support machinery.**
- Caches: TT, child rows per `(state, action)`, preference lists per
  `(state, who)`, evaluation cache (shared `heuristic_cache`), all
  board-scoped via `board.serial`.
- `_bounded_allocations`: cyclic GC is **disabled during a decision**
  (refcounting still runs) because profiling found >100 ms GC pauses;
  re-entrant and exception-safe.
- Rule-3 submission: `choose_action` computes the preference list
  inside the budget, hoists the chosen action to the front, and stores
  it so the GUI's `preference_list()` submits exactly what the search
  used.
- Telemetry: `LAST_SEARCH` records engine name, completed depth, nodes,
  cache hits, per-root values with per-reply diagnostics (executed
  actions, conflict flag, immediate value, evaluation components, both
  sides' preference lists), iterations, and wall time — consumed by
  `competitive_benchmark.py` and the GUI.

**Opponent modeling caveat (documented approximation).** The opponent
is assumed to follow the same `round_preferences` fallback policy; the
engine itself defaults a side without a submitted list to its own
evaluate-ranked fallback. The search models worst-case *choices* over
all rule-permitted replies, not the opponent's hidden intent.

### 3.4 Experiments

- `benchmark.py`: UCS vs A\* race with an equality assertion on cost —
  validates admissibility-by-construction in practice.
- `verify_heuristic.py`: sampled-state admissibility/consistency audit
  of the matching heuristic against UCS ground truth.
- `competitive_benchmark.py`: reproducible decision + seat-swapped match
  comparison (baseline snapshot vs live engine) with full search
  telemetry per decision, invariant assertions each turn
  (no overlap, box count conserved), JSON output.

### Note on versions/history

The competitive AI has had two architectures: an earlier
**GBFS/beam-style planner with staged tactical windows, deep/strike/
robust ranking keys and weight-tuned evaluation** (fully documented —
including its phases, audits and fixes — in
`docs/ai_search_improvement_plan.md` and `docs/audit_findings_report.md`),
and the **current** iterative-deepening pure-action maximin described
above (see `docs/competitive_rewrite.md`, which supersedes those
docs). `optimization.md` likewise describes the old agent. When reading
older docs, check them against `src/competitive/agent_a.py` and
`src/competitive/evaluation.py` before assuming a constant or helper
still exists.

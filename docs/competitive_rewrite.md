# Competitive AI rewrite — 2026-10-08

This report supersedes the older GBFS architecture and optimization proposals.
The existing uncommitted working tree was the starting point, not Git HEAD.
A pre-edit copy was saved at
`C:\Users\PROTEC~1\AppData\Local\Temp\sokoban-before-rewrite-20261008-202809`.

## Diagnosis verified against the working tree

- The supplied AI crashed on an undefined `_Planner._joint_prefs` call. The
  immediate-scoring regression reproduced that error before any rewrite.
- The exact shallow tactical value and strike distance outranked almost all
  deeper GBFS evidence. Reported maximum path depth did not measure the horizon
  actually used to compare decisions.
- Seeding could return a result from partially evaluated roots. Forced-event
  extensions could swallow a timeout and retain unextended leaf values.
- Search supplied only its own fallback list; the live GUI supplied both.
  The engine's legacy fallback evaluated alternatives after seeing the other
  move. Those were different policies despite sharing the transition function.
- Several heuristic terms projected independent deliveries into points without
  proving that a single agent could complete those deliveries sequentially.
- Human Shift keys submitted WAIT, and invalid directional submissions were
  not rejected in the GUI.

Historical diagnoses were not assumed correct. A useful example is the old
"safe immediate scoring" fixture: A at (3,2), B at (5,5), box at (3,3), goal at
(3,4). After A scores south, B can walk west, west, then push north from (3,5)
to strip the point. The test now checks this contested position against a full
five-round reference search. A separate corner-goal fixture checks genuinely
safe immediate scoring. This does not prove that declining every contested
delivery is optimal; it establishes that the old unconditional assertion was
not a safe-scoring test.

## Current responsibilities

| File | Responsibility |
| --- | --- |
| `transition.py` | Physical legality, simultaneous conflict resolution, fixed fallback order, credits |
| `preferences.py` | Cheap starting-state ranking of all legal alternatives; no recursive planner |
| `evaluation.py` | Actual credit plus bounded next-delivery and approach estimates |
| `agent_a.py` | Iterative deepening, joint maximin backups, pruning, deadline handling and diagnostics |
| `agent_b.py` | Same search from B's perspective; parity priority still uses the actual player labels |

Each node represents a whole round. For each primary direction, the search
considers every legal opponent primary direction and resolves their pair
jointly. Both sides' preference lists are generated from that node's starting
state using the same policy as live AI play. This is pure-action maximin: the
best worst-case continuation of a fixed action, not a mixed-strategy equilibrium
and not knowledge of an opponent's live submission. Other agents/humans may
have different preference policies; that remains an opponent-model assumption.

Every root receives an exact search value at the same completed horizon. An
unfinished deeper iteration cannot replace any completed root value. Internal
alpha/beta cuts and depth-specific exact/lower/upper transposition entries
reduce work without discarding relevant responses. No optimistic descendant
peak or forced scoring sequence selects a move. Exact value ties prefer the
immediate worst-case value, then avoid recent repetitions, then preserve stable
preference order. Before any iteration completes, the agent has a legal fallback.

The search stops at the round limit and returns score difference in points.
Nonterminal evaluation estimates at most one next delivery per side rather
than adding independent delivery times into a fictitious schedule. Its first
push respects current box blockage; subsequent single-box distances are relaxed.
A bounded approach term encourages pressure on opposing credited boxes without
claiming that proximity proves a successful removal. Future conflicts and
ownership changes are resolved by search. Physically legal sacrifices and
deadlock pushes remain searchable.

Board neighbors, walking distances and delivery costs are cached after profiling
identified walking BFS as the main cost. State/transition/evaluation cache keys
include the applicable board, credits and time; transition keys also include
preferences. Search tables are scoped to one decision and distinguish depth and
bound type. Board preprocessing is done at level load, outside decision timing.

Python cyclic garbage collection caused measured 100–216 ms pauses inside single
expansions, including deadline overruns. Decision calls temporarily defer cyclic
collection while building acyclic search tables; reference-count cleanup remains
active. The previous GC setting is restored even on exceptions and after nested
or concurrent calls. A 40 ms maximum return reserve covers normal cleanup and
telemetry. The live preference submission is cached inside the decision budget.

## Rules retained and discrepancies resolved

- Remaining rounds before the move determine priority: odd A, even B.
- Same destination, swapping agents, shared-box pushes, player/box collisions
  and different boxes sharing a destination use that priority.
- A loser uses the first compatible alternative in its fixed ranked list.
- Forced standstill remains possible. When a stationary loser blocks the
  winner, the winner tries another compatible move. This preserves the existing
  engine behavior; the earlier proposed "both always stay" default was not
  silently adopted. Both stay when neither has a compatible alternative.
- Blocked rounds consume time. Following into a successfully vacated player
  cell is allowed; swaps are conflicts.
- Removing a box strips credit; goal-to-goal pushes credit the successful
  pusher. Parser-created boxes start uncredited. Filling goals does not end play
  before the limit.
- WAIT remains an internal/legacy sentinel. Humans cannot select it or submit
  an individually illegal direction. Boxed-in humans automatically submit the
  forced sentinel so the UI cannot wait forever for an impossible input.
- Movement and reverse-push preprocessing now respect board dimensions even
  when border walls are absent. Previously an open-border board could cause
  unbounded preprocessing.

## Validation and reproducibility

Tests cover engine conflicts, ownership, forced blockage, legal fallback lists,
board/time cache identity, zero-sum evaluation, exact terminal scores, safe and
contested scoring, legal sacrifices, repeated-position tie-breaking, completed
iteration atomicity, and GC restoration. An independent unpruned joint-matrix
reference verifies root values across both player perspectives and both parities,
including terminal horizons, reversed traversal and previously populated bounds.
Three GUI input tests run without opening a window when Pygame is installed.

Tests tied to removed buckets, descendant peaks, forced-event extensions and
hard strategic gates were replaced with tests of the new observable contract.
Engine and state-identity regressions were retained. No engine invariant was
relaxed to accommodate the rewrite.

Commands:

```powershell
python -B -m unittest discover -s tests
py -3.11 -B -m unittest discover -s tests
python debug2.py
python -B -m src.experiments.competitive_benchmark --budget 1 --rounds 50
```

For baseline comparisons, pass `--baseline-dir` pointing to the snapshot's
`competitive` directory. The harness loads those files separately and repairs
only the undefined hook in memory, delegating it to the old documented one-sided
fallback method. It does not compare against a silently rewritten baseline.
The shared live engine and round limit are identical for both agents; each
agent receives only the round's starting state. Seat-swapped matches cover both
priority roles, but are not claimed to make asymmetric maps symmetric.

`LAST_SEARCH` and each agent's `last_search` expose completed horizons, each
iteration's root values, submitted alternatives, conflict flags, executed
actions, and immediate evaluation components. The benchmark writes JSON traces
under `results/`. The older baseline's reported path depth is not comparable to
the rewrite's completed adversarial horizon.

## Remaining approximations

The evaluator estimates only the next opportunity, not a full multi-box plan.
Walking guidance does not treat the moving opponent as a permanent wall, and
later delivery steps relax other-box blockage. Finite horizons can still favor
delaying a threat past the leaf. Pure-action maximin may be overly defensive
against a weaker opponent. No universal optimality or strength claim follows
from this small test/match collection. OS scheduling can still affect wall time;
the measurements below report actual overruns rather than hiding them.


## Measured results

Final correctness run: **71 tests passed on Python 3.11.9**, including the three
Pygame input tests. Python 3.14 also passed the 68 non-GUI tests; its three GUI
tests were skipped because Pygame is installed only in Python 3.11. A headless
GUI input smoke test also passed there. No interactive visual playtest was run.

The following decision/match measurements use Python 3.14 on this machine,
with a one-second allowance per decision. Preprocessing is excluded; search,
submission ranking and return overhead are included. Results vary with CPU
scheduling and cache state. Baseline means the working-tree snapshot with only
the undefined preference hook repaired in memory.

Starting states, 50-round limit. Depth is a fully completed tactical horizon;
baseline depth 0 means no tactical window completed (only its one-step robust
seed controlled the decision), not that it performed no computation.

| Map | Baseline action A / B | Rewrite action A / B | Baseline horizon A / B | Rewrite horizon A / B | Rewrite time A / B (ms) |
| --- | --- | --- | --- | --- | --- |
| arena_open | SOUTH / SOUTH | SOUTH / SOUTH | 0 / 0 | 11 / 10 | 966.6 / 967.7 |
| capacity_lab | SOUTH / SOUTH | SOUTH / SOUTH | 0 / 0 | 8 / 8 | 964.6 / 964.3 |
| dense_goals | EAST / WEST | SOUTH / WEST | 0 / 0 | 7 / 7 | 964.1 / 963.2 |
| corridors | EAST / WEST | SOUTH / SOUTH | 0 / 0 | 10 / 10 | 965.2 / 967.0 |
| test_race | WEST / SOUTH | WEST / SOUTH | 3 / 3 | 19 / 17 | 968.4 / 970.3 |

Detailed root values, evaluation components, preferences and resolved actions:
[`competitive_comparison.json`](../results/competitive_comparison.json).
The changed corridor starting actions have no full-budget match validation yet;
no claim of proven improvement is made for those individual actions.

Full-budget matches, 30 rounds, with both seat assignments:

| Map | Rewrite seat | Final A:B score | Rewrite outcome |
| --- | --- | --- | --- |
| capacity_lab | A | 2:0 | Win |
| capacity_lab | B | 0:2 | Win |
| dense_goals | A | 2:0 | Win |
| dense_goals | B | 0:1 | Win |

Across these **120 rewrite decisions**, median wall time was
**965.9 ms**, maximum **992.4 ms**,
and **0 exceeded one second**. The four wins are encouraging,
but this is a small deterministic sample, not a statistical strength guarantee.
The repaired baseline exceeded its one-second allowance 10 times in this run.
Full replay traces: [`competitive_matches_1s_final.json`](../results/competitive_matches_1s_final.json).

Earlier measurements are retained rather than hidden:

- At 0.1 seconds across five maps and both seats: 4 wins, 5 draws, 1 loss.
  The loss was capacity_lab as A (1:2). The final one-second rerun of that matchup
  won 2:0; short-budget strength is not established by the full-budget result.
  The old planner also reserves a fixed 50 ms, so this is not an equal usable
  search-time comparison at such a small allowance.
  [`competitive_matches.json`](../results/competitive_matches.json)
- Before the GC fix, the one-second matches were 3 wins and 1 draw, with 9/120
  rewrite decisions over budget and a maximum of 1285.6 ms. Those results
  exposed the pause problem and are not the final timing claim.
  [`competitive_matches_1s.json`](../results/competitive_matches_1s.json)

No known failing correctness test remains. The short-budget loss, unvalidated
corridor action changes, finite-horizon behavior and limited opponent sample
remain explicit limitations.

An additional Python 3.11 check on the same ten starting-position decisions
completed 7–21 rounds, with maximum elapsed time 970.5 ms
and 0 deadline overruns. See
[`competitive_python311.json`](../results/competitive_python311.json).

> **ARCHIVED** — This document describes an *older* architecture and is kept
> for history only; it is no longer accurate. Current docs:
> `docs/competitive_rewrite.md` (competitive AI), `docs/project_analysis.md`
> (project overview). Root scripts referenced below now live in `tools/`,
> and the `src/` layout was later reorganized (`src/core` → `src/single/core`,
> `src/gui` → `src/shared` / `src/single/gui` — see `docs/project_analysis.md`).

---

# AI Search Improvement Plan

## Objective
Improve the competitive Sokoban agent from a heuristic-driven beam search into a stronger, more objective-driven planner that still respects the game’s tactical rules and conflict semantics.

This plan keeps the proven tactical constraints already implemented, while re-centering the planner around projected score and deeper search quality instead of raw heuristic dominance.

---

## 1. Current diagnosis

The current AI is functional but still structurally limited.

### What works
- Tactical rules are now enforced consistently:
  - no own finished-box pushes
  - no normal WAIT actions in play
  - parity conflict winner/loser resolution
  - deterministic yielding to a legal alternative move
  - immediate steal logic
  - concrete tactical target selection
  - iterative deepening wrapper around the search
- The test suite passes and the conflict logic is stable.

### What is still weak
- The planner is still optimizing a heuristic score, not a projected end-game objective.
- The search is mostly ranking states rather than planning complete winning sequences.
- Tactical constraints are acting like patches rather than the main objective.
- Heuristic weights can shift behavior unpredictably across maps.
- A deeper search can still be misled by a locally attractive branch.
- The root action can be chosen from the wrong branch because a misleading deeper descendant is favored too early.

### Core problem statement
The AI should not ask: what does the heuristic say this state is worth?
It should ask: what eventual score can this move realistically lead to, given opponent responses, box ownership, and conflict rules?

---

## 2. Design principles

1. Tactical safety remains mandatory.
   - never push your own finished box away
   - never stand still in normal play
   - respect conflict priority rules
   - prefer legal alternative moves when forced to yield

2. Search quality must improve before reward tuning.
   - deeper search should reduce heuristic brittleness
   - iterative deepening should be the mechanism, not a blind depth increase

3. Objective should be projected final score, not heuristic dominance.
   - search should prefer states that lead to more eventual points
   - terminal-value estimates should dominate local heuristic noise

4. Search results must be robust under time pressure.
   - use the last completed depth if the next deeper pass times out
   - never overwrite a valid result with a partial deeper iteration

5. A deeper search must stay aligned with tactical priorities.
   - parity conflicts still matter
   - immediate steals still matter
   - concrete push progress still matters

---

## 3. Planned implementation phases

### Phase 1: tighten the objective function

#### Goal
Make the planner optimize projected score more directly instead of pure heuristic state value.

#### Planned changes
- Keep tactical rules as hard constraints.
- Introduce a projected terminal-value estimator for each search node.
- Define a score estimate as:
  - current credited boxes
  - potential additional future boxes
  - expected steal opportunities
  - expected defensive value
  - expected penalty if an opponent can immediately steal a box
- Distinguish between:
  - immediate tactical value
  - future strategic value
  - terminal value estimate

#### Design rule
The scoring function should prioritize:
1. score gain from credited boxes
2. future box completion opportunities
3. steal opportunities and opponent scoring denial
4. positional safety and defense
5. heuristic tie-breaks only after the above

#### Output
A clearer objective function that is less dependent on ad hoc weights.

---

### Phase 2: make the search depth-aware and time-safe

#### Goal
Replace the crude fixed-depth logic with a disciplined iterative-deepening beam search.

#### Planned changes
- Keep the current beam search structure.
- Search in increasing depth limits: 1, 2, 3, ... up to a cap.
- Keep the best result from the last fully completed depth.
- If time expires mid-depth, use the previous completed depth result as fallback.
- Do not let a partial deeper search overwrite a valid shallow result.

#### Important safeguards
- maintain a per-depth best root action
- only promote a new root action when the deeper iteration fully completes
- keep tactical constraints at all depths
- keep anti-loop filtering, but do not reject necessary tactical revisits that produce progress

#### Output
A search that becomes more informed with time, without destabilizing proven tactical behavior.

---

### Phase 3: clean up root-action selection

#### Goal
Fix the recurring issue where a deeper but misleading descendant beats a correct immediate tactical move.

#### Planned changes (implemented)
- separate immediate root evaluation from deeper descendant evaluation
- root action ranking, in order:
  1. best descendant value found by the search (quantized by `ROBUST_MARGIN`)
  2. immediate robust value after the opponent's worst-case reply (quantized)
  3. the child's strike distance to the boxes still in play - equal lines
     step toward the fight instead of repeating one tuple-biased direction
  4. exact descendant value, exact robust value, fixed action order
- one open list per root; a root trailing the busiest root by more than
  `FAIR_LAG` expansions is served anyway, so a line that must cross a value
  valley (approach, absorb the worst-case reply, cash in deeper) is still
  developed instead of starving behind shallower nodes
- root-only shaping penalties: `REVISIT_PENALTY` (no shuffling back onto a
  just-visited cell) and `AWAY_PENALTY` (no opening that walks away from
  every box still in play - all evaluation terms are relative, so two agents
  drifting in parallel look perfectly neutral)

#### Rule
The original robust-first ranking was replaced after it caused permanent
passivity: every attack line dips under the opponent's worst-case reply, so
"do nothing" always won the robust bucket and agents defended to a draw
(47-step center dances, 0-0 games where nobody even pushed). Quantized
robust value still decides whenever descendant values agree, so a clearly
superior immediate move never loses to noise - but a descendant line that
survives the reply and converts deeper may now win. That is the difference
between "safe" and "trying to win".

#### Output
Agents approach, steal, strip and re-score instead of oscillating: across
the five benchmark maps, cell repeats dropped (39->6, 32->0, 29->1), boxes
scored rose (dense_goals 1/5 -> 3/5) and games show lead changes instead of
frozen 1-1 standoffs.

---

### Phase 4: improve opponent modeling

#### Goal
Make the opponent response model less brittle and less over-pessimistic.

#### Planned changes
- keep worst-case response as a safety mechanism for now
- but add a second evaluation mode for likely opponent responses
- compare:
  - worst-case response value
  - likely-opponent-response value
  - tie-break using tactical priority rules
- this helps avoid an overly pessimistic branch that never takes a fight

#### Why this matters
The current evaluation treats the opponent as a single blunt response model. A more nuanced model makes the agent better at:
- intercepting a box before the opponent finishes it
- closing a scoring race
- not overreacting to harmless blocked paths

#### Output
Better strategic anticipation without exploding search cost.

---

### Phase 5: reduce noisy heuristic overlap

#### Goal
Break the heuristic into clearer, non-overlapping terms.

#### Planned changes
- split the heuristic into:
  - immediate score gain
  - future score potential
  - connection to a tactical target
  - steal pressure
  - defense pressure
  - mobility and congestion value
  - deadlock penalty
- normalize each term by horizon and map size
- remove duplicated scoring effects that are counted in more than one term
- ensure each term measures a distinct strategic effect

#### Output
A cleaner heuristic that is easier to reason about and less brittle under map changes.

---

### Phase 6: structured audit — identity, aggregation, window, timing (completed)

#### Goal
Verify that state identity, caches, opponent search, evaluation and root
selection actually represent the objective, and fix the two reported
regressions: agent B never finishing its easy box on `capacity_lab`, and
every decision burning the full ~1 s even when the agents are far apart.

#### Findings and fixes (classified)

**State identity / caches**
- Search keys did not distinguish same-position states from different
  rounds: `_key` now includes `state.step`.
- Eval and transition caches collided across boards/rounds: keys now
  include `Board.serial` (eval) and `serial + max_steps` (transition).

**Root aggregation**
- A state reached under a second root was skipped silently when already
  in the closed set, so expansion order decided which root saw shared
  states. The credit to `deep[root]` now runs *before* the closed check
  (`agent_a.py` expansion loop). Regression:
  `tests/test_search_identity.py::test_cross_root_transposition_credits_both_roots`
  fails on the pre-fix order (`deep[WEST] = 572 < stored 596`) and pins
  the strict premises (seeds 559, alternatives 552/572, all < 596).

**Opponent coverage / justification gates**
- `denial_justified` now gates denied pushes in `_forbidden` and
  `_pick_diversion` (Kuhn augmenting-path matching in `_match_costs`,
  independent per-box min costs in `_race`), so a push is only forbidden
  when the opponent really wins the race for the loose boxes.
- `reposition_justified` was refuted and removed (no rule-valid position
  ever justified standing still).

**Evaluation inversions**
- Winner-restricted `_advantage` with `W_ADV` (was: raw cost difference
  across both agents), exact walk+push `_away_penalty` basis (was:
  strike basis that moved the target set along with the push — the
  capacity wrong-push scored *better* than the correct approach),
  `TACTICAL_MARGIN` / `ROBUST_MARGIN` bucketing so a tactical window win
  cannot be overridden by optimistic descendant peaks, and
  `_rank_best` refactor.
- Greedy-matching artifact: partial greedy assignment produced `INF`
  distances that flipped race/denial decisions; replaced by maximum-
  cardinality Kuhn matching.

**Tactical window**
- Two complete simultaneous rounds over all rule-permitted pairs, deep
  third round only when `WINDOW_DEEP_GATE` (0.15) headroom and
  `WINDOW_BUDGET` (0.35) of the decision remain; all-or-nothing on
  timeout with a robust one-ply fallback (never a stale window).
  Documented as pure-action maximin (not mixed strategies).

**Discounting**
- `DISCOUNT_ALPHA = 0.0`: undiscounted baseline first; speculative
  points are only ever scaled by remaining horizon, never re-counted.

**Time budget / early exit**
- All deadlines on `time.monotonic()`.
- Early exit originally keyed on `EXIT_STABLE = 4000` pops — unreachable
  (≈3500 pops per 1 s budget; ~1.5–2 pops/ms) — so agents always ran the
  full budget. Fixed: `EXIT_STABLE = 300` pops **and**
  `EXIT_STABLE_MS = 0.15` s of wall-clock silence (both required),
  compared against `_rank_structure` (winner + bucket key) instead of the
  full creeping `-deep` value, above `EXIT_MIN_NODES` / `EXIT_MIN_DEPTH`
  floors. Quiet capacity positions now exit in 188–719 ms at a 1.0 s
  budget (avg ≈ 570 ms, was ≈ 950 ms).

#### Output
63 tests green (41 original + 22 in `tests/test_search_identity.py`), the
capacity trace completes with agent B scoring its easy box (final A=1
B=2, all 3 boxes), and both reported regressions are fixed with
discriminating regression tests for identity, window, aggregation,
penalties, matching/race, push gates and the early-exit bounds.

### Phase 7: play quality — "they don't even try for the point" (completed)

#### Symptom
Playtesting after the audit still showed agents refusing obvious points,
circling instead of cashing, and games ending with few boxes scored
(e.g. `capacity_lab` ending A=0 B=3 with A shut out). Action filters were
exempt: at every flagged moment every push was legal (`_forbidden=False`,
nothing filtered).

#### Diagnosis (instrumented games: `record_game.py` flags every root where
#### a SCORE/PROD push exists but is not chosen; `trace_flag2.py` decomposes
#### the exact states and walks the window's min-max leaves)

The blocker was the **evaluation of holding a credited box under attack**,
not the search or the gates:

- `_steal_potential` charged a race-won threat as `1.0 + 0.5*gradient`
  per credited box × `W_STEAL = 1200` → **1200–1800 per box versus the
  1000 the box is worth**. Crediting a box *added* a steal target, so the
  second point could evaluate as a net loss.
- Measured on a real mid-game flag: after A pushed `(7,8)→(8,8)` = goal,
  eval(A) was **−806** for a 2:1 lead (`locked +1000, steal_b +2094`), and
  the 3-round window saw **−818 for the scoring push versus −36 for
  walking in circles** — 9 tactical buckets apart, so the walk won
  decisively and the agent circled.
- The steal preview also double-counts: it charges the future strip while
  the `locked` term still holds the same cred; the strip is charged again
  later when it actually happens. Zero-sum and label symmetry were intact
  — the inversion was economic: a threatened cred evaluated negative.
- Secondary finding (kept, not a bug): advancing a loose box toward its
  goal can reduce *your own* projected race margin because delivery costs
  are opponent-relative (`(7,7)→(8,7)` cut A's cost 4→2 but B's 6→2, so
  the won race collapsed to a tie). Refusing that push while the opponent
  hovers is defensible; the flag remains only as a one-bucket close call.

#### Fix
Bound the race-won threat per box: `STEAL_WIN_BASE = 0.35` +
`STEAL_WIN_GRAD = 0.15` (max 0.5 × `W_STEAL` = 600 < `W_LOCKED` = 1000,
`else` gradient 0.3 → 0.2) so a credited box always keeps at least +400
while the *completed* strip is still charged exactly once by the locked
term. The winner-restriction jump (≥300 vs ≤240) is preserved.

#### Result
Same flag state: after-score eval **+385** (was −806), window **+388 for
the score versus +373 for the walk** (was −818 vs −36); recorded capacity
game now ends **A=1 B=1** with 5/6 scoring roots taken (was A=0 B=3).
63 tests stay green.

---

### Phase 8: early-exit soundness — a bucket tie keeps thinking (completed)

#### Symptom
The Phase 6 early exit (structure stability + floors) is what bought the
latency fix (p50 951 ms → 15–525 ms, misses ~200 → 11), but the
`--no-exit` ablation scored **19 boxes versus the exit run's 10** with
identical evaluation: the exit was freezing tie-breaks that further
search would refine. The visible class is the one-bucket miss — a
scoring push that leads on the window's *exact* tactical value loses a
bucket tie to a walk whose optimistic deep value happens to be higher
(step-14 PROD; capacity step-15 in `record_game.py`).

#### Diagnosis

- `_rank_structure` stability proves the ranking *stopped* changing, not
  that it *cannot*: a rival mid-valley-crossing is undersampled (GBFS
  serves it only via `FAIR_LAG`), so "300 pops with no verdict change"
  can be an artifact of expansion order — one terminal descendant can
  jump its deep bucket past the winner after the exit fires.
- What IS provable: after the window completes, tactical values,
  penalties, strikes and robust values are frozen (all set before the
  loop); only deep climbs, monotonically. So the winner is final when it
  is strictly ahead on the frozen tactical bucket, or when every
  bucket-tied rival has an empty frontier (its deep value can no longer
  move, and the winner already leads the frozen remainder of the key).
- `tie_probe.py`: **~50 % of all moves are bucket-tied** (45–54 % per
  map), so a rule that simply refuses to exit on a tie would hand the
  full deadline back to half the moves and undo the latency fix. The
  uncertifiable case therefore needs a *bounded* gate, not a veto.

#### Fix
`_exit_decided(heaps, tie_gate)` gates the exit on top of the existing
stability check (`agent_a.py`):

1. bucket-separated on the frozen tactical key → decided (lexicographically
   dominant `key[0]` cannot be overtaken by any later deep refinement);
2. every bucket-tied rival drained → decided (frozen keys, winner leads);
3. live tie → decided only after `EXIT_TIE_FRACTION = 0.60` of the
   budget (`tie_gate = t0 + 0.60 × budget`): even a fully tied move now
   stops at 60 % of the budget instead of either ~25 % (old exit) or
   100 % (deadline).

The ranking itself is untouched; `_rank_best` was factored into
`_rank_keys()` (identical order — stable sort over the same key — pinned
by the existing ranking tests) so the check can see the runner-up's
bucket.

#### Result

- **68 tests green** (5 new: live tie blocks before the gate, gate
  releases after it, drained rival is decided, bucket separation needs no
  gate, single root is trivially decided).
- `record_game.py` A/B on `dense_goals` (same code, `GATE=0.0` = exact
  pre-change behavior vs `0.6`): **flags 40 → 13, scoring roots taken
  2 → 4, final A=0 B=0 → A=1 B=0**. `capacity_lab`: final **A=2 B=0**
  with 6/8 scoring roots taken, flags=2 — both the documented one-bucket
  class where the deep term trails by ≥ 2 buckets (no amount of thinking
  within budget flips those; kept as a limitation).
- Benchmarks (`bench_tiegate.txt`, final shipped code): p50 **17–572 ms**
  (≈290 ms median), p95 ≤ 1296, misses **44 / 1200 decisions** (3.7 %;
  baseline 222 = 18.5 %, pre-gate 11 = 0.9 %), boxes scored **10 → 14**
  (no-exit ceiling 19) — gains on the contested maps (dense 2 → 4,
  corridors 2 → 4), capacity still cashes all 3 boxes (3/3). The gate is
  the measured middle of the quality/latency curve; `EXIT_TIE_FRACTION`
  is the knob.

### Phase 9: root ranking fidelity — the window's exact verdict beats optimistic descendants (completed)

#### Symptom

Probe v4/v5 (root-table dumps on dense/corridors/capacity): among
bucket-tied moves the exact window margin favored the approach or push
in 21 sampled cases and deep/strike overrode it in **all 21** — the
Phase 6 key consulted the optimistic descendant *before* the window's
own exact value. The second visible class: dense_goals played a frozen
ping-pong — both agents stepped (4,3)↔(5,3) / (10,3)↔(9,3) every round
for 40+ steps with a scoring push on the table every cycle
(`record_game`: **46 flags, A never scored**).

#### Diagnosis

- **Ranking**: `_rank_keys` was `[tact_bucket, deep_bucket, strike,
  exact_tact, ...]` — within a tied tactical bucket the GBFS deep value
  (monotone, expansion-order dependent, the audit's own "optimistic
  peak") settled the verdict *before* the window's exact margin, so a
  decisive window separation (e.g. −420 vs −574) could be discarded for
  a deep peak. The early exit certified stability on the then-first three
  components, so any key change must move the frozen prefix with it.
- **The dense ping-pong is horizon-gaming, not eval noise.** Exact
  replication of the gated 3-round window at the anchor state: R2 ranks
  SCORE above the walk (−420 vs −574); R3 flips it (−574 vs +416).
  Each extra round flips the advance verdict by ≈ ±1000 — one box's
  credit entering or leaving the horizon: from (4,3) on odd (A-parity)
  steps the line ends at A's high-water mark (seizes the (7,3) approach
  cell, *censored before B's counter*) = +416; from (5,3) on even steps
  it includes B's counter (walk into the vacated approach cell after
  A's cash: the cash leaf computes −1346 — A's +1000 is swamped by
  losing every remaining race by one cell) = −570. Advance when the
  window censors B's reply, retreat when it doesn't, forever.
  The return half of that loop *is* a delivery-cost-lowering return.

#### Fix

1. `_rank_keys` → `[tact_bucket, exact_tact, strike, deep_bucket,
   exact_deep, exact_robust, order]`: the window's exact margin becomes
   the second key and decides every bucket tie it can see; deep/strike
   only refine exact window *ties* (a transient deep peak can no longer
   override a decisive window margin). Bucket precedence (tactical
   bucket first) is unchanged.
2. `_exit_decided`'s frozen prefix generalised from `key[0]` to the
   frozen prefix `(tact_bucket, exact_tact, strike)` — the three
   components settled once the window completes — so the exit certifies
   exactly what the sort can still see; tie gate and drained-rival
   logic untouched.
3. Revisit pricing kept **flat**: an attempted productive-return
   exemption (free when the return lowers walk+push delivery cost) was
   A/B-tested in `record_game` (`PRODRET=0/1`) and **removed**. Dense:
   exemption → 46 flags and A frozen; flat → **11 flags, A cashes**
   (the flat 250 makes the return cost more than the advance, breaking
   the parity cycle). Capacity: flat cashes **3/3 — better than the
   exemption's run** (2/3), flags 5 vs 2 (flags are a nudge, boxes are
   the objective). The exemption's original motive (no-WAIT forces an
   agent off its push cell) stays a documented trade-off, not a rule.

#### Result

- **72 tests green** (4 new: exact window margin beats a deep peak
  within the bucket; strike breaks exact window ties before deep; exact
  window separation decides before the tie gate; revisit penalty flat
  even when the return lowers delivery cost — with the A/B numbers in
  the comment).
- `record_game` final: dense **flags 46 → 11, scoring roots taken 3 → 4,
  A frozen → A=1 cashes**; capacity **3/3 boxes cashed** (best of any
  run), flags 5; arena 4/4 in the bench.
- Bench `reorder_flat` (5 maps × 2 assignments, 1200 decisions): p50
  **89–314 ms**, p95 ≤ 953, misses **3/1200 = 0.25 %** (baseline 222 =
  18.5 %, tie gate 44 = 3.7 %), boxes **17/28** (baseline 11, tie gate
  14, Phase 9 exemption attempt 12): arena 4/4, dense 5/10 (mirrored
  4/5 with a 3:1 win), capacity 3/6, corridors 4/6, race 1/2.
- Label symmetry re-verified (`test_evaluate_is_label_symmetric`,
  `mirror_test` 3 maps), AI beats random in both seats (arena 2-0,
  dense 2-0 / 3-0), `ui_smoke` OK (8 screenshots).

#### Known limitations

- **dense orig still cashes 1/5** in the bench game (mirrored 4/5):
  the window-content class — the eval ranks contest promises over cash
  on that map at the anchor state; no weight retuning was done
  (`W_LOCKED`/`W_PROJECTED` untouched by design; the R2/R3 decomposition
  is recorded here instead).
- `wander5` TRUE_WANDER counts (59 pre → 79 post) are confounded: the
  games diverge, the sampling is not normalized, and the raw-tie
  sub-class (window exact tie, flat pen breaks toward a walk) is the
  documented cost of the flat charge whose benefit is the cycle break —
  record metrics (flags/cashes) are the primary evidence.
- 3 deadline misses at 1042–1098 ms (deep-window tail on dense/race);
  ≥2-bucket separations remain unflippable in budget; `STEAL_RANGE=40`
  else-gradient ≈ constant; no claim of depth-25 coverage or an
  exhausted frontier.

---

## 4. Implementation checklist

### Search structure
- [x] Keep the current GBFS and beam search as the base engine
- [x] Add iterative deepening around the search loop
- [x] Keep the deepest fully completed result
- [x] Fall back to the last valid completed depth when the time limit is reached
- [x] Prevent partial deeper searches from overwriting a valid shallower result

### Tactical constraints
- [x] keep own finished-box protection
- [x] keep no-WAIT rule for normal play
- [x] keep parity conflict priority
- [x] keep deterministic yield selection
- [x] keep immediate-steal priority
- [x] keep tactical progress as an explicit objective filter

### Objective and evaluation
- [x] add projected score evaluation by horizon
- [ ] separate tactical value from future strategic value
- [x] add terminal-score estimate for deeper evaluation nodes
- [ ] normalize heuristic terms to reduce map dependence

### Opponent modeling
- [ ] compare worst-case and likely-response values
- [ ] keep tactical safeguards in place during opponent modeling
- [ ] add stress tests for intercept vs. defend decisions

### Regression coverage
- [ ] iterative-deepening fallback after timeout
- [x] deeper search not overwriting valid shallower results
- [x] immediate parity-winning conflict actions
- [x] yielding alternatives excluding occupied destinations
- [x] no WAIT during normal play
- [x] own finished-box protection
- [x] immediate steals
- [x] finish-box-then-return-defense missions
- [x] state identity across rounds (`state.step` in search keys, serial-keyed caches)
- [x] odd/even parity priority and conflict fallback
- [x] transient descendant peak cannot override the window
- [x] delayed threat / window-timeout fallback to robust values
- [x] cross-root transposition credits every root (discriminating test)
- [x] root penalties: scoring push free, approach free, retreat/wrong-push charged
- [x] race + matching costs (cardinality, independent per-box costs)
- [x] denial push gates (own-cred and remaining-loose rules)
- [x] early-exit latency bounds (quiet exit fast, hard cap on dense boards)
- [x] exhaustive pure-action maximin reference for the tactical window
- [x] exact window margin beats an optimistic deep peak inside a bucket
- [x] exact window separation decides before the tie gate (frozen prefix)
- [x] strike breaks exact window ties before the deep bucket
- [x] revisit penalty stays flat even when the return lowers delivery cost
      (dense ping-pong A/B: exemption 46 flags/0 vs flat 11 flags/cash)

---

## 5. Expected outcome
The agent should become more strategic and less heuristic-driven, while still respecting the game rules and the tactical constraints that have already been validated.

The target end state is:
- better move selection in competitive situations
- fewer stupid local choices
- stronger box completion and denial behavior
- deeper but safer planning under time limits
- search that improves with time instead of simply reacting to a fixed heuristic snapshot

---

## 6. Recommended order of execution
1. project score estimator
2. iterative deepening wrapper safety rules
3. root-action ranking fix
4. opponent model refinement
5. heuristic cleanup and normalization
6. repeated cross-map benchmark replay
7. final regression and performance pass

This should be implemented in small, test-backed changes so each improvement is validated before moving on to the next one.

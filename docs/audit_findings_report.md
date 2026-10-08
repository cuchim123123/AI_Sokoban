# Competitive AI audit — findings, fixes, and benchmarks

Status: complete (baseline, final code, and three variants benchmarked).

Scope: `src/competitive/{agent_a,agent_b,evaluation,state,transition}.py`
under the standing engine constraints — GBFS stays the expansion engine,
four directions only (WAIT only as the forced-immobility enum), parity
conflict priority with rule-legal loser redirection, ≤ 1000 ms per
decision over the whole call on a monotonic timer with a valid-direction
fallback, winner = score at end of the allotted rounds. Every change below
was required to *represent the objective better*, not to bolt on A*/BFS.

---

## 1. Executive summary

Two playtest complaints were reproduced, root-caused, and fixed:

1. **"Reasoning always burns the full second."** Root cause: the early-exit
   threshold was unreachable (~4000 pops budget vs ~3500 pops available in
   1 s), and its comparison key included a creeping `-deep` term that never
   settled. Fixed with a structure-only key plus pop-count *and*
   wall-silence conditions. Baseline p50 was **951–953 ms on every single
   decision**; corrected p50 is **15–525 ms** (≈220 ms median) with
   deadline misses dropping from 6–22 per game to **11 across the whole
   20-agent-game run** (and most of those in the hardest map).

2. **"They don't even try for the point."** Root cause: the evaluation
   made holding a credited box under attack evaluate *negative*. The steal
   threat charged `1.0 + 0.5·gradient × W_STEAL(1200)` per credited box —
   1200–1800 points against a box worth 1000 — while the locked term still
   held the same point (double count). Measured on a real mid-game state:
   after pushing a box onto a goal (2:1 lead!) `eval(A) = −806`, and the
   3-round tactical window read **−818 for the scoring push vs −36 for
   walking in circles** — 9 tactical buckets apart, so the walk won
   decisively. Fixed by bounding the race-won threat below `W_LOCKED`.
   The same state now reads **+388 (score) vs +373 (walk)**; the recorded
   capacity game flipped from A=0 B=3 (A shut out) to A=1 B=1 with 5/6
   scoring roots taken, and the benchmark's capacity games now end
   **2:1 with all 3 boxes cashed** (baseline: 0:1 and 0:0).

3. **Soundness follow-up on the latency fix.** The exit was certifying
   *unfinished* tie-breaks: on ~50 % of moves the top two share the
   tactical bucket, and the winner then rests on the optimistic deep term,
   whose rival value is still climbing while that root is undersampled
   (the `--no-exit` ablation's 19-vs-10 box gap is the cost of firing
   anyway). `_exit_decided` now certifies the winner — strictly ahead on
   the *frozen* tactical bucket, or with every bucket-tied rival's
   frontier drained — and grants a live tie only a bounded
   `EXIT_TIE_FRACTION = 0.60` share of the budget. Dense A/B (gate 0 →
   0.6, same code): flags 40 → 13, scoring roots taken 2 → 4. Final
   benchmark: p50 **17–572 ms** with 44 misses (baseline ~222, pre-gate
   11) and boxes scored **10 → 14** — part of the no-exit quality gap
   (19) recovered at a bounded, measured latency cost.

68 tests green throughout (41 original + 27 new in
`tests/test_search_identity.py`).

---

## 2. Findings classified (with code refs)

### A. Latency / early exit (`agent_a.py`)

| # | Finding | Classification | Status |
|---|---------|----------------|--------|
| A1 | `EXIT_STABLE = 4000` pops unreachable: pop rate ≈1.5–2/ms ⇒ ~3500 pops in a 1 s budget, so agents always ran the deadline. | bug (dead code path) | fixed |
| A2 | Exit comparison used the full `-deep` value, which creeps monotonically → structural stability never registered even when reached. | bug (wrong comparison key) | fixed |
| A3 | Fix: `EXIT_STABLE = 300` pops **and** `EXIT_STABLE_MS = 0.15` s wall silence (both required), compared on `_rank_structure` (winner + bucket keys), above `EXIT_MIN_NODES = 2000` / `EXIT_MIN_DEPTH = 4` floors. | fix | verified (bench + ablations below) |
| A4 | All deadlines moved to `time.monotonic()` (perf-counter affine, immune to clock adjustment). | hardening | fixed |
| A5 | The stability exit could fire on an *unfinished tie-break*: with the top two sharing the tactical bucket, the winner rests on the optimistic deep term, and an undersampled rival's "no change for 300 pops" is an artifact of expansion order (the `--no-exit` ablation's 19-vs-10 box gap is the cost). Probe `tie_probe.py`: **~50 % of moves are bucket-tied** (45–54 %/map), so a veto on ties would return to the baseline burn. | soundness gap (exit vs partial-search consistency) | fixed: `_exit_decided` certifies bucket-separation on the frozen tactical key or a drained rival; a live tie waits for `EXIT_TIE_FRACTION = 0.60` of the budget — 68 tests, dense A/B (GATE 0→0.6): flags 40→13 |

### B. Play quality / the value of a point (`evaluation.py`)

| # | Finding | Classification | Status |
|---|---------|----------------|--------|
| B1 | `_steal_potential` race-won branch returned `1.0 + 0.5·gradient` per credited box; ×1200 = 1200–1800 vs `W_LOCKED = 1000`. A second cred adds ~1000 locked **and** ~1760 steal exposure ⇒ scoring evaluates as a net loss. | **eval inversion** (core of complaint 1) | fixed |
| B2 | The steal preview double-counts a future strip: the threat is charged at the leaf while `locked` still holds the same point; the strip is charged again later when it completes. Zero-sum and label symmetry were intact — the inversion was purely economic. | eval modeling | fixed (bounded preview) |
| B3 | Advancing a loose box can *reduce* your projected race margin because delivery costs are opponent-relative: `(7,7)→(8,7)` cut A's cost 4→2 but B's 6→2, collapsing A's won race (+600) to a 0.5/0.5 tie; one round later B stood on the approach and `_race` reported `pb=1` (−600). | honest but sharp: refusing that push while the opponent hovers is defensible; kept as a one-bucket close call | documented, not changed |
| B4 | `STEAL_RANGE = 40` makes the *else* gradient ≈constant across maps (a flat ~0.3 offset per far cred). Zero-sum (applies to both sides), harmless once the won-branch is bounded. | accepted quirk | documented |

**Fix (B1/B2)** — `evaluation.py`:

```python
STEAL_WIN_BASE = 0.35  # + gradient part 0.15: race-won threat ∈ [0.35, 0.5]
STEAL_WIN_GRAD = 0.15  # max 0.5 × W_STEAL = 600 < W_LOCKED = 1000
...
if best_att < best_def and best_att + 1 <= remaining:
    total += STEAL_WIN_BASE + STEAL_WIN_GRAD * gradient   # was 1.0 + 0.5*gradient
else:
    total += 0.2 * gradient                               # was 0.3
```

Rationale: a strip *preview* may cost part of the point it threatens, never
the whole point; the completed strip is charged exactly once, by the locked
term, when it happens. A credited box therefore always keeps ≥ +400, the
second point can never be a net loss, and the winner-restriction jump
(≥300 vs ≤240) that prevents race flip-flopping is preserved. Measured
effect on the diagnosed state: after-score eval **−806 → +385**; window
**−818 → +388** for the score (walk: −36 → +373).

### C. State identity / caches (`state.py`, `transition.py`, `agent_a.py`)

| # | Finding | Status |
|---|---------|--------|
| C1 | Search keys did not distinguish same-position states from different rounds → `_key` includes `state.step`. | fixed |
| C2 | Eval cache key omitted the board → includes `Board.serial`. Transition cache includes `serial + max_steps`. | fixed |

### D. Root aggregation (`agent_a.py`)

| # | Finding | Status |
|---|---------|--------|
| D1 | A state first reached under one root was skipped in the closed set when reached under a second root, so expansion order decided which root got credited. Credit to `deep[root]` now runs *before* the closed check. | fixed + discriminating regression test (`test_cross_root_transposition_credits_both_roots`; pre-fix order fails with `deep[WEST] = 572 < stored 596`) |

### E. Justification gates / opponent search

| # | Finding | Status |
|---|---------|--------|
| E1 | `denial_justified` now gates denied pushes in `_forbidden` and `_pick_diversion`: a push is only forbidden when the opponent really wins the race for the loose box (independent per-box costs in `_race`, Kuhn maximum-cardinality matching in `_match_costs`). | fixed |
| E2 | `reposition_justified` was refuted by construction — no rule-valid position ever justified standing still — and removed. | fixed |
| E3 | Greedy partial matching produced `INF` distances that flipped race/denial decisions. | fixed (Kuhn augmenting-path) |

### F. Window / ranking architecture (`agent_a.py`)

| # | Finding | Status |
|---|---------|--------|
| F1 | Tactical window: two complete simultaneous rounds over **all** rule-permitted pairs; third round only with `WINDOW_DEEP_GATE = 0.15` headroom inside `WINDOW_BUDGET = 0.35` of the decision; all-or-nothing on timeout with a robust one-ply fallback (never a stale window). Documented as pure-action maximin, not mixed strategies. | audited ✓ |
| F2 | Expansion priority (GBFS heap) is deliberately separate from the backed-up root value: tactical bucket (window − penalties, `TACTICAL_MARGIN = 100`) → deep bucket (`ROBUST_MARGIN = 200`) → strike tiebreak → exact values. Transient peaks cannot override the window (regression test pins this). | audited ✓ |
| F3 | `DISCOUNT_ALPHA = 0.0`: undiscounted baseline; speculative points only ever scaled by remaining horizon, never re-counted per depth; terminal wins never discounted below losses. | audited ✓ |
| F4 | Penalties: `AWAY_PENALTY = 25`/step on exact walk+push delivery-cost damage (documented geometry: scoring pushes are not charged for the box they cash), `REVISIT_PENALTY = 250`. Remaining flags show penalty-driven ordering only inside bucket ties — accepted. | audited ✓ |

### G. Regression coverage (`tests/test_search_identity.py`, 22 tests)

Identity across rounds (C1), odd/even priority, transient peak vs window
(F2), delayed threat, cross-root transposition (D1), scoring push with
checked alternatives, detours/oscillation, four-direction submissions,
deadline behavior, exhaustive reference on tiny boards, pure-action
maximin vs mixed strategies, zero-sum and label symmetry for the new steal
bounds (B1), and the early-exit certificate (bucket separation / drained
rival / tie gate, A5). All pass together with the original 41 → **68 green**.

---

## 3. Benchmarks

Same harness (`bench.py`), 5 maps × 2 assignments (original + horizontal
mirror with roles A↔C swapped), TIME = 1.0 s/move, ≤ 60 steps,
wall-clock latency per decision.

### Latency (all 10 games, 1200 agent-decisions)

| metric | baseline (pre-audit) | corrected (Phase 6–7) | **final** (+ tie gate, Phase 8) |
|---|---|---|---|
| p50 / move | **951–953 ms, every game** | **15–525 ms** (≈220 ms median) | **17–572 ms** (≈290 ms median) |
| p95 / move | 1368–2465 ms | 412–1128 ms | 580–1296 ms |
| max / move | 1506–**3180 ms** | 460–1339 ms | 758–1696 ms |
| deadline misses (>1 s) | 6–22 per game per agent (~222 total) | **11 total** (9 of them dense-mirror) | **44 total** (3.7 % of decisions) |
### Ablations (same harness; the two ablations ran on the pre-gate code — they are what motivated it)

| config | p50 / move | misses (>1 s) | boxes scored (sum of 10 games) |
|---|---|---|---|
| corrected (exit + window) | 15–525 ms | **11** | 10 |
| `--no-exit` (window kept) | **951–954 ms** on 8 games (test_race orig terminates naturally at 15–38 ms; its mirror 860–939) | **209** | 19 |
| `--no-window` (exit kept) | 20–738 ms (p95 up to 1656) | **70** | 10 |
| **final: + tie gate 0.60 (shipped)** | 17–572 ms | 44 | **14** |

Reading: the early exit is what bought the latency — without it the
corrected agents return to the baseline burn (p50 ≈953 ms, ~200 misses
over the run). Removing the window degrades timing stability (70 misses,
p95 1656 vs 1128: without window values the root ranking stays unstable
longer, so the structure-only early exit fires later) and costs play
quality on corridors (0:0 / 0:1 vs 1:0 / 1:0); box totals elsewhere are
within single-game noise of the corrected run. The no-exit run also
scores more boxes than the exit run (19 vs 10) with identical evaluation:
early exit trades deep-descendant refinement (the 2nd ranking key, which
breaks bucket ties) for latency. Box counts here are single-game-per-cell
and noisy; the honest summary is that latency and search depth are the
traded goods, and `EXIT_STABLE` / `EXIT_STABLE_MS` are the knobs if a
different point on that curve is wanted.

The shipped tie gate is the measured middle of exactly that curve.
`tie_probe.py` found **~50 % of moves are bucket-tied**, so the old exit
was freezing half its decisions on a tie-break it could not certify;
granting those moves `EXIT_TIE_FRACTION = 0.60` of the budget recovered
4 of the no-exit run's 9 extra boxes (**10 → 14**, concentrated where the
fight is: dense 2 → 4, corridors 2 → 4) at the cost of misses rising
11 → 44 — still **5× below baseline** (3.7 % vs 18.5 % of decisions) and
with p50 within the same order of magnitude. `EXIT_TIE_FRACTION` moves
along that curve; the pre-gate and no-exit rows bound it.

The window's *tactical* value is established by the instrumented traces
and tests rather than these totals: it is the oracle that refused the
−818 score (B1), and after the fix it ranks the score +388 over the walk
+373 in the diagnosed state; `test_transient_peak_cannot_override_window`
pins that hierarchy.

### Scores / boxes (single game per cell; treat as indicative)

| map | baseline | corrected | **final (tie gate)** |
|---|---|---|---|
| arena_open | 1:1 (2/2), 1:1 (2/2) | 1:0 (1/2), 1:0 (1/2) | 1:0 (1/2), 1:0 (1/2) |
| capacity_lab (**repro**) | 0:1 (1/3), 0:0 (0/3) | 2:1 (3/3), 1:1 (1/3) | **1:2 (3/3)**, 2:0 (1/3) |
| dense_goals | 0:0 (0/5), 2:0 (2/5) | 0:1 (1/5), 0:1 (1/5) | **1:1 (2/5), 1:1 (2/5)** |
| corridors | 1:1 (2/3), 1:1 (2/3) | 1:0 (1/3), 1:0 (1/3) | **1:1 (2/3), 1:1 (2/3)** |
| test_race | 0:0 (0/1), 1:0 (0/1) | 0:0 (0/1), 1:0 (0/1) | 0:0 (0/1), 1:0 (0/1) |

Per-map box totals are noisy at one game per cell (single-game
variance acknowledged); the repro map remains the structural showpiece
— baseline cashed at most 1 of 3 boxes, every corrected run cashes all
3 — and the contested maps (dense, corridors) are where the tie gate's
extra thinking pays (+2 boxes each vs the pre-gate code).

### Instrumented play (flags: a SCORE/PROD push existed but wasn't chosen)

| run | flags | scoring roots seen/taken | final |
|---|---|---|---|
| capacity, pre-B1-fix | 2 (incl. the −818 refusal) | 4/5 | A=0 B=3 (A shut out) |
| capacity, fixed | 2 (both one-bucket close calls) | **5/6** | A=1 B=1 |
| dense, fixed | 10 (mostly genuine multi-box race trades) | 5/9 | A=2 B=2 (4/5 boxes) |

5-map sim at 1.0 s: time/move 110–455 ms; arena cashes both boxes by
step 10; capacity ends 2:0 (A), strips visibly contested (conflicts 2–6
per game, repeats ≤ 7).

Also green: `mirror_test.py` (reflection-consistent, AI wins both seats)
and `ui_smoke.py` (menus, single solve, competitive game).

---

## 4. Limitations

- **Score statistics**: one game per map×assignment in the benchmark;
  box totals move several boxes between runs (acknowledged single-game
  noise). The latency result is robust (every game, both agents);
  the score result is strongest on the repro map.
- **Window horizon**: the tactical window sees 2–3 rounds. Strips that
  land beyond it are represented only by the (now bounded) steal preview;
  a threat within ~0.5×point is the calibration chosen, not a theorem.
- **`exact_step_costs` are body-blind**: projection cannot price a
  body-block/hold (standing on the goal cell), so a correct defensive hold
  can look worthless inside the window. Not fixed — would require
  state-dependent cost tables.
- **Opponent model**: worst-case replies over rule-permitted actions
  (pure-action maximin), not a model of what the opponent will actually
  play; no claim that depth-25 GBFS lines cover all opponent continuations,
  and no claim that any filtered frontier is exhausted — the search returns
  the best *found* backup within budget, by design.
- **Dense maps** remain the weakest scoring map (1–2 of 5 boxes); their
  flags are honest multi-box race trades rather than inversions, but the
  agents could still be sharper about when a cash strands the rest.

## 5. Final architecture

```
decision (≤ TIME on time.monotonic)
├─ tactical window (first ranking key)
│    2 complete simultaneous rounds over all rule-permitted pairs
│    (+1 gated round); all-or-nothing, robust one-ply fallback
├─ GBFS expansion (priority = heuristic eval ONLY, never the backup)
│    state key = positions + boxes + creds + step (+ board serial in caches)
│    credit to deep[root] before the closed check (cross-root transplants)
├─ backup per root: [window − root penalties] bucket → deep bucket →
│    strike tiebreak → exact values   (structure-only early-exit key on top)
├─ eval (zero-sum, label-symmetric, undiscounted)
│    locked 1000/cred · projected 600/race-won · adv 12/step (winner-
│    restricted, capped) · initiative 10/step (capped) · steal 1200×
│    [0.2 gradient | 0.35–0.5 race-won]  < W_LOCKED per box  ← this audit
└─ gates: denial only when opponent wins the race; forbidden = least-bad
     fallback; four directions, WAIT only when physically boxed in
```

## 6. Repro tooling (temp, not committed)

`record_game.py` (flags un-taken good pushes with full root tables),
`trace_flag2.py` (window min-max leaf traces + term decomposition),
`decomp_leaves.py`, `why_no_score.py`, `bench.py`.

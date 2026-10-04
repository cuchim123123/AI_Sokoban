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

#### Planned changes
- separate immediate root evaluation from deeper descendant evaluation
- root action ranking should consider:
  - immediate robust value
  - tactical progress value
  - deeper descendant value only as a secondary tie-breaker
- do not let a misleading descendant override a clearly superior immediate action

#### Rule
If a root action produces immediate tactical progress or parity-winning conflict resolution, it should not lose to a merely deeper but less relevant branch.

#### Output
Reduced stupid-move behavior where the agent chooses a branch that looks deeper but does not protect its box or win the conflict.

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

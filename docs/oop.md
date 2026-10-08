# Structure review — 2026-10-09

## Assessment

The project is reasonably organized for a small Pygame game. It separates the
two game modes, shared presentation code, domain rules, evaluation and search.
It uses useful OOP abstractions, but splitting a large app into mixin files does
not fully separate responsibilities: each mixin still reads and mutates the
same large application object. No broad rewrite is warranted before release.

## Current layout

| Location | Responsibility | Keep in source repository? |
| --- | --- | --- |
| `main.py` | Launcher and advertised CLI commands | Yes |
| `src/shared/` | Shared widgets, animation loader and mode launcher | Yes |
| `src/single/core/` | Single-player states, parsing, legal moves | Yes |
| `src/single/search/` | Search interface, nodes, UCS and A* implementations | Yes |
| `src/single/heuristics/` | Push distances, matching and deadlock detection | Yes; competitive preprocessing also imports push distances |
| `src/single/gui/` | Puzzle setup, input, solution playback and rendering | Yes |
| `src/competitive/` | Simultaneous transition, credits, evaluation, preferences and controllers | Yes |
| `src/competitive/gui/` | Competitive setup, input, round computation and rendering | Yes |
| `src/experiments/` | Optional benchmarks and heuristic verification | Yes while documented/exposed commands remain; not needed in a player-only distribution |
| `tests/` | Rule, search, loop-breaker and input regressions | Yes; omit from player-only packaging if desired |
| `maps/` | Playable maps and test fixtures, discovered dynamically by menus | Yes |
| `src/assets/` | Four cardinal directions for each animation, six board tiles and asset license | Yes |

Agent A owns the shared planner. Agent B subclasses it to reuse normal search
and applies a one-round legal-action restriction after observing two complete
position cycles. The UI only reads its loop-breaker telemetry; it does not
control this decision. Preferences and joint resolution remain separate from
the renderer. Pure functions for transitions/evaluation are appropriate here;
turning every helper into a class would not improve the design.

## Refactors worth considering (analysis only)

1. **Competitive worker lifecycle — highest priority.** `gui/game.py` reads
   mutable application fields during a background decision and accepts results
   using only the round number. A restart/new game can also begin at round zero.
   Snapshot state, board, agents and limit before dispatch; tag results with a
   game-generation token and reject stale results. The single-player solver
   already has a `_solve_token` pattern. This is a correctness boundary, not an
   AI-strategy change; it was not changed during this presentation/cleanup task.
2. **Make hashed states actually immutable.** `CompetitiveState` caches its hash
   but exposes assignable attributes; changing a position, credit or step after
   insertion into a cache could violate dictionary/set invariants. A frozen
   dataclass with slots, or enforced read-only fields, would express the intended
   model. Apply the same discipline to the single-player state. Current normal
   transitions construct new objects; this is preventive hardening.
3. **Replace implicit mixin coupling gradually.** Both GUI apps combine Setup,
   Events, Game and Render mixins which assume many undocumented `self` fields.
   Prefer a small session/model object and explicit renderer/controller
   collaborators when the UI grows. Do not undertake a wholesale MVC rewrite
   just to increase the number of classes.
4. **Public controller result instead of private-field writes.** The GUI writes
   `_last_action`, and search diagnostics also use module-global `LAST_SEARCH`.
   A typed decision/result plus an `on_round_resolved` method would make ownership
   clearer and remove fragile coupling, especially if agents become concurrent.
5. **Shared board preprocessing and asset paths.** Competitive state imports
   push-distance preprocessing from the single-player package; move that generic
   helper to a neutral domain utility in a future focused change. Resolve assets
   and maps relative to the project/install directory rather than the shell's
   working directory to make packaged/distributed launches reliable.
6. **Benchmark controller fidelity.** `competitive_benchmark.py` currently creates
   `AgentA` and changes its perspective for both seats. That measures the base
   search but does not exercise B's loop-breaker subclass. Use actual controller
   factories before claiming those matches represent the complete current game.

The `SearchAlgorithm` interface with A*/UCS implementations is a good use of
polymorphism. Agent B's small specialization is also reasonable; prefer
composition only if many independent strategies/policies are added. Keep game
rules independent of Pygame. Existing GUI mixins should not be treated as domain
models or reused as engine classes.

## Cleanup performed

- Removed obsolete `tools/debug2.py` and `tools/sim.py` and the historical
  competitive rewrite report with machine-specific temporary paths.
- Removed 40 unused diagonal animation files (`dir2`, `dir4`, `dir6`, `dir8`,
  PNG + JSON across five animation states). The loader only uses 1, 3, 5, 7.
- Removed the menu-art `Preview.png` and its now-unused blur/gradient helpers.
- Kept both game engines, legal-move rules, scoring, AI/search behavior, required
  cardinal animation data, tile textures, license and public CLI experiments.
- Retired references to the intentionally deleted dense-goals map. Tests use
  the remaining `main.txt` fixture without dropping symmetry or movement checks.
- Solid cyan menus now share one background constant, opaque dark containers,
  white labels and distinctly filled selected/hover controls.

## Validation

Run `py -3.11 -B -m unittest discover -s tests`. Render all three menus and both
boards headlessly; ensure every cardinal animation has frames, the cyan pixels
remain exact outside panels, and both modes can execute real transitions. No
claim of universal AI optimality follows from UI/cleanup validation.

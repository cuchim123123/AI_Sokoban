# Heuristic Research Context --- Single-Agent Sokoban

## Goal

Design an A\* heuristic for the original single-agent Sokoban
assignment.

The assignment explicitly prohibits: - Manhattan distance - Euclidean
distance

Therefore, the heuristic should use Sokoban-specific structure rather
than simple geometric distance.

## Candidate direction

A strong research direction is:

**Minimum legal push distance + box-goal matching**

For each box `B_i` and goal `D_j`, estimate the minimum number of
**legal box pushes** required to move the box to that goal while
respecting walls and push geometry.

Construct a cost matrix:

``` text
          D1   D2   D3
B1         ?    ?    ?
B2         ?    ?    ?
B3         ?    ?    ?
```

Then determine the minimum-cost one-to-one assignment between boxes and
goals.

Conceptually:

`h(s) = minimum assignment cost over box-goal pairs`

## Critical distinction

The assignment's action output consists of player movements: - North -
East - West - South

Therefore, the actual path cost is based on **agent actions**, not
simply box pushes.

A push-distance heuristic is consequently an estimate/lower bound
candidate, not automatically the exact remaining solution cost.

The heuristic must be analyzed rather than assumed to be admissible or
consistent.

## What needs to be investigated

### 1. How to calculate legal push distance

A push is only possible when: - the destination cell for the box is
free - the player can stand on the opposite side of the box -
walls/obstacles are respected

A useful approach is reverse-push analysis or a search over feasible
box-push configurations.

### 2. Box-goal matching

Because boxes and goals can be paired in different ways, simply summing
each box's nearest goal can produce a weak or incorrect assignment.

Investigate minimum-cost bipartite matching between boxes and goals.

### 3. Deadlocks

The heuristic/search should account for impossible box placements where
appropriate.

Important examples: - non-goal corners - wall-line deadlocks - 2x2
frozen configurations - boxes that cannot reach any goal

Be careful: adding arbitrary deadlock penalties can destroy
admissibility if the heuristic claims to be a lower bound.

### 4. Admissibility

A heuristic `h(s)` is admissible if:

`h(s) <= h*(s)`

for every state `s`, where `h*(s)` is the true minimum remaining
solution cost.

Do not assume that a sophisticated-looking heuristic is admissible.
Prove or experimentally test the relevant lower-bound property.

### 5. Consistency

For every transition:

`h(s) <= c(s,a,s') + h(s')`

and `h(goal) = 0`.

Again, this needs to be reasoned about and experimentally verified for
the exact heuristic definition.

## Important warning

Do not silently equate:

`minimum box pushes`

with:

`minimum player movement actions`.

A box-push count ignores some walking cost. If it is used as an A\*
heuristic, the relationship between push count and actual action cost
must be explicitly justified.

## Experimental verification

Useful experiment categories: - generate/collect solvable states -
compute an exact optimal cost using an appropriate exhaustive/optimal
solver on small instances - compare heuristic values against exact
costs - test admissibility violations - test consistency across
parent-child state transitions - compare UCS and A\* runtime and
explored/frontier state counts

The assignment explicitly requires experimental verification of
admissibility and consistency, so these properties should be treated as
empirical research questions rather than assumptions.

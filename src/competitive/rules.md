Work on my competitive, simultaneous two-agent Sokoban game. Use this specification when inspecting the engine and improving the AI. Distinguish confirmed rules from proposed defaults; report discrepancies before changing gameplay.

**1. Board and actions — confirmed**

- Agents A and B share one board containing walls, boxes, and goals.
- Both choose an action for the same round. Resolve their actions jointly, rather than treating the game as ordinary alternating-turn Sokoban.
- Available submissions are UP, DOWN, LEFT, and RIGHT. There is no selectable WAIT.
- Agents may submit only individually legal directions. Directions into walls or otherwise impossible moves must be excluded before selection.
- An individually legal action can still fail because of the other agent’s simultaneous action.
- Players and boxes cannot finish a round sharing a square. Boxes cannot overlap.

**2. Conflict priority — confirmed**

Determine priority using the number of remaining rounds BEFORE resolving the current round:

- Odd remaining rounds: A has priority.
- Even remaining rounds: B has priority.

Apply this priority to:

- Both agents attempting to enter the same square.
- Agents attempting to swap positions.
- Both agents attempting to push the same box.
- An agent attempting to enter the same square that the opponent is pushing a box into.

The priority agent’s action wins the conflict. The other agent yields and must move to an alternative position if possible.

Example: A moves into X while B pushes a box into X. With odd remaining rounds, A enters X and B’s push loses. With even remaining rounds, B pushes the box into X and A yields.

**3. Alternative moves — implementation requested**

Use a strategically ranked fallback instead of an arbitrary fixed direction order.

Each agent should provide its legal directions in preference order. When its preferred action loses a conflict, try its next preferred direction that can legally coexist with the winner’s resolved action.

Recheck each fallback against the resulting occupancy and box movements. A fallback cannot overturn the winner’s action or create another unresolved conflict. The AI’s search must simulate the same fallback behavior as the actual engine.

**4. Scoring — confirmed**

- Each agent’s score is the number of boxes currently on goals credited to that agent.
- Successfully delivering a box onto a goal credits the delivering agent.
- Moving a credited box off a goal removes its owner’s point.
- If the other agent subsequently delivers that box onto a goal, that agent gains the point.
- Points are not permanent or cumulative.
- The game has a fixed round limit. Final current scores determine the winner; equal scores produce a draw.

**5. Proposed defaults — not yet confirmed**

Check the existing implementation and flag these choices for confirmation:

- If the conflict loser has no valid alternative, it stays in place.
- The winner acts only if still physically legal. If the stationary loser blocks the winner, both stay.
- A blocked round still consumes one round.
- Following into the opponent’s successfully vacated square is allowed, except a swap uses the conflict rule.
- Two different boxes pushed toward the same destination use odd/even conflict priority.
- Pushing an opponent’s credited box directly from one goal onto another transfers credit to the successful pusher.
- Starting boxes on goals are uncredited unless the level assigns ownership.
- Filling every goal does not end the game early; play continues until the round limit.

**6. Verification**

Inspect the actual transition code, action validation, fallback selection, credit bookkeeping, and termination logic. Verify that search predictions match engine outcomes.

Do not confuse AI preferences—such as avoiding deadlocks or refusing to push its own box off a goal—with game rules prohibiting those actions.
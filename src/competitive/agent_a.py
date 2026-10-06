"""
Competitive agent, shared by Agent A and Agent B.

Engine: time-bounded GBFS with a persistent open list
(docs/ai_search_improvement_plan.md: "keep the current GBFS and beam search
as the base engine").

How a decision is made
----------------------
1. Every node is a joint state produced by the REAL transition function, so
   the search can never disagree with the rules: parity conflict priority
   (odd/even remaining steps), forced diversion of the loser, credit changes
   and occupancy rules are all simulated exactly.
2. Expansion is greedy best-first (GBFS): a priority queue ordered by
   `evaluation.evaluate` from my perspective, popped until the deadline.
   From a node, each own action is paired with the opponent's *worst-case*
   reply for us - the documented safety mechanism, evaluated through the
   real joint transition - so the open list climbs through lines that stay
   good even when the opponent answers well.
3. A per-call closed set keeps each state at most once; states that did not
   make the cut stay in the open list and remain expandable, so a promising
   line is never lost by a width cap (this replaces the optional beam /
   iterative-deepening scaffolding: the root ranking is monotone - robust
   values are fixed after the first level and descendant values only grow -
   so a partial search can never degrade the answer, it only refines it).
4. Root ranking follows the documented priority: immediate robust value
   (after the opponent's worst reply, quantized by ROBUST_MARGIN) decides
   first, the best descendant found by the search breaks ties. A misleading
   deep branch therefore cannot override a clearly superior immediate move,
   while near-equal lines are resolved by depth.

Speed comes from caching, never from thinking shallower:
  * persistent evaluation cache shared with the rule engine,
  * memoized joint transitions (a state is never resolved twice),
  * a per-call closed set: each state is expanded at most once.

Hard tactical constraints (never traded away for heuristic points):
  * never WAIT while a legal move exists,
  * never push your own credited box off its goal,
  * never push a box into a static deadlock,
  * never step back into a cell you just left while the board is unchanged
    (root REVISIT_PENALTY, kills oscillation).
"""

import heapq
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from src.competitive.evaluation import (
    creates_deadlock,
    evaluate,
)
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (
    get_valid_actions,
    resolve_joint_action_outcome,
)


TIME_LIMIT = 1.0         # seconds one choose_action call may use
MAX_DEPTH = 64           # node-depth cap (also bounded by remaining steps)
ROBUST_MARGIN = 200.0    # quantum of the primary (immediate) root ranking
REVISIT_PENALTY = 250.0  # root penalty for stepping onto a cell just visited
_TICK_MASK = 127         # check the clock every 127 node generations


class _Timeout(Exception):
    """Raised inside the search when the time budget is used up."""


# Telemetry of the most recent `best_action` call (engine, depth reached,
# expanded nodes, wall time). Handy for benchmarks and the GUI's debug output.
LAST_SEARCH: Dict[str, object] = {
    "engine": "GBFS",
    "perspective": None,
    "depth": 0,
    "nodes": 0,
    "time": 0.0,
}


def _key(state: CompetitiveState) -> tuple:
    """Identity of a position for the closed set: every joint transition
    advances the step counter by one, so the step count is implied by the
    board configuration and one entry blocks every cycle."""
    return (
        state.agent_a,
        state.agent_b,
        state.boxes,
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
    )


def history_entry(state: CompetitiveState, perspective: str) -> tuple:
    """
    Compact fingerprint of "where I stood, what the board looked like".
    Used by the anti-oscillation penalty: returning to the same cell while
    boxes and credits are unchanged means the agent is shuffling its feet.
    """
    pos = state.agent_a if perspective == "A" else state.agent_b
    return (pos, state.boxes, state.boxes_on_goals_a, state.boxes_on_goals_b)


class _Planner:
    """One time-bounded GBFS decision for one perspective."""

    __slots__ = (
        "state", "board", "max_steps", "me", "opp",
        "deadline", "recent", "banned", "eval_cache",
        "root_actions", "robust", "deep", "closed",
        "nodes", "final_depth", "best",
    )

    def __init__(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        perspective: str,
        deadline: float,
        recent_positions,
        banned_actions,
        eval_cache: Optional[Dict[tuple, float]],
    ):
        self.state = state
        self.board = board
        self.max_steps = max_steps
        self.me = perspective
        self.opp = "B" if perspective == "A" else "A"
        self.deadline = deadline
        self.recent = recent_positions
        self.banned: Optional[Set[Action]] = set(banned_actions) if banned_actions else None
        self.eval_cache = eval_cache
        self.root_actions: List[Action] = []
        self.robust: Dict[Action, float] = {}
        self.deep: Dict[Action, float] = {}
        self.closed: Set[tuple] = set()
        self.nodes = 0
        self.final_depth = 0
        self.best = Action.WAIT

    # ── bookkeeping ─────────────────────────────────────────────────────

    def _tick(self) -> None:
        self.nodes += 1
        if not (self.nodes & _TICK_MASK) and time.time() > self.deadline:
            raise _Timeout()

    def _my_pos(self, s: CompetitiveState) -> Tuple[int, int]:
        return s.agent_a if self.me == "A" else s.agent_b

    def _opp_pos(self, s: CompetitiveState) -> Tuple[int, int]:
        return s.agent_b if self.me == "A" else s.agent_a

    def _eval(self, s: CompetitiveState) -> float:
        return evaluate(s, self.board, self.me, self.max_steps, cache=self.eval_cache)

    # ── action generation ───────────────────────────────────────────────

    def _forbidden(self, s: CompetitiveState, action: Action, own_credited) -> bool:
        """Tactical moves that are never worth playing (unless forced)."""
        if action is Action.WAIT:
            return False
        pos = self._my_pos(s)
        dest = (pos[0] + action.value[0], pos[1] + action.value[1])
        if dest not in s.boxes:
            return False
        new_cell = (dest[0] + action.value[0], dest[1] + action.value[1])
        if dest in own_credited and new_cell not in self.board.goals:
            return True                      # throwing away my own point
        return creates_deadlock(pos, action, s.boxes, self.board)

    def _my_actions(self, s: CompetitiveState, root: bool = False):
        actions = get_valid_actions(
            self._my_pos(s), self._opp_pos(s), s.boxes, self.board
        )
        if not actions:
            return [Action.WAIT]             # completely boxed in: forced

        if root and self.banned:
            unbanned = [a for a in actions if a not in self.banned]
            if unbanned:
                actions = unbanned

        own = s.boxes_on_goals_a if self.me == "A" else s.boxes_on_goals_b
        allowed = [a for a in actions if not self._forbidden(s, a, own)]
        return allowed or actions            # everything bad -> least bad

    def _opp_actions(self, s: CompetitiveState):
        # No WAIT modelled: both agents must move every turn. If the opponent
        # has no legal move its forced stand-still arises naturally inside the
        # joint transition (conflict diversion / blocked entry).
        actions = get_valid_actions(
            self._opp_pos(s), self._my_pos(s), s.boxes, self.board
        )
        return actions or [Action.WAIT]

    def _is_revisit(self, s: CompetitiveState) -> bool:
        if not self.recent:
            return False
        return history_entry(s, self.me) in self.recent

    # ── GBFS expansion ──────────────────────────────────────────────────

    def _worst_reply(self, s: CompetitiveState, my_action: Action):
        """
        The opponent's reply that hurts me the most (lowest value from my
        perspective), applied through the real joint transition. Returns
        (child_state, value). Actions are passed in their owner's slot:
        action_a for agent A, action_b for agent B.
        """
        best_child: Optional[CompetitiveState] = None
        best_value = float("inf")
        for reply in self._opp_actions(s):
            if self.me == "A":
                child = resolve_joint_action_outcome(
                    s, my_action, reply, self.board, self.max_steps
                ).state
            else:
                child = resolve_joint_action_outcome(
                    s, reply, my_action, self.board, self.max_steps
                ).state
            value = self._eval(child)
            if value < best_value:
                best_value = value
                best_child = child
            self._tick()
        return best_child, best_value

    # ── root ranking (docs Phase 3) ─────────────────────────────────────

    def _rank_root(self) -> Action:
        """Immediate robust value decides (quantized), deeper descendant
        value breaks ties, then the exact robust value (progress within a
        margin), finally the fixed action order."""
        best: Optional[Action] = None
        best_key: Optional[tuple] = None
        for action in self.root_actions:
            if action not in self.robust:
                continue
            bucket = math.floor(self.robust[action] / ROBUST_MARGIN)
            key = (
                -bucket,
                -self.deep.get(action, self.robust[action]),
                -self.robust[action],
                action.value,
            )
            if best_key is None or key < best_key:
                best_key, best = key, action
        return best if best is not None else Action.WAIT

    # ── search loop ─────────────────────────────────────────────────────

    def run(self) -> Action:
        roots = self._my_actions(self.state, root=True)
        if not roots:
            return Action.WAIT
        self.root_actions = roots
        self.closed.add(_key(self.state))

        # First level: immediate robust values (small, always completes).
        heap: List[tuple] = []
        try:
            for seq, action in enumerate(roots):
                child, value = self._worst_reply(self.state, action)
                if self._is_revisit(child):
                    value -= REVISIT_PENALTY
                self.robust[action] = value
                self.deep[action] = value
                self.closed.add(_key(child))
                heapq.heappush(heap, (-value, seq, 1, child, action))
        except _Timeout:
            self.best = self._rank_root()
            return self.best

        self.final_depth = 1
        self.best = self._rank_root()
        max_depth = min(MAX_DEPTH, max(self.max_steps - self.state.step, 1))

        # GBFS: repeatedly expand the highest-valued open node until the
        # deadline. Nothing is discarded by a width cap, the closed set
        # blocks cycles, and the root ranking is monotone (robust values
        # are fixed, descendant values only grow) - so the answer returned
        # at any moment is a valid refinement of the previous one.
        seq = len(roots)
        try:
            while heap:
                if time.time() >= self.deadline:
                    break
                neg_value, _, depth, node, root = heapq.heappop(heap)
                if depth >= max_depth:
                    continue                   # leaf: beyond the depth cap
                new_depth = depth + 1
                for action in self._my_actions(node):
                    child, value = self._worst_reply(node, action)
                    key = _key(child)
                    if key in self.closed:
                        continue
                    self.closed.add(key)
                    if value > self.deep[root]:
                        self.deep[root] = value
                    if new_depth > self.final_depth:
                        self.final_depth = new_depth
                    if new_depth < max_depth:
                        heapq.heappush(
                            heap, (-value, seq, new_depth, child, root)
                        )
                        seq += 1
                    self._tick()
        except _Timeout:
            pass                               # keep everything found so far

        self.best = self._rank_root()
        return self.best


def best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str = "A",
    recent_positions=None,
    tt: Optional[Dict[tuple, tuple]] = None,
    eval_cache: Optional[Dict[tuple, float]] = None,
    time_limit: float = TIME_LIMIT,
    banned_actions=None,
    **_ignored,
) -> Action:
    """
    Best action for `perspective` within `time_limit` seconds.

    The GBFS open list runs to the deadline; the root ranking is monotone,
    so whatever has been found when time runs out is returned unchanged.
    `tt` is accepted for compatibility with older callers; the engine keeps
    its own per-call closed set instead.
    """
    if eval_cache is None:
        eval_cache = _ignored.get("heuristic_cache")
    started = time.time()
    deadline = started + max(0.05, time_limit - 0.05)
    planner = _Planner(
        state, board, max_steps, perspective, deadline,
        recent_positions, banned_actions, eval_cache,
    )
    action = planner.run()
    LAST_SEARCH.update(
        engine="GBFS",
        perspective=perspective,
        depth=planner.final_depth,
        nodes=planner.nodes,
        time=time.time() - started,
    )
    return action


class AgentA:
    """Agent A controller."""

    perspective = "A"

    def __init__(self, time_limit: float = TIME_LIMIT):
        self.time_limit = time_limit
        self._history: Deque = deque(maxlen=6)
        self.tt: Dict[tuple, tuple] = {}
        # Shared evaluation cache (the historical name used by debug tools).
        self.heuristic_cache: Dict[tuple, float] = {}
        self._last_action: Optional[Action] = None

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions=None,
    ) -> Action:
        action = best_action(
            state,
            board,
            max_steps,
            perspective=self.perspective,
            recent_positions=self._history,
            tt=self.tt,
            eval_cache=self.heuristic_cache,
            time_limit=self.time_limit,
            banned_actions=banned_actions,
        )
        self._history.append(history_entry(state, self.perspective))
        self._last_action = action
        return action

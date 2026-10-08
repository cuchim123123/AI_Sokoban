"""
Competitive agent, shared by Agent A and Agent B.

Engine: time-bounded GBFS with a persistent open list
(docs/ai_search_improvement_plan.md: "keep the current GBFS and beam search
as the base engine") plus a bounded joint-action tactical window at the root.

How a decision is made
----------------------
1. Every node is a joint state produced by the REAL transition function, so
   the search can never disagree with the rules: parity conflict priority
   (odd/even remaining steps), forced diversion of the loser, credit changes
   and occupancy rules are all simulated exactly.
2. Bounded tactical window (before the GBFS run): for every root action a
   pure-action maximin search over WINDOW_ROUNDS complete simultaneous
   rounds - ALL rule-permitted direction pairs, the opponent's minimum taken
   over CONTINUATION values, not one-step evaluations. Completed rounds
   only: if the budget runs out mid-window the whole result is discarded
   and the always-complete one-ply robust values are used instead. This is
   the root's BACKUP VALUE; it sees parity conflicts and delayed opponent
   threats that a one-step reply model misses.
3. Expansion is greedy best-first (GBFS): a priority queue ordered by
   `evaluation.evaluate` from my perspective - this is only the EXPANSION
   PRIORITY and never decides the root move. From a node each own action is
   paired with the opponent's worst one-step reply for the open-list
   ordering; the root decision comes from the tactical window above.
4. A per-call closed set (keyed by configuration AND step) keeps each state
   at most once; when a state is reached under a second root its value is
   still credited to that root's backup (transposition value transfer), so
   expansion order cannot decide which root sees a shared continuation. One
   open list is kept per root and a root that trails the busiest one by
   FAIR_LAG expansions is served anyway, so a line that has to cross a value
   valley (approach, absorb the worst-case reply, cash in deeper) is still
   developed instead of starving behind shallower but safer nodes. Within
   one search the ranking is monotone: the window is fixed once completed
   and descendant values only grow, so a partial search only refines the
   answer.
5. Root ranking priority: the tactical-window value decides first
   (bucketed by TACTICAL_MARGIN - fine enough that any difference the
   window sees decides), the best GBFS descendant value breaks ties
   (bucketed by ROBUST_MARGIN), then the child's strike distance to the
   boxes still in play, then the exact values, finally the fixed action
   order. Optimistic descendant peaks therefore cannot override the
   tactical analysis - they only separate roots the window considers
   equal.

Speed comes from caching, never from thinking shallower:
  * persistent evaluation cache keyed by board identity + configuration +
    remaining steps (shared with the rule engine),
  * memoized joint transitions keyed by board + max steps + configuration,
  * a per-call closed set: each state is expanded at most once,
  * early exit: with the window fixed and deep values monotone, a ranking
    whose structure (winner and buckets) stops changing for EXIT_STABLE
    pops and EXIT_STABLE_MS of wall time means further thinking is idle -
    quiet positions return in a few hundred ms instead of always burning
    the full budget (the monotonic deadline stays the hard cap; early
    exit can only undercut it).

Hard tactical constraints (never traded away for heuristic points):
  * never WAIT while a legal move exists,
  * never push your own credited box off its goal (the candidate
    counterexample - freeing a loose box that needs exactly that cell -
    was constructed and refuted, see evaluation.py),
  * never push a box into a static deadlock - unless the opponent already
    wins the delivery race for it (a justified denial preserving a lead
    or converting a loss into a draw),
  * never step back into a cell you just left while the board is unchanged
    (root REVISIT_PENALTY, kills oscillation),
  * never open a root by walking away from every box still in play
    (root AWAY_PENALTY, kills aimless laps - relative eval terms cannot
    see a symmetric retreat).

Deadlines use `time.monotonic()`; if even the seeding phase cannot finish,
a legal direction (the first root action) is returned - WAIT only when the
agent is physically boxed in.
"""

import heapq
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from src.competitive.evaluation import (
    INF,
    creates_deadlock,
    denial_justified,
    evaluate,
    strike_distance,
)
from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import (
    get_valid_actions,
    resolve_joint_action_outcome,
)


TIME_LIMIT = 1.0         # seconds one choose_action call may use
MAX_DEPTH = 64           # node-depth cap (also bounded by remaining steps)
ROBUST_MARGIN = 200.0    # quantum of the deep-descendant bucket (2nd key)
TACTICAL_MARGIN = 100.0  # quantum of the tactical bucket (1st key): finer
                         # than the deep bucket so a difference the window
                         # can actually see is never folded into one bucket
                         # and handed over to an optimistic descendant peak
REVISIT_PENALTY = 250.0  # root penalty for stepping onto a cell just visited
AWAY_PENALTY = 25.0      # root penalty per step of delivery cost GONE UP:
                         # all eval terms are relative, so two agents wandering
                         # in parallel look perfectly neutral - this stops a
                         # root that walks away from every box still in play
                         # (and prices a push that drags a box AWAY from its
                         # goal: delivery cost, not strike distance - see
                         # _away_penalty)
FAIR_LAG = 64            # expansions a root may trail the leader before it
                         # is served anyway (keeps valley-crossing lines alive)
EXIT_STABLE = 300         # pops with the root ranking STRUCTURALLY unchanged
                          # (same winner, same buckets, same strike) -> stop
                          # early. A pop evaluates up to 16 children, so the
                          # real rate is only ~1.5-2 pops/ms: thresholds in
                          # the thousands need more silence than a 1s budget
                          # has left after the window and the initial churn
                          # (measured: never fired within budget). Exact
                          # values are deliberately NOT compared: deep keeps
                          # creeping inside its bucket, which is refinement,
                          # not change - bit-stability made the exit
                          # unreachable in practice.
EXIT_STABLE_MS = 0.15     # ...AND at least this much WALL silence, because a
                          # pop count alone is speed-dependent: on tiny boards
                          # 300 pops may be only ~10ms of evidence, which made
                          # the agent bail out of an active box wrestle
                          # almost immediately. Both conditions must hold -
                          # the pop floor proves enough of the tree was
                          # sampled, the time floor proves the machine really
                          # did keep thinking without a verdict change.
EXIT_MIN_NODES = 2000    # never stop early before this many evaluated
                         # children (window + seeding included)
EXIT_MIN_DEPTH = 4       # ...nor before the deepest line reached this depth
WINDOW_ROUNDS = 2        # complete simultaneous rounds the root window searches
WINDOW_EXTRA_ROUNDS = 1  # extra rounds attempted only while they are cheap
WINDOW_BUDGET = 0.35     # share of the remaining budget the window may use
WINDOW_DEEP_GATE = 0.15  # deeper rounds only while the window stays this cheap
_TICK_MASK = 15         # check the clock every 16 node generations (a node
                        # grew heavier with per-root bookkeeping; checking
                        # often keeps the deadline overshoot under ~4 ms)


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
    "window": 0,        # completed tactical-window rounds (0 = fallback)
}


def _key(state: CompetitiveState) -> tuple:
    """Identity of a position for the closed set.

    The step count is rule-relevant and therefore part of identity:
    identical configurations reached at different rounds have different
    remaining time (evaluation gates such as `cost <= remaining`, the
    terminal score freeze) and - when the difference is odd, which needs
    an agent to have stood still on the way - the OPPOSITE conflict
    priority. One entry blocks cycles at one specific round only, which
    is exactly the distinction the rules make (state.__eq__ includes the
    step for the same reason).
    """
    return (
        state.agent_a,
        state.agent_b,
        state.boxes,
        state.boxes_on_goals_a,
        state.boxes_on_goals_b,
        state.step,
    )


def history_entry(state: CompetitiveState, perspective: str) -> tuple:
    """
    Compact fingerprint of "where I stood, what the board looked like".
    Used by the anti-oscillation penalty: returning to the same cell while
    boxes and credits are unchanged means the agent is shuffling its feet.
    """
    pos = state.agent_a if perspective == "A" else state.agent_b
    return (pos, state.boxes, state.boxes_on_goals_a, state.boxes_on_goals_b)


def _rank_structure(rank: Tuple[Optional[Action], Optional[tuple]]) -> tuple:
    """
    Structural skeleton of a `_rank_best` result: winner + its tactical
    bucket, deep bucket and strike (the first three key components).

    The early-exit stability check compares this instead of the full key.
    Exact values inside a bucket keep creeping while the monotone deep
    term refines - that is improvement within the same verdict, not a
    change of verdict, and demanding bit-stability on it made the exit
    practically unreachable (measured: never fired once within a 1s
    budget). Roots can only reorder across buckets or on exact ties, so
    if the winner and its buckets hold for EXIT_STABLE pops, the ranking
    has structurally settled; a rival crossing a bucket boundary changes
    the winner or the winner's bucket and resets the counter either way.
    """
    action, key = rank
    if key is None:
        return (action, None)
    return (action, key[0], key[1], key[2])


class _Planner:
    """One time-bounded GBFS decision for one perspective."""

    __slots__ = (
        "state", "board", "max_steps", "me", "opp",
        "deadline", "recent", "banned", "eval_cache",
        "root_actions", "robust", "deep", "strikes", "penalties",
        "closed", "window", "window_rounds",
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
        self.strikes: Dict[Action, int] = {}
        self.penalties: Dict[Action, float] = {}
        # closed maps state identity -> its (root-independent) value so a
        # state reached under a second root can still transfer its value.
        self.closed: Dict[tuple, float] = {}
        self.window: Optional[Dict[Action, float]] = None
        self.window_rounds = 0
        self.nodes = 0
        self.final_depth = 0
        self.best = Action.WAIT

    # ── bookkeeping ─────────────────────────────────────────────────────

    def _tick(self) -> None:
        self.nodes += 1
        if not (self.nodes & _TICK_MASK) and time.monotonic() > self.deadline:
            raise _Timeout()

    def _my_pos(self, s: CompetitiveState) -> Tuple[int, int]:
        return s.agent_a if self.me == "A" else s.agent_b

    def _opp_pos(self, s: CompetitiveState) -> Tuple[int, int]:
        return s.agent_b if self.me == "A" else s.agent_a

    def _eval(self, s: CompetitiveState) -> float:
        return evaluate(s, self.board, self.me, self.max_steps, cache=self.eval_cache)

    # ── action generation ───────────────────────────────────────────────

    def _forbidden(self, s: CompetitiveState, action: Action, own_credited) -> bool:
        """Tactical moves that are never worth playing (unless forced).

        The deadlock filter is strategic pruning, NOT a game rule, and it
        keeps a constructed counterexample that unlocks it: the opponent
        already wins the delivery race for that box, so killing it
        preserves a lead or converts a loss into a draw (denial_justified);
        killing my own race only loses points.

        The own-credited-off-goal filter stays hard: the candidate
        counterexample (free a loose box that needs exactly my cell while
        I re-seat elsewhere) was constructed and refuted - see the audit
        note in evaluation.py.
        """
        if action is Action.WAIT:
            return False
        pos = self._my_pos(s)
        dest = (pos[0] + action.value[0], pos[1] + action.value[1])
        if dest not in s.boxes:
            return False
        new_cell = (dest[0] + action.value[0], dest[1] + action.value[1])
        if dest in own_credited and new_cell not in self.board.goals:
            return True                      # throwing away my own point
        if not creates_deadlock(pos, action, s.boxes, self.board):
            return False
        return not denial_justified(s, self.board, dest, self.me, self.max_steps)

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

    def _away_penalty(self, s: CompetitiveState, child: CompetitiveState) -> float:
        """
        Root-only shaping priced on my actual OBJECTIVE margin: the exact
        walk+push cost of cashing the nearest box still in play, before vs
        after the move.

        A move that leaves that cost higher is shuffling - or, for a push,
        dragging a box away from its goal - and pays AWAY_PENALTY per step
        of damage; a move that lowers it is progress and is never charged.

        Why delivery cost and not strike distance (the previous basis):
          * strike distance is 0 while standing ON any approach cell, so
            stepping around to the CORRECT pushing side scored as
            "fleeing" and was charged - while the actual wrong push
            scored 0, because pushing the box moved the target set along
            with it and hid the damage. On capacity_lab that inversion
            ranked the wrong push BEST of the four roots.
          * exact walk+push cost sees both: approach maneuvers lower it,
            wrong pushes raise it (3 -> 6 steps there = 75 charged).

        Geometry bookkeeping: both sides are priced over the boxes that
        are LOOSE AFTER the move (a scoring push must not be charged for
        "losing" the box it just cashed in), with a pushed box priced at
        its PRE-move position on the before side, and boxes that were
        opponent-credited before the move excluded from the before side
        (a strip adds a steal target; it is not a cost increase).
        """
        if self._my_pos(child) == self._my_pos(s):
            return 0.0                      # I did not move: cannot have fled
        cred_s = s.boxes_on_goals_a if self.me == "A" else s.boxes_on_goals_b
        cred_c = (child.boxes_on_goals_a if self.me == "A"
                  else child.boxes_on_goals_b)
        loose_c = child.boxes - cred_c
        if not loose_c:
            return 0.0                      # everything cashed in: nothing left
        before_set = set(loose_c)
        old_pos = next(iter(s.boxes - child.boxes), None)
        new_pos = next(iter(child.boxes - s.boxes), None)
        if old_pos is not None and new_pos is not None and new_pos in before_set:
            before_set.remove(new_pos)      # loose push: price the box where
            before_set.add(old_pos)         # it was BEFORE the move
        before_set &= (s.boxes - cred_s)    # drops strips (wasn't mine to cash)
        if not before_set:
            return 0.0
        before = self._my_cost(self._my_pos(s), before_set)
        after = self._my_cost(self._my_pos(child), loose_c)
        if before >= INF or after >= INF:
            return 0.0                      # nothing reachable to chase
        return AWAY_PENALTY * max(0, after - before)

    def _my_cost(self, pos, targets) -> int:
        """Cheapest walk+push delivery for me from `pos` to cash any box in
        `targets` (precomputed exact step costs, nearest goal)."""
        best = INF
        for box in targets:
            for goal in self.board.goals:
                cost = self.board.exact_step_costs.get(goal, {}).get(
                    (box[0], box[1], pos[0], pos[1]), INF
                )
                if cost < best:
                    best = cost
        return best

    def _min_strike(self, pos, targets, boxes) -> int:
        if not targets:
            return INF                      # no target: nothing to walk to
        return min(strike_distance(pos, b, boxes, self.board) for b in targets)

    def _child_strike(self, child: CompetitiveState) -> int:
        """Root tiebreak measure: my walk to the nearest box still in play
        after the child state. Cash in everything and there is nothing left
        to chase - that ranks best of all (mapped below every finite walk)."""
        mine = (
            child.boxes_on_goals_a
            if self.me == "A"
            else child.boxes_on_goals_b
        )
        targets = child.boxes - mine
        if not targets:
            return -INF
        d = self._min_strike(self._my_pos(child), targets, child.boxes)
        return 0 if d >= INF else d         # unreachable: neutral direction

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

    # ── bounded joint-action tactical window (root backup) ──────────────

    def _w_reply(
        self, s: CompetitiveState, my_action: Action, rounds: int, wd: float
    ) -> float:
        """One simultaneous round: the opponent commits against my
        `my_action` (min over EVERY rule-permitted reply through the real
        joint transition), then the game continues with `rounds - 1`
        rounds left. The minimum is taken over continuation values, so a
        reply that looks harmless immediately but is stronger several
        rounds later is not discarded."""
        worst = float("inf")
        for reply in self._opp_actions(s):
            if time.monotonic() > wd:
                raise _Timeout()
            if self.me == "A":
                child = resolve_joint_action_outcome(
                    s, my_action, reply, self.board, self.max_steps
                ).state
            else:
                child = resolve_joint_action_outcome(
                    s, reply, my_action, self.board, self.max_steps
                ).state
            value = self._w_value(child, rounds - 1, wd)
            if value < worst:
                worst = value
            self._tick()
        return worst

    def _w_value(
        self, s: CompetitiveState, rounds: int, wd: float
    ) -> float:
        """Maximin continuation with `rounds` rounds left: I commit first
        (the conservative pure-action ordering - the opponent answers
        knowing my action), then their minimum. At rounds == 0 the leaf
        value is `evaluate`, except when the rounds ran the game out,
        where `evaluate` returns the exact frozen final score."""
        if rounds <= 0:
            return self._eval(s)
        best = -float("inf")
        for action in self._my_actions(s):
            value = self._w_reply(s, action, rounds, wd)
            if value > best:
                best = value
            self._tick()
        return best

    def _tactical_window(self, rounds: int, wd: float) -> Dict[Action, float]:
        """Backup value per root: `_w_reply` over the root action for
        `rounds` complete simultaneous rounds. Raises `_Timeout` if the
        window budget runs out - the caller then discards partial results
        (completed depth is only comparable when every root has it)."""
        values: Dict[Action, float] = {}
        for action in self.root_actions:
            if time.monotonic() > wd:
                raise _Timeout()
            values[action] = self._w_reply(self.state, action, rounds, wd)
        return values

    # ── root ranking (docs Phase 3) ─────────────────────────────────────

    def _rank_best(self) -> Tuple[Optional[Action], Optional[tuple]]:
        """Winning (action, key) of the current root ranking, or
        (None, None) if nothing was seeded. The ranking is the BACKUP
        VALUE of the search, deliberately separate from the expansion
        priority (the GBFS heap ordering, which is only heuristic
        guidance for which node to expand next).

        Key order:
          1. tactical bucket (TACTICAL_MARGIN): pure-action maximin over
             COMPLETED rounds with root penalties applied; the one-ply
             robust value while no window completed - always a valid
             fallback. Finer than the deep bucket, so any difference the
             window can actually see decides before the optimistic term,
          2. best GBFS descendant bucket (ROBUST_MARGIN) - optimistic and
             monotone, so it may only separate roots the tactical
             analysis considers equal; a transient heuristic peak can
             never override the window's verdict on delayed opponent
             threats,
          3. the child's strike distance to the boxes still in play
             (equal lines step toward the fight instead of repeating
             whichever direction the raw tuple order prefers),
          4-6. exact window, descendant and robust values, then the fixed
             action order.
        """
        best: Optional[Action] = None
        best_key: Optional[tuple] = None
        for action in self.root_actions:
            if action not in self.robust:
                continue
            if self.window is not None and action in self.window:
                tactical = self.window[action] - self.penalties.get(action, 0.0)
            else:
                tactical = self.robust[action]   # already penalized
            deep = self.deep.get(action, tactical)
            key = (
                -math.floor(tactical / TACTICAL_MARGIN),
                -math.floor(deep / ROBUST_MARGIN),
                self.strikes.get(action, INF),
                -tactical,
                -deep,
                -self.robust[action],
                action.value,
            )
            if best_key is None or key < best_key:
                best_key, best = key, action
        return best, best_key

    def _rank_root(self) -> Action:
        """Best root action under the ranking in `_rank_best`.

        Partial search: the window exists in completed form for every
        root or not at all; deep values only grow, so the ranking at any
        instant is a valid refinement of the previous one. Terminal
        window leaves carry the exact frozen final score; non-terminal
        leaves are heuristic estimates, not proven game bounds.

        If seeding did not finish, the first root action is returned (a
        legal direction - WAIT only when the agent is physically boxed
        in, in which case WAIT is the sole action)."""
        best, _ = self._rank_best()
        if best is not None:
            return best
        return self.root_actions[0] if self.root_actions else Action.WAIT

    # ── search loop ─────────────────────────────────────────────────────

    def run(self) -> Action:
        roots = self._my_actions(self.state, root=True)
        if not roots:
            return Action.WAIT
        self.root_actions = roots
        t0 = time.monotonic()
        budget = max(1e-9, self.deadline - t0)
        self.closed[_key(self.state)] = self._eval(self.state)

        # First level: immediate robust values (small, always completes).
        heaps: Dict[Action, List[tuple]] = {}
        try:
            for seq, action in enumerate(roots):
                child, plain = self._worst_reply(self.state, action)
                penalty = 0.0
                if self._is_revisit(child):
                    penalty += REVISIT_PENALTY
                penalty += self._away_penalty(self.state, child)
                value = plain - penalty
                self.penalties[action] = penalty
                self.robust[action] = value
                self.deep[action] = value
                self.strikes[action] = self._child_strike(child)
                self.closed[_key(child)] = plain
                heaps[action] = [(-value, seq, 1, child, action)]
        except _Timeout:
            self.best = self._rank_root()
            return self.best

        # Bounded joint-action tactical window: the root BACKUP value.
        # Completed rounds for every root or nothing - a partially built
        # window is not comparable across roots, so on timeout the whole
        # result is discarded and _rank_root falls back to the one-ply
        # robust values (a valid fallback that always completes).
        window_started = time.monotonic()
        wd = min(self.deadline, window_started + WINDOW_BUDGET * budget)
        try:
            self.window = self._tactical_window(WINDOW_ROUNDS, wd)
            self.window_rounds = WINDOW_ROUNDS
            if time.monotonic() - window_started < WINDOW_DEEP_GATE * budget:
                try:
                    self.window = self._tactical_window(
                        WINDOW_ROUNDS + WINDOW_EXTRA_ROUNDS, wd
                    )
                    self.window_rounds = WINDOW_ROUNDS + WINDOW_EXTRA_ROUNDS
                except _Timeout:
                    pass               # keep the completed shallower window
        except _Timeout:
            self.window = None
            self.window_rounds = 0

        self.final_depth = 1
        self.best = self._rank_root()
        max_depth = min(MAX_DEPTH, max(self.max_steps - self.state.step, 1))

        # GBFS with one open list per root: the globally best node is
        # expanded normally, but a root trailing the busiest root by more
        # than FAIR_LAG expansions is served next. A line that must cross
        # a value valley (approach, absorb the opponent's worst-case
        # reply, cash in deeper) therefore still gets developed instead
        # of starving behind shallower nodes. Nothing is discarded by a
        # width cap and the closed set keys on configuration AND step, so
        # cycle/transposition blocking never conflates different rounds.
        seq = len(roots)
        expanded = {action: 0 for action in roots}
        # Early exit: with the window fixed after completion and deep
        # values monotone, the ranking is only REFINED, never
        # invalidated. Once the STRUCTURE of the ranking (winner, its
        # tactical and deep buckets, its strike - key[:3]) has not changed
        # for EXIT_STABLE pops AND EXIT_STABLE_MS of wall time past the
        # minimum floors, further thinking is measurably idle: exact
        # values may still creep inside their buckets, but that cannot
        # reorder roots across buckets or flip the winner without changing
        # this structure. Quiet positions - agents far apart, nothing
        # contested - return in a few hundred ms instead of burning the
        # full budget, while positions whose bucket structure keeps
        # improving reset both counters and search to the deadline. The
        # deadline check stays monotonic; early exit can only undercut
        # it, never exceed it.
        stable = 0
        last_rank = self._rank_best()
        last_change = time.monotonic()
        try:
            while True:
                if time.monotonic() >= self.deadline:
                    break
                live = [a for a in roots if heaps.get(a)]
                if not live:
                    break
                best_root = min(live, key=lambda a: heaps[a][0][0])
                leader = max(expanded[a] for a in live)
                lagging = [a for a in live if expanded[a] <= leader - FAIR_LAG]
                target = (
                    min(lagging, key=lambda a: heaps[a][0][0])
                    if lagging
                    else best_root
                )
                neg_value, _, depth, node, root = heapq.heappop(heaps[target])
                expanded[target] += 1
                if depth >= max_depth:
                    continue                   # leaf: beyond the depth cap
                new_depth = depth + 1
                for action in self._my_actions(node):
                    child, value = self._worst_reply(node, action)
                    # The value of a state is root-independent: credit it
                    # to this root BEFORE the closed check, so a state
                    # reached under a second root still transfers its
                    # continuation value instead of being skipped
                    # silently (expansion order would otherwise decide
                    # which root sees shared states).
                    if value > self.deep[root]:
                        self.deep[root] = value
                    if new_depth > self.final_depth:
                        self.final_depth = new_depth
                    key = _key(child)
                    if key in self.closed:
                        continue
                    self.closed[key] = value
                    if new_depth < max_depth:
                        heapq.heappush(
                            heaps[root], (-value, seq, new_depth, child, root)
                        )
                        seq += 1
                    self._tick()
                rank_now = self._rank_best()
                if _rank_structure(rank_now) == _rank_structure(last_rank):
                    stable += 1
                    if (
                        self.window is not None
                        and stable >= EXIT_STABLE
                        and time.monotonic() - last_change >= EXIT_STABLE_MS
                        and self.nodes >= EXIT_MIN_NODES
                        and self.final_depth >= EXIT_MIN_DEPTH
                    ):
                        break               # ranking settled: stop early
                else:
                    last_rank = rank_now
                    stable = 0
                    last_change = time.monotonic()
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

    The tactical window (up to WINDOW_BUDGET of the budget) fixes the
    root backup value first; the GBFS open list then runs to the
    deadline refining the descendant term. Both keys are monotone, so
    whatever has been found when time runs out is returned unchanged.
    Deadlines use `time.monotonic()`; on any timeout the ranking falls
    back to the always-completed one-ply values, and if even seeding
    cannot finish, to the first legal root direction.
    `tt` is accepted for compatibility with older callers; the engine keeps
    its own per-call closed set instead.
    """
    if eval_cache is None:
        eval_cache = _ignored.get("heuristic_cache")
    started = time.monotonic()
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
        time=time.monotonic() - started,
        window=planner.window_rounds,
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

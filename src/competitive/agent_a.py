"""One time-bounded pure-action maximin search over simultaneous rounds.

Both actions are applied together. Max/min describes a conservative security
value, not alternating physical turns or knowledge of a live submission.
Every completed depth replaces the previous root values wholesale. No best
intermediate position, forced scoring plan, or partially searched root wins.
"""
import time
import gc
from collections import deque
from functools import wraps
from threading import Lock

from src.competitive.state import Action
from src.competitive.evaluation import evaluate, evaluation_components
from src.competitive.preferences import round_preferences
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome

TIME_LIMIT = 1.0
LAST_SEARCH = {}
_gc_lock = Lock()
_gc_users = 0
_gc_was_enabled = False


def _bounded_allocations(function):
    """Defer cyclic-GC pauses while building acyclic search tables.

    Refcount cleanup still runs. Restore the prior GC setting after the last
    nested/concurrent decision, including exceptions. Profiling found full
    collections could pause a single expansion for more than 100 ms.
    """
    @wraps(function)
    def call(*args, **kwargs):
        global _gc_users, _gc_was_enabled
        with _gc_lock:
            if _gc_users == 0:
                _gc_was_enabled = gc.isenabled()
                gc.disable()
            _gc_users += 1
        try:
            return function(*args, **kwargs)
        finally:
            with _gc_lock:
                _gc_users -= 1
                if _gc_users == 0 and _gc_was_enabled:
                    gc.enable()
    return call


class _Timeout(Exception):
    pass


def _key(state):
    return (state.agent_a, state.agent_b, state.boxes,
            state.boxes_on_goals_a, state.boxes_on_goals_b, state.step)


def history_entry(state, perspective):
    pos = state.agent_a if perspective == "A" else state.agent_b
    return (pos, state.boxes, state.boxes_on_goals_a, state.boxes_on_goals_b)


class _Planner:
    def __init__(self, state, board, max_steps, perspective, deadline,
                 recent_positions=None, banned_actions=None, eval_cache=None):
        self.state, self.board, self.max_steps = state, board, max_steps
        self.me = perspective
        self.deadline = deadline
        self.recent = tuple(recent_positions or ())
        self.banned = set(banned_actions or ())
        self.eval_cache = {} if eval_cache is None else eval_cache
        # All search-local caches share this immutable board/limit/perspective
        # and fallback policy. State identity includes credits AND round.
        self.tt, self.rows, self.prefs, self.pv = {}, {}, {}, {}
        self.nodes = self.hits = 0
        self.completed_depth = 0
        self.root_values = {}
        self.iterations = []
        self.components = {}

    def check(self):
        if time.monotonic() >= self.deadline:
            raise _Timeout()

    def score(self, state):
        self.check()
        value = evaluate(state, self.board, self.me, self.max_steps, self.eval_cache)
        self.check()
        return value

    def preferences(self, state, who, primary=None):
        key = (state, who)
        if key not in self.prefs:
            self.check()
            self.prefs[key] = round_preferences(state, self.board, self.max_steps, who)
        base = self.prefs[key]
        if primary in base:
            return (primary,) + tuple(a for a in base if a != primary)
        return base

    def actions(self, state, who):
        return self.preferences(state, who) or (Action.WAIT,)

    def children(self, state, action):
        key = (state, action)
        if key not in self.rows:
            result = []
            opponent = "B" if self.me == "A" else "A"
            for reply in self.actions(state, opponent):
                self.check()
                aa, ab = (action, reply) if self.me == "A" else (reply, action)
                pa = self.preferences(state, "A", aa)
                pb = self.preferences(state, "B", ab)
                outcome = resolve_joint_action_outcome(
                    state, aa, ab, self.board, self.max_steps, pa, pb)
                result.append((self.score(outcome.state), reply, outcome))
                if state == self.state:
                    self.check()
                    self.components[outcome.state] = evaluation_components(
                        outcome.state, self.board, self.max_steps)
                self.nodes += 1
            # Ordering helps pruning; every reply remains available.
            self.rows[key] = sorted(result, key=lambda item: item[0])
        return self.rows[key]

    def row_value(self, state, action, depth, alpha, beta):
        worst = float("inf")
        for _, reply, outcome in self.children(state, action):
            self.check()
            value = self.value(outcome.state, depth - 1, alpha, min(beta, worst))
            worst = min(worst, value)
            if worst <= alpha:
                break
        return worst

    def value(self, state, depth, alpha=-float("inf"), beta=float("inf")):
        self.check()
        if depth <= 0 or state.is_terminal(self.max_steps):
            return self.score(state)
        key = (state, depth)
        original_alpha, original_beta = alpha, beta
        entry = self.tt.get(key)
        if entry is not None:
            value, flag = entry
            self.hits += 1
            if flag == "exact":
                return value
            if flag == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        actions = list(self.actions(state, self.me))
        preferred = self.pv.get(state)
        if preferred in actions:
            actions.remove(preferred)
            actions.insert(0, preferred)
        best, selected = -float("inf"), actions[0]
        for action in actions:
            value = self.row_value(state, action, depth, alpha, beta)
            if value > best:
                best, selected = value, action
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        flag = "upper" if best <= original_alpha else "lower" if best >= original_beta else "exact"
        self.tt[key] = (best, flag)
        self.pv[state] = selected
        return best

    def root_iteration(self, roots, depth):
        # Each root gets an exact value at the SAME horizon. Root alpha is
        # deliberately reset: bounds on losing roots are not logged as values.
        values = {}
        for action in roots:
            self.check()
            values[action] = self.row_value(self.state, action, depth,
                                             -float("inf"), float("inf"))
        self.check()
        return values

    def tie_key(self, action):
        children = self.rows.get((self.state, action), ())
        immediate = min((v for v, _, _ in children), default=-float("inf"))
        repeats = sum(history_entry(out.state, self.me) in self.recent
                      for _, _, out in children)
        return (immediate, -repeats)

    def run(self):
        pos, other = ((self.state.agent_a, self.state.agent_b) if self.me == "A"
                      else (self.state.agent_b, self.state.agent_a))
        roots = get_valid_actions(pos, other, self.state.boxes, self.board)
        roots = [a for a in roots if a not in self.banned] or roots
        if not roots or self.state.is_terminal(self.max_steps):
            return Action.WAIT
        # Legal even if no iteration (or even preference ranking) completes.
        best = roots[0]
        try:
            order = self.actions(self.state, self.me)
            roots.sort(key=lambda a: order.index(a))
            best = roots[0]
            for depth in range(1, self.max_steps - self.state.step + 1):
                values = self.root_iteration(roots, depth)
                best = max(roots, key=lambda a: (values[a], self.tie_key(a)))
                self.root_values = values
                self.completed_depth = depth
                self.iterations.append(dict(depth=depth, action=best.name,
                                            values={a.name:v for a,v in values.items()}))
                roots.sort(key=lambda a: (values[a], self.tie_key(a)), reverse=True)
        except _Timeout:
            pass
        return best

    def diagnostics(self):
        # Cached rows only: no expensive post-deadline evaluation/search.
        result = {}
        for action, value in self.root_values.items():
            replies = []
            for leaf_value, reply, out in self.rows.get((self.state, action), ()):
                replies.append(dict(reply=reply.name, conflict=out.conflict,
                                    executed_a=out.resolved_action_a.name,
                                    executed_b=out.resolved_action_b.name,
                                    immediate_value=leaf_value,
                                    components_a=self.components.get(out.state, {}),
                                    preferences_a=self._logged_preferences(action, reply, "A"),
                                    preferences_b=self._logged_preferences(action, reply, "B")))
            result[action.name] = dict(value=value, replies=replies)
        return result

    def _logged_preferences(self, action, reply, who):
        primary = action if who == self.me else reply
        base = self.prefs.get((self.state, who), ())
        return [a.name for a in ((primary,) + tuple(a for a in base if a != primary)
                                if primary in base else base)]


@_bounded_allocations
def best_action(state, board, max_steps, perspective="A", recent_positions=None,
                tt=None, eval_cache=None, time_limit=TIME_LIMIT, banned_actions=None,
                **kwargs):
    if eval_cache is None:
        eval_cache = kwargs.get("heuristic_cache")
    started = time.monotonic()
    budget = max(0.0, time_limit)
    # Leave a small reserve for returning telemetry and the cached submission.
    deadline = started + max(0.0, budget - min(0.04, budget * 0.1))
    planner = _Planner(state, board, max_steps, perspective, deadline,
                       recent_positions, banned_actions, eval_cache)
    action = planner.run()
    LAST_SEARCH.clear()
    LAST_SEARCH.update(engine="simultaneous-maximin", perspective=perspective,
                       depth=planner.completed_depth, window=planner.completed_depth,
                       nodes=planner.nodes, cache_hits=planner.hits,
                       roots=planner.diagnostics(), iterations=planner.iterations,
                       preferences={who: [a.name for a in planner.prefs.get((state, who), ())]
                                    for who in ("A", "B")},
                       time=time.monotonic()-started)
    return action


class AgentA:
    perspective = "A"

    def __init__(self, time_limit=TIME_LIMIT):
        self.time_limit = time_limit
        self._history = deque(maxlen=6)
        self.tt = {}  # accepted historical API; search tables are per decision
        self.heuristic_cache = {}
        self._last_action = None
        self.last_search = {}
        self._submission = None

    @_bounded_allocations
    def choose_action(self, state, board, max_steps, banned_actions=None):
        started = time.monotonic()
        # Cache the live preference list inside the decision's budget.
        prefs = round_preferences(state, board, max_steps, self.perspective)
        action = best_action(state, board, max_steps, self.perspective,
                             self._history, self.tt, self.heuristic_cache,
                             max(0.0, self.time_limit-(time.monotonic()-started)),
                             banned_actions)
        if action in prefs:
            prefs = (action,) + tuple(a for a in prefs if a != action)
        self._submission = (state, board.serial, max_steps, action, prefs)
        self.last_search = dict(LAST_SEARCH)
        self.last_search["time"] = time.monotonic()-started
        self._history.append(history_entry(state, self.perspective))
        self._last_action = action
        return action

    def preference_list(self, state, board, max_steps, primary=None):
        if self._submission and self._submission[:4] == (state, board.serial, max_steps, primary):
            return self._submission[4]
        return round_preferences(state, board, max_steps, self.perspective, primary)

"""
Agent B controller.

The normal search is identical to Agent A, with the opposite perspective.
After two complete repetitions of a position cycle, B alone yields by asking
that same search to choose an alternative legal root move for this round.
"""

from src.competitive.agent_a import AgentA, TIME_LIMIT, _key
from src.competitive.transition import get_valid_actions


class AgentB(AgentA):
    """Agent B controller."""

    perspective = "B"

    def __init__(self, time_limit=TIME_LIMIT):
        super().__init__(time_limit)
        self._loop_states = []
        self._loop_actions = []
        self._loop_context = None
        self._loop_step = None

    def _observe_cycle(self, state, board, max_steps):
        # Compare physical positions AND ownership, not remaining time: a
        # repetition consumes rounds. Box/credit changes cannot masquerade as
        # a loop unless the entire configuration repeats twice too.
        position = _key(state)[:-1]
        context = (board.serial, max_steps)
        repeated_call = (context == self._loop_context
                         and state.step == self._loop_step
                         and self._loop_states
                         and position == self._loop_states[-1])
        if not repeated_call:
            if (context != self._loop_context or self._loop_step is None
                    or state.step != self._loop_step + 1):
                # New game, rewind, skipped observations, or changed state at
                # the same round: no evidence of consecutive executed loops.
                self._loop_states.clear()
                self._loop_actions.clear()
            self._loop_states.append(position)
            self._loop_actions.append(None)
        self._loop_context, self._loop_step = context, state.step
        history = self._loop_states
        for period in range(1, (len(history) - 1) // 2 + 1):
            if history[-1] != history[-period - 1]:
                continue
            if (history[-2 * period - 1:-period]
                    == history[-period - 1:]):
                return period
        return 0

    def choose_action(self, state, board, max_steps, banned_actions=None):
        period = self._observe_cycle(state, board, max_steps)
        exclusions = set(banned_actions or ())
        yielding = set()
        if period and not state.is_terminal(max_steps):
            legal = get_valid_actions(state.agent_b, state.agent_a, state.boxes, board)
            available = [a for a in legal if a not in exclusions] or legal
            cycle_cells = {s[1] for s in self._loop_states[-period - 1:]}
            previous = self._loop_actions[-period - 1]

            def destination(action):
                dx, dy = action.value
                return (state.agent_b[0] + dx, state.agent_b[1] + dy)

            # Prefer genuinely leaving the cycle and yielding A's current
            # square. If that is impossible, at least change the submission
            # that repeated the cycle (also handles conflict-induced stalls).
            alternatives = [a for a in available
                            if a != previous and destination(a) not in cycle_cells
                            and destination(a) != state.agent_a]
            if not alternatives:
                alternatives = [a for a in available if a != previous]
            if alternatives:
                yielding = set(available) - set(alternatives)
                exclusions.update(yielding)

        # One ordinary search, with a one-round restriction only after the
        # trigger. It may choose a costly push/sacrifice; evaluation, backups,
        # deadline and primary-pinned fallback submission are unchanged.
        action = super().choose_action(state, board, max_steps, exclusions)
        self._loop_actions[-1] = action
        self.last_search["loop_breaker"] = {
            "active": bool(yielding), "period": period,
            "excluded": sorted(a.name for a in yielding),
        }
        return action

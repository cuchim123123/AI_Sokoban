"""Per-round turn computation and result application."""


import pygame
import time

from src.competitive.state import Action
from src.competitive.transition import (
    resolve_joint_action_outcome, get_valid_actions, conflict_winner,
)

from src.shared.common import (
    TILE,
)






class GameMixin:
    """Per-round turn computation and result application."""

    def _compute_step(self):
        if self.state.is_terminal(self.max_steps):
            self.pending_out = "TERMINAL"
            return
            
        # No key submission is possible for a physically boxed-in human.
        for who in ("a", "b"):
            if getattr(self, "agent_" + who) is None:
                pos = self.state.agent_a if who == "a" else self.state.agent_b
                other = self.state.agent_b if who == "a" else self.state.agent_a
                if not get_valid_actions(pos, other, self.state.boxes, self.board):
                    setattr(self, "pending_human_" + who, Action.WAIT)

        expected_step = self.state.step

        t0 = time.time()
        action_a = self.agent_a.choose_action(self.state, self.board, self.max_steps) if self.agent_a else None
        dt_a = time.time() - t0

        t1 = time.time()
        action_b = self.agent_b.choose_action(self.state, self.board, self.max_steps) if self.agent_b else None
        dt_b = time.time() - t1
        
        while (not self.agent_a and self.pending_human_a is None) or \
              (not self.agent_b and self.pending_human_b is None):
            if not self.running or not self.computing or self.state.step != expected_step:
                return
            time.sleep(0.01)

        action_a = action_a if self.agent_a else self.pending_human_a
        action_b = action_b if self.agent_b else self.pending_human_b

        # Rule 3: submit each AGENT's preference list for this round, the
        # chosen action pinned first - the exact root list the search
        # ranked with. Human sides (or agents without the method) submit
        # no list and fall back to the shared starting-state preference policy.
        prefs_a = (
            self.agent_a.preference_list(
                self.state, self.board, self.max_steps, primary=action_a
            )
            if self.agent_a is not None and hasattr(
                self.agent_a, "preference_list"
            )
            else None
        )
        prefs_b = (
            self.agent_b.preference_list(
                self.state, self.board, self.max_steps, primary=action_b
            )
            if self.agent_b is not None and hasattr(
                self.agent_b, "preference_list"
            )
            else None
        )
        out = resolve_joint_action_outcome(
            self.state, action_a, action_b, self.board, self.max_steps,
            prefs_a=prefs_a, prefs_b=prefs_b,
        )
        self.pending_out = (expected_step, action_a, action_b, out, dt_a, dt_b)

    def _apply_computed_step(self):
        self.computing = False
        
        self.pending_human_a = None
        self.pending_human_b = None
        
        if self.pending_out == "TERMINAL":
            self.finished = True
            self.running = False
            self.finish_time = pygame.time.get_ticks()
            self.pending_out = None
            return
            
        step_idx, action_a, action_b, out, dt_a, dt_b = self.pending_out
        self.pending_out = None
        
        if step_idx != self.state.step:
            return
        
        if self.agent_a:
            self.agent_a._last_action = out.resolved_action_a
        if self.agent_b:
            self.agent_b._last_action = out.resolved_action_b
        
        self.prev_state = self.state
        self.state = out.state
        self.anim_t = 0.0

        # Conflict bookkeeping (UI display only): record which side held
        # parity priority for the round that just resolved, keyed by the
        # resulting state's step so history scrubbing stays consistent.
        # Drop stale entries from an abandoned timeline when branching.
        self.step_conflicts = {
            k: v for k, v in self.step_conflicts.items() if k <= self.state.step
        }
        self.conflict_badge = None
        if out.conflict:
            winner = conflict_winner(self.max_steps, step_idx)
            self.step_conflicts[self.state.step] = winner
            wcell = self.state.agent_a if winner == "A" else self.state.agent_b
            self.conflict_badge = {
                "cell": wcell, "winner": winner,
                "life": 80, "max_life": 80,
            }

        import random
        for nb in (self.state.boxes - self.prev_state.boxes):
            # Spawn dust particles for pushed box
            for _ in range(8):
                self.particles.append({
                    "pos": [nb[0] * TILE + TILE//2, nb[1] * TILE + TILE - 5],
                    "vx": random.uniform(-1.5, 1.5),
                    "vy": random.uniform(-1, 0),
                    "life": 30,
                    "max_life": 30
                })
        
        prev_goals = len(self.prev_state.boxes & self.board.goals)
        curr_goals = len(self.state.boxes & self.board.goals)
        if curr_goals > prev_goals:
            self.screen_shake = 15
            for g in (self.state.boxes & self.board.goals) - (self.prev_state.boxes & self.board.goals):
                self.floating_texts.append({
                    "pos": [g[0] * TILE + TILE//2, g[1] * TILE],
                    "life": 60,
                    "max_life": 60
                })
        metric = (self.state.step, action_a, action_b, dt_a, dt_b)
        self._metrics = self._metrics[:self.step_index]
        self._metrics.append(metric)

        self.history = self.history[:self.step_index + 1]
        self.history.append((self.state, metric))
        self.step_index += 1

        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False
            self.finish_time = pygame.time.get_ticks()


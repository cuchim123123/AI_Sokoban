"""Keyboard/mouse input handling."""


import pygame
import sys

from src.competitive.state import Action
from src.competitive.transition import (
    get_valid_actions,
)







class EventsMixin:
    """Keyboard/mouse input handling."""

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if self.in_menu:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._back()
                        return
                    if event.key == pygame.K_RETURN:
                        self._start_game()
                        return
                for b in self.menu_buttons:
                    b.handle_event(event)
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_ESCAPE:
                        self._to_menu()
                        continue
                    
                    if event.key == pygame.K_SPACE:
                        if self.finished and self.step_index == len(self.history) - 1:
                            self._to_menu()
                        else:
                            self.running = not self.running
                            self._last_step_time = pygame.time.get_ticks()
                    elif event.key == pygame.K_m:
                        self._to_menu()
                    elif event.key == pygame.K_r:
                        self._start_game()
                    elif event.key == pygame.K_COMMA:
                        self.running = False
                        self.computing = False
                        self.pending_out = None
                        if self.step_index > 0:
                            self.step_index -= 1
                            self.prev_state = self.history[self.step_index + 1][0]
                            self.state, metric = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.finished = self.state.is_terminal(self.max_steps)
                            if metric:
                                self._metrics = [m for _, m in self.history[1:self.step_index+1]]
                            else:
                                self._metrics = []
                    elif event.key == pygame.K_PERIOD:
                        self.running = False
                        self.computing = False
                        self.pending_out = None
                        if self.step_index < len(self.history) - 1:
                            self.prev_state = self.state
                            self.step_index += 1
                            self.state, metric = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.finished = self.state.is_terminal(self.max_steps)
                            self._metrics = [m for _, m in self.history[1:self.step_index+1]]

                    
                    if self.running and not self.finished and self.anim_t >= 1.0:
                        if self.agent_a is None:
                            if event.key == pygame.K_w: self.pending_human_a = Action.NORTH
                            elif event.key == pygame.K_s: self.pending_human_a = Action.SOUTH
                            elif event.key == pygame.K_a: self.pending_human_a = Action.WEST
                            elif event.key == pygame.K_d: self.pending_human_a = Action.EAST
                        
                        if self.agent_b is None:
                            if event.key == pygame.K_UP: self.pending_human_b = Action.NORTH
                            elif event.key == pygame.K_DOWN: self.pending_human_b = Action.SOUTH
                            elif event.key == pygame.K_LEFT: self.pending_human_b = Action.WEST
                            elif event.key == pygame.K_RIGHT: self.pending_human_b = Action.EAST

                        for who in ("a", "b"):
                            pos = self.state.agent_a if who == "a" else self.state.agent_b
                            other = self.state.agent_b if who == "a" else self.state.agent_a
                            legal = get_valid_actions(pos, other, self.state.boxes, self.board)
                            pending = getattr(self, "pending_human_" + who)
                            if not legal and getattr(self, "agent_" + who) is None:
                                setattr(self, "pending_human_" + who, Action.WAIT)
                            elif pending is not None and pending not in legal:
                                setattr(self, "pending_human_" + who, None)


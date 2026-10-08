"""Keyboard/mouse input handling."""


import sys

import pygame







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
                        self._back_to_launcher()
                        return
                    if event.key == pygame.K_RETURN:
                        self._start_game()
                        return
                for b in self.menu_buttons:
                    b.handle_event(event)
                continue

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self._to_menu()
                elif event.key == pygame.K_SPACE:
                    if self.finished and self.step_index == len(self.history) - 1:
                        self._to_menu()
                    else:
                        self.running = not self.running
                        self._last_step_time = pygame.time.get_ticks()
                elif event.key == pygame.K_r:
                    # Replay the found solution from the start.
                    if not self.solving and len(self.history) > 1:
                        self.step_index = 0
                        self.prev_state = self.history[0]
                        self.state = self.history[0]
                        self.anim_t = 1.0
                        self.finished = False
                        self.running = True
                        self.screen_shake = 0
                        self.particles = []
                        self.floating_texts = []
                        self._last_step_time = pygame.time.get_ticks()
                elif event.key == pygame.K_COMMA:
                    self.running = False
                    self._rewind()
                elif event.key == pygame.K_PERIOD:
                    self.running = False
                    if self.step_index < len(self.history) - 1:
                        self._advance()


import re

with open("src/competitive/gui/app.py", "r") as f:
    code = f.read()

# 1. Add set_repeat
code = code.replace(
    'pygame.display.set_caption("Sokoban — Competitive Mode")',
    'pygame.display.set_caption("Sokoban — Competitive Mode")\n        pygame.key.set_repeat(200, 50)'
)

# 2. Add history init in _start_game
code = code.replace(
    'self.state = self.initial_state\n        self.prev_state = self.initial_state',
    'self.state = self.initial_state\n        self.prev_state = self.initial_state\n        self.box_owners = {}\n        self.history = [(self.state, self.box_owners, None)]\n        self.step_index = 0'
)

# 3. Add Left/Right handlers in _handle_events
events_replacement = """                    elif event.key == pygame.K_m:
                        self.in_menu = True
                        self.screen = pygame.display.set_mode((1024, 768))
                    elif event.key == pygame.K_r:
                        self._start_game()
                    elif event.key == pygame.K_LEFT and not self.running:
                        if self.step_index > 0:
                            self.step_index -= 1
                            self.prev_state = self.history[self.step_index + 1][0]
                            self.state, self.box_owners, metric = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.finished = self.state.is_terminal(self.max_steps)
                            if metric:
                                self._metrics = [m for _, _, m in self.history[1:self.step_index+1]]
                            else:
                                self._metrics = []
                    elif event.key == pygame.K_RIGHT and not self.running:
                        if self.step_index < len(self.history) - 1:
                            self.prev_state = self.state
                            self.step_index += 1
                            self.state, self.box_owners, metric = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.finished = self.state.is_terminal(self.max_steps)
                            self._metrics = [m for _, _, m in self.history[1:self.step_index+1]]
"""
code = code.replace(
    'elif event.key == pygame.K_m:\n                        self.in_menu = True\n                        self.screen = pygame.display.set_mode((1024, 768))\n                    elif event.key == pygame.K_r:\n                        self._start_game()',
    events_replacement
)

# 4. Modify space behavior to let you unpause if finished was true but is now false
space_replacement = """                    if event.key == pygame.K_SPACE:
                        if self.finished and self.step_index == len(self.history) - 1:
                            self.in_menu = True
                            self.screen = pygame.display.set_mode((1024, 768))
                        else:
                            self.running = not self.running
                            self._last_step_time = pygame.time.get_ticks()"""
code = code.replace(
    'if event.key == pygame.K_SPACE:\n                        if self.finished:\n                            self.in_menu = True\n                            self.screen = pygame.display.set_mode((1024, 768))\n                        else:\n                            self.running = not self.running\n                            self._last_step_time = pygame.time.get_ticks()',
    space_replacement
)


# 5. Modify _apply_computed_step to track history
apply_replacement = """        self.box_owners = new_box_owners

        metric = (self.state.step, action_a, action_b, dt_a, dt_b)
        self._metrics = self._metrics[:self.step_index]
        self._metrics.append(metric)

        self.history = self.history[:self.step_index + 1]
        self.history.append((self.state, self.box_owners, metric))
        self.step_index += 1

        if self.state.is_terminal(self.max_steps):"""

code = code.replace(
    'self.box_owners = new_box_owners\n\n        self._metrics.append((self.state.step, action_a, action_b, dt_a, dt_b))\n\n        if self.state.is_terminal(self.max_steps):',
    apply_replacement
)

with open("src/competitive/gui/app.py", "w") as f:
    f.write(code)

print("Patched successfully!")

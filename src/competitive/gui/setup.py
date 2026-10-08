"""Menu/setup screen: buttons, options, game start, navigation."""


import pygame
import sys
import os

from src.competitive.parser import parse_competitive_map
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB

from src.shared.common import (
    TILE,
    UI_H,
    create_gradient_surface,
    Button,
)






class SetupMixin:
    """Menu/setup screen: buttons, options, game start, navigation."""

    def _create_gradient_bg(self):
        self.bg_surface = create_gradient_surface(1024, 768, (10, 15, 30), (50, 20, 80))

    def _setup_menu_buttons(self):
        self.menu_buttons = []
        
        # Map list buttons
        start_y = 170
        for m in self.available_maps:
            m_name = os.path.basename(m)
            b = Button(70, start_y, 260, 50, m_name, lambda m=m: self._select_map(m))
            self.menu_buttons.append(b)
            start_y += 65

        # Settings buttons
        self.btn_ai_a = Button(430, 520, 245, 50, f"Agent A: {self.ai_a_type.upper()}", self._toggle_ai_a)
        self.btn_ai_b = Button(695, 520, 245, 50, f"Agent B: {self.ai_b_type.upper()}", self._toggle_ai_b)
        
        self.btn_steps_dn = Button(700, 450, 110, 45, "Steps -5", self._dec_steps)
        self.btn_steps_up = Button(830, 450, 110, 45, "Steps +5", self._inc_steps)
        
        # Start game button spanning width (primary action)
        self.btn_start = Button(430, 600, 510, 70, "START GAME", self._start_game,
                                selected=True, accent=(120, 220, 120))
        # Back to the main menu (or quit when running standalone)
        self.btn_back = Button(70, 640, 260, 48, "< BACK", self._back)

        self.menu_buttons.extend([self.btn_ai_a, self.btn_ai_b, self.btn_steps_up,
                                  self.btn_steps_dn, self.btn_start, self.btn_back])

    def _back(self):
        """Leave the competitive menu: back to the launcher, or quit."""
        if self.launcher:
            self._exit_requested = True
        else:
            pygame.quit()
            sys.exit()

    def _to_menu(self):
        self.in_menu = True
        self.running = False
        self.screen = pygame.display.set_mode((1024, 768))

    def _select_map(self, m):
        self.map_file = m

    def _toggle_ai_a(self):
        self.ai_a_type = "AI" if self.ai_a_type == "human" else "human"
        self.btn_ai_a.text = f"Agent A: {self.ai_a_type.upper()}"

    def _toggle_ai_b(self):
        self.ai_b_type = "AI" if self.ai_b_type == "human" else "human"
        self.btn_ai_b.text = f"Agent B: {self.ai_b_type.upper()}"

    def _inc_steps(self):
        self.max_steps += 5
    
    def _dec_steps(self):
        self.max_steps = max(5, self.max_steps - 5)

    def _start_game(self):
        self.in_menu = False
        
        self.agent_a = AgentA() if self.ai_a_type == "AI" else None
        self.agent_b = AgentB() if self.ai_b_type == "AI" else None
        
        self.initial_state, self.board = parse_competitive_map(self.map_file)
        self.state = self.initial_state
        self.prev_state = self.initial_state
        self.history = [(self.state, None)]
        self.step_index = 0
        self.anim_t = 1.0
        self.screen_shake = 0
        self.particles = []
        self.floating_texts = []
        self.step_conflicts = {}
        self.step_loop_breaks = {}
        self.conflict_badge = None
        
        self.computing = False
        self.pending_out = None
        
        self.finished = False
        self.running = True
        self.finish_time = 0
        self._metrics = []
        self.pending_human_a = None
        self.pending_human_b = None
        self._last_step_time = pygame.time.get_ticks()

        # Resize screen to fit game
        w = self.board.width * TILE
        h = self.board.height * TILE + UI_H
        self.screen_w = max(w, 800)
        self.screen_h = max(h, 600)
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))

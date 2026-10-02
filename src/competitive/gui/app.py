"""
Competitive Sokoban GUI — Pygame visualiser for the two-agent game.

Controls:
    SPACE       Start / pause the game
    R           Reset to initial state
    ESC / Q     Quit

The game steps are run automatically at a configurable speed (STEP_DELAY ms).
Both agents decide simultaneously; their actions are resolved and the board
is updated before the next frame is drawn.

Box colour coding (Requirement 8):
    Grey    = uncredited (not on any goal)
    Blue    = on a goal, credited to Agent A
    Orange  = on a goal, credited to Agent B
    Gold    = on a goal but credit system shows both (shouldn't happen — sanity)
"""

import pygame
import sys
import time
import os

from src.competitive.state import Board, CompetitiveState, Action
from src.competitive.parser import parse_competitive_map
from src.competitive.transition import resolve_joint_action
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB

# ── Layout constants ──────────────────────────────────────────────────────────
TILE  = 64          # pixels per tile
UI_H  = 110         # height of info bar at the bottom
FPS   = 60
STEP_DELAY = 800    # ms between automatic game steps

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG         = (28,  28,  36)
C_WALL       = (80,  80,  95)
C_FLOOR      = (45,  45,  58)
C_GOAL       = (60,  90,  60)
C_BOX        = (130, 130, 145)   # uncredited
C_BOX_A      = (70,  130, 210)   # credited to A (blue)
C_BOX_B      = (220, 120,  50)   # credited to B (orange)
C_AGENT_A    = (100, 180, 255)
C_AGENT_B    = (255, 160,  60)
C_UI_BG      = (20,  20,  28)
C_TEXT       = (230, 230, 240)
C_TEXT_DIM   = (130, 130, 150)
C_WIN_A      = (100, 180, 255)
C_WIN_B      = (255, 160,  60)
C_WIN_DRAW   = (200, 200, 100)


class CompetitiveApp:
    def __init__(self, map_file: str, max_steps: int):
        pygame.init()
        pygame.display.set_caption("Sokoban — Competitive Mode")

        self.map_file  = map_file
        self.max_steps = max_steps
        self.agent_a   = AgentA()
        self.agent_b   = AgentB()

        self._load_map()

        w = self.board.width  * TILE
        h = self.board.height * TILE + UI_H
        self.screen = pygame.display.set_mode((w, h))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 20)
        self.font_sm = pygame.font.SysFont("Arial", 16)

        self.running  = False   # paused until SPACE
        self.finished = False
        self._last_step_time = 0
        self._metrics: list = []   # list of (step, action_a, action_b, dt_a, dt_b)

    # ── Map loading ────────────────────────────────────────────────────────────

    def _load_map(self):
        self.initial_state, self.board = parse_competitive_map(self.map_file)
        self.state = self.initial_state
        self.finished = False
        self.running = False
        self._metrics = []

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        while True:
            self._handle_events()

            now = pygame.time.get_ticks()
            if self.running and not self.finished:
                if now - self._last_step_time >= STEP_DELAY:
                    self._step()
                    self._last_step_time = now

            self._draw()
            self.clock.tick(FPS)

    # ── Event handling ─────────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    if self.finished:
                        self._load_map()
                    else:
                        self.running = not self.running
                        self._last_step_time = pygame.time.get_ticks()
                elif event.key == pygame.K_r:
                    self._load_map()

    # ── Game step ──────────────────────────────────────────────────────────────

    def _step(self):
        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False
            return

        # Both agents decide simultaneously
        t0 = time.time()
        action_a = self.agent_a.choose_action(self.state, self.board, self.max_steps)
        dt_a = time.time() - t0

        t1 = time.time()
        action_b = self.agent_b.choose_action(self.state, self.board, self.max_steps)
        dt_b = time.time() - t1

        self.state = resolve_joint_action(self.state, action_a, action_b, self.board)
        self._metrics.append((self.state.step, action_a, action_b, dt_a, dt_b))

        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(C_BG)
        self._draw_board()
        self._draw_ui()
        if self.finished:
            self._draw_result_overlay()
        pygame.display.flip()

    def _draw_board(self):
        board = self.board
        state = self.state

        for y in range(board.height):
            for x in range(board.width):
                rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                if (x, y) in board.walls:
                    pygame.draw.rect(self.screen, C_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, C_FLOOR, rect)
                    if (x, y) in board.goals:
                        inner = rect.inflate(-TILE // 2, -TILE // 2)
                        pygame.draw.rect(self.screen, C_GOAL, inner, border_radius=4)

        # Boxes
        for bx, by in state.boxes:
            rect = pygame.Rect(bx * TILE + 6, by * TILE + 6, TILE - 12, TILE - 12)
            pos = (bx, by)
            if pos in state.boxes_on_goals_a:
                col = C_BOX_A
            elif pos in state.boxes_on_goals_b:
                col = C_BOX_B
            else:
                col = C_BOX
            pygame.draw.rect(self.screen, col, rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255, 60), rect, 2, border_radius=8)

        # Agent B (draw first so A appears on top when overlapping)
        bx2, by2 = state.agent_b
        r2 = pygame.Rect(bx2 * TILE + 10, by2 * TILE + 10, TILE - 20, TILE - 20)
        pygame.draw.ellipse(self.screen, C_AGENT_B, r2)
        lbl = self.font_sm.render("B", True, (0, 0, 0))
        self.screen.blit(lbl, lbl.get_rect(center=r2.center))

        # Agent A
        ax, ay = state.agent_a
        r1 = pygame.Rect(ax * TILE + 10, ay * TILE + 10, TILE - 20, TILE - 20)
        pygame.draw.ellipse(self.screen, C_AGENT_A, r1)
        lbl = self.font_sm.render("A", True, (0, 0, 0))
        self.screen.blit(lbl, lbl.get_rect(center=r1.center))

    def _draw_ui(self):
        state = self.state
        board = self.board
        sw = self.screen.get_width()
        ui_top = board.height * TILE
        ui_rect = pygame.Rect(0, ui_top, sw, UI_H)
        pygame.draw.rect(self.screen, C_UI_BG, ui_rect)

        # Score row
        score_txt_a = self.font_lg.render(f"A: {state.score_a()}", True, C_AGENT_A)
        score_txt_b = self.font_lg.render(f"B: {state.score_b()}", True, C_AGENT_B)
        step_txt    = self.font_lg.render(
            f"Step {state.step} / {self.max_steps}", True, C_TEXT
        )
        self.screen.blit(score_txt_a, (14, ui_top + 10))
        self.screen.blit(score_txt_b, (14 + score_txt_a.get_width() + 30, ui_top + 10))
        self.screen.blit(step_txt, (sw - step_txt.get_width() - 14, ui_top + 10))

        # Last action row
        if self._metrics:
            _, act_a, act_b, dt_a, dt_b = self._metrics[-1]
            last_txt = self.font_sm.render(
                f"A→{act_a}  B→{act_b}   "
                f"time A: {dt_a*1000:.0f}ms  B: {dt_b*1000:.0f}ms",
                True, C_TEXT_DIM
            )
            self.screen.blit(last_txt, (14, ui_top + 50))

        # Controls
        if not self.running and not self.finished:
            ctrl = self.font_sm.render("SPACE = start  |  R = reset  |  Q = quit", True, C_TEXT_DIM)
        elif self.running:
            ctrl = self.font_sm.render("SPACE = pause  |  R = reset  |  Q = quit", True, C_TEXT_DIM)
        else:
            ctrl = self.font_sm.render("SPACE = restart  |  R = reset  |  Q = quit", True, C_TEXT_DIM)
        self.screen.blit(ctrl, (14, ui_top + 76))

    def _draw_result_overlay(self):
        state = self.state
        sa, sb = state.score_a(), state.score_b()
        sw, sh = self.screen.get_size()
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        if sa > sb:
            msg, col = "Agent A Wins!", C_WIN_A
        elif sb > sa:
            msg, col = "Agent B Wins!", C_WIN_B
        else:
            msg, col = "Draw!", C_WIN_DRAW

        txt = self.font_lg.render(msg, True, col)
        sub = self.font_md.render(f"Final score — A: {sa}   B: {sb}", True, C_TEXT)
        hint = self.font_sm.render("Press SPACE to play again", True, C_TEXT_DIM)

        cx = sw // 2
        cy = (self.board.height * TILE) // 2
        self.screen.blit(txt, txt.get_rect(center=(cx, cy - 30)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 14)))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 48)))

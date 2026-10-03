"""
Competitive Sokoban GUI — Pygame visualiser for the two-agent game.
"""

import pygame
import sys
import time
import os

from src.competitive.state import Board, CompetitiveState, Action
from src.competitive.parser import parse_competitive_map
from src.competitive.transition import resolve_joint_action_outcome
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB

# ── Config ──
TILE = 40
UI_H = 100
FPS  = 60
STEP_DELAY = 150  # ms between steps for AI

# ── Colors ──
C_BG       = (30, 30, 30)
C_WALL     = (100, 100, 100)
C_FLOOR    = (40, 40, 40)
C_GOAL     = (60, 60, 60)
C_BOX      = (200, 180, 140)
C_BOX_DONE = (140, 200, 140)

C_AGENT_A  = (255, 100, 100)
C_AGENT_B  = (100, 150, 255)
C_A_DONE   = (200,  50,  50)
C_B_DONE   = ( 50, 100, 200)

C_UI_BG    = (20, 20, 20)
C_TEXT     = (220, 220, 220)
C_TEXT_DIM = (120, 120, 120)

C_WIN_A    = (255, 150, 150)
C_WIN_B    = (150, 200, 255)
C_WIN_DRAW = (200, 200, 200)


class CompetitiveApp:
    def __init__(self, map_file: str, max_steps: int, ai_a: str = "aggressive", ai_b: str = "aggressive"):
        pygame.init()
        pygame.display.set_caption("Sokoban — Competitive Mode")

        self.map_file  = map_file
        self.max_steps = max_steps
        self.ai_a_type = ai_a
        self.ai_b_type = ai_b
        
        # Instantiate agents (None if human)
        self.agent_a = AgentA(ai_a) if ai_a != "human" else None
        self.agent_b = AgentB(ai_b) if ai_b != "human" else None

        self._load_map()

        w = self.board.width  * TILE
        h = self.board.height * TILE + UI_H
        self.screen = pygame.display.set_mode((w, h))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 20)
        self.font_sm = pygame.font.SysFont("Arial", 16)

        self.running  = False
        self.finished = False
        self._last_step_time = 0
        self._metrics: list = []
        
        # Human Input State
        self.pending_human_a = None
        self.pending_human_b = None
        self.conflict_state = False
        self.conflict_action_a = None
        self.conflict_action_b = None

    def _load_map(self):
        self.initial_state, self.board = parse_competitive_map(self.map_file)
        self.state = self.initial_state
        self.finished = False
        self.running = False
        self._metrics = []
        self.pending_human_a = None
        self.pending_human_b = None
        self.conflict_state = False

    def run(self):
        while True:
            self._handle_events()

            now = pygame.time.get_ticks()
            if self.running and not self.finished and not self.conflict_state:
                if self.agent_a is None or self.agent_b is None:
                    # If any human, don't use timer delay, wait for input
                    self._step()
                elif now - self._last_step_time >= STEP_DELAY:
                    self._step()
                    self._last_step_time = now

            self._draw()
            self.clock.tick(FPS)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                
                # Conflict resolution overrides
                if self.conflict_state:
                    if event.key == pygame.K_a:
                        self._apply_conflict_resolution('A')
                    elif event.key == pygame.K_b:
                        self._apply_conflict_resolution('B')
                    continue

                if event.key == pygame.K_SPACE:
                    if self.finished:
                        self._load_map()
                    else:
                        self.running = not self.running
                        self._last_step_time = pygame.time.get_ticks()
                elif event.key == pygame.K_r:
                    self._load_map()
                
                # Human inputs
                if self.running and not self.finished and not self.conflict_state:
                    # Player A: WASD + Left Shift (Wait)
                    if self.agent_a is None:
                        if event.key == pygame.K_w: self.pending_human_a = Action.NORTH
                        elif event.key == pygame.K_s: self.pending_human_a = Action.SOUTH
                        elif event.key == pygame.K_a: self.pending_human_a = Action.WEST
                        elif event.key == pygame.K_d: self.pending_human_a = Action.EAST
                        elif event.key == pygame.K_LSHIFT: self.pending_human_a = Action.WAIT
                    
                    # Player B: Arrows + Right Shift (Wait)
                    if self.agent_b is None:
                        if event.key == pygame.K_UP: self.pending_human_b = Action.NORTH
                        elif event.key == pygame.K_DOWN: self.pending_human_b = Action.SOUTH
                        elif event.key == pygame.K_LEFT: self.pending_human_b = Action.WEST
                        elif event.key == pygame.K_RIGHT: self.pending_human_b = Action.EAST
                        elif event.key == pygame.K_RSHIFT: self.pending_human_b = Action.WAIT

    def _apply_conflict_resolution(self, yielded_agent: str):
        act_a = self.conflict_action_a
        act_b = self.conflict_action_b
        
        if yielded_agent == 'A':
            if self.agent_a is not None:
                act_a = self.agent_a.choose_action(self.state, self.board, self.max_steps, banned_actions=[self.conflict_action_a])
            else:
                act_a = Action.WAIT
        elif yielded_agent == 'B':
            if self.agent_b is not None:
                act_b = self.agent_b.choose_action(self.state, self.board, self.max_steps, banned_actions=[self.conflict_action_b])
            else:
                act_b = Action.WAIT
                
        out = resolve_joint_action_outcome(self.state, act_a, act_b, self.board)
        
        if out.conflict and act_a != Action.WAIT and act_b != Action.WAIT:
            self.conflict_state = True
            self.conflict_action_a = act_a
            self.conflict_action_b = act_b
            return
            
        self.conflict_state = False
        self.state = out.state
        self._metrics.append((self.state.step, act_a, act_b, 0.0, 0.0))
        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running = False
        self._last_step_time = pygame.time.get_ticks()

    def _step(self):
        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False
            return
            
        if self.agent_a is None and self.pending_human_a is None:
            return
        if self.agent_b is None and self.pending_human_b is None:
            return

        t0 = time.time()
        action_a = self.agent_a.choose_action(self.state, self.board, self.max_steps) if self.agent_a else self.pending_human_a
        dt_a = time.time() - t0

        t1 = time.time()
        action_b = self.agent_b.choose_action(self.state, self.board, self.max_steps) if self.agent_b else self.pending_human_b
        dt_b = time.time() - t1

        self.pending_human_a = None
        self.pending_human_b = None

        out = resolve_joint_action_outcome(self.state, action_a, action_b, self.board)
        
        # Detect Conflict
        if out.conflict and action_a != Action.WAIT and action_b != Action.WAIT:
            self.conflict_state = True
            self.conflict_action_a = action_a
            self.conflict_action_b = action_b
            return
            
        self.state = out.state
        self._metrics.append((self.state.step, action_a, action_b, dt_a, dt_b))

        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False

    def _draw(self):
        self.screen.fill(C_BG)
        self._draw_board()
        self._draw_ui()
        if self.finished:
            self._draw_result_overlay()
        elif self.conflict_state:
            self._draw_conflict_overlay()
        pygame.display.flip()
        
    def _draw_conflict_overlay(self):
        sw, sh = self.screen.get_size()
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        cx = sw // 2
        cy = (self.board.height * TILE) // 2
        
        txt = self.font_lg.render("CONFLICT!", True, (255, 50, 50))
        sub = self.font_md.render("Agents collided! Who yields?", True, C_TEXT)
        hint1 = self.font_sm.render("Press 'A' to force Agent A to yield (WAIT)", True, C_AGENT_A)
        hint2 = self.font_sm.render("Press 'B' to force Agent B to yield (WAIT)", True, C_AGENT_B)
        
        self.screen.blit(txt, txt.get_rect(center=(cx, cy - 40)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy - 10)))
        self.screen.blit(hint1, hint1.get_rect(center=(cx, cy + 20)))
        self.screen.blit(hint2, hint2.get_rect(center=(cx, cy + 45)))

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
                
                # Goal marker
                if (x, y) in board.goals:
                    inner = rect.inflate(-TILE//2, -TILE//2)
                    pygame.draw.ellipse(self.screen, C_GOAL, inner)

        # Boxes
        for bx, by in state.boxes:
            rect = pygame.Rect(bx * TILE + 4, by * TILE + 4, TILE - 8, TILE - 8)
            col = C_BOX
            if (bx, by) in state.boxes_on_goals_a:
                col = C_A_DONE
            elif (bx, by) in state.boxes_on_goals_b:
                col = C_B_DONE
            elif (bx, by) in board.goals:
                col = C_BOX_DONE
            pygame.draw.rect(self.screen, col, rect, border_radius=4)
            pygame.draw.rect(self.screen, (20, 20, 20), rect, width=2, border_radius=4)

        # Agents
        self._draw_agent(state.agent_a, "A", C_AGENT_A)
        self._draw_agent(state.agent_b, "B", C_AGENT_B)

    def _draw_agent(self, pos, label, color):
        x, y = pos
        rect = pygame.Rect(x * TILE + 4, y * TILE + 4, TILE - 8, TILE - 8)
        pygame.draw.ellipse(self.screen, color, rect)
        pygame.draw.ellipse(self.screen, (20, 20, 20), rect, width=2)
        
        lbl = self.font_sm.render(label, True, (20, 20, 20))
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_ui(self):
        state = self.state
        board = self.board
        sw = self.screen.get_width()
        ui_top = board.height * TILE
        ui_rect = pygame.Rect(0, ui_top, sw, UI_H)
        pygame.draw.rect(self.screen, C_UI_BG, ui_rect)

        # Score row
        score_txt_a = self.font_lg.render(f"A ({self.ai_a_type}): {state.score_a()}", True, C_AGENT_A)
        score_txt_b = self.font_lg.render(f"B ({self.ai_b_type}): {state.score_b()}", True, C_AGENT_B)
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

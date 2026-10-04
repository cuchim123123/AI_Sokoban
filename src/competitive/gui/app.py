"""
Competitive Sokoban GUI — Pygame visualiser for the two-agent game.
"""

import pygame
import sys
import time
import os
import glob
import threading
import json

from src.competitive.state import Board, CompetitiveState, Action
from src.competitive.parser import parse_competitive_map
from src.competitive.transition import resolve_joint_action_outcome
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB

# ── Config ──
TILE = 64
UI_H = 120
FPS  = 60
STEP_DELAY = 400  # ms between steps for AI
ANIM_DUR = 0.4 # seconds for animation

# ── Colors ──
C_BG       = (30, 30, 30)
C_WALL     = (100, 100, 100)
C_FLOOR    = (40, 40, 40)
C_GOAL     = (60, 160, 60)
C_BOX      = (200, 180, 140)
C_BOX_DONE = (140, 200, 140)

C_AGENT_A  = (255, 100, 100)
C_AGENT_B  = (100, 150, 255)
C_A_DONE   = (200,  50,  50)
C_B_DONE   = ( 50, 100, 200)

C_UI_BG    = (20, 20, 20)
C_TEXT     = (255, 255, 255)
C_TEXT_DIM = (180, 180, 180)

C_WIN_A    = (255, 150, 150)
C_WIN_B    = (150, 200, 255)
C_WIN_DRAW = (200, 200, 200)


def load_placeholder_image(path, size, color, label, tint=None):
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (size, size))
        if tint:
            tint_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            tint_surf.fill((*tint, 255))
            img.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        return img
    except Exception as e:
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0, 0, size, size), border_radius=8)
        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        font = pygame.font.SysFont(font_name, max(10, size // 5), bold=True)
        txt = font.render(label, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=(size // 2, size // 2)))
        return surf


def lerp(a, b, t):
    return a + (b - a) * t

def lerp_pos(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


class Animator:
    def __init__(self, base_dir):
        self.states = ["Running", "Push", "Die", "Uppercut", "Idle"]
        self.animations = {}
        self.frame_dims = {}
        
        # 1: South, 3: West, 5: North, 7: East
        self.dir_map = {
            (0, 1): 1,
            (-1, 0): 3,
            (0, -1): 5,
            (1, 0): 7
        }
        
        for state in self.states:
            self.animations[state] = {}
            self.frame_dims[state] = {}
            base_path = f"{base_dir}/{state}/Businessman_{state}"
            for vec, idx in self.dir_map.items():
                img_path = f"{base_path}_dir{idx}.png"
                json_path = f"{base_path}_dir{idx}.json"
                if os.path.exists(img_path) and os.path.exists(json_path):
                    try:
                        sheet = pygame.image.load(img_path).convert_alpha()
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        
                        # 1. Find union bounding box
                        union_rect = None
                        for f_data in data['frames']:
                            r = f_data['frame']
                            rect = pygame.Rect(r['x'], r['y'], r['w'], r['h'])
                            sub = sheet.subsurface(rect)
                            bbox = sub.get_bounding_rect()
                            if bbox.width > 0 and bbox.height > 0:
                                if union_rect is None:
                                    union_rect = bbox.copy()
                                else:
                                    union_rect.union_ip(bbox)
                                    
                        if union_rect is None:
                            union_rect = pygame.Rect(0, 0, 256, 256)
                        else:
                            union_rect.inflate_ip(4, 4)

                        # 2. Extract and scale cropped frames
                        frames = []
                        scale_factor = (TILE * 0.9) / union_rect.h
                        new_w = int(union_rect.w * scale_factor)
                        new_h = int(union_rect.h * scale_factor)
                        self.frame_dims[state][vec] = (new_w, new_h)
                        
                        for f_data in data['frames']:
                            r = f_data['frame']
                            crop_rect = pygame.Rect(r['x'] + union_rect.x, r['y'] + union_rect.y, union_rect.w, union_rect.h)
                            frame_surf = sheet.subsurface(crop_rect)
                            frame_surf = pygame.transform.scale(frame_surf, (new_w, new_h))
                            frames.append(frame_surf)
                        self.animations[state][vec] = frames
                    except Exception as e:
                        print(f"Failed to load animation {img_path}: {e}")
                        self.animations[state][vec] = None
                else:
                    self.animations[state][vec] = None

    def get_frame(self, state, vec, t, time_ms, start_time_ms=0):
        state_anims = self.animations.get(state, {})
        state_dims = self.frame_dims.get(state, {})
        
        frames = state_anims.get(vec)
        if not frames:
            frames = state_anims.get((0, 1))
            
        if not frames:
            return None, (TILE, TILE)
            
        dims = state_dims.get(vec, state_dims.get((0, 1), (TILE, TILE)))
            
        if state in ["Running", "Push"] and t < 1.0:
            frame_idx = int(t * len(frames)) % len(frames)
            if frame_idx == 0 and t > 0:
                frame_idx = 1
            return frames[frame_idx], dims
        else:
            # Idle, Die, Uppercut -> use absolute time
            elapsed = time_ms - start_time_ms
            frame_idx = (elapsed // 62) % len(frames)
            
            # Clamp end-game animations so they don't loop endlessly
            if state in ["Die", "Uppercut"]:
                if (elapsed // 62) >= len(frames):
                    frame_idx = len(frames) - 1
                    
            return frames[frame_idx], dims

def draw_glass_panel(surface, rect, alpha=40, border_alpha=120, radius=15):
    temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(temp, (255, 255, 255, alpha), temp.get_rect(), border_radius=radius)
    pygame.draw.rect(temp, (255, 255, 255, border_alpha), temp.get_rect(), width=1, border_radius=radius)
    surface.blit(temp, rect.topleft)

class Button:
    def __init__(self, x, y, w, h, text, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        self.font = pygame.font.SysFont(font_name, 20, bold=True)
        self.hovered = False

    def draw(self, screen):
        alpha = 60 if self.hovered else 25
        draw_glass_panel(screen, self.rect, alpha=alpha, border_alpha=150, radius=10)
        
        # Shadow for text
        txt_shadow = self.font.render(self.text, True, (0, 0, 0))
        txt_surf = self.font.render(self.text, True, (255, 255, 255))
        
        c = self.rect.center
        screen.blit(txt_shadow, txt_shadow.get_rect(center=(c[0]+1, c[1]+1)))
        screen.blit(txt_surf, txt_surf.get_rect(center=c))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.action()

class CompetitiveApp:
    def __init__(self, map_file: str, max_steps: int, ai_a: str = "AI", ai_b: str = "AI"):
        pygame.init()
        pygame.display.set_caption("Sokoban — Competitive Mode")

        self.map_file = map_file
        self.max_steps = max_steps
        self.ai_a_type = ai_a if ai_a in ["human", "AI"] else "AI"
        self.ai_b_type = ai_b if ai_b in ["human", "AI"] else "AI"

        self.screen_w = 1024
        self.screen_h = 768
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        
        # Create gradient background surface
        self._create_gradient_bg()

        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        self.font_title = pygame.font.SysFont(font_name, 48, bold=True)
        self.font_lg = pygame.font.SysFont(font_name, 28, bold=True)
        self.font_md = pygame.font.SysFont(font_name, 22)
        self.font_sm = pygame.font.SysFont(font_name, 16)
        # Assets
        base_soko = "src/assets/soko/PNG/Retina"
        self.img_wall = load_placeholder_image(f"{base_soko}/Blocks/block_03.png", TILE, C_WALL, "WALL")
        self.img_floor = load_placeholder_image(f"{base_soko}/Ground/ground_06.png", TILE, C_FLOOR, "")
        self.img_goal = load_placeholder_image(f"{base_soko}/Environment/environment_02.png", TILE, C_GOAL, "GOAL")
        self.img_box = load_placeholder_image(f"{base_soko}/Crates/crate_02.png", TILE, C_BOX, "BOX")
        self.img_box_done_a = load_placeholder_image(f"{base_soko}/Crates/crate_03.png", TILE, C_A_DONE, "BOX A")
        self.img_box_done_b = load_placeholder_image(f"{base_soko}/Crates/crate_04.png", TILE, C_B_DONE, "BOX B")
        self.img_box_a = load_placeholder_image(f"{base_soko}/Crates/crate_03.png", TILE, C_AGENT_A, "BOX A")
        self.img_box_b = load_placeholder_image(f"{base_soko}/Crates/crate_04.png", TILE, C_AGENT_B, "BOX B")
        self.img_agent_a = load_placeholder_image("assets/agent_a.png", TILE, C_AGENT_A, "A")
        self.img_agent_b = load_placeholder_image("assets/agent_b.png", TILE, C_AGENT_B, "B")
        
        self.animator = Animator("src/assets")
        self._last_vecs = {"A": (0, 1), "B": (0, 1)}
        self.finish_time = 0
        self.box_owners = {}

        # Load maps list
        self.available_maps = glob.glob("maps/competitive/*.txt")
        if not self.available_maps:
            self.available_maps = [map_file]

        self.in_menu = True
        self._setup_menu_buttons()

        self.running = False
        self.finished = False
        self._last_step_time = 0
        self._metrics = []
        self.pending_human_a = None
        self.pending_human_b = None

        # Animation state
        self.prev_state = None
        self.anim_t = 1.0
        
        # Threading state
        self.computing = False
        self.pending_out = None

    def _create_gradient_bg(self):
        self.bg_surface = pygame.Surface((1024, 768))
        c1 = pygame.Color(10, 15, 30)
        c2 = pygame.Color(50, 20, 80)
        for y in range(768):
            t = y / 768.0
            r = int(lerp(c1.r, c2.r, t))
            g = int(lerp(c1.g, c2.g, t))
            b = int(lerp(c1.b, c2.b, t))
            pygame.draw.line(self.bg_surface, (r, g, b), (0, y), (1024, y))

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
        
        # Start game button spanning width
        self.btn_start = Button(430, 600, 510, 70, "START GAME", self._start_game)

        self.menu_buttons.extend([self.btn_ai_a, self.btn_ai_b, self.btn_steps_up, self.btn_steps_dn, self.btn_start])

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
        self.box_owners = {}
        self.history = [(self.state, self.box_owners, None)]
        self.step_index = 0
        self.anim_t = 1.0
        
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

    def run(self):
        last_time = time.time()
        while True:
            now_time = time.time()
            dt = min(now_time - last_time, 0.1) 
            last_time = now_time

            self._handle_events()

            if not self.in_menu:
                if self.anim_t < 1.0:
                    self.anim_t = min(1.0, self.anim_t + dt / ANIM_DUR)
                
                now_ticks = pygame.time.get_ticks()
                if self.running and not self.finished and self.anim_t >= 1.0:
                    if not self.computing:
                        if now_ticks - self._last_step_time >= STEP_DELAY:
                            self.computing = True
                            threading.Thread(target=self._compute_step, daemon=True).start()
                
                # Check for thread completion
                if self.computing and self.pending_out is not None:
                    self._apply_computed_step()
                    self._last_step_time = pygame.time.get_ticks()

            self._draw()
            self.clock.tick(FPS)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if self.in_menu:
                for b in self.menu_buttons:
                    b.handle_event(event)
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    
                    if event.key == pygame.K_SPACE:
                        if self.finished and self.step_index == len(self.history) - 1:
                            self.in_menu = True
                            self.screen = pygame.display.set_mode((1024, 768))
                        else:
                            self.running = not self.running
                            self._last_step_time = pygame.time.get_ticks()
                    elif event.key == pygame.K_m:
                        self.in_menu = True
                        self.screen = pygame.display.set_mode((1024, 768))
                    elif event.key == pygame.K_r:
                        self._start_game()
                    elif event.key == pygame.K_COMMA:
                        self.running = False
                        self.computing = False
                        self.pending_out = None
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
                    elif event.key == pygame.K_PERIOD:
                        self.running = False
                        self.computing = False
                        self.pending_out = None
                        if self.step_index < len(self.history) - 1:
                            self.prev_state = self.state
                            self.step_index += 1
                            self.state, self.box_owners, metric = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.finished = self.state.is_terminal(self.max_steps)
                            self._metrics = [m for _, _, m in self.history[1:self.step_index+1]]

                    
                    if self.running and not self.finished and self.anim_t >= 1.0:
                        if self.agent_a is None:
                            if event.key == pygame.K_w: self.pending_human_a = Action.NORTH
                            elif event.key == pygame.K_s: self.pending_human_a = Action.SOUTH
                            elif event.key == pygame.K_a: self.pending_human_a = Action.WEST
                            elif event.key == pygame.K_d: self.pending_human_a = Action.EAST
                            elif event.key == pygame.K_LSHIFT: self.pending_human_a = Action.WAIT
                        
                        if self.agent_b is None:
                            if event.key == pygame.K_UP: self.pending_human_b = Action.NORTH
                            elif event.key == pygame.K_DOWN: self.pending_human_b = Action.SOUTH
                            elif event.key == pygame.K_LEFT: self.pending_human_b = Action.WEST
                            elif event.key == pygame.K_RIGHT: self.pending_human_b = Action.EAST
                            elif event.key == pygame.K_RSHIFT: self.pending_human_b = Action.WAIT

    def _compute_step(self):
        if self.state.is_terminal(self.max_steps):
            self.pending_out = "TERMINAL"
            return
            
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

        out = resolve_joint_action_outcome(self.state, action_a, action_b, self.board, self.max_steps)
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
        
        self.prev_state = self.state
        self.state = out.state
        self.anim_t = 0.0

        new_box_owners = {}
        for b in self.prev_state.boxes:
            if b in self.box_owners and b in self.state.boxes:
                new_box_owners[b] = self.box_owners[b]
                
        for nb in (self.state.boxes - self.prev_state.boxes):
            if self.state.agent_a in self.prev_state.boxes:
                new_box_owners[nb] = "A"
            elif self.state.agent_b in self.prev_state.boxes:
                new_box_owners[nb] = "B"
        self.box_owners = new_box_owners

        metric = (self.state.step, action_a, action_b, dt_a, dt_b)
        self._metrics = self._metrics[:self.step_index]
        self._metrics.append(metric)

        self.history = self.history[:self.step_index + 1]
        self.history.append((self.state, self.box_owners, metric))
        self.step_index += 1

        if self.state.is_terminal(self.max_steps):
            self.finished = True
            self.running  = False
            self.finish_time = pygame.time.get_ticks()

    def _draw(self):
        if self.in_menu:
            self._draw_menu()
        else:
            self.screen.fill(C_BG)
            self._draw_board()
            self._draw_ui()
            if self.finished:
                self._draw_result_overlay()
        pygame.display.flip()
        
    def _draw_menu(self):
        # Background
        self.screen.blit(self.bg_surface, (0, 0))

        # Title
        title_shadow = self.font_title.render("SOKOBAN COMPETITIVE", True, (0, 0, 0))
        title = self.font_title.render("SOKOBAN COMPETITIVE", True, (255, 255, 255))
        self.screen.blit(title_shadow, title_shadow.get_rect(center=(513, 61)))
        self.screen.blit(title, title.get_rect(center=(512, 60)))

        # Left Panel (Map List)
        panel_left = pygame.Rect(50, 110, 300, 600)
        draw_glass_panel(self.screen, panel_left, alpha=20, radius=20)
        
        lbl = self.font_lg.render("Select Map", True, (255, 255, 255))
        self.screen.blit(lbl, (70, 130))

        # Right Panel (Settings & Preview)
        panel_right = pygame.Rect(400, 110, 570, 600)
        draw_glass_panel(self.screen, panel_right, alpha=20, radius=20)
        
        lbl = self.font_lg.render("Game Settings", True, (255, 255, 255))
        self.screen.blit(lbl, (430, 130))

        # Map Preview Area (glass look inside panel)
        preview_rect = pygame.Rect(430, 180, 510, 240)
        draw_glass_panel(self.screen, preview_rect, alpha=40, border_alpha=80, radius=15)
        
        prev_txt = self.font_lg.render("MAP PREVIEW", True, (220, 220, 255))
        self.screen.blit(prev_txt, prev_txt.get_rect(center=(preview_rect.centerx, preview_rect.centery - 15)))
        map_name_txt = self.font_md.render(os.path.basename(self.map_file), True, (255, 255, 255))
        self.screen.blit(map_name_txt, map_name_txt.get_rect(center=(preview_rect.centerx, preview_rect.centery + 20)))
        
        # Max Steps display
        steps_txt = self.font_lg.render(f"Max Steps: {self.max_steps}", True, (255, 255, 255))
        self.screen.blit(steps_txt, (430, 458))

        # Highlight for selected map button
        for b in self.menu_buttons:
            if b.text == os.path.basename(self.map_file):
                # Glowing border behind active button
                pygame.draw.rect(self.screen, (255, 255, 255, 80), b.rect.inflate(8, 8), border_radius=12)
            b.draw(self.screen)

    def _get_box_visual_positions(self, t):
        prev_boxes = self.prev_state.boxes
        next_boxes = self.state.boxes
        
        prev_unmatched = list(prev_boxes - next_boxes)
        next_unmatched = list(next_boxes - prev_boxes)
        
        positions = []
        for b in (prev_boxes & next_boxes):
            positions.append((b, b)) 
            
        for pb in prev_unmatched:
            best_nb = None
            best_dist = 999
            for nb in next_unmatched:
                dist = abs(pb[0]-nb[0]) + abs(pb[1]-nb[1])
                if dist < best_dist:
                    best_dist = dist
                    best_nb = nb
            if best_nb:
                positions.append((pb, best_nb))
                next_unmatched.remove(best_nb)
                
        for nb in next_unmatched:
            positions.append((nb, nb))
            
        res = []
        for pb, nb in positions:
            res.append((lerp_pos(pb, nb, t), nb)) 
        return res

    def _draw_board(self):
        board = self.board
        
        board_w_px = board.width * TILE
        board_h_px = board.height * TILE
        
        offset_x = max(0, (self.screen_w - board_w_px) // 2)
        offset_y = 0

        for y in range(board.height):
            for x in range(board.width):
                px = offset_x + x * TILE
                py = offset_y + y * TILE
                
                self.screen.blit(self.img_floor, (px, py))
                
                if (x, y) in board.walls:
                    self.screen.blit(self.img_wall, (px, py))
                
                if (x, y) in board.goals:
                    self.screen.blit(self.img_goal, (px, py))

        t = self.anim_t
        t = t * t * (3 - 2 * t) 
        
        for (vx, vy), dest in self._get_box_visual_positions(t):
            px = offset_x + vx * TILE
            py = offset_y + vy * TILE
            
            owner = self.box_owners.get(dest)
            img = self.img_box
            if owner == "A":
                img = self.img_box_a
            elif owner == "B":
                img = self.img_box_b
                
            if dest in self.state.boxes_on_goals_a:
                img = self.img_box_done_a
            elif dest in self.state.boxes_on_goals_b:
                img = self.img_box_done_b
            elif dest in board.goals:
                img = self.img_box_done_a 
                
            self.screen.blit(img, (px, py))

        self._draw_agent_anim("A", self.prev_state.agent_a, self.state.agent_a, t, offset_x, offset_y, C_AGENT_A)
        self._draw_agent_anim("B", self.prev_state.agent_b, self.state.agent_b, t, offset_x, offset_y, C_AGENT_B)

    def _draw_agent_anim(self, label, prev_pos, next_pos, t, ox, oy, color):
        ax, ay = lerp_pos(prev_pos, next_pos, t)
        px, py = ox + ax * TILE, oy + ay * TILE
        
        vec = (next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1])
        if vec == (0, 0):
            vec = self._last_vecs.get(label, (0, 1))
        else:
            self._last_vecs[label] = vec
            
        anim_state = "Idle"
        is_pushing = (next_pos in self.prev_state.boxes)
        
        if t < 1.0 and next_pos != prev_pos:
            anim_state = "Push" if is_pushing else "Running"
        elif self.finished:
            sa, sb = self.state.score_a(), self.state.score_b()
            if label == "A":
                anim_state = "Uppercut" if sa >= sb else "Die"
            else:
                anim_state = "Uppercut" if sb >= sa else "Die"

        start_t = self.finish_time if self.finished else 0
        frame, dims = self.animator.get_frame(anim_state, vec, t, pygame.time.get_ticks(), start_t)
        if frame:
            fw, fh = dims
            # Draw shadow/aura specifically under the feet
            pygame.draw.ellipse(self.screen, color, (px + TILE//2 - 15, py + TILE - 10, 30, 10))
            
            # Since the frame is tightly cropped around the character, 
            # place it so the bottom of the frame aligns with the bottom of the tile
            blit_x = px + (TILE - fw) / 2
            blit_y = py + (TILE - fh)
            self.screen.blit(frame, (blit_x, blit_y))
        else:
            pygame.draw.ellipse(self.screen, color, (px + 10, py + TILE - 15, TILE - 20, 10))
            img = self.img_agent_a if label == "A" else self.img_agent_b
            self.screen.blit(img, (px, py))
            
        # Draw label above head
        lbl = self.font_sm.render(label, True, (255, 255, 255))
        lbl_shadow = self.font_sm.render(label, True, (0, 0, 0))
        self.screen.blit(lbl_shadow, lbl_shadow.get_rect(center=(px + TILE//2 + 1, py - 5 + 1)))
        self.screen.blit(lbl, lbl.get_rect(center=(px + TILE//2, py - 5)))

    def _draw_ui(self):
        state = self.state
        ui_top = self.screen_h - UI_H
        ui_rect = pygame.Rect(0, ui_top, self.screen_w, UI_H)
        pygame.draw.rect(self.screen, C_UI_BG, ui_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (0, ui_top), (self.screen_w, ui_top), 2)

        score_txt_a = self.font_lg.render(f"Agent A: {state.score_a()}", True, C_AGENT_A)
        score_txt_b = self.font_lg.render(f"Agent B: {state.score_b()}", True, C_AGENT_B)
        step_txt    = self.font_lg.render(f"Step {state.step} / {self.max_steps}", True, C_TEXT)
        
        self.screen.blit(score_txt_a, (30, ui_top + 15))
        self.screen.blit(score_txt_b, (30 + score_txt_a.get_width() + 40, ui_top + 15))
        self.screen.blit(step_txt, (self.screen_w - step_txt.get_width() - 30, ui_top + 15))

        if self._metrics:
            _, act_a, act_b, dt_a, dt_b = self._metrics[-1]
            act_a_str = act_a.name if hasattr(act_a, 'name') else str(act_a)
            act_b_str = act_b.name if hasattr(act_b, 'name') else str(act_b)
            last_txt = self.font_md.render(
                f"A: {act_a_str} ({dt_a*1000:.0f}ms) | B: {act_b_str} ({dt_b*1000:.0f}ms)",
                True, C_TEXT_DIM
            )
            self.screen.blit(last_txt, (30, ui_top + 55))

        if not self.running and not self.finished:
            ctrl = self.font_sm.render("SPACE=start | [ , ]=scrub | R=restart | M=menu", True, C_TEXT_DIM)
        elif self.running:
            ctrl = self.font_sm.render("SPACE=pause | [ , ]=scrub | R=restart | M=menu", True, C_TEXT_DIM)
        else:
            ctrl = self.font_sm.render("SPACE=unpause | [ , ]=scrub | R=restart | M=menu", True, C_TEXT_DIM)
        self.screen.blit(ctrl, (30, ui_top + 90))

    def _draw_result_overlay(self):
        state = self.state
        sa, sb = state.score_a(), state.score_b()
        
        draw_glass_panel(self.screen, pygame.Rect(0, 0, self.screen_w, self.screen_h), alpha=150, border_alpha=0, radius=0)

        if sa > sb:
            msg, col = "AGENT A WINS!", C_WIN_A
        elif sb > sa:
            msg, col = "AGENT B WINS!", C_WIN_B
        else:
            msg, col = "DRAW!", C_WIN_DRAW

        txt = self.font_lg.render(msg, True, col)
        sub = self.font_md.render(f"Final Score — A: {sa}  |  B: {sb}", True, C_TEXT)
        hint = self.font_md.render("Press SPACE to return to Menu", True, (200, 255, 200))

        cx = self.screen_w // 2
        cy = self.screen_h // 2
        
        # Panel for result
        res_rect = pygame.Rect(0, 0, 400, 200)
        res_rect.center = (cx, cy)
        draw_glass_panel(self.screen, res_rect, alpha=40, border_alpha=150, radius=20)

        self.screen.blit(txt, txt.get_rect(center=(cx, cy - 40)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 10)))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 60)))


if __name__ == "__main__":
    app = CompetitiveApp("maps/competitive/arena_open.txt", 50)
    app.run()

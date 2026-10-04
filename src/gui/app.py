import pygame
import sys
import os
import time
import json
from enum import Enum
from src.core.parser import parse_map
from src.core.actions import get_successors
from src.search.ucs import UniformCostSearch
from src.search.astar import AStarSearch
from src.heuristics.push_distance import precompute_push_costs
from src.heuristics.matching import MatchingHeuristic

TILE_SIZE = 64
BACKGROUND_COLOR = (40, 40, 40)
C_WALL = (100, 100, 100)
C_FLOOR = (40, 40, 40)
C_GOAL = (50, 200, 50)
C_BOX = (200, 150, 50)
C_BOX_DONE = (255, 215, 0)
C_AGENT = (50, 150, 255)

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

def draw_glass_panel(surface, rect, alpha=40, border_alpha=120, radius=15):
    temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(temp, (255, 255, 255, alpha), temp.get_rect(), border_radius=radius)
    pygame.draw.rect(temp, (255, 255, 255, border_alpha), temp.get_rect(), width=1, border_radius=radius)
    surface.blit(temp, rect.topleft)

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_pos(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))

class Animator:
    def __init__(self, base_dir):
        self.states = ["Running", "Push", "Die", "Uppercut", "Idle"]
        self.animations = {}
        self.frame_dims = {}
        self.dir_map = { (0, 1): 1, (-1, 0): 3, (0, -1): 5, (1, 0): 7 }
        
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
                        union_rect = None
                        for f_data in data['frames']:
                            r = f_data['frame']
                            rect = pygame.Rect(r['x'], r['y'], r['w'], r['h'])
                            sub = sheet.subsurface(rect)
                            bbox = sub.get_bounding_rect()
                            if bbox.width > 0 and bbox.height > 0:
                                if union_rect is None: union_rect = bbox.copy()
                                else: union_rect.union_ip(bbox)
                        if union_rect is None: union_rect = pygame.Rect(0, 0, 256, 256)
                        else: union_rect.inflate_ip(4, 4)

                        frames = []
                        scale_factor = (TILE_SIZE * 0.9) / union_rect.h
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
                    except: self.animations[state][vec] = None
                else: self.animations[state][vec] = None

    def get_frame(self, state, vec, t, time_ms, start_time_ms=0):
        state_anims = self.animations.get(state, {})
        state_dims = self.frame_dims.get(state, {})
        frames = state_anims.get(vec)
        if not frames: frames = state_anims.get((0, 1))
        if not frames: return None, (TILE_SIZE, TILE_SIZE)
        dims = state_dims.get(vec, state_dims.get((0, 1), (TILE_SIZE, TILE_SIZE)))
        if state in ["Running", "Push"] and t < 1.0:
            frame_idx = int(t * len(frames)) % len(frames)
            if frame_idx == 0 and t > 0: frame_idx = 1
            return frames[frame_idx], dims
        else:
            elapsed = time_ms - start_time_ms
            frame_idx = (elapsed // 62) % len(frames)
            if state in ["Die", "Uppercut"]:
                if (elapsed // 62) >= len(frames): frame_idx = len(frames) - 1
            return frames[frame_idx], dims

class App:
    def __init__(self, map_file):
        pygame.init()
        self.initial_state, self.board = parse_map(map_file)
        self.state = self.initial_state
        self.prev_state = self.initial_state
        self.history = [self.initial_state]
        self.step_index = 0
        self.actions = []
        
        self.width = max(800, self.board.width * TILE_SIZE + 100)
        self.height = max(600, self.board.height * TILE_SIZE + 150)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sokoban AI — Single Agent")
        
        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        self.font = pygame.font.SysFont(font_name, 24, bold=True)
        self.font_lg = pygame.font.SysFont(font_name, 32, bold=True)
        
        self.computing = False
        self.algorithm = "None"
        
        self.anim_t = 1.0
        self.anim_start_time = 0
        self.animator = Animator("src/assets")
        self._last_vec = (0, 1)

        base_soko = "src/assets/soko/PNG/Retina"
        self.img_wall = load_placeholder_image(f"{base_soko}/Blocks/block_03.png", TILE_SIZE, C_WALL, "WALL")
        self.img_floor = load_placeholder_image(f"{base_soko}/Ground/ground_06.png", TILE_SIZE, C_FLOOR, "")
        # Just tint the goal with a nice green target color
        self.img_goal = load_placeholder_image(f"{base_soko}/Environment/environment_02.png", TILE_SIZE, C_GOAL, "GOAL", tint=C_GOAL)
        self.img_box = load_placeholder_image(f"{base_soko}/Crates/crate_02.png", TILE_SIZE, C_BOX, "BOX")
        self.img_box_done = load_placeholder_image(f"{base_soko}/Crates/crate_03.png", TILE_SIZE, C_BOX_DONE, "DONE")
        
    def _create_gradient_bg(self):
        self.bg_surf = pygame.Surface((self.width, self.height))
        c1 = (15, 20, 30)
        c2 = (40, 50, 70)
        for y in range(self.height):
            ratio = y / self.height
            c = (int(c1[0] * (1 - ratio) + c2[0] * ratio),
                 int(c1[1] * (1 - ratio) + c2[1] * ratio),
                 int(c1[2] * (1 - ratio) + c2[2] * ratio))
            pygame.draw.line(self.bg_surf, c, (0, y), (self.width, y))

    def solve_ucs(self):
        self.computing = True
        self.draw()
        t0 = time.time()
        actions, cost, gen, exp = UniformCostSearch().search(self.initial_state, self.board)
        dt = time.time() - t0
        self.last_metrics = (gen, exp, dt)
        self.apply_solution(actions, "UCS")
        
    def solve_astar(self):
        self.computing = True
        self.draw()
        push_costs = precompute_push_costs(self.board)
        h = MatchingHeuristic(self.board, push_costs)
        t0 = time.time()
        actions, cost, gen, exp = AStarSearch(h).search(self.initial_state, self.board)
        dt = time.time() - t0
        self.last_metrics = (gen, exp, dt)
        self.apply_solution(actions, "A*")
        
    def apply_solution(self, actions, algo_name):
        self.computing = False
        self.algorithm = algo_name
        if actions is None:
            self.actions = []
            return
            
        self.actions = actions
        self.history = [self.initial_state]
        
        curr = self.initial_state
        for act in actions:
            for a, s in get_successors(curr, self.board):
                if a == act:
                    curr = s
                    self.history.append(curr)
                    break
        self.step_index = 0
        self.prev_state = self.history[0]
        self.state = self.history[0]
        self.anim_t = 1.0

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

    def draw(self):
        if not hasattr(self, 'bg_surf'):
            self._create_gradient_bg()
        self.screen.blit(self.bg_surf, (0, 0))
        
        board_w_px = self.board.width * TILE_SIZE
        board_h_px = self.board.height * TILE_SIZE
        offset_x = (self.width - board_w_px) // 2
        offset_y = max(20, (self.height - board_h_px - 100) // 2)

        for y in range(self.board.height):
            for x in range(self.board.width):
                px = offset_x + x * TILE_SIZE
                py = offset_y + y * TILE_SIZE
                self.screen.blit(self.img_floor, (px, py))
                if (x, y) in self.board.walls:
                    self.screen.blit(self.img_wall, (px, py))
                elif (x, y) in self.board.goals:
                    self.screen.blit(self.img_goal, (px, py))

        t = self.anim_t
        t = t * t * (3 - 2 * t)
                    
        for (vx, vy), dest in self._get_box_visual_positions(t):
            px = offset_x + vx * TILE_SIZE
            py = offset_y + vy * TILE_SIZE
            img = self.img_box_done if dest in self.board.goals else self.img_box
            self.screen.blit(img, (px, py))
            
        # Draw Agent
        prev_pos = self.prev_state.agent
        next_pos = self.state.agent
        ax, ay = lerp_pos(prev_pos, next_pos, t)
        px = offset_x + ax * TILE_SIZE
        py = offset_y + ay * TILE_SIZE
        
        vec = (next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1])
        if vec == (0, 0): vec = self._last_vec
        else: self._last_vec = vec
        
        is_pushing = False
        if vec != (0, 0) and next_pos in self.prev_state.boxes:
            is_pushing = True

        anim_state = "Idle"
        if t < 1.0: anim_state = "Push" if is_pushing else "Running"
        elif set(self.state.boxes) == set(self.board.goals): anim_state = "Uppercut"
        
        frame, (fw, fh) = self.animator.get_frame(anim_state, vec, t, pygame.time.get_ticks(), self.anim_start_time)
        if frame:
            cx = px + TILE_SIZE // 2
            cy = py + TILE_SIZE
            self.screen.blit(frame, frame.get_rect(midbottom=(cx, cy)))
        else:
            pygame.draw.ellipse(self.screen, C_AGENT, (px + 10, py + 10, TILE_SIZE - 20, TILE_SIZE - 20))
        
        # Draw UI Glass Panel
        ui_rect = pygame.Rect(20, self.height - 100, self.width - 40, 80)
        draw_glass_panel(self.screen, ui_rect, alpha=40, radius=20)
        
        info = f"[{self.algorithm}] Step {self.step_index}/{max(0, len(self.history)-1)}"
        if hasattr(self, 'last_metrics') and self.algorithm != "None":
            gen, exp, dt = self.last_metrics
            info += f" | Gen: {gen} | Exp: {exp} | {dt:.3f}s"
            
        text = self.font.render(info, True, (255, 255, 255))
        self.screen.blit(text, (40, self.height - 85))
        
        controls = "[1] UCS  [2] A*  [Left] Prev  [Right] Next"
        if self.computing: controls = "Computing... Please wait."
        elif not self.actions and self.algorithm != "None": controls = "No Solution Found"
        
        ctrl_text = self.font.render(controls, True, (200, 200, 200))
        self.screen.blit(ctrl_text, (40, self.height - 50))
        
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(60)
            if self.anim_t < 1.0:
                self.anim_t += dt / 250.0  # 250ms animation
                if self.anim_t > 1.0: self.anim_t = 1.0
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and not self.computing:
                    if event.key == pygame.K_1:
                        self.solve_ucs()
                    elif event.key == pygame.K_2:
                        self.solve_astar()
                    elif event.key == pygame.K_LEFT:
                        if self.step_index > 0:
                            self.prev_state = self.history[self.step_index]
                            self.step_index -= 1
                            self.state = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.anim_start_time = pygame.time.get_ticks()
                    elif event.key == pygame.K_RIGHT:
                        if self.step_index < len(self.history) - 1:
                            self.prev_state = self.history[self.step_index]
                            self.step_index += 1
                            self.state = self.history[self.step_index]
                            self.anim_t = 0.0
                            self.anim_start_time = pygame.time.get_ticks()
                            
            self.draw()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        map_file = sys.argv[1]
    else:
        map_file = "maps/test_solvable.txt"
    app = App(map_file)
    app.run()

"""
Single-player Sokoban — setup screen and gameplay, styled like the
competitive mode.  The chosen algorithm (A* or UCS) solves the puzzle in
the background, then the solution is replayed with full animation.
"""

import glob
import os
import sys
import threading
import time

import pygame

from src.core.parser import parse_map
from src.core.actions import get_successors
from src.search.astar import AStarSearch
from src.search.ucs import UniformCostSearch
from src.heuristics.push_distance import precompute_push_costs
from src.heuristics.matching import MatchingHeuristic

from src.gui.common import (
    TILE, UI_H,
    C_WALL, C_FLOOR, C_GOAL, C_BOX, C_BOX_DONE,
    C_UI_BG, C_TEXT, C_TEXT_DIM, C_ACCENT,
    load_placeholder_image, lerp_pos,
    make_fonts, draw_glass_panel, draw_glow, create_gradient_surface,
    load_blurred_image, draw_map_preview,
    Button, Animator,
)

FPS = 60
STEP_DELAY = 260   # ms between replayed solution steps
ANIM_DUR = 0.25    # seconds per animation

C_AGENT = (100, 170, 255)


class SinglePlayerApp:
    def __init__(self, map_file=None):
        pygame.init()

        self.fonts = make_fonts()
        self.font_title = self.fonts["title"]
        self.font_lg = self.fonts["lg"]
        self.font_md = self.fonts["md"]
        self.font_sm = self.fonts["sm"]

        self.available_maps = sorted(glob.glob(os.path.join("maps", "*.txt")))
        if not self.available_maps:
            self.available_maps = ["maps/test_solvable.txt"]
        # Normalize separators so "maps/x.txt" matches "maps\x.txt" on Windows.
        self.map_file = self._resolve_map(map_file) \
            or self._resolve_map("maps/benchmark_2.txt") \
            or self.available_maps[0]
        self.algorithm = "A*"

        # Assets
        base_soko = "src/assets/soko/PNG/Retina"
        self.img_wall = load_placeholder_image(f"{base_soko}/Blocks/block_03.png", TILE, C_WALL, "WALL")
        self.img_floor = load_placeholder_image(f"{base_soko}/Ground/ground_06.png", TILE, C_FLOOR, "")
        self.img_goal = load_placeholder_image(f"{base_soko}/Environment/environment_02.png", TILE, C_GOAL, "GOAL")
        self.img_box = load_placeholder_image(f"{base_soko}/Crates/crate_02.png", TILE, C_BOX, "BOX")
        self.img_box_done = load_placeholder_image(f"{base_soko}/Crates/crate_03.png", TILE, C_BOX_DONE, "DONE")

        self.animator = Animator("src/assets")
        self._last_vec = (0, 1)

        self.menu_bg_img = load_blurred_image("src/assets/soko/Preview.png", 1024, 768)
        self.bg_surface = create_gradient_surface(1024, 768, (10, 15, 30), (50, 20, 80))

        self._exit_requested = False
        self._solve_token = 0
        self.screen = pygame.display.set_mode((1024, 768))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Sokoban — Puzzle Mode")

        self.in_menu = True
        self._setup_menu_buttons()

        # Game state (populated by _start_game)
        self.running = False
        self.finished = False
        self.solving = True
        self.solve_out = None
        self.history = []
        self.step_index = 0
        self.state = None
        self.prev_state = None
        self.anim_t = 1.0
        self.anim_start_time = 0
        self.screen_shake = 0
        self.particles = []
        self.floating_texts = []
        self.finish_time = 0
        self._last_step_time = 0
        self.result_metrics = None   # (cost, gen, exp, dt)

    def _resolve_map(self, map_file):
        """Match a requested path against the available list (any separator)."""
        if not map_file:
            return None
        norm = os.path.normpath(map_file)
        for m in self.available_maps:
            if os.path.normpath(m) == norm:
                return m
        return None

    # ── Setup screen ─────────────────────────────────────────────────────

    def _setup_menu_buttons(self):
        self.menu_buttons = []

        start_y = 170
        for m in self.available_maps:
            name = os.path.basename(m)
            b = Button(70, start_y, 260, 48, name, lambda m=m: self._select_map(m))
            self.menu_buttons.append(b)
            start_y += 60

        # Algorithm choice (one row: label + toggles)
        self.btn_astar = Button(575, 452, 145, 48, "A*",
                                lambda: self._set_algorithm("A*"), accent=C_ACCENT)
        self.btn_ucs = Button(740, 452, 145, 48, "UCS",
                              lambda: self._set_algorithm("UCS"), accent=C_ACCENT)

        self.btn_back = Button(70, 640, 125, 48, "< BACK", self._back_to_launcher)
        self.btn_start = Button(770, 640, 170, 48, "START", self._start_game,
                                selected=True, accent=(120, 220, 120))

        self.menu_buttons.extend([self.btn_astar, self.btn_ucs,
                                  self.btn_back, self.btn_start])
        self._refresh_menu_selection()

    def _refresh_menu_selection(self):
        for b in self.menu_buttons:
            if b.text.endswith(".txt"):
                b.selected = (os.path.basename(b.text) == os.path.basename(self.map_file))
        self.btn_astar.selected = (self.algorithm == "A*")
        self.btn_ucs.selected = (self.algorithm == "UCS")

    def _select_map(self, m):
        self.map_file = m
        self._refresh_menu_selection()

    def _set_algorithm(self, algo):
        self.algorithm = algo
        self._refresh_menu_selection()

    def _back_to_launcher(self):
        self._exit_requested = True

    # ── Game lifecycle ───────────────────────────────────────────────────

    def _start_game(self):
        try:
            self.initial_state, self.board = parse_map(self.map_file)
        except Exception as e:
            print(f"Failed to load map {self.map_file}: {e}")
            return

        self.in_menu = False
        self.state = self.initial_state
        self.prev_state = self.initial_state
        self.history = [self.initial_state]
        self.step_index = 0
        self.anim_t = 1.0
        self.screen_shake = 0
        self.particles = []
        self.floating_texts = []
        self.result_metrics = None

        self.running = True
        self.finished = False
        self.solving = True
        self.solve_out = None
        self.finish_time = 0

        w = self.board.width * TILE
        h = self.board.height * TILE + UI_H
        self.screen_w = max(w, 800)
        self.screen_h = max(h, 600)
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self._last_step_time = pygame.time.get_ticks()

        self._solve_token += 1
        token = self._solve_token
        threading.Thread(target=self._compute_solution, args=(token,),
                         daemon=True).start()

    def _compute_solution(self, token):
        t0 = time.time()
        if self.algorithm == "A*":
            push_costs = precompute_push_costs(self.board)
            h = MatchingHeuristic(self.board, push_costs)
            out = AStarSearch(h).search(self.initial_state, self.board)
        else:
            out = UniformCostSearch().search(self.initial_state, self.board)
        dt = time.time() - t0
        if token != self._solve_token:
            return  # stale search from an abandoned game
        actions, cost, gen, exp = out
        self.solve_out = (actions, cost, gen, exp, dt)

    def _apply_solution(self):
        actions, cost, gen, exp, dt = self.solve_out
        self.solve_out = None
        self.solving = False
        self.result_metrics = (cost, gen, exp, dt)

        if actions is None:
            self.running = False
            self.finished = True
            self.finish_time = pygame.time.get_ticks()
            return

        # Rebuild every state along the solution so we can replay/scrub.
        curr = self.initial_state
        for act in actions:
            for a, s in get_successors(curr, self.board):
                if a == act:
                    curr = s
                    self.history.append(curr)
                    break

        if len(self.history) == 1:
            # Already solved on the first frame.
            self.running = False
            self.finished = True
            self.finish_time = pygame.time.get_ticks()
        else:
            self._last_step_time = pygame.time.get_ticks()

    def _advance(self):
        if self.step_index < len(self.history) - 1:
            self.prev_state = self.history[self.step_index]
            self.step_index += 1
            self.state = self.history[self.step_index]
            self.anim_t = 0.0
            self.anim_start_time = pygame.time.get_ticks()
            self._spawn_step_effects()
        if self.state.boxes == self.board.goals:
            self.running = False
            self.finished = True
            self.finish_time = pygame.time.get_ticks()

    def _rewind(self):
        if self.step_index <= 0:
            return
        self.prev_state = self.history[self.step_index]
        self.step_index -= 1
        self.state = self.history[self.step_index]
        self.anim_t = 0.0
        self.anim_start_time = pygame.time.get_ticks()
        # Stepping away from the final frame hides the result overlay.
        self.finished = False

    def _spawn_step_effects(self):
        import random
        prev_boxes = self.prev_state.boxes
        new_boxes = self.state.boxes - prev_boxes

        for nb in new_boxes:
            for _ in range(8):
                self.particles.append({
                    "pos": [nb[0] * TILE + TILE // 2, nb[1] * TILE + TILE - 5],
                    "vx": random.uniform(-1.5, 1.5),
                    "vy": random.uniform(-1, 0),
                    "life": 30,
                    "max_life": 30,
                })

        prev_goals = len(prev_boxes & self.board.goals)
        curr_goals = len(self.state.boxes & self.board.goals)
        if curr_goals > prev_goals:
            self.screen_shake = 15
            for g in (self.state.boxes & self.board.goals) - (prev_boxes & self.board.goals):
                self.floating_texts.append({
                    "pos": [g[0] * TILE + TILE // 2, g[1] * TILE],
                    "life": 60,
                    "max_life": 60,
                })

    def _to_menu(self):
        self.in_menu = True
        self.running = False
        self.screen = pygame.display.set_mode((1024, 768))

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self):
        self._exit_requested = False
        while not self._exit_requested:
            dt_ms = self.clock.tick(FPS)
            dt = dt_ms / 1000.0

            self._handle_events()

            if not self.in_menu:
                if self.anim_t < 1.0:
                    self.anim_t = min(1.0, self.anim_t + dt / ANIM_DUR)

                # Awaiting the search thread?
                if self.solving and self.solve_out is not None:
                    self._apply_solution()

                # Auto-replay the solution
                if (self.running and not self.finished and not self.solving
                        and self.anim_t >= 1.0):
                    if pygame.time.get_ticks() - self._last_step_time >= STEP_DELAY:
                        self._advance()
                        self._last_step_time = pygame.time.get_ticks()

            self._draw()

        # Back to launcher — keep pygame running for the parent shell.

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

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw(self):
        if self.in_menu:
            self._draw_menu()
        else:
            self.screen.fill((30, 30, 30))

            sx, sy = 0, 0
            if self.screen_shake > 0:
                import random
                sx = random.randint(-self.screen_shake, self.screen_shake)
                sy = random.randint(-self.screen_shake, self.screen_shake)
                self.screen_shake -= 1

            game_surf = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            old_screen = self.screen
            self.screen = game_surf
            self._draw_board()
            self._draw_particles()
            self._draw_floating_texts()
            self.screen = old_screen
            self.screen.blit(game_surf, (sx, sy))

            self._draw_ui()
            if self.solving:
                self._draw_solving_overlay()
            elif self.finished:
                self._draw_result_overlay()
        pygame.display.flip()

    def _draw_menu(self):
        if self.menu_bg_img:
            self.screen.blit(self.menu_bg_img, (0, 0))
        else:
            self.screen.blit(self.bg_surface, (0, 0))
        # Darken the background so the UI reads clearly
        overlay = pygame.Surface((1024, 768), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        # Title
        title_shadow = self.font_title.render("SOKOBAN PUZZLE", True, (0, 0, 0))
        title = self.font_title.render("SOKOBAN PUZZLE", True, (255, 255, 255))
        self.screen.blit(title_shadow, title_shadow.get_rect(center=(513, 61)))
        self.screen.blit(title, title.get_rect(center=(512, 60)))

        # Left panel — map list
        panel_left = pygame.Rect(50, 110, 300, 600)
        draw_glass_panel(self.screen, panel_left, alpha=20, radius=20)
        lbl = self.font_lg.render("Select Map", True, C_TEXT)
        self.screen.blit(lbl, (70, 130))

        # Right panel — settings
        panel_right = pygame.Rect(400, 110, 570, 600)
        draw_glass_panel(self.screen, panel_right, alpha=20, radius=20)
        lbl = self.font_lg.render("Game Settings", True, C_TEXT)
        self.screen.blit(lbl, (430, 130))

        preview_rect = pygame.Rect(430, 180, 510, 240)
        draw_map_preview(self.screen, preview_rect, self.map_file, "single", self.fonts)

        algo_lbl = self.font_lg.render("Algorithm", True, C_TEXT)
        self.screen.blit(algo_lbl, (430, 459))
        algo_hint = self.font_sm.render("A* = heuristic search   UCS = uniform cost",
                                        True, C_TEXT_DIM)
        self.screen.blit(algo_hint, (430, 520))

        hint = self.font_sm.render("ENTER = start  |  ESC = back", True, C_TEXT_DIM)
        self.screen.blit(hint, (430, 560))

        # Buttons (with glow behind the selected map entry)
        for b in self.menu_buttons:
            if b.selected and b.text.endswith(".txt"):
                draw_glow(self.screen, b.rect, color=(255, 255, 255), alpha=60)
            b.draw(self.screen)

    def _board_offsets(self):
        board_w = self.board.width * TILE
        return max(0, (self.screen_w - board_w) // 2), 0

    def _draw_board(self):
        offset_x, offset_y = self._board_offsets()

        for y in range(self.board.height):
            for x in range(self.board.width):
                px = offset_x + x * TILE
                py = offset_y + y * TILE
                self.screen.blit(self.img_floor, (px, py))
                if (x, y) in self.board.walls:
                    self.screen.blit(self.img_wall, (px, py))
                if (x, y) in self.board.goals:
                    self.screen.blit(self.img_goal, (px, py))

        t = self.anim_t
        t = t * t * (3 - 2 * t)

        for (vx, vy), dest in self._box_visual_positions(t):
            px = offset_x + vx * TILE
            py = offset_y + vy * TILE
            img = self.img_box_done if dest in self.board.goals else self.img_box
            self.screen.blit(img, (px, py))

        self._draw_agent(t, offset_x, offset_y)

    def _box_visual_positions(self, t):
        prev_boxes = self.prev_state.boxes
        next_boxes = self.state.boxes

        prev_unmatched = list(prev_boxes - next_boxes)
        next_unmatched = list(next_boxes - prev_boxes)

        positions = [(b, b) for b in (prev_boxes & next_boxes)]

        for pb in prev_unmatched:
            best_nb = None
            best_dist = 999
            for nb in next_unmatched:
                dist = abs(pb[0] - nb[0]) + abs(pb[1] - nb[1])
                if dist < best_dist:
                    best_dist = dist
                    best_nb = nb
            if best_nb:
                positions.append((pb, best_nb))
                next_unmatched.remove(best_nb)

        for nb in next_unmatched:
            positions.append((nb, nb))

        return [(lerp_pos(pb, nb, t), nb) for pb, nb in positions]

    def _draw_agent(self, t, ox, oy):
        prev_pos = self.prev_state.agent
        next_pos = self.state.agent
        ax, ay = lerp_pos(prev_pos, next_pos, t)
        px, py = ox + ax * TILE, oy + ay * TILE

        vec = (next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1])
        if vec == (0, 0):
            vec = self._last_vec
        else:
            self._last_vec = vec

        is_pushing = vec != (0, 0) and next_pos in self.prev_state.boxes

        anim_state = "Idle"
        if t < 1.0 and next_pos != prev_pos:
            anim_state = "Push" if is_pushing else "Running"
        elif self.finished and self.state.boxes == self.board.goals:
            anim_state = "Uppercut"

        start_t = self.finish_time if self.finished else self.anim_start_time
        frame, dims = self.animator.get_frame(anim_state, vec, t,
                                              pygame.time.get_ticks(), start_t)
        # Aura under the feet so the agent reads well on any tile.
        pygame.draw.ellipse(self.screen, C_AGENT,
                            (px + TILE // 2 - 15, py + TILE - 10, 30, 10))

        if frame:
            fw, fh = dims
            self.screen.blit(frame, (px + (TILE - fw) / 2, py + (TILE - fh)))
        else:
            pygame.draw.ellipse(self.screen, C_AGENT,
                                (px + 10, py + 10, TILE - 20, TILE - 20))

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200, 200, 200, alpha), (4, 4), 4)
            self.screen.blit(surf, (p["pos"][0] - 4, p["pos"][1] - 4))
            p["pos"][0] += p["vx"]
            p["pos"][1] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _draw_floating_texts(self):
        for t in self.floating_texts:
            alpha = int(255 * (t["life"] / t["max_life"]))
            txt = self.font_lg.render("+1", True, (255, 215, 0))
            txt.set_alpha(alpha)
            self.screen.blit(txt, txt.get_rect(center=(t["pos"][0], t["pos"][1])))
            t["pos"][1] -= 0.5
            t["life"] -= 1
        self.floating_texts = [t for t in self.floating_texts if t["life"] > 0]

    def _draw_ui(self):
        ui_top = self.screen_h - UI_H
        ui_rect = pygame.Rect(0, ui_top, self.screen_w, UI_H)
        pygame.draw.rect(self.screen, C_UI_BG, ui_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (0, ui_top),
                         (self.screen_w, ui_top), 2)

        goals_done = len(self.state.boxes & self.board.goals)
        goals_total = len(self.board.goals)

        algo_txt = self.font_lg.render(self.algorithm, True, (120, 200, 255))
        step_txt = self.font_lg.render(
            f"Step {self.step_index} / {max(0, len(self.history) - 1)}", True, C_TEXT)
        goal_txt = self.font_lg.render(
            f"Goals {goals_done}/{goals_total}", True, (255, 215, 120))

        self.screen.blit(algo_txt, (30, ui_top + 15))
        self.screen.blit(step_txt, (30 + algo_txt.get_width() + 30, ui_top + 15))

        if self.result_metrics and not self.solving:
            cost, gen, exp, dt = self.result_metrics
            stats = self.font_md.render(
                f"Cost {cost}  |  Generated {gen}  |  Expanded {exp}  |  {dt*1000:.0f}ms",
                True, C_TEXT_DIM)
            self.screen.blit(stats, (30, ui_top + 55))
        elif self.solving:
            stats = self.font_md.render("Searching for a solution...", True, C_TEXT_DIM)
            self.screen.blit(stats, (30, ui_top + 55))

        self.screen.blit(goal_txt,
                         (self.screen_w - goal_txt.get_width() - 30, ui_top + 15))

        if self.solving:
            ctrl = "ESC=menu — solving, please wait"
        elif self.finished:
            ctrl = "SPACE=menu  |  ,=prev  |  .=next  |  R=replay"
        elif self.running:
            ctrl = "SPACE=pause  |  ,=prev  |  .=next  |  R=replay  |  M=menu"
        else:
            ctrl = "SPACE=play  |  ,=prev  |  .=next  |  R=replay  |  M=menu"
        ctrl_txt = self.font_sm.render(ctrl, True, C_TEXT_DIM)
        self.screen.blit(ctrl_txt, (30, ui_top + 90))

    def _draw_solving_overlay(self):
        draw_glass_panel(self.screen, pygame.Rect(0, 0, self.screen_w, self.screen_h),
                         alpha=120, border_alpha=0, radius=0)
        dots = "." * (1 + (pygame.time.get_ticks() // 350) % 3)
        msg = self.font_lg.render(f"SOLVING WITH {self.algorithm}{dots}",
                                  True, (180, 220, 255))
        sub = self.font_md.render("Searching for the optimal path", True, C_TEXT_DIM)

        cx, cy = self.screen_w // 2, self.screen_h // 2
        panel = pygame.Rect(0, 0, 430, 150)
        panel.center = (cx, cy)
        draw_glass_panel(self.screen, panel, alpha=40, border_alpha=150, radius=20)
        self.screen.blit(msg, msg.get_rect(center=(cx, cy - 25)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 25)))

    def _draw_result_overlay(self):
        draw_glass_panel(self.screen, pygame.Rect(0, 0, self.screen_w, self.screen_h),
                         alpha=150, border_alpha=0, radius=0)

        solved = bool(self.history) and self.state.boxes == self.board.goals

        if solved:
            moves = len(self.history) - 1
            msg, col = "PUZZLE SOLVED!", (150, 255, 150)
            detail = f"{moves} moves"
            if self.result_metrics:
                cost, gen, exp, dt = self.result_metrics
                detail += f"  |  Cost {cost}  |  Gen {gen}  |  Exp {exp}  |  {dt:.3f}s"
        else:
            msg, col = "NO SOLUTION FOUND", (255, 150, 150)
            detail = "The puzzle is unsolvable from this position"

        cx, cy = self.screen_w // 2, self.screen_h // 2
        panel = pygame.Rect(0, 0, 560, 190)
        panel.center = (cx, cy)
        draw_glass_panel(self.screen, panel, alpha=40, border_alpha=150, radius=20)

        txt = self.font_lg.render(msg, True, col)
        sub = self.font_md.render(detail, True, C_TEXT)
        hint = self.font_md.render("Press SPACE to return to the menu", True,
                                   (200, 255, 200))
        self.screen.blit(txt, txt.get_rect(center=(cx, cy - 40)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 10)))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 60)))

"""All drawing: menu, board, agent, HUD, overlays, effects."""



import pygame


from src.shared.common import (
    TILE,
    UI_H,
    C_UI_BG,
    C_TEXT,
    C_TEXT_DIM,
    lerp_pos,
    draw_glass_panel,
    draw_glow,
    draw_map_preview,
)


C_AGENT = (100, 170, 255)



class RenderMixin:
    """All drawing: menu, board, agent, HUD, overlays, effects."""

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


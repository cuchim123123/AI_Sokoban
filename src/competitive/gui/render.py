"""All drawing: menu, board, agents, HUD, overlays, effects."""


import pygame
import os


from src.shared.common import (
    TILE,
    UI_H,
    C_BG,
    C_UI_BG,
    C_TEXT,
    C_TEXT_DIM,
    lerp_pos,
    draw_glass_panel,
    draw_result_panel,
    draw_menu_panel, C_MENU_BG, C_MENU_INK,
    draw_glow,
    draw_map_preview,
)


# ── Mode-specific colors ──
C_AGENT_A  = (255, 100, 100)
C_AGENT_B  = (100, 150, 255)

C_WIN_A    = (255, 150, 150)
C_WIN_B    = (150, 200, 255)
C_WIN_DRAW = (200, 200, 200)



class RenderMixin:
    """All drawing: menu, board, agents, HUD, overlays, effects."""

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200, 200, 200, alpha), (4, 4), 4)
            self.screen.blit(surf, (p["pos"][0]-4, p["pos"][1]-4))
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

    def _draw(self):
        if self.in_menu:
            self._draw_menu()
        else:
            self.screen.fill(C_BG)
            
            sx, sy = 0, 0
            if getattr(self, 'screen_shake', 0) > 0:
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
            if self.finished:
                self._draw_result_overlay()
        pygame.display.flip()
        
    def _draw_menu(self):
        self.screen.fill(C_MENU_BG)

        title = self.font_title.render("SOKOBAN COMPETITIVE", True, C_MENU_INK)
        self.screen.blit(title, title.get_rect(center=(512, 60)))

        # Left Panel (Map List)
        panel_left = pygame.Rect(50, 110, 300, 600)
        draw_menu_panel(self.screen, panel_left)
        
        lbl = self.font_lg.render("Select Map", True, (255, 255, 255))
        self.screen.blit(lbl, (70, 130))

        # Right Panel (Settings & Preview)
        panel_right = pygame.Rect(400, 110, 570, 600)
        draw_menu_panel(self.screen, panel_right)
        
        lbl = self.font_lg.render("Game Settings", True, (255, 255, 255))
        self.screen.blit(lbl, (430, 130))

        # Map Preview Area (live render of the selected map)
        preview_rect = pygame.Rect(430, 180, 510, 240)
        draw_map_preview(self.screen, preview_rect, self.map_file, "competitive",
                         self.fonts)
        
        # Max Steps display
        steps_txt = self.font_lg.render(f"Max Steps: {self.max_steps}", True, (255, 255, 255))
        self.screen.blit(steps_txt, (430, 458))

        hint = self.font_sm.render("ENTER = start  |  ESC = back", True, C_TEXT_DIM)
        self.screen.blit(hint, (430, 574))

        # Highlight for selected map button
        for b in self.menu_buttons:
            if b.text.endswith(".txt"):
                b.selected = (b.text == os.path.basename(self.map_file))
                if b.selected:
                    draw_glow(self.screen, b.rect, color=(255, 255, 255), alpha=60)
            b.draw(self.screen)

    def _get_box_visual_positions(self, t):
        # Boxes have no IDs. Recover actual pushes from executed player
        # movement instead of guessing nearest matches between unordered sets.
        # Always infer in chronological order, then reverse paths for rewind.
        reverse = self.state.step < self.prev_state.step
        earlier, later = ((self.state, self.prev_state) if reverse
                          else (self.prev_state, self.state))
        sources = {}
        if later.step == earlier.step + 1:
            for start, end in ((earlier.agent_a, later.agent_a),
                               (earlier.agent_b, later.agent_b)):
                dx, dy = end[0] - start[0], end[1] - start[1]
                if abs(dx) + abs(dy) != 1 or end not in earlier.boxes:
                    continue
                box_end = (end[0] + dx, end[1] + dy)
                if box_end in later.boxes:
                    source, dest = (box_end, end) if reverse else (end, box_end)
                    sources[dest] = source

        return [(lerp_pos(sources.get(dest, dest), dest, t), dest)
                for dest in sorted(self.state.boxes)]

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
            
            # Off-goal boxes are ALWAYS neutral: color is decided purely
            # by whether this box currently sits on a goal (credited or
            # not), so pushing a box away drops it straight back to the
            # neutral sprite.  (Owner-based tinting used the same crate
            # sprites as the "done" colors, so a pushed box never
            # reverted.)
            if dest in self.state.boxes_on_goals_a:
                img = self.img_box_done_a
            elif dest in self.state.boxes_on_goals_b:
                img = self.img_box_done_b
            elif dest in board.goals:
                img = self.img_box_done_a
            else:
                img = self.img_box

            self.screen.blit(img, (px, py))

        self._draw_agent_anim("A", self.prev_state.agent_a, self.state.agent_a, t, offset_x, offset_y, C_AGENT_A)
        self._draw_agent_anim("B", self.prev_state.agent_b, self.state.agent_b, t, offset_x, offset_y, C_AGENT_B)

        # Conflict-advantage badge: floats up over the priority winner of
        # the last conflicting round and fades out (~1.3s).
        if self.conflict_badge:
            b = self.conflict_badge
            col = C_AGENT_A if b["winner"] == "A" else C_AGENT_B
            label = f"{b['winner']} WINS CONFLICT"
            rise = (b["max_life"] - b["life"]) * 0.4
            cx = offset_x + b["cell"][0] * TILE + TILE // 2
            cy = max(12, offset_y + b["cell"][1] * TILE - 14 - rise)
            alpha = int(255 * b["life"] / b["max_life"])
            for text, color, dx, dy in (
                (label, (0, 0, 0), 1, 1),
                (label, col, 0, 0),
            ):
                s = self.font_md.render(text, True, color)
                s.set_alpha(alpha)
                self.screen.blit(s, s.get_rect(center=(cx + dx, cy + dy)))
            b["life"] -= 1
            if b["life"] <= 0:
                self.conflict_badge = None

        # Yielding is a deliberate B action, not a conflict-priority win.
        # Use a warm, solid pill BELOW B; conflict text floats ABOVE its winner.
        if self.step_loop_breaks.get(self.state.step):
            label = self.font_sm.render("B is breaking the loop", True, (255, 218, 145))
            width, height = label.get_width() + 44, label.get_height() + 16
            bx, by = lerp_pos(self.prev_state.agent_b, self.state.agent_b, t)
            x = max(8, min(self.screen_w - width - 8,
                           offset_x + bx * TILE + TILE // 2 - width // 2))
            y = min(self.screen_h - UI_H - height - 6,
                    offset_y + (by + 1) * TILE + 8)
            rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(self.screen, (55, 41, 24), rect, border_radius=10)
            pygame.draw.rect(self.screen, (222, 166, 73), rect, width=1, border_radius=10)
            # Bent arrow visually distinguishes yielding from winning.
            cx, cy = rect.x + 17, rect.centery
            pygame.draw.lines(self.screen, (255, 218, 145), False,
                              [(cx - 5, cy - 5), (cx - 5, cy + 4), (cx + 6, cy + 4)], 2)
            pygame.draw.lines(self.screen, (255, 218, 145), False,
                              [(cx + 2, cy), (cx + 6, cy + 4), (cx + 2, cy + 8)], 2)
            self.screen.blit(label, (rect.x + 32, rect.y + 8))

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
            frame = frame.copy()
            tint_surf = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
            tint_color = (255, 150, 150, 255) if label == "A" else (150, 150, 255, 255)
            tint_surf.fill(tint_color)
            frame.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
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
        remaining = max(0, self.max_steps - state.step)
        step_txt    = self.font_lg.render(f"Steps Left: {remaining}", True, C_TEXT)
        
        self.screen.blit(score_txt_a, (30, ui_top + 15))
        self.screen.blit(score_txt_b, (30 + score_txt_a.get_width() + 40, ui_top + 15))
        self.screen.blit(step_txt, (self.screen_w - step_txt.get_width() - 30, ui_top + 15))

        if self._metrics:
            _, act_a, act_b, _, _ = self._metrics[-1]
            act_a_str = act_a.name if hasattr(act_a, 'name') else str(act_a)
            act_b_str = act_b.name if hasattr(act_b, 'name') else str(act_b)
            last_txt = self.font_md.render(
                f"A: {act_a_str} | B: {act_b_str}",
                True, C_TEXT_DIM
            )
            self.screen.blit(last_txt, (30, ui_top + 55))

            # Who held conflict priority for the displayed round (UI only)
            winner = self.step_conflicts.get(state.step)
            if winner is not None:
                conf_txt = self.font_md.render(
                    f"| CONFLICT: {winner} had priority", True,
                    C_AGENT_A if winner == "A" else C_AGENT_B,
                )
                self.screen.blit(
                    conf_txt, (30 + last_txt.get_width() + 20, ui_top + 55)
                )

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
        
        if sa > sb:
            msg, col = "AGENT A WINS!", C_WIN_A
        elif sb > sa:
            msg, col = "AGENT B WINS!", C_WIN_B
        else:
            msg, col = "DRAW!", C_WIN_DRAW

        draw_result_panel(self.screen, self.fonts, msg,
                          [f"Final Score:  A {sa}   |   B {sb}"], col)



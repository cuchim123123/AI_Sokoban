import re

with open("src/competitive/gui/app.py", "r") as f:
    code = f.read()

# 1. Menu Background Blur
menu_bg_init = """        self.clock = pygame.time.Clock()
        
        # Load and blur background
        try:
            raw_bg = pygame.image.load("src/assets/soko/Preview.png").convert()
            small = pygame.transform.scale(raw_bg, (raw_bg.get_width()//16, raw_bg.get_height()//16))
            self.menu_bg_img = pygame.transform.smoothscale(small, (self.screen_w, self.screen_h))
        except:
            self.menu_bg_img = None
"""
code = code.replace("        self.clock = pygame.time.Clock()", menu_bg_init)

menu_bg_draw = """    def _draw_menu(self):
        if hasattr(self, 'menu_bg_img') and self.menu_bg_img:
            self.screen.blit(self.menu_bg_img, (0, 0))
        else:
            self.screen.blit(self.bg_surface, (0, 0))"""
code = code.replace(
    '    def _draw_menu(self):\n        # Background\n        self.screen.blit(self.bg_surface, (0, 0))',
    menu_bg_draw
)

# 2. Setup game juice vars in _start_game
start_game_juice = """        self.anim_t = 1.0
        self.screen_shake = 0
        self.particles = []
        self.floating_texts = []"""
code = code.replace("        self.anim_t = 1.0", start_game_juice)

# 3. Apply game juice logic in _apply_computed_step
apply_juice = """        new_box_owners = {}
        
        import random
        for nb in (self.state.boxes - self.prev_state.boxes):
            # Spawn dust particles for pushed box
            for _ in range(8):
                self.particles.append({
                    "pos": [nb[0] * TILE + TILE//2, nb[1] * TILE + TILE - 5],
                    "vx": random.uniform(-1.5, 1.5),
                    "vy": random.uniform(-1, 0),
                    "life": 30,
                    "max_life": 30
                })
        
        prev_goals = len(self.prev_state.boxes & self.board.goals)
        curr_goals = len(self.state.boxes & self.board.goals)
        if curr_goals > prev_goals:
            self.screen_shake = 15
            for g in (self.state.boxes & self.board.goals) - (self.prev_state.boxes & self.board.goals):
                self.floating_texts.append({
                    "pos": [g[0] * TILE + TILE//2, g[1] * TILE],
                    "life": 60,
                    "max_life": 60
                })"""

code = code.replace("        new_box_owners = {}", apply_juice + "\n        new_box_owners = {}")

# 4. Agent Differentiation tinting in _draw_agent_anim
draw_agent_anim = """        frame = frame.copy()
        tint_surf = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
        tint_color = (255, 100, 100, 255) if agent_id == "A" else (100, 150, 255, 255)
        tint_surf.fill(tint_color)
        frame.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Draw shadow"""
code = code.replace("        # Draw shadow", draw_agent_anim)

# 5. Drawing shake, particles and texts
draw_func = """    def _draw_particles(self):
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
        pygame.display.flip()"""

# Replace the old _draw
code = re.sub(r'    def _draw\(self\):.*?pygame\.display\.flip\(\)', draw_func, code, flags=re.DOTALL)


with open("src/competitive/gui/app.py", "w") as f:
    f.write(code)

print("Juice patched successfully!")

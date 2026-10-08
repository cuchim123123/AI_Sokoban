"""Competitive Sokoban GUI - orchestrator.

`CompetitiveApp` owns construction and the main loop only;
menu/setup, input, turn computation and rendering live in the
sibling concern modules (setup.py, events.py, game.py,
render.py) as mixins.
"""


import pygame
import time
import glob
import threading


from src.shared.common import (
    TILE,
    C_WALL,
    C_FLOOR,
    C_GOAL,
    C_BOX,
    load_placeholder_image,
    make_fonts,
    load_blurred_image,
    Animator,
)

# ── Config ──
FPS  = 60
STEP_DELAY = 400  # ms between steps for AI
ANIM_DUR = 0.4 # seconds for animation

# ── Mode-specific colors ──
C_AGENT_A  = (255, 100, 100)
C_AGENT_B  = (100, 150, 255)
C_A_DONE   = (200,  50,  50)
C_B_DONE   = ( 50, 100, 200)




from src.competitive.gui.setup import SetupMixin
from src.competitive.gui.game import GameMixin
from src.competitive.gui.events import EventsMixin
from src.competitive.gui.render import RenderMixin

class CompetitiveApp(SetupMixin, GameMixin, EventsMixin, RenderMixin):
    def __init__(self, map_file: str, max_steps: int, ai_a: str = "AI",
                 ai_b: str = "AI", launcher: bool = False):
        pygame.init()

        self.launcher = launcher
        self._exit_requested = False
        pygame.display.set_caption("Sokoban — Competitive Mode")

        self.map_file = map_file
        self.max_steps = max_steps
        self.ai_a_type = ai_a if ai_a in ["human", "AI"] else "AI"
        self.ai_b_type = ai_b if ai_b in ["human", "AI"] else "AI"

        self.screen_w = 1024
        self.screen_h = 768
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        
        # Load and blur background
        try:
            self.menu_bg_img = load_blurred_image("src/assets/soko/Preview.png",
                                                  self.screen_w, self.screen_h)
        except Exception:
            self.menu_bg_img = None

        # Create gradient background surface
        self._create_gradient_bg()

        self.fonts = make_fonts()
        self.font_title = self.fonts["title"]
        self.font_lg = self.fonts["lg"]
        self.font_md = self.fonts["md"]
        self.font_sm = self.fonts["sm"]
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
        self.screen_shake = 0
        self.particles = []
        self.floating_texts = []

        # Conflict-advantage UI: per-step priority winner (UI display
        # only - never fed back into search or scoring).
        self.step_conflicts = {}
        self.conflict_badge = None
        
        # Threading state
        self.computing = False
        self.pending_out = None

    def run(self):
        last_time = time.time()
        self._exit_requested = False
        while not self._exit_requested:
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

if __name__ == "__main__":
    app = CompetitiveApp("maps/competitive/arena_open.txt", 50)
    app.run()

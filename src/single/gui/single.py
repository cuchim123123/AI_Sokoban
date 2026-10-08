"""Single-player Sokoban - orchestrator.

`SinglePlayerApp` owns construction and the main loop only;
menu/setup, input, game lifecycle and rendering live in the
sibling concern modules (setup.py, events.py, game.py,
render.py) as mixins.
"""


import glob
import os

import pygame


from src.shared.common import (
    TILE,
    C_WALL,
    C_FLOOR,
    C_GOAL,
    C_BOX,
    C_BOX_DONE,
    load_placeholder_image,
    make_fonts,
    Animator,
)

FPS = 60
STEP_DELAY = 260   # ms between replayed solution steps
ANIM_DUR = 0.25    # seconds per animation




from src.single.gui.setup import SetupMixin
from src.single.gui.game import GameMixin
from src.single.gui.events import EventsMixin
from src.single.gui.render import RenderMixin

class SinglePlayerApp(SetupMixin, GameMixin, EventsMixin, RenderMixin):
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


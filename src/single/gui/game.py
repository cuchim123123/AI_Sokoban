"""Game lifecycle: solve thread, solution replay, step/rewind."""


import threading
import time

import pygame

from src.single.core.parser import parse_map
from src.single.core.actions import get_successors
from src.single.search.astar import AStarSearch
from src.single.search.ucs import UniformCostSearch
from src.single.heuristics.push_distance import precompute_push_costs
from src.single.heuristics.matching import MatchingHeuristic

from src.shared.common import (
    TILE,
    UI_H,
)





class GameMixin:
    """Game lifecycle: solve thread, solution replay, step/rewind."""

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


import pygame
import sys
import os
import time
from enum import Enum
from src.core.parser import parse_map
from src.core.actions import get_successors
from src.search.ucs import UniformCostSearch
from src.search.astar import AStarSearch
from src.heuristics.push_distance import precompute_push_costs
from src.heuristics.matching import MatchingHeuristic

TILE_SIZE = 64
BACKGROUND_COLOR = (40, 40, 40)
WALL_COLOR = (100, 100, 100)
AGENT_COLOR = (50, 150, 255)
BOX_COLOR = (200, 150, 50)
GOAL_COLOR = (50, 200, 50)
BOX_ON_GOAL_COLOR = (255, 215, 0)

class App:
    def __init__(self, map_file):
        pygame.init()
        self.initial_state, self.board = parse_map(map_file)
        self.state = self.initial_state
        self.history = [self.initial_state]
        self.step_index = 0
        self.actions = []
        
        self.width = self.board.width * TILE_SIZE
        self.height = self.board.height * TILE_SIZE + 80  # Extra space for UI
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sokoban AI")
        self.font = pygame.font.SysFont("Arial", 24)
        
        self.computing = False
        self.algorithm = "None"
        
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
            # Find the successor that matches the action
            for a, s in get_successors(curr, self.board):
                if a == act:
                    curr = s
                    self.history.append(curr)
                    break
        self.step_index = 0
        self.state = self.history[0]
        
    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw board
        for x in range(self.board.width):
            for y in range(self.board.height):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if (x, y) in self.board.walls:
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                elif (x, y) in self.board.goals:
                    pygame.draw.circle(self.screen, GOAL_COLOR, rect.center, TILE_SIZE // 4)
                    
        # Draw dynamic entities
        for bx, by in self.state.boxes:
            rect = pygame.Rect(bx * TILE_SIZE + 4, by * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8)
            color = BOX_ON_GOAL_COLOR if (bx, by) in self.board.goals else BOX_COLOR
            pygame.draw.rect(self.screen, color, rect)
            
        ax, ay = self.state.agent
        agent_rect = pygame.Rect(ax * TILE_SIZE + 10, ay * TILE_SIZE + 10, TILE_SIZE - 20, TILE_SIZE - 20)
        pygame.draw.ellipse(self.screen, AGENT_COLOR, agent_rect)
        
        # Draw UI
        ui_rect = pygame.Rect(0, self.board.height * TILE_SIZE, self.width, 80)
        pygame.draw.rect(self.screen, (30, 30, 30), ui_rect)
        
        info = f"Algo: {self.algorithm} | Step: {self.step_index}/{max(0, len(self.history)-1)}"
        if hasattr(self, 'last_metrics') and self.algorithm != "None":
            gen, exp, dt = self.last_metrics
            info += f" | Gen: {gen} | Exp: {exp} | {dt:.3f}s"
            
        text = self.font.render(info, True, (255, 255, 255))
        self.screen.blit(text, (10, self.board.height * TILE_SIZE + 10))
        
        controls = "[1] UCS  [2] A*  [Left] Prev  [Right] Next"
        if self.computing:
            controls = "Computing..."
        elif not self.actions and self.algorithm != "None":
            controls = "No Solution Found"
            
        ctrl_text = self.font.render(controls, True, (200, 200, 200))
        self.screen.blit(ctrl_text, (10, self.board.height * TILE_SIZE + 40))
        
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
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
                            self.step_index -= 1
                            self.state = self.history[self.step_index]
                    elif event.key == pygame.K_RIGHT:
                        if self.step_index < len(self.history) - 1:
                            self.step_index += 1
                            self.state = self.history[self.step_index]
                            
            self.draw()
            clock.tick(30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        map_file = sys.argv[1]
    else:
        map_file = "maps/test_solvable.txt"
    app = App(map_file)
    app.run()

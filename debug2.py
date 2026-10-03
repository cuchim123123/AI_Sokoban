import sys
from src.competitive.parser import parse_competitive_map
from src.competitive.agent_a import best_action, AgentA
from src.competitive.state import CompetitiveState

state, board = parse_competitive_map('maps/competitive/dense_goals.txt')
state.agent_a = (4, 2)
state.agent_b = (10, 3)
state.step = 2
state.boxes = frozenset([(4, 4), (7, 4), (5, 2), (10, 4), (8, 2)])
state.boxes_on_goals_a = frozenset()
state.boxes_on_goals_b = frozenset()

import src.competitive.agent_a as aa
import inspect
source = inspect.getsource(aa.best_action)
source = source.replace('for act in root_acts:', 'for act in root_acts:\n                if d == 4: print(f"Evaluating B={act.name} at depth {d}")')
source = source.replace('if val > best_val:', 'if d == 4: print(f"B={act.name} min_val={val}")\n                if val > best_val:')

exec(source, aa.__dict__)

b = AgentA()
act_b = aa.best_action(state, board, 40, 'B', b._history, b.tt, b.heuristic_cache)
print(f"Agent B chose: {act_b}")

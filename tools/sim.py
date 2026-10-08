import os
import sys
import time

# Allow running from anywhere: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.competitive.state import Action
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB
from src.competitive.transition import resolve_joint_action_outcome
from src.competitive.parser import parse_competitive_map

def run_sim():
    state, board = parse_competitive_map('maps/competitive/arena_open.txt')
    agent_a = AgentA()
    agent_b = AgentB()
    
    for i in range(40):
        print(f"\n--- STEP {i} ---")
        print(f"A: {state.agent_a}  B: {state.agent_b}")
        print(f"Boxes: {list(state.boxes)}")
        
        act_a = agent_a.choose_action(state, board, max_steps=50)
        act_b = agent_b.choose_action(state, board, max_steps=50)
        
        print(f"A chose: {act_a.name}")
        print(f"B chose: {act_b.name}")
        
        out = resolve_joint_action_outcome(
            state, act_a, act_b, board, 50,
            agent_a.preference_list(state, board, 50, act_a),
            agent_b.preference_list(state, board, 50, act_b),
        )
        print(f"Executed: A={out.resolved_action_a.name}, B={out.resolved_action_b.name}")
        print(f"Completed horizons: A={agent_a.last_search['depth']}, B={agent_b.last_search['depth']}")
        if out.conflict:
            print(">>> CONFLICT! <<<")

        agent_a._last_action = out.resolved_action_a
        agent_b._last_action = out.resolved_action_b
        state = out.state
        
if __name__ == "__main__":
    run_sim()

import sys
import time
from src.competitive.state import Action
from src.competitive.agent_a import AgentA
from src.competitive.agent_b import AgentB
from src.competitive.transition import resolve_joint_action_outcome
from src.competitive.parser import parse_competitive_map

def run_sim():
    state, board = parse_competitive_map('maps/competitive/dense_goals.txt')
    agent_a = AgentA()
    agent_b = AgentB()
    
    for i in range(15):
        print(f"\n--- STEP {i} ---")
        print(f"A: {state.agent_a}  B: {state.agent_b}")
        print(f"Boxes: {list(state.boxes)}")
        
        act_a = agent_a.choose_action(state, board, max_steps=20)
        act_b = agent_b.choose_action(state, board, max_steps=20)
        
        print(f"A chose: {act_a.name}")
        print(f"B chose: {act_b.name}")
        
        out = resolve_joint_action_outcome(state, act_a, act_b, board)
        if out.conflict:
            print(">>> CONFLICT! <<<")
            
        state = out.state
        
run_sim()

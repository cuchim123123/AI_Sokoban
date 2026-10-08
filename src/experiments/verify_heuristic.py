from src.single.core.parser import parse_map
from src.single.core.actions import get_successors
from src.single.heuristics.push_distance import precompute_push_costs
from src.single.heuristics.matching import MatchingHeuristic
from src.single.search.ucs import UniformCostSearch
from src.single.heuristics.deadlock import is_deadlock
from collections import deque
import sys

def verify_properties(map_file):
    print(f"Verifying heuristic on {map_file}...")
    state, board = parse_map(map_file)
    push_costs = precompute_push_costs(board)
    heuristic = MatchingHeuristic(board, push_costs)
    
    visited = set()
    queue = deque([state])
    visited.add(state)
    
    states_to_test = []
    
    while queue:
        curr = queue.popleft()
        states_to_test.append(curr)
        
        if len(states_to_test) >= 200:
            break
            
        for action, next_state in get_successors(curr, board):
            if not is_deadlock(next_state, board) and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                
    violations_admissibility = 0
    violations_consistency = 0
    max_admissibility_diff = 0
    
    for s in states_to_test:
        h_s = heuristic(s)
        
        actions, true_cost, _, _ = UniformCostSearch().search(s, board)
        if actions is not None:
            if h_s > true_cost:
                violations_admissibility += 1
            max_admissibility_diff = max(max_admissibility_diff, true_cost - h_s)
            
        for action, s_prime in get_successors(s, board):
            if is_deadlock(s_prime, board):
                continue
            h_s_prime = heuristic(s_prime)
            c = 1
            if h_s > c + h_s_prime:
                violations_consistency += 1
                
    print(f"States tested: {len(states_to_test)}")
    print(f"Admissibility violations: {violations_admissibility}")
    print(f"Consistency violations: {violations_consistency}")
    print(f"Max true_cost - h(s) margin: {max_admissibility_diff}")
    
    with open("results/verification_results.txt", "w") as f:
        f.write(f"Map: {map_file}\n")
        f.write(f"States tested: {len(states_to_test)}\n")
        f.write(f"Admissibility violations: {violations_admissibility}\n")
        f.write(f"Consistency violations: {violations_consistency}\n")
        f.write(f"Max true_cost - h(s) margin: {max_admissibility_diff}\n")

if __name__ == "__main__":
    verify_properties("maps/test_solvable.txt" if len(sys.argv) == 1 else sys.argv[1])

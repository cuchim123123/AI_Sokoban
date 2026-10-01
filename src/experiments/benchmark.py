import time
import csv
from src.core.parser import parse_map
from src.search.ucs import UniformCostSearch
from src.search.astar import AStarSearch
from src.heuristics.push_distance import precompute_push_costs
from src.heuristics.matching import MatchingHeuristic

def run_benchmarks(map_files, output_csv="results/benchmark_results.csv"):
    results = []
    
    for map_file in map_files:
        print(f"Benchmarking {map_file}...")
        state, board = parse_map(map_file)
        
        # UCS
        start_time = time.time()
        actions_ucs, cost_ucs, gen_ucs, exp_ucs = UniformCostSearch().search(state, board)
        time_ucs = time.time() - start_time
        
        # A*
        push_costs = precompute_push_costs(board)
        heuristic = MatchingHeuristic(board, push_costs)
        
        start_time = time.time()
        actions_astar, cost_astar, gen_astar, exp_astar = AStarSearch(heuristic).search(state, board)
        time_astar = time.time() - start_time
        
        assert cost_ucs == cost_astar, f"Cost mismatch on {map_file}: UCS={cost_ucs}, A*={cost_astar}"
        
        results.append({
            "Map": map_file,
            "Algorithm": "UCS",
            "Cost": cost_ucs,
            "Length": len(actions_ucs) if actions_ucs else 0,
            "Generated": gen_ucs,
            "Expanded": exp_ucs,
            "Time (s)": round(time_ucs, 4)
        })
        
        results.append({
            "Map": map_file,
            "Algorithm": "A*",
            "Cost": cost_astar,
            "Length": len(actions_astar) if actions_astar else 0,
            "Generated": gen_astar,
            "Expanded": exp_astar,
            "Time (s)": round(time_astar, 4)
        })
        
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Map", "Algorithm", "Cost", "Length", "Generated", "Expanded", "Time (s)"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    run_benchmarks(["maps/test_solvable.txt"])

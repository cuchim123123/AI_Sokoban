import sys
import os

# Ensure the root directory is in the path so 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_help():
    print("Usage: python main.py [command] [args]")
    print("")
    print("Commands:")
    print("  gui [map_file]        - Run the Pygame visualizer (default)")
    print("  benchmark             - Run the UCS vs A* benchmark on all maps")
    print("  verify [map_file]     - Run the heuristic verification experiment")
    print("")
    print("Examples:")
    print("  python main.py")
    print("  python main.py gui maps/benchmark_1.txt")
    print("  python main.py benchmark")
    print("  python main.py verify maps/test_solvable.txt")

def check_dependencies():
    try:
        import pygame
        import scipy
        import numpy
    except ImportError as e:
        print(f"Dependency Error: {e}")
        print("It looks like the dependencies are not installed for this exact version of Python.")
        print("Please run this command to fix it:")
        print("    python -m pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        sys.exit(0)
        
    check_dependencies()
    
    command = "gui"
    if len(sys.argv) > 1 and sys.argv[1] in ["gui", "benchmark", "verify"]:
        command = sys.argv[1]
        args = sys.argv[2:]
    else:
        args = sys.argv[1:]

    if command == "gui":
        from src.gui.app import App
        map_file = args[0] if args else "maps/benchmark_2.txt"
        print(f"Starting GUI with map: {map_file}")
        app = App(map_file)
        app.run()
        
    elif command == "benchmark":
        from src.experiments.benchmark import run_benchmarks
        maps = ["maps/test_solvable.txt", "maps/benchmark_1.txt", "maps/benchmark_2.txt"]
        run_benchmarks(maps)
        
    elif command == "verify":
        from src.experiments.verify_heuristic import verify_properties
        map_file = args[0] if args else "maps/test_solvable.txt"
        verify_properties(map_file)

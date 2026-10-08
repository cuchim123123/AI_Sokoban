# Sokoban Single-Agent Solver

An AI-powered Sokoban game solver implemented in Python, featuring Uniform Cost Search (UCS) and A* Search.

## Features
- Complete single-agent Sokoban rule implementation.
- Immutable state representation for efficient hashing.
- Uniform Cost Search (UCS).
- A* Search with a custom admissible and consistent heuristic (Minimum Legal Push Distance Matching).
- Static deadlock detection (corners, wall lines) integrated directly into the heuristic calculation.
- Dynamic deadlock detection (2x2 blocked squares) for search space pruning.
- Comprehensive experiment framework for benchmarking and verifying heuristic properties.
- Interactive Pygame GUI for visualizing solutions step-by-step.

## Prerequisites
- Python 3.11+
- Requirements listed in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Running the GUI
```bash
$env:PYTHONPATH="."
python src/gui/app.py maps/benchmark_2.txt
```
Controls in GUI:
- `1`: Solve using UCS
- `2`: Solve using A*
- `Left Arrow`: Previous step
- `Right Arrow`: Next step

## Running Tests
Unit tests cover parser, state management, actions, and search algorithms.
```bash
$env:PYTHONPATH="."
python -m unittest discover tests
```

## Running Experiments

### Benchmarking (UCS vs A*)
Benchmarks the algorithms on multiple maps, measuring runtime, expanded states, generated states, and solution cost. Results are saved to `results/benchmark_results.csv`.
```bash
$env:PYTHONPATH="."
python -c "from src.experiments.benchmark import run_benchmarks; run_benchmarks(['maps/test_solvable.txt', 'maps/benchmark_1.txt', 'maps/benchmark_2.txt'])"
```

### Heuristic Verification
Empirically verifies the admissibility and consistency of the A* heuristic by exploring the state space and comparing heuristic estimates against optimal paths found by UCS.
```bash
$env:PYTHONPATH="."
python src/experiments/verify_heuristic.py maps/benchmark_2.txt
```

## Architecture
- `src/core/`: Domain logic (parsing, states, action validation).
- `src/heuristics/`: Heuristic calculations (reverse push BFS, bipartite matching, deadlocks).
- `src/search/`: Search algorithms (UCS, A*).
- `src/experiments/`: Evaluation scripts.
- `src/gui/`: Pygame application.

## Competitive simultaneous mode

Run `py -3.11 main.py competitive maps/competitive/arena_open.txt 50` on
Windows (or use a Python installation with `requirements.txt` installed).
Both agents choose from the same starting state and resolve one joint round.
The AI uses iterative-deepening pure-action maximin with a one-second decision
budget and the same fallback policy as live execution. Human inputs are WASD
and arrow keys; WAIT is only a forced-immobility outcome.

See [the current AI rewrite report](docs/competitive_rewrite.md) for rules,
search approximations, validation and measured comparisons. Earlier audit
and optimization documents describe previous implementations; their GBFS,
alternating-turn and hard-filter proposals do not describe the current AI.

Run all tests with `python -m unittest discover -s tests`. Run
`python debug2.py` to print root values, completed horizons, fallbacks and
executed outcomes for a fixed decision. The benchmark command is
`python -m src.experiments.competitive_benchmark --help`.

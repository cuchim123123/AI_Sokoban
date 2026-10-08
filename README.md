# Sokoban — Puzzle and Competitive Modes

A Python/Pygame game with two modes:

- **Puzzle:** watch one agent solve Sokoban using UCS or A* with a minimum push-distance matching heuristic.
- **Competitive:** play or watch two agents choosing simultaneous moves, competing for current goal-box credit within a round limit. The AI uses iterative-deepening maximin search; B yields after two repeated configuration cycles.

## Requirements

- **Python 3.11 recommended.** This project has been tested with Python 3.11.9 and Pygame 2.6.1. Other Python versions are not verified here.
- A graphical desktop to display the game.
- Dependencies in `requirements.txt`: pygame, scipy, and numpy.
- Git if cloning the repository; alternatively download and extract it as a ZIP.

Run all commands from the project root, where `main.py`, `maps/`, and `src/` are located. Maps and assets currently use paths relative to this directory.

## Setup on Windows (PowerShell)

Install Python 3.11 first. Then clone the repository, or skip the first two commands if you already extracted a ZIP and open a terminal in its folder:

```powershell
git clone https://github.com/cuchim123123/AI_Sokoban.git
cd AI_Sokoban
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

These commands use the virtual environment's interpreter directly, so PowerShell script activation is unnecessary. After closing the terminal, return to the project folder and launch again with:

```powershell
.\.venv\Scripts\python.exe main.py
```

If the `py` launcher is unavailable but `python --version` reports Python 3.11, use `python -m venv .venv` instead.

## Setup on macOS / Linux

With Python 3.11 installed, run:

```bash
git clone https://github.com/cuchim123123/AI_Sokoban.git
cd AI_Sokoban
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python main.py
```

These platforms have not been validated in this project's current local test run. On Linux, install your distribution's Python venv/pip support if environment creation reports that it is missing.

## Playing

The launcher offers **1 PLAYER** and **2 PLAYERS**. Click a card, press `1`/`2`, or press Enter for puzzle mode.

### Puzzle mode

Choose a map and click **UCS** or **A***, then click **START** or press Enter. Search runs in the background; the resulting solution plays automatically. The HUD shows solution cost, generated states, expanded states, and search time. Cost includes both walking and pushing actions.

### Competitive mode

Choose a map, set each agent to **AI** or **HUMAN**, and adjust the round limit with the steps buttons. Start the match with **START GAME** or Enter.

Both actions resolve together. Odd remaining rounds favor A in a conflict; even remaining rounds favor B. A credited box on a goal contributes one point. Moving it off removes that credit; delivering it again awards credit to the pusher. The round limit determines the final result, even if every goal is occupied. Equal final scores produce a draw.

An amber notice below B marks an active loop-breaking decision. Conflict priority appears separately above the priority agent. AI directions can be diverted by joint conflict resolution. WAIT is only a forced-immobility outcome.

### Controls

| Control | Puzzle | Competitive |
|---|---|---|
| Space | Pause/resume replay; return to menu at the final result | Pause/resume match; return to menu at the final result |
| Comma `,` | Previous replay step | Previous recorded round |
| Period `.` | Next replay step | Next recorded round |
| R | Replay the found solution | Restart the match |
| M / Escape | Return to setup | Return to setup |
| WASD | — | Human agent A |
| Arrow keys | — | Human agent B |
| Close window | Quit | Quit |

The setup screen's Back/Escape returns to the launcher when embedded; standalone competitive mode exits instead. Puzzle mode is an AI solution viewer rather than manual player control.

## Command-line examples

After setup, use the virtual environment's Python in place of `python` below (`.\.venv\Scripts\python.exe` on Windows; `./.venv/bin/python` on macOS/Linux).

```bash
python main.py --help
python main.py gui maps/test_solvable.txt
python main.py competitive maps/competitive/arena_open.txt 50
python main.py competitive maps/competitive/test_race.txt 30 human AI
```

`gui` preselects a puzzle map in the unified launcher. `competitive` opens the competitive setup directly with the supplied map, round limit, and optional A/B controller types. Controller arguments are exactly `human` or `AI`.

## Tests and optional experiments

```bash
python -B -m unittest discover -s tests
python main.py verify maps/test_solvable.txt
python main.py benchmark
python -m src.experiments.competitive_benchmark --help
```

The test suite covers puzzle search, competitive conflicts/scoring, evaluation, caches, deadline behavior, B's loop breaker, and human input. Larger puzzle benchmarks and heuristic verification can take considerably longer than playing a small map; puzzle search has no time limit.

Experiment commands create `results/` as needed:

- Puzzle benchmark: `results/benchmark_results.csv`.
- Sampled heuristic verification: `results/verification_results.txt`.
- Competitive comparison: `results/competitive_comparison.json` by default. Baseline matches require a compatible external baseline snapshot; one is not bundled.

The competitive comparison script uses the base controller with either perspective, so its matches do not exercise AgentB's loop-breaker extension.

## Custom maps

Place puzzle maps in `maps/` and competitive maps in `maps/competitive/`. Menus discover `.txt` maps when the application is constructed; relaunch after adding a file.

| Symbol | Meaning |
|---|---|
| `%` | Wall |
| `.` or space | Floor |
| `A` | Puzzle agent or competitive agent A |
| `E` | Competitive agent B |
| `B` | Box |
| `D` | Goal |
| `C` | Box already on a goal, puzzle mode only |

Use rectangular maps with enclosing walls, exactly one start per required agent, and equal box/goal counts for puzzles. Competitive maps require at least one goal. Keep agents, boxes, and goals within the playable board. Parsers do not fully validate malformed maps; unknown characters do not create entities. Initial competitive boxes have no owner credit.

## Troubleshooting

- **Missing pygame/scipy/numpy:** install using the same interpreter that launches the game: `python -m pip install -r requirements.txt`, replacing `python` with the environment path above. Installing with a different Python version does not fix the active environment.
- **Map not found or missing graphics:** change to the project root and retain the bundled `maps/` and `src/assets/` folders.
- **PowerShell blocks activation:** use `.\.venv\Scripts\python.exe` directly; activation is optional.
- **Dependency installation fails on another Python version:** recreate the environment with Python 3.11, the tested version.
- **Puzzle search seems slow:** try `maps/test_solvable.txt` with A* first. Larger Sokoban state spaces can require substantial time and memory.
- **No window on a server:** the GUI requires a graphical desktop; the automated tests and experiment commands can be run separately.

## Code layout and further reading

- `main.py`: launch and command dispatch.
- `src/single/`: puzzle state, parser, actions, UCS/A*, heuristics, and GUI.
- `src/competitive/`: simultaneous rules, state/preprocessing, evaluation, preferences, A/B controllers, and GUI.
- `src/shared/`: common widgets, animation, rendering helpers, and mode launcher.
- `src/experiments/`: benchmarks and sampled heuristic verification.
- `tests/`: automated regression checks.
- `maps/` and `src/assets/`: level data, textures, cardinal sprite sheets, and frame metadata.

See [the project defense guide](docs/dive.md), [the OOP review](docs/oop.md), and [competitive rule notes](src/competitive/rules.md). The implementation is authoritative where older rule proposals or snapshot documentation differ. Kenney board-texture license information is retained in `src/assets/soko/License.txt`.

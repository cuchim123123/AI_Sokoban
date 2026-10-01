# Sokoban AI Assignment --- Source Context

## Source

Ton Duc Thang University, Faculty of Information Technology\
Course: 503043 --- Introduction to AI\
Midterm Presentation, 3-week project.

The source PDF defines Task 1 as Sokoban worth 8.0 points.

## Original single-agent problem

The assignment says students apply search strategies to enable **the
agent** to move boxes to their designated positions.

Input: - Path to a layout file such as `example_map.txt`.

Output: - List of actions: `North`, `East`, `West`, `South`. - Total
cost.

Map symbols: - `%` = obstacle/wall - `A` = initial location of the
agent - `B` = box - `D` = designated position / goal - `C` = box
currently on a designated position - space = blank cell

## Requirements 1--5

### Requirement 1 --- State-space formulation (1.0 point)

Formulate the problem as a state-space search problem and determine the
details of its relevant components.

The context should therefore cover: - state representation - initial
state - actions/operators - transition model - goal test - path/action
cost

### Requirement 2 --- UCS and A\* (1.0 point)

Implement: - Uniform Cost Search (UCS) - A\*

Propose a heuristic for A\*.

**Explicit restriction:** Euclidean and Manhattan distances are not
allowed.

### Requirement 3 --- Experimental comparison (1.0 point)

Design and implement an experiment comparing UCS and A\* in: - time
complexity / runtime - space complexity / memory/search-space usage

### Requirement 4 --- Heuristic properties (1.0 point)

Discuss the proposed heuristic's: - admissibility - consistency

Design and implement experiments to verify these properties and report
experimental results.

### Requirement 5 --- Pygame GUI (1.0 point)

Implement a user-friendly Pygame GUI.

The GUI must: - provide UCS and A\* as the two algorithm options -
display number of actions - allow pause - allow moving forward/backward
using the specified keys - follow OOP - be compact, well-structured and
reasonable - run on macOS 13.7.8 (Ventura), Intel Core i5 - have
relevant Python library versions verified

## Important scope boundary

This file intentionally focuses on the **original single-agent Sokoban
problem**.

Do not mix the competitive two-agent rules into Requirements 1--5.

The competitive version is introduced separately in Requirement 6.

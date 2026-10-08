"""Reproducible decision and seat-swapped match comparisons.

Run with python -m src.experiments.competitive_benchmark --help.
An optional external baseline snapshot contains evaluation.py, transition.py,
and agent_a.py. Its missing _joint_prefs hook is repaired only in memory using
its documented one-sided fallback model. The live engine is always current.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from src.competitive import agent_a as current
from src.competitive.parser import parse_competitive_map
from src.competitive.transition import resolve_joint_action_outcome, clear_cache
from src.competitive.evaluation import clear_cache as clear_evaluation


def load_baseline(directory):
    saved = {}
    try:
        for name in ("evaluation", "transition", "agent_a"):
            fullname = "src.competitive." + name
            saved[fullname] = sys.modules.get(fullname)
            spec = importlib.util.spec_from_file_location(fullname, Path(directory) / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)
        baseline = module
        if not hasattr(baseline._Planner, "_joint_prefs"):
            baseline._Planner._joint_prefs = lambda self, s, a, b: self._round_prefs_for(s, a)
        return baseline
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def run(args):
    baseline = load_baseline(args.baseline_dir) if args.baseline_dir else None
    report = dict(budget=args.budget, rounds=args.rounds,
                  baseline_dir=args.baseline_dir,
                  baseline_repair="missing _joint_prefs -> _round_prefs_for, in memory only"
                  if baseline else None, decisions=[], matches=[])
    for name in args.maps:
        state, board = parse_competitive_map(f"maps/competitive/{name}.txt")
        if not args.matches_only:
            for label, engine in (("rewrite", current), ("baseline", baseline)):
                if engine is None:
                    continue
                for who in ("A", "B"):
                    clear_cache()
                    clear_evaluation()
                    started = time.perf_counter()
                    action = engine.best_action(state, board, args.rounds, who, time_limit=args.budget,
                                                eval_cache={})
                    report["decisions"].append(dict(
                        case=name, engine=label, perspective=who, action=action.name,
                        elapsed=time.perf_counter()-started, search=dict(engine.LAST_SEARCH)))
        if args.matches and baseline:
            for new_side in ("A", "B"):
                clear_cache()
                clear_evaluation()
                agents = {}
                for side in ("A", "B"):
                    engine = current if side == new_side else baseline
                    agents[side] = engine.AgentA(args.budget)
                    agents[side].perspective = side
                s = state
                turns = []
                while not s.is_terminal(args.rounds):
                    actions, prefs, elapsed, depths = {}, {}, {}, {}
                    for side in ("A", "B"):
                        started = time.perf_counter()
                        actions[side] = agents[side].choose_action(s, board, args.rounds)
                        prefs[side] = agents[side].preference_list(s, board, args.rounds, actions[side])
                        elapsed[side] = time.perf_counter()-started
                        engine = current if side == new_side else baseline
                        depths[side] = engine.LAST_SEARCH.get("window",0)
                    out = resolve_joint_action_outcome(s, actions["A"], actions["B"], board,
                                                       args.rounds, prefs["A"], prefs["B"])
                    assert out.state.agent_a != out.state.agent_b
                    assert out.state.agent_a not in out.state.boxes and out.state.agent_b not in out.state.boxes
                    assert len(out.state.boxes) == len(s.boxes)
                    turns.append(dict(step=s.step, submitted={k:v.name for k,v in actions.items()},
                                      executed=dict(A=out.resolved_action_a.name,B=out.resolved_action_b.name),
                                      preferences={k:[a.name for a in v] for k,v in prefs.items()},
                                      elapsed=elapsed, completed_depth=depths, conflict=out.conflict,
                                      score=[out.state.score_a(),out.state.score_b()]))
                    s = out.state
                match = dict(case=name,rewrite_side=new_side,score_a=s.score_a(),score_b=s.score_b(),turns=turns)
                report["matches"].append(match)
                print(f"{name} rewrite={new_side}: {s.score_a()}:{s.score_b()}",flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--budget",type=float,default=1.0)
    parser.add_argument("--rounds",type=int,default=50)
    parser.add_argument("--maps",nargs="+",default=["arena_open","capacity_lab","main","corridors","test_race"])
    parser.add_argument("--matches",action="store_true")
    parser.add_argument("--matches-only",action="store_true")
    parser.add_argument("--output",default="results/competitive_comparison.json")
    run(parser.parse_args())

"""
Agent A controller — Iterative Deepening Minimax/Maximin with Global TT & PV Ordering.
"""
import time
from collections import deque
from typing import Optional, Tuple, Deque, Dict, Any, List

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action
from src.competitive.evaluation import competitive_heuristic

def _step(pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
    if action == Action.NORTH: return (pos[0], pos[1] - 1)
    if action == Action.SOUTH: return (pos[0], pos[1] + 1)
    if action == Action.EAST:  return (pos[0] + 1, pos[1])
    if action == Action.WEST:  return (pos[0] - 1, pos[1])
    return pos

def _single_agent_transition(state: CompetitiveState, act: Action, perspective: str, board: Board) -> CompetitiveState:
    """Simulates a move assuming the opponent doesn't exist (yields)."""
    if act == Action.WAIT:
        return state
        
    pos = state.agent_a if perspective == 'A' else state.agent_b
    dest = _step(pos, act)
    
    # We assume get_valid_actions already cleared dest from walls.
    # What about boxes?
    new_boxes = set(state.boxes)
    new_bga = set(state.boxes_on_goals_a)
    new_bgb = set(state.boxes_on_goals_b)
    
    if dest in state.boxes:
        push_dest = _step(dest, act)
        new_boxes.discard(dest)
        new_boxes.add(push_dest)
        
        # Credit
        if dest in board.goals:
            new_bga.discard(dest)
            new_bgb.discard(dest)
        if push_dest in board.goals:
            if perspective == 'A':
                new_bga.add(push_dest)
            else:
                new_bgb.add(push_dest)
                
    ns = CompetitiveState(
        agent_a=dest if perspective == 'A' else state.agent_a,
        agent_b=dest if perspective == 'B' else state.agent_b,
        boxes=frozenset(new_boxes),
        boxes_on_goals_a=frozenset(new_bga),
        boxes_on_goals_b=frozenset(new_bgb),
        step=state.step + 1
    )
    return ns

# ── Search Constants ──────────────────────────────────────────────────────────
TIME_LIMIT       = 0.90
MAX_SEARCH_DEPTH = 20
LOOP_PENALTY     = 30
# ──────────────────────────────────────────────────────────────────────────────

class _Deadline(Exception): pass

def best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str,
    recent_positions: Deque[Tuple[int, int]],
    ai_type: str,
    tt: Dict[int, Tuple[int, float, Action]],
    heuristic_cache: Dict[int, float],
    time_limit: float = TIME_LIMIT,
    banned_actions: List[Action] = None
) -> Action:
    deadline = time.time() + time_limit

    def _state_value(
        curr: CompetitiveState,
        depth: int,
        is_max: bool,
    ) -> float:
        if time.time() > deadline:
            raise _Deadline()
        
        # Check TT (strict depth match is required to prevent caching bugs)
        board_key = curr._hash ^ (hash(is_max))
        pv_action = None
        if board_key in tt:
            cached_depth, cached_value, cached_action = tt[board_key]
            if cached_depth >= depth:
                return cached_value
            pv_action = cached_action

        if depth == 0 or curr.is_terminal(max_steps):
            if curr._hash in heuristic_cache:
                val = heuristic_cache[curr._hash]
            else:
                val = competitive_heuristic(curr, board, perspective, max_steps)
                heuristic_cache[curr._hash] = val
                
            if ai_type != "aggressive":
                my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
                if my_pos in recent_positions:
                    val -= LOOP_PENALTY
                    
            tt[board_key] = (0, val, Action.WAIT)
            return val

        my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
        op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
        
        if ai_type == "aggressive":
            op_pos = (-1, -1)
        
        if is_max:
            acts = get_valid_actions(my_pos, op_pos, curr.boxes, board)
            if not acts:
                acts = [Action.WAIT]
                
            # PV Ordering
            if pv_action in acts:
                acts.remove(pv_action)
                acts.insert(0, pv_action)
                
            best_val = -float('inf')
            best_act = acts[0]
            
            for act in acts:
                if ai_type == "aggressive":
                    # Assume opponent yields completely (ignore them)
                    ns = _single_agent_transition(curr, act, perspective, board)
                    # Next is max (single-agent search)
                    val = _state_value(ns, depth - 1, True)
                else:
                    # Transition to MIN node
                    val = _min_value(curr, depth, act)
                    
                if val > best_val:
                    best_val = val
                    best_act = act
                    
            tt[board_key] = (depth, best_val, best_act)
            return best_val
            
    def _min_value(curr: CompetitiveState, depth: int, max_act: Action) -> float:
        my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
        op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
        
        acts = get_valid_actions(op_pos, my_pos, curr.boxes, board)
        if not acts:
            acts = [Action.WAIT]
            
        worst_val = float('inf')
        for op_act in acts:
            ns = resolve_joint_action(
                curr, 
                max_act if perspective == 'A' else op_act, 
                op_act if perspective == 'A' else max_act, 
                board
            )
            # Transition to MAX node
            val = _state_value(ns, depth - 1, True)
            if val < worst_val:
                worst_val = val
        return worst_val

    # ITERATIVE DEEPENING
    best_act_overall = Action.WAIT
    reached_depth = 0
    my_pos = state.agent_a if perspective == 'A' else state.agent_b
    op_pos = state.agent_b if perspective == 'A' else state.agent_a
    
    if ai_type == "aggressive":
        op_pos = (-1, -1)
        
    root_acts = get_valid_actions(my_pos, op_pos, state.boxes, board)
    
    if banned_actions:
        root_acts = [a for a in root_acts if a not in banned_actions]
    
    if not root_acts:
        return Action.WAIT
        
    stable_count = 0
    prev_best = None
    
    try:
        for d in range(1, MAX_SEARCH_DEPTH + 1):
            board_key = state._hash ^ (hash(True))
            pv = tt.get(board_key, (0, 0, None))[2]
            if pv in root_acts:
                root_acts.remove(pv)
                root_acts.insert(0, pv)
                
            best_val = -float('inf')
            best_act = root_acts[0]
            
            for act in root_acts:
                if ai_type == "aggressive":
                    ns = _single_agent_transition(state, act, perspective, board)
                    val = _state_value(ns, d - 1, True)
                else:
                    val = _min_value(state, d, act)
                    
                if val > best_val:
                    best_val = val
                    best_act = act
                    
            best_act_overall = best_act
            reached_depth = d
            tt[board_key] = (d, best_val, best_act)
            
            # ── Adaptive Search Depth ──
            if best_act == prev_best:
                stable_count += 1
            else:
                stable_count = 0
            prev_best = best_act
            
            # Stop early if the best action has been stable for 2 depths (so 3 total identical depths)
            # AND the action is just a walking move (not pushing a box).
            # Deep search is unreliable for pure walking because the opponent will modify the board.
            # We want to reserve deep search (and time) for actual box pushes and tactical encounters.
            if d >= 6 and stable_count >= 2:
                dest = _step(my_pos, best_act)
                is_push = (dest in state.boxes)
                if not is_push:
                    break
                    
    except _Deadline:
        pass
        
    print(f"Agent {perspective} [{ai_type}] reached depth {reached_depth}")
    return best_act_overall


class AgentA:
    """
    Agent A controller. Uses a global Transposition Table and Heuristic Cache
    to retain knowledge across turns.
    """

    def __init__(self, ai_type: str = "aggressive"):
        self.ai_type = ai_type
        self._history: Deque[Tuple[int, int]] = deque(maxlen=4)
        self.tt: Dict[int, Tuple[int, float, Action]] = {}
        self.heuristic_cache: Dict[int, float] = {}

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions: List[Action] = None
    ) -> Action:
        
        # Optional: prevent TT from growing to infinity (unlikely in short matches, but safe)
        if len(self.tt) > 500000:
            self.tt.clear()
            
        action = best_action(
            state, board, max_steps,
            perspective='A',
            recent_positions=self._history,
            ai_type=self.ai_type,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=banned_actions
        )
        
        dx, dy = action.value
        expected = (state.agent_a[0] + dx, state.agent_a[1] + dy)
        self._history.append(state.agent_a)
        
        return action

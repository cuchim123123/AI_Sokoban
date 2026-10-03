"""
Agent controller — Greedy Best-First Search (GBFS).
"""
import time
import heapq
import random
from collections import deque
from typing import Optional, Tuple, Deque, Dict, Any, List

from src.competitive.state import Action, Board, CompetitiveState
from src.competitive.transition import get_valid_actions, resolve_joint_action_outcome
from src.competitive.evaluation import competitive_heuristic

# ── Search Constants ──────────────────────────────────────────────────────────
TIME_LIMIT = 0.90
# ──────────────────────────────────────────────────────────────────────────────

def _get_greedy_opp_act(curr: CompetitiveState, curr_my_pos, curr_op_pos, board: Board, perspective: str, max_steps: int, heuristic_cache: Dict[int, float]) -> Action:
    opp_acts = get_valid_actions(curr_op_pos, curr_my_pos, curr.boxes, board)
    if not opp_acts:
        return Action.WAIT
    random.shuffle(opp_acts)
    
    # Opponent wants to MINIMIZE our heuristic (since heuristic is zero-sum)
    best_val_for_opp = float('inf') 
    best_act = opp_acts[0]
    
    for act in opp_acts:
        act_a = Action.WAIT if perspective == 'A' else act
        act_b = act if perspective == 'A' else Action.WAIT
        
        ns = resolve_joint_action_outcome(curr, act_a, act_b, board).state
        if ns._hash in heuristic_cache:
            val = heuristic_cache[ns._hash]
        else:
            val = competitive_heuristic(ns, board, perspective, max_steps)
            heuristic_cache[ns._hash] = val
            
        if val < best_val_for_opp:
            best_val_for_opp = val
            best_act = act
            
    return best_act


def best_action(
    state: CompetitiveState,
    board: Board,
    max_steps: int,
    perspective: str,
    recent_positions: Deque[Tuple[int, int]],
    tt: Dict[int, Tuple[int, float, Action]],
    heuristic_cache: Dict[int, float],
    time_limit: float = TIME_LIMIT,
    banned_actions: List[Action] = None
) -> Action:
    deadline = time.time() + time_limit
    
    # Priority queue for GBFS: (-heuristic_value, tiebreaker, state, first_action)
    pq = []
    tiebreaker = 0
    
    visited = set()
    visited.add(state._hash)
    
    my_pos = state.agent_a if perspective == 'A' else state.agent_b
    op_pos = state.agent_b if perspective == 'A' else state.agent_a
    
    root_acts = get_valid_actions(my_pos, op_pos, state.boxes, board)
    if banned_actions:
        root_acts = [a for a in root_acts if a not in banned_actions]
    if not root_acts:
        root_acts = [Action.NORTH] # Fallback if totally stuck
    random.shuffle(root_acts)
        
    best_terminal_val = -float('inf')
    best_terminal_act = root_acts[0]
    
    # Initialize PQ with root's successors
    opp_act = _get_greedy_opp_act(state, my_pos, op_pos, board, perspective, max_steps, heuristic_cache)
    for act in root_acts:
        act_a = act if perspective == 'A' else opp_act
        act_b = opp_act if perspective == 'A' else act
        
        out = resolve_joint_action_outcome(state, act_a, act_b, board)
        ns = out.state
        
        if ns._hash in heuristic_cache:
            val = heuristic_cache[ns._hash]
        else:
            val = competitive_heuristic(ns, board, perspective, max_steps)
            heuristic_cache[ns._hash] = val
            
        new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
        if new_my_pos in recent_positions and new_my_pos != my_pos:
            val -= 1000
            
        val -= ns.step * 2.0
        
        if ns.is_terminal(max_steps) and val > best_terminal_val:
            best_terminal_val = val
            best_terminal_act = act
            
        if state.step >= 6:
            print(f"[{perspective}] Root {act} -> val {val} (new_pos {new_my_pos}, recent {list(recent_positions)})")
        heapq.heappush(pq, (-val, tiebreaker, ns, act))
        tiebreaker += 1
        visited.add(ns._hash)
        
    nodes_expanded = 0
    
    while pq and time.time() < deadline:
        neg_val, _, curr, first_act = heapq.heappop(pq)
        
        if curr.is_terminal(max_steps):
            continue
            
        nodes_expanded += 1
        
        curr_my_pos = curr.agent_a if perspective == 'A' else curr.agent_b
        curr_op_pos = curr.agent_b if perspective == 'A' else curr.agent_a
        
        opp_act = _get_greedy_opp_act(curr, curr_my_pos, curr_op_pos, board, perspective, max_steps, heuristic_cache)
        
        acts = get_valid_actions(curr_my_pos, curr_op_pos, curr.boxes, board)
        random.shuffle(acts)
        for act in acts:
            act_a = act if perspective == 'A' else opp_act
            act_b = opp_act if perspective == 'A' else act
            
            out = resolve_joint_action_outcome(curr, act_a, act_b, board)
            ns = out.state
            
            if ns._hash in visited:
                continue
            visited.add(ns._hash)
            
            if ns._hash in heuristic_cache:
                n_val = heuristic_cache[ns._hash]
            else:
                n_val = competitive_heuristic(ns, board, perspective, max_steps)
                heuristic_cache[ns._hash] = n_val
                
            n_val -= ns.step * 2.0
            
            if ns.is_terminal(max_steps) and n_val > best_terminal_val:
                best_terminal_val = n_val
                best_terminal_act = first_act
                
            heapq.heappush(pq, (-n_val, tiebreaker, ns, first_act))
            tiebreaker += 1

    if pq:
        best_frontier_val = -pq[0][0]
        best_frontier_act = pq[0][3]
    else:
        best_frontier_val = -float('inf')
        best_frontier_act = root_acts[0]
        
    if best_terminal_val > best_frontier_val:
        best_act = best_terminal_act
    else:
        best_act = best_frontier_act

    print(f"Agent {perspective} [GBFS] expanded {nodes_expanded} nodes, chose {best_act}")
    return best_act


class AgentA:
    """
    Agent A controller (Unified).
    """
    def __init__(self):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=4)
        self.tt: Dict[int, Tuple[int, float, Action]] = {}
        self.heuristic_cache: Dict[int, float] = {}
        self._last_action: Optional[Action] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    def choose_action(
        self,
        state: CompetitiveState,
        board: Board,
        max_steps: int,
        banned_actions: List[Action] = None
    ) -> Action:
        
        if len(self.heuristic_cache) > 500000:
            self.heuristic_cache.clear()
            
        auto_banned = list(banned_actions) if banned_actions else []
        if self._last_pos == state.agent_a and self._last_action is not None and self._last_action != Action.WAIT:
            auto_banned.append(self._last_action)
            
        action = best_action(
            state, board, max_steps,
            perspective='A',
            recent_positions=self._history,
            tt=self.tt,
            heuristic_cache=self.heuristic_cache,
            banned_actions=auto_banned
        )
        
        self._history.append(state.agent_a)
        self._last_pos = state.agent_a
        self._last_action = action
        
        return action

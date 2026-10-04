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

def _get_greedy_opp_act(
    curr: CompetitiveState,
    curr_my_pos,
    curr_op_pos,
    board: Board,
    perspective: str,
    max_steps: int,
    heuristic_cache: Dict[int, float]
) -> Action:
    """
    Predict the opponent's next action by simulating joint transitions.
    We pair each opponent action with a simple greedy own-action (move toward
    the best-known target) so that conflict rules fire realistically.
    The opponent wants to MINIMISE our heuristic value.
    """
    opp_acts = get_valid_actions(curr_op_pos, curr_my_pos, curr.boxes, board)
    if not opp_acts:
        return Action.WAIT
    random.shuffle(opp_acts)

    # Use WAIT for own action in the opponent model — keeps the model honest
    # without requiring a full nested greedy search (too expensive).
    # This is simpler but already captures conflict rules correctly.
    best_val_for_opp = float('inf')
    best_act = opp_acts[0]

    for act in opp_acts:
        act_a = Action.WAIT if perspective == 'A' else act
        act_b = act        if perspective == 'A' else Action.WAIT

        ns = resolve_joint_action_outcome(curr, act_a, act_b, board, max_steps).state
        key = ns.board_hash
        if key in heuristic_cache:
            val = heuristic_cache[key]
        else:
            val = competitive_heuristic(ns, board, perspective, max_steps)
            heuristic_cache[key] = val

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
        root_acts = [Action.NORTH]
    random.shuffle(root_acts)
        
    root_action_best_val = {act: -float('inf') for act in root_acts}
    
    # Initialize PQ with root's successors
    opp_act = _get_greedy_opp_act(state, my_pos, op_pos, board, perspective, max_steps, heuristic_cache)
    for act in root_acts:
        act_a = act if perspective == 'A' else opp_act
        act_b = opp_act if perspective == 'A' else act
        
        out = resolve_joint_action_outcome(state, act_a, act_b, board, max_steps)
        ns = out.state
        
        # ── Cache lookup (keyed on board_hash, which excludes step) ──────────
        # board_hash gives much higher cache hit rate: the same spatial board
        # reached at different steps has the same heuristic value.
        key = ns.board_hash
        if key in heuristic_cache:
            val = heuristic_cache[key]
        else:
            val = competitive_heuristic(ns, board, perspective, max_steps)
            heuristic_cache[key] = val

        new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
        # Penalise revisiting real recent positions (not search-tree positions)
        if new_my_pos in recent_positions and new_my_pos != my_pos:
            val -= 300

        # Smart yielding: if we stood still (e.g. collision), yield if we lack priority
        if new_my_pos == my_pos:
            a_has_priority = ((max_steps - state.step) % 2 != 0)
            we_have_priority = a_has_priority if perspective == 'A' else not a_has_priority
            if not we_have_priority:
                val -= 10.0  # Big penalty: forces yielding
            else:
                val -= 0.1   # Tiny penalty: holds ground

        # Small step cost to prefer shorter paths (much softer than before)
        val -= (ns.step - state.step) * 0.3
        
        if val > root_action_best_val[act]:
            root_action_best_val[act] = val
            
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
            
            out = resolve_joint_action_outcome(curr, act_a, act_b, board, max_steps)
            ns = out.state
            
            if ns._hash in visited:
                continue
            visited.add(ns._hash)
            
            # ── Cache lookup (board_hash) ─────────────────────────────────
            key = ns.board_hash
            n_val = heuristic_cache.setdefault(
                key, competitive_heuristic(ns, board, perspective, max_steps)
            )

            new_my_pos = ns.agent_a if perspective == 'A' else ns.agent_b
            if new_my_pos in recent_positions and new_my_pos != curr_my_pos:
                n_val -= 300
                
            # Smart yielding: if we stood still (e.g. collision), yield if we lack priority
            if new_my_pos == curr_my_pos:
                a_has_priority = ((max_steps - curr.step) % 2 != 0)
                we_have_priority = a_has_priority if perspective == 'A' else not a_has_priority
                if not we_have_priority:
                    n_val -= 10.0  # Big penalty: forces yielding
                else:
                    n_val -= 0.1   # Tiny penalty: holds ground

            # Small step cost: prefer shorter paths but not at the expense of strategy.
            # We must accumulate this penalty from the ROOT, otherwise deep 6-step loops
            # will have the same penalty as 1-step moves, causing pointless wandering.
            n_val -= (ns.step - state.step) * 0.3
            
            if n_val > root_action_best_val[first_act]:
                root_action_best_val[first_act] = n_val
                
            heapq.heappush(pq, (-n_val, tiebreaker, ns, first_act))
            tiebreaker += 1

    best_act = max(root_action_best_val, key=root_action_best_val.get)

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

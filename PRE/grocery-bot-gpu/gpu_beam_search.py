"""GPU-accelerated beam search for grocery bot optimization.

Uses PyTorch CUDA tensors for massive parallel state evaluation.
Single-bot: beam width 10,000-50,000+ with ALL valid actions per round.

On RTX 5090: ~50M state evaluations in <5 seconds (vs ~6K/s on CPU).

Usage:
    python gpu_beam_search.py [difficulty] [--seed SEED] [--beam 10000] [--capture]
"""
import time
import sys
import numpy as np
import torch

from game_engine import (
    init_game, init_game_from_capture, step as cpu_step,
    GameState, MapState, Order,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
)
from pathfinding import precompute_all_distances
from precompute import PrecomputedTables
from configs import CONFIGS


class GPUBeamSearcher:
    """Fully vectorized GPU beam search for single-bot grocery game.

    State representation (all tensors have batch dimension B):
        bot_x, bot_y: [B] int16 — bot position
        bot_inv: [B, 3] int8 — inventory (-1=empty, 0-15=type_id)
        active_idx: [B] int32 — index into all_orders for current active order
        active_del: [B, 6] int8 — delivered state for active order (0/1)
        score: [B] int32 — cumulative score
        orders_comp: [B] int32 — orders completed count
    """

    def __init__(self, map_state, all_orders, device='cuda', num_bots=1,
                 locked_trajectories=None, pipeline_mode=False, pipeline_depth=1,
                 preferred_types=None):
        self.device = device
        self.ms = map_state
        self.all_orders = all_orders
        self.num_items = map_state.num_items
        self.num_orders = len(all_orders)
        self.num_types = map_state.num_types
        self.W = map_state.width
        self.H = map_state.height
        self.drop_x = map_state.drop_off[0]
        self.drop_y = map_state.drop_off[1]
        self.num_bots = num_bots
        self.pipeline_mode = pipeline_mode  # pre-fetches preview items proactively
        # pipeline_depth: which order ahead to target (1=order+1, 2=order+2, etc.)
        # Only used when pipeline_mode=True.
        self.pipeline_depth = max(1, pipeline_depth)
        # preferred_types: set of item type IDs this bot should specialize in.
        # Soft hint (bonus) to avoid competition between bots for the same types.
        # Set via compute_type_assignments() in gpu_sequential_solver.py.
        self.preferred_types = preferred_types  # set[int] or None

        # === Locked trajectories for sequential multi-bot DP ===
        self.candidate_bot_id = 0  # Set by dp_search()
        if locked_trajectories is not None:
            self.num_locked = locked_trajectories['locked_actions'].shape[0]
            self.locked_actions = torch.tensor(
                locked_trajectories['locked_actions'], dtype=torch.int8, device=device)
            self.locked_action_items = torch.tensor(
                locked_trajectories['locked_action_items'], dtype=torch.int16, device=device)
            self.locked_pos_x = torch.tensor(
                locked_trajectories['locked_pos_x'], dtype=torch.int16, device=device)
            self.locked_pos_y = torch.tensor(
                locked_trajectories['locked_pos_y'], dtype=torch.int16, device=device)
            # CPU numpy copies for fast scalar lookup in _eval (avoid GPU sync overhead)
            self._locked_actions_np = locked_trajectories['locked_actions']  # [L, 300] int8
            self._locked_action_items_np = locked_trajectories['locked_action_items']  # [L, 300] int16
            # Map locked index -> real bot ID for correct processing order
            self.locked_bot_ids = locked_trajectories.get(
                'locked_bot_ids', list(range(self.num_locked)))
            self.locked_idx_map = {
                real_id: idx for idx, real_id in enumerate(self.locked_bot_ids)}
            # Precomputed lookup tables (built in __init__ after item_types is ready)
            self._locked_all_planned_mask = None      # set after item_types upload
            self._locked_remaining_planned = None     # [MAX_ROUNDS+1, num_types] bool GPU
        else:
            self.num_locked = 0
            self.locked_bot_ids = []
            self.locked_idx_map = {}
            self._locked_actions_np = None
            self._locked_action_items_np = None
            self._locked_all_planned_mask = None
            self._locked_remaining_planned = None

        # Actions per state: wait + 4 moves + dropoff + num_items pickups
        self.num_actions = 6 + self.num_items

        t0 = time.time()

        # === Upload static map data to GPU ===
        self.grid = torch.tensor(map_state.grid, dtype=torch.int8, device=device)
        self.item_pos_x = torch.tensor(
            map_state.item_positions[:, 0], dtype=torch.int16, device=device)
        self.item_pos_y = torch.tensor(
            map_state.item_positions[:, 1], dtype=torch.int16, device=device)
        self.item_types = torch.tensor(
            map_state.item_types, dtype=torch.int8, device=device)

        # === Precompute locked-trajectory lookup tables (avoid per-round alloc in _eval) ===
        if self.num_locked > 0 and self._locked_actions_np is not None:
            _itp = self.item_types.cpu().numpy().astype(np.int16)  # [num_items]
            _LOOKAHEAD = 60

            # _locked_all_planned_mask: bool [num_types] — True if any locked bot ever picks
            planned_set_all = set()
            for lb in range(self.num_locked):
                for r in range(MAX_ROUNDS):
                    if self._locked_actions_np[lb, r] == ACT_PICKUP:
                        iidx = int(self._locked_action_items_np[lb, r])
                        if 0 <= iidx < self.num_items:
                            planned_set_all.add(int(_itp[iidx]))
            if planned_set_all:
                mask_t = torch.zeros(self.num_types, dtype=torch.bool, device=device)
                for tp in planned_set_all:
                    if 0 <= tp < self.num_types:
                        mask_t[tp] = True
                self._locked_all_planned_mask = mask_t
            else:
                self._locked_all_planned_mask = None

            # _locked_remaining_planned: bool [MAX_ROUNDS+1, num_types] — for each round r,
            # True if any locked bot plans to pick that type in rounds [r, MAX_ROUNDS).
            # Suffix union: remaining[r] = remaining[r+1] | (any pickup at round r).
            # Used in _eval for accurate coverage: a type is "covered" if a locked bot
            # will still pick it this game (not just within 60 rounds).
            remaining = np.zeros((MAX_ROUNDS + 1, self.num_types), dtype=np.uint8)
            for r in range(MAX_ROUNDS - 1, -1, -1):
                remaining[r] = remaining[r + 1]  # inherit future pickups
                for lb in range(self.num_locked):
                    if self._locked_actions_np[lb, r] == ACT_PICKUP:
                        iidx = int(self._locked_action_items_np[lb, r])
                        if 0 <= iidx < self.num_items:
                            tp = int(_itp[iidx])
                            if 0 <= tp < self.num_types:
                                remaining[r, tp] = 1
            # Upload to GPU as bool [MAX_ROUNDS+1, num_types]
            self._locked_remaining_planned = torch.tensor(
                remaining, dtype=torch.bool, device=device)  # [MAX_ROUNDS+1, num_types]

        # Walkable mask [H, W]
        self.walkable = (
            (self.grid == CELL_FLOOR) | (self.grid == CELL_DROPOFF)
        ).contiguous()

        # Precompute valid move directions [H, W, 4]: up, down, left, right
        valid_moves = torch.zeros(self.H, self.W, 4, dtype=torch.bool, device=device)
        valid_moves[1:, :, 0]  = self.walkable[:-1, :]  # up:   target y-1 walkable
        valid_moves[:-1, :, 1] = self.walkable[1:, :]   # down: target y+1 walkable
        valid_moves[:, 1:, 2]  = self.walkable[:, :-1]  # left: target x-1 walkable
        valid_moves[:, :-1, 3] = self.walkable[:, 1:]   # right: target x+1 walkable
        self.valid_moves = valid_moves  # [H, W, 4]

        # Pack all orders into tensor: [num_orders, MAX_ORDER_SIZE] int8
        order_req = np.full((self.num_orders, MAX_ORDER_SIZE), -1, dtype=np.int8)
        for i, o in enumerate(all_orders):
            for j in range(len(o.required)):
                order_req[i, j] = int(o.required[j])
        self.order_req = torch.tensor(order_req, dtype=torch.int8, device=device)

        # Order sizes (number of actual items, not padding)
        order_sizes = np.array([len(o.required) for o in all_orders], dtype=np.int32)
        self.order_sizes = torch.tensor(order_sizes, dtype=torch.int32, device=device)

        # === Precompute BFS distances via GPU-cached tables ===
        tables = PrecomputedTables.get(map_state)
        gpu_tables = tables.to_gpu_tensors(device)

        self.dist_to_dropoff = gpu_tables['dist_to_dropoff']
        self.dist_to_type = gpu_tables['dist_to_type']
        self.first_step_to_dropoff = gpu_tables['step_to_dropoff']
        self.first_step_to_type = gpu_tables['step_to_type']

        # === Build action expansion pattern (all actions for fallback) ===
        acts = torch.zeros(self.num_actions, dtype=torch.int8, device=device)
        items = torch.full((self.num_actions,), -1, dtype=torch.int16, device=device)
        acts[0] = ACT_WAIT
        acts[1] = ACT_MOVE_UP
        acts[2] = ACT_MOVE_DOWN
        acts[3] = ACT_MOVE_LEFT
        acts[4] = ACT_MOVE_RIGHT
        acts[5] = ACT_DROPOFF
        for i in range(self.num_items):
            acts[6 + i] = ACT_PICKUP
            items[6 + i] = i
        self.action_pattern = acts
        self.action_item_pattern = items

        # === Position-based pickup adjacency lookup for _dp_expand() ===
        # For each walkable cell (y, x), precompute which items are adjacent
        MAX_ADJ = 4  # max items adjacent to any walkable cell (safety margin)
        adj_items = torch.full((self.H, self.W, MAX_ADJ), -1, dtype=torch.int16, device=device)
        adj_count = torch.zeros((self.H, self.W), dtype=torch.int8, device=device)
        for item_idx in range(self.num_items):
            ix, iy = int(self.item_pos_x[item_idx]), int(self.item_pos_y[item_idx])
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx_, ny_ = ix + ddx, iy + ddy
                if 0 <= nx_ < self.W and 0 <= ny_ < self.H:
                    if bool(self.walkable[ny_, nx_]):
                        c = int(adj_count[ny_, nx_])
                        if c < MAX_ADJ:
                            adj_items[ny_, nx_, c] = item_idx
                            adj_count[ny_, nx_] = c + 1
        self.adj_items = adj_items  # [H, W, MAX_ADJ]
        self.adj_count = adj_count  # [H, W]
        self.MAX_ADJ = MAX_ADJ
        self.dp_num_actions = 6 + MAX_ADJ  # wait + 4 moves + dropoff + MAX_ADJ pickups

        # Direction lookup tables
        self.DX = torch.tensor([0, 0, 0, -1, 1, 0, 0], dtype=torch.int32, device=device)
        self.DY = torch.tensor([0, -1, 1, 0, 0, 0, 0], dtype=torch.int32, device=device)

        # Pre-allocate constant tensors for torch.where
        self._neg1_i8 = torch.tensor(-1, dtype=torch.int8, device=device)
        self._zero_i8 = torch.tensor(0, dtype=torch.int8, device=device)
        self._one_i8 = torch.tensor(1, dtype=torch.int8, device=device)

        dt = time.time() - t0
        walkable_cells = int(self.walkable.sum())
        max_adj_actual = int(adj_count.max())
        print(f"  GPU init: {self.num_items} items, {self.num_types} types, "
              f"{walkable_cells} cells, {self.num_actions} acts(full)/{self.dp_num_actions} acts(dp), "
              f"max_adj={max_adj_actual}, {dt:.2f}s",
              file=sys.stderr)

    def _from_game_state(self, gs, bot_id=0):
        """Convert CPU GameState to GPU beam state dict (batch size 1).

        Args:
            bot_id: Which bot to use as the candidate (default 0).
        """
        d = self.device
        state = {
            'bot_x': torch.tensor(
                [int(gs.bot_positions[bot_id, 0])], dtype=torch.int16, device=d),
            'bot_y': torch.tensor(
                [int(gs.bot_positions[bot_id, 1])], dtype=torch.int16, device=d),
            'bot_inv': torch.tensor(
                [[int(gs.bot_inventories[bot_id, s]) for s in range(INV_CAP)]],
                dtype=torch.int8, device=d),
            'active_idx': torch.tensor([0], dtype=torch.int32, device=d),
            'active_del': torch.zeros((1, MAX_ORDER_SIZE), dtype=torch.int8, device=d),
            'score': torch.tensor([0], dtype=torch.int32, device=d),
            'orders_comp': torch.tensor([0], dtype=torch.int32, device=d),
        }
        if self.num_locked > 0:
            state['locked_inv'] = torch.full(
                (1, self.num_locked, INV_CAP), -1, dtype=torch.int8, device=d)
            # Dynamic locked bot positions (start at spawn)
            sx, sy = self.ms.spawn
            state['locked_bx'] = torch.full(
                (1, self.num_locked), sx, dtype=torch.int16, device=d)
            state['locked_by'] = torch.full(
                (1, self.num_locked), sy, dtype=torch.int16, device=d)
        return state

    def _expand(self, state):
        """Expand beam: each state -> num_actions copies with action tensors.

        Returns (expanded_state, actions[B*N], action_items[B*N]).
        """
        N = self.num_actions
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(N)
            elif k == 'locked_inv':
                # [B, num_locked, INV_CAP] -> [B*N, num_locked, INV_CAP]
                expanded[k] = v.repeat_interleave(N, dim=0)
            else:
                expanded[k] = v.repeat_interleave(N, dim=0)

        B = state['bot_x'].shape[0]
        actions = self.action_pattern.repeat(B)
        action_items = self.action_item_pattern.repeat(B)
        return expanded, actions, action_items

    def _dp_expand(self, state):
        """Position-aware expand for dp_search(): only generate valid actions per position.

        Fixed actions (always valid): wait, 4 moves, dropoff = 6
        Variable pickups: 0-MAX_ADJ items adjacent to current position
        Total: 6+MAX_ADJ actions per state (vs 6+num_items in _expand)

        Returns (expanded_state, actions[B*N], action_items[B*N], valid_mask[B*N], N).
        """
        B = state['bot_x'].shape[0]
        d = self.device
        N = self.dp_num_actions  # 6 + MAX_ADJ (e.g., 10)

        # Look up adjacent items for each state's position
        by = state['bot_y'].long()
        bx = state['bot_x'].long()
        per_state_adj = self.adj_items[by, bx]       # [B, MAX_ADJ]
        per_state_count = self.adj_count[by, bx]      # [B]

        # Build action tensor [B, N]
        acts = torch.zeros(B, N, dtype=torch.int8, device=d)
        items = torch.full((B, N), -1, dtype=torch.int16, device=d)

        # Fixed actions: wait(0), up(1), down(2), left(3), right(4), dropoff(5)
        acts[:, 0] = ACT_WAIT
        acts[:, 1] = ACT_MOVE_UP
        acts[:, 2] = ACT_MOVE_DOWN
        acts[:, 3] = ACT_MOVE_LEFT
        acts[:, 4] = ACT_MOVE_RIGHT
        acts[:, 5] = ACT_DROPOFF

        # Variable pickups: per-state adjacent items
        for a in range(self.MAX_ADJ):
            acts[:, 6 + a] = ACT_PICKUP
            items[:, 6 + a] = per_state_adj[:, a]

        # Expand: [B, N] -> [B*N]
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(N)
            else:
                expanded[k] = v.repeat_interleave(N, dim=0)

        actions = acts.reshape(-1)
        action_items = items.reshape(-1)

        # Valid mask: fixed actions always valid; pickups only where adj exists
        valid = torch.ones(B * N, dtype=torch.bool, device=d)
        for a in range(self.MAX_ADJ):
            # Mark pickup slots as invalid where no adjacent item exists
            pickup_idx = torch.arange(B, device=d) * N + (6 + a)
            invalid = per_state_adj[:, a] < 0  # no item at this adj slot
            valid[pickup_idx[invalid]] = False

        # Wall pruning: mask invalid move directions (slots 1-4: UP, DOWN, LEFT, RIGHT)
        move_valid = self.valid_moves[by, bx]  # [B, 4]
        valid.view(B, N)[:, 1:5] &= move_valid

        return expanded, actions, action_items, valid, N

    @torch.no_grad()
    def _smart_expand(self, state, max_cands=20):
        """Generate guided candidates per state using BFS first-step tables.

        Generates move-toward for EACH needed active+preview item type (not just nearest),
        ensuring beam diversity when multiple item types are needed.

        Returns (expanded_state, actions[B*C], action_items[B*C], valid_mask[B*C], C)
        """
        B = state['bot_x'].shape[0]
        d = self.device
        C = max_cands

        bot_x = state['bot_x']
        bot_y = state['bot_y']
        inv = state['bot_inv']
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]
        act_del = state['active_del']

        bx_l = bot_x.long()
        by_l = bot_y.long()
        inv_count = (inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP
        has_items = inv_count > 0
        at_drop = (bot_x == self.drop_x) & (bot_y == self.drop_y)

        # Check for active items in inventory
        has_active_inv = torch.zeros(B, dtype=torch.bool, device=d)
        for s in range(INV_CAP):
            for os in range(MAX_ORDER_SIZE):
                has_active_inv = has_active_inv | (
                    (act_req[:, os] == inv[:, s]) &
                    (act_del[:, os] == 0) & (inv[:, s] >= 0))

        active_remaining = ((act_req >= 0) & (act_del == 0)).sum(dim=1)

        # Preview order (pipeline_depth determines which order ahead to target)
        # pipeline_mode bots target active_idx + pipeline_depth, others target +1
        _pdepth = self.pipeline_depth if self.pipeline_mode else 1
        pidx = (aidx + _pdepth).clamp(0, self.num_orders - 1)
        prev_req = self.order_req[pidx]

        # All action/item slots
        all_acts = torch.zeros((B, C), dtype=torch.int8, device=d)
        all_items = torch.full((B, C), -1, dtype=torch.int16, device=d)
        valid = torch.zeros((B, C), dtype=torch.bool, device=d)
        slot = 0

        # Slot 0: Wait
        all_acts[:, slot] = ACT_WAIT
        valid[:, slot] = True
        slot += 1

        # Slot 1: Dropoff
        all_acts[:, slot] = ACT_DROPOFF
        valid[:, slot] = at_drop & has_items & has_active_inv
        slot += 1

        # Slot 2: Move toward dropoff
        move_drop = self.first_step_to_dropoff[by_l, bx_l]
        all_acts[:, slot] = move_drop
        if self.pipeline_mode or self.num_locked > 0:
            # Pipeline/multi-bot: also valid when holding preview items
            # (pre-fetch bot should position near dropoff for instant auto-delivery)
            has_prev_inv = torch.zeros(B, dtype=torch.bool, device=d)
            for s in range(INV_CAP):
                for os in range(MAX_ORDER_SIZE):
                    has_prev_inv = has_prev_inv | (
                        (prev_req[:, os] == inv[:, s]) & (inv[:, s] >= 0))
            valid[:, slot] = (has_active_inv | has_prev_inv) & (move_drop > 0)
        else:
            valid[:, slot] = has_active_inv & (move_drop > 0)
        slot += 1

        # Slots 3-8: Move toward EACH distinct active item type (up to 6)
        seen_types = set()
        for os in range(MAX_ORDER_SIZE):
            if slot >= C - 6:  # leave room for moves + pickups
                break
            # Get needed type for this order slot
            needed = (act_req[:, os] >= 0) & (act_del[:, os] == 0)
            if not needed.any():
                continue
            # Use the type from the first needing state (for dedup check)
            sample_type = int(act_req[needed.nonzero(as_tuple=True)[0][0], os])
            if sample_type in seen_types:
                continue
            seen_types.add(sample_type)

            ntype = act_req[:, os].long().clamp(0, self.num_types - 1)
            move = self.first_step_to_type[ntype, by_l, bx_l]
            all_acts[:, slot] = move.to(torch.int8)
            valid[:, slot] = needed & has_space & (move > 0)
            slot += 1

        # Move toward EACH distinct preview item type (up to 4)
        seen_prev_types = set()
        for os in range(MAX_ORDER_SIZE):
            if slot >= C - 6:
                break
            pneeded = prev_req[:, os] >= 0
            if not pneeded.any():
                continue
            sample_type = int(prev_req[pneeded.nonzero(as_tuple=True)[0][0], os])
            if sample_type in seen_prev_types or sample_type in seen_types:
                continue
            seen_prev_types.add(sample_type)

            ptype = prev_req[:, os].long().clamp(0, self.num_types - 1)
            move = self.first_step_to_type[ptype, by_l, bx_l]
            all_acts[:, slot] = move.to(torch.int8)
            # Pipeline mode or near-complete: always allow preview moves with space.
            # Otherwise restrict to nearly-done or bot has room to spare.
            if self.pipeline_mode or self.num_locked > 0:
                prev_cond = pneeded & has_space & (move > 0)
            else:
                prev_cond = (pneeded & has_space & (move > 0) &
                             ((active_remaining <= 1) | (inv_count < INV_CAP - 1)))
            valid[:, slot] = prev_cond
            slot += 1

        # Move toward order+2 item types (deep pipeline, multi-bot only)
        # _eval rewards this when active+preview are fully covered by locked bots.
        # Without generating these candidates, the beam can't follow that eval signal.
        if self.num_locked > 0 and slot < C - 4:
            p2idx = (aidx + 2).clamp(0, self.num_orders - 1)
            req_p2 = self.order_req[p2idx]
            seen_p2_types = set()
            for os in range(MAX_ORDER_SIZE):
                if slot >= C - 4:
                    break
                p2needed = req_p2[:, os] >= 0
                if not p2needed.any():
                    continue
                sample_p2 = int(req_p2[p2needed.nonzero(as_tuple=True)[0][0], os])
                if sample_p2 in seen_p2_types or sample_p2 in seen_types or sample_p2 in seen_prev_types:
                    continue
                seen_p2_types.add(sample_p2)
                p2type = req_p2[:, os].long().clamp(0, self.num_types - 1)
                move_p2 = self.first_step_to_type[p2type, by_l, bx_l]
                all_acts[:, slot] = move_p2.to(torch.int8)
                valid[:, slot] = p2needed & has_space & (move_p2 > 0)
                slot += 1

        # Move toward preferred types (type specialization, multi-bot only)
        # Only when bot has space and NOT already carrying active-order items
        # (active items should be delivered first, not detoured for specialization)
        if self.preferred_types is not None and slot < C - 4 and self.num_locked > 0:
            seen_pref = set()
            for pt in sorted(self.preferred_types):  # deterministic order
                if slot >= C - 4:
                    break
                if pt in seen_types or pt in seen_prev_types or pt in seen_pref:
                    continue
                seen_pref.add(pt)
                pt_t = torch.full((B,), pt, dtype=torch.long, device=d)
                move_pt = self.first_step_to_type[pt_t, by_l, bx_l]
                all_acts[:, slot] = move_pt.to(torch.int8)
                valid[:, slot] = has_space & ~has_active_inv & (move_pt > 0)
                slot += 1

        # Precompute order+2 requirements for multi-bot pickup below
        _req_p2_pu = None
        if self.num_locked > 0:
            _p2idx_pu = (aidx + 2).clamp(0, self.num_orders - 1)
            _req_p2_pu = self.order_req[_p2idx_pu]  # [B, MAX_ORDER_SIZE]

        # Pickup adjacent items (active + preview + order+2 in multi-bot)
        for item_idx in range(self.num_items):
            if slot >= C - 4:  # leave room for 4 moves
                break
            ix = self.item_pos_x[item_idx].to(torch.int32)
            iy = self.item_pos_y[item_idx].to(torch.int32)
            adj = ((bot_x.to(torch.int32) - ix).abs() +
                   (bot_y.to(torch.int32) - iy).abs()) == 1
            if not adj.any():
                continue
            type_id = int(self.item_types[item_idx])
            # Check active order match
            matches = torch.zeros(B, dtype=torch.bool, device=d)
            for os in range(MAX_ORDER_SIZE):
                matches = matches | (
                    (act_req[:, os] == type_id) & (act_del[:, os] == 0))
            # Also check preview match
            matches_prev = torch.zeros(B, dtype=torch.bool, device=d)
            for os in range(MAX_ORDER_SIZE):
                matches_prev = matches_prev | (prev_req[:, os] == type_id)
            if self.pipeline_mode or self.num_locked > 0:
                # Multi-bot: also allow picking up items for order+2
                if _req_p2_pu is not None:
                    matches_p2 = torch.zeros(B, dtype=torch.bool, device=d)
                    for os in range(MAX_ORDER_SIZE):
                        matches_p2 = matches_p2 | (_req_p2_pu[:, os] == type_id)
                    can_pick = adj & has_space & (matches | matches_prev | matches_p2)
                else:
                    can_pick = adj & has_space & (matches | matches_prev)
            else:
                can_pick = adj & has_space & (matches | (matches_prev & (active_remaining <= 1)))
            if can_pick.any():
                all_acts[:, slot] = ACT_PICKUP
                all_items[:, slot] = item_idx
                valid[:, slot] = can_pick
                slot += 1

        # 4 directional moves (exploration fallbacks)
        for act_id, dx, dy in [(ACT_MOVE_UP, 0, -1), (ACT_MOVE_DOWN, 0, 1),
                                (ACT_MOVE_LEFT, -1, 0), (ACT_MOVE_RIGHT, 1, 0)]:
            if slot >= C:
                break
            nx = (bot_x.to(torch.int32) + dx)
            ny = (bot_y.to(torch.int32) + dy)
            in_b = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
            ns_y = ny.clamp(0, self.H - 1).long()
            ns_x = nx.clamp(0, self.W - 1).long()
            is_walk = in_b & self.walkable[ns_y, ns_x]
            all_acts[:, slot] = act_id
            valid[:, slot] = is_walk
            slot += 1

        # Trim unused slots
        C_actual = slot
        all_acts = all_acts[:, :C_actual]
        all_items = all_items[:, :C_actual]
        valid = valid[:, :C_actual]

        # Expand state
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(C_actual)
            else:
                expanded[k] = v.repeat_interleave(C_actual, dim=0)

        actions = all_acts.reshape(-1)
        action_items = all_items.reshape(-1)
        valid_mask = valid.reshape(-1)

        return expanded, actions, action_items, valid_mask, C_actual

    @torch.no_grad()
    def _step(self, state, actions, action_items, round_num=-1):
        """Apply actions to batch of states. Returns new state dict.

        Replicates game_engine.step() exactly: processes bots in real ID
        order (0, 1, ..., N-1). Each bot does movement/pickup/dropoff
        before the next, matching CPU collision resolution exactly.

        Args:
            round_num: Current round (required when num_locked > 0).
        """
        B = state['bot_x'].shape[0]
        d = self.device

        bot_x = state['bot_x'].clone()
        bot_y = state['bot_y'].clone()
        bot_inv = state['bot_inv'].clone()
        active_idx = state['active_idx'].clone()
        active_del = state['active_del'].clone()
        score = state['score'].clone()
        orders_comp = state['orders_comp'].clone()
        locked_inv = state['locked_inv'].clone() if 'locked_inv' in state else None
        locked_bx = state['locked_bx'].clone() if 'locked_bx' in state else None
        locked_by = state['locked_by'].clone() if 'locked_by' in state else None

        # === INTERLEAVED PROCESSING: bots in real ID order ===
        if self.num_locked > 0 and round_num >= 0:
            spawn_x, spawn_y = self.ms.spawn

            for real_bid in range(self.num_bots):
                if real_bid == self.candidate_bot_id:
                    # ===== CANDIDATE BOT =====
                    bot_x, bot_y, bot_inv, active_idx, active_del, \
                        score, orders_comp, locked_inv, locked_bx, locked_by = \
                        self._step_candidate(
                            actions, action_items, bot_x, bot_y, bot_inv,
                            active_idx, active_del, score, orders_comp,
                            locked_inv, locked_bx, locked_by, B, d, spawn_x, spawn_y)

                elif real_bid in self.locked_idx_map:
                    b = self.locked_idx_map[real_bid]
                    # ===== LOCKED BOT =====
                    bot_x, bot_y, bot_inv, active_idx, active_del, \
                        score, orders_comp, locked_inv, locked_bx, locked_by = \
                        self._step_locked_bot(
                            b, round_num, bot_x, bot_y, bot_inv,
                            active_idx, active_del, score, orders_comp,
                            locked_inv, locked_bx, locked_by, B, d, spawn_x, spawn_y)
                # else: unplanned bot at spawn (collision-exempt), skip

        else:
            # === SINGLE BOT (no locked) — original fast path ===
            bot_x, bot_y, bot_inv, active_idx, active_del, \
                score, orders_comp = \
                self._step_candidate_only(
                    actions, action_items, bot_x, bot_y, bot_inv,
                    active_idx, active_del, score, orders_comp, B, d)

        result = {
            'bot_x': bot_x,
            'bot_y': bot_y,
            'bot_inv': bot_inv,
            'active_idx': active_idx,
            'active_del': active_del,
            'score': score,
            'orders_comp': orders_comp,
        }
        if locked_inv is not None:
            result['locked_inv'] = locked_inv
        if locked_bx is not None:
            result['locked_bx'] = locked_bx
            result['locked_by'] = locked_by
        return result

    def _vectorized_deliver(self, bot_inv_slice, act_req, active_del, can_deliver, B, d):
        """Vectorized delivery: replace 3x6 nested loop with tensor ops.

        Matches items in inventory against order requirements, delivering
        each inv slot to first matching undelivered order slot.

        Args:
            bot_inv_slice: [B, INV_CAP] int8 inventory
            act_req: [B, MAX_ORDER_SIZE] int8 order requirements
            active_del: [B, MAX_ORDER_SIZE] int8 delivery state
            can_deliver: [B] bool mask — which states can deliver
            B: batch size
            d: device

        Returns:
            (bot_inv_slice, active_del, score_add) — modified in-place where can_deliver
        """
        # [B, INV_CAP, 1] vs [B, 1, MAX_ORDER_SIZE] -> [B, INV_CAP, MAX_ORDER_SIZE]
        inv_exp = bot_inv_slice.unsqueeze(2)    # [B, INV_CAP, 1]
        req_exp = act_req.unsqueeze(1)          # [B, 1, MAX_ORDER_SIZE]
        del_exp = active_del.unsqueeze(1)       # [B, 1, MAX_ORDER_SIZE]
        can_exp = can_deliver.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

        # Full match grid: which (inv_slot, order_slot) pairs can deliver
        match = ((inv_exp == req_exp) & (del_exp == 0) &
                 (inv_exp >= 0) & can_exp)      # [B, INV_CAP, MAX_ORDER_SIZE]

        # Greedy first-match: each inv slot delivers to first matching order slot
        score_add = torch.zeros(B, dtype=torch.int32, device=d)
        used_order = torch.zeros(B, MAX_ORDER_SIZE, dtype=torch.bool, device=d)

        for s in range(INV_CAP):
            available = match[:, s, :] & ~used_order   # [B, MAX_ORDER_SIZE]
            # First available order slot per state (first True only)
            cumsum = available.to(torch.int32).cumsum(dim=1)
            first_match = available & (cumsum == 1)
            any_match = first_match.any(dim=1)          # [B]

            # Update: mark order slot delivered, clear inv slot, add score
            active_del = active_del + first_match.to(torch.int8)
            bot_inv_slice[:, s] = torch.where(any_match, self._neg1_i8, bot_inv_slice[:, s])
            score_add = score_add + first_match.sum(dim=1).to(torch.int32)
            used_order = used_order | first_match

        return bot_inv_slice, active_del, score_add

    def _auto_deliver_all(self, order_complete, bot_x, bot_y, bot_inv,
                          locked_inv, locked_bx, locked_by,
                          active_idx, active_del, score, B, d):
        """Auto-delivery when an order completes: all bots at dropoff deliver to new active."""
        if not order_complete.any():
            return bot_inv, active_del, score, active_idx

        score = score + order_complete.to(torch.int32) * 5
        new_aidx = (active_idx + order_complete.to(torch.int32)).clamp(
            0, self.num_orders - 1)
        oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
        active_del = torch.where(oc_mask, self._zero_i8, active_del)

        new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)
        new_req = self.order_req[new_aidx_l]

        # Auto-deliver ALL locked bots at dropoff (vectorized per bot)
        if locked_bx is not None:
            for b2 in range(self.num_locked):
                b2_at_drop = (locked_bx[:, b2] == self.drop_x) & (locked_by[:, b2] == self.drop_y)
                b2_auto = order_complete & b2_at_drop
                if b2_auto.any():
                    linv_b2 = locked_inv[:, b2, :]  # [B, INV_CAP]
                    linv_b2, active_del, score_add = self._vectorized_deliver(
                        linv_b2, new_req, active_del, b2_auto, B, d)
                    locked_inv[:, b2, :] = linv_b2
                    score = score + score_add

        # Auto-deliver candidate bot at dropoff (vectorized)
        cand_at_drop = (bot_x == self.drop_x) & (bot_y == self.drop_y)
        auto_cand = order_complete & cand_at_drop
        if auto_cand.any():
            bot_inv, active_del, score_add = self._vectorized_deliver(
                bot_inv, new_req, active_del, auto_cand, B, d)
            score = score + score_add
            sort_key = (bot_inv < 0).to(torch.int8)
            _, sort_idx = sort_key.sort(dim=1, stable=True)
            bot_inv = bot_inv.gather(1, sort_idx.long())

        return bot_inv, active_del, score, new_aidx

    def _step_locked_bot(self, b, round_num, bot_x, bot_y, bot_inv,
                         active_idx, active_del, score, orders_comp,
                         locked_inv, locked_bx, locked_by, B, d, spawn_x, spawn_y):
        """Process one locked bot's action (movement/pickup/dropoff)."""
        lb_act = int(self.locked_actions[b, round_num])
        lb_item = int(self.locked_action_items[b, round_num])
        lbx = locked_bx[:, b]
        lby = locked_by[:, b]

        # --- Movement ---
        if ACT_MOVE_UP <= lb_act <= ACT_MOVE_RIGHT:
            lb_dx = int(self.DX[lb_act])
            lb_dy = int(self.DY[lb_act])
            lnx = lbx.to(torch.int32) + lb_dx
            lny = lby.to(torch.int32) + lb_dy

            lb_in_b = (lnx >= 0) & (lnx < self.W) & (lny >= 0) & (lny < self.H)
            lny_s = lny.clamp(0, self.H - 1)
            lnx_s = lnx.clamp(0, self.W - 1)
            lb_walk = lb_in_b & self.walkable[lny_s.long(), lnx_s.long()]

            lb_at_spawn = (lnx == spawn_x) & (lny == spawn_y)
            # Collision with candidate bot
            cand_coll = (lnx == bot_x.to(torch.int32)) & (lny == bot_y.to(torch.int32)) & ~lb_at_spawn
            lb_can_move = lb_walk & ~cand_coll

            # Collision with ALL other locked bots (vectorized)
            if self.num_locked > 1:
                # [B, 1] vs [B, num_locked] -> [B, num_locked]
                all_lb_x = locked_bx.to(torch.int32)  # [B, num_locked]
                all_lb_y = locked_by.to(torch.int32)
                all_coll = ((lnx.unsqueeze(1) == all_lb_x) &
                            (lny.unsqueeze(1) == all_lb_y) &
                            ~lb_at_spawn.unsqueeze(1))
                # Exclude self-collision (column b)
                all_coll[:, b] = False
                any_coll = all_coll.any(dim=1)
                lb_can_move = lb_can_move & ~any_coll

            locked_bx[:, b] = torch.where(lb_can_move, lnx.to(torch.int16), lbx)
            locked_by[:, b] = torch.where(lb_can_move, lny.to(torch.int16), lby)
            lbx = locked_bx[:, b]
            lby = locked_by[:, b]

        # --- Pickup ---
        elif lb_act == ACT_PICKUP and lb_item >= 0:
            item_x = int(self.item_pos_x[lb_item])
            item_y = int(self.item_pos_y[lb_item])
            lb_mdist = (lbx.to(torch.int32) - item_x).abs() + (lby.to(torch.int32) - item_y).abs()
            lb_adjacent = lb_mdist == 1

            pickup_type_lb = self.item_types[lb_item]
            added_lb = torch.zeros(B, dtype=torch.bool, device=d)
            for s in range(INV_CAP):
                slot_empty = locked_inv[:, b, s] < 0
                add_here = slot_empty & ~added_lb & lb_adjacent
                locked_inv[:, b, s] = torch.where(
                    add_here, pickup_type_lb, locked_inv[:, b, s])
                added_lb = added_lb | add_here

        # --- Dropoff ---
        elif lb_act == ACT_DROPOFF:
            lb_at_drop = (lbx == self.drop_x) & (lby == self.drop_y)

            aidx_l = active_idx.long().clamp(0, self.num_orders - 1)
            act_req_lb = self.order_req[aidx_l]

            # Vectorized delivery for locked bot
            linv_b = locked_inv[:, b, :]  # [B, INV_CAP]
            linv_b, active_del, score_add = self._vectorized_deliver(
                linv_b, act_req_lb, active_del, lb_at_drop, B, d)
            locked_inv[:, b, :] = linv_b
            score = score + score_add

            # Compact locked bot inventory
            linv_b = locked_inv[:, b, :]
            sort_key = (linv_b < 0).to(torch.int8)
            _, sort_idx = sort_key.sort(dim=1, stable=True)
            locked_inv[:, b, :] = linv_b.gather(1, sort_idx.long())

            # Check order completion
            act_req_lb = self.order_req[aidx_l]
            slot_done = (active_del == 1) | (act_req_lb < 0)
            has_required = (act_req_lb >= 0).any(dim=1)
            order_complete_lb = slot_done.all(dim=1) & has_required & lb_at_drop

            if order_complete_lb.any():
                orders_comp = orders_comp + order_complete_lb.to(torch.int32)
                bot_inv, active_del, score, active_idx = self._auto_deliver_all(
                    order_complete_lb, bot_x, bot_y, bot_inv,
                    locked_inv, locked_bx, locked_by,
                    active_idx, active_del, score, B, d)

        return (bot_x, bot_y, bot_inv, active_idx, active_del,
                score, orders_comp, locked_inv, locked_bx, locked_by)

    def _step_candidate(self, actions, action_items, bot_x, bot_y, bot_inv,
                        active_idx, active_del, score, orders_comp,
                        locked_inv, locked_bx, locked_by, B, d, spawn_x, spawn_y):
        """Process candidate bot's expanded actions (movement/pickup/dropoff)."""
        acts_i32 = actions.to(torch.int32)

        # === MOVEMENT ===
        is_move = (actions >= 1) & (actions <= 4)
        dx = self.DX[acts_i32.long()]
        dy = self.DY[acts_i32.long()]
        nx = bot_x.to(torch.int32) + dx
        ny = bot_y.to(torch.int32) + dy

        in_bounds = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
        ny_safe = ny.clamp(0, self.H - 1)
        nx_safe = nx.clamp(0, self.W - 1)
        is_walkable = self.walkable[ny_safe.long(), nx_safe.long()]
        can_move = is_move & in_bounds & is_walkable

        # Collision with locked bots (vectorized)
        if locked_bx is not None:
            at_spawn = (nx == spawn_x) & (ny == spawn_y)
            # [B, 1] vs [B, num_locked] -> [B, num_locked]
            all_lb_x = locked_bx.to(torch.int32)  # [B, num_locked]
            all_lb_y = locked_by.to(torch.int32)
            all_coll = ((nx.unsqueeze(1) == all_lb_x) &
                        (ny.unsqueeze(1) == all_lb_y) &
                        ~at_spawn.unsqueeze(1))
            any_coll = all_coll.any(dim=1)  # [B]
            can_move = can_move & ~any_coll

        bot_x = torch.where(can_move, nx.to(torch.int16), bot_x)
        bot_y = torch.where(can_move, ny.to(torch.int16), bot_y)

        # === PICKUP ===
        is_pickup = (actions == ACT_PICKUP)
        item_idx = action_items.long().clamp(0, self.num_items - 1)
        inv_count = (bot_inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP

        ix = self.item_pos_x[item_idx].to(torch.int32)
        iy = self.item_pos_y[item_idx].to(torch.int32)
        mdist = (bot_x.to(torch.int32) - ix).abs() + (bot_y.to(torch.int32) - iy).abs()
        adjacent = mdist == 1

        can_pickup = is_pickup & has_space & adjacent
        pickup_type = self.item_types[item_idx]

        added = torch.zeros(B, dtype=torch.bool, device=d)
        for s in range(INV_CAP):
            slot_empty = bot_inv[:, s] < 0
            add_here = can_pickup & slot_empty & ~added
            bot_inv[:, s] = torch.where(add_here, pickup_type, bot_inv[:, s])
            added = added | add_here

        # === DROPOFF (vectorized) ===
        is_dropoff = (actions == ACT_DROPOFF)
        at_drop = (bot_x == self.drop_x) & (bot_y == self.drop_y)
        has_items = (bot_inv >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items

        aidx = active_idx.long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]

        bot_inv, active_del, score_add = self._vectorized_deliver(
            bot_inv, act_req, active_del, can_dropoff, B, d)
        score = score + score_add

        sort_key = (bot_inv < 0).to(torch.int8)
        _, sort_idx = sort_key.sort(dim=1, stable=True)
        bot_inv = bot_inv.gather(1, sort_idx.long())

        # === ORDER COMPLETION ===
        slot_done = (active_del == 1) | (act_req < 0)
        has_required = (act_req >= 0).any(dim=1)
        order_complete = slot_done.all(dim=1) & can_dropoff & has_required

        if order_complete.any():
            orders_comp = orders_comp + order_complete.to(torch.int32)
            bot_inv, active_del, score, active_idx = self._auto_deliver_all(
                order_complete, bot_x, bot_y, bot_inv,
                locked_inv, locked_bx, locked_by,
                active_idx, active_del, score, B, d)

        return (bot_x, bot_y, bot_inv, active_idx, active_del,
                score, orders_comp, locked_inv, locked_bx, locked_by)

    def _step_candidate_only(self, actions, action_items, bot_x, bot_y, bot_inv,
                             active_idx, active_del, score, orders_comp, B, d):
        """Process candidate bot only (no locked bots). Original fast path."""
        acts_i32 = actions.to(torch.int32)

        # === MOVEMENT ===
        is_move = (actions >= 1) & (actions <= 4)
        dx = self.DX[acts_i32.long()]
        dy = self.DY[acts_i32.long()]
        nx = bot_x.to(torch.int32) + dx
        ny = bot_y.to(torch.int32) + dy

        in_bounds = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
        ny_safe = ny.clamp(0, self.H - 1)
        nx_safe = nx.clamp(0, self.W - 1)
        is_walkable = self.walkable[ny_safe.long(), nx_safe.long()]
        can_move = is_move & in_bounds & is_walkable

        bot_x = torch.where(can_move, nx.to(torch.int16), bot_x)
        bot_y = torch.where(can_move, ny.to(torch.int16), bot_y)

        # === PICKUP ===
        is_pickup = (actions == ACT_PICKUP)
        item_idx = action_items.long().clamp(0, self.num_items - 1)
        inv_count = (bot_inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP

        ix = self.item_pos_x[item_idx].to(torch.int32)
        iy = self.item_pos_y[item_idx].to(torch.int32)
        mdist = (bot_x.to(torch.int32) - ix).abs() + (bot_y.to(torch.int32) - iy).abs()
        adjacent = mdist == 1

        can_pickup = is_pickup & has_space & adjacent
        pickup_type = self.item_types[item_idx]

        added = torch.zeros(B, dtype=torch.bool, device=d)
        for s in range(INV_CAP):
            slot_empty = bot_inv[:, s] < 0
            add_here = can_pickup & slot_empty & ~added
            bot_inv[:, s] = torch.where(add_here, pickup_type, bot_inv[:, s])
            added = added | add_here

        # === DROPOFF (vectorized) ===
        is_dropoff = (actions == ACT_DROPOFF)
        at_drop = (bot_x == self.drop_x) & (bot_y == self.drop_y)
        has_items = (bot_inv >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items

        aidx = active_idx.long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]

        bot_inv, active_del, score_add = self._vectorized_deliver(
            bot_inv, act_req, active_del, can_dropoff, B, d)
        score = score + score_add

        sort_key = (bot_inv < 0).to(torch.int8)
        _, sort_idx = sort_key.sort(dim=1, stable=True)
        bot_inv = bot_inv.gather(1, sort_idx.long())

        # === ORDER COMPLETION ===
        slot_done = (active_del == 1) | (act_req < 0)
        has_required = (act_req >= 0).any(dim=1)
        order_complete = slot_done.all(dim=1) & can_dropoff & has_required

        score = score + order_complete.to(torch.int32) * 5
        orders_comp = orders_comp + order_complete.to(torch.int32)

        new_aidx = (active_idx + order_complete.to(torch.int32)).clamp(
            0, self.num_orders - 1)
        oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
        active_del = torch.where(oc_mask, self._zero_i8, active_del)

        # Auto-deliver candidate at dropoff (vectorized)
        new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)
        new_req = self.order_req[new_aidx_l]

        bot_inv, active_del, score_add = self._vectorized_deliver(
            bot_inv, new_req, active_del, order_complete, B, d)
        score = score + score_add

        sort_key = (bot_inv < 0).to(torch.int8)
        _, sort_idx = sort_key.sort(dim=1, stable=True)
        bot_inv = bot_inv.gather(1, sort_idx.long())

        active_idx = new_aidx
        return bot_x, bot_y, bot_inv, active_idx, active_del, score, orders_comp

    @torch.no_grad()
    def _hash(self, state):
        """Compute state hash for deduplication. Returns [B] int64.

        Two states with the same hash are functionally identical
        (same position, inventory, order state). Score is NOT included
        in the hash — we keep the highest-scoring representative.
        """
        # Sort inventory for canonical form
        sorted_inv, _ = state['bot_inv'].sort(dim=1)

        h = (state['bot_x'].long()
             | (state['bot_y'].long() << 5)
             | ((sorted_inv[:, 0].long() + 1) << 10)
             | ((sorted_inv[:, 1].long() + 1) << 15)
             | ((sorted_inv[:, 2].long() + 1) << 20)
             | (state['active_idx'].long() << 25))

        # Pack active_del as bits (vectorized)
        del_bits = state['active_del'].long()  # [B, MAX_ORDER_SIZE]
        shifts = torch.arange(32, 32 + MAX_ORDER_SIZE, device=self.device)
        h = h | (del_bits << shifts.unsqueeze(0)).sum(dim=1)

        return h

    @torch.no_grad()
    def _eval(self, state, round_num=0):
        """Evaluate states for beam pruning. Returns [B] float32 (higher=better).

        Key insight: each pending delivery is worth ~1 point (100000 in eval units).
        Items in inventory matching the active order should be valued at nearly
        a full point, discounted by distance to dropoff. This prevents the beam
        from pruning states that are about to score.

        Vectorized: replaces 3x6 nested loops with batched tensor ops.
        """
        B = state['bot_x'].shape[0]
        d = self.device

        bot_x = state['bot_x'].long()
        bot_y = state['bot_y'].long()
        inv = state['bot_inv']   # [B, INV_CAP] int8 (type index, -1=empty)
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)

        rounds_left = MAX_ROUNDS - round_num - 1
        rl_f = float(max(rounds_left, 1))

        # Score is dominant factor (1 point = 100000 eval units)
        ev = state['score'].float() * 100000

        # Distance to dropoff (BFS, exact)
        dist_drop = self.dist_to_dropoff[bot_y, bot_x].float()

        # Active order analysis
        act_req = self.order_req[aidx]  # [B, MAX_ORDER_SIZE] int8
        act_del = state['active_del']   # [B, MAX_ORDER_SIZE] int8

        active_needed = (act_req >= 0) & (act_del == 0)  # [B, 6]
        active_done = (act_req >= 0) & (act_del == 1)     # [B, 6]
        active_remaining = active_needed.sum(dim=1).float()
        active_delivered = active_done.sum(dim=1).float()
        active_total = ((act_req >= 0)).sum(dim=1).float()

        # === Order completion progress ===
        # Completing an order gives +5 bonus. Value progress toward completion.
        # fraction_done ranges from 0.0 to 1.0
        fraction_done = active_delivered / active_total.clamp(min=1)
        # Completion is imminent when remaining == 0 (order bonus about to trigger)
        ev = ev + fraction_done * 30000     # Partial progress toward +5 bonus
        ev = ev + (active_remaining == 0).float() * 50000  # About to complete!
        ev = ev + (active_remaining == 1).float() * 20000  # Almost there
        ev = ev + (active_remaining == 2).float() * 5000

        # === Vectorized inventory-order matching ===
        # inv: [B, 3], act_req: [B, 6] -> match grid [B, 3, 6]
        inv_exp = inv.unsqueeze(2)     # [B, 3, 1]
        req_exp = act_req.unsqueeze(1) # [B, 1, 6]
        del_exp = act_del.unsqueeze(1) # [B, 1, 6]

        has_item_exp = (inv_exp >= 0)  # [B, 3, 1]
        # An inv slot matches an order slot if same type, slot not yet delivered, slot valid
        match_active = (inv_exp == req_exp) & (del_exp == 0) & (req_exp >= 0) & has_item_exp  # [B, 3, 6]
        # Any match per inv slot (across order slots)
        inv_matches_active = match_active.any(dim=2)  # [B, 3]
        has_active_inv = inv_matches_active.any(dim=1)  # [B]
        num_active_items = inv_matches_active.sum(dim=1).float()  # [B]

        # Active inventory value: ~70000 per item, discounted by distance to dropoff
        # At dist_drop=0: 70000 (0.7 of a delivery — high but not overriding score)
        # At dist_drop=20: 30000
        # At dist_drop=40: 0 (item far from deliverable)
        active_inv_value = (70000 - dist_drop * 1750).clamp(min=0)
        ev = ev + num_active_items * active_inv_value

        # Preview order matching (pipeline_depth determines which order ahead to target)
        _pdepth = self.pipeline_depth if self.pipeline_mode else 1
        pidx = (aidx + _pdepth).clamp(0, self.num_orders - 1)
        prev_req = self.order_req[pidx]  # [B, 6]
        prev_req_exp = prev_req.unsqueeze(1)  # [B, 1, 6]
        match_preview = (inv_exp == prev_req_exp) & (prev_req_exp >= 0) & has_item_exp  # [B, 3, 6]
        inv_matches_preview = match_preview.any(dim=2) & ~inv_matches_active  # [B, 3]

        # Preview items: value depends on planning mode.
        #
        # pipeline_mode=True (pre-fetch bot): always value preview items highly —
        #   this bot's job IS to pre-pick the next order. Never penalize.
        #
        # has locked bots (multi-bot sequential DP): use progressive valuation —
        #   locked bots are delivering the active order, so preview items become
        #   more valuable as the active order progresses. No early penalty.
        #
        # single-bot (no locked bots): conservative — penalize early hoarding
        #   because the sole bot must deliver active items first.
        if self.pipeline_mode:
            # Pipeline bot: always strongly value preview items
            preview_val_per_item = 15000.0
            preview_value = inv_matches_preview.float() * preview_val_per_item
        elif self.num_locked > 0:
            # Multi-bot with locked delivery bots: progressive value, no penalty.
            # With many locked bots delivering, active order completes fast;
            # start valuing preview items higher from the beginning.
            # At 0% done: 5000 (was 2000 — encourage pre-fetching earlier)
            # At 50% done: 10000 (was 8500)
            # At 100% done: 15000
            frac_exp = fraction_done.unsqueeze(1).expand_as(inv_matches_preview)
            preview_val_per_item = 5000.0 + 10000.0 * frac_exp
            preview_value = inv_matches_preview.float() * preview_val_per_item
        else:
            # Single-bot: original behaviour — penalise early hoarding
            near_complete = (active_remaining <= 2)  # [B]
            preview_value = torch.where(
                near_complete.unsqueeze(1).expand_as(inv_matches_preview),
                inv_matches_preview.float() * 15000,
                inv_matches_preview.float() * -5000)
        ev = ev + preview_value.sum(dim=1)

        # Dead inventory: items matching neither active nor preview
        has_item_flat = (inv >= 0)  # [B, 3]
        is_dead = has_item_flat & ~inv_matches_active & ~inv_matches_preview  # [B, 3]
        # Multi-bot mode: also check orders +2, +3 before declaring dead.
        # With multiple bots, orders complete faster so pre-fetching further ahead is useful.
        if self.num_locked > 0 and is_dead.any():
            for _extra_d in [2, 3]:
                _eidx = (aidx + _extra_d).clamp(0, self.num_orders - 1)
                _ereq = self.order_req[_eidx]   # [B, MAX_ORDER_SIZE]
                _ereq_exp = _ereq.unsqueeze(1)  # [B, 1, MAX_ORDER_SIZE]
                _match_extra = (inv_exp == _ereq_exp) & (_ereq_exp >= 0) & has_item_exp
                _matches_extra = _match_extra.any(dim=2) & ~inv_matches_active & ~inv_matches_preview
                is_dead = is_dead & ~_matches_extra
        # Dead items are more catastrophic early (waste whole game) vs late (just 1-2 wasted rounds)
        dead_penalty = 50000 * min(1.0, rounds_left / 150.0) + 5000
        ev = ev - is_dead.sum(dim=1).float() * dead_penalty

        # === Distance to needed items (with inventory space) ===
        inv_count = has_item_flat.sum(dim=1)
        has_space = inv_count < INV_CAP

        best_item_dist = torch.full((B,), 9999.0, device=d)
        for os in range(MAX_ORDER_SIZE):
            needed = active_needed[:, os]
            if not needed.any():
                continue
            needed_type = act_req[:, os].long().clamp(0, self.num_types - 1)
            d_type = self.dist_to_type[needed_type, bot_y, bot_x].float()
            better = needed & has_space & (d_type < best_item_dist)
            best_item_dist = torch.where(better, d_type, best_item_dist)

        close = best_item_dist < 9999

        # Trip feasibility check
        min_trip_time = best_item_dist + dist_drop + 2  # pickup + dropoff actions
        trip_feasible = close & (min_trip_time <= rounds_left)

        # Approaching needed items — only for single-bot mode.
        # Multi-bot mode uses differentiated guidance in the coordination block below.
        if self.num_locked == 0:
            ev = ev + torch.where(
                trip_feasible & has_space & ~has_active_inv,
                20000.0 - best_item_dist * 800,
                torch.zeros(B, device=d))
            ev = ev + torch.where(
                trip_feasible & has_space & has_active_inv,
                5000.0 - best_item_dist * 200,
                torch.zeros(B, device=d))

        # Bot with active items: closer to dropoff is better
        delivery_urgency = max(1.0, (100 - rounds_left) / 20.0) if rounds_left < 100 else 1.0
        ev = ev - torch.where(
            has_active_inv, dist_drop * 200 * delivery_urgency, torch.zeros(B, device=d))

        # Pipeline/multi-bot: bot holding preview items should position near dropoff
        # as active order nears completion (ready for instant auto-delivery).
        if self.pipeline_mode or self.num_locked > 0:
            has_preview_inv_any = inv_matches_preview.any(dim=1)  # [B]
            # Scale guidance by order completion fraction: stronger as order nears done
            preview_drop_guidance = 80.0 * fraction_done + 20.0  # 20..100 per unit dist
            ev = ev - torch.where(
                has_preview_inv_any & ~has_active_inv,
                dist_drop * preview_drop_guidance,
                torch.zeros(B, device=d))

        # Can't reach dropoff in time: inventory is completely wasted
        cant_deliver = has_active_inv & (dist_drop >= rounds_left)
        ev = ev - cant_deliver.float() * 80000

        # Empty bot camping at dropoff: penalize blocking deliveries
        at_drop = (state['bot_x'].long() == self.drop_x) & (state['bot_y'].long() == self.drop_y)
        camping_penalty = at_drop & ~has_active_inv
        ev = ev - camping_penalty.float() * 12000

        # === Coordination with locked bots ===
        if 'locked_bx' in state and self.num_locked > 0:
            locked_bx_state = state['locked_bx']  # [B, num_locked]
            locked_by_state = state['locked_by']
            # Congestion: penalize being near dropoff when many locked bots are too
            locked_at_drop = ((locked_bx_state == self.drop_x) &
                              (locked_by_state == self.drop_y)).sum(dim=1).float()
            near_drop = dist_drop < 3
            ev = ev - near_drop.float() * locked_at_drop * 500

            locked_inv_state = state.get('locked_inv')  # [B, num_locked, INV_CAP]
            if locked_inv_state is not None:
                # === Assignment-aware coordination ===
                # For each active order slot, determine if locked bots already cover it
                # (either via current inventory OR planned future pickup).
                # Candidate should target UNCOVERED slots (division of labor).
                locked_inv_l = locked_inv_state.long()  # [B, L, INV_CAP]

                # Use precomputed lookup tables (built in __init__) instead of
                # creating new CUDA tensors / scanning numpy on every _eval call.
                locked_plans_type_mask = self._locked_all_planned_mask  # [num_types] bool | None

                # _locked_remaining_planned[r, t]: True if any locked bot will pick type t
                # in rounds [r, MAX_ROUNDS). Accurate full-game coverage tracking.
                remaining_cover = None
                if self._locked_remaining_planned is not None and round_num >= 0:
                    rn = min(round_num, MAX_ROUNDS)
                    remaining_cover = self._locked_remaining_planned[rn]  # [num_types] bool

                # locked_covers_slot[b, os]: True if any locked bot carries or will pick
                # the type needed by order slot os (full remaining-game coverage).
                locked_covers_slot = torch.zeros(
                    B, MAX_ORDER_SIZE, dtype=torch.bool, device=d)
                for os in range(MAX_ORDER_SIZE):
                    ntype = act_req[:, os].long().clamp(0, self.num_types - 1)  # [B]
                    ntype_e = ntype.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                    # Covers by current inventory
                    has_type = ((locked_inv_l == ntype_e) &
                                (locked_inv_state >= 0)).any(dim=2).any(dim=1)  # [B]
                    # Also covers if any locked bot PLANS to pick that type this game
                    if remaining_cover is not None:
                        has_type = has_type | remaining_cover[ntype]  # [B] bool
                    locked_covers_slot[:, os] = has_type

                # For each active order slot, check if candidate carries that type
                # needed: active_needed [B, MAX_ORDER_SIZE]
                covered_by_locked = active_needed & locked_covers_slot      # [B, 6]
                uncovered_by_locked = active_needed & ~locked_covers_slot   # [B, 6]

                # Candidate inventory items: [B, INV_CAP]
                cand_redundant = torch.zeros(B, dtype=torch.float32, device=d)
                cand_covers_unique = torch.zeros(B, dtype=torch.float32, device=d)
                for s in range(INV_CAP):
                    for os in range(MAX_ORDER_SIZE):
                        type_match = ((inv[:, s].long() == act_req[:, os].long()) &
                                      (inv[:, s] >= 0))
                        # Carrying a type already covered by locked bots → redundant
                        cand_redundant = cand_redundant + (
                            type_match & covered_by_locked[:, os]).float()
                        # Carrying a type NOT covered by any locked bot → uniquely useful
                        cand_covers_unique = cand_covers_unique + (
                            type_match & uncovered_by_locked[:, os]).float()

                # Strong incentive: reward unique coverage, penalize redundancy
                ev = ev + cand_covers_unique * 30000
                ev = ev - cand_redundant * 20000

                # === Differentiated distance guidance (replaces general guidance) ===
                # Priority 1: Uncovered active types → strong signal (same as single-bot).
                # Priority 2: All active slots covered → strong signal toward PREVIEW items.
                # Priority 3: No preview either → weak signal toward covered active types.
                best_uncovered_dist = torch.full((B,), 9999.0, device=d)
                best_covered_dist = torch.full((B,), 9999.0, device=d)
                for os in range(MAX_ORDER_SIZE):
                    needed_type = act_req[:, os].long().clamp(0, self.num_types - 1)
                    d_os = self.dist_to_type[needed_type, bot_y, bot_x].float()
                    is_uncov = uncovered_by_locked[:, os]
                    is_cov = covered_by_locked[:, os]
                    better_u = is_uncov & (d_os < best_uncovered_dist)
                    best_uncovered_dist = torch.where(better_u, d_os, best_uncovered_dist)
                    better_c = is_cov & (d_os < best_covered_dist)
                    best_covered_dist = torch.where(better_c, d_os, best_covered_dist)

                close_u = best_uncovered_dist < 9999
                close_c = best_covered_dist < 9999
                trip_u = close_u & ((best_uncovered_dist + dist_drop + 2) <= rounds_left)
                trip_c = close_c & ((best_covered_dist + dist_drop + 2) <= rounds_left)

                # --- Priority 1: Uncovered active types ---
                ev = ev + torch.where(
                    trip_u & has_space & ~has_active_inv,
                    20000.0 - best_uncovered_dist * 800,
                    torch.zeros(B, device=d))
                ev = ev + torch.where(
                    trip_u & has_space & has_active_inv,
                    5000.0 - best_uncovered_dist * 200,
                    torch.zeros(B, device=d))

                # --- Priority 2: All active slots covered → pipeline toward preview ---
                # Bot becomes a pipeline bot: pre-fetch next order's items.
                # If locked bots already plan to pick those preview types → go to order+2.
                all_active_covered = ~close_u  # no reachable uncovered active type
                if all_active_covered.any():
                    best_prev_dist = torch.full((B,), 9999.0, device=d)
                    best_pp_dist = torch.full((B,), 9999.0, device=d)  # order+2

                    # Order+1 (preview) distance — skip types that locked bots WILL STILL pick.
                    # Use remaining_cover (suffix union, future-only) if available;
                    # fall back to locked_plans_type_mask (whole-game) otherwise.
                    # This prevents wrongly excluding types locked bots already picked in past rounds.
                    _preview_cover = remaining_cover if remaining_cover is not None else locked_plans_type_mask
                    pidx_p1 = (aidx + 1).clamp(0, self.num_orders - 1)
                    req_p1 = self.order_req[pidx_p1]
                    for os in range(MAX_ORDER_SIZE):
                        pt = req_p1[:, os].long().clamp(0, self.num_types - 1)
                        d_pt = self.dist_to_type[pt, bot_y, bot_x].float()
                        has_slot = req_p1[:, os] >= 0
                        # Exclude types that locked bots will still pick → those bots handle it
                        if _preview_cover is not None:
                            already_planned = _preview_cover[pt]
                            has_slot = has_slot & ~already_planned
                        better_p = has_slot & (d_pt < best_prev_dist)
                        best_prev_dist = torch.where(better_p, d_pt, best_prev_dist)

                    # Order+2 distance (deep pipeline when order+1 is fully covered)
                    pidx_p2 = (aidx + 2).clamp(0, self.num_orders - 1)
                    req_p2 = self.order_req[pidx_p2]
                    for os in range(MAX_ORDER_SIZE):
                        pt2 = req_p2[:, os].long().clamp(0, self.num_types - 1)
                        d_pt2 = self.dist_to_type[pt2, bot_y, bot_x].float()
                        has_slot2 = req_p2[:, os] >= 0
                        better_pp = has_slot2 & (d_pt2 < best_pp_dist)
                        best_pp_dist = torch.where(better_pp, d_pt2, best_pp_dist)

                    close_p = best_prev_dist < 9999
                    trip_p = close_p & ((best_prev_dist + dist_drop + 2) <= rounds_left)
                    close_pp = best_pp_dist < 9999
                    trip_pp = close_pp & ((best_pp_dist + dist_drop + 2) <= rounds_left)

                    # Pipeline toward order+1 uncovered types
                    ev = ev + torch.where(
                        all_active_covered & trip_p & has_space & ~has_active_inv,
                        16000.0 - best_prev_dist * 800,
                        torch.zeros(B, device=d))

                    # Deep pipeline toward order+2 (when order+1 fully covered by locked bots)
                    ev = ev + torch.where(
                        all_active_covered & ~close_p & trip_pp & has_space & ~has_active_inv,
                        14000.0 - best_pp_dist * 800,
                        torch.zeros(B, device=d))

                    # --- Priority 3: No preview options → weak covered fallback ---
                    no_options = all_active_covered & ~close_p & ~close_pp
                    ev = ev + torch.where(
                        no_options & trip_c & has_space & ~has_active_inv,
                        10000.0 - best_covered_dist * 500,
                        torch.zeros(B, device=d))

        # === Type specialization bonus (soft assignment) ===
        # Encourages this bot to pick items of its preferred types, reducing
        # competition with other bots and improving natural work division.
        if self.preferred_types is not None and len(self.preferred_types) > 0:
            # Build preferred_mask: [num_types] bool
            pref_mask = torch.zeros(self.num_types + 1, dtype=torch.bool, device=d)
            for t in self.preferred_types:
                if 0 <= t < self.num_types:
                    pref_mask[t] = True

            # Bonus for carrying preferred type items in inventory
            inv_safe = inv.long().clamp(0, self.num_types)  # [B, INV_CAP]
            has_item_flat2 = (inv >= 0)  # [B, INV_CAP]
            for s in range(INV_CAP):
                is_pref = pref_mask[inv_safe[:, s]]  # [B]
                ev = ev + (is_pref & has_item_flat2[:, s]).float() * 8000

            # Guidance bonus: moving toward nearest preferred type when empty/space
            best_pref_dist = torch.full((B,), 9999.0, device=d)
            for t in self.preferred_types:
                if 0 <= t < self.num_types:
                    t_t = torch.tensor(t, dtype=torch.long, device=d)
                    d_t = self.dist_to_type[t_t, bot_y, bot_x].float()  # [B]
                    better = d_t < best_pref_dist
                    best_pref_dist = torch.where(better, d_t, best_pref_dist)
            close_pref = best_pref_dist < 9999
            trip_pref = close_pref & ((best_pref_dist + dist_drop + 2) <= rounds_left)
            # Only guide toward preferred types when empty (not already carrying active items)
            ev = ev + torch.where(
                trip_pref & has_space & ~has_active_inv,
                6000.0 - best_pref_dist * 300,
                torch.zeros(B, device=d))

        return ev

    @torch.no_grad()
    def search(self, game_state, beam_width=10000, verbose=True):
        """Run GPU beam search over 300 rounds.

        Returns (best_score, action_sequence) where action_sequence is
        list of [(action_type, item_idx)] per round.
        """
        t0 = time.time()
        state = self._from_game_state(game_state)
        N = self.num_actions

        # Parent pointer tracking for action reconstruction
        parent_history = []
        act_offset_history = []

        for rnd in range(MAX_ROUNDS):
            t_rnd = time.time()
            B = state['bot_x'].shape[0]

            # Smart expand: guided candidates + exploration fallbacks
            # Multi-bot needs more slots: active(6)+preview(6)+order+2(6)+pickups+dropoff+moves
            _expand_cands = 30 if self.num_locked > 0 else 20
            expanded, actions, action_items, valid_mask, C = \
                self._smart_expand(state, max_cands=_expand_cands)

            # Step: apply all actions in parallel
            new_state = self._step(expanded, actions, action_items, round_num=rnd)

            # Detect no-ops (invalid actions producing same state as wait)
            old_x = expanded['bot_x']
            old_y = expanded['bot_y']
            old_inv = expanded['bot_inv']
            old_score = expanded['score']
            no_change = (
                (new_state['bot_x'] == old_x) &
                (new_state['bot_y'] == old_y) &
                ((new_state['bot_inv'] == old_inv).all(dim=1)) &
                (new_state['score'] == old_score)
            )
            is_noop_dup = no_change & (actions != ACT_WAIT)

            # Eval
            evals = self._eval(new_state, round_num=rnd)
            evals[~valid_mask] = float('-inf')
            evals[is_noop_dup] = float('-inf')

            # Top-K
            new_B = new_state['bot_x'].shape[0]
            valid_count = int((valid_mask & ~is_noop_dup).sum().item())
            k = min(beam_width, max(valid_count, 1))
            _, topk_idx = torch.topk(evals, k)

            N = C  # candidates per parent state

            # Count unique states for diagnostics
            num_unique = 0
            if verbose and (rnd < 5 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
                hashes = self._hash({key: val[topk_idx] for key, val in new_state.items()})
                num_unique = int(torch.unique(hashes).shape[0])

            # Gather selected states
            state = {}
            for key, val in new_state.items():
                state[key] = val[topk_idx]

            # Track lineage for action reconstruction
            parent = topk_idx // N  # which parent state
            # Save actual actions taken (not offsets into pattern)
            taken_acts = actions[topk_idx].cpu()
            taken_items = action_items[topk_idx].cpu()
            parent_history.append(parent.cpu())
            act_offset_history.append((taken_acts, taken_items))

            dt = time.time() - t_rnd
            if verbose and (rnd < 5 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
                best_score = state['score'].max().item()
                uniq_str = f", unique={num_unique}" if num_unique > 0 else ""
                print(f"  R{rnd:3d}: score={best_score:3d}, beam={k}, "
                      f"valid={valid_count}/{new_B}{uniq_str}, dt={dt:.3f}s",
                      file=sys.stderr)

        # Find best final state
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Reconstruct action sequence by backtracking through parent pointers
        actions_seq = []
        idx = best_idx

        wait_pad = [(ACT_WAIT, -1)] * (self.num_bots - 1)
        for rnd in range(MAX_ROUNDS - 1, -1, -1):
            taken_acts, taken_items = act_offset_history[rnd]
            act_type = int(taken_acts[idx])
            item = int(taken_items[idx])
            actions_seq.append([(act_type, item)] + wait_pad)
            idx = int(parent_history[rnd][idx])
        actions_seq.reverse()

        total_time = time.time() - t0
        if verbose:
            print(f"\nGPU beam search: score={best_score}, time={total_time:.1f}s, "
                  f"beam={beam_width}, actions/state={N}",
                  file=sys.stderr)

        return best_score, actions_seq

    def verify_against_cpu(self, game_state, all_orders, num_rounds=100):
        """Verify GPU step matches CPU step for guided action sequences.

        Uses a simple greedy policy that actually picks up items and delivers them,
        ensuring that pickup, dropoff, order completion, and auto-delivery are tested.

        Returns True if all rounds match, raises AssertionError otherwise.
        """
        from pathfinding import precompute_all_distances, get_first_step, get_distance
        ms = game_state.map_state
        dist_maps = precompute_all_distances(ms)

        # Run CPU simulation with greedy policy
        cpu_state = game_state.copy()
        cpu_actions_log = []

        for rnd in range(num_rounds):
            bx = int(cpu_state.bot_positions[0, 0])
            by = int(cpu_state.bot_positions[0, 1])
            bot_pos = (bx, by)
            inv_count = cpu_state.bot_inv_count(0)
            inv_items = cpu_state.bot_inv_list(0)

            active = cpu_state.get_active_order()
            action = (ACT_WAIT, -1)

            if active:
                # Priority 1: dropoff if at dropoff with matching items
                if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
                        inv_count > 0 and any(active.needs_type(t) for t in inv_items)):
                    action = (ACT_DROPOFF, -1)

                # Priority 2: pickup adjacent needed item
                elif inv_count < INV_CAP:
                    for item_idx in range(ms.num_items):
                        ix = int(ms.item_positions[item_idx, 0])
                        iy = int(ms.item_positions[item_idx, 1])
                        if abs(bx - ix) + abs(by - iy) == 1:
                            tid = int(ms.item_types[item_idx])
                            if active.needs_type(tid):
                                action = (ACT_PICKUP, item_idx)
                                break

                # Priority 3: move toward dropoff if carrying active items
                elif inv_count > 0 and any(active.needs_type(t) for t in inv_items):
                    act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                    if act > 0:
                        action = (act, -1)

                # Priority 4: move toward nearest needed item
                if action[0] == ACT_WAIT and inv_count < INV_CAP:
                    best_dist = 9999
                    best_act = None
                    for item_idx in range(ms.num_items):
                        tid = int(ms.item_types[item_idx])
                        if not active.needs_type(tid):
                            continue
                        adj = ms.item_adjacencies.get(item_idx, [])
                        for ax, ay in adj:
                            d = get_distance(dist_maps, bot_pos, (ax, ay))
                            if d < best_dist:
                                best_dist = d
                                act = get_first_step(dist_maps, bot_pos, (ax, ay))
                                if act > 0:
                                    best_act = (act, -1)
                                    best_dist = d
                    if best_act:
                        action = best_act

                # Priority 5: move to dropoff with items
                if action[0] == ACT_WAIT and inv_count > 0:
                    act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                    if act > 0:
                        action = (act, -1)

            cpu_actions_log.append(action)
            cpu_state.round = rnd
            # Pad with wait actions for other bots in multi-bot games
            num_bots = len(cpu_state.bot_positions)
            round_actions = [action] + [(ACT_WAIT, -1)] * (num_bots - 1)
            cpu_step(cpu_state, round_actions, all_orders)

        # Run GPU simulation with same actions
        gpu_state = self._from_game_state(game_state)

        for rnd in range(num_rounds):
            act_type, item_idx = cpu_actions_log[rnd]
            actions = torch.tensor([act_type], dtype=torch.int8, device=self.device)
            action_items = torch.tensor(
                [item_idx if item_idx >= 0 else 0], dtype=torch.int16, device=self.device)
            gpu_state = self._step(gpu_state, actions, action_items, round_num=rnd)

            # Compare
            gpu_score = gpu_state['score'][0].item()
            cpu_score = cpu_state.score if rnd == num_rounds - 1 else None

            gpu_x = gpu_state['bot_x'][0].item()
            gpu_y = gpu_state['bot_y'][0].item()
            cpu_x = int(cpu_state.bot_positions[0, 0]) if rnd == num_rounds - 1 else None

        # Final comparison
        gpu_score = gpu_state['score'][0].item()
        gpu_x = gpu_state['bot_x'][0].item()
        gpu_y = gpu_state['bot_y'][0].item()
        gpu_oc = gpu_state['orders_comp'][0].item()

        cpu_score = cpu_state.score
        cpu_x = int(cpu_state.bot_positions[0, 0])
        cpu_y = int(cpu_state.bot_positions[0, 1])
        cpu_oc = cpu_state.orders_completed

        print(f"  Verification ({num_rounds} rounds):", file=sys.stderr)
        print(f"    CPU: score={cpu_score}, pos=({cpu_x},{cpu_y}), "
              f"orders={cpu_oc}", file=sys.stderr)
        print(f"    GPU: score={gpu_score}, pos=({gpu_x},{gpu_y}), "
              f"orders={gpu_oc}", file=sys.stderr)

        # Check score match
        if gpu_score != cpu_score:
            # Detailed round-by-round comparison to find divergence
            gpu_state2 = self._from_game_state(game_state)
            cpu_state2 = game_state.copy()

            for rnd in range(num_rounds):
                act_type, item_idx = cpu_actions_log[rnd]

                # CPU step
                cpu_state2.round = rnd
                num_bots_v = len(cpu_state2.bot_positions)
                round_acts_v = [(act_type, item_idx)] + [(ACT_WAIT, -1)] * (num_bots_v - 1)
                cpu_step(cpu_state2, round_acts_v, all_orders)

                # GPU step
                a = torch.tensor([act_type], dtype=torch.int8, device=self.device)
                ai = torch.tensor(
                    [item_idx if item_idx >= 0 else 0],
                    dtype=torch.int16, device=self.device)
                gpu_state2 = self._step(gpu_state2, a, ai, round_num=rnd)

                gs = gpu_state2['score'][0].item()
                cs = cpu_state2.score
                gx = gpu_state2['bot_x'][0].item()
                gy = gpu_state2['bot_y'][0].item()
                cx = int(cpu_state2.bot_positions[0, 0])
                cy = int(cpu_state2.bot_positions[0, 1])

                if gs != cs or gx != cx or gy != cy:
                    gi = [int(gpu_state2['bot_inv'][0, s].item()) for s in range(INV_CAP)]
                    ci = [int(cpu_state2.bot_inventories[0, s]) for s in range(INV_CAP)]
                    print(f"    DIVERGENCE at round {rnd}!", file=sys.stderr)
                    print(f"      Action: type={act_type}, item={item_idx}",
                          file=sys.stderr)
                    print(f"      CPU: score={cs}, pos=({cx},{cy}), inv={ci}",
                          file=sys.stderr)
                    print(f"      GPU: score={gs}, pos=({gx},{gy}), inv={gi}",
                          file=sys.stderr)
                    return False

        assert gpu_score == cpu_score, \
            f"Score mismatch: CPU={cpu_score}, GPU={gpu_score}"
        assert gpu_x == cpu_x and gpu_y == cpu_y, \
            f"Position mismatch: CPU=({cpu_x},{cpu_y}), GPU=({gpu_x},{gpu_y})"
        assert gpu_oc == cpu_oc, \
            f"Orders mismatch: CPU={cpu_oc}, GPU={gpu_oc}"

        print(f"    MATCH! Score={gpu_score}, Orders={gpu_oc}", file=sys.stderr)
        return True

    @torch.no_grad()
    def dp_search(self, game_state, max_states=500000, verbose=True, on_round=None,
                  bot_id=0, start_rnd=0, max_rounds=None, init_state=None):
        """Exact dynamic programming search via exhaustive BFS + dedup on GPU.

        For single-bot games, finds the OPTIMAL solution by exploring ALL
        reachable states each round. State space is bounded:
        - ~45 walkable cells (Easy) * ~35 inventory combos * ~64 order states
        - Typically <20K unique states per round

        Falls back to eval-based pruning only if states exceed max_states.

        Args:
            on_round: Optional callback(rnd, best_score, unique, expanded, elapsed)
                      for streaming progress updates.
            bot_id: Which bot in the game state to optimize (default 0).
            start_rnd: Starting round index (default 0). Used for locked bot lookup.
            max_rounds: Max rounds to run (default MAX_ROUNDS - start_rnd).
            init_state: Optional GPU state dict to start from instead of game_state.
                        If provided, game_state may be None.

        Returns (best_score, action_sequence).
        """
        t0 = time.time()
        self.candidate_bot_id = bot_id
        if init_state is not None:
            state = init_state
        else:
            state = self._from_game_state(game_state, bot_id=bot_id)
        d = self.device

        if max_rounds is None:
            max_rounds = MAX_ROUNDS - start_rnd

        # Use position-filtered expand (6+MAX_ADJ actions vs 6+num_items)
        use_dp_expand = True
        N = self.dp_num_actions if use_dp_expand else self.num_actions

        # History for action reconstruction (kept on CPU to save GPU mem)
        parent_idx_history = []  # parent_idx_history[r][i] = index into round r-1's set
        act_history = []         # act_history[r][i] = action type taken
        item_history = []        # item_history[r][i] = action item index

        pruned_rounds = 0

        for i in range(max_rounds):
            rnd = start_rnd + i
            B = state['bot_x'].shape[0]

            # Expand: position-aware (10 actions) vs full (136 actions)
            if use_dp_expand:
                expanded, actions, action_items, dp_valid, N = self._dp_expand(state)
            else:
                expanded, actions, action_items = self._expand(state)
                dp_valid = None
            BN = B * N

            # Step all expanded states
            new_state = self._step(expanded, actions, action_items, round_num=rnd)

            # Detect no-ops (actions producing same state as wait)
            no_change = (
                (new_state['bot_x'] == expanded['bot_x']) &
                (new_state['bot_y'] == expanded['bot_y']) &
                ((new_state['bot_inv'] == expanded['bot_inv']).all(dim=1)) &
                (new_state['score'] == expanded['score']) &
                ((new_state['active_del'] == expanded['active_del']).all(dim=1)) &
                (new_state['active_idx'] == expanded['active_idx'])
            )
            is_noop = no_change & (actions != ACT_WAIT)

            # Also mark invalid dp_expand slots as noop
            if dp_valid is not None:
                is_noop = is_noop | ~dp_valid

            # Hash states for dedup
            hashes = self._hash(new_state)
            hashes[is_noop] = -1  # mark invalid

            # Dedup: sort by hash, keep highest-scoring representative per hash
            sorted_h, sort_idx = hashes.sort()
            is_first = torch.ones(BN, dtype=torch.bool, device=d)
            is_first[1:] = sorted_h[1:] != sorted_h[:-1]
            valid = is_first & (sorted_h != -1)

            # Indices of unique valid states in the expanded array
            unique_idx = sort_idx[valid]
            B_new = unique_idx.shape[0]

            # Parent tracking: which state in previous round produced this one
            parent_idx = unique_idx // N
            parent_idx_history.append(parent_idx.cpu())
            act_history.append(actions[unique_idx].cpu())
            item_history.append(action_items[unique_idx].cpu())

            # Gather unique states
            state = {k: v[unique_idx] for k, v in new_state.items()}

            # Adaptive beam: allow 2x headroom when state diversity is high
            diversity = B_new / BN if BN > 0 else 0
            effective_max = int(max_states * 2) if diversity > 0.5 else max_states

            # If too many states, prune by eval (lossy fallback)
            if B_new > effective_max:
                pruned_rounds += 1
                evals = self._eval(state, round_num=rnd)
                keep = min(effective_max, B_new)

                # Step 7: Threshold pruning before topk
                # Filter out low-eval states first, then topk on survivors
                if B_new > keep * 2:  # only bother if filtering saves >50%
                    max_eval = evals.max()
                    # Keep states within reasonable range of best
                    threshold = max_eval - 50000  # ~5 score points
                    survivor_mask = evals >= threshold
                    survivor_count = survivor_mask.sum().item()
                    if survivor_count > keep and survivor_count < B_new * 0.9:
                        # Filter first, then topk on smaller set
                        survivor_idx = survivor_mask.nonzero(as_tuple=True)[0]
                        survivor_evals = evals[survivor_idx]
                        _, local_topk = torch.topk(survivor_evals, keep)
                        topk = survivor_idx[local_topk]
                    else:
                        _, topk = torch.topk(evals, keep)
                else:
                    _, topk = torch.topk(evals, keep)

                topk_sorted, _ = topk.sort()  # maintain order for consistency
                state = {k: v[topk_sorted] for k, v in state.items()}
                parent_idx_history[-1] = parent_idx_history[-1][topk_sorted.cpu()]
                act_history[-1] = act_history[-1][topk_sorted.cpu()]
                item_history[-1] = item_history[-1][topk_sorted.cpu()]
                B_new = keep

            dt = time.time() - t0
            if i < 10 or i % 5 == 0 or i == max_rounds - 1:
                best_score = state['score'].max().item()
                if verbose and (i < 10 or i % 25 == 0 or i == max_rounds - 1):
                    print(f"  R{rnd:3d}: score={best_score:3d}, "
                          f"unique={B_new}, expanded={BN}, "
                          f"t={dt:.1f}s", file=sys.stderr)
                if on_round:
                    on_round(rnd, best_score, B_new, BN, dt)

        # Find best final state
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Backtrack to reconstruct action sequence (single-bot actions only)
        actions_seq = []
        idx = best_idx
        for j in range(max_rounds - 1, -1, -1):
            act = int(act_history[j][idx])
            item = int(item_history[j][idx])
            actions_seq.append((act, item))
            idx = int(parent_idx_history[j][idx])
        actions_seq.reverse()

        total_time = time.time() - t0
        if verbose:
            print(f"\nGPU DP search: score={best_score}, time={total_time:.1f}s, "
                  f"pruned_rounds={pruned_rounds}", file=sys.stderr)

        return best_score, actions_seq


def gpu_dp_search(difficulty=None, seed=None, max_states=500000, verbose=True,
                  game_factory=None, capture_data=None):
    """Convenience function for GPU DP search (optimal for single-bot).

    Returns (best_score, action_sequence) where action_sequence is
    list of [(action_type, item_idx)] per round per bot (old format).
    """
    if game_factory:
        gs, all_orders = game_factory()
    elif capture_data:
        gs, all_orders = init_game_from_capture(capture_data)
    else:
        gs, all_orders = init_game(seed, difficulty)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available, using CPU (will be slow)", file=sys.stderr)

    num_bots = len(gs.bot_positions)
    searcher = GPUBeamSearcher(gs.map_state, all_orders, device=device, num_bots=num_bots)

    if verbose:
        print("Verifying GPU step correctness...", file=sys.stderr)
        gs_copy, _ = (game_factory() if game_factory
                      else (init_game_from_capture(capture_data)
                            if capture_data
                            else init_game(seed, difficulty)))
        searcher.verify_against_cpu(gs_copy, all_orders, num_rounds=100)

    score, bot_actions = searcher.dp_search(gs, max_states=max_states, verbose=verbose)
    # Wrap single-bot actions into per-round multi-bot format
    wait_pad = [(ACT_WAIT, -1)] * (num_bots - 1)
    actions_seq = [[(a, i)] + wait_pad for a, i in bot_actions]
    return score, actions_seq


def gpu_beam_search(difficulty=None, seed=None, beam_width=10000, verbose=True,
                    game_factory=None, capture_data=None):
    """Convenience function for GPU beam search.

    Returns (best_score, action_sequence).
    """
    if game_factory:
        gs, all_orders = game_factory()
    elif capture_data:
        gs, all_orders = init_game_from_capture(capture_data)
    else:
        gs, all_orders = init_game(seed, difficulty)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available, using CPU (will be slow)", file=sys.stderr)

    num_bots2 = len(gs.bot_positions)
    searcher = GPUBeamSearcher(gs.map_state, all_orders, device=device, num_bots=num_bots2)

    if verbose:
        print("Verifying GPU step correctness...", file=sys.stderr)
        gs_copy, _ = (game_factory() if game_factory
                      else (init_game_from_capture(capture_data)
                            if capture_data
                            else init_game(seed, difficulty)))
        searcher.verify_against_cpu(gs_copy, all_orders, num_rounds=100)

    return searcher.search(gs, beam_width=beam_width, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GPU-accelerated search for grocery bot')
    parser.add_argument('difficulty', nargs='?', default='easy',
                        choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--seed', type=int, default=7001)
    parser.add_argument('--beam', type=int, default=10000)
    parser.add_argument('--max-states', type=int, default=500000)
    parser.add_argument('--mode', choices=['dp', 'beam'], default='dp',
                        help='Search mode: dp (exact) or beam (heuristic)')
    parser.add_argument('--capture', action='store_true',
                        help='Use saved capture data instead of sim server seed')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify GPU vs CPU, no search')
    args = parser.parse_args()

    if args.capture:
        from solution_store import load_capture
        capture = load_capture(args.difficulty)
        if capture is None:
            print(f"No capture for {args.difficulty}", file=sys.stderr)
            sys.exit(1)
        gs, all_orders = init_game_from_capture(capture)
    else:
        gs, all_orders = init_game(args.seed, args.difficulty)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_bots = len(gs.bot_positions)
    searcher = GPUBeamSearcher(gs.map_state, all_orders, device=device, num_bots=num_bots)

    print(f"Verifying GPU step correctness...", file=sys.stderr)
    gs_copy = gs.copy() if not args.capture else init_game_from_capture(capture)[0]
    ok = searcher.verify_against_cpu(gs_copy, all_orders, num_rounds=200)
    if not ok:
        print("VERIFICATION FAILED - aborting", file=sys.stderr)
        sys.exit(1)

    if args.verify_only:
        sys.exit(0)

    gs2 = gs.copy() if not args.capture else init_game_from_capture(capture)[0]

    if args.mode == 'dp':
        print(f"\nRunning GPU DP search: {args.difficulty}, max_states={args.max_states}",
              file=sys.stderr)
        score, actions = searcher.dp_search(gs2, max_states=args.max_states)
    else:
        print(f"\nRunning GPU beam search: {args.difficulty}, beam={args.beam}",
              file=sys.stderr)
        score, actions = searcher.search(gs2, beam_width=args.beam)

    print(f"\nFinal score: {score}")

    # Save solution for replay
    from solution_store import save_solution
    # actions is [(act, item)] per round; save_solution expects [[(a,i),...]] per round per bot
    actions_to_save = [[(act, item)] for (act, item) in actions]
    saved = save_solution(args.difficulty, score, actions_to_save, seed=args.seed or 0, force=True)
    if saved:
        print(f"Solution saved: solutions/{args.difficulty}/best.json (score={score})")

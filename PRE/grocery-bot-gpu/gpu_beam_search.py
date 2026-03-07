"""GPU-accelerated beam search for grocery bot optimization.

Uses PyTorch CUDA tensors for massive parallel state evaluation.
Single-bot: beam width 10,000-50,000+ with ALL valid actions per round.

On RTX 5090: ~50M state evaluations in <5 seconds (vs ~6K/s on CPU).

Usage:
    python gpu_beam_search.py [difficulty] [--seed SEED] [--beam 10000] [--capture]
"""
from __future__ import annotations

import time
import sys
from typing import Any, Callable

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

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 device: str = 'cuda', num_bots: int = 1,
                 locked_trajectories: dict[str, np.ndarray] | None = None,
                 pipeline_mode: bool = False, pipeline_depth: int = 1,
                 preferred_types: set[int] | None = None,
                 no_compile: bool = False,
                 order_cap: int | None = None,
                 speed_bonus: float = 0.0,
                 preferred_zone: tuple[int, ...] | None = None,
                 order_modulo: int | None = None,
                 order_slot: int | None = None,
                 num_rounds: int | None = None) -> None:
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
        # Multi-dropoff zones (for nightmare etc.)
        self.drop_off_zones = getattr(map_state, 'drop_off_zones', [map_state.drop_off])
        self._multi_drop = len(self.drop_off_zones) > 1
        self.num_bots = num_bots
        # num_rounds: actual game length (300 for easy-expert, 500 for nightmare)
        # Defaults to MAX_ROUNDS (500) for backward compat; callers should pass explicitly
        self.num_rounds = num_rounds if num_rounds is not None else MAX_ROUNDS
        self.pipeline_mode = pipeline_mode  # pre-fetches preview items proactively
        # pipeline_depth: which order ahead to target (1=order+1, 2=order+2, etc.)
        # Only used when pipeline_mode=True.
        self.pipeline_depth = max(1, pipeline_depth)
        # order_cap: max orders this bot should complete before switching to
        # pre-fetch/idle. Forces work distribution across bots in sequential DP.
        # None = unlimited (default). Only used in Pass 1 for Hard/Expert.
        self.order_cap = order_cap
        self.speed_bonus = speed_bonus
        # coord_temperature: 0.0 = full coordination penalties, 1.0 = zero penalties.
        # Used for eval annealing in early refinement iterations (simulated annealing).
        self.coord_temperature = 0.0
        # LNS Order Assignment: soft order-to-bot assignment.
        # order_modulo = number of primary bots, order_slot = this bot's slot.
        # When active_idx % modulo != slot, active delivery value is dampened to 15%.
        self.order_modulo = order_modulo
        self.order_slot = order_slot if order_slot is not None else 0
        # preferred_types: set of item type IDs this bot should specialize in.
        # Soft hint (bonus) to avoid competition between bots for the same types.
        # Set via compute_type_assignments() in gpu_sequential_solver.py.
        self.preferred_types = preferred_types  # set[int] or None
        # Pre-build preferred type mask (used 300x per search in _eval)
        self._pref_mask = None  # set after device is ready, below


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
            self._locked_remaining_planned = None     # [num_rounds+1, num_types] bool GPU
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

        # Pre-build dropoff zone tensors for multi-dropoff checks
        self._drop_xs = [dz[0] for dz in self.drop_off_zones]
        self._drop_ys = [dz[1] for dz in self.drop_off_zones]

        # Enable TF32 for any float32 matmuls (BFS precompute, etc.)
        # TF32 uses Tensor Cores with 10-bit mantissa — exact for integer-valued results
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
                for r in range(self.num_rounds):
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

            # _locked_remaining_planned: int16 [num_rounds+1, num_types] — for each round r,
            # COUNT of how many pickups of type t locked bots will make in [r, num_rounds).
            # Suffix sum: remaining[r] = remaining[r+1] + (pickups at round r).
            # Used in _eval for count-aware coordination.
            remaining = np.zeros((self.num_rounds + 1, self.num_types), dtype=np.int16)
            for r in range(self.num_rounds - 1, -1, -1):
                remaining[r] = remaining[r + 1]  # inherit future pickup counts
                for lb in range(self.num_locked):
                    if self._locked_actions_np[lb, r] == ACT_PICKUP:
                        iidx = int(self._locked_action_items_np[lb, r])
                        if 0 <= iidx < self.num_items:
                            tp = int(_itp[iidx])
                            if 0 <= tp < self.num_types:
                                remaining[r, tp] += 1
            # Upload to GPU as float [num_rounds+1, num_types] for easy arithmetic
            self._locked_remaining_planned = torch.tensor(
                remaining, dtype=torch.float32, device=device)  # [num_rounds+1, num_types]

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

        # === Aisle / corridor detection (for congestion penalty in multi-bot _eval) ===
        # Mirrors spacetime.zig aisle detection: 1-tile wide passages flanked by shelf/wall
        grid_np = map_state.grid  # [H, W] int8
        _corridor_rows = np.zeros(self.H, dtype=bool)
        for y in range(self.H):
            floor_count = np.sum((grid_np[y] == CELL_FLOOR) | (grid_np[y] == CELL_DROPOFF))
            if floor_count * 10 > self.W * 6:
                _corridor_rows[y] = True

        _aisle_columns = np.zeros(self.W, dtype=bool)
        for x in range(1, self.W - 1):
            aisle_cells = 0
            total_floor = 0
            for y in range(self.H):
                if _corridor_rows[y]:
                    continue
                if grid_np[y, x] != CELL_FLOOR and grid_np[y, x] != CELL_DROPOFF:
                    continue
                total_floor += 1
                left = grid_np[y, x - 1]
                right = grid_np[y, x + 1]
                if (left == CELL_SHELF or left == CELL_WALL) and (right == CELL_SHELF or right == CELL_WALL):
                    aisle_cells += 1
            if total_floor > 2 and aisle_cells * 4 > total_floor * 3:
                _aisle_columns[x] = True

        self._aisle_columns = torch.tensor(_aisle_columns, dtype=torch.bool, device=device)  # [W]
        self._corridor_rows = torch.tensor(_corridor_rows, dtype=torch.bool, device=device)  # [H]
        _n_aisles = int(_aisle_columns.sum())

        # Zone-based aisle assignment: bot prefers picking from specific aisle columns.
        # Each aisle column has adjacent shelves with items. Being in your zone
        # reduces cross-traffic and collision with other bots.
        self.preferred_zone = preferred_zone
        self._zone_mask = None  # [W] bool — True for columns in/near preferred aisles
        if preferred_zone and len(preferred_zone) > 0:
            zone_mask = np.zeros(self.W, dtype=bool)
            for col in preferred_zone:
                # Zone includes the aisle column and adjacent shelf columns
                for dx in range(-1, 2):
                    c = col + dx
                    if 0 <= c < self.W:
                        zone_mask[c] = True
            self._zone_mask = torch.tensor(zone_mask, dtype=torch.bool, device=device)

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

        # === Trip table for exact multi-item trip costs ===
        trips = tables.get_trips(map_state)
        trip_gpu = trips.to_gpu(device)
        self.trip_cost_gpu = trip_gpu['cost']             # [N_cells, N_combos] int16
        self.trip_cell_idx_map = trip_gpu['cell_idx_map'] # [H, W] int32
        self._trip_n_cells = trips.n_cells
        self._trip_n_combos = trips.n_combos
        self._del_weights = torch.tensor(
            [1 << i for i in range(MAX_ORDER_SIZE)], dtype=torch.long, device=device)

        # Precompute per (order_idx, delivery_bitmask):
        #   remaining_trip_combo: combo_idx for the next ≤3 items to pick
        #   remaining_after_cost: cost from dropoff for items beyond the first trip
        #   order_full_cost: total rounds to complete full order from dropoff
        from itertools import combinations as _combs
        drop_ci = tables.pos_to_idx[
            (map_state.drop_off[0], map_state.drop_off[1])]
        _bitmask_dim = 1 << MAX_ORDER_SIZE  # 128 for 7-item orders
        _rtc = np.full((self.num_orders, _bitmask_dim), -1, dtype=np.int32)
        _rac = np.zeros((self.num_orders, _bitmask_dim), dtype=np.int16)
        _ofc = np.zeros(self.num_orders, dtype=np.int16)
        for oi in range(self.num_orders):
            oreq = [int(x) for x in all_orders[oi].required]
            n = len(oreq)
            _ofc[oi] = min(trips.order_cost(drop_ci, oreq, drop_ci), 9999)
            for bitmask in range(1 << n):
                remaining = [oreq[j] for j in range(n)
                             if not (bitmask & (1 << j))]
                nr = len(remaining)
                if nr == 0:
                    continue
                if nr <= 3:
                    combo = tuple(sorted(remaining))
                    ci = trips.combo_to_idx.get(combo)
                    if ci is not None:
                        _rtc[oi, bitmask] = ci
                else:
                    # Find best 3-item first trip + rest from dropoff
                    best_total = 9999
                    best_ci = -1
                    best_rest = 0
                    seen = set()
                    for idx3 in _combs(range(nr), 3):
                        fc = tuple(sorted(remaining[i] for i in idx3))
                        if fc in seen:
                            continue
                        seen.add(fc)
                        fci = trips.combo_to_idx.get(fc)
                        if fci is None:
                            continue
                        f_cost = int(trips.cost[drop_ci, fci])
                        rest = [remaining[i] for i in range(nr)
                                if i not in set(idx3)]
                        r_cost = trips.order_cost(
                            drop_ci, rest, drop_ci)
                        if f_cost + r_cost < best_total:
                            best_total = f_cost + r_cost
                            best_ci = fci
                            best_rest = r_cost
                    if best_ci >= 0:
                        _rtc[oi, bitmask] = best_ci
                        _rac[oi, bitmask] = min(best_rest, 9999)
        self.remaining_trip_combo = torch.tensor(
            _rtc, dtype=torch.int32, device=device)
        self.remaining_after_cost = torch.tensor(
            _rac, dtype=torch.int16, device=device)
        self.order_full_cost = torch.tensor(
            _ofc, dtype=torch.int16, device=device)

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

        # Pre-allocate hash shift constants (used 300x per search in _hash)
        self._hash_shifts = torch.arange(
            32, 32 + MAX_ORDER_SIZE, dtype=torch.int64, device=device)

        # Pre-allocate DX/DY as int32 to avoid repeated .to(int32) in _step
        self.DX_i32 = self.DX.to(torch.int32)
        self.DY_i32 = self.DY.to(torch.int32)

        # Pre-allocate fixed action template for _dp_expand [1, N]
        # (broadcast-expanded per state, avoids creating from scratch each call)
        _dp_acts_tmpl = torch.zeros(1, self.dp_num_actions, dtype=torch.int8, device=device)
        _dp_acts_tmpl[0, 0] = ACT_WAIT
        _dp_acts_tmpl[0, 1] = ACT_MOVE_UP
        _dp_acts_tmpl[0, 2] = ACT_MOVE_DOWN
        _dp_acts_tmpl[0, 3] = ACT_MOVE_LEFT
        _dp_acts_tmpl[0, 4] = ACT_MOVE_RIGHT
        _dp_acts_tmpl[0, 5] = ACT_DROPOFF
        _dp_acts_tmpl[0, 6:6 + MAX_ADJ] = ACT_PICKUP
        self._dp_acts_template = _dp_acts_tmpl

        # torch.compile hot-path functions for reduced overhead on repeated calls (300x per search).
        # Only on CUDA; CPU path sees negligible benefit and compile adds overhead.
        # Disabled in live/multi-threaded mode (no_compile=True) to avoid dynamo FX tracing conflicts.
        # Fallback chain: inductor/default (triton kernel fusion) → reduce-overhead
        #   (cudagraphs) → aot_eager
        # Note: reduce-overhead uses CUDA graphs which require careful tensor lifetime
        # management (no referencing outputs across graph replays). 'default' mode gets
        # Triton kernel fusion without the buffer-reuse pitfall.
        if device == 'cuda' and int(torch.__version__.split('.')[0]) >= 2 and not no_compile:
            _compile = None
            try:
                import triton  # noqa: F401
                # 'default' mode: Triton codegen + kernel fusion (no CUDA graphs)
                _compile = lambda fn: torch.compile(fn, mode='default', fullgraph=False)
            except ImportError:
                # Try inductor with C++ wrapper (kernel fusion without triton)
                try:
                    _test = torch.compile(lambda x: x + 1, backend='inductor',
                                          options={'cpp_wrapper': True})
                    _test(torch.zeros(1, device='cuda'))
                    _compile = lambda fn: torch.compile(
                        fn, backend='inductor', options={'cpp_wrapper': True})
                except Exception:
                    try:
                        _compile = lambda fn: torch.compile(
                            fn, backend='aot_eager', fullgraph=False)
                    except Exception:
                        pass  # All compile backends unavailable; will use uncompiled kernels
            if _compile is not None:
                try:
                    self._eval = _compile(self._eval)
                    self._step_candidate_only = _compile(self._step_candidate_only)
                    self._hash = _compile(self._hash)
                except Exception:
                    pass  # Silently fall back to uncompiled

        # Pre-build preferred type mask on GPU (avoids per-_eval allocation)
        if preferred_types is not None and len(preferred_types) > 0:
            pref_mask = torch.zeros(self.num_types + 1, dtype=torch.bool, device=device)
            for t in preferred_types:
                if 0 <= t < self.num_types:
                    pref_mask[t] = True
            self._pref_mask = pref_mask

        dt = time.time() - t0
        walkable_cells = int(self.walkable.sum())
        max_adj_actual = int(adj_count.max())
        print(f"  GPU init: {self.num_items} items, {self.num_types} types, "
              f"{walkable_cells} cells, {self.num_actions} acts(full)/{self.dp_num_actions} acts(dp), "
              f"max_adj={max_adj_actual}, {dt:.2f}s",
              file=sys.stderr)

    def _from_game_state(self, gs: GameState, bot_id: int = 0) -> dict[str, torch.Tensor]:
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

    def _at_drop(self, x, y):
        """Check if position (x, y) tensors are at any dropoff zone."""
        if not self._multi_drop:
            return (x == self.drop_x) & (y == self.drop_y)
        result = (x == self._drop_xs[0]) & (y == self._drop_ys[0])
        for i in range(1, len(self._drop_xs)):
            result = result | ((x == self._drop_xs[i]) & (y == self._drop_ys[i]))
        return result

    def _expand(self, state):
        """Expand beam: each state -> num_actions copies with action tensors.

        Returns (expanded_state, actions[B*N], action_items[B*N]).
        """
        N = self.num_actions
        B = state['bot_x'].shape[0]
        BN = B * N
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(N, output_size=BN)
            else:
                expanded[k] = v.repeat_interleave(N, dim=0, output_size=BN)
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

        # Build action tensor [B, N] from pre-allocated template (avoids zeroing + filling)
        acts = self._dp_acts_template.expand(B, N).clone()
        items = torch.full((B, N), -1, dtype=torch.int16, device=d)

        # Variable pickups: per-state adjacent items (vectorized)
        items[:, 6:6 + self.MAX_ADJ] = per_state_adj

        # Expand: [B, N] -> [B*N]
        BN = B * N
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(N, output_size=BN)
            else:
                expanded[k] = v.repeat_interleave(N, dim=0, output_size=BN)

        actions = acts.reshape(-1)
        action_items = items.reshape(-1)

        # Valid mask: fixed actions valid; pickups only where adj exists (vectorized)
        valid = torch.ones(B, N, dtype=torch.bool, device=d)
        # Pickup slots 6..6+MAX_ADJ: invalid where no adjacent item
        valid[:, 6:6 + self.MAX_ADJ] = per_state_adj >= 0

        # Wall pruning: mask invalid move directions (slots 1-4: UP, DOWN, LEFT, RIGHT)
        valid[:, 1:5] &= self.valid_moves[by, bx]  # [B, 4]

        valid = valid.reshape(-1)

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
        at_drop = self._at_drop(bot_x, bot_y)

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
        valid[:, slot] = at_drop & has_items & has_active_inv & (state['active_idx'] < self.num_orders)
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

        # === Dedup: mark duplicate (action, item) pairs per state as invalid ===
        # Many "move toward type X" candidates produce the same directional move.
        # Hash (action, item) into a key, sort per row, mark later duplicates invalid.
        # Key: action * 1000 + (item + 1) gives unique int per (action, item) pair.
        act_key = all_acts.to(torch.int32) * 1000 + (all_items.to(torch.int32) + 1)  # [B, C]
        # Set invalid slots to unique negative keys so they don't match anything
        act_key = torch.where(valid, act_key, torch.full_like(act_key, -1) - torch.arange(C_actual, device=d).unsqueeze(0))
        # Sort each row to group duplicates
        sorted_keys, sort_perm = act_key.sort(dim=1)
        # Mark first occurrence as valid, later duplicates as invalid
        is_dup = torch.zeros_like(valid)
        is_dup[:, 1:] = (sorted_keys[:, 1:] == sorted_keys[:, :-1]) & (sorted_keys[:, 1:] >= 0)
        # Unsort the dup mask back to original slot order
        _, unsort_perm = sort_perm.sort(dim=1)
        is_dup_orig = is_dup.gather(1, unsort_perm)
        valid = valid & ~is_dup_orig

        # Expand state
        BC = B * C_actual
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(C_actual, output_size=BC)
            else:
                expanded[k] = v.repeat_interleave(C_actual, dim=0, output_size=BC)

        actions = all_acts.reshape(-1)
        action_items = all_items.reshape(-1)
        valid_mask = valid.reshape(-1)

        return expanded, actions, action_items, valid_mask, C_actual

    @torch.no_grad()
    def _step(self, state: dict[str, torch.Tensor], actions: torch.Tensor,
              action_items: torch.Tensor, round_num: int = -1) -> dict[str, torch.Tensor]:
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
                    lb_act = int(self.locked_actions[b, round_num])
                    # Skip WAIT actions entirely — no movement, no game state change
                    if lb_act == ACT_WAIT:
                        continue
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
        # Uses out-of-place ops only (no slice assignment) for CUDA graph compatibility
        score_add = torch.zeros(B, dtype=torch.int32, device=d)
        used_order = torch.zeros(B, MAX_ORDER_SIZE, dtype=torch.bool, device=d)
        new_inv_cols = []

        for s in range(INV_CAP):
            available = match[:, s, :] & ~used_order   # [B, MAX_ORDER_SIZE]
            # First available order slot per state (first True only)
            cumsum = available.to(torch.int32).cumsum(dim=1)
            first_match = available & (cumsum == 1)
            any_match = first_match.any(dim=1)          # [B]

            # Accumulate: mark order slot delivered, collect new inv value, add score
            active_del = active_del + first_match.to(torch.int8)
            new_inv_cols.append(torch.where(any_match, self._neg1_i8, bot_inv_slice[:, s]))
            score_add = score_add + first_match.sum(dim=1).to(torch.int32)
            used_order = used_order | first_match

        bot_inv_slice = torch.stack(new_inv_cols, dim=1)
        return bot_inv_slice, active_del, score_add

    def _auto_deliver_all(self, order_complete, bot_x, bot_y, bot_inv,
                          locked_inv, locked_bx, locked_by,
                          active_idx, active_del, score, B, d):
        """Auto-delivery when an order completes: all bots at dropoff deliver to new active."""
        if not order_complete.any():
            return bot_inv, active_del, score, active_idx

        score = score + order_complete.to(torch.int32) * 5
        new_aidx = active_idx + order_complete.to(torch.int32)  # no clamp — may exceed num_orders
        oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
        active_del = torch.where(oc_mask, self._zero_i8, active_del)

        valid_auto = order_complete & (new_aidx < self.num_orders)
        new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)  # clamp only for table lookup
        new_req = self.order_req[new_aidx_l]

        # Auto-deliver ALL locked bots at dropoff (vectorized per bot)
        if locked_bx is not None:
            for b2 in range(self.num_locked):
                b2_at_drop = self._at_drop(locked_bx[:, b2], locked_by[:, b2])
                b2_auto = valid_auto & b2_at_drop
                if b2_auto.any():
                    linv_b2 = locked_inv[:, b2, :]  # [B, INV_CAP]
                    linv_b2, active_del, score_add = self._vectorized_deliver(
                        linv_b2, new_req, active_del, b2_auto, B, d)
                    locked_inv[:, b2, :] = linv_b2
                    score = score + score_add

        # Auto-deliver candidate bot at dropoff (vectorized)
        cand_at_drop = self._at_drop(bot_x, bot_y)
        auto_cand = valid_auto & cand_at_drop
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
            lb_at_drop = self._at_drop(lbx, lby)
            lb_can_drop = lb_at_drop & (active_idx < self.num_orders)

            aidx_l = active_idx.long().clamp(0, self.num_orders - 1)
            act_req_lb = self.order_req[aidx_l]

            # Vectorized delivery for locked bot
            linv_b = locked_inv[:, b, :]  # [B, INV_CAP]
            linv_b, active_del, score_add = self._vectorized_deliver(
                linv_b, act_req_lb, active_del, lb_can_drop, B, d)
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
            order_complete_lb = slot_done.all(dim=1) & has_required & lb_can_drop

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
        acts_long = actions.long()

        # === MOVEMENT ===
        is_move = (actions >= 1) & (actions <= 4)
        dx = self.DX_i32[acts_long]
        dy = self.DY_i32[acts_long]
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
        at_drop = self._at_drop(bot_x, bot_y)
        has_items = (bot_inv >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items & (active_idx < self.num_orders)

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
        acts_long = actions.long()

        # === MOVEMENT ===
        is_move = (actions >= 1) & (actions <= 4)
        dx = self.DX_i32[acts_long]
        dy = self.DY_i32[acts_long]
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

        # Out-of-place inventory update for CUDA graph compatibility
        added = torch.zeros(B, dtype=torch.bool, device=d)
        inv_cols = []
        for s in range(INV_CAP):
            slot_empty = bot_inv[:, s] < 0
            add_here = can_pickup & slot_empty & ~added
            inv_cols.append(torch.where(add_here, pickup_type, bot_inv[:, s]))
            added = added | add_here
        bot_inv = torch.stack(inv_cols, dim=1)

        # === DROPOFF (vectorized) ===
        is_dropoff = (actions == ACT_DROPOFF)
        at_drop = self._at_drop(bot_x, bot_y)
        has_items = (bot_inv >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items & (active_idx < self.num_orders)

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

        new_aidx = active_idx + order_complete.to(torch.int32)  # no clamp — may exceed num_orders
        oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
        active_del = torch.where(oc_mask, self._zero_i8, active_del)

        # Auto-deliver candidate at dropoff (vectorized)
        valid_auto = order_complete & (new_aidx < self.num_orders)
        new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)  # clamp only for table lookup
        new_req = self.order_req[new_aidx_l]

        bot_inv, active_del, score_add = self._vectorized_deliver(
            bot_inv, new_req, active_del, valid_auto, B, d)
        score = score + score_add

        sort_key = (bot_inv < 0).to(torch.int8)
        _, sort_idx = sort_key.sort(dim=1, stable=True)
        bot_inv = bot_inv.gather(1, sort_idx.long())

        active_idx = new_aidx
        return bot_x, bot_y, bot_inv, active_idx, active_del, score, orders_comp

    @torch.no_grad()
    def _hash(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
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

        # Pack active_del as bits (vectorized, pre-allocated shifts)
        del_bits = state['active_del'].long()  # [B, MAX_ORDER_SIZE]
        h = h | (del_bits << self._hash_shifts.unsqueeze(0)).sum(dim=1)

        return h

    @torch.no_grad()
    def _eval_cheap(self, state: dict[str, torch.Tensor], round_num: int = 0) -> torch.Tensor:
        """Cheap evaluation for two-phase beam pre-filtering. Returns [B] float32.

        Only computes: score, distance to dropoff, inventory matching (active),
        and trip-table feasibility. ~5x faster than full _eval.
        """
        B = state['bot_x'].shape[0]
        d = self.device

        bot_x = state['bot_x'].long()
        bot_y = state['bot_y'].long()
        inv = state['bot_inv']
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)

        rounds_left = self.num_rounds - round_num - 1

        # Score is dominant
        ev = state['score'].float() * 100000

        # Speed bonus: prefer higher scores earlier (more rounds left to discover orders)
        if self.speed_bonus > 0:
            ev = ev + self.speed_bonus * state['score'].float() * rounds_left

        # Distance to dropoff
        dist_drop = self.dist_to_dropoff[bot_y, bot_x].float()

        # Active order analysis
        act_req = self.order_req[aidx]
        act_del = state['active_del']
        active_needed = (act_req >= 0) & (act_del == 0)
        active_done = (act_req >= 0) & (act_del == 1)
        active_remaining = active_needed.sum(dim=1).float()
        active_delivered = active_done.sum(dim=1).float()
        active_total = ((act_req >= 0)).sum(dim=1).float()
        fraction_done = active_delivered / active_total.clamp(min=1)

        ev = ev + fraction_done * 30000
        ev = ev + (active_remaining == 0).float() * 50000

        # Inventory-order matching
        inv_exp = inv.unsqueeze(2)
        req_exp = act_req.unsqueeze(1)
        del_exp = act_del.unsqueeze(1)
        has_item_exp = (inv_exp >= 0)
        match_active = (inv_exp == req_exp) & (del_exp == 0) & (req_exp >= 0) & has_item_exp
        inv_matches_active = match_active.any(dim=2)
        has_active_inv = inv_matches_active.any(dim=1)
        num_active_items = inv_matches_active.sum(dim=1).float()

        active_inv_value = (70000 - dist_drop * 1750).clamp(min=0)
        ev = ev + num_active_items * active_inv_value

        # Dead inventory (quick check)  # _eval_2bot
        has_item_flat = (inv >= 0)
        is_dead = has_item_flat & ~inv_matches_active
        ev = ev - is_dead.sum(dim=1).float() * 40000

        # Trip-table cost
        del_bits = (act_del.long() * self._del_weights).sum(dim=1)
        cell_idx = self.trip_cell_idx_map[bot_y, bot_x]
        trip_combo = self.remaining_trip_combo[aidx, del_bits]
        has_trip_info = (trip_combo >= 0) & (cell_idx >= 0)
        safe_cell = cell_idx.clamp(0, self._trip_n_cells - 1)
        safe_combo = trip_combo.clamp(0, self._trip_n_combos - 1)
        next_trip_cost = self.trip_cost_gpu[safe_cell, safe_combo].float()
        after_cost = self.remaining_after_cost[aidx, del_bits].float()
        total_order_remaining = (
            has_trip_info.float() * (next_trip_cost + after_cost) +
            (~has_trip_info).float() * 9999.0)
        order_achievable = has_trip_info & (total_order_remaining <= rounds_left)
        ev = ev + order_achievable.float() * 8000
        ev = ev - has_trip_info.float() * total_order_remaining.clamp(0, 300) * 50

        # Delivery urgency
        ev = ev - has_active_inv.float() * dist_drop * 200

        return ev

    @torch.no_grad()
    def _eval(self, state: dict[str, torch.Tensor], round_num: int = 0) -> torch.Tensor:
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

        rounds_left = self.num_rounds - round_num - 1
        rl_f = float(max(rounds_left, 1))

        # Score is dominant factor (1 point = 100000 eval units)
        ev = state['score'].float() * 100000

        # Speed bonus: prefer higher scores earlier (more rounds left to discover orders)
        if self.speed_bonus > 0:
            ev = ev + self.speed_bonus * state['score'].float() * rounds_left

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
        # At dist_drop=20: 35000
        # At dist_drop=40: 0 (item far from deliverable)
        active_inv_value = (70000 - dist_drop * 1750).clamp(min=0)
        # LNS Order Assignment: dampen delivery value for non-assigned orders.
        # My order (aidx % modulo == slot): full value. Others: 50%.
        if self.order_modulo is not None:
            is_my_order = ((aidx % self.order_modulo) == self.order_slot).float()
            active_inv_value = active_inv_value * (0.5 + 0.5 * is_my_order)
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
            # Multi-bot: aggressively value preview items from the start.
            # Orders complete in 30-50 rounds with multiple bots delivering,
            # so pre-fetching early creates seamless order transitions.
            # At 0% done: 12000 (was 5000 — much stronger pre-fetch signal)
            # At 50% done: 16000
            # At 100% done: 20000
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
        if is_dead.any():
            for _extra_d in [2, 3]:
                _eidx = (aidx + _extra_d).clamp(0, self.num_orders - 1)
                _ereq = self.order_req[_eidx]   # [B, MAX_ORDER_SIZE]
                _ereq_exp = _ereq.unsqueeze(1)  # [B, 1, MAX_ORDER_SIZE]
                _match_extra = (inv_exp == _ereq_exp) & (_ereq_exp >= 0) & has_item_exp
                _matches_extra = _match_extra.any(dim=2) & ~inv_matches_active & ~inv_matches_preview
                is_dead = is_dead & ~_matches_extra
        # Dead items: catastrophic early (waste whole game), less so late.
        # Multi-bot: softer penalty — speculative items may match future orders.
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

        # Trip feasibility check (single-item, used for distance guidance)
        min_trip_time = best_item_dist + dist_drop + 2  # pickup + dropoff actions
        trip_feasible = close & (min_trip_time <= rounds_left)

        # === Trip-table: exact remaining-order cost from current position ===
        del_bits = (act_del.long() * self._del_weights).sum(dim=1)  # [B]
        cell_idx = self.trip_cell_idx_map[bot_y, bot_x]  # [B]
        trip_combo = self.remaining_trip_combo[aidx, del_bits]  # [B]
        has_trip_info = (trip_combo >= 0) & (cell_idx >= 0)
        safe_cell = cell_idx.clamp(0, self._trip_n_cells - 1)
        safe_combo = trip_combo.clamp(0, self._trip_n_combos - 1)
        next_trip_cost = self.trip_cost_gpu[safe_cell, safe_combo].float()
        after_cost = self.remaining_after_cost[aidx, del_bits].float()
        total_order_remaining = (
            has_trip_info.float() * (next_trip_cost + after_cost) +
            (~has_trip_info).float() * 9999.0)

        # Bonus: entire remaining order is achievable in time
        order_achievable = has_trip_info & (total_order_remaining <= rounds_left)
        ev = ev + order_achievable.float() * 8000
        # Prefer states with less remaining work (smoother gradient than just feasibility)
        # Prefer states with less remaining work (smoother gradient than just feasibility)
        ev = ev - has_trip_info.float() * total_order_remaining.clamp(0, 300) * 50

        # === Endgame partial delivery: when order can't be completed, maximize item pickups ===
        # Each item delivered is +1 point even without completing the order (+5 bonus).
        # When order is unachievable, strongly incentivize: pick up matching items → deliver.
        if rounds_left < 30:
            cant_complete = ~order_achievable & has_space & close
            # Per matching item in inventory: boost delivery urgency
            partial_value = num_active_items * 15000 * (~order_achievable).float()
            ev = ev + partial_value
            # Incentivize picking up more matching items when order can't be completed
            ev = ev + cant_complete.float() * (12000.0 - best_item_dist * 600)

        # Approaching needed items — only for single-bot mode.
        # Multi-bot mode uses differentiated guidance in the coordination block below.
        if self.num_locked == 0:
            _m1 = (trip_feasible & has_space & ~has_active_inv).float()
            ev = ev + _m1 * (20000.0 - best_item_dist * 800)
            _m2 = (trip_feasible & has_space & has_active_inv).float()
            ev = ev + _m2 * (5000.0 - best_item_dist * 200)

        # Bot with active items: closer to dropoff is better
        delivery_urgency = max(1.0, (100 - rounds_left) / 20.0) if rounds_left < 100 else 1.0
        ev = ev - has_active_inv.float() * dist_drop * 200 * delivery_urgency

        # Pipeline/multi-bot: bot holding preview items should position near dropoff
        # as active order nears completion (ready for instant auto-delivery).
        if self.pipeline_mode or self.num_locked > 0:
            has_preview_inv_any = inv_matches_preview.any(dim=1)  # [B]
            preview_drop_guidance = 80.0 * fraction_done + 20.0  # 20..100 per unit dist
            _m3 = (has_preview_inv_any & ~has_active_inv).float()
            ev = ev - _m3 * dist_drop * preview_drop_guidance

        # Can't reach dropoff in time: inventory is completely wasted
        cant_deliver = has_active_inv & (dist_drop >= rounds_left)
        ev = ev - cant_deliver.float() * 80000

        # Empty bot camping at dropoff: penalize blocking deliveries.
        at_drop = self._at_drop(state['bot_x'].long(), state['bot_y'].long())
        camping_penalty = at_drop & ~has_active_inv
        ev = ev - camping_penalty.float() * 12000

        # === Aisle congestion penalty (multi-bot) ===
        # Penalize candidate bot for being in the same 1-tile aisle column as
        # a locked bot. Narrow aisles create deadlocks and wasted rounds.
        # coord_temperature anneals this: 0.0=full penalty, 1.0=no penalty.
        _ct_scale = 1.0 - self.coord_temperature
        if self.num_locked > 0 and 'locked_bx' in state and self._aisle_columns.any():
            _cand_in_aisle = self._aisle_columns[bot_x]  # [B] bool
            _cand_not_corridor = ~self._corridor_rows[bot_y]  # [B] bool
            _cand_in_narrow = _cand_in_aisle & _cand_not_corridor  # [B]
            if _cand_in_narrow.any():
                _lbx = state['locked_bx']  # [B, L]
                _lby = state['locked_by']  # [B, L]
                # Locked bot in same aisle column AND not in corridor row
                _same_col = (_lbx == bot_x.unsqueeze(1))  # [B, L]
                _lby_long = _lby.long().clamp(0, self.H - 1)
                _locked_not_corr = ~self._corridor_rows[_lby_long]  # [B, L]
                _locked_same_aisle = _same_col & _locked_not_corr  # [B, L]
                _num_locked_same = _locked_same_aisle.sum(dim=1).float()  # [B]
                # Scale: -4000 per locked bot in same aisle (moderate — less than
                # active_inv_value but enough to prefer different-aisle routing)
                ev = ev - _cand_in_narrow.float() * _num_locked_same * 4000 * _ct_scale

        # === Dropoff queue system ===
        # When multiple bots are heading to dropoff with items, they should
        # form a queue along the bottom row (y = height-2) approaching from
        # the right. Penalize crowding at dropoff; reward queuing positions.
        if self.num_locked > 0 and 'locked_bx' in state:
            locked_bx_state = state.get('locked_bx')
            locked_by_state = state.get('locked_by')
            if locked_bx_state is not None:
                # Count locked bots near dropoff (within 2 cells)
                # Count locked bots near any dropoff (within 2 cells)
                if not self._multi_drop:
                    locked_near_drop = (
                        (locked_bx_state - self.drop_x).abs() +
                        (locked_by_state - self.drop_y).abs() <= 2
                    ).sum(dim=1).float()  # [B]
                else:
                    # Multi-dropoff: check distance to each zone, take min
                    near_any = None
                    for dz in self.drop_off_zones:
                        near_dz = (locked_bx_state - dz[0]).abs() + (locked_by_state - dz[1]).abs() <= 2
                        near_any = near_dz if near_any is None else (near_any | near_dz)
                    locked_near_drop = near_any.sum(dim=1).float()
        # === Coordination with locked bots ===
        if 'locked_bx' in state and self.num_locked > 0:
            locked_bx_state = state['locked_bx']  # [B, num_locked]
            locked_by_state = state['locked_by']
            # Congestion: penalize being near dropoff when many locked bots are too.
            locked_at_drop_mask = self._at_drop(locked_bx_state[:, 0:1], locked_by_state[:, 0:1])
            for lb_i in range(1, self.num_locked):
                locked_at_drop_mask = locked_at_drop_mask | self._at_drop(
                    locked_bx_state[:, lb_i:lb_i+1], locked_by_state[:, lb_i:lb_i+1])
            # Count per batch: how many locked bots are at any dropoff
            locked_at_drop_counts = torch.zeros(locked_bx_state.shape[0], device=locked_bx_state.device)
            for lb_i in range(self.num_locked):
                locked_at_drop_counts += self._at_drop(
                    locked_bx_state[:, lb_i], locked_by_state[:, lb_i]).float()
            locked_at_drop = locked_at_drop_counts
            near_drop = dist_drop < 3
            ev = ev - near_drop.float() * locked_at_drop * 500 * _ct_scale

            locked_inv_state = state.get('locked_inv')  # [B, num_locked, INV_CAP]
            if locked_inv_state is not None:
                # === Assignment-aware coordination ===
                # For each active order slot, determine if locked bots already cover it
                # (either via current inventory OR planned future pickup).
                # Candidate should target UNCOVERED slots (division of labor).
                locked_inv_l = locked_inv_state.long()  # [B, L, INV_CAP]

                locked_plans_type_mask = self._locked_all_planned_mask  # [num_types] bool | None

                remaining_cover = None
                if self._locked_remaining_planned is not None and round_num >= 0:
                    rn = min(round_num, self.num_rounds)
                    remaining_cover = self._locked_remaining_planned[rn]  # [num_types] float (counts)

                # locked_covers_slot[b, os]: True if any locked bot carries or will pick
                # the type needed by order slot os.
                ntypes_all = act_req.long().clamp(0, self.num_types - 1)  # [B, 6]
                ntypes_4d = ntypes_all.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 6]
                linv_4d = locked_inv_l.unsqueeze(3)  # [B, L, C, 1]
                linv_valid = (locked_inv_state >= 0).unsqueeze(3)  # [B, L, C, 1]
                # [B, L, C, 6] -> any over L,C -> [B, 6]
                locked_covers_slot = ((linv_4d == ntypes_4d) & linv_valid).any(dim=2).any(dim=1)
                if remaining_cover is not None:
                    locked_covers_slot = locked_covers_slot | (remaining_cover[ntypes_all] > 0)

                covered_by_locked = active_needed & locked_covers_slot      # [B, 6]
                uncovered_by_locked = active_needed & ~locked_covers_slot   # [B, 6]

                # Candidate inventory items matching order slots
                type_match_grid = ((inv_exp == req_exp) & has_item_exp &
                                   (req_exp >= 0))  # [B, INV_CAP, MAX_ORDER_SIZE]
                cov_exp = covered_by_locked.unsqueeze(1)   # [B, 1, 6]
                uncov_exp = uncovered_by_locked.unsqueeze(1)  # [B, 1, 6]
                cand_redundant = (type_match_grid & cov_exp).float().sum(dim=(1, 2))
                cand_covers_unique = (type_match_grid & uncov_exp).float().sum(dim=(1, 2))

                # Coordination: unique coverage is rewarded, redundancy penalized.
                # Penalty is intentionally LESS than active_inv_value (~70K) because
                # "redundant" is approximate — locked bots deliver and free slots,
                # so temporary redundancy often resolves naturally during the game.
                _nl = self.num_locked
                unique_bonus = 30000 + _nl * 3000   # 33K (medium) to 57K (expert)
                redundant_pen = (20000 + _nl * 2000) * _ct_scale  # annealed
                ev = ev + cand_covers_unique * unique_bonus
                ev = ev - cand_redundant * redundant_pen

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
                _mu1 = (trip_u & has_space & ~has_active_inv).float()
                ev = ev + _mu1 * (20000.0 - best_uncovered_dist * 800)
                _mu2 = (trip_u & has_space & has_active_inv).float()
                ev = ev + _mu2 * (5000.0 - best_uncovered_dist * 200)

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
                            already_planned = (_preview_cover[pt] > 0) if _preview_cover.dtype != torch.bool else _preview_cover[pt]
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
                    _mp1 = (all_active_covered & trip_p & has_space & ~has_active_inv).float()
                    ev = ev + _mp1 * (16000.0 - best_prev_dist * 800)

                    # Deep pipeline toward order+2 (when order+1 fully covered by locked bots)
                    _mp2 = (all_active_covered & ~close_p & trip_pp & has_space & ~has_active_inv).float()
                    ev = ev + _mp2 * (14000.0 - best_pp_dist * 800)

                    # --- Priority 3: No preview options → weak covered fallback ---
                    no_options = all_active_covered & ~close_p & ~close_pp
                    _mp3 = (no_options & trip_c & has_space & ~has_active_inv).float()
                    ev = ev + _mp3 * (10000.0 - best_covered_dist * 500)

        # === Type specialization bonus (soft assignment) ===
        # Encourages this bot to pick items of its preferred types, reducing
        # competition with other bots and improving natural work division.
        if self._pref_mask is not None:
            pref_mask = self._pref_mask

            # Bonus for carrying preferred type items in inventory (vectorized)
            inv_safe = inv.long().clamp(0, self.num_types)  # [B, INV_CAP]
            has_item_flat2 = (inv >= 0)  # [B, INV_CAP]
            is_pref_all = pref_mask[inv_safe]  # [B, INV_CAP]
            ev = ev + (is_pref_all & has_item_flat2).float().sum(dim=1) * 8000

            # Guidance bonus: moving toward nearest preferred type when empty/space
            # Vectorized: stack all preferred type distances, take min
            pref_list = [t for t in self.preferred_types if 0 <= t < self.num_types]
            if pref_list:
                pref_t = torch.tensor(pref_list, dtype=torch.long, device=d)
                # dist_to_type[pref_types, bot_y, bot_x] → [len(pref_list), B]
                pref_dists = self.dist_to_type[pref_t.unsqueeze(1), bot_y.unsqueeze(0), bot_x.unsqueeze(0)]
                best_pref_dist = pref_dists.float().min(dim=0).values  # [B]
            else:
                best_pref_dist = torch.full((B,), 9999.0, device=d)
            close_pref = best_pref_dist < 9999
            trip_pref = close_pref & ((best_pref_dist + dist_drop + 2) <= rounds_left)
            _mpref = (trip_pref & has_space & ~has_active_inv).float()
            ev = ev + _mpref * (6000.0 - best_pref_dist * 300)

        # === Order cap: force work distribution across bots ===
        # After completing order_cap orders, this bot should STOP actively
        # delivering and switch to pre-fetching for the next order.
        # This prevents bot 0 from hogging all orders in sequential Pass 1.
        if self.order_cap is not None and self.order_cap > 0:
            oc = state['orders_comp']  # [B] int32 — orders completed by THIS bot
            over_cap = (oc >= self.order_cap)  # [B] bool
            if over_cap.any():
                # Kill delivery incentive: penalize carrying/delivering active items
                ev = ev - over_cap.float() * has_active_inv.float() * 100000
                ev = ev - over_cap.float() * num_active_items * 60000
                # Kill pickup incentive: penalize being near active items
                ev = ev - over_cap.float() * close.float() * 30000
                # Boost pre-fetch: strongly encourage picking preview/future items
                has_preview = inv_matches_preview.any(dim=1)  # [B]
                ev = ev + over_cap.float() * has_preview.float() * 50000
                # Position near dropoff for auto-delivery when order transitions
                ev = ev - over_cap.float() * dist_drop * 300

        return ev

    @torch.no_grad()
    def search(self, game_state: GameState, beam_width: int = 10000,
               verbose: bool = True) -> tuple[int, list[tuple[int, int]]]:
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

        for rnd in range(self.num_rounds):
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

            # Two-phase beam: cheap pre-filter → full eval on survivors
            new_B = new_state['bot_x'].shape[0]
            k = min(beam_width, new_B)

            # Phase 1: cheap pre-filter when expanded set is large enough
            prefilter_k = min(k * 2, new_B)
            if new_B > k * 3:
                # Cheap eval: score + distance + basic matching (~5x faster)
                cheap_evals = self._eval_cheap(new_state, round_num=rnd)
                cheap_evals[~valid_mask] = float('-inf')
                cheap_evals[is_noop_dup] = float('-inf')
                _, prefilter_idx = torch.topk(cheap_evals, prefilter_k)
                # Gather pre-filtered states
                pf_state = {key: val[prefilter_idx] for key, val in new_state.items()}
                pf_actions = actions[prefilter_idx]
                pf_action_items = action_items[prefilter_idx]
                # Phase 2: full eval on reduced set
                evals = self._eval(pf_state, round_num=rnd)
                _, topk_local = torch.topk(evals, k)
                topk_idx = prefilter_idx[topk_local]
            else:
                # Small beam: full eval on everything
                evals = self._eval(new_state, round_num=rnd)
                evals[~valid_mask] = float('-inf')
                evals[is_noop_dup] = float('-inf')
                _, topk_idx = torch.topk(evals, k)

            N = C  # candidates per parent state

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

            # Verbose diagnostics (only sync to CPU when printing)
            if verbose and (rnd < 5 or rnd % 25 == 0 or rnd == self.num_rounds - 1):
                dt = time.time() - t_rnd
                best_score = state['score'].max().item()
                num_unique = 0
                if rnd < 5 or rnd % 50 == 0 or rnd == self.num_rounds - 1:
                    hashes = self._hash(state)
                    num_unique = int(torch.unique(hashes).shape[0])
                uniq_str = f", unique={num_unique}" if num_unique > 0 else ""
                print(f"  R{rnd:3d}: score={best_score:3d}, beam={k}, "
                      f"states={new_B}{uniq_str}, dt={dt:.3f}s",
                      file=sys.stderr)

        # Find best final state
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Reconstruct action sequence by backtracking through parent pointers
        actions_seq = []
        idx = best_idx

        wait_pad = [(ACT_WAIT, -1)] * (self.num_bots - 1)
        for rnd in range(self.num_rounds - 1, -1, -1):
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

    def verify_against_cpu(self, game_state: GameState, all_orders: list[Order],
                           num_rounds: int = 100) -> bool:
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
            f"Score mismatch: CPU={cpu_score}, GPU={gpu_score}"  # nosec B101
        assert gpu_x == cpu_x and gpu_y == cpu_y, \
            f"Position mismatch: CPU=({cpu_x},{cpu_y}), GPU=({gpu_x},{gpu_y})"  # nosec B101
        assert gpu_oc == cpu_oc, \
            f"Orders mismatch: CPU={cpu_oc}, GPU={gpu_oc}"  # nosec B101

        print(f"    MATCH! Score={gpu_score}, Orders={gpu_oc}", file=sys.stderr)
        return True

    @torch.no_grad()
    def dp_search(self, game_state: GameState | None, max_states: int = 500000,
                  verbose: bool = True,
                  on_round: Callable[[int, int, int, int, float], None] | None = None,
                  bot_id: int = 0, start_rnd: int = 0,
                  max_rounds: int | None = None,
                  init_state: dict[str, torch.Tensor] | None = None
                  ) -> tuple[int, list[tuple[int, int]]]:
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
            max_rounds: Max rounds to run (default num_rounds - start_rnd).
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
            max_rounds = self.num_rounds - start_rnd

        # Use position-filtered expand (6+MAX_ADJ actions vs 6+num_items)
        use_dp_expand = True
        N = self.dp_num_actions if use_dp_expand else self.num_actions

        # History for action reconstruction (kept on GPU to avoid per-round CPU sync,
        # transferred to CPU only during final backtracking).
        # With 32GB VRAM, ~200K states × 300 rounds uses <1GB total.
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

            # === Pre-filter: step only valid actions (saves ~40% of _step work) ===
            if dp_valid is not None:
                filt_idx = dp_valid.nonzero(as_tuple=True)[0]
                filt_expanded = {k: v[filt_idx] for k, v in expanded.items()}
                filt_actions = actions[filt_idx]
                filt_items = action_items[filt_idx]
                filt_parent = filt_idx // N  # maps to parent index in state
                BF = filt_idx.shape[0]
            else:
                filt_expanded = expanded
                filt_actions = actions
                filt_items = action_items
                filt_parent = None
                BF = B * N

            # Step only valid expanded states
            new_state = self._step(filt_expanded, filt_actions, filt_items, round_num=rnd)

            # Detect no-ops among valid states
            no_change = (
                (new_state['bot_x'] == filt_expanded['bot_x']) &
                (new_state['bot_y'] == filt_expanded['bot_y']) &
                ((new_state['bot_inv'] == filt_expanded['bot_inv']).all(dim=1)) &
                (new_state['score'] == filt_expanded['score']) &
                (new_state['active_idx'] == filt_expanded['active_idx'])
            )
            is_noop = no_change & (filt_actions != ACT_WAIT)

            # Hash states for dedup
            hashes = self._hash(new_state)
            hashes[is_noop] = -1  # mark invalid

            # Dedup: sort by hash, keep highest-scoring representative per hash.
            # Two stable sorts: first by score descending, then by hash ascending.
            # Stable sort preserves score ordering within same-hash groups,
            # so is_first picks the highest-scoring state per hash.
            if self.num_locked > 0:
                _, score_order = new_state['score'].sort(descending=True, stable=True)
                hashes_scored = hashes[score_order]
                sorted_h, hash_order = hashes_scored.sort(stable=True)
                sort_idx = score_order[hash_order]
            else:
                # Single-bot: score is deterministic from hash, arbitrary pick is fine
                sorted_h, sort_idx = hashes.sort()
            is_first = torch.ones(BF, dtype=torch.bool, device=d)
            is_first[1:] = sorted_h[1:] != sorted_h[:-1]
            valid = is_first & (sorted_h != -1)

            # Indices of unique valid states in the filtered array
            unique_idx = sort_idx[valid]
            B_new = unique_idx.shape[0]

            # Parent tracking: map filtered→original parent index
            if filt_parent is not None:
                parent_idx = filt_parent[unique_idx]
            else:
                parent_idx = unique_idx // N
            parent_idx_history.append(parent_idx)
            act_history.append(filt_actions[unique_idx])
            item_history.append(filt_items[unique_idx])

            # Gather unique states
            state = {k: v[unique_idx] for k, v in new_state.items()}

            # Adaptive beam: allow 2x headroom when state diversity is high
            diversity = B_new / BF if BF > 0 else 0
            effective_max = int(max_states * 2) if diversity > 0.5 else max_states

            # Periodic pruning: every 10 rounds, prune to 80% of max_states
            # even when below the hard limit. Prevents slow growth to limit.
            soft_prune = (i > 0 and i % 10 == 0 and
                          B_new > int(max_states * 0.6) and
                          B_new <= effective_max)
            if soft_prune:
                effective_max = int(max_states * 0.8)

            # If too many states, prune by eval (lossy fallback)
            if B_new > effective_max:
                pruned_rounds += 1
                evals = self._eval(state, round_num=rnd)
                keep = min(effective_max, B_new)

                # Threshold pruning before topk (avoids .item() sync)
                # Filter out obviously bad states, then topk on survivors
                if B_new > keep * 2:
                    max_eval = evals.max()
                    threshold = max_eval - 50000  # ~5 score points
                    survivor_mask = evals >= threshold
                    survivor_idx = survivor_mask.nonzero(as_tuple=True)[0]
                    sc = survivor_idx.shape[0]
                    if sc > keep and sc < B_new * 0.9:
                        survivor_evals = evals[survivor_idx]
                        _, local_topk = torch.topk(survivor_evals, keep)
                        topk = survivor_idx[local_topk]
                    else:
                        _, topk = torch.topk(evals, keep)
                else:
                    _, topk = torch.topk(evals, keep)

                topk_sorted, _ = topk.sort()  # maintain order for consistency
                state = {k: v[topk_sorted] for k, v in state.items()}
                # History is on GPU — index directly (no CPU transfer needed)
                parent_idx_history[-1] = parent_idx_history[-1][topk_sorted]
                act_history[-1] = act_history[-1][topk_sorted]
                item_history[-1] = item_history[-1][topk_sorted]
                B_new = keep

            # Only sync to CPU when needed for verbose output or callbacks
            _need_score = on_round and (i < 10 or i % 5 == 0 or i == max_rounds - 1)
            _need_verbose = verbose and (i < 10 or i % 25 == 0 or i == max_rounds - 1)
            if _need_score or _need_verbose:
                dt = time.time() - t0
                best_score = state['score'].max().item()
                if _need_verbose:
                    print(f"  R{rnd:3d}: score={best_score:3d}, "
                          f"unique={B_new}, expanded={BF}, "
                          f"t={dt:.1f}s", file=sys.stderr)
                if on_round:
                    on_round(rnd, best_score, B_new, BF, dt)

        # Find best final state
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Backtrack to reconstruct action sequence (transfer history from GPU to CPU)
        # Batch transfer: move all history to CPU at once before backtracking
        parent_idx_history = [h.cpu() for h in parent_idx_history]
        act_history = [h.cpu() for h in act_history]
        item_history = [h.cpu() for h in item_history]

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


class GPUBeamSearcher2Bot(GPUBeamSearcher):
    """Joint 2-bot GPU DP search.

    Plans 2 candidate bots simultaneously via Cartesian product of actions.
    For Hard (5 bots): plan pairs (0,1), (2,3), solo 4 → better coordination.

    State representation:
        bot1_x, bot1_y: [B] int16  — first candidate bot position
        bot1_inv: [B, 3] int8      — first candidate inventory
        bot2_x, bot2_y: [B] int16  — second candidate bot position
        bot2_inv: [B, 3] int8      — second candidate inventory
        active_idx: [B] int32      — shared order index
        active_del: [B, 6] int8    — shared delivery state
        score: [B] int32           — joint cumulative score
        orders_comp: [B] int32     — joint orders completed
        (plus locked_inv/locked_bx/locked_by when other bots are locked)
    """

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 device: str = 'cuda', num_bots: int = 1,
                 locked_trajectories: dict[str, np.ndarray] | None = None,
                 candidate_bot_ids: tuple[int, int] = (0, 1),
                 no_compile: bool = False,
                 speed_bonus: float = 0.0,
                 num_rounds: int | None = None) -> None:
        # Store candidate IDs before parent init (parent sets candidate_bot_id=0)
        self.bot1_real_id = candidate_bot_ids[0]
        self.bot2_real_id = candidate_bot_ids[1]
        super().__init__(
            map_state, all_orders, device=device, num_bots=num_bots,
            locked_trajectories=locked_trajectories,
            no_compile=no_compile, speed_bonus=speed_bonus,
            num_rounds=num_rounds)

    def _from_game_state_2bot(self, gs: GameState,
                               bot1_id: int = 0, bot2_id: int = 1
                               ) -> dict[str, torch.Tensor]:
        """Convert CPU GameState to 2-bot GPU state dict (batch size 1)."""
        d = self.device
        state = {
            'bot1_x': torch.tensor(
                [int(gs.bot_positions[bot1_id, 0])], dtype=torch.int16, device=d),
            'bot1_y': torch.tensor(
                [int(gs.bot_positions[bot1_id, 1])], dtype=torch.int16, device=d),
            'bot1_inv': torch.tensor(
                [[int(gs.bot_inventories[bot1_id, s]) for s in range(INV_CAP)]],
                dtype=torch.int8, device=d),
            'bot2_x': torch.tensor(
                [int(gs.bot_positions[bot2_id, 0])], dtype=torch.int16, device=d),
            'bot2_y': torch.tensor(
                [int(gs.bot_positions[bot2_id, 1])], dtype=torch.int16, device=d),
            'bot2_inv': torch.tensor(
                [[int(gs.bot_inventories[bot2_id, s]) for s in range(INV_CAP)]],
                dtype=torch.int8, device=d),
            'active_idx': torch.tensor([0], dtype=torch.int32, device=d),
            'active_del': torch.zeros((1, MAX_ORDER_SIZE), dtype=torch.int8, device=d),
            'score': torch.tensor([0], dtype=torch.int32, device=d),
            'orders_comp': torch.tensor([0], dtype=torch.int32, device=d),
        }
        if self.num_locked > 0:
            state['locked_inv'] = torch.full(
                (1, self.num_locked, INV_CAP), -1, dtype=torch.int8, device=d)
            sx, sy = self.ms.spawn
            state['locked_bx'] = torch.full(
                (1, self.num_locked), sx, dtype=torch.int16, device=d)
            state['locked_by'] = torch.full(
                (1, self.num_locked), sy, dtype=torch.int16, device=d)
        return state

    @torch.no_grad()
    def _hash_2bot(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute 2-bot state hash for dedup. Returns [B] int64.

        Packing (61 bits total, fits int64):
          bot1_x(5) + bot1_y(5) + bot1_inv(15) +
          bot2_x(5) + bot2_y(5) + bot2_inv(15) +
          active_idx(5) + active_del(6)
        """
        sorted_inv1, _ = state['bot1_inv'].sort(dim=1)
        sorted_inv2, _ = state['bot2_inv'].sort(dim=1)

        h = (state['bot1_x'].long()
             | (state['bot1_y'].long() << 5)
             | ((sorted_inv1[:, 0].long() + 1) << 10)
             | ((sorted_inv1[:, 1].long() + 1) << 15)
             | ((sorted_inv1[:, 2].long() + 1) << 20)
             | (state['bot2_x'].long() << 25)
             | (state['bot2_y'].long() << 30)
             | ((sorted_inv2[:, 0].long() + 1) << 35)
             | ((sorted_inv2[:, 1].long() + 1) << 40)
             | ((sorted_inv2[:, 2].long() + 1) << 45)
             | (state['active_idx'].long() << 50))

        del_bits = state['active_del'].long()
        _shifts_2bot = torch.arange(55, 55 + MAX_ORDER_SIZE,
                                    dtype=torch.int64, device=state['bot1_x'].device)
        h = h | (del_bits << _shifts_2bot.unsqueeze(0)).sum(dim=1)

        return h

    @torch.no_grad()
    def _score_single_bot_actions(self, state, bot_key: str, round_num: int):
        """Lightweight per-action scoring for one bot in 2-bot state.

        Scores each of N actions based on: pickup value > dropoff > move-toward-item > wait.
        Uses distance heuristics — fast enough to run every round.

        Returns [B, N] float32 action values (higher = better).
        """
        B = state[f'{bot_key}_x'].shape[0]
        d = self.device
        N = self.dp_num_actions

        bx = state[f'{bot_key}_x'].long()
        by = state[f'{bot_key}_y'].long()
        inv = state[f'{bot_key}_inv']
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]  # [B, MAX_ORDER_SIZE]
        act_del = state['active_del']

        # Base action scores
        scores = torch.zeros(B, N, device=d)

        # Score pickups (indices 6+): huge bonus for matching active/preview order
        adj = self.adj_items[by, bx]  # [B, MAX_ADJ]
        inv_count = (inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP
        active_needed = (act_req >= 0) & (act_del == 0)
        needed_types = set()
        for os in range(MAX_ORDER_SIZE):
            needed = active_needed[:, os]
            if needed.any():
                needed_types.add(os)

        for ai in range(self.MAX_ADJ):
            item_idx = adj[:, ai].long()
            item_valid = item_idx >= 0
            itype = self.item_types[item_idx.clamp(0)]  # [B]
            # Check if type matches active order
            matches_active = torch.zeros(B, dtype=torch.bool, device=d)
            for os in range(MAX_ORDER_SIZE):
                matches_active = matches_active | (
                    (itype == act_req[:, os]) & (act_del[:, os] == 0) & (act_req[:, os] >= 0))
            # Preview match
            pidx = (aidx + 1).clamp(0, self.num_orders - 1)
            prev_req = self.order_req[pidx]
            matches_preview = torch.zeros(B, dtype=torch.bool, device=d)
            for os in range(MAX_ORDER_SIZE):
                matches_preview = matches_preview | ((itype == prev_req[:, os]) & (prev_req[:, os] >= 0))

            slot = 6 + ai
            if slot < N:
                scores[:, slot] += (item_valid & has_space & matches_active).float() * 100.0
                scores[:, slot] += (item_valid & has_space & matches_preview & ~matches_active).float() * 40.0
                scores[:, slot] += (item_valid & has_space & ~matches_active & ~matches_preview).float() * -50.0

        # Score dropoff (index 5): high if carrying active items
        has_active = (inv.unsqueeze(2) == act_req.unsqueeze(1)).any(dim=2).any(dim=1)
        at_drop = self._at_drop(bx, by)
        scores[:, 5] += (has_active & at_drop).float() * 120.0
        scores[:, 5] += (has_active & ~at_drop).float() * -10.0  # don't dropoff when not at dropoff

        # Score moves (indices 1-4): prefer moving toward needed items / dropoff
        dist_drop = self.dist_to_dropoff[by, bx].float()
        for mi, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx = (bx + dx).clamp(0, self.W - 1)
            ny = (by + dy).clamp(0, self.H - 1)
            new_dist_drop = self.dist_to_dropoff[ny, nx].float()
            # Toward dropoff when carrying items
            scores[:, mi + 1] += has_active.float() * (dist_drop - new_dist_drop) * 5.0
            # Toward nearest needed item when not carrying
            if not has_active.all():
                for os in range(MAX_ORDER_SIZE):
                    needed = active_needed[:, os] & ~has_active
                    if needed.any():
                        nt = act_req[:, os].long().clamp(0, self.num_types - 1)
                        old_d = self.dist_to_type[nt, by, bx].float()
                        new_d = self.dist_to_type[nt, ny, nx].float()
                        scores[:, mi + 1] += needed.float() * (old_d - new_d) * 3.0

        # Wait (index 0): slight penalty to prefer action
        scores[:, 0] -= 2.0

        return scores

    @torch.no_grad()
    def _dp_expand_2bot_sparse(self, state, round_num: int = 0):
        """Spatially-sparse joint expansion for 2-bot DP.

        Instead of expanding the full N*N grid for all states, computes
        which (state, action1, action2) triples are worth exploring:
        - dist > 5: top-1 x top-1 = 1 combo per state (independent)
        - dist 3-5: top-3 x top-3 = 9 combos per state (mild coupling)
        - dist <= 2: all valid combos (tight coupling for collision resolution)

        Uses .nonzero() to extract sparse valid combinations, reducing
        expanded states from B*N*N (~5M) to ~150K-300K.

        Returns (expanded_states, b1_acts, b1_items, b2_acts, b2_items,
                 parent_indices) — all already filtered to valid combos only.
        """
        B = state['bot1_x'].shape[0]
        d = self.device
        N = self.dp_num_actions

        # Bot1 actions from bot1's position
        by1 = state['bot1_y'].long()
        bx1 = state['bot1_x'].long()
        adj1 = self.adj_items[by1, bx1]

        acts1 = self._dp_acts_template.expand(B, N).clone()
        items1 = torch.full((B, N), -1, dtype=torch.int16, device=d)
        items1[:, 6:6 + self.MAX_ADJ] = adj1
        valid1 = torch.ones(B, N, dtype=torch.bool, device=d)
        valid1[:, 6:6 + self.MAX_ADJ] = adj1 >= 0
        valid1[:, 1:5] &= self.valid_moves[by1, bx1]

        # Bot2 actions from bot2's position
        by2 = state['bot2_y'].long()
        bx2 = state['bot2_x'].long()
        adj2 = self.adj_items[by2, bx2]

        acts2 = self._dp_acts_template.expand(B, N).clone()
        items2 = torch.full((B, N), -1, dtype=torch.int16, device=d)
        items2[:, 6:6 + self.MAX_ADJ] = adj2
        valid2 = torch.ones(B, N, dtype=torch.bool, device=d)
        valid2[:, 6:6 + self.MAX_ADJ] = adj2 >= 0
        valid2[:, 1:5] &= self.valid_moves[by2, bx2]

        # === Build [B, N, N] validity mask with distance-adaptive pruning ===
        # Start with physical validity: both actions must be independently valid
        joint_valid = valid1.unsqueeze(2) & valid2.unsqueeze(1)  # [B, N, N]

        # Distance-based pruning
        mdist = (bx1 - bx2).abs() + (by1 - by2).abs()  # [B]
        is_close = mdist <= 2
        is_medium = (mdist > 2) & (mdist <= 5)
        is_far = mdist > 5

        n_needs_pruning = (is_medium | is_far).sum().item()

        if n_needs_pruning > 0:
            # Lightweight proxy scores for top-K selection
            scores1 = self._score_single_bot_actions(state, 'bot1', round_num)
            scores2 = self._score_single_bot_actions(state, 'bot2', round_num)
            scores1 = torch.where(valid1, scores1, torch.full_like(scores1, -1e9))
            scores2 = torch.where(valid2, scores2, torch.full_like(scores2, -1e9))

            # === FAR states: top-1 x top-1 ===
            if is_far.any():
                far_idx = is_far.nonzero(as_tuple=True)[0]
                top1_a1 = scores1[far_idx].argmax(dim=1)  # [F]
                top1_a2 = scores2[far_idx].argmax(dim=1)  # [F]
                # Zero out joint_valid for far states, then re-enable only top1
                joint_valid[far_idx] = False
                joint_valid[far_idx, top1_a1, top1_a2] = True

            # === MEDIUM states: top-3 x top-3 ===
            if is_medium.any():
                med_idx = is_medium.nonzero(as_tuple=True)[0]
                M = med_idx.shape[0]
                top3_a1 = scores1[med_idx].topk(min(3, N), dim=1).indices  # [M, 3]
                top3_a2 = scores2[med_idx].topk(min(3, N), dim=1).indices  # [M, 3]
                # Zero out, then re-enable top3 x top3
                joint_valid[med_idx] = False
                t1 = top3_a1.unsqueeze(2)  # [M, 3, 1]
                t2 = top3_a2.unsqueeze(1)  # [M, 1, 3]
                joint_valid[
                    med_idx.unsqueeze(1).unsqueeze(2).expand(M, min(3, N), min(3, N)),
                    t1.expand(M, min(3, N), min(3, N)),
                    t2.expand(M, min(3, N), min(3, N))
                ] = True

        # === Sparse extraction via .nonzero() ===
        state_idx, a1_idx, a2_idx = joint_valid.nonzero(as_tuple=True)
        BF = state_idx.shape[0]

        if BF == 0:
            return None, None, None, None, None, None

        # Gather parent states (only the valid combinations)
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v[state_idx]
            else:
                expanded[k] = v[state_idx]

        # Gather corresponding actions
        b1_acts = acts1[state_idx, a1_idx]
        b1_items = items1[state_idx, a1_idx]
        b2_acts = acts2[state_idx, a2_idx]
        b2_items = items2[state_idx, a2_idx]

        return expanded, b1_acts, b1_items, b2_acts, b2_items, state_idx

    def _move_candidate(self, acts, cand_x, cand_y,
                        other_xs, other_ys,
                        locked_bx, locked_by,
                        B, d, spawn_x, spawn_y):
        """Move one candidate bot with collision checks against others."""
        acts_long = acts.long()
        is_move = (acts >= 1) & (acts <= 4)
        dx = self.DX_i32[acts_long]
        dy = self.DY_i32[acts_long]
        nx = cand_x.to(torch.int32) + dx
        ny = cand_y.to(torch.int32) + dy

        in_bounds = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
        ny_safe = ny.clamp(0, self.H - 1)
        nx_safe = nx.clamp(0, self.W - 1)
        is_walkable = self.walkable[ny_safe.long(), nx_safe.long()]
        can_move = is_move & in_bounds & is_walkable

        at_spawn = (nx == spawn_x) & (ny == spawn_y)

        # Collision with other candidate bots
        for ox, oy in zip(other_xs, other_ys):
            coll = ((nx == ox.to(torch.int32)) &
                    (ny == oy.to(torch.int32)) & ~at_spawn)
            can_move = can_move & ~coll

        # Collision with locked bots
        if locked_bx is not None and self.num_locked > 0:
            all_lb_x = locked_bx.to(torch.int32)
            all_lb_y = locked_by.to(torch.int32)
            all_coll = ((nx.unsqueeze(1) == all_lb_x) &
                        (ny.unsqueeze(1) == all_lb_y) &
                        ~at_spawn.unsqueeze(1))
            any_coll = all_coll.any(dim=1)
            can_move = can_move & ~any_coll

        new_x = torch.where(can_move, nx.to(torch.int16), cand_x)
        new_y = torch.where(can_move, ny.to(torch.int16), cand_y)
        return new_x, new_y

    def _pickup_candidate(self, acts, items_arg, cand_x, cand_y, cand_inv, B, d):
        """Pickup for one candidate bot."""
        is_pickup = (acts == ACT_PICKUP)
        item_idx = items_arg.long().clamp(0, self.num_items - 1)
        inv_count = (cand_inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP

        ix = self.item_pos_x[item_idx].to(torch.int32)
        iy = self.item_pos_y[item_idx].to(torch.int32)
        mdist = (cand_x.to(torch.int32) - ix).abs() + (cand_y.to(torch.int32) - iy).abs()
        adjacent = mdist == 1

        can_pickup = is_pickup & has_space & adjacent
        pickup_type = self.item_types[item_idx]

        added = torch.zeros(B, dtype=torch.bool, device=d)
        inv_cols = []
        for s in range(INV_CAP):
            slot_empty = cand_inv[:, s] < 0
            add_here = can_pickup & slot_empty & ~added
            inv_cols.append(torch.where(add_here, pickup_type, cand_inv[:, s]))
            added = added | add_here
        return torch.stack(inv_cols, dim=1)

    def _dropoff_candidate(self, acts, cand_x, cand_y, cand_inv,
                           active_idx, active_del, score, orders_comp,
                           other_cand_xs, other_cand_ys, other_cand_invs,
                           locked_inv, locked_bx, locked_by, B, d):
        """Dropoff + order completion + auto-deliver for one candidate bot.

        After order completion, auto-delivers from: locked bots, then other
        candidate bots, then this candidate bot — all in real ID order.
        """
        is_dropoff = (acts == ACT_DROPOFF)
        at_drop = self._at_drop(cand_x, cand_y)
        has_items = (cand_inv >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items & (active_idx < self.num_orders)

        aidx = active_idx.long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]

        cand_inv, active_del, score_add = self._vectorized_deliver(
            cand_inv, act_req, active_del, can_dropoff, B, d)
        score = score + score_add

        sort_key = (cand_inv < 0).to(torch.int8)
        _, sort_idx = sort_key.sort(dim=1, stable=True)
        cand_inv = cand_inv.gather(1, sort_idx.long())

        # Order completion check
        slot_done = (active_del == 1) | (act_req < 0)
        has_required = (act_req >= 0).any(dim=1)
        order_complete = slot_done.all(dim=1) & can_dropoff & has_required

        if order_complete.any():
            orders_comp = orders_comp + order_complete.to(torch.int32)
            score = score + order_complete.to(torch.int32) * 5
            new_aidx = active_idx + order_complete.to(torch.int32)
            oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
            active_del = torch.where(oc_mask, self._zero_i8, active_del)

            valid_auto = order_complete & (new_aidx < self.num_orders)
            new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)
            new_req = self.order_req[new_aidx_l]

            # Auto-deliver locked bots at dropoff
            if locked_bx is not None:
                for lb in range(self.num_locked):
                    lb_at_drop = self._at_drop(locked_bx[:, lb], locked_by[:, lb])
                    lb_auto = valid_auto & lb_at_drop
                    if lb_auto.any():
                        linv = locked_inv[:, lb, :]
                        linv, active_del, sa = self._vectorized_deliver(
                            linv, new_req, active_del, lb_auto, B, d)
                        locked_inv[:, lb, :] = linv
                        score = score + sa

            # Auto-deliver other candidate bots at dropoff
            for ox, oy, oinv_ref in zip(other_cand_xs, other_cand_ys, other_cand_invs):
                o_at_drop = self._at_drop(ox, oy)
                o_auto = valid_auto & o_at_drop
                if o_auto.any():
                    oinv_ref[0], active_del, sa = self._vectorized_deliver(
                        oinv_ref[0], new_req, active_del, o_auto, B, d)
                    score = score + sa
                    sk = (oinv_ref[0] < 0).to(torch.int8)
                    _, si = sk.sort(dim=1, stable=True)
                    oinv_ref[0] = oinv_ref[0].gather(1, si.long())

            # Auto-deliver this candidate at dropoff
            cand_at_drop = self._at_drop(cand_x, cand_y)
            auto_cand = valid_auto & cand_at_drop
            if auto_cand.any():
                cand_inv, active_del, sa = self._vectorized_deliver(
                    cand_inv, new_req, active_del, auto_cand, B, d)
                score = score + sa
                sk = (cand_inv < 0).to(torch.int8)
                _, si = sk.sort(dim=1, stable=True)
                cand_inv = cand_inv.gather(1, si.long())

            active_idx = new_aidx

        return cand_inv, active_idx, active_del, score, orders_comp

    @torch.no_grad()
    def _step_2bot(self, state, b1_acts, b1_items, b2_acts, b2_items,
                   round_num: int = -1):
        """Apply joint actions to 2-bot batch. Process all bots in real ID order.

        Returns new state dict with both candidate bots updated.
        """
        B = state['bot1_x'].shape[0]
        d = self.device

        bot1_x = state['bot1_x'].clone()
        bot1_y = state['bot1_y'].clone()
        bot1_inv = state['bot1_inv'].clone()
        bot2_x = state['bot2_x'].clone()
        bot2_y = state['bot2_y'].clone()
        bot2_inv = state['bot2_inv'].clone()
        active_idx = state['active_idx'].clone()
        active_del = state['active_del'].clone()
        score = state['score'].clone()
        orders_comp = state['orders_comp'].clone()
        locked_inv = state['locked_inv'].clone() if 'locked_inv' in state else None
        locked_bx = state['locked_bx'].clone() if 'locked_bx' in state else None
        locked_by = state['locked_by'].clone() if 'locked_by' in state else None

        spawn_x, spawn_y = self.ms.spawn

        # Process all bots in real ID order (lower ID moves first)
        for real_bid in range(self.num_bots):
            if real_bid == self.bot1_real_id:
                # === Bot 1 movement (collision with bot2 + locked) ===
                bot1_x, bot1_y = self._move_candidate(
                    b1_acts, bot1_x, bot1_y,
                    [bot2_x], [bot2_y],
                    locked_bx, locked_by,
                    B, d, spawn_x, spawn_y)

                # === Bot 1 pickup ===
                bot1_inv = self._pickup_candidate(
                    b1_acts, b1_items, bot1_x, bot1_y, bot1_inv, B, d)

                # === Bot 1 dropoff + order completion ===
                # Wrap bot2_inv in a mutable list so auto-delivery can update it
                bot2_inv_ref = [bot2_inv]
                bot1_inv, active_idx, active_del, score, orders_comp = \
                    self._dropoff_candidate(
                        b1_acts, bot1_x, bot1_y, bot1_inv,
                        active_idx, active_del, score, orders_comp,
                        [bot2_x], [bot2_y], [bot2_inv_ref],
                        locked_inv, locked_bx, locked_by, B, d)
                bot2_inv = bot2_inv_ref[0]

            elif real_bid == self.bot2_real_id:
                # === Bot 2 movement (collision with bot1's NEW pos + locked) ===
                bot2_x, bot2_y = self._move_candidate(
                    b2_acts, bot2_x, bot2_y,
                    [bot1_x], [bot1_y],
                    locked_bx, locked_by,
                    B, d, spawn_x, spawn_y)

                # === Bot 2 pickup ===
                bot2_inv = self._pickup_candidate(
                    b2_acts, b2_items, bot2_x, bot2_y, bot2_inv, B, d)

                # === Bot 2 dropoff + order completion ===
                bot1_inv_ref = [bot1_inv]
                bot2_inv, active_idx, active_del, score, orders_comp = \
                    self._dropoff_candidate(
                        b2_acts, bot2_x, bot2_y, bot2_inv,
                        active_idx, active_del, score, orders_comp,
                        [bot1_x], [bot1_y], [bot1_inv_ref],
                        locked_inv, locked_bx, locked_by, B, d)
                bot1_inv = bot1_inv_ref[0]

            elif real_bid in self.locked_idx_map:
                b = self.locked_idx_map[real_bid]
                lb_act = int(self.locked_actions[b, round_num])
                if lb_act == ACT_WAIT:
                    continue
                # Reuse parent's _step_locked_bot but with 2 candidate bots
                # We need a temp single-bot state for the locked bot processing
                # Use _step_locked_bot_2cand instead
                (bot1_x, bot1_y, bot1_inv, bot2_x, bot2_y, bot2_inv,
                 active_idx, active_del, score, orders_comp,
                 locked_inv, locked_bx, locked_by) = \
                    self._step_locked_bot_2cand(
                        b, round_num,
                        bot1_x, bot1_y, bot1_inv,
                        bot2_x, bot2_y, bot2_inv,
                        active_idx, active_del, score, orders_comp,
                        locked_inv, locked_bx, locked_by,
                        B, d, spawn_x, spawn_y)

        result = {
            'bot1_x': bot1_x, 'bot1_y': bot1_y, 'bot1_inv': bot1_inv,
            'bot2_x': bot2_x, 'bot2_y': bot2_y, 'bot2_inv': bot2_inv,
            'active_idx': active_idx, 'active_del': active_del,
            'score': score, 'orders_comp': orders_comp,
        }
        if locked_inv is not None:
            result['locked_inv'] = locked_inv
        if locked_bx is not None:
            result['locked_bx'] = locked_bx
            result['locked_by'] = locked_by
        return result

    def _step_locked_bot_2cand(self, b, round_num,
                                bot1_x, bot1_y, bot1_inv,
                                bot2_x, bot2_y, bot2_inv,
                                active_idx, active_del, score, orders_comp,
                                locked_inv, locked_bx, locked_by,
                                B, d, spawn_x, spawn_y):
        """Process locked bot action with 2 candidate bots for collision/auto-delivery."""
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
            # Collision with both candidates
            cand1_coll = ((lnx == bot1_x.to(torch.int32)) &
                          (lny == bot1_y.to(torch.int32)) & ~lb_at_spawn)
            cand2_coll = ((lnx == bot2_x.to(torch.int32)) &
                          (lny == bot2_y.to(torch.int32)) & ~lb_at_spawn)
            lb_can_move = lb_walk & ~cand1_coll & ~cand2_coll

            # Collision with other locked bots
            if self.num_locked > 1:
                all_lb_x = locked_bx.to(torch.int32)
                all_lb_y = locked_by.to(torch.int32)
                all_coll = ((lnx.unsqueeze(1) == all_lb_x) &
                            (lny.unsqueeze(1) == all_lb_y) &
                            ~lb_at_spawn.unsqueeze(1))
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
            lb_mdist = ((lbx.to(torch.int32) - item_x).abs() +
                        (lby.to(torch.int32) - item_y).abs())
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
            lb_at_drop = self._at_drop(lbx, lby)
            lb_can_drop = lb_at_drop & (active_idx < self.num_orders)

            aidx_l = active_idx.long().clamp(0, self.num_orders - 1)
            act_req_lb = self.order_req[aidx_l]

            linv_b = locked_inv[:, b, :]
            linv_b, active_del, score_add = self._vectorized_deliver(
                linv_b, act_req_lb, active_del, lb_can_drop, B, d)
            locked_inv[:, b, :] = linv_b
            score = score + score_add

            linv_b = locked_inv[:, b, :]
            sort_key = (linv_b < 0).to(torch.int8)
            _, sort_idx = sort_key.sort(dim=1, stable=True)
            locked_inv[:, b, :] = linv_b.gather(1, sort_idx.long())

            # Order completion
            act_req_lb = self.order_req[aidx_l]
            slot_done = (active_del == 1) | (act_req_lb < 0)
            has_required = (act_req_lb >= 0).any(dim=1)
            order_complete_lb = slot_done.all(dim=1) & has_required & lb_can_drop

            if order_complete_lb.any():
                orders_comp = orders_comp + order_complete_lb.to(torch.int32)
                score = score + order_complete_lb.to(torch.int32) * 5
                new_aidx = active_idx + order_complete_lb.to(torch.int32)
                oc_mask = order_complete_lb.unsqueeze(1).expand_as(active_del)
                active_del = torch.where(oc_mask, self._zero_i8, active_del)

                valid_auto = order_complete_lb & (new_aidx < self.num_orders)
                new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)
                new_req = self.order_req[new_aidx_l]

                # Auto-deliver all locked bots
                for lb2 in range(self.num_locked):
                    lb2_at = self._at_drop(locked_bx[:, lb2], locked_by[:, lb2])
                    lb2_auto = valid_auto & lb2_at
                    if lb2_auto.any():
                        li = locked_inv[:, lb2, :]
                        li, active_del, sa = self._vectorized_deliver(
                            li, new_req, active_del, lb2_auto, B, d)
                        locked_inv[:, lb2, :] = li
                        score = score + sa

                # Auto-deliver bot1
                b1_at = self._at_drop(bot1_x, bot1_y)
                b1_auto = valid_auto & b1_at
                if b1_auto.any():
                    bot1_inv, active_del, sa = self._vectorized_deliver(
                        bot1_inv, new_req, active_del, b1_auto, B, d)
                    score = score + sa
                    sk = (bot1_inv < 0).to(torch.int8)
                    _, si = sk.sort(dim=1, stable=True)
                    bot1_inv = bot1_inv.gather(1, si.long())

                # Auto-deliver bot2
                b2_at = self._at_drop(bot2_x, bot2_y)
                b2_auto = valid_auto & b2_at
                if b2_auto.any():
                    bot2_inv, active_del, sa = self._vectorized_deliver(
                        bot2_inv, new_req, active_del, b2_auto, B, d)
                    score = score + sa
                    sk = (bot2_inv < 0).to(torch.int8)
                    _, si = sk.sort(dim=1, stable=True)
                    bot2_inv = bot2_inv.gather(1, si.long())

                active_idx = new_aidx

        return (bot1_x, bot1_y, bot1_inv, bot2_x, bot2_y, bot2_inv,
                active_idx, active_del, score, orders_comp,
                locked_inv, locked_bx, locked_by)

    @torch.no_grad()
    def _eval_2bot(self, state: dict[str, torch.Tensor],
                   round_num: int = 0) -> torch.Tensor:
        """Evaluate 2-bot states for beam pruning. Returns [B] float32.

        Joint evaluation: considers both bots' inventories, positions, and
        their coordination (anti-redundancy, work division).
        """
        B = state['bot1_x'].shape[0]
        d = self.device

        bot1_x = state['bot1_x'].long()
        bot1_y = state['bot1_y'].long()
        inv1 = state['bot1_inv']
        bot2_x = state['bot2_x'].long()
        bot2_y = state['bot2_y'].long()
        inv2 = state['bot2_inv']
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)

        rounds_left = self.num_rounds - round_num - 1

        # Score is dominant
        ev = state['score'].float() * 100000

        if self.speed_bonus > 0:
            ev = ev + self.speed_bonus * state['score'].float() * rounds_left

        # Distance to dropoff for each bot
        dist_drop1 = self.dist_to_dropoff[bot1_y, bot1_x].float()
        dist_drop2 = self.dist_to_dropoff[bot2_y, bot2_x].float()

        # Active order analysis
        act_req = self.order_req[aidx]
        act_del = state['active_del']
        active_needed = (act_req >= 0) & (act_del == 0)
        active_done = (act_req >= 0) & (act_del == 1)
        active_remaining = active_needed.sum(dim=1).float()
        active_delivered = active_done.sum(dim=1).float()
        active_total = (act_req >= 0).sum(dim=1).float()
        fraction_done = active_delivered / active_total.clamp(min=1)

        ev = ev + fraction_done * 30000
        ev = ev + (active_remaining == 0).float() * 50000
        ev = ev + (active_remaining == 1).float() * 20000
        ev = ev + (active_remaining == 2).float() * 5000

        # === Inventory matching for BOTH bots (joint) ===
        req_exp = act_req.unsqueeze(1)      # [B, 1, 6]
        del_exp = act_del.unsqueeze(1)      # [B, 1, 6]

        # Bot1 active matching
        inv1_exp = inv1.unsqueeze(2)        # [B, 3, 1]
        has1 = (inv1_exp >= 0)
        match1 = (inv1_exp == req_exp) & (del_exp == 0) & (req_exp >= 0) & has1
        inv1_matches = match1.any(dim=2)    # [B, 3]
        has_active1 = inv1_matches.any(dim=1)
        n_active1 = inv1_matches.sum(dim=1).float()

        # Bot2 active matching
        inv2_exp = inv2.unsqueeze(2)
        has2 = (inv2_exp >= 0)
        match2 = (inv2_exp == req_exp) & (del_exp == 0) & (req_exp >= 0) & has2
        inv2_matches = match2.any(dim=2)
        has_active2 = inv2_matches.any(dim=1)
        n_active2 = inv2_matches.sum(dim=1).float()

        # Active inventory value per bot (discounted by distance)
        active_val1 = (70000 - dist_drop1 * 1750).clamp(min=0)
        active_val2 = (70000 - dist_drop2 * 1750).clamp(min=0)
        ev = ev + n_active1 * active_val1 + n_active2 * active_val2

        # Preview order matching for both bots
        pidx = (aidx + 1).clamp(0, self.num_orders - 1)
        prev_req = self.order_req[pidx]
        prev_req_exp = prev_req.unsqueeze(1)

        match_prev1 = ((inv1_exp == prev_req_exp) & (prev_req_exp >= 0) & has1).any(dim=2) & ~inv1_matches
        match_prev2 = ((inv2_exp == prev_req_exp) & (prev_req_exp >= 0) & has2).any(dim=2) & ~inv2_matches

        # Preview items valued progressively as active order completes
        frac_exp1 = fraction_done.unsqueeze(1).expand_as(match_prev1)
        prev_val = 5000.0 + 10000.0 * frac_exp1
        ev = ev + (match_prev1.float() * prev_val).sum(dim=1)
        ev = ev + (match_prev2.float() * prev_val).sum(dim=1)

        # Dead inventory for both bots
        has1_flat = (inv1 >= 0)
        is_dead1 = has1_flat & ~inv1_matches & ~match_prev1
        has2_flat = (inv2 >= 0)
        is_dead2 = has2_flat & ~inv2_matches & ~match_prev2

        # Check orders +2, +3 before declaring dead
        for _extra in [2, 3]:
            _eidx = (aidx + _extra).clamp(0, self.num_orders - 1)
            _ereq = self.order_req[_eidx].unsqueeze(1)
            if is_dead1.any():
                _m1 = ((inv1_exp == _ereq) & (_ereq >= 0) & has1).any(dim=2) & ~inv1_matches & ~match_prev1
                is_dead1 = is_dead1 & ~_m1
            if is_dead2.any():
                _m2 = ((inv2_exp == _ereq) & (_ereq >= 0) & has2).any(dim=2) & ~inv2_matches & ~match_prev2
                is_dead2 = is_dead2 & ~_m2

        dead_penalty = 50000 * min(1.0, rounds_left / 150.0) + 5000
        ev = ev - (is_dead1.sum(dim=1).float() + is_dead2.sum(dim=1).float()) * dead_penalty

        # === Joint coordination: reward DIFFERENT targets, penalize redundancy ===
        # Count order slots covered by each bot's inventory
        # match1_grid: [B, 3, 6], match2_grid: [B, 3, 6]
        # A slot is "jointly covered" if EITHER bot covers it
        slot_covered1 = match1.any(dim=1)  # [B, 6] — any inv1 slot matches this order slot
        slot_covered2 = match2.any(dim=1)  # [B, 6]
        both_cover = active_needed & slot_covered1 & slot_covered2  # redundancy
        unique_cover = active_needed & (slot_covered1 ^ slot_covered2)  # unique

        ev = ev + unique_cover.sum(dim=1).float() * 35000   # reward unique coverage
        ev = ev - both_cover.sum(dim=1).float() * 15000      # penalize redundancy

        # === Distance guidance for both bots ===
        for bot_x, bot_y, has_active_inv, dist_drop in [
            (bot1_x, bot1_y, has_active1, dist_drop1),
            (bot2_x, bot2_y, has_active2, dist_drop2),
        ]:
            inv_count_b = 0  # simplified; check space via has_space below
            has_space_b = True  # will compute per-bot below

            best_dist = torch.full((B,), 9999.0, device=d)
            for os in range(MAX_ORDER_SIZE):
                needed = active_needed[:, os]
                if not needed.any():
                    continue
                nt = act_req[:, os].long().clamp(0, self.num_types - 1)
                dt = self.dist_to_type[nt, bot_y, bot_x].float()
                better = needed & (dt < best_dist)
                best_dist = torch.where(better, dt, best_dist)

            close = best_dist < 9999
            trip_ok = close & ((best_dist + dist_drop + 2) <= rounds_left)
            _m = (trip_ok & ~has_active_inv).float()
            ev = ev + _m * (20000.0 - best_dist * 800)

            # Delivery urgency
            ev = ev - has_active_inv.float() * dist_drop * 200

        # Trip table for remaining order cost
        del_bits = (act_del.long() * self._del_weights).sum(dim=1)
        # Use bot that's closer to items for trip evaluation
        min_dist_drop = torch.min(dist_drop1, dist_drop2)
        # Use bot1's position for trip cost (heuristic — either bot could do it)
        cell_idx = self.trip_cell_idx_map[bot1_y, bot1_x]
        trip_combo = self.remaining_trip_combo[aidx, del_bits]
        has_trip = (trip_combo >= 0) & (cell_idx >= 0)
        safe_cell = cell_idx.clamp(0, self._trip_n_cells - 1)
        safe_combo = trip_combo.clamp(0, self._trip_n_combos - 1)
        next_cost = self.trip_cost_gpu[safe_cell, safe_combo].float()
        after_cost = self.remaining_after_cost[aidx, del_bits].float()
        total_remaining = (has_trip.float() * (next_cost + after_cost) +
                           (~has_trip).float() * 9999.0)
        achievable = has_trip & (total_remaining <= rounds_left)
        ev = ev + achievable.float() * 8000
        ev = ev - has_trip.float() * total_remaining.clamp(0, 300) * 50

        # Penalize both bots camping at dropoff without active items
        at_drop1 = self._at_drop(state['bot1_x'].long(), state['bot1_y'].long())
        at_drop2 = self._at_drop(state['bot2_x'].long(), state['bot2_y'].long())
        camp1 = at_drop1 & ~has_active1
        camp2 = at_drop2 & ~has_active2
        ev = ev - camp1.float() * 12000
        ev = ev - camp2.float() * 12000

        # Penalize bots being too close to each other (congestion)
        bot_dist = ((bot1_x - bot2_x).abs() + (bot1_y - bot2_y).abs()).float()
        too_close = bot_dist <= 1
        ev = ev - too_close.float() * 3000

        # Aisle congestion with locked bots
        if self.num_locked > 0 and 'locked_bx' in state and self._aisle_columns.any():
            for bx, by in [(bot1_x, bot1_y), (bot2_x, bot2_y)]:
                _in_aisle = self._aisle_columns[bx]
                _not_corr = ~self._corridor_rows[by]
                _in_narrow = _in_aisle & _not_corr
                if _in_narrow.any():
                    _lbx = state['locked_bx']
                    _lby = state['locked_by']
                    _same = (_lbx == bx.unsqueeze(1))
                    _lby_l = _lby.long().clamp(0, self.H - 1)
                    _lnc = ~self._corridor_rows[_lby_l]
                    _lsa = _same & _lnc
                    _nls = _lsa.sum(dim=1).float()
                    ev = ev - _in_narrow.float() * _nls * 4000

        return ev

    @torch.no_grad()
    def dp_search_2bot(self, game_state: GameState | None,
                       max_states: int = 100000,
                       verbose: bool = True,
                       on_round=None,
                       bot_ids: tuple[int, int] = (0, 1),
                       init_state: dict[str, torch.Tensor] | None = None,
                       ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
        """Joint 2-bot DP search via exhaustive BFS + dedup.

        Returns (best_score, bot1_actions, bot2_actions) where each is
        list of (action_type, item_idx) per round.
        """
        t0 = time.time()
        self.bot1_real_id = bot_ids[0]
        self.bot2_real_id = bot_ids[1]
        d = self.device

        if init_state is not None:
            state = init_state
        else:
            state = self._from_game_state_2bot(
                game_state, bot1_id=bot_ids[0], bot2_id=bot_ids[1])

        N = self.dp_num_actions  # per bot

        # History for backtracking (on GPU)
        parent_idx_history = []
        b1_act_history = []
        b1_item_history = []
        b2_act_history = []
        b2_item_history = []

        pruned_rounds = 0

        for rnd in range(self.num_rounds):
            B = state['bot1_x'].shape[0]

            # Sparse expand: distance-adaptive, returns only valid combos
            result = self._dp_expand_2bot_sparse(state, round_num=rnd)
            filt_exp, filt_b1a, filt_b1i, filt_b2a, filt_b2i, parent_idx_raw = result

            if filt_exp is None:
                # No valid actions — just keep current state with wait
                parent_idx_history.append(torch.zeros(B, dtype=torch.long, device=d))
                b1_act_history.append(torch.zeros(B, dtype=torch.int8, device=d))
                b1_item_history.append(torch.full((B,), -1, dtype=torch.int16, device=d))
                b2_act_history.append(torch.zeros(B, dtype=torch.int8, device=d))
                b2_item_history.append(torch.full((B,), -1, dtype=torch.int16, device=d))
                continue

            BF = filt_b1a.shape[0]

            # Step both bots
            new_state = self._step_2bot(
                filt_exp, filt_b1a, filt_b1i, filt_b2a, filt_b2i,
                round_num=rnd)

            # No-op detection: both bots unchanged
            no_change = (
                (new_state['bot1_x'] == filt_exp['bot1_x']) &
                (new_state['bot1_y'] == filt_exp['bot1_y']) &
                (new_state['bot2_x'] == filt_exp['bot2_x']) &
                (new_state['bot2_y'] == filt_exp['bot2_y']) &
                ((new_state['bot1_inv'] == filt_exp['bot1_inv']).all(dim=1)) &
                ((new_state['bot2_inv'] == filt_exp['bot2_inv']).all(dim=1)) &
                (new_state['score'] == filt_exp['score']) &
                (new_state['active_idx'] == filt_exp['active_idx'])
            )
            is_noop = no_change & ((filt_b1a != ACT_WAIT) | (filt_b2a != ACT_WAIT))

            # Hash + dedup
            hashes = self._hash_2bot(new_state)
            hashes[is_noop] = -1

            # Sort by score desc then hash asc for dedup
            _, score_order = new_state['score'].sort(descending=True, stable=True)
            hashes_scored = hashes[score_order]
            sorted_h, hash_order = hashes_scored.sort(stable=True)
            sort_idx = score_order[hash_order]
            is_first = torch.ones(BF, dtype=torch.bool, device=d)
            is_first[1:] = sorted_h[1:] != sorted_h[:-1]
            valid_mask = is_first & (sorted_h != -1)

            unique_idx = sort_idx[valid_mask]
            B_new = unique_idx.shape[0]

            # Parent tracking (sparse: parent_idx_raw already has state indices)
            parent_idx = parent_idx_raw[unique_idx]
            parent_idx_history.append(parent_idx)
            b1_act_history.append(filt_b1a[unique_idx])
            b1_item_history.append(filt_b1i[unique_idx])
            b2_act_history.append(filt_b2a[unique_idx])
            b2_item_history.append(filt_b2i[unique_idx])

            # Gather unique states
            state = {k: v[unique_idx] for k, v in new_state.items()}

            # Prune if too many states
            if B_new > max_states:
                pruned_rounds += 1
                evals = self._eval_2bot(state, round_num=rnd)
                keep = min(max_states, B_new)
                _, topk = torch.topk(evals, keep)
                topk_sorted, _ = topk.sort()
                state = {k: v[topk_sorted] for k, v in state.items()}
                parent_idx_history[-1] = parent_idx_history[-1][topk_sorted]
                b1_act_history[-1] = b1_act_history[-1][topk_sorted]
                b1_item_history[-1] = b1_item_history[-1][topk_sorted]
                b2_act_history[-1] = b2_act_history[-1][topk_sorted]
                b2_item_history[-1] = b2_item_history[-1][topk_sorted]
                B_new = keep

            # Verbose output
            if verbose and (rnd < 10 or rnd % 25 == 0 or rnd == self.num_rounds - 1):
                dt = time.time() - t0
                best_score = state['score'].max().item()
                print(f"  R{rnd:3d}: score={best_score:3d}, "
                      f"unique={B_new}, expanded={BF}, "
                      f"t={dt:.1f}s", file=sys.stderr)
                if on_round:
                    on_round(rnd, best_score, B_new, BF, dt)

        # Find best and backtrack
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Transfer history to CPU
        parent_idx_history = [h.cpu() for h in parent_idx_history]
        b1_act_history = [h.cpu() for h in b1_act_history]
        b1_item_history = [h.cpu() for h in b1_item_history]
        b2_act_history = [h.cpu() for h in b2_act_history]
        b2_item_history = [h.cpu() for h in b2_item_history]

        bot1_seq = []
        bot2_seq = []
        idx = best_idx
        for j in range(self.num_rounds - 1, -1, -1):
            bot1_seq.append((int(b1_act_history[j][idx]),
                             int(b1_item_history[j][idx])))
            bot2_seq.append((int(b2_act_history[j][idx]),
                             int(b2_item_history[j][idx])))
            idx = int(parent_idx_history[j][idx])
        bot1_seq.reverse()
        bot2_seq.reverse()

        total_time = time.time() - t0
        if verbose:
            print(f"\n2-Bot GPU DP: score={best_score}, time={total_time:.1f}s, "
                  f"pruned={pruned_rounds}", file=sys.stderr)

        return best_score, bot1_seq, bot2_seq

    @torch.no_grad()
    def verify_2bot_against_cpu(self, game_state: GameState,
                                 all_orders: list[Order],
                                 bot1_id: int = 0, bot2_id: int = 1,
                                 num_rounds: int = 50) -> bool:
        """Verify 2-bot GPU step matches CPU step with greedy actions."""
        from pathfinding import precompute_all_distances, get_first_step, get_distance
        ms = game_state.map_state
        dist_maps = precompute_all_distances(ms)
        num_bots = len(game_state.bot_positions)

        cpu_state = game_state.copy()
        cpu_log = []

        def greedy_action(gs, bid):
            bx = int(gs.bot_positions[bid, 0])
            by = int(gs.bot_positions[bid, 1])
            inv_count = gs.bot_inv_count(bid)
            inv_items = gs.bot_inv_list(bid)
            active = gs.get_active_order()
            if not active:
                return (ACT_WAIT, -1)
            if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
                    inv_count > 0 and any(active.needs_type(t) for t in inv_items)):
                return (ACT_DROPOFF, -1)
            if inv_count < INV_CAP:
                for iidx in range(ms.num_items):
                    ix = int(ms.item_positions[iidx, 0])
                    iy = int(ms.item_positions[iidx, 1])
                    if abs(bx - ix) + abs(by - iy) == 1:
                        tid = int(ms.item_types[iidx])
                        if active.needs_type(tid):
                            return (ACT_PICKUP, iidx)
            if inv_count > 0 and any(active.needs_type(t) for t in inv_items):
                act = get_first_step(dist_maps, (bx, by), ms.drop_off)
                if act > 0:
                    return (act, -1)
            if inv_count < INV_CAP:
                best_d = 9999
                best_a = None
                for iidx in range(ms.num_items):
                    tid = int(ms.item_types[iidx])
                    if not active.needs_type(tid):
                        continue
                    for ax, ay in ms.item_adjacencies.get(iidx, []):
                        dd = get_distance(dist_maps, (bx, by), (ax, ay))
                        if dd < best_d:
                            act = get_first_step(dist_maps, (bx, by), (ax, ay))
                            if act > 0:
                                best_a = (act, -1)
                                best_d = dd
                if best_a:
                    return best_a
            if inv_count > 0:
                act = get_first_step(dist_maps, (bx, by), ms.drop_off)
                if act > 0:
                    return (act, -1)
            return (ACT_WAIT, -1)

        for rnd in range(num_rounds):
            a1 = greedy_action(cpu_state, bot1_id)
            a2 = greedy_action(cpu_state, bot2_id)
            cpu_log.append((a1, a2))
            round_acts = [(ACT_WAIT, -1)] * num_bots
            round_acts[bot1_id] = a1
            round_acts[bot2_id] = a2
            cpu_state.round = rnd
            cpu_step(cpu_state, round_acts, all_orders)

        # Replay on GPU
        gpu_state = self._from_game_state_2bot(game_state, bot1_id, bot2_id)
        for rnd in range(num_rounds):
            a1, a2 = cpu_log[rnd]
            b1a = torch.tensor([a1[0]], dtype=torch.int8, device=self.device)
            b1i = torch.tensor([a1[1] if a1[1] >= 0 else 0], dtype=torch.int16, device=self.device)
            b2a = torch.tensor([a2[0]], dtype=torch.int8, device=self.device)
            b2i = torch.tensor([a2[1] if a2[1] >= 0 else 0], dtype=torch.int16, device=self.device)
            gpu_state = self._step_2bot(gpu_state, b1a, b1i, b2a, b2i, round_num=rnd)

        gs = gpu_state['score'][0].item()
        cs = cpu_state.score
        print(f"  2-Bot verify ({num_rounds} rounds): CPU={cs}, GPU={gs}",
              file=sys.stderr)
        if gs != cs:
            # Detailed replay to find divergence
            gpu2 = self._from_game_state_2bot(game_state, bot1_id, bot2_id)
            cpu2 = game_state.copy()
            for rnd in range(num_rounds):
                a1, a2 = cpu_log[rnd]
                round_acts = [(ACT_WAIT, -1)] * num_bots
                round_acts[bot1_id] = a1
                round_acts[bot2_id] = a2
                cpu2.round = rnd
                cpu_step(cpu2, round_acts, all_orders)

                b1a = torch.tensor([a1[0]], dtype=torch.int8, device=self.device)
                b1i = torch.tensor([a1[1] if a1[1] >= 0 else 0], dtype=torch.int16, device=self.device)
                b2a = torch.tensor([a2[0]], dtype=torch.int8, device=self.device)
                b2i = torch.tensor([a2[1] if a2[1] >= 0 else 0], dtype=torch.int16, device=self.device)
                gpu2 = self._step_2bot(gpu2, b1a, b1i, b2a, b2i, round_num=rnd)

                g_s = gpu2['score'][0].item()
                c_s = cpu2.score
                if g_s != c_s:
                    print(f"    DIVERGENCE at round {rnd}!", file=sys.stderr)
                    print(f"      Bot1 act={a1}, Bot2 act={a2}", file=sys.stderr)
                    print(f"      CPU: score={c_s}", file=sys.stderr)
                    print(f"      GPU: score={g_s}", file=sys.stderr)
                    return False
        print(f"    MATCH! Score={gs}", file=sys.stderr)
        return True


def gpu_dp_search(difficulty: str | None = None, seed: int | None = None,
                  max_states: int = 500000, verbose: bool = True,
                  game_factory: Callable | None = None,
                  capture_data: dict | None = None) -> tuple[int, list[tuple[int, int]]]:
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


def gpu_beam_search(difficulty: str | None = None, seed: int | None = None,
                    beam_width: int = 10000, verbose: bool = True,
                    game_factory: Callable | None = None,
                    capture_data: dict | None = None) -> tuple[int, list[tuple[int, int]]]:
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
                        choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
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
        print(f"Solution saved to DB ({args.difficulty}, score={score})")

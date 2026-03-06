"""N-bot joint DP solver — the core breakthrough for B200.

Instead of planning bots sequentially (each locked to others' fixed trajectories),
this plans 2-5 bots JOINTLY in a single search. The state space is the Cartesian
product of all candidate bots' states, enabling true coordination.

Key innovation: distance-adaptive action expansion. Bots far apart get 1 action
(greedy best), bots nearby get 5-10 actions. This keeps the effective branching
factor at ~20-40 combos/state instead of the naive 7^N.

B200 targets:
  2-bot: 10M states (37GB)
  3-bot: 5-10M states (18-70GB)
  4-bot: 2M states (60GB)
  5-bot: 1M states (30GB) — all of Hard in ONE search
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np
import torch

import _shared  # noqa: F401
from game_engine import (
    GameState, MapState, Order,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
)
from gpu_beam_search import GPUBeamSearcher
from precompute import PrecomputedTables
from b200_beam_search import CPUHistory


# Mixing primes for N-bot hash
_HASH_PRIMES = [
    1,
    6364136223846793005,
    1442695040888963407,
    3141592653589793238,
    2718281828459045235,
]


class JointBeamSearcher:
    """N-bot joint DP solver.

    Searches over joint states of N candidate bots simultaneously, with
    locked bots (outside the squad) following fixed trajectories.

    State per candidate bot: (x, y, inv[3]) — 8 bytes
    Shared state: (active_idx, active_del[6], score, orders_comp) — ~12 bytes
    Locked bots: (inv[L,3], bx[L], by[L]) — tracked for collision + delivery

    Total per-state: ~53 bytes (same scale as single-bot with locked bots).
    """

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 n_candidates: int, candidate_bot_ids: tuple[int, ...],
                 num_bots: int, device: str = 'cuda',
                 locked_trajectories: dict[str, np.ndarray] | None = None,
                 no_compile: bool = False,
                 speed_bonus: float = 100.0,
                 chunk_size: int = 500_000):
        self.device = device
        self.ms = map_state
        self.all_orders = all_orders
        self.n_candidates = n_candidates
        self.candidate_bot_ids = candidate_bot_ids
        self.num_bots = num_bots
        self.num_items = map_state.num_items
        self.num_orders = len(all_orders)
        self.num_types = map_state.num_types
        self.W = map_state.width
        self.H = map_state.height
        self.drop_x = map_state.drop_off[0]
        self.drop_y = map_state.drop_off[1]
        self.speed_bonus = speed_bonus
        self.chunk_size = chunk_size

        # Enable TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Upload static map data
        self.grid = torch.tensor(map_state.grid, dtype=torch.int8, device=device)
        self.item_pos_x = torch.tensor(
            map_state.item_positions[:, 0], dtype=torch.int16, device=device)
        self.item_pos_y = torch.tensor(
            map_state.item_positions[:, 1], dtype=torch.int16, device=device)
        self.item_types = torch.tensor(
            map_state.item_types, dtype=torch.int8, device=device)

        # Walkable mask
        self.walkable = (
            (self.grid == CELL_FLOOR) | (self.grid == CELL_DROPOFF)
        ).contiguous()

        # Valid moves per cell [H, W, 4]
        valid_moves = torch.zeros(self.H, self.W, 4, dtype=torch.bool, device=device)
        valid_moves[1:, :, 0] = self.walkable[:-1, :]
        valid_moves[:-1, :, 1] = self.walkable[1:, :]
        valid_moves[:, 1:, 2] = self.walkable[:, :-1]
        valid_moves[:, :-1, 3] = self.walkable[:, 1:]
        self.valid_moves = valid_moves

        # Pack orders
        order_req = np.full((self.num_orders, MAX_ORDER_SIZE), -1, dtype=np.int8)
        for i, o in enumerate(all_orders):
            for j in range(len(o.required)):
                order_req[i, j] = int(o.required[j])
        self.order_req = torch.tensor(order_req, dtype=torch.int8, device=device)
        order_sizes = np.array([len(o.required) for o in all_orders], dtype=np.int32)
        self.order_sizes = torch.tensor(order_sizes, dtype=torch.int32, device=device)

        # Precomputed distances
        tables = PrecomputedTables.get(map_state)
        gpu_tables = tables.to_gpu_tensors(device)
        self.dist_to_dropoff = gpu_tables['dist_to_dropoff']
        self.dist_to_type = gpu_tables['dist_to_type']
        self.first_step_to_dropoff = gpu_tables['step_to_dropoff']
        self.first_step_to_type = gpu_tables['step_to_type']

        # Item adjacency lookup [H, W, MAX_ADJ]
        MAX_ADJ = 4
        self.MAX_ADJ = MAX_ADJ
        adj_items = torch.full((self.H, self.W, MAX_ADJ), -1, dtype=torch.int16, device=device)
        adj_count = torch.zeros((self.H, self.W), dtype=torch.int8, device=device)
        for item_idx in range(self.num_items):
            ix = int(map_state.item_positions[item_idx, 0])
            iy = int(map_state.item_positions[item_idx, 1])
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx_, ny_ = ix + ddx, iy + ddy
                if 0 <= nx_ < self.W and 0 <= ny_ < self.H:
                    if bool(self.walkable[ny_, nx_]):
                        c = int(adj_count[ny_, nx_])
                        if c < MAX_ADJ:
                            adj_items[ny_, nx_, c] = item_idx
                            adj_count[ny_, nx_] = c + 1
        self.adj_items = adj_items
        self.adj_count = adj_count

        # Direction tables
        self.DX = torch.tensor([0, 0, 0, -1, 1, 0, 0], dtype=torch.int32, device=device)
        self.DY = torch.tensor([0, -1, 1, 0, 0, 0, 0], dtype=torch.int32, device=device)

        # Constants
        self._neg1_i8 = torch.tensor(-1, dtype=torch.int8, device=device)
        self._zero_i8 = torch.tensor(0, dtype=torch.int8, device=device)
        self._one_i8 = torch.tensor(1, dtype=torch.int8, device=device)
        self._hash_shifts = torch.arange(
            32, 32 + MAX_ORDER_SIZE, dtype=torch.int64, device=device)
        self._del_weights = torch.tensor(
            [1, 2, 4, 8, 16, 32], dtype=torch.long, device=device)

        # Locked trajectories
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
            self.locked_bot_ids = locked_trajectories.get(
                'locked_bot_ids', list(range(self.num_locked)))
            self.locked_idx_map = {
                real_id: idx for idx, real_id in enumerate(self.locked_bot_ids)}
        else:
            self.num_locked = 0
            self.locked_bot_ids = []
            self.locked_idx_map = {}

        # Aisle detection (for congestion penalty)
        grid_np = map_state.grid
        _aisle_columns = np.zeros(self.W, dtype=bool)
        _corridor_rows = np.zeros(self.H, dtype=bool)
        for y in range(self.H):
            floor_count = np.sum((grid_np[y] == CELL_FLOOR) | (grid_np[y] == CELL_DROPOFF))
            if floor_count * 10 > self.W * 6:
                _corridor_rows[y] = True
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
        self._aisle_columns = torch.tensor(_aisle_columns, dtype=torch.bool, device=device)
        self._corridor_rows = torch.tensor(_corridor_rows, dtype=torch.bool, device=device)

        print(f"  JointBeamSearcher: {n_candidates} candidates {candidate_bot_ids}, "
              f"{self.num_locked} locked, {self.num_items} items, "
              f"{self.num_types} types, chunk={chunk_size:,}",
              file=sys.stderr)

    def _init_state(self, gs: GameState) -> dict[str, torch.Tensor]:
        """Create initial joint state from GameState (batch size 1)."""
        d = self.device
        state = {}

        # Per-candidate bot state
        for i, bid in enumerate(self.candidate_bot_ids):
            state[f'bot{i}_x'] = torch.tensor(
                [int(gs.bot_positions[bid, 0])], dtype=torch.int16, device=d)
            state[f'bot{i}_y'] = torch.tensor(
                [int(gs.bot_positions[bid, 1])], dtype=torch.int16, device=d)
            state[f'bot{i}_inv'] = torch.tensor(
                [[int(gs.bot_inventories[bid, s]) for s in range(INV_CAP)]],
                dtype=torch.int8, device=d)

        # Shared order state
        state['active_idx'] = torch.tensor([0], dtype=torch.int32, device=d)
        state['active_del'] = torch.zeros((1, MAX_ORDER_SIZE), dtype=torch.int8, device=d)
        state['score'] = torch.tensor([0], dtype=torch.int32, device=d)
        state['orders_comp'] = torch.tensor([0], dtype=torch.int32, device=d)

        # Locked bot state
        if self.num_locked > 0:
            sx, sy = self.ms.spawn
            state['locked_inv'] = torch.full(
                (1, self.num_locked, INV_CAP), -1, dtype=torch.int8, device=d)
            state['locked_bx'] = torch.full(
                (1, self.num_locked), sx, dtype=torch.int16, device=d)
            state['locked_by'] = torch.full(
                (1, self.num_locked), sy, dtype=torch.int16, device=d)

        return state

    @torch.no_grad()
    def _hash_nbot(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Probabilistic 63-bit hash for joint state. Returns [B] int64."""
        B = state['bot0_x'].shape[0]
        d = self.device
        h = torch.zeros(B, dtype=torch.int64, device=d)

        for i in range(self.n_candidates):
            sorted_inv, _ = state[f'bot{i}_inv'].sort(dim=1)
            # Pack: x(5) | y(5) | inv0(5) | inv1(5) | inv2(5) = 25 bits
            bot_hash = (
                state[f'bot{i}_x'].long()
                | (state[f'bot{i}_y'].long() << 5)
                | ((sorted_inv[:, 0].long() + 1) << 10)
                | ((sorted_inv[:, 1].long() + 1) << 15)
                | ((sorted_inv[:, 2].long() + 1) << 20)
            )
            prime = _HASH_PRIMES[i % len(_HASH_PRIMES)]
            h = h ^ (bot_hash * prime)

        # Order state
        order_hash = state['active_idx'].long() << 25
        del_bits = state['active_del'].long()
        order_hash = order_hash | (del_bits << self._hash_shifts.unsqueeze(0)).sum(dim=1)
        h = h ^ order_hash

        return h & 0x7FFFFFFFFFFFFFFF

    def _per_bot_actions(self, state: dict[str, torch.Tensor], bot_idx: int,
                         round_num: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate valid actions for one candidate bot.

        Returns (actions[B, K], items[B, K], valid[B, K]) where K = 6 + MAX_ADJ.
        """
        B = state[f'bot{bot_idx}_x'].shape[0]
        d = self.device
        K = 6 + self.MAX_ADJ

        bx = state[f'bot{bot_idx}_x'].long()
        by = state[f'bot{bot_idx}_y'].long()

        # Adjacent items
        per_adj = self.adj_items[by, bx]   # [B, MAX_ADJ]
        per_count = self.adj_count[by, bx]  # [B]

        acts = torch.zeros(B, K, dtype=torch.int8, device=d)
        items = torch.full((B, K), -1, dtype=torch.int16, device=d)
        valid = torch.ones(B, K, dtype=torch.bool, device=d)

        # Fixed actions
        acts[:, 0] = ACT_WAIT
        acts[:, 1] = ACT_MOVE_UP
        acts[:, 2] = ACT_MOVE_DOWN
        acts[:, 3] = ACT_MOVE_LEFT
        acts[:, 4] = ACT_MOVE_RIGHT
        acts[:, 5] = ACT_DROPOFF

        # Variable pickups
        for j in range(self.MAX_ADJ):
            acts[:, 6 + j] = ACT_PICKUP
            items[:, 6 + j] = per_adj[:, j]
            valid[:, 6 + j] = per_adj[:, j] >= 0

        # Wall pruning
        valid[:, 1:5] &= self.valid_moves[by, bx]

        return acts, items, valid

    def _proxy_eval_per_bot(self, state: dict[str, torch.Tensor], bot_idx: int,
                            acts: torch.Tensor, items: torch.Tensor,
                            round_num: int) -> torch.Tensor:
        """Quick proxy score for each action of one bot. Returns [B, K] float32.

        Used to rank actions for distance-adaptive expansion.
        """
        B, K = acts.shape
        d = self.device

        bx = state[f'bot{bot_idx}_x'].long()
        by = state[f'bot{bot_idx}_y'].long()
        inv = state[f'bot{bot_idx}_inv']
        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]
        act_del = state['active_del']

        dist_drop = self.dist_to_dropoff[by, bx].float()

        # Check if bot has active items in inventory
        inv_exp = inv.unsqueeze(2)    # [B, 3, 1]
        req_exp = act_req.unsqueeze(1)  # [B, 1, 6]
        del_exp = act_del.unsqueeze(1)  # [B, 1, 6]
        match_active = (inv_exp == req_exp) & (del_exp == 0) & (req_exp >= 0) & (inv_exp >= 0)
        has_active_inv = match_active.any(dim=2).any(dim=1)  # [B]

        scores = torch.zeros(B, K, dtype=torch.float32, device=d)

        # Dropoff: high priority when carrying active items at dropoff
        at_drop = (bx == self.drop_x) & (by == self.drop_y)
        scores[:, 5] = (at_drop & has_active_inv).float() * 100

        # Move toward dropoff when carrying active items
        move_to_drop = self.first_step_to_dropoff[by, bx]  # [B] int8
        for k_idx in range(1, 5):
            act_val = k_idx  # 1=up, 2=down, 3=left, 4=right
            matches_drop = (move_to_drop == act_val)
            scores[:, k_idx] += (matches_drop & has_active_inv).float() * 50

        # Move toward needed items when has space
        inv_count = (inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP
        active_needed = (act_req >= 0) & (act_del == 0)

        for os_idx in range(MAX_ORDER_SIZE):
            needed = active_needed[:, os_idx]
            if not needed.any():
                continue
            ntype = act_req[:, os_idx].long().clamp(0, self.num_types - 1)
            move_to_type = self.first_step_to_type[ntype, by, bx]
            for k_idx in range(1, 5):
                matches = (move_to_type == k_idx) & needed & has_space
                scores[:, k_idx] += matches.float() * 30

        # Pickup: bonus for picking active/preview items
        for j in range(self.MAX_ADJ):
            item_idx_t = items[:, 6 + j].long().clamp(0, self.num_items - 1)
            pickup_valid = items[:, 6 + j] >= 0
            ptype = self.item_types[item_idx_t]
            for os_idx in range(MAX_ORDER_SIZE):
                type_match = (ptype == act_req[:, os_idx]) & (act_del[:, os_idx] == 0)
                scores[:, 6 + j] += (type_match & pickup_valid & has_space).float() * 40

        return scores

    @torch.no_grad()
    def _expand_nbot(self, state: dict[str, torch.Tensor],
                     round_num: int, max_combos_per_state: int = 50
                     ) -> tuple[dict, list[torch.Tensor], list[torch.Tensor]]:
        """Distance-adaptive N-bot action expansion.

        For each state, computes per-bot action budgets based on pairwise distances.
        Bots far apart get fewer actions (greedy best), nearby bots get more.

        Returns:
            (expanded_state, per_bot_actions[N][B*C], per_bot_items[N][B*C])
        """
        B = state['bot0_x'].shape[0]
        d = self.device
        N = self.n_candidates

        # Collect per-bot actions and proxy scores
        all_acts = []    # [N] of [B, K]
        all_items = []   # [N] of [B, K]
        all_valid = []   # [N] of [B, K]
        all_scores = []  # [N] of [B, K]

        for i in range(N):
            acts, items, valid = self._per_bot_actions(state, i, round_num)
            scores = self._proxy_eval_per_bot(state, i, acts, items, round_num)
            scores[~valid] = float('-inf')
            all_acts.append(acts)
            all_items.append(items)
            all_valid.append(valid)
            all_scores.append(scores)

        # Compute pairwise distances between candidates
        # For each bot pair, compute Manhattan distance
        K = all_acts[0].shape[1]  # actions per bot

        # Determine per-bot action budget based on min distance to any other candidate
        budgets = []
        for i in range(N):
            bx_i = state[f'bot{i}_x'].to(torch.int32)
            by_i = state[f'bot{i}_y'].to(torch.int32)
            min_dist = torch.full((B,), 999, dtype=torch.int32, device=d)
            for j in range(N):
                if i == j:
                    continue
                bx_j = state[f'bot{j}_x'].to(torch.int32)
                by_j = state[f'bot{j}_y'].to(torch.int32)
                dist = (bx_i - bx_j).abs() + (by_i - by_j).abs()
                min_dist = torch.minimum(min_dist, dist)

            # Budget: far=1, medium=3, close=5, adjacent=K (all)
            budget = torch.ones(B, dtype=torch.int32, device=d)
            budget = torch.where(min_dist <= 8, torch.full_like(budget, 3), budget)
            budget = torch.where(min_dist <= 3, torch.full_like(budget, 5), budget)
            budget = torch.where(min_dist <= 1, torch.full_like(budget, min(K, 10)), budget)
            budgets.append(budget)

        # For simplicity and GPU efficiency, use the MAX budget across states for each bot.
        # This over-expands some states but keeps tensor shapes uniform.
        max_budgets = [int(b.max().item()) for b in budgets]

        # Cap total combos
        total_combos = 1
        for mb in max_budgets:
            total_combos *= max(1, mb)
        while total_combos > max_combos_per_state and any(mb > 1 for mb in max_budgets):
            # Reduce largest budget
            largest_idx = max(range(N), key=lambda i: max_budgets[i])
            max_budgets[largest_idx] = max(1, max_budgets[largest_idx] - 1)
            total_combos = 1
            for mb in max_budgets:
                total_combos *= max(1, mb)

        # Select top-K actions per bot (by proxy score)
        selected_acts = []   # [N] of [B, budget_i]
        selected_items = []  # [N] of [B, budget_i]

        for i in range(N):
            budget_i = max_budgets[i]
            if budget_i >= K:
                selected_acts.append(all_acts[i])
                selected_items.append(all_items[i])
            else:
                _, topk_idx = torch.topk(all_scores[i], budget_i, dim=1)
                sel_a = all_acts[i].gather(1, topk_idx)
                sel_it = all_items[i].gather(1, topk_idx.to(torch.int16).long())
                selected_acts.append(sel_a)
                selected_items.append(sel_it)

        # Cartesian product via iterative cross
        # Start with bot 0's actions, then cross with bot 1, etc.
        C = total_combos
        BC = B * C

        # Build expanded state and per-bot action arrays
        # Start by repeating state for total_combos per original state
        expanded = {}
        for k, v in state.items():
            if v.dim() == 1:
                expanded[k] = v.repeat_interleave(C, output_size=BC)
            else:
                expanded[k] = v.repeat_interleave(C, dim=0, output_size=BC)

        # Build per-bot action tensors via Cartesian product indexing
        per_bot_acts = []
        per_bot_items = []

        # Compute strides for indexing into Cartesian product
        strides = []
        stride = 1
        for i in range(N - 1, -1, -1):
            strides.append(stride)
            stride *= max_budgets[i]
        strides.reverse()

        for i in range(N):
            budget_i = max_budgets[i]
            # For each of the B*C expanded states, compute which action index for bot i
            # combo_idx goes 0..C-1 for each original state
            combo_idx = torch.arange(C, device=d).repeat(B)  # [B*C]
            bot_action_idx = (combo_idx // strides[i]) % budget_i  # [B*C]

            # Original state index
            state_idx = torch.arange(B, device=d).repeat_interleave(C, output_size=BC)

            # Gather actions
            acts_flat = selected_acts[i][state_idx, bot_action_idx]
            items_flat = selected_items[i][state_idx, bot_action_idx]
            per_bot_acts.append(acts_flat)
            per_bot_items.append(items_flat)

        return expanded, per_bot_acts, per_bot_items

    @torch.no_grad()
    def _step_nbot(self, state: dict[str, torch.Tensor],
                   per_bot_acts: list[torch.Tensor],
                   per_bot_items: list[torch.Tensor],
                   round_num: int) -> dict[str, torch.Tensor]:
        """Apply joint actions to all bots in real ID order.

        Processes all bots (candidates + locked) in ascending real bot ID order,
        matching the CPU game engine's collision resolution exactly.
        """
        B = state['bot0_x'].shape[0]
        d = self.device
        spawn_x, spawn_y = self.ms.spawn

        # Clone mutable state
        bot_xs = {i: state[f'bot{i}_x'].clone() for i in range(self.n_candidates)}
        bot_ys = {i: state[f'bot{i}_y'].clone() for i in range(self.n_candidates)}
        bot_invs = {i: state[f'bot{i}_inv'].clone() for i in range(self.n_candidates)}
        active_idx = state['active_idx'].clone()
        active_del = state['active_del'].clone()
        score = state['score'].clone()
        orders_comp = state['orders_comp'].clone()

        locked_inv = state['locked_inv'].clone() if 'locked_inv' in state else None
        locked_bx = state['locked_bx'].clone() if 'locked_bx' in state else None
        locked_by = state['locked_by'].clone() if 'locked_by' in state else None

        # Map candidate index -> real bot ID
        cand_id_map = {bid: i for i, bid in enumerate(self.candidate_bot_ids)}

        # Process all bots in real ID order
        for real_bid in range(self.num_bots):
            if real_bid in cand_id_map:
                ci = cand_id_map[real_bid]
                self._step_one_candidate(
                    ci, per_bot_acts[ci], per_bot_items[ci],
                    bot_xs, bot_ys, bot_invs,
                    active_idx, active_del, score, orders_comp,
                    locked_inv, locked_bx, locked_by,
                    B, d, spawn_x, spawn_y)

            elif real_bid in self.locked_idx_map:
                li = self.locked_idx_map[real_bid]
                lb_act = int(self.locked_actions[li, round_num])
                if lb_act == ACT_WAIT:
                    continue
                self._step_one_locked(
                    li, round_num,
                    bot_xs, bot_ys, bot_invs,
                    active_idx, active_del, score, orders_comp,
                    locked_inv, locked_bx, locked_by,
                    B, d, spawn_x, spawn_y)

        # Build result
        result = {
            'active_idx': active_idx,
            'active_del': active_del,
            'score': score,
            'orders_comp': orders_comp,
        }
        for i in range(self.n_candidates):
            result[f'bot{i}_x'] = bot_xs[i]
            result[f'bot{i}_y'] = bot_ys[i]
            result[f'bot{i}_inv'] = bot_invs[i]
        if locked_inv is not None:
            result['locked_inv'] = locked_inv
            result['locked_bx'] = locked_bx
            result['locked_by'] = locked_by

        return result

    def _step_one_candidate(self, ci, actions, action_items,
                            bot_xs, bot_ys, bot_invs,
                            active_idx, active_del, score, orders_comp,
                            locked_inv, locked_bx, locked_by,
                            B, d, spawn_x, spawn_y):
        """Process one candidate bot's action with collision checks against all others."""
        bx = bot_xs[ci]
        by = bot_ys[ci]
        inv = bot_invs[ci]
        acts_long = actions.long()

        # === MOVEMENT ===
        is_move = (actions >= 1) & (actions <= 4)
        dx = self.DX[acts_long]
        dy = self.DY[acts_long]
        nx = bx.to(torch.int32) + dx
        ny = by.to(torch.int32) + dy

        in_bounds = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
        ny_safe = ny.clamp(0, self.H - 1)
        nx_safe = nx.clamp(0, self.W - 1)
        is_walkable = self.walkable[ny_safe.long(), nx_safe.long()]
        can_move = is_move & in_bounds & is_walkable

        at_spawn = (nx == spawn_x) & (ny == spawn_y)

        # Collision with other candidate bots
        for j in range(self.n_candidates):
            if j == ci:
                continue
            coll = ((nx == bot_xs[j].to(torch.int32)) &
                    (ny == bot_ys[j].to(torch.int32)) & ~at_spawn)
            can_move = can_move & ~coll

        # Collision with locked bots
        if locked_bx is not None:
            all_lb_x = locked_bx.to(torch.int32)
            all_lb_y = locked_by.to(torch.int32)
            all_coll = ((nx.unsqueeze(1) == all_lb_x) &
                        (ny.unsqueeze(1) == all_lb_y) & ~at_spawn.unsqueeze(1))
            any_coll = all_coll.any(dim=1)
            can_move = can_move & ~any_coll

        bot_xs[ci] = torch.where(can_move, nx.to(torch.int16), bx)
        bot_ys[ci] = torch.where(can_move, ny.to(torch.int16), by)

        # === PICKUP ===
        is_pickup = (actions == ACT_PICKUP)
        item_idx = action_items.long().clamp(0, self.num_items - 1)
        inv_count = (inv >= 0).sum(dim=1)
        has_space = inv_count < INV_CAP

        ix = self.item_pos_x[item_idx].to(torch.int32)
        iy = self.item_pos_y[item_idx].to(torch.int32)
        mdist = (bot_xs[ci].to(torch.int32) - ix).abs() + (bot_ys[ci].to(torch.int32) - iy).abs()
        adjacent = mdist == 1
        can_pickup = is_pickup & has_space & adjacent
        pickup_type = self.item_types[item_idx]

        added = torch.zeros(B, dtype=torch.bool, device=d)
        inv_cols = []
        for s in range(INV_CAP):
            slot_empty = inv[:, s] < 0
            add_here = can_pickup & slot_empty & ~added
            inv_cols.append(torch.where(add_here, pickup_type, inv[:, s]))
            added = added | add_here
        bot_invs[ci] = torch.stack(inv_cols, dim=1)

        # === DROPOFF ===
        is_dropoff = (actions == ACT_DROPOFF)
        at_drop = (bot_xs[ci] == self.drop_x) & (bot_ys[ci] == self.drop_y)
        has_items = (bot_invs[ci] >= 0).any(dim=1)
        can_dropoff = is_dropoff & at_drop & has_items & (active_idx < self.num_orders)

        if can_dropoff.any():
            aidx = active_idx.long().clamp(0, self.num_orders - 1)
            act_req = self.order_req[aidx]

            bot_invs[ci], active_del_new, score_add = self._vectorized_deliver(
                bot_invs[ci], act_req, active_del, can_dropoff, B, d)
            active_del[:] = active_del_new
            score[:] = score + score_add

            # Compact inventory
            sort_key = (bot_invs[ci] < 0).to(torch.int8)
            _, sort_idx = sort_key.sort(dim=1, stable=True)
            bot_invs[ci] = bot_invs[ci].gather(1, sort_idx.long())

            # Check order completion
            aidx = active_idx.long().clamp(0, self.num_orders - 1)
            act_req = self.order_req[aidx]
            slot_done = (active_del == 1) | (act_req < 0)
            has_required = (act_req >= 0).any(dim=1)
            order_complete = slot_done.all(dim=1) & can_dropoff & has_required

            if order_complete.any():
                orders_comp_new = orders_comp + order_complete.to(torch.int32)
                orders_comp[:] = orders_comp_new

                # Auto-delivery on order completion
                new_aidx, active_del_auto, score_auto = self._auto_deliver_nbot(
                    order_complete, bot_xs, bot_ys, bot_invs,
                    locked_inv, locked_bx, locked_by,
                    active_idx, active_del, score, B, d)
                active_idx[:] = new_aidx
                active_del[:] = active_del_auto
                score[:] = score_auto

    def _step_one_locked(self, li, round_num,
                         bot_xs, bot_ys, bot_invs,
                         active_idx, active_del, score, orders_comp,
                         locked_inv, locked_bx, locked_by,
                         B, d, spawn_x, spawn_y):
        """Process one locked bot's action."""
        lb_act = int(self.locked_actions[li, round_num])
        lb_item = int(self.locked_action_items[li, round_num])
        lbx = locked_bx[:, li]
        lby = locked_by[:, li]

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

            lb_can_move = lb_walk

            # Collision with candidate bots
            for ci in range(self.n_candidates):
                coll = ((lnx == bot_xs[ci].to(torch.int32)) &
                        (lny == bot_ys[ci].to(torch.int32)) & ~lb_at_spawn)
                lb_can_move = lb_can_move & ~coll

            # Collision with other locked bots
            if self.num_locked > 1:
                all_lb_x = locked_bx.to(torch.int32)
                all_lb_y = locked_by.to(torch.int32)
                all_coll = ((lnx.unsqueeze(1) == all_lb_x) &
                            (lny.unsqueeze(1) == all_lb_y) & ~lb_at_spawn.unsqueeze(1))
                all_coll[:, li] = False
                any_coll = all_coll.any(dim=1)
                lb_can_move = lb_can_move & ~any_coll

            locked_bx[:, li] = torch.where(lb_can_move, lnx.to(torch.int16), lbx)
            locked_by[:, li] = torch.where(lb_can_move, lny.to(torch.int16), lby)

        elif lb_act == ACT_PICKUP and lb_item >= 0:
            item_x = int(self.item_pos_x[lb_item])
            item_y = int(self.item_pos_y[lb_item])
            lbx_cur = locked_bx[:, li]
            lby_cur = locked_by[:, li]
            lb_mdist = (lbx_cur.to(torch.int32) - item_x).abs() + (lby_cur.to(torch.int32) - item_y).abs()
            lb_adjacent = lb_mdist == 1

            pickup_type_lb = self.item_types[lb_item]
            added_lb = torch.zeros(B, dtype=torch.bool, device=d)
            for s in range(INV_CAP):
                slot_empty = locked_inv[:, li, s] < 0
                add_here = slot_empty & ~added_lb & lb_adjacent
                locked_inv[:, li, s] = torch.where(
                    add_here, pickup_type_lb, locked_inv[:, li, s])
                added_lb = added_lb | add_here

        elif lb_act == ACT_DROPOFF:
            lbx_cur = locked_bx[:, li]
            lby_cur = locked_by[:, li]
            lb_at_drop = (lbx_cur == self.drop_x) & (lby_cur == self.drop_y)
            lb_can_drop = lb_at_drop & (active_idx < self.num_orders)

            if lb_can_drop.any():
                aidx_l = active_idx.long().clamp(0, self.num_orders - 1)
                act_req_lb = self.order_req[aidx_l]

                linv_b = locked_inv[:, li, :]
                linv_b, active_del_new, score_add = self._vectorized_deliver(
                    linv_b, act_req_lb, active_del, lb_can_drop, B, d)
                locked_inv[:, li, :] = linv_b
                active_del[:] = active_del_new
                score[:] = score + score_add

                sort_key = (linv_b < 0).to(torch.int8)
                _, sort_idx = sort_key.sort(dim=1, stable=True)
                locked_inv[:, li, :] = linv_b.gather(1, sort_idx.long())

                # Check order completion
                act_req_lb = self.order_req[active_idx.long().clamp(0, self.num_orders - 1)]
                slot_done = (active_del == 1) | (act_req_lb < 0)
                has_required = (act_req_lb >= 0).any(dim=1)
                order_complete_lb = slot_done.all(dim=1) & lb_can_drop & has_required

                if order_complete_lb.any():
                    orders_comp[:] = orders_comp + order_complete_lb.to(torch.int32)
                    new_aidx, ad_auto, sc_auto = self._auto_deliver_nbot(
                        order_complete_lb, bot_xs, bot_ys, bot_invs,
                        locked_inv, locked_bx, locked_by,
                        active_idx, active_del, score, B, d)
                    active_idx[:] = new_aidx
                    active_del[:] = ad_auto
                    score[:] = sc_auto

    def _vectorized_deliver(self, bot_inv_slice, act_req, active_del, can_deliver, B, d):
        """Vectorized delivery matching items to order slots."""
        inv_exp = bot_inv_slice.unsqueeze(2)
        req_exp = act_req.unsqueeze(1)
        del_exp = active_del.unsqueeze(1)
        can_exp = can_deliver.unsqueeze(1).unsqueeze(2)

        match = ((inv_exp == req_exp) & (del_exp == 0) &
                 (inv_exp >= 0) & can_exp)

        score_add = torch.zeros(B, dtype=torch.int32, device=d)
        used_order = torch.zeros(B, MAX_ORDER_SIZE, dtype=torch.bool, device=d)
        new_inv_cols = []

        for s in range(INV_CAP):
            available = match[:, s, :] & ~used_order
            cumsum = available.to(torch.int32).cumsum(dim=1)
            first_match = available & (cumsum == 1)
            any_match = first_match.any(dim=1)

            active_del = active_del + first_match.to(torch.int8)
            new_inv_cols.append(torch.where(any_match, self._neg1_i8, bot_inv_slice[:, s]))
            score_add = score_add + first_match.sum(dim=1).to(torch.int32)
            used_order = used_order | first_match

        bot_inv_slice = torch.stack(new_inv_cols, dim=1)
        return bot_inv_slice, active_del, score_add

    def _auto_deliver_nbot(self, order_complete, bot_xs, bot_ys, bot_invs,
                           locked_inv, locked_bx, locked_by,
                           active_idx, active_del, score, B, d):
        """Auto-delivery when an order completes — all bots at dropoff deliver to new active."""
        score = score + order_complete.to(torch.int32) * 5
        new_aidx = active_idx + order_complete.to(torch.int32)
        oc_mask = order_complete.unsqueeze(1).expand_as(active_del)
        active_del = torch.where(oc_mask, self._zero_i8, active_del)

        valid_auto = order_complete & (new_aidx < self.num_orders)
        new_aidx_l = new_aidx.long().clamp(0, self.num_orders - 1)
        new_req = self.order_req[new_aidx_l]

        # Auto-deliver locked bots at dropoff
        if locked_bx is not None:
            for li in range(self.num_locked):
                b_at_drop = (locked_bx[:, li] == self.drop_x) & (locked_by[:, li] == self.drop_y)
                b_auto = valid_auto & b_at_drop
                if b_auto.any():
                    linv = locked_inv[:, li, :]
                    linv, active_del, sa = self._vectorized_deliver(
                        linv, new_req, active_del, b_auto, B, d)
                    locked_inv[:, li, :] = linv
                    score = score + sa

        # Auto-deliver candidate bots at dropoff
        for ci in range(self.n_candidates):
            c_at_drop = (bot_xs[ci] == self.drop_x) & (bot_ys[ci] == self.drop_y)
            c_auto = valid_auto & c_at_drop
            if c_auto.any():
                bot_invs[ci], active_del, sa = self._vectorized_deliver(
                    bot_invs[ci], new_req, active_del, c_auto, B, d)
                score = score + sa
                sort_key = (bot_invs[ci] < 0).to(torch.int8)
                _, sort_idx = sort_key.sort(dim=1, stable=True)
                bot_invs[ci] = bot_invs[ci].gather(1, sort_idx.long())

        return new_aidx, active_del, score

    @torch.no_grad()
    def _eval_nbot(self, state: dict[str, torch.Tensor], round_num: int) -> torch.Tensor:
        """Evaluate joint states for beam pruning. Returns [B] float32."""
        B = state['bot0_x'].shape[0]
        d = self.device
        rounds_left = MAX_ROUNDS - round_num - 1

        ev = state['score'].float() * 100000

        if self.speed_bonus > 0:
            ev = ev + self.speed_bonus * state['score'].float() * rounds_left

        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)
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

        # Preview order
        pidx = (aidx + 1).clamp(0, self.num_orders - 1)
        prev_req = self.order_req[pidx]

        # Per-candidate bot evaluation
        for ci in range(self.n_candidates):
            bx = state[f'bot{ci}_x'].long()
            by = state[f'bot{ci}_y'].long()
            inv = state[f'bot{ci}_inv']
            dist_drop = self.dist_to_dropoff[by, bx].float()

            # Active inventory matching
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

            # Preview matching
            prev_req_exp = prev_req.unsqueeze(1)
            match_preview = (inv_exp == prev_req_exp) & (prev_req_exp >= 0) & has_item_exp
            inv_matches_preview = match_preview.any(dim=2) & ~inv_matches_active
            frac_exp = fraction_done.unsqueeze(1).expand_as(inv_matches_preview)
            preview_val = 5000.0 + 10000.0 * frac_exp
            ev = ev + (inv_matches_preview.float() * preview_val).sum(dim=1)

            # Dead inventory
            has_item_flat = (inv >= 0)
            is_dead = has_item_flat & ~inv_matches_active & ~inv_matches_preview
            dead_penalty = 50000 * min(1.0, rounds_left / 150.0) + 5000
            ev = ev - is_dead.sum(dim=1).float() * dead_penalty

            # Distance guidance
            inv_count = has_item_flat.sum(dim=1)
            has_space = inv_count < INV_CAP
            best_dist = torch.full((B,), 9999.0, device=d)
            for os_idx in range(MAX_ORDER_SIZE):
                needed = active_needed[:, os_idx]
                if not needed.any():
                    continue
                ntype = act_req[:, os_idx].long().clamp(0, self.num_types - 1)
                d_type = self.dist_to_type[ntype, by, bx].float()
                better = needed & has_space & (d_type < best_dist)
                best_dist = torch.where(better, d_type, best_dist)

            close = best_dist < 9999
            min_trip = best_dist + dist_drop + 2
            trip_feasible = close & (min_trip <= rounds_left)

            _m1 = (trip_feasible & has_space & ~has_active_inv).float()
            ev = ev + _m1 * (20000.0 - best_dist * 800)

            # Delivery urgency
            delivery_urgency = max(1.0, (100 - rounds_left) / 20.0) if rounds_left < 100 else 1.0
            ev = ev - has_active_inv.float() * dist_drop * 200 * delivery_urgency

            cant_deliver = has_active_inv & (dist_drop >= rounds_left)
            ev = ev - cant_deliver.float() * 80000

        # === JOINT coordination bonuses (the key advantage) ===
        # Slot coverage: which candidate covers which order slot
        for os_idx in range(MAX_ORDER_SIZE):
            slot_needed = active_needed[:, os_idx]
            if not slot_needed.any():
                continue
            slot_type = act_req[:, os_idx]

            covering_count = torch.zeros(B, dtype=torch.int32, device=d)
            for ci in range(self.n_candidates):
                inv_ci = state[f'bot{ci}_inv']
                has_type = (inv_ci == slot_type.unsqueeze(1)).any(dim=1) & (slot_type >= 0)
                covering_count = covering_count + has_type.to(torch.int32)

            # Unique coverage bonus
            unique = slot_needed & (covering_count == 1)
            ev = ev + unique.float() * 35000

            # Redundancy penalty
            redundant = slot_needed & (covering_count > 1)
            ev = ev - redundant.float() * 15000

        # Inter-bot spacing: penalize candidates too close together
        for i in range(self.n_candidates):
            for j in range(i + 1, self.n_candidates):
                dist_ij = (
                    (state[f'bot{i}_x'].to(torch.int32) - state[f'bot{j}_x'].to(torch.int32)).abs() +
                    (state[f'bot{i}_y'].to(torch.int32) - state[f'bot{j}_y'].to(torch.int32)).abs()
                )
                too_close = dist_ij <= 1
                ev = ev - too_close.float() * 3000

        # Aisle congestion between candidates
        if self._aisle_columns.any():
            for ci in range(self.n_candidates):
                bx_ci = state[f'bot{ci}_x'].long()
                by_ci = state[f'bot{ci}_y'].long()
                in_aisle = self._aisle_columns[bx_ci] & ~self._corridor_rows[by_ci]
                if not in_aisle.any():
                    continue
                for cj in range(self.n_candidates):
                    if cj == ci:
                        continue
                    same_col = (state[f'bot{cj}_x'] == state[f'bot{ci}_x'])
                    by_cj = state[f'bot{cj}_y'].long().clamp(0, self.H - 1)
                    cj_not_corr = ~self._corridor_rows[by_cj]
                    cong = in_aisle & same_col & cj_not_corr
                    ev = ev - cong.float() * 4000

        return ev

    @torch.no_grad()
    def dp_search_nbot(self, game_state: GameState,
                       max_states: int = 100_000,
                       verbose: bool = True,
                       on_round: Callable | None = None,
                       max_combos: int = 50,
                       ) -> tuple[int, dict[int, list[tuple[int, int]]]]:
        """Run N-bot joint DP search.

        Returns:
            (best_score, {bot_id: [(act, item)] * 300})
        """
        t0 = time.time()
        state = self._init_state(game_state)
        d = self.device
        history = CPUHistory()

        # We need to track per-bot actions for reconstruction.
        # Store them as concatenated per-round: [bot0_act, bot0_item, bot1_act, bot1_item, ...]
        per_bot_act_history: list[list[np.ndarray]] = [[] for _ in range(self.n_candidates)]
        per_bot_item_history: list[list[np.ndarray]] = [[] for _ in range(self.n_candidates)]
        parent_history: list[np.ndarray] = []

        for rnd in range(MAX_ROUNDS):
            t_rnd = time.time()
            B = state['bot0_x'].shape[0]

            # Expand with distance-adaptive action generation
            expanded, per_bot_acts, per_bot_items = self._expand_nbot(
                state, rnd, max_combos_per_state=max_combos)

            total_cands = expanded['bot0_x'].shape[0]

            # Step
            new_state = self._step_nbot(expanded, per_bot_acts, per_bot_items, rnd)

            # Eval + topk
            k = min(max_states, total_cands)
            if total_cands > k * 3:
                # Two-phase: cheap then full
                cheap_ev = new_state['score'].float() * 100000
                if self.speed_bonus > 0:
                    cheap_ev = cheap_ev + self.speed_bonus * new_state['score'].float() * (MAX_ROUNDS - rnd - 1)
                prefilter_k = min(k * 2, total_cands)
                _, pf_idx = torch.topk(cheap_ev, prefilter_k)
                pf_state = {key: val[pf_idx] for key, val in new_state.items()}
                evals = self._eval_nbot(pf_state, rnd)
                _, topk_local = torch.topk(evals, k)
                topk_idx = pf_idx[topk_local]
            else:
                evals = self._eval_nbot(new_state, rnd)
                _, topk_idx = torch.topk(evals, k)

            # Dedup
            selected = {key: val[topk_idx] for key, val in new_state.items()}
            hashes = self._hash_nbot(selected)
            if k > 1:
                sorted_h, sort_perm = hashes.sort()
                dups = torch.zeros(k, dtype=torch.bool, device=d)
                dups[1:] = sorted_h[1:] == sorted_h[:-1]
                _, unsort = sort_perm.sort()
                unique_mask = ~dups[unsort]
                # Remove duplicates
                unique_idx = unique_mask.nonzero(as_tuple=True)[0]
                if len(unique_idx) < k:
                    topk_idx = topk_idx[unique_idx]
                    k = len(unique_idx)

            # Gather final beam
            state = {key: new_state[key][topk_idx] for key in new_state}

            # Compute parent indices
            C = total_cands // B if B > 0 else 1
            parents = topk_idx // C if C > 0 else torch.zeros(k, dtype=torch.long, device=d)

            parent_history.append(parents.cpu().numpy().astype(np.int32))
            for ci in range(self.n_candidates):
                per_bot_act_history[ci].append(
                    per_bot_acts[ci][topk_idx].cpu().numpy().astype(np.int8))
                per_bot_item_history[ci].append(
                    per_bot_items[ci][topk_idx].cpu().numpy().astype(np.int16))

            if verbose and (rnd < 5 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
                dt = time.time() - t_rnd
                best_score = state['score'].max().item()
                print(f"  R{rnd:3d}: score={best_score:3d}, beam={k:,}, "
                      f"cands={total_cands:,}, dt={dt:.3f}s",
                      file=sys.stderr)

            if on_round:
                best_score = state['score'].max().item()
                on_round(rnd, best_score, k, total_cands, time.time() - t_rnd)

        # Find best and backtrack
        best_idx = state['score'].argmax().item()
        best_score = state['score'][best_idx].item()

        # Reconstruct per-bot action sequences
        bot_actions = {bid: [] for bid in self.candidate_bot_ids}
        idx = best_idx
        for rnd in range(MAX_ROUNDS - 1, -1, -1):
            for ci, bid in enumerate(self.candidate_bot_ids):
                act = int(per_bot_act_history[ci][rnd][idx])
                item = int(per_bot_item_history[ci][rnd][idx])
                bot_actions[bid].append((act, item))
            idx = int(parent_history[rnd][idx])

        # Reverse all sequences
        for bid in bot_actions:
            bot_actions[bid].reverse()

        total_time = time.time() - t0
        if verbose:
            bots_str = ','.join(str(b) for b in self.candidate_bot_ids)
            print(f"\nJoint DP ({self.n_candidates}-bot [{bots_str}]): "
                  f"score={best_score}, time={total_time:.1f}s, beam={max_states:,}",
                  file=sys.stderr)

        return best_score, bot_actions

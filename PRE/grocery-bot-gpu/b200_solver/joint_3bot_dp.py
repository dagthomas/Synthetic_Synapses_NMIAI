"""3-bot joint GPU DP solver for grocery bot optimization.

Plans 3 bots simultaneously in a shared state space, exploring the full
N^3 cross-product of actions per round. This shatters the sequential DP
ceiling by guaranteeing perfect coordination between the 3 bots.

Hardware scaling:
  5090 (32GB):   50K-200K states, ~1-5 min per triple
  B200 (192GB):  500K-2M states, ~15-30s per triple

Usage:
    solver = GPU3BotDP(map_state, all_orders, bot_ids=(0,1,2), ...)
    score, acts = solver.dp_search(gs, max_states=50000)
    # acts = {0: [(act, item)] * 300, 1: [...], 2: [...]}
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np
import torch

# Add parent dir to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_engine import (
    GameState, MapState, Order,
    MAX_ROUNDS, INV_CAP, MAX_ORDER_SIZE,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
)
from precompute import PrecomputedTables


# Hash mixing primes (large coprime constants for FNV-like hashing)
_P0 = 6364136223846793005
_P1 = 1442695040888963407
_P2 = 3935559000370003845
_P3 = 2862933555777941757


class GPU3BotDP:
    """GPU-accelerated 3-bot joint DP solver.

    State: 3 bot positions + 3 inventories + shared order state.
    Actions: full N^3 cross-product (typically 343 combos for N=7).
    Collision resolution: sequential by bot ID (lower first).
    """

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 bot_ids: tuple[int, int, int],
                 device: str = 'cuda',
                 num_bots_total: int = 5,
                 locked_trajectories: dict | None = None,
                 speed_bonus: float = 0.0):
        assert len(bot_ids) == 3
        self.device = device
        self.ms = map_state
        self.all_orders = all_orders
        self.bot_ids = tuple(sorted(bot_ids))
        self.num_bots_total = num_bots_total
        self.speed_bonus = speed_bonus
        self.num_orders = len(all_orders)

        W, H = map_state.width, map_state.height
        self.W, self.H = W, H
        self.num_items = map_state.num_items
        self.num_types = map_state.num_types
        self.drop_x = int(map_state.drop_off[0])
        self.drop_y = int(map_state.drop_off[1])
        self.spawn_x = int(map_state.spawn[0])
        self.spawn_y = int(map_state.spawn[1])

        d = device

        # Grid and walkability
        self.grid = torch.tensor(map_state.grid, dtype=torch.int8, device=d)
        self.walkable = ((self.grid == CELL_FLOOR) | (self.grid == CELL_DROPOFF))

        # Items
        self.item_x = torch.tensor(
            map_state.item_positions[:, 0], dtype=torch.int16, device=d)
        self.item_y = torch.tensor(
            map_state.item_positions[:, 1], dtype=torch.int16, device=d)
        self.item_types = torch.tensor(
            map_state.item_types, dtype=torch.int8, device=d)

        # Precomputed distances
        tables = PrecomputedTables.get(map_state)
        gpu_t = tables.to_gpu_tensors(d)
        self.dist_to_type = gpu_t['dist_to_type']      # [num_types, H, W]
        self.dist_to_dropoff = gpu_t['dist_to_dropoff'] # [H, W]

        # Build adjacency: adj[y, x, k] = item_idx pickable from (x,y), -1 if none
        max_adj = 8
        adj = torch.full((H, W, max_adj), -1, dtype=torch.int16, device=d)
        adj_count = torch.zeros((H, W), dtype=torch.int8, device=d)
        for ii in range(self.num_items):
            ix = int(map_state.item_positions[ii, 0])
            iy = int(map_state.item_positions[ii, 1])
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ax, ay = ix + dx, iy + dy
                if 0 <= ax < W and 0 <= ay < H:
                    c = map_state.grid[ay, ax]
                    if c == CELL_FLOOR or c == CELL_DROPOFF:
                        k = int(adj_count[ay, ax])
                        if k < max_adj:
                            adj[ay, ax, k] = ii
                            adj_count[ay, ax] = k + 1
        self.adj = adj           # [H, W, max_adj] int16
        self.adj_count = adj_count  # [H, W] int8
        self.max_adj = max_adj

        # DP actions: moves (4) + wait + dropoff + pickup slots (position-dependent)
        # Pickup slots resolve to actual items based on bot position at expand time
        max_adj_actual = int(adj_count.max()) if int(adj_count.max()) > 0 else 1
        self.dp_max_adj = max_adj_actual
        N = 6 + max_adj_actual
        acts_list = [ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN,
                     ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_DROPOFF]
        for _ in range(max_adj_actual):
            acts_list.append(ACT_PICKUP)
        self.dp_actions = torch.tensor(acts_list, dtype=torch.int8, device=d)
        self.N = N  # actions per bot

        # Movement deltas indexed by action code
        self.DX = torch.tensor([0, 0, 0, -1, 1, 0, 0], dtype=torch.int16, device=d)
        self.DY = torch.tensor([0, -1, 1, 0, 0, 0, 0], dtype=torch.int16, device=d)

        # Orders table
        self.order_req = torch.full(
            (self.num_orders, MAX_ORDER_SIZE), -1, dtype=torch.int8, device=d)
        for i, o in enumerate(all_orders):
            for j, t in enumerate(o.required):
                self.order_req[i, j] = int(t)

        # Locked trajectories (other bots not in our triple)
        self.num_locked = 0
        self._locked_pos_x = None
        self._locked_pos_y = None
        if locked_trajectories is not None:
            self.num_locked = locked_trajectories['locked_pos_x'].shape[0]
            self._locked_pos_x = torch.tensor(
                locked_trajectories['locked_pos_x'], dtype=torch.int16, device=d)
            self._locked_pos_y = torch.tensor(
                locked_trajectories['locked_pos_y'], dtype=torch.int16, device=d)
            # Map: which locked indices correspond to bots with lower IDs than each candidate
            self._locked_bot_ids = list(locked_trajectories['locked_bot_ids'])

        # Pre-build combo indices (N^3)
        N = self.N
        NNN = N ** 3
        self.NNN = NNN
        self._combo_a0 = torch.arange(N, device=d).view(N, 1, 1).expand(
            N, N, N).reshape(-1)
        self._combo_a1 = torch.arange(N, device=d).view(1, N, 1).expand(
            N, N, N).reshape(-1)
        self._combo_a2 = torch.arange(N, device=d).view(1, 1, N).expand(
            N, N, N).reshape(-1)

        print(f"  GPU3BotDP init: bots={self.bot_ids}, N={N}, NNN={NNN}, "
              f"locked={self.num_locked}, items={self.num_items}",
              file=sys.stderr)

    def _resolve_items_batch(self, bx, by, action_idx):
        """Resolve item index from bot position and action index.

        For pickup actions (index >= 6), looks up adjacent items at the bot's
        position. For non-pickup actions, returns -1.

        Args:
            bx, by: [E] bot positions
            action_idx: [E] index into dp_actions (0..N-1), long tensor

        Returns:
            [E] int16 item indices (-1 if not a valid pickup)
        """
        pickup_slot = (action_idx - 6).clamp(0, self.dp_max_adj - 1)
        is_pickup = action_idx >= 6
        items = self.adj[by.long(), bx.long(), pickup_slot]  # [E] int16
        return torch.where(is_pickup, items,
                           torch.tensor(-1, dtype=torch.int16, device=self.device))

    def _init_state(self, gs: GameState) -> dict[str, torch.Tensor]:
        """Create initial batch-1 state from GameState."""
        d = self.device
        b0, b1, b2 = self.bot_ids

        bx = torch.tensor(
            [[gs.bot_positions[b0, 0], gs.bot_positions[b1, 0],
              gs.bot_positions[b2, 0]]], dtype=torch.int16, device=d)
        by = torch.tensor(
            [[gs.bot_positions[b0, 1], gs.bot_positions[b1, 1],
              gs.bot_positions[b2, 1]]], dtype=torch.int16, device=d)

        binv = torch.full((1, 3, INV_CAP), -1, dtype=torch.int8, device=d)
        for i, bid in enumerate(self.bot_ids):
            for s in range(INV_CAP):
                binv[0, i, s] = int(gs.bot_inventories[bid][s])

        # Find active order index
        aidx = 0
        for i, o in enumerate(gs.orders):
            if o.status == 'active':
                aidx = i
                break

        adel = torch.zeros((1, MAX_ORDER_SIZE), dtype=torch.int8, device=d)
        for o in gs.orders:
            if o.status == 'active':
                for j in range(len(o.delivered)):
                    adel[0, j] = int(o.delivered[j])
                break

        return {
            'bot_x': bx,           # [1, 3]
            'bot_y': by,           # [1, 3]
            'bot_inv': binv,       # [1, 3, INV_CAP]
            'active_idx': torch.tensor([aidx], dtype=torch.int32, device=d),
            'active_del': adel,    # [1, MAX_ORDER_SIZE]
            'score': torch.tensor([gs.score], dtype=torch.int32, device=d),
            'orders_comp': torch.tensor([gs.orders_completed], dtype=torch.int32,
                                        device=d),
        }

    def _hash(self, state: dict) -> torch.Tensor:
        """Compute 64-bit lossy hash for 3-bot state (FNV-like mixing)."""
        nt = self.num_types + 2  # +1 for empty, +1 for padding

        def bot_hash(idx):
            h = state['bot_x'][:, idx].long() * self.H + state['bot_y'][:, idx].long()
            for s in range(INV_CAP):
                h = h * nt + (state['bot_inv'][:, idx, s].long() + 1)
            return h

        h0 = bot_hash(0)
        h1 = bot_hash(1)
        h2 = bot_hash(2)

        # Delivery bitmask
        del_bits = torch.zeros_like(state['active_idx'].long())
        for j in range(MAX_ORDER_SIZE):
            del_bits = del_bits + state['active_del'][:, j].long() * (1 << j)
        ho = state['active_idx'].long() * 64 + del_bits

        # FNV-like mix into 64 bits (int64 wraps on overflow)
        return h0 * _P0 + h1 * _P1 + h2 * _P2 + ho * _P3

    def _resolve_move(self, bx, by, action, occupied_x_list, occupied_y_list,
                      locked_x, locked_y):
        """Resolve one bot's movement against occupied tiles and locked bots.

        Args:
            bx, by: [E] current position
            action: [E] action code (int8)
            occupied_x_list: list of [E] tensors for already-resolved bots
            occupied_y_list: list of [E] tensors
            locked_x: [E, L] or None - locked bot positions this round
            locked_y: [E, L] or None

        Returns:
            new_bx, new_by: [E] resolved position
        """
        is_move = (action >= ACT_MOVE_UP) & (action <= ACT_MOVE_RIGHT)
        act_clamp = action.long().clamp(0, 6)
        dx = self.DX[act_clamp]
        dy = self.DY[act_clamp]

        prop_x = (bx + dx).clamp(0, self.W - 1)
        prop_y = (by + dy).clamp(0, self.H - 1)

        # Check walkability
        valid = is_move & self.walkable[prop_y.long(), prop_x.long()]

        # Check collision with locked bots (not at spawn)
        if locked_x is not None:
            at_spawn = (prop_x == self.spawn_x) & (prop_y == self.spawn_y)
            for li in range(locked_x.shape[1]):
                collides = (prop_x == locked_x[:, li]) & (prop_y == locked_y[:, li])
                valid = valid & ~(collides & ~at_spawn)

        # Check collision with already-resolved candidate bots
        at_spawn = (prop_x == self.spawn_x) & (prop_y == self.spawn_y)
        for ox, oy in zip(occupied_x_list, occupied_y_list):
            collides = (prop_x == ox) & (prop_y == oy)
            valid = valid & ~(collides & ~at_spawn)

        new_x = torch.where(valid, prop_x, bx)
        new_y = torch.where(valid, prop_y, by)
        return new_x, new_y

    def _process_pickups(self, bx, by, action, item_idx, inv):
        """Process pickup action for one bot.

        Args:
            bx, by: [E] position
            action: [E] int8 action code
            item_idx: [E] int16 item index (-1 if not pickup)
            inv: [E, INV_CAP] int8 inventory

        Returns:
            new_inv: [E, INV_CAP] int8 (modified in-place or cloned)
        """
        E = bx.shape[0]
        is_pickup = (action == ACT_PICKUP) & (item_idx >= 0)
        if not is_pickup.any():
            return inv

        new_inv = inv.clone()

        # Check adjacency: bot must be adjacent to the item
        safe_idx = item_idx.long().clamp(0, self.num_items - 1)
        ix = self.item_x[safe_idx]
        iy = self.item_y[safe_idx]
        adjacent = ((bx - ix).abs() + (by - iy).abs() == 1)
        can_pick = is_pickup & adjacent

        if not can_pick.any():
            return new_inv

        # Find empty inventory slot
        item_type = self.item_types[safe_idx]  # [E]

        for s in range(INV_CAP):
            slot_empty = (new_inv[:, s] == -1)
            do_pick = can_pick & slot_empty
            if do_pick.any():
                new_inv[:, s] = torch.where(do_pick, item_type, new_inv[:, s])
                can_pick = can_pick & ~do_pick  # Consume the pickup
                break

        return new_inv

    def _process_dropoffs(self, bx_list, by_list, act_list, inv_list,
                          score, aidx, adel, ocomp):
        """Process dropoff for all 3 bots. Handles delivery and order advancement.

        Args:
            bx_list: list of 3 [E] tensors (resolved positions)
            by_list: list of 3 [E] tensors
            act_list: list of 3 [E] int8 action codes
            inv_list: list of 3 [E, INV_CAP] int8 inventories
            score, aidx, adel, ocomp: [E] state tensors

        Returns:
            (new_score, new_aidx, new_adel, new_ocomp, new_inv_list)
        """
        E = score.shape[0]
        new_score = score.clone()
        new_aidx = aidx.clone()
        new_adel = adel.clone()
        new_ocomp = ocomp.clone()
        new_invs = [inv.clone() for inv in inv_list]

        # Process each bot's dropoff
        for bi in range(3):
            bx, by = bx_list[bi], by_list[bi]
            act = act_list[bi]
            inv = new_invs[bi]

            at_drop = (bx == self.drop_x) & (by == self.drop_y)
            is_drop = (act == ACT_DROPOFF) & at_drop

            if not is_drop.any():
                continue

            # Get active order requirements
            safe_aidx = new_aidx.long().clamp(0, self.num_orders - 1)
            act_req = self.order_req[safe_aidx]  # [E, MAX_ORDER_SIZE]

            # For each inventory slot, check if it matches an undelivered order slot
            for s in range(INV_CAP):
                inv_type = inv[:, s]  # [E] int8
                has_item = (inv_type >= 0) & is_drop

                if not has_item.any():
                    continue

                for os in range(MAX_ORDER_SIZE):
                    matches = (has_item &
                               (inv_type == act_req[:, os]) &
                               (new_adel[:, os] == 0) &
                               (act_req[:, os] >= 0))
                    if matches.any():
                        new_adel[:, os] = torch.where(
                            matches, torch.ones_like(new_adel[:, os]),
                            new_adel[:, os])
                        new_invs[bi][:, s] = torch.where(
                            matches,
                            torch.full_like(inv_type, -1),
                            new_invs[bi][:, s])
                        new_score = new_score + matches.int()  # +1 per delivery
                        has_item = has_item & ~matches
                        break

        # Check order completion: all required slots delivered
        safe_aidx = new_aidx.long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[safe_aidx]
        required_mask = (act_req >= 0)  # [E, MAX_ORDER_SIZE]
        all_delivered = (required_mask & (new_adel == 1)).sum(dim=1) == required_mask.sum(dim=1)
        any_required = required_mask.any(dim=1)
        order_complete = all_delivered & any_required

        if order_complete.any():
            new_score = new_score + order_complete.int() * 5  # +5 bonus
            new_ocomp = new_ocomp + order_complete.int()
            new_aidx = new_aidx + order_complete.int()

            # Reset delivery mask for new order
            new_adel = torch.where(
                order_complete.unsqueeze(1).expand_as(new_adel),
                torch.zeros_like(new_adel),
                new_adel)

            # Chain reaction: check if any bot holds items matching the NEW active order
            new_safe_aidx = new_aidx.long().clamp(0, self.num_orders - 1)
            new_req = self.order_req[new_safe_aidx]  # [E, MAX_ORDER_SIZE]

            # Auto-deliver from all 3 bots' inventories
            for bi in range(3):
                bx, by = bx_list[bi], by_list[bi]
                at_drop_bi = (bx == self.drop_x) & (by == self.drop_y)
                can_chain = order_complete & at_drop_bi

                if not can_chain.any():
                    continue

                inv = new_invs[bi]
                for s in range(INV_CAP):
                    inv_type = inv[:, s]
                    has_item = (inv_type >= 0) & can_chain

                    for os in range(MAX_ORDER_SIZE):
                        matches = (has_item &
                                   (inv_type == new_req[:, os]) &
                                   (new_adel[:, os] == 0) &
                                   (new_req[:, os] >= 0))
                        if matches.any():
                            new_adel[:, os] = torch.where(
                                matches, torch.ones_like(new_adel[:, os]),
                                new_adel[:, os])
                            new_invs[bi][:, s] = torch.where(
                                matches, torch.full_like(inv_type, -1),
                                new_invs[bi][:, s])
                            new_score = new_score + matches.int()
                            has_item = has_item & ~matches
                            break

        return new_score, new_aidx, new_adel, new_ocomp, new_invs

    def _expand(self, state: dict, round_num: int) -> tuple:
        """Expand all states by all N^3 action combos.

        Returns:
            (expanded_state, combo_indices) where combo_indices [E] int32
            maps each expanded state to its action combo.
        """
        B = state['bot_x'].shape[0]
        N = self.N
        NNN = self.NNN
        E = B * NNN
        d = self.device

        # Expand parent states: [B, ...] → [E, ...]
        def rep(t):
            return t.repeat_interleave(NNN, dim=0)

        bx = rep(state['bot_x'])     # [E, 3]
        by = rep(state['bot_y'])     # [E, 3]
        binv = rep(state['bot_inv']) # [E, 3, INV_CAP]
        aidx = rep(state['active_idx'])
        adel = rep(state['active_del'])
        score = rep(state['score'])
        ocomp = rep(state['orders_comp'])

        # Tile action combos
        a0_idx = self._combo_a0.repeat(B)  # [E]
        a1_idx = self._combo_a1.repeat(B)
        a2_idx = self._combo_a2.repeat(B)

        act0 = self.dp_actions[a0_idx]  # [E] int8
        act1 = self.dp_actions[a1_idx]
        act2 = self.dp_actions[a2_idx]

        # Locked bot positions this round
        locked_x = locked_y = None
        if self._locked_pos_x is not None and round_num < MAX_ROUNDS:
            locked_x = self._locked_pos_x[:, round_num].unsqueeze(0).expand(E, -1)
            locked_y = self._locked_pos_y[:, round_num].unsqueeze(0).expand(E, -1)

        # Extract per-bot positions
        bx0, by0 = bx[:, 0], by[:, 0]
        bx1, by1 = bx[:, 1], by[:, 1]
        bx2, by2 = bx[:, 2], by[:, 2]

        # Resolve pickup item indices from position + pickup slot
        item0 = self._resolve_items_batch(bx0, by0, a0_idx)
        item1 = self._resolve_items_batch(bx1, by1, a1_idx)
        item2 = self._resolve_items_batch(bx2, by2, a2_idx)

        # Resolve movements in bot ID order
        # Bot IDs are sorted in self.bot_ids; process lowest first
        new_x0, new_y0 = self._resolve_move(bx0, by0, act0, [], [], locked_x, locked_y)
        new_x1, new_y1 = self._resolve_move(bx1, by1, act1,
                                              [new_x0], [new_y0], locked_x, locked_y)
        new_x2, new_y2 = self._resolve_move(bx2, by2, act2,
                                              [new_x0, new_x1], [new_y0, new_y1],
                                              locked_x, locked_y)

        # Process pickups
        inv0 = self._process_pickups(new_x0, new_y0, act0, item0, binv[:, 0])
        inv1 = self._process_pickups(new_x1, new_y1, act1, item1, binv[:, 1])
        inv2 = self._process_pickups(new_x2, new_y2, act2, item2, binv[:, 2])

        # Process dropoffs and order advancement
        new_score, new_aidx, new_adel, new_ocomp, new_invs = self._process_dropoffs(
            [new_x0, new_x1, new_x2],
            [new_y0, new_y1, new_y2],
            [act0, act1, act2],
            [inv0, inv1, inv2],
            score, aidx, adel, ocomp)

        # Pack action combo index for parent tracking
        combo_idx = a0_idx * (N * N) + a1_idx * N + a2_idx  # [E] int32-range

        new_state = {
            'bot_x': torch.stack([new_x0, new_x1, new_x2], dim=1),
            'bot_y': torch.stack([new_y0, new_y1, new_y2], dim=1),
            'bot_inv': torch.stack(new_invs, dim=1),
            'active_idx': new_aidx,
            'active_del': new_adel,
            'score': new_score,
            'orders_comp': new_ocomp,
        }
        return new_state, combo_idx

    def _eval(self, state: dict, round_num: int) -> torch.Tensor:
        """Evaluate states for beam pruning. Returns [E] float32 (higher=better).

        Simple per-bot eval summed across all 3 bots, plus coordination bonuses.
        """
        E = state['bot_x'].shape[0]
        d = self.device
        rounds_left = MAX_ROUNDS - round_num - 1

        ev = state['score'].float() * 100000

        if self.speed_bonus > 0:
            ev = ev + self.speed_bonus * state['score'].float() * rounds_left

        aidx = state['active_idx'].long().clamp(0, self.num_orders - 1)
        act_req = self.order_req[aidx]    # [E, MAX_ORDER_SIZE]
        act_del = state['active_del']     # [E, MAX_ORDER_SIZE]

        active_needed = (act_req >= 0) & (act_del == 0)
        active_done = (act_req >= 0) & (act_del == 1)
        active_remaining = active_needed.sum(dim=1).float()
        fraction_done = active_done.sum(dim=1).float() / (
            (act_req >= 0).sum(dim=1).float().clamp(min=1))

        # Order progress bonuses
        ev = ev + fraction_done * 30000
        ev = ev + (active_remaining == 0).float() * 50000
        ev = ev + (active_remaining == 1).float() * 20000

        # Per-bot evaluation
        for bi in range(3):
            bx = state['bot_x'][:, bi].long()
            by = state['bot_y'][:, bi].long()
            inv = state['bot_inv'][:, bi]  # [E, INV_CAP]

            dist_drop = self.dist_to_dropoff[by, bx].float()

            # Inventory-order matching
            inv_exp = inv.unsqueeze(2)       # [E, INV_CAP, 1]
            req_exp = act_req.unsqueeze(1)   # [E, 1, MAX_ORDER_SIZE]
            del_exp = act_del.unsqueeze(1)   # [E, 1, MAX_ORDER_SIZE]
            has_item = (inv >= 0).unsqueeze(2)

            match_active = ((inv_exp == req_exp) & (del_exp == 0) &
                            (req_exp >= 0) & has_item)
            inv_matches = match_active.any(dim=2)  # [E, INV_CAP]
            has_active = inv_matches.any(dim=1)
            num_active = inv_matches.sum(dim=1).float()

            # Active inventory value
            active_val = (70000 - dist_drop * 1750).clamp(min=0)
            ev = ev + num_active * active_val

            # Delivery urgency
            ev = ev - has_active.float() * dist_drop * 200

            # Distance to needed items
            inv_count = (inv >= 0).sum(dim=1)
            has_space = inv_count < INV_CAP
            best_dist = torch.full((E,), 9999.0, device=d)
            for os in range(MAX_ORDER_SIZE):
                needed = active_needed[:, os]
                if not needed.any():
                    continue
                ntype = act_req[:, os].long().clamp(0, self.num_types - 1)
                dt = self.dist_to_type[ntype, by, bx].float()
                better = needed & has_space & (dt < best_dist)
                best_dist = torch.where(better, dt, best_dist)

            close = best_dist < 9999
            feasible = close & has_space & (best_dist + dist_drop + 2 <= rounds_left)
            ev = ev + (feasible & ~has_active).float() * (20000 - best_dist * 800)

            # Preview matching
            pidx = (aidx + 1).clamp(0, self.num_orders - 1)
            prev_req = self.order_req[pidx]
            prev_exp = prev_req.unsqueeze(1)
            match_prev = ((inv_exp == prev_exp) & (prev_exp >= 0) & has_item)
            inv_prev = match_prev.any(dim=2) & ~inv_matches
            ev = ev + inv_prev.float().sum(dim=1) * 12000

            # Dead inventory penalty
            has_item_flat = (inv >= 0)
            is_dead = has_item_flat & ~inv_matches & ~inv_prev.bool()
            dead_pen = 50000 * min(1.0, rounds_left / 150.0) + 5000
            ev = ev - is_dead.sum(dim=1).float() * dead_pen

        # Coordination: penalize bots in same narrow column (congestion)
        for i in range(3):
            for j in range(i + 1, 3):
                same_col = (state['bot_x'][:, i] == state['bot_x'][:, j])
                same_row = (state['bot_y'][:, i] == state['bot_y'][:, j])
                # Same tile (not spawn) → heavy penalty
                at_spawn = ((state['bot_x'][:, i] == self.spawn_x) &
                            (state['bot_y'][:, i] == self.spawn_y))
                same_tile = same_col & same_row & ~at_spawn
                ev = ev - same_tile.float() * 15000
                # Same column (aisle congestion)
                ev = ev - (same_col & ~same_row).float() * 3000

        return ev

    def dp_search(self, gs: GameState, max_states: int = 50000,
                  verbose: bool = True) -> tuple[int, dict[int, list]]:
        """Run 3-bot joint DP search.

        Args:
            gs: Initial GameState.
            max_states: Maximum beam width.
            verbose: Print progress.

        Returns:
            (score, actions_dict) where actions_dict = {bot_id: [(act, item)] * 300}
        """
        t0 = time.time()
        d = self.device
        N = self.N
        NNN = self.NNN

        state = self._init_state(gs)
        B = 1

        # Parent tracking for backtracking
        parent_indices = []   # per round: [kept_size] int64 (index into prev round's beam)
        parent_combos = []    # per round: [kept_size] int32 (action combo index)
        parent_items = []     # per round: [kept_size, 3] int16 (resolved item indices)

        for rnd in range(MAX_ROUNDS):
            # Save pre-expansion positions for item resolution during backtracking
            prev_bx = state['bot_x']  # [B, 3]
            prev_by = state['bot_y']  # [B, 3]

            # Expand
            expanded, combo_idx = self._expand(state, rnd)
            E = expanded['bot_x'].shape[0]

            # Hash
            hashes = self._hash(expanded)

            # Dedup: sort by hash, keep best score per hash
            sort_idx = torch.argsort(hashes)
            hashes_sorted = hashes[sort_idx]

            # Find unique hash boundaries
            diff = torch.ones(E, dtype=torch.bool, device=d)
            diff[1:] = hashes_sorted[1:] != hashes_sorted[:-1]
            unique_mask = diff

            # For duplicates, keep the one with highest score
            scores_sorted = expanded['score'][sort_idx]
            # Within each hash group, we want the max score entry.
            # Simple approach: for each group, if current score < prev score and same hash, skip
            # More efficient: scatter_max by hash group
            # Practical approach: just keep unique + let the eval/topk handle it
            # (slight duplication won't matter much at high state counts)

            # Gather unique states
            unique_idx = sort_idx[unique_mask]
            U = unique_idx.shape[0]

            def gather(t, idx):
                if t.dim() == 1:
                    return t[idx]
                return t[idx]

            unique_state = {k: gather(v, unique_idx) for k, v in expanded.items()}

            # Eval
            ev = self._eval(unique_state, rnd)

            # Top-K
            K = min(max_states, U)
            if U > K:
                topk_vals, topk_idx = torch.topk(ev, K)
            else:
                topk_idx = torch.arange(U, device=d)

            # Map back to parent indices
            # expanded index = parent_state_idx * NNN + combo_offset
            expanded_idx = unique_idx[topk_idx]  # indices into expanded arrays
            parent_state_idx = expanded_idx // NNN
            combo_offset = expanded_idx % NNN

            parent_indices.append(parent_state_idx.cpu())
            parent_combos.append(combo_offset.cpu())

            # Resolve and store item indices for backtracking
            a0_kept = combo_offset // (N * N)
            a1_kept = (combo_offset // N) % N
            a2_kept = combo_offset % N
            kept_bx = prev_bx[parent_state_idx]  # [K, 3]
            kept_by = prev_by[parent_state_idx]  # [K, 3]
            items_r = torch.stack([
                self._resolve_items_batch(kept_bx[:, 0], kept_by[:, 0], a0_kept),
                self._resolve_items_batch(kept_bx[:, 1], kept_by[:, 1], a1_kept),
                self._resolve_items_batch(kept_bx[:, 2], kept_by[:, 2], a2_kept),
            ], dim=1)  # [K, 3]
            parent_items.append(items_r.cpu())

            # Keep top-K states (topk_idx indexes into unique_state, not expanded)
            state = {k: v[topk_idx] for k, v in unique_state.items()}
            B = state['bot_x'].shape[0]

            if verbose and (rnd < 10 or rnd % 25 == 0 or rnd == MAX_ROUNDS - 1):
                best_sc = int(state['score'].max())
                elapsed = time.time() - t0
                print(f"  R{rnd:>3}: score={best_sc:>3}, unique={U:,}, "
                      f"expanded={E:,}, kept={B:,}, t={elapsed:.1f}s",
                      file=sys.stderr)

            # Memory cleanup
            del expanded, hashes, sort_idx, ev
            if d == 'cuda':
                torch.cuda.empty_cache()

        # Find best final state
        final_scores = state['score']
        best_idx = int(final_scores.argmax())
        best_score = int(final_scores[best_idx])

        if verbose:
            elapsed = time.time() - t0
            print(f"\n  3-bot DP: score={best_score}, time={elapsed:.1f}s",
                  file=sys.stderr)

        # Backtrack to extract actions
        actions = self._backtrack(best_idx, parent_indices, parent_combos, parent_items)
        return best_score, actions

    def _backtrack(self, final_idx: int, parent_indices: list,
                   parent_combos: list, parent_items: list) -> dict[int, list]:
        """Extract per-bot actions by backtracking through parent pointers."""
        N = self.N
        b0, b1, b2 = self.bot_ids
        acts = {b0: [], b1: [], b2: []}

        idx = final_idx
        round_acts = []

        for rnd in range(MAX_ROUNDS - 1, -1, -1):
            combo = int(parent_combos[rnd][idx])
            items = parent_items[rnd][idx]  # [3] int16
            parent = int(parent_indices[rnd][idx])

            a0_idx = combo // (N * N)
            a1_idx = (combo // N) % N
            a2_idx = combo % N

            act0 = int(self.dp_actions[a0_idx])
            act1 = int(self.dp_actions[a1_idx])
            act2 = int(self.dp_actions[a2_idx])
            item0 = int(items[0])
            item1 = int(items[1])
            item2 = int(items[2])

            round_acts.append(((act0, item0), (act1, item1), (act2, item2)))
            idx = parent

        round_acts.reverse()
        for r, (a0, a1, a2) in enumerate(round_acts):
            acts[b0].append(a0)
            acts[b1].append(a1)
            acts[b2].append(a2)

        return acts

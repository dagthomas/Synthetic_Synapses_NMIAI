"""Nightmare solver V5: Chain reaction pipeline.

Strategy:
- Active order: handled reactively (like V3) — fetch & deliver ASAP
- Preview order: 3 COMMITTED chain bots pick up preview items, stage at dropoff zones
- When active completes → chain bots' preview items auto-deliver → potential chain!
- Remaining bots: help with active, or pre-fetch for future orders

Key difference from V3: dedicated chain staging bots with multi-round commitment.
V3 gets ZERO chains. V5 targets 30-50% chain rate.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class NightmareSolverV5:

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']

        # Subsystems
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # Item lookup: type_id -> [(item_idx, [adj_cells])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            self.type_items.setdefault(tid, []).append((idx, adj))

        # Position-adjacent items for opportunistic pickup
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            for adj in ms.item_adjacencies.get(idx, []):
                self.pos_to_items.setdefault(adj, []).append((idx, tid))

        self.all_orders_list = future_orders or []

        # Chain bot state (committed across rounds)
        # {bid: {'phase': 'fetch'|'stage', 'fetch_type': tid, 'fetch_item': idx,
        #        'fetch_pos': (x,y), 'dropoff': (x,y), 'remaining_types': [tid...]}}
        self.chain_bots: dict[int, dict] = {}
        self.last_completed = -1

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Chain stats
        self.chain_events: list[tuple[int, int]] = []

    # ---- helpers ----

    def _nearest_drop(self, pos):
        return min(self.drop_zones, key=lambda dz: self.tables.get_distance(pos, dz))

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _balanced_drop(self, pos, loads):
        return min(self.drop_zones,
                   key=lambda dz: self.tables.get_distance(pos, dz) + loads.get(dz, 0) * 5)

    def _nearest_item(self, pos, tid, claimed=None):
        """Find nearest item of type. Returns (item_idx, adj_pos) or (None, None)."""
        best_idx, best_adj, best_d = None, None, 9999
        for idx, adjs in self.type_items.get(tid, []):
            if claimed and idx in claimed:
                continue
            for adj in adjs:
                d = self.tables.get_distance(pos, adj)
                if d < best_d:
                    best_d = d
                    best_idx, best_adj = idx, adj
        return best_idx, best_adj

    def _best_item_for_needs(self, pos, needs, assigned, claimed):
        """Find best (type, item_idx, adj) for given needs dict."""
        best_t, best_idx, best_adj, best_cost = None, None, None, 9999
        for t, need in needs.items():
            if assigned.get(t, 0) >= need:
                continue
            idx, adj = self._nearest_item(pos, t, claimed)
            if adj is not None:
                cost = self.tables.get_distance(pos, adj)
                if cost < best_cost:
                    best_cost = cost
                    best_t, best_idx, best_adj = t, idx, adj
        return best_t, best_idx, best_adj

    def _escape(self, bid, pos, rnd):
        dirs = MOVES[:]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def _corridor_park(self, pos, occupied):
        best, best_d = self.spawn, 9999
        for cy in [1, self.ms.height // 2, self.ms.height - 3]:
            for dx in range(15):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width:
                        cell = (cx, cy)
                        if cell in self.tables.pos_to_idx and cell not in occupied:
                            if any(self.tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                                continue
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
        return best

    # ---- chain bot management ----

    def _setup_chain_bots(self, bot_pos, bot_inv, preview, busy_bids, claimed):
        """Assign 3 chain bots (one per dropoff zone) to carry preview items."""
        self.chain_bots.clear()
        if not preview:
            return

        preview_needs: dict[int, int] = {}
        for t in preview.needs():
            preview_needs[t] = preview_needs.get(t, 0) + 1
        if not preview_needs:
            return

        # Distribute preview types across 3 zones round-robin
        types_flat = []
        for t, n in preview_needs.items():
            types_flat.extend([t] * n)

        zone_types: list[list[int]] = [[], [], []]
        for i, t in enumerate(types_flat):
            zone_types[i % 3].append(t)

        for zone_idx in range(3):
            if not zone_types[zone_idx]:
                continue
            dz = self.drop_zones[zone_idx]

            # Find best available bot
            best_bid, best_score = None, 9999
            for bid in range(self.num_bots):
                if bid in busy_bids or bid in self.chain_bots:
                    continue
                inv = bot_inv.get(bid, [])
                pos = bot_pos[bid]
                # Prefer bots already carrying matching preview types
                matching = sum(1 for t in inv if t in preview_needs)
                free = INV_CAP - len(inv)
                if matching == 0 and free == 0:
                    continue
                # Estimate cost: distance to first needed item + distance to dropoff
                first_type = next((t for t in zone_types[zone_idx] if t not in inv), None)
                if first_type:
                    _, adj = self._nearest_item(pos, first_type, claimed)
                    d_item = self.tables.get_distance(pos, adj) if adj else 30
                else:
                    d_item = 0  # Already has all needed types
                d_drop = self.tables.get_distance(pos, dz)
                score = d_item + d_drop * 0.5 - matching * 8
                if score < best_score:
                    best_score = score
                    best_bid = bid

            if best_bid is None:
                continue

            inv = bot_inv.get(best_bid, [])
            inv_types = list(inv)
            still_need = []
            for t in zone_types[zone_idx]:
                if t in inv_types:
                    inv_types.remove(t)
                else:
                    still_need.append(t)
            free = INV_CAP - len(inv)
            still_need = still_need[:free]

            if still_need:
                first_t = still_need[0]
                idx, adj = self._nearest_item(bot_pos[best_bid], first_t, claimed)
                if idx is not None:
                    claimed.add(idx)
                    self.chain_bots[best_bid] = {
                        'phase': 'fetch',
                        'fetch_type': first_t,
                        'fetch_item': idx,
                        'fetch_pos': adj,
                        'dropoff': dz,
                        'remaining_types': still_need[1:],
                    }
                else:
                    self.chain_bots[best_bid] = {
                        'phase': 'stage', 'dropoff': dz,
                        'fetch_type': None, 'fetch_item': None,
                        'fetch_pos': None, 'remaining_types': [],
                    }
            else:
                self.chain_bots[best_bid] = {
                    'phase': 'stage', 'dropoff': dz,
                    'fetch_type': None, 'fetch_item': None,
                    'fetch_pos': None, 'remaining_types': [],
                }

    def _advance_chain_bot(self, bid, pos, inv, claimed):
        """Check if chain bot picked up its target, advance to next step."""
        info = self.chain_bots.get(bid)
        if not info or info['phase'] != 'fetch':
            return

        ft = info['fetch_type']
        if ft is not None and ft in inv:
            # Picked up successfully. More types to fetch?
            remaining = info['remaining_types']
            if remaining and len(inv) < INV_CAP:
                next_t = remaining[0]
                idx, adj = self._nearest_item(pos, next_t, claimed)
                if idx is not None:
                    claimed.add(idx)
                    info['fetch_type'] = next_t
                    info['fetch_item'] = idx
                    info['fetch_pos'] = adj
                    info['remaining_types'] = remaining[1:]
                    return

            # Done fetching → go to dropoff
            info['phase'] = 'stage'
            info['fetch_type'] = None
            info['fetch_item'] = None
            info['fetch_pos'] = None

    # ---- main action method ----

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        num_bots = self.num_bots
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        # ---- extract state ----
        bot_pos: dict[int, tuple[int, int]] = {}
        bot_inv: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_pos[bid] = (int(state.bot_positions[bid, 0]),
                           int(state.bot_positions[bid, 1]))
            bot_inv[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_pos.values()))
        for bid in range(num_bots):
            if self.prev_positions.get(bid) == bot_pos[bid]:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = bot_pos[bid]

        active = state.get_active_order()
        preview = state.get_preview_order()

        # Reset chain bots on order completion
        if state.orders_completed != self.last_completed:
            self.last_completed = state.orders_completed
            self.chain_bots.clear()

        # ---- active order analysis ----
        active_needs: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Count carried active items
        carrying_active: dict[int, int] = {}
        for bid, inv in bot_inv.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # ---- classify bots ----
        preview_needs: dict[int, int] = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        active_carrier_bids: list[int] = []
        preview_carrier_bids: list[int] = []
        dead_bids: list[int] = []
        empty_bids: list[int] = []

        for bid in range(num_bots):
            inv = bot_inv[bid]
            if not inv:
                empty_bids.append(bid)
                continue
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            if has_active:
                active_carrier_bids.append(bid)
            elif has_preview:
                preview_carrier_bids.append(bid)
            else:
                dead_bids.append(bid)

        # ---- build goals ----
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}
        claimed_items: set[int] = set()
        type_assigned: dict[int, int] = dict(carrying_active)
        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        # 1. ACTIVE CARRIERS → deliver (with fill-up option)
        for bid in active_carrier_bids:
            pos = bot_pos[bid]
            inv = bot_inv[bid]
            free = INV_CAP - len(inv)

            if free > 0 and active_short:
                bot_types = set(inv)
                filtered = {t: s for t, s in active_short.items() if t not in bot_types or s > 1}
                t, idx, adj = self._best_item_for_needs(pos, filtered, type_assigned, claimed_items)
                if t is not None:
                    dz = self._nearest_drop(pos)
                    d_direct = self.tables.get_distance(pos, dz)
                    d_via = self.tables.get_distance(pos, adj) + self._drop_dist(adj)
                    if d_via - d_direct <= 4:
                        claimed_items.add(idx)
                        type_assigned[t] = type_assigned.get(t, 0) + 1
                        goals[bid] = adj
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = idx
                        continue
                    # Also try preview fill-up
                    if preview_needs:
                        pt, pidx, padj = self._best_item_for_needs(
                            pos, {t2: n for t2, n in preview_needs.items() if t2 not in bot_types},
                            {}, claimed_items)
                        if pt is not None:
                            d_via_p = self.tables.get_distance(pos, padj) + self._drop_dist(padj)
                            if d_via_p - d_direct <= 6:
                                claimed_items.add(pidx)
                                goals[bid] = padj
                                goal_types[bid] = 'pickup'
                                pickup_targets[bid] = pidx
                                continue

            dz = self._balanced_drop(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # 2. ACTIVE FETCHERS → empty bots pick up active items
        empty_bids.sort(key=lambda b: min(
            (self.tables.get_distance(bot_pos[b], adj)
             for t in active_short for _, adjs in self.type_items.get(t, []) for adj in adjs),
            default=9999))

        for bid in empty_bids[:]:
            if not active_short:
                break
            t, idx, adj = self._best_item_for_needs(
                bot_pos[bid], active_short, type_assigned, claimed_items)
            if t is not None:
                claimed_items.add(idx)
                type_assigned[t] = type_assigned.get(t, 0) + 1
                goals[bid] = adj
                goal_types[bid] = 'pickup'
                pickup_targets[bid] = idx
                empty_bids.remove(bid)

        # 3. CHAIN STAGING → 3 committed bots for preview order
        busy_bids = set(goals.keys())
        if not self.chain_bots and preview:
            self._setup_chain_bots(bot_pos, bot_inv, preview, busy_bids, claimed_items)

        # Advance chain bots that completed their current fetch
        for bid in list(self.chain_bots.keys()):
            self._advance_chain_bot(bid, bot_pos[bid], bot_inv[bid], claimed_items)

        # Set goals for chain bots (DON'T override deliver goals)
        for bid, info in self.chain_bots.items():
            if bid in goals and goal_types.get(bid) == 'deliver':
                continue  # Active delivery takes priority over chain staging
            if info['phase'] == 'fetch':
                # Can't fetch if inventory is full
                if len(bot_inv.get(bid, [])) >= INV_CAP:
                    # Full inventory → deliver to free space, then resume chain
                    dz = self._balanced_drop(bot_pos[bid], dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                else:
                    goals[bid] = info['fetch_pos']
                    goal_types[bid] = 'chain_fetch'
                    pickup_targets[bid] = info['fetch_item']
            elif info['phase'] == 'stage':
                goals[bid] = info['dropoff']
                goal_types[bid] = 'stage'
            # Remove from empty/preview carrier lists if present
            if bid in empty_bids:
                empty_bids.remove(bid)
            if bid in preview_carrier_bids:
                preview_carrier_bids.remove(bid)

        # 4. PREVIEW CARRIERS → stage at dropoff (non-chain preview carriers)
        for bid in preview_carrier_bids:
            if bid in goals:
                continue
            pos = bot_pos[bid]
            dz = self._nearest_drop(pos)
            goals[bid] = dz
            goal_types[bid] = 'stage'

        # 5. DEAD BOTS → go to nearest dropoff (items might auto-deliver on chain)
        for bid in dead_bids:
            if bid in goals:
                continue
            pos = bot_pos[bid]
            # Dead bots have free slots → pick up preview items if close
            inv = bot_inv[bid]
            free = INV_CAP - len(inv)
            if free > 0 and preview_needs:
                bot_types = set(inv)
                t, idx, adj = self._best_item_for_needs(
                    pos, {t2: n for t2, n in preview_needs.items() if t2 not in bot_types},
                    {}, claimed_items)
                if t is not None and self.tables.get_distance(pos, adj) < 10:
                    claimed_items.add(idx)
                    goals[bid] = adj
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = idx
                    continue
            # Otherwise stage at dropoff
            dz = self._nearest_drop(pos)
            goals[bid] = dz
            goal_types[bid] = 'stage'

        # 6. REMAINING EMPTY BOTS → preview pickup or pre-position
        occupied = set(goals.values())
        preview_type_assigned: dict[int, int] = {}
        # Count chain bots' preview coverage
        for bid, info in self.chain_bots.items():
            if info.get('fetch_type'):
                preview_type_assigned[info['fetch_type']] = preview_type_assigned.get(info['fetch_type'], 0) + 1
            for t in info.get('remaining_types', []):
                preview_type_assigned[t] = preview_type_assigned.get(t, 0) + 1

        max_preview_pick = min(8, len(empty_bids))
        preview_picked = 0

        for bid in empty_bids:
            if bid in goals:
                continue
            pos = bot_pos[bid]

            # Preview pickup
            if preview_needs and preview_picked < max_preview_pick:
                p_short = {}
                for t, need in preview_needs.items():
                    s = need - preview_type_assigned.get(t, 0)
                    if s > 0:
                        p_short[t] = s
                if p_short:
                    t, idx, adj = self._best_item_for_needs(pos, p_short, {}, claimed_items)
                    if t is not None:
                        claimed_items.add(idx)
                        preview_type_assigned[t] = preview_type_assigned.get(t, 0) + 1
                        goals[bid] = adj
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = idx
                        preview_picked += 1
                        continue

            # Future pre-fetch
            future_idx = state.orders_completed + 2
            if future_idx < len(all_orders):
                future_types = set(int(t) for t in all_orders[future_idx].required)
                if future_types:
                    t, idx, adj = self._best_item_for_needs(
                        pos, {ft: 1 for ft in future_types}, {}, claimed_items)
                    if t is not None:
                        claimed_items.add(idx)
                        goals[bid] = adj
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = idx
                        continue

            # Idle: spread in corridors
            park = self._corridor_park(pos, occupied)
            occupied.add(park)
            goals[bid] = park
            goal_types[bid] = 'idle'

        # ---- pathfinding (PIBT) ----
        priority_map = {'deliver': 0, 'chain_fetch': 1, 'pickup': 2,
                        'stage': 3, 'preview': 4, 'idle': 5}
        urgency_order = sorted(range(num_bots), key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'idle'), 5),
            self.tables.get_distance(bot_pos[bid], goals.get(bid, self.spawn))
        ))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency_order, goal_types=goal_types)

        # ---- build final actions ----
        for bid in range(num_bots):
            pos = bot_pos[bid]
            inv = bot_inv[bid]
            gt = goal_types.get(bid, 'idle')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 4:
                act = self._escape(bid, pos, rnd)
                actions[bid] = (act, -1)
                self.stall_counts[bid] = 0
                continue

            # At dropoff: deliver if role is 'deliver'
            if pos in self.drop_set and gt == 'deliver' and inv:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            # At dropoff staging: drop off to deliver any active-matching items
            # Non-matching items stay in inventory for auto-delivery on chain
            if pos in self.drop_set and gt == 'stage' and inv:
                # Always try dropoff — delivers active items, keeps non-matching
                if active and any(active.needs_type(t) for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                else:
                    actions[bid] = (ACT_WAIT, -1)
                continue

            # At pickup target
            if gt in ('pickup', 'chain_fetch', 'preview') and bid in pickup_targets:
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                    continue

            # Opportunistic adjacent pickup
            if len(inv) < INV_CAP and pos in self.pos_to_items:
                opp = self._opp_pickup(pos, inv, active_needs, active_short, preview_needs)
                if opp is not None:
                    actions[bid] = (ACT_PICKUP, opp)
                    continue

            # Pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _opp_pickup(self, pos, inv, active_needs, active_short, preview_needs):
        """Opportunistic adjacent pickup: ONLY active-needed items.

        Preview items are NOT picked up opportunistically to avoid filling
        inventory with wrong types that prevent active item pickup.
        """
        bot_types = Counter(inv)
        best_idx = None
        for item_idx, tid in self.pos_to_items.get(pos, []):
            if tid in active_short and active_short[tid] > 0:
                if bot_types.get(tid, 0) > 0 and active_short[tid] <= 1:
                    continue
                best_idx = item_idx
                break  # Active items are always highest priority
        return best_idx

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV5(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
        action_log = []
        drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd

            pre_at_drop = {}
            for b in range(len(state.bot_positions)):
                bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                if bp in drop_set:
                    inv = state.bot_inv_list(b)
                    if inv:
                        pre_at_drop[b] = (bp, inv[:])

            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN×{c}!" if c > 1 else ""
                dropoff_info = ""
                if c >= 1:
                    at_drop = []
                    for b in range(len(state.bot_positions)):
                        bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                        if bp in drop_set:
                            binv = state.bot_inv_list(b)
                            at_drop.append(f"b{b}:{binv}")
                    if at_drop:
                        dropoff_info = f" AtDrop=[{', '.join(at_drop)}]"
                    if pre_at_drop:
                        pre_info = ", ".join(
                            f"b{b}@{p}:{i}" for b, (p, i) in pre_at_drop.items())
                        dropoff_info += f" PRE=[{pre_info}]"

                chain_info = ""
                if solver.chain_bots:
                    cb = [f"b{bid}:{info['phase']}" for bid, info in solver.chain_bots.items()]
                    chain_info = f" CB=[{','.join(cb)}]"

                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra + dropoff_info + chain_info)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7001-7010')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    for seed in seeds:
        score, _ = NightmareSolverV5.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f"Seed {seed}: {score}")

    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cascade-aware nightmare solver: team-split V6 with cascade staging.

Key design:
- Active team (bots 0-9): pick up active items + deliver
- Preview team (bots 10-19): pick up preview items + stage at dropoffs
- Preview items NEVER become dead (preview→active transition preserves them)
- Active deliverers fill up with preview items when detour is cheap
- Stagers at dropoffs DON'T drop off (wait for cascade auto-delivery)

Usage: python nightmare_cascade.py --seed 7009 --verbose
"""
from __future__ import annotations
import sys, time
import numpy as np
from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class CascadeSolver:
    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order], num_bots: int = 20,
                 active_team_size: int = 8):
        self.ms = ms
        self.tables = tables
        self.future_orders = future_orders
        self.num_bots = num_bots
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # Team assignment
        self.active_team_size = active_team_size
        self.active_team = set(range(active_team_size))
        self.preview_team = set(range(active_team_size, num_bots))

        # Zone infrastructure
        sorted_drops = sorted(self.drop_zones, key=lambda d: d[0])
        self.zone_dropoff = {i: dz for i, dz in enumerate(sorted_drops)}
        self.bot_zone = {bid: (0 if bid < 7 else (1 if bid < 14 else 2))
                         for bid in range(num_bots)}

        # Item lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            ix = int(ms.item_positions[idx, 0])
            adj = ms.item_adjacencies.get(idx, [])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))
            for a in adj:
                self.pos_to_items.setdefault(a, []).append((idx, tid))

        # Corridor parking
        near_drop = set()
        for cell in tables.pos_to_idx:
            if any(tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                near_drop.add(cell)
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28), -1: (0, ms.width - 1)}
        corridor_ys = [1, ms.height // 2, ms.height - 3]
        self._corridor_cells: dict[int, list[tuple[int, int]]] = {}
        for zone_id in [-1, 0, 1, 2]:
            x_lo, x_hi = zone_x_ranges[zone_id]
            cells = []
            for cy in corridor_ys:
                for cx in range(x_lo, x_hi + 1):
                    cell = (cx, cy)
                    if cell in tables.pos_to_idx and cell not in near_drop:
                        cells.append(cell)
            self._corridor_cells[zone_id] = cells

        # Bot tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._preview_bot_types: dict[int, int] = {}
        self._last_preview_id: int = -1

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _balanced_dropoff(self, pos, loads):
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 5
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _corridor_parking(self, pos, occupied, zone=-1):
        best = self.spawn
        best_d = 9999
        candidates = self._corridor_cells.get(zone, self._corridor_cells[-1])
        for cell in candidates:
            if cell not in occupied:
                d = self.tables.get_distance(pos, cell)
                if 0 < d < best_d:
                    best_d = d
                    best = cell
        return best

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        num_bots = self.num_bots

        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Active order needs
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Preview needs
        preview_needs = {}
        preview_oid = -1
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1
            preview_oid = preview_order.id

        # Reset preview tracking on order change
        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._last_preview_id = preview_oid

        # Count what's already being carried
        carrying_active = {}
        carrying_preview = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1
                if t in preview_needs:
                    carrying_preview[t] = carrying_preview.get(t, 0) + 1

        # Active shortfall
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Preview shortfall
        preview_short = {}
        for t, need in preview_needs.items():
            s = need - carrying_preview.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Classify all bots
        active_carriers = []
        preview_carriers = []
        dead_bots = []
        empty_active = []   # Active team, can pick up
        empty_preview = []  # Preview team, can pick up

        for bid, inv in bot_inventories.items():
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)

            if not inv:
                if bid in self.active_team:
                    empty_active.append(bid)
                else:
                    empty_preview.append(bid)
            elif has_active:
                active_carriers.append(bid)
            elif has_preview:
                preview_carriers.append(bid)
            elif len(inv) < INV_CAP:
                # Has useless items but free slots
                if bid in self.active_team:
                    empty_active.append(bid)
                else:
                    empty_preview.append(bid)
            else:
                dead_bots.append(bid)

        # Build goals
        goals = {}
        goal_types = {}
        pickup_targets = {}
        type_assigned = {}
        preview_type_assigned = dict(carrying_preview)
        claimed_items = set()
        dropoff_loads = {dz: 0 for dz in self.drop_zones}
        occupied_goals = set()

        # === ACTIVE CARRIERS: deliver (with fill-up detour) ===
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)
            bot_types = set(inv)

            if free_slots > 0:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)

                # Try active fill-up first
                best_detour = 9999
                best_item = None
                best_adj = None
                for tid in active_short:
                    if tid in bot_types and active_short[tid] <= 1:
                        continue
                    for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                        if item_idx in claimed_items:
                            continue
                        for adj in adj_cells:
                            d_to = self.tables.get_distance(pos, adj)
                            d_back = self._drop_dist(adj)
                            detour = d_to + d_back - drop_dist
                            if detour < best_detour:
                                best_detour = detour
                                best_item = item_idx
                                best_adj = adj

                # Try preview fill-up (for cascade potential)
                if best_detour >= 4 and preview_short:
                    for tid in preview_short:
                        if tid in bot_types:
                            continue
                        for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                            if item_idx in claimed_items:
                                continue
                            for adj in adj_cells:
                                d_to = self.tables.get_distance(pos, adj)
                                d_back = self._drop_dist(adj)
                                detour = d_to + d_back - drop_dist
                                if detour < best_detour:
                                    best_detour = detour
                                    best_item = item_idx
                                    best_adj = adj

                if best_detour < 6 and best_item is not None:
                    goals[bid] = best_adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = best_item
                    tid = int(self.ms.item_types[best_item])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(best_item)
                    continue

            # Go deliver
            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # === PREVIEW CARRIERS: stage at dropoff ===
        # Find which dropoffs have deliverers
        deliver_zones = set()
        for b2 in goals:
            if goal_types.get(b2) == 'deliver' and goals[b2] in self.drop_set:
                deliver_zones.add(goals[b2])

        staging_counts = {dz: 0 for dz in self.drop_zones}

        for bid in preview_carriers:
            pos = bot_positions[bid]

            if pos in self.drop_set:
                # Already at dropoff - stay for cascade
                goals[bid] = pos
                goal_types[bid] = 'stage'
                staging_counts[pos] = staging_counts.get(pos, 0) + 1
                continue

            # Go to nearest FREE dropoff (not occupied by deliverer)
            best_dz = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_zones:
                    continue
                if staging_counts.get(dz, 0) >= 3:  # Limit per-zone staging
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_dz = dz

            if best_dz is not None:
                staging_counts[best_dz] = staging_counts.get(best_dz, 0) + 1
                goals[bid] = best_dz
                goal_types[bid] = 'stage'
            else:
                # All dropoffs busy, go to nearest anyway
                dz = self._nearest_drop(pos)
                goals[bid] = dz
                goal_types[bid] = 'stage'

        # === DEAD BOTS: park ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)
            park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === ACTIVE TEAM EMPTY BOTS: pick up active items ===
        empty_active.sort(key=lambda bid: self._min_dist_types(
            bot_positions[bid], active_short.keys() if active_short else active_needs.keys()))

        for bid in empty_active:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bz = self.bot_zone.get(bid, 2)
            bot_types = set(inv)

            if len(inv) >= INV_CAP:
                if any(t in active_needs for t in inv):
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                else:
                    park = self._corridor_parking(pos, occupied_goals, zone=bz)
                    occupied_goals.add(park)
                    goals[bid] = park
                    goal_types[bid] = 'flee'
                continue

            assigned = False

            # Active pickup
            if active_short:
                remaining = sum(max(0, s - type_assigned.get(t, 0))
                                for t, s in active_short.items())
                if remaining > 0:
                    item_idx, adj_pos = self._assign_item(
                        pos, active_short, type_assigned, claimed_items, bot_types)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        assigned = True

            # Active team bots can also help with preview if active covered
            if not assigned and preview_short:
                remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                       for t, s in active_short.items())
                if remaining_active == 0:
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_short, preview_type_assigned, claimed_items,
                        bot_types, strict=True)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        assigned = True

            if not assigned:
                park = self._corridor_parking(pos, occupied_goals, zone=bz)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'

        # === PREVIEW TEAM EMPTY BOTS: pick up preview items ===
        empty_preview.sort(key=lambda bid: self._min_dist_types(
            bot_positions[bid], preview_short.keys() if preview_short else preview_needs.keys()))

        for bid in empty_preview:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bz = self.bot_zone.get(bid, 2)
            bot_types = set(inv)

            if len(inv) >= INV_CAP:
                # Full with preview/other items - stage at dropoff
                if any(t in preview_needs for t in inv):
                    dz = self._nearest_drop(pos)
                    goals[bid] = dz
                    goal_types[bid] = 'stage'
                elif any(t in active_needs for t in inv):
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                else:
                    park = self._corridor_parking(pos, occupied_goals, zone=bz)
                    occupied_goals.add(park)
                    goals[bid] = park
                    goal_types[bid] = 'flee'
                continue

            assigned = False

            # Preview pickup FIRST (team's primary duty)
            if preview_short:
                item_idx, adj_pos = self._assign_item(
                    pos, preview_short, preview_type_assigned, claimed_items,
                    bot_types, strict=True)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    assigned = True

            # If no preview items needed, help with active
            if not assigned and active_short:
                remaining = sum(max(0, s - type_assigned.get(t, 0))
                                for t, s in active_short.items())
                if remaining > 0:
                    item_idx, adj_pos = self._assign_item(
                        pos, active_short, type_assigned, claimed_items, bot_types)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        assigned = True

            if not assigned:
                park = self._corridor_parking(pos, occupied_goals, zone=bz)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'

        # === URGENCY ORDER ===
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'stage':
                return (0, dist)  # Stagers get SAME priority as deliverers
            elif gt == 'pickup':
                return (1, dist)
            elif gt == 'preview':
                return (2, dist)
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 3, dist)
            else:
                return (4, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # === BUILD ACTIONS ===
        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # At dropoff
            if pos in self.drop_set:
                if gt == 'deliver' and inv:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                if gt == 'stage':
                    # Stagers WAIT at dropoff for cascade (don't drop off!)
                    # UNLESS they also have active items
                    has_active_items = any(t in active_needs for t in inv)
                    if has_active_items and inv:
                        actions[bid] = (ACT_DROPOFF, -1)
                        continue
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # At pickup target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if len(inv) < INV_CAP:
                opp = self._opportunistic_pickup(bid, pos, inv, active_short,
                                                  active_needs, preview_needs)
                if opp is not None:
                    actions[bid] = opp
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _opportunistic_pickup(self, bid, pos, inv, active_short,
                               active_needs, preview_needs):
        """Pick up adjacent items that match active or preview needs."""
        adj_items = self.pos_to_items.get(pos, [])
        if not adj_items:
            return None
        bot_types = set(inv)

        # Active items first
        for item_idx, tid in adj_items:
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
                active_short[tid] -= 1
                return (ACT_PICKUP, item_idx)

        # Preview items for preview team bots
        if bid in self.preview_team:
            for item_idx, tid in adj_items:
                if tid in preview_needs and tid not in bot_types:
                    return (ACT_PICKUP, item_idx)

        # Preview items when active is covered
        if not active_short:
            for item_idx, tid in adj_items:
                if tid in preview_needs and tid not in bot_types:
                    return (ACT_PICKUP, item_idx)

        return None

    def _assign_item(self, bot_pos, needs, assigned_counts, claimed,
                     bot_types, strict=False):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need in needs.items():
            if need <= 0:
                continue
            if tid in bot_types:
                continue
            max_assign = need if strict else need + 1
            if assigned_counts.get(tid, 0) >= max_assign:
                continue
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    d_back = self._drop_dist(adj)
                    cost = d + d_back * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def _min_dist_types(self, pos, types):
        best = 9999
        for tid in types:
            for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

    def _escape_action(self, bid, pos, rnd):
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed: int, verbose: bool = False,
                active_team_size: int = 8) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = CascadeSolver(ms, tables, all_orders,
                               active_team_size=active_team_size)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7009')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--active-team', type=int, default=8)
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    for seed in seeds:
        score, actions = CascadeSolver.run_sim(
            seed, verbose=args.verbose, active_team_size=args.active_team)
        scores.append(score)
        print(f"Seed {seed}: {score}")

        if score > 0:
            from solution_store import save_solution
            save_solution('nightmare', score, actions, seed=seed)

    if len(scores) > 1:
        print(f"\nMean: {np.mean(scores):.1f}")


if __name__ == '__main__':
    main()

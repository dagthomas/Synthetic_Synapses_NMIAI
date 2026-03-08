#!/usr/bin/env python3
"""Nightmare Trip-Based Solver: Plan full trips for each bot, execute, repeat.

Instead of per-round greedy decisions, plans complete trips:
  bot → item1_adj → pickup → item2_adj → pickup → dropoff → dropoff

This naturally enables pipelining: while active-order bots deliver,
preview-order bots start trips.

Usage: python nightmare_trip_solver.py --seed 7009
"""
from __future__ import annotations

import sys
import time
import heapq
from collections import defaultdict

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables


MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class Trip:
    """A planned trip for a bot: pickup N items then deliver."""

    def __init__(self, bot_id: int, waypoints: list[tuple[int, int]],
                 actions_at_waypoint: list[tuple[int, int]],
                 item_types: list[int], purpose: str = 'active'):
        self.bot_id = bot_id
        self.waypoints = waypoints  # List of (x, y) positions to visit
        self.actions_at_waypoint = actions_at_waypoint  # Action to take when arriving
        self.item_types = item_types  # Types being picked up
        self.purpose = purpose  # 'active', 'preview', or 'stage'
        self.current_wp = 0
        self.path: list[int] = []  # Sequence of movement actions
        self.path_idx = 0

    def is_done(self) -> bool:
        return self.current_wp >= len(self.waypoints)


class TripSolver:
    """Plan and execute trips for 20 nightmare bots."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = tuple(ms.spawn)
        self.future_orders = future_orders
        self.num_bots = CONFIGS['nightmare']['bots']

        # Item info
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            ix = int(ms.item_positions[idx, 0])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        # Bot zones
        self.bot_zone = {i: (0 if i < 7 else (1 if i < 14 else 2)) for i in range(20)}

        # Active trips per bot
        self.bot_trips: dict[int, Trip | None] = {i: None for i in range(self.num_bots)}
        self.claimed_items: set[int] = set()

        # Pathfinding cache
        self._bfs_cache: dict[tuple, dict] = {}

        # Stall tracking
        self.prev_pos: dict[int, tuple[int, int]] = {}
        self.stall: dict[int, int] = {}

    def _nearest_drop(self, pos: tuple[int, int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _drop_dist(self, pos: tuple[int, int]) -> int:
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _bfs_path(self, start: tuple[int, int], goal: tuple[int, int],
                  blocked: set[tuple[int, int]] | None = None) -> list[int]:
        """BFS shortest path, returns list of movement actions."""
        if start == goal:
            return []
        blocked = blocked or set()
        blocked_clean = blocked - {goal}  # goal is always reachable

        queue = [(start, [])]
        visited = {start}
        while queue:
            pos, path = queue.pop(0)
            for act in MOVES:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                npos = (nx, ny)
                if npos == goal:
                    return path + [act]
                if npos not in visited and npos in self.tables.pos_to_idx:
                    if npos not in blocked_clean:
                        visited.add(npos)
                        queue.append((npos, path + [act]))
        return []  # No path found

    def _plan_trip(self, bot_id: int, bot_pos: tuple[int, int],
                   bot_inv: list[int], needs: dict[int, int],
                   assigned_counts: dict[int, int],
                   purpose: str = 'active',
                   occupied: set[tuple[int, int]] | None = None) -> Trip | None:
        """Plan a trip: pick up needed items, then deliver at dropoff."""
        free_slots = INV_CAP - len(bot_inv)
        if free_slots <= 0:
            # Full inventory → just deliver
            if any(t in needs for t in bot_inv):
                dz = self._nearest_drop(bot_pos)
                path = self._bfs_path(bot_pos, dz, occupied)
                if path:
                    return Trip(bot_id, [dz], [(ACT_DROPOFF, -1)],
                                [], purpose)
            return None

        # Find items to pick up
        pickup_plan = []
        remaining_types = set(bot_inv)
        current_pos = bot_pos

        for _ in range(free_slots):
            best_item = None
            best_adj = None
            best_cost = 9999
            best_tid = -1

            for tid, need_count in needs.items():
                if assigned_counts.get(tid, 0) >= need_count:
                    continue
                # Don't pick up duplicate types (waste)
                if tid in remaining_types and purpose == 'active':
                    if assigned_counts.get(tid, 0) >= need_count - 1:
                        continue

                for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                    if item_idx in self.claimed_items:
                        continue
                    for adj in adj_cells:
                        d_to_adj = self.tables.get_distance(current_pos, adj)
                        d_adj_to_drop = self._drop_dist(adj)
                        cost = d_to_adj + d_adj_to_drop * 0.4
                        if cost < best_cost:
                            best_cost = cost
                            best_item = item_idx
                            best_adj = adj
                            best_tid = tid

            if best_item is not None:
                pickup_plan.append((best_adj, best_item, best_tid))
                self.claimed_items.add(best_item)
                assigned_counts[best_tid] = assigned_counts.get(best_tid, 0) + 1
                remaining_types.add(best_tid)
                current_pos = best_adj
            else:
                break

        if not pickup_plan and not bot_inv:
            return None

        # Build trip: pickup all items, then go to dropoff
        waypoints = []
        actions = []
        types_picked = []

        for adj, item_idx, tid in pickup_plan:
            waypoints.append(adj)
            actions.append((ACT_PICKUP, item_idx))
            types_picked.append(tid)

        # Add dropoff if we have items to deliver
        if pickup_plan or any(t in needs for t in bot_inv):
            dz = self._nearest_drop(current_pos)
            waypoints.append(dz)
            actions.append((ACT_DROPOFF, -1))

        if waypoints:
            return Trip(bot_id, waypoints, actions, types_picked, purpose)
        return None

    def _plan_all_trips(self, state: GameState, all_orders: list[Order]):
        """Plan trips for all bots based on current game state."""
        active = state.get_active_order()
        preview = state.get_preview_order()

        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Check what's already being carried
        bot_positions = {}
        bot_inventories = {}
        for bid in range(self.num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Count items already being carried for active
        carrying_active = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        # Adjusted needs
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Classify bots
        active_carriers = []
        preview_carriers = []
        free_bots = []
        dead_bots = []

        for bid in range(self.num_bots):
            if self.bot_trips[bid] is not None and not self.bot_trips[bid].is_done():
                continue  # Already on a trip

            inv = bot_inventories[bid]
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)

            if has_active:
                active_carriers.append(bid)
            elif has_preview:
                preview_carriers.append(bid)
            elif len(inv) >= INV_CAP:
                dead_bots.append(bid)
            else:
                free_bots.append(bid)

        self.claimed_items.clear()
        active_assigned = dict(carrying_active)
        preview_assigned = {}

        occupied = set(bot_positions.values())

        # 1. Active carriers → deliver
        for bid in active_carriers:
            dz = self._nearest_drop(bot_positions[bid])
            self.bot_trips[bid] = Trip(bid, [dz], [(ACT_DROPOFF, -1)],
                                       [], 'active')

        # 2. Preview carriers → stage at dropoff
        for bid in preview_carriers:
            # Find dropoff not used by active carriers
            used_drops = set()
            for ac_bid in active_carriers:
                used_drops.add(self._nearest_drop(bot_positions[ac_bid]))
            best_dz = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in used_drops:
                    continue
                d = self.tables.get_distance(bot_positions[bid], dz)
                if d < best_d:
                    best_d = d
                    best_dz = dz
            if best_dz is None:
                best_dz = self._nearest_drop(bot_positions[bid])
            self.bot_trips[bid] = Trip(bid, [best_dz], [(ACT_DROPOFF, -1)],
                                       [], 'stage')

        # 3. Free bots → plan trips for active then preview
        # Sort by distance to nearest needed item
        free_bots.sort(key=lambda bid: min(
            (self.tables.get_distance(bot_positions[bid], adj)
             for tid in (active_short if active_short else preview_needs)
             for _, adjs, _ in self.type_items.get(tid, [])
             for adj in adjs),
            default=9999))

        for bid in free_bots:
            pos = bot_positions[bid]

            # Try active trip first
            if active_short:
                trip = self._plan_trip(bid, pos, bot_inventories[bid],
                                       active_short, active_assigned,
                                       'active', occupied)
                if trip:
                    self.bot_trips[bid] = trip
                    continue

            # Try preview trip
            if preview_needs:
                trip = self._plan_trip(bid, pos, bot_inventories[bid],
                                       preview_needs, preview_assigned,
                                       'preview', occupied)
                if trip:
                    self.bot_trips[bid] = trip
                    continue

            # No trip needed → wait
            self.bot_trips[bid] = None

        # Dead bots → move out of the way
        for bid in dead_bots:
            self.bot_trips[bid] = None

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        """Generate actions for all bots at round rnd."""
        num_bots = self.num_bots
        bot_positions = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))

        # Update stall tracking
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_pos.get(bid) == pos:
                self.stall[bid] = self.stall.get(bid, 0) + 1
            else:
                self.stall[bid] = 0
            self.prev_pos[bid] = pos

        # Re-plan trips if needed
        needs_replan = False
        for bid in range(num_bots):
            trip = self.bot_trips[bid]
            if trip is None or trip.is_done():
                needs_replan = True
                break

        if needs_replan or rnd % 15 == 0:
            self._plan_all_trips(state, all_orders)

        actions = [(ACT_WAIT, -1)] * num_bots
        occupied_next = set()

        for bid in range(num_bots):
            pos = bot_positions[bid]
            trip = self.bot_trips[bid]

            # Stall escape
            if self.stall.get(bid, 0) >= 3:
                dirs = list(MOVES)
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    npos = (nx, ny)
                    if npos in self.tables.pos_to_idx and npos not in occupied_next:
                        actions[bid] = (a, -1)
                        occupied_next.add(npos)
                        break
                continue

            if trip is None or trip.is_done():
                # No trip — just wait (or do opportunistic pickup)
                opp = self._opportunistic_pickup(state, bid, pos)
                if opp is not None:
                    actions[bid] = opp
                occupied_next.add(pos)
                continue

            wp = trip.waypoints[trip.current_wp]
            wp_action = trip.actions_at_waypoint[trip.current_wp]

            if pos == wp:
                # At waypoint → execute waypoint action
                actions[bid] = wp_action
                trip.current_wp += 1
                occupied_next.add(pos)
            else:
                # Move toward waypoint
                best_act = ACT_WAIT
                best_d = self.tables.get_distance(pos, wp)
                for act in MOVES:
                    nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                    npos = (nx, ny)
                    if npos in self.tables.pos_to_idx and npos not in occupied_next:
                        d = self.tables.get_distance(npos, wp)
                        if d < best_d:
                            best_d = d
                            best_act = act
                if best_act != ACT_WAIT:
                    nx, ny = pos[0] + DX[best_act], pos[1] + DY[best_act]
                    occupied_next.add((nx, ny))
                else:
                    occupied_next.add(pos)
                actions[bid] = (best_act, -1)

        return actions

    def _opportunistic_pickup(self, state: GameState, bid: int,
                              pos: tuple[int, int]):
        """Pick up adjacent items that match active/preview needs."""
        active = state.get_active_order()
        if not active or state.bot_inv_count(bid) >= INV_CAP:
            return None

        active_types = set()
        for t in active.needs():
            active_types.add(t)

        ms = self.ms
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_types:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = TripSolver(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']

        action_log = []
        t0 = time.time()
        chains = 0

        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains}"
                  f" Time={elapsed:.1f}s")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if '-' in args.seeds:
        lo, hi = args.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]

    scores = []
    for seed in seeds:
        score, _ = TripSolver.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f'Seed {seed}: {score}')

    if len(scores) > 1:
        print(f'Mean: {sum(scores)/len(scores):.1f}  '
              f'Min: {min(scores)}  Max: {max(scores)}')


if __name__ == '__main__':
    main()

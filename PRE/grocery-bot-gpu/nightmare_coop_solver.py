"""Cooperative A* rolling-horizon solver for nightmare mode.

Plans collision-free paths for all 20 bots over a rolling horizon,
using full order knowledge for pipelined item fetching.

Usage:
    python nightmare_coop_solver.py --seeds 7005 -v
    python nightmare_coop_solver.py --seeds 1000-1009
"""
from __future__ import annotations

import heapq
import time
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
MOVE_DELTAS = {
    ACT_MOVE_UP: (0, -1),
    ACT_MOVE_DOWN: (0, 1),
    ACT_MOVE_LEFT: (-1, 0),
    ACT_MOVE_RIGHT: (1, 0),
}


class CooperativeAStar:
    """Spacetime A* with reservation table for collision-free multi-agent paths."""

    def __init__(self, walkable: set[tuple[int, int]],
                 tables: PrecomputedTables,
                 spawn: tuple[int, int]):
        self.walkable = walkable
        self.tables = tables
        self.spawn = spawn
        # Reservation table: {(x, y, t): bot_id}
        self.reservations: dict[tuple[int, int, int], int] = {}

    def clear(self):
        self.reservations.clear()

    def reserve_path(self, bot_id: int, path: list[tuple[int, int]], start_time: int):
        """Reserve all cells along a path in the reservation table."""
        for dt, pos in enumerate(path):
            t = start_time + dt
            self.reservations[(pos[0], pos[1], t)] = bot_id

    def plan_path(self, bot_id: int, start: tuple[int, int], goal: tuple[int, int],
                  start_time: int, max_horizon: int = 30) -> list[tuple[int, int]] | None:
        """Find shortest collision-free path from start to goal.

        Returns list of (x, y) positions for each timestep, or None if no path found.
        """
        if start == goal:
            return [start]

        h0 = self.tables.get_distance(start, goal)
        if h0 >= 9999:
            return None

        # A* in spacetime: state = (x, y, t)
        # Priority: (f_score, tie_break, x, y, t)
        open_set = [(h0, 0, start[0], start[1], 0)]
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], int] = {(start[0], start[1], 0): 0}
        counter = 1

        while open_set:
            f, _, x, y, t = heapq.heappop(open_set)

            if (x, y) == goal:
                # Reconstruct path
                path = [(x, y)]
                state = (x, y, t)
                while state in came_from:
                    state = came_from[state]
                    path.append((state[0], state[1]))
                path.reverse()
                return path

            if t >= max_horizon:
                continue

            g = g_score.get((x, y, t), 9999)

            # Neighbors: 4 directions + wait
            neighbors = [
                (x, y),  # wait
                (x + 1, y), (x - 1, y),
                (x, y + 1), (x, y - 1),
            ]

            for nx, ny in neighbors:
                if (nx, ny) not in self.walkable:
                    continue

                nt = t + 1

                # Check reservation (cell occupied at time nt)
                reserved_by = self.reservations.get((nx, ny, start_time + nt))
                if reserved_by is not None and reserved_by != bot_id:
                    # Allow spawn stacking
                    if (nx, ny) != self.spawn:
                        continue

                # Swap detection: don't swap with another bot
                other_at_dest_now = self.reservations.get((nx, ny, start_time + t))
                other_at_src_next = self.reservations.get((x, y, start_time + nt))
                if (other_at_dest_now is not None and other_at_dest_now != bot_id
                        and other_at_src_next == other_at_dest_now):
                    continue

                new_g = g + 1
                if new_g < g_score.get((nx, ny, nt), 9999):
                    g_score[(nx, ny, nt)] = new_g
                    h = self.tables.get_distance((nx, ny), goal)
                    if h >= 9999:
                        continue
                    new_f = new_g + h
                    came_from[(nx, ny, nt)] = (x, y, t)
                    heapq.heappush(open_set, (new_f, counter, nx, ny, nt))
                    counter += 1

        return None


class NightmareCoopSolver:
    """Rolling-horizon cooperative solver for nightmare mode.

    Plans paths for all 20 bots using Cooperative A* with a rolling
    horizon window. Uses full order knowledge for pipelined fetching.
    """

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 all_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.all_orders = all_orders or []
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = CONFIGS['nightmare']['bots']
        self.walkable = self._build_walkable()

        # Item lookup: type_id -> [(item_idx, [(adj_x, adj_y), ...])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Bot task state (persists across rounds)
        # task: None | ('pickup', item_idx, adj_pos) | ('deliver', drop_zone) | ('stage', drop_zone)
        self.bot_tasks: dict[int, tuple | None] = {b: None for b in range(self.num_bots)}
        # Planned paths (from cooperative A*)
        self.bot_paths: dict[int, list[tuple[int, int]]] = {}
        self.bot_path_step: dict[int, int] = {}  # current step in path

        self._coop = CooperativeAStar(self.walkable, tables, self.spawn)

    def _build_walkable(self) -> set[tuple[int, int]]:
        w = set()
        for y in range(self.ms.height):
            for x in range(self.ms.width):
                if self.ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    w.add((x, y))
        return w

    def _nearest_drop(self, pos: tuple[int, int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _balanced_drop(self, pos: tuple[int, int],
                       loads: dict[tuple[int, int], int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_s = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            s = d + loads.get(dz, 0) * 6
            if s < best_s:
                best_s = s
                best = dz
        return best

    def _assign_items_to_bots(self, bot_positions: dict[int, tuple[int, int]],
                               bot_inventories: dict[int, list[int]],
                               needed_types: dict[int, int],
                               available_bots: list[int],
                               claimed_items: set[int]) -> dict[int, tuple[int, tuple[int, int]]]:
        """Greedily assign items to bots. Returns {bid: (item_idx, adj_pos)}."""
        assignments: dict[int, tuple[int, tuple[int, int]]] = {}
        type_assigned: dict[int, int] = {}

        # Sort bots by distance to nearest needed item
        bot_costs = []
        for bid in available_bots:
            pos = bot_positions[bid]
            min_d = 9999
            for tid in needed_types:
                for item_idx, adj_list in self.type_items.get(tid, []):
                    if item_idx in claimed_items:
                        continue
                    for adj in adj_list:
                        d = self.tables.get_distance(pos, adj)
                        if d < min_d:
                            min_d = d
            bot_costs.append((min_d, bid))
        bot_costs.sort()

        for _, bid in bot_costs:
            if not needed_types:
                break
            pos = bot_positions[bid]
            best_cost = 9999
            best_item = None
            best_adj = None
            best_tid = None

            for tid, need in needed_types.items():
                if need <= 0:
                    continue
                if type_assigned.get(tid, 0) >= need:
                    continue
                for item_idx, adj_list in self.type_items.get(tid, []):
                    if item_idx in claimed_items:
                        continue
                    for adj in adj_list:
                        d = self.tables.get_distance(pos, adj)
                        drop_d = min(self.tables.get_distance(adj, dz) for dz in self.drop_zones)
                        cost = d + drop_d * 0.3
                        if cost < best_cost:
                            best_cost = cost
                            best_item = item_idx
                            best_adj = adj
                            best_tid = tid

            if best_item is not None:
                assignments[bid] = (best_item, best_adj)
                claimed_items.add(best_item)
                type_assigned[best_tid] = type_assigned.get(best_tid, 0) + 1
                # Remove type if fully assigned
                if type_assigned[best_tid] >= needed_types[best_tid]:
                    del needed_types[best_tid]

        return assignments

    def _plan_cooperative_paths(self, bot_positions: dict[int, tuple[int, int]],
                                 goals: dict[int, tuple[int, int]],
                                 priority_order: list[int],
                                 horizon: int = 25) -> dict[int, list[tuple[int, int]]]:
        """Plan collision-free paths for all bots using Cooperative A*."""
        self._coop.clear()
        paths: dict[int, list[tuple[int, int]]] = {}

        for bid in priority_order:
            pos = bot_positions.get(bid)
            goal = goals.get(bid)
            if pos is None or goal is None:
                continue

            path = self._coop.plan_path(bid, pos, goal, start_time=0, max_horizon=horizon)
            if path:
                paths[bid] = path
                self._coop.reserve_path(bid, path, start_time=0)
            else:
                # Fallback: stay in place
                paths[bid] = [pos]
                self._coop.reserve_path(bid, [pos], start_time=0)

        return paths

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        """Per-round entry point."""
        ms = self.ms
        num_bots = len(state.bot_positions)

        # Extract positions and inventories
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Compute needs
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid, inv in bot_inventories.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_needs: dict[int, int] = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Classify bots
        active_carriers = []
        preview_carriers = []
        empty_bots = []
        dead_bots = []

        for bid in range(num_bots):
            inv = bot_inventories[bid]
            if not inv:
                empty_bots.append(bid)
                continue
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            if has_active:
                active_carriers.append(bid)
            elif has_preview:
                preview_carriers.append(bid)
            else:
                dead_bots.append(bid)

        # === TASK ASSIGNMENT ===
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}
        claimed_items: set[int] = set()
        drop_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        # Active carriers → deliver
        for bid in active_carriers:
            inv = bot_inventories[bid]
            free = INV_CAP - len(inv)
            total_short = sum(active_short.values())

            if free == 0 or total_short == 0:
                dz = self._balanced_drop(bot_positions[bid], drop_loads)
                drop_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
            else:
                # Try fill-up: pick more active items if close
                dz = self._nearest_drop(bot_positions[bid])
                drop_dist = self.tables.get_distance(bot_positions[bid], dz)
                assigned = self._assign_items_to_bots(
                    bot_positions, bot_inventories,
                    dict(active_short), [bid], claimed_items)
                if bid in assigned:
                    item_idx, adj_pos = assigned[bid]
                    d_to_item = self.tables.get_distance(bot_positions[bid], adj_pos)
                    if d_to_item < drop_dist and d_to_item < 10:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        tid = int(ms.item_types[item_idx])
                        active_short[tid] = max(0, active_short.get(tid, 0) - 1)
                        if active_short.get(tid) == 0:
                            del active_short[tid]
                        continue
                dz = self._balanced_drop(bot_positions[bid], drop_loads)
                drop_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'

        # Preview carriers → stage at non-deliver zones or pick more
        deliver_zones = set(goals[b] for b in goals if goal_types.get(b) == 'deliver'
                           and goals[b] in self.drop_set)
        staging_counts: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        for bid in preview_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free = INV_CAP - len(inv)

            # If partial inventory, pick more preview items
            if free > 0 and preview_needs:
                bot_types = set(inv)
                pick_needs = {t: n for t, n in preview_needs.items() if t not in bot_types}
                if pick_needs:
                    assigned = self._assign_items_to_bots(
                        bot_positions, bot_inventories,
                        pick_needs, [bid], claimed_items)
                    if bid in assigned:
                        item_idx, adj_pos = assigned[bid]
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        continue

            # Stage at non-deliver zone
            best_zone = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_zones:
                    continue
                if staging_counts[dz] >= 1:
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_zone = dz
            if best_zone is not None:
                staging_counts[best_zone] += 1
                goals[bid] = best_zone
                goal_types[bid] = 'stage'
            else:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'

        # Dead bots → park at spawn
        for bid in dead_bots:
            goals[bid] = self.spawn
            goal_types[bid] = 'park'

        # Empty bots → pick active items, then preview items
        remaining_short = dict(active_short)
        active_assigned = self._assign_items_to_bots(
            bot_positions, bot_inventories,
            remaining_short, empty_bots, claimed_items)

        preview_pickers = 0
        max_preview = 6

        for bid in empty_bots:
            if bid in active_assigned:
                item_idx, adj_pos = active_assigned[bid]
                goals[bid] = adj_pos
                goal_types[bid] = 'pickup'
                pickup_targets[bid] = item_idx
            elif preview_needs and preview_pickers < max_preview:
                pick_needs = dict(preview_needs)
                assigned = self._assign_items_to_bots(
                    bot_positions, bot_inventories,
                    pick_needs, [bid], claimed_items)
                if bid in assigned:
                    item_idx, adj_pos = assigned[bid]
                    goals[bid] = adj_pos
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = item_idx
                    preview_pickers += 1
                else:
                    goals[bid] = self.spawn
                    goal_types[bid] = 'park'
            else:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'

        # === COOPERATIVE PATHFINDING ===
        # Priority: deliver > pickup > stage > preview > park
        priority_map = {'deliver': 0, 'pickup': 1, 'stage': 2, 'preview': 3, 'park': 5}
        priority_order = sorted(range(num_bots), key=lambda b: (
            priority_map.get(goal_types.get(b, 'park'), 5),
            self.tables.get_distance(bot_positions[b], goals.get(b, self.spawn))
        ))

        paths = self._plan_cooperative_paths(bot_positions, goals, priority_order, horizon=20)

        # === BUILD ACTIONS ===
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # At dropoff: deliver
            if pos in self.drop_set and gt == 'deliver' and bot_inventories[bid]:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            # At pickup target: pick up
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if gt in ('pickup', 'preview', 'deliver') and len(bot_inventories[bid]) < INV_CAP:
                opp = self._opp_pickup(bid, pos, active_needs, preview_needs, bot_inventories[bid])
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Follow cooperative path
            path = paths.get(bid, [pos])
            if len(path) > 1:
                next_pos = path[1]
                # Convert position delta to action
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]
                if dx == 1:
                    actions[bid] = (ACT_MOVE_RIGHT, -1)
                elif dx == -1:
                    actions[bid] = (ACT_MOVE_LEFT, -1)
                elif dy == 1:
                    actions[bid] = (ACT_MOVE_DOWN, -1)
                elif dy == -1:
                    actions[bid] = (ACT_MOVE_UP, -1)
                else:
                    actions[bid] = (ACT_WAIT, -1)
            else:
                actions[bid] = (ACT_WAIT, -1)

        return actions

    def _opp_pickup(self, bid: int, pos: tuple[int, int],
                     active_needs: dict[int, int],
                     preview_needs: dict[int, int],
                     bot_inv: list[int]) -> tuple[int, int] | None:
        """Opportunistic adjacent pickup for active or preview items."""
        bot_types = set(bot_inv)
        total_active_short = sum(max(0, n) for n in active_needs.values())

        for item_idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[item_idx])
            if tid in active_needs and active_needs[tid] > 0:
                if tid in bot_types:
                    continue
            elif total_active_short == 0 and tid in preview_needs:
                if tid in bot_types:
                    continue
            else:
                continue
            for adj in self.ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        """Run full simulation. Returns (score, action_log)."""
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareCoopSolver(ms, tables, all_orders=all_orders)
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

            if verbose and (rnd < 3 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN×{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


DB_URL = "postgres://grocery:grocery123@localhost:5433/grocery_bot"


def record_to_pg(seed, score, orders_completed, items_delivered, action_log, elapsed):
    """Record run to PostgreSQL."""
    import json
    import os
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        return None

    db_url = os.environ.get("GROCERY_DB_URL", DB_URL)
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        from game_engine import build_map, CELL_WALL, CELL_SHELF, state_to_ws_format, actions_to_ws_format
        ms = build_map('nightmare')
        cfg = CONFIGS['nightmare']

        walls, shelves = [], []
        for y in range(ms.height):
            for x in range(ms.width):
                c = int(ms.grid[y, x])
                if c == CELL_WALL:
                    walls.append([x, y])
                elif c == CELL_SHELF:
                    shelves.append([x, y])

        items = [{"id": it["id"], "type": it["type"], "position": list(it["position"])}
                 for it in ms.items]

        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, spawn,
                              final_score, items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            seed, 'nightmare', ms.width, ms.height, cfg['bots'],
            ms.num_types, cfg['order_size'][0], cfg['order_size'][1],
            json.dumps(walls), json.dumps(shelves),
            json.dumps(items), json.dumps(list(ms.drop_off)),
            json.dumps(list(ms.spawn)),
            score, items_delivered, orders_completed, 'synthetic',
        ))
        run_id = cur.fetchone()[0]

        if action_log:
            gs, ao = init_game(seed, 'nightmare', num_orders=100)
            round_tuples = []
            for r in range(min(len(action_log), 500)):
                gs.round = r
                ws_data = state_to_ws_format(gs, ao)
                ws_acts = actions_to_ws_format(action_log[r], gs.map_state)
                bots = [{"id": b["id"], "position": b["position"],
                         "inventory": b.get("inventory", [])} for b in ws_data["bots"]]
                orders = [{"id": o["id"], "items_required": o["items_required"],
                           "items_delivered": o.get("items_delivered", []),
                           "status": o.get("status", "active")}
                          for o in ws_data.get("orders", [])]
                round_tuples.append((
                    run_id, r, json.dumps(bots), json.dumps(orders),
                    json.dumps(ws_acts), ws_data["score"], json.dumps([])
                ))
                step(gs, action_log[r], ao)

            execute_values(cur, """
                INSERT INTO rounds (run_id, round_number, bots, orders, actions, score, events)
                VALUES %s
            """, round_tuples)

        conn.commit()
        cur.close()
        conn.close()
        return run_id
    except Exception as e:
        print(f"  DB error: {e}", file=__import__('sys').stderr)
        return None


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    # Parse seeds
    seeds = []
    for part in args.seeds.split(','):
        if '-' in part and not part.startswith('-'):
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    scores = []
    t_total = time.time()
    for seed in seeds:
        score, action_log = NightmareCoopSolver.run_sim(seed, verbose=args.verbose)
        scores.append(score)

        if not args.no_record:
            # Compute stats from action_log
            gs, ao = init_game(seed, 'nightmare', num_orders=100)
            for r in range(len(action_log)):
                gs.round = r
                step(gs, action_log[r], ao)
            rid = record_to_pg(seed, gs.score, gs.orders_completed,
                               gs.items_delivered, action_log, time.time() - t_total)
            if rid:
                print(f"  Recorded to DB: run_id={rid}", file=sys.stderr)

        print(f"Seed {seed}: {score}")

    print(f"\n{'='*40}")
    print(f"Seeds: {len(scores)}")
    print(f"Mean: {sum(scores)/len(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {time.time()-t_total:.1f}s ({(time.time()-t_total)/len(scores):.1f}s/seed)")

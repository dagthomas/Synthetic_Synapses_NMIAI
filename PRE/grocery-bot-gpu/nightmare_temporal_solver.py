"""Temporal Planner V2 for Nightmare mode.

Collision-free offline planning using space-time A* with reservation tables.
- Batched pickups (up to 3 items per trip)
- Pipeline: active + preview order processing simultaneously
- Correct action timing (pickup/dropoff on separate rounds after arrival)

Usage:
    python nightmare_temporal_solver.py --seeds 7005 -v
    python nightmare_temporal_solver.py --seeds 7001-7010 --compare
"""
from __future__ import annotations

import heapq
import time
from typing import Optional

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import build_walkable

NUM_ROUNDS = 500
NUM_BOTS = 20


class ReservationTable:
    """Tracks occupied (x, y, t) cells AND edge conflicts for collision avoidance."""

    def __init__(self):
        self._reserved: set[tuple[int, int, int]] = set()
        # Edge conflicts: (from_x, from_y, to_x, to_y, t) means
        # "no bot can move from (from_x,from_y) to (to_x,to_y) at time t"
        self._edges: set[tuple[int, int, int, int, int]] = set()

    def is_free(self, x: int, y: int, t: int) -> bool:
        return (x, y, t) not in self._reserved

    def is_edge_free(self, fx: int, fy: int, tx: int, ty: int, t: int) -> bool:
        """Check if moving from (fx,fy) to (tx,ty) at time t is not blocked."""
        return (fx, fy, tx, ty, t) not in self._edges

    def reserve(self, x: int, y: int, t: int):
        self._reserved.add((x, y, t))

    def reserve_path(self, path: list[tuple[int, int, int]]):
        for x, y, t in path:
            self.reserve(x, y, t)
        # Reserve edge conflicts + departure cells for each move
        for i in range(1, len(path)):
            x0, y0, t0 = path[i - 1]
            x1, y1, t1 = path[i]
            if (x1, y1) != (x0, y0):  # actual move, not wait
                # Block the REVERSE direction at the same time step
                self._edges.add((x1, y1, x0, y0, t0))
                # Departure reservation: block the source cell at arrival time
                # Prevents lower-ID bots from following (they process first
                # in the game engine and would see the departing bot still there)
                self.reserve(x0, y0, t1)

    def reserve_stay(self, x: int, y: int, t_start: int, t_end: int):
        for t in range(t_start, min(t_end + 1, NUM_ROUNDS + 1)):
            self.reserve(x, y, t)

    def unreserve_stay(self, x: int, y: int, t_start: int, t_end: int):
        for t in range(t_start, min(t_end + 1, NUM_ROUNDS + 1)):
            self._reserved.discard((x, y, t))


def spacetime_astar(start: tuple[int, int], goal: tuple[int, int],
                    start_t: int, walkable: set, res_table: ReservationTable,
                    tables: PrecomputedTables,
                    spawn: tuple[int, int],
                    max_t: int = NUM_ROUNDS) -> Optional[list[tuple[int, int, int]]]:
    """A* in (x, y, t) space avoiding reserved cells.
    Spawn exempt from collision checks.
    """
    if start_t >= max_t:
        return None

    sx, sy = start
    gx, gy = goal

    if (sx, sy) == (gx, gy):
        return [(sx, sy, start_t)]

    h0 = tables.get_distance(start, goal)
    open_set = [(start_t + h0, start_t, sx, sy)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    best_g: dict[tuple[int, int, int], int] = {(sx, sy, start_t): start_t}

    while open_set:
        f, t, x, y = heapq.heappop(open_set)

        if (x, y) == (gx, gy):
            path = [(x, y, t)]
            key = (x, y, t)
            while key in came_from:
                key = came_from[key]
                path.append(key)
            return list(reversed(path))

        if t >= max_t - 1:
            continue

        state = (x, y, t)
        if best_g.get(state, 9999) < t:
            continue

        nt = t + 1

        neighbors = [(x, y)]  # wait
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + ddx, y + ddy
            if (nx, ny) in walkable:
                neighbors.append((nx, ny))

        for nx, ny in neighbors:
            is_spawn = (nx, ny) == spawn
            if not is_spawn and not res_table.is_free(nx, ny, nt):
                continue
            # Edge conflict: block reverse-direction swaps
            if (nx, ny) != (x, y) and not res_table.is_edge_free(x, y, nx, ny, t):
                continue

            ns = (nx, ny, nt)
            if ns in best_g and best_g[ns] <= nt:
                continue

            best_g[ns] = nt
            h = tables.get_distance((nx, ny), goal)
            heapq.heappush(open_set, (nt + h, nt, nx, ny))
            came_from[ns] = state

    return None


def path_to_actions(path: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Convert (x,y,t) path to list of (t, action) tuples."""
    actions = []
    for i in range(1, len(path)):
        x0, y0, _ = path[i - 1]
        x1, y1, t1 = path[i]
        if x1 > x0:
            actions.append((t1, ACT_MOVE_RIGHT))
        elif x1 < x0:
            actions.append((t1, ACT_MOVE_LEFT))
        elif y1 > y0:
            actions.append((t1, ACT_MOVE_DOWN))
        elif y1 < y0:
            actions.append((t1, ACT_MOVE_UP))
        else:
            actions.append((t1, ACT_WAIT))
    return actions


class TemporalPlanner:
    """Plans collision-free task schedules for all bots."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 all_orders: list[Order]):
        self.ms = ms
        self.tables = tables
        self.all_orders = all_orders
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = (int(ms.spawn[0]), int(ms.spawn[1]))  # Must be tuple

        # Item locations: type_id -> [(item_idx, adj_pos)]
        self.type_items: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj_list = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            for a in adj_list:
                self.type_items[tid].append((idx, (int(a[0]), int(a[1]))))

    def _nearest_dropoff(self, pos):
        return min(self.drop_zones,
                   key=lambda dz: self.tables.get_distance(pos, dz))

    def plan(self, num_rounds=NUM_ROUNDS, num_bots=NUM_BOTS,
             verbose=False) -> list[list[tuple[int, int]]]:
        """Plan all actions. Returns action_log[round][bot] = (act, item_idx)."""
        res = ReservationTable()
        action_log = [[(ACT_WAIT, -1)] * num_bots for _ in range(num_rounds)]

        bot_pos = {bid: self.spawn for bid in range(num_bots)}
        bot_avail = {bid: 0 for bid in range(num_bots)}
        # Track parking: bid -> (pos, t_start, t_end) for unreserving
        bot_park: dict[int, tuple[tuple[int, int], int, int]] = {}

        order_ready_time = 0
        orders_planned = 0
        total_items_planned = 0
        order_idx = 0

        while order_idx < len(self.all_orders) and order_ready_time < num_rounds - 10:
            order = self.all_orders[order_idx]
            needs = [int(t) for t in order.required]
            if not needs:
                order_idx += 1
                continue

            # Group into trips of up to INV_CAP items
            trips = []
            for i in range(0, len(needs), INV_CAP):
                trips.append(needs[i:i + INV_CAP])

            trip_finish_times = []
            for trip_types in trips:
                ft = self._assign_and_plan(
                    trip_types, bot_pos, bot_avail, bot_park,
                    res, action_log, num_rounds, order_ready_time,
                    mode='deliver')
                if ft is not None:
                    trip_finish_times.append(ft)
                    total_items_planned += len(trip_types)

            if trip_finish_times:
                active_complete_t = max(trip_finish_times)

                # --- Pipeline: stage preview items at dropoffs ---
                preview_idx = order_idx + 1
                staged_items = 0
                if preview_idx < len(self.all_orders):
                    preview_order = self.all_orders[preview_idx]
                    preview_needs = [int(t) for t in preview_order.required]
                    preview_trips = []
                    for i in range(0, len(preview_needs), INV_CAP):
                        preview_trips.append(preview_needs[i:i + INV_CAP])

                    for pt_types in preview_trips:
                        ft = self._assign_and_plan(
                            pt_types, bot_pos, bot_avail, bot_park,
                            res, action_log, num_rounds, order_ready_time,
                            mode='stage', arrive_by=active_complete_t)
                        if ft is not None:
                            staged_items += len(pt_types)
                            total_items_planned += len(pt_types)

                    # If ALL preview items staged, chain will complete it
                    if staged_items >= len(preview_needs):
                        order_idx += 1  # Skip preview order
                        if verbose:
                            print(f"  Order {preview_idx}: fully staged ({staged_items} items)")

                order_ready_time = active_complete_t + 1
                orders_planned += 1
                if verbose:
                    print(f"  Order {order_idx}: done t={active_complete_t} "
                          f"(staged={staged_items})")
            else:
                order_ready_time += 20

            order_idx += 1

        if verbose:
            print(f"\n  Plan: {orders_planned} orders, {total_items_planned} items, "
                  f"t_end={order_ready_time}")
        return action_log

    def _assign_and_plan(self, trip_types, bot_pos, bot_avail, bot_park,
                         res, action_log, num_rounds, order_ready_time,
                         mode='deliver', arrive_by=None):
        """Find best bot and plan a batched trip. Returns finish_t or None."""
        candidates = []
        for bid in range(len(bot_pos)):
            avail_t = max(bot_avail[bid], order_ready_time)
            if avail_t >= num_rounds - 10:
                continue
            bpos = bot_pos[bid]
            # Estimate: distance to nearest item of first type
            est = avail_t
            if trip_types:
                items = self.type_items.get(trip_types[0], [])
                if items:
                    est += min(self.tables.get_distance(bpos, adj)
                               for _, adj in items)
            candidates.append((est, -bid, bid))  # prefer higher IDs (planned first)

        candidates.sort()
        for _, _, bid in candidates:
            avail_t = max(bot_avail[bid], order_ready_time)
            result = self._plan_trip(
                bid, trip_types, bot_pos[bid], avail_t,
                res, action_log, num_rounds, bot_pos, bot_avail, bot_park,
                mode=mode, arrive_by=arrive_by)
            if result is not None:
                return result
        return None

    def _plan_trip(self, bot_id, trip_types, start_pos, start_t,
                   res, action_log, num_rounds, bot_pos, bot_avail, bot_park,
                   mode='deliver', arrive_by=None):
        """Plan multi-pickup trip. Returns finish_t or None."""
        # Unreserve old parking
        if bot_id in bot_park:
            old_pos, old_ts, old_te = bot_park[bot_id]
            if old_pos != self.spawn:
                res.unreserve_stay(old_pos[0], old_pos[1], old_ts, old_te)
            del bot_park[bot_id]

        current_pos = start_pos
        current_t = start_t

        # Nearest-neighbor routing for item pickups
        pickups = []
        remaining = list(trip_types)
        route_pos = current_pos

        while remaining:
            best_dist = 999999
            best_type = None
            best_item_idx = None
            best_adj = None

            for tid in remaining:
                for item_idx, adj_pos in self.type_items.get(tid, []):
                    d = self.tables.get_distance(route_pos, adj_pos)
                    if d < best_dist:
                        best_dist = d
                        best_type = tid
                        best_item_idx = item_idx
                        best_adj = adj_pos

            if best_adj is None:
                break
            remaining.remove(best_type)
            pickups.append((best_item_idx, best_adj))
            route_pos = best_adj

        if not pickups:
            return None

        # Plan collision-free path for each pickup stop
        for item_idx, adj_pos in pickups:
            path = spacetime_astar(
                current_pos, adj_pos, current_t,
                self.walkable, res, self.tables, self.spawn,
                max_t=num_rounds)
            if path is None:
                return None

            arrive_t = path[-1][2]
            pickup_t = arrive_t + 1  # Pickup AFTER arriving

            if pickup_t >= num_rounds:
                return None

            px, py = adj_pos
            # Wait if position busy at pickup time
            if not res.is_free(px, py, pickup_t):
                found = False
                for wt in range(pickup_t, min(pickup_t + 20, num_rounds)):
                    if res.is_free(px, py, wt):
                        pickup_t = wt
                        found = True
                        break
                if not found:
                    return None

            # Commit
            res.reserve_path(path)
            for wt in range(arrive_t + 1, pickup_t + 1):
                res.reserve(px, py, wt)

            for t, act in path_to_actions(path):
                if 0 <= t < num_rounds:
                    action_log[t][bot_id] = (act, -1)

            if 0 <= pickup_t < num_rounds:
                action_log[pickup_t][bot_id] = (ACT_PICKUP, item_idx)

            current_pos = adj_pos
            current_t = pickup_t + 1

        # Path to dropoff
        dropoff = self._nearest_dropoff(current_pos)
        path_to_drop = spacetime_astar(
            current_pos, dropoff, current_t,
            self.walkable, res, self.tables, self.spawn,
            max_t=num_rounds)
        if path_to_drop is None:
            return None

        arrive_drop_t = path_to_drop[-1][2]
        dx, dy = dropoff

        if mode == 'deliver':
            # Dropoff one round AFTER arriving
            drop_act_t = arrive_drop_t + 1
            if drop_act_t >= num_rounds:
                return None

            # Wait for dropoff cell to be free
            if not res.is_free(dx, dy, drop_act_t):
                found = False
                for wt in range(drop_act_t, min(drop_act_t + 20, num_rounds)):
                    if res.is_free(dx, dy, wt):
                        drop_act_t = wt
                        found = True
                        break
                if not found:
                    return None

            # Commit path + dropoff
            res.reserve_path(path_to_drop)
            for wt in range(arrive_drop_t + 1, drop_act_t + 1):
                res.reserve(dx, dy, wt)

            for t, act in path_to_actions(path_to_drop):
                if 0 <= t < num_rounds:
                    action_log[t][bot_id] = (act, -1)

            if 0 <= drop_act_t < num_rounds:
                action_log[drop_act_t][bot_id] = (ACT_DROPOFF, -1)

            finish_t = drop_act_t
            park_start = drop_act_t + 1

        else:
            # Stage mode: arrive at dropoff and wait (chain will deliver)
            # Need to arrive BEFORE arrive_by
            if arrive_by is not None and arrive_drop_t > arrive_by:
                return None  # Too late

            res.reserve_path(path_to_drop)
            for t, act in path_to_actions(path_to_drop):
                if 0 <= t < num_rounds:
                    action_log[t][bot_id] = (act, -1)

            # Reserve dropoff cell while staging
            stage_end = arrive_by + 3 if arrive_by else arrive_drop_t + 30
            stage_end = min(stage_end, num_rounds)
            for wt in range(arrive_drop_t + 1, stage_end + 1):
                res.reserve(dx, dy, wt)

            finish_t = arrive_drop_t
            park_start = stage_end + 1

        # Park the bot after trip (move off dropoff to nearby cell)
        self._park_bot(bot_id, dropoff, park_start, res, action_log,
                       num_rounds, bot_pos, bot_avail, bot_park)
        return finish_t

    def _park_bot(self, bot_id, from_pos, start_t, res, action_log,
                  num_rounds, bot_pos, bot_avail, bot_park):
        """Move bot away from dropoff to a parking cell."""
        if start_t >= num_rounds:
            bot_pos[bot_id] = from_pos
            bot_avail[bot_id] = start_t
            return

        # Find nearby parking (non-dropoff walkable cell)
        fx, fy = from_pos
        park_pos = None
        best_d = 999
        for ddx in range(-3, 4):
            for ddy in range(-3, 4):
                px, py = fx + ddx, fy + ddy
                if (px, py) in self.walkable and (px, py) not in self.drop_set:
                    d = abs(ddx) + abs(ddy)
                    if 0 < d < best_d and res.is_free(px, py, start_t):
                        best_d = d
                        park_pos = (px, py)

        if park_pos is None:
            # Fallback to spawn
            park_pos = self.spawn

        if park_pos == from_pos or park_pos == self.spawn:
            bot_pos[bot_id] = park_pos if park_pos == self.spawn else from_pos
            bot_avail[bot_id] = start_t
            return

        park_path = spacetime_astar(
            from_pos, park_pos, start_t,
            self.walkable, res, self.tables, self.spawn,
            max_t=min(start_t + 10, num_rounds))

        if park_path and len(park_path) > 1:
            res.reserve_path(park_path)
            for t, act in path_to_actions(park_path):
                if 0 <= t < num_rounds:
                    action_log[t][bot_id] = (act, -1)
            final_pos = (park_path[-1][0], park_path[-1][1])
            final_t = park_path[-1][2]
            # Reserve parking for idle time
            park_end = min(final_t + 50, num_rounds)
            if final_pos != self.spawn:
                res.reserve_stay(final_pos[0], final_pos[1],
                                 final_t + 1, park_end)
                bot_park[bot_id] = (final_pos, final_t + 1, park_end)
            bot_pos[bot_id] = final_pos
            bot_avail[bot_id] = final_t + 1
        else:
            bot_pos[bot_id] = from_pos
            bot_avail[bot_id] = start_t


class TemporalSolver:
    """Nightmare solver using pre-computed temporal plans."""

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)

        t_plan_start = time.time()
        planner = TemporalPlanner(ms, tables, all_orders)
        action_log = planner.plan(verbose=verbose)
        plan_time = time.time() - t_plan_start
        if verbose:
            print(f"  Planning time: {plan_time:.2f}s")

        drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
        chains, max_chain = 0, 0
        t0 = time.time()
        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = action_log[rnd]
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}"
                         if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            dead = sum(1 for b in range(len(state.bot_positions))
                       if state.bot_inv_list(b))
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"DeadBots={dead} "
                  f"PlanTime={plan_time:.2f}s "
                  f"SimTime={elapsed:.1f}s")
        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_tp, scores_v4 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - Temporal Planner V2")
        print(f"{'='*50}")
        score, _ = TemporalSolver.run_sim(seed, verbose=args.verbose)
        scores_tp.append(score)

        if args.compare:
            from nightmare_lmapf_solver import LMAPFSolver
            print(f"\n--- V4 ---")
            s4, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
            scores_v4.append(s4)
            print(f"\nTemporal={score} vs V4={s4} (delta={score - s4:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nTemporal: mean={statistics.mean(scores_tp):.1f} "
              f"max={max(scores_tp)} min={min(scores_tp)}")
        if scores_v4:
            print(f"V4: mean={statistics.mean(scores_v4):.1f}")

"""Global offline planner for multi-bot grocery game.

Plans all bots' actions using full order knowledge and space-time A*
pathfinding for collision-free multi-agent coordination.

Key idea: since the game is deterministic per day, plan the ENTIRE game
offline using captured order data. This is fundamentally different from
reactive solvers (CascadeSolver) or sequential DP (gpu_sequential_solver).

Architecture:
  1. Order decomposition: orders -> pickup-delivery trips
  2. Task assignment: items -> bots (min-cost Hungarian matching)
  3. Route planning: space-time A* for collision-free paths
  4. DZ scheduling: coordinate dropoff arrivals (max 1 bot per DZ tile)
  5. Cascade preparation: stage prefetch bots at DZ before order completion
  6. Multi-restart: perturb assignments, keep best

Usage:
    python global_planner.py --seeds 7005 --difficulty nightmare -v
    python global_planner.py --seeds 42 --difficulty hard -v
    python global_planner.py --seeds 1000-1009 --difficulty nightmare
"""
from __future__ import annotations

import heapq
import random as _random
import sys
import time
from collections import defaultdict
from itertools import permutations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from game_engine import (
    init_game, init_game_from_capture, step as cpu_step,
    GameState, MapState, Order, CaptureData,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import CONFIGS, DIFF_ROUNDS, parse_seeds
from precompute import PrecomputedTables


# ---------------------------------------------------------------------------
# Reservation table for multi-agent collision avoidance
# ---------------------------------------------------------------------------

class ReservationTable:
    """Track cell occupancy at each timestep for MAPF collision avoidance.

    Spawn is always free (multiple bots can stack there).
    """

    __slots__ = ['spawn', '_res']

    def __init__(self, spawn: tuple[int, int]):
        self.spawn = spawn
        self._res: dict[tuple[int, int, int], int] = {}  # (x, y, t) -> bot_id

    def reserve(self, x: int, y: int, t: int, bot_id: int):
        self._res[(x, y, t)] = bot_id

    def is_free(self, x: int, y: int, t: int, bot_id: int = -1) -> bool:
        if (x, y) == self.spawn:
            return True
        occ = self._res.get((x, y, t))
        return occ is None or occ == bot_id

    def clear_bot_from(self, bot_id: int, from_t: int):
        """Remove all reservations for bot_id at time >= from_t."""
        to_del = [k for k, v in self._res.items()
                  if v == bot_id and k[2] >= from_t]
        for k in to_del:
            del self._res[k]


# ---------------------------------------------------------------------------
# Global Planner
# ---------------------------------------------------------------------------

class GlobalPlanner:
    """Offline planner using task allocation + space-time A* MAPF.

    Plans all bots' actions for the full game using known order sequences.
    Works for any difficulty (1-20 bots).
    """

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 num_bots: int, num_rounds: int = 500,
                 verbose: bool = False, rng_seed: int = 0):
        self.ms = map_state
        self.all_orders = all_orders
        self.num_bots = num_bots
        self.num_rounds = num_rounds
        self.verbose = verbose
        self.rng = _random.Random(rng_seed)

        # Precomputed tables (BFS shortest paths)
        self.tables = PrecomputedTables.get(map_state)

        # Item lookup by type: type_id -> [(item_idx, shelf_pos, [adj_positions])]
        self.type_to_items: dict[int, list[tuple[int, tuple, list]]] = defaultdict(list)
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            pos = (int(map_state.item_positions[idx, 0]),
                   int(map_state.item_positions[idx, 1]))
            adj = [(int(a[0]), int(a[1]))
                   for a in map_state.item_adjacencies.get(idx, [])]
            self.type_to_items[tid].append((idx, pos, adj))

        # DZ positions
        self.dz_list = [tuple(int(c) for c in dz) for dz in map_state.drop_off_zones]
        self.dz_set = set(self.dz_list)
        self.spawn = (int(map_state.spawn[0]), int(map_state.spawn[1]))

        # Build walkable set and neighbor lists
        self._walkable: set[tuple[int, int]] = set()
        for y in range(map_state.height):
            for x in range(map_state.width):
                if map_state.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    self._walkable.add((x, y))

        self._neighbors: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for pos in self._walkable:
            nbrs = []
            for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in self._walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _dist(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """BFS shortest path distance (from precomputed tables)."""
        ai = self.tables.pos_to_idx.get(a)
        bi = self.tables.pos_to_idx.get(b)
        if ai is None or bi is None:
            return 9999
        return int(self.tables.dist_matrix[ai, bi])

    def _nearest_dz(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Nearest dropoff zone by BFS distance."""
        best = self.dz_list[0]
        best_d = self._dist(pos, best)
        for dz in self.dz_list[1:]:
            d = self._dist(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _find_nearest_item(self, type_id: int, from_pos: tuple[int, int]
                           ) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
        """Find nearest item of type. Returns (item_idx, shelf_pos, best_adj) or None."""
        candidates = self.type_to_items.get(type_id, [])
        if not candidates:
            return None
        best = None
        best_d = 9999
        for item_idx, shelf_pos, adj_list in candidates:
            for adj in adj_list:
                d = self._dist(from_pos, adj)
                if d < best_d:
                    best_d = d
                    best = (item_idx, shelf_pos, adj)
        return best

    def _find_nearest_item_avoiding(self, type_id: int, from_pos: tuple[int, int],
                                    avoid_adj: set[tuple[int, int]]
                                    ) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
        """Like _find_nearest_item but avoids specified adjacent positions."""
        candidates = self.type_to_items.get(type_id, [])
        if not candidates:
            return None
        best = None
        best_d = 9999
        for item_idx, shelf_pos, adj_list in candidates:
            for adj in adj_list:
                if adj in avoid_adj:
                    continue
                d = self._dist(from_pos, adj)
                if d < best_d:
                    best_d = d
                    best = (item_idx, shelf_pos, adj)
        # Fallback to any item if all avoided
        if best is None:
            return self._find_nearest_item(type_id, from_pos)
        return best

    # ------------------------------------------------------------------
    # Space-Time A* pathfinder
    # ------------------------------------------------------------------

    def _astar(self, start_pos: tuple[int, int], goal_pos: tuple[int, int],
               start_t: int, res: ReservationTable, bot_id: int,
               max_extra_wait: int = 40
               ) -> tuple[list[tuple[int, int]], int] | None:
        """Space-time A* from start_pos at time start_t to goal_pos.

        Returns (actions, end_t) where:
          actions: list of (action_type, -1) for each timestep
          end_t: the round when goal is reached
        Returns None if no path found.
        """
        if start_pos == goal_pos:
            return [], start_t

        gi = self.tables.pos_to_idx.get(goal_pos)
        si = self.tables.pos_to_idx.get(start_pos)
        if gi is None or si is None:
            return None

        # Heuristic: precomputed BFS distance from goal to all cells
        h_from_goal = self.tables.dist_matrix[gi]
        optimal_dist = int(h_from_goal[si])
        max_t = min(start_t + optimal_dist + max_extra_wait, self.num_rounds)

        start_state = (start_pos[0], start_pos[1], start_t)

        counter = 0
        pq = [(optimal_dist, counter, start_state)]
        counter += 1

        g_score = {start_state: 0}
        came_from: dict[tuple, tuple[tuple, int]] = {}

        while pq:
            f, _, state = heapq.heappop(pq)
            x, y, t = state

            if (x, y) == goal_pos:
                # Reconstruct path
                actions = []
                s = state
                while s in came_from:
                    prev_s, act = came_from[s]
                    actions.append((act, -1))
                    s = prev_s
                actions.reverse()
                return actions, t

            if g_score.get(state, 9999) < f - int(h_from_goal[self.tables.pos_to_idx.get((x, y), 0)]):
                continue  # outdated

            if t >= max_t:
                continue

            g = g_score[state]

            # Try wait + 4 moves
            candidates = [(ACT_WAIT, (x, y))]
            candidates.extend(self._neighbors.get((x, y), []))

            for act, (nx, ny) in candidates:
                nt = t + 1
                if not res.is_free(nx, ny, nt, bot_id):
                    continue
                # For moves: also check destination at source time.
                # The game engine checks target occupancy BEFORE the move,
                # and higher-ID bots haven't moved yet. If a lower-ID bot
                # reserved this cell at time t (via move-target reservation),
                # we must avoid it.
                if act != ACT_WAIT and not res.is_free(nx, ny, t, bot_id):
                    continue

                new_state = (nx, ny, nt)
                new_g = g + 1

                if new_state in g_score and g_score[new_state] <= new_g:
                    continue

                ci = self.tables.pos_to_idx.get((nx, ny))
                if ci is None:
                    continue

                g_score[new_state] = new_g
                h = int(h_from_goal[ci])
                came_from[new_state] = (state, act)
                heapq.heappush(pq, (new_g + h, counter, new_state))
                counter += 1

        return None  # no path found

    # ------------------------------------------------------------------
    # Trip planning
    # ------------------------------------------------------------------

    def _plan_trip(self, bot_id: int, item_types: list[int],
                   bot_pos: tuple[int, int], start_t: int,
                   res: ReservationTable,
                   deliver: bool = True,
                   target_dz: tuple[int, int] | None = None
                   ) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
        """Plan a pickup-delivery trip for one bot.

        Args:
            bot_id: Bot index.
            item_types: List of type_ids to pick up (1-3 items).
            bot_pos: Current bot position.
            start_t: Round when bot is available.
            res: Reservation table.
            deliver: If True, go to DZ and dropoff after pickups.
            target_dz: Specific DZ tile to use (None = nearest).

        Returns (actions, end_pos, end_t).
        """
        all_actions: list[tuple[int, int]] = []
        pos = bot_pos
        t = start_t

        # Find items for each type
        items_info: list[tuple[int, int, tuple, tuple]] = []  # (tid, item_idx, shelf, adj)
        used_adj: set[tuple[int, int]] = set()
        for tid in item_types:
            info = self._find_nearest_item_avoiding(tid, pos, used_adj)
            if info is None:
                continue
            item_idx, shelf_pos, adj_pos = info
            items_info.append((tid, item_idx, shelf_pos, adj_pos))
            used_adj.add(adj_pos)

        if not items_info:
            return all_actions, pos, t

        # TSP: find best visit order (max 6 permutations for 3 items)
        best_perm = list(range(len(items_info)))
        if len(items_info) > 1:
            best_cost = float('inf')
            for perm in permutations(range(len(items_info))):
                cost = 0
                p = pos
                for idx in perm:
                    adj = items_info[idx][3]
                    cost += self._dist(p, adj) + 1
                    p = adj
                if deliver:
                    dz = target_dz or self._nearest_dz(p)
                    cost += self._dist(p, dz) + 1
                if cost < best_cost:
                    best_cost = cost
                    best_perm = list(perm)

        # Execute pickups in order
        for perm_idx in best_perm:
            tid, item_idx, shelf_pos, adj_pos = items_info[perm_idx]

            # Re-find nearest from current position
            info = self._find_nearest_item(tid, pos)
            if info:
                item_idx, shelf_pos, adj_pos = info

            # Navigate to adjacent cell
            result = self._astar(pos, adj_pos, t, res, bot_id)
            if result is None:
                continue

            path_actions, end_t = result

            # Reserve and record path
            p = pos
            for i, (act, _) in enumerate(path_actions):
                if act == ACT_WAIT:
                    np_ = p
                else:
                    np_ = (p[0] + DX[act], p[1] + DY[act])
                    # Reserve destination at source time (prevents higher-ID
                    # bots from blocking this move in sequential execution)
                    res.reserve(np_[0], np_[1], t + i, bot_id)
                res.reserve(np_[0], np_[1], t + i + 1, bot_id)
                p = np_
            all_actions.extend(path_actions)

            pos = adj_pos
            t = end_t

            # Pickup action
            all_actions.append((ACT_PICKUP, item_idx))
            t += 1
            res.reserve(pos[0], pos[1], t, bot_id)

        if not deliver:
            return all_actions, pos, t

        # Navigate to DZ (try all, pick fastest)
        dz = target_dz or self._nearest_dz(pos)
        best_dz = dz
        best_dz_result = None
        best_dz_t = 99999

        for dz_candidate in self.dz_list:
            result = self._astar(pos, dz_candidate, t, res, bot_id)
            if result and result[1] < best_dz_t:
                best_dz_t = result[1]
                best_dz_result = result
                best_dz = dz_candidate

        if best_dz_result is None:
            return all_actions, pos, t

        path_actions, end_t = best_dz_result

        # Reserve DZ path
        p = pos
        for i, (act, _) in enumerate(path_actions):
            if act == ACT_WAIT:
                np_ = p
            else:
                np_ = (p[0] + DX[act], p[1] + DY[act])
                res.reserve(np_[0], np_[1], t + i, bot_id)
            res.reserve(np_[0], np_[1], t + i + 1, bot_id)
            p = np_
        all_actions.extend(path_actions)

        pos = best_dz
        t = end_t

        # Dropoff action
        all_actions.append((ACT_DROPOFF, -1))
        t += 1
        res.reserve(pos[0], pos[1], t, bot_id)

        return all_actions, pos, t

    # ------------------------------------------------------------------
    # Item-to-bot assignment
    # ------------------------------------------------------------------

    def _assign_trips(self, trips: list[list[int]],
                      bot_pos: list[tuple[int, int]],
                      bot_free_at: list[int],
                      current_min_t: int
                      ) -> list[tuple[int, int, list[int]]]:
        """Assign trips to bots using Hungarian algorithm.

        Returns list of (bot_id, trip_idx, trip_items).
        """
        n_trips = len(trips)
        n_bots = self.num_bots

        # Build cost matrix [n_bots, n_trips]
        cost = np.full((n_bots, n_trips), 1e6, dtype=np.float64)

        for bi in range(n_bots):
            for ti in range(n_trips):
                trip = trips[ti]
                # Estimate trip cost: travel to items + travel to DZ
                wait_cost = max(0, bot_free_at[bi] - current_min_t)
                p = bot_pos[bi]
                travel = 0
                for tid in trip:
                    info = self._find_nearest_item(tid, p)
                    if info:
                        _, _, adj = info
                        travel += self._dist(p, adj) + 1
                        p = adj
                dz = self._nearest_dz(p)
                travel += self._dist(p, dz) + 1
                cost[bi, ti] = travel + wait_cost

        # Solve assignment (may have more bots than trips)
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except ValueError:
            return []

        assignments = []
        for bi, ti in zip(row_ind, col_ind):
            if cost[bi, ti] < 1e5:  # filter out infeasible
                assignments.append((bi, ti, trips[ti]))

        return assignments

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def plan(self) -> list[list[tuple[int, int]]]:
        """Generate complete action plan for all bots.

        Order-sequential: plan delivery for one order at a time, ensuring
        bots deliver to the ACTUAL active order. After each order's trips
        complete, the order transitions and we plan the next one.

        Returns list of num_rounds round_actions, each [(act, item)] * num_bots.
        """
        bot_pos = [self.spawn] * self.num_bots
        bot_free_at = [0] * self.num_bots
        bot_actions: list[list[tuple[int, int]]] = [[] for _ in range(self.num_bots)]

        res = ReservationTable(self.spawn)

        # Reserve spawn for all bots at t=0
        for bid in range(self.num_bots):
            res.reserve(self.spawn[0], self.spawn[1], 0, bid)

        order_idx = 0
        orders_planned = 0
        t0 = time.time()

        while order_idx < len(self.all_orders):
            order = self.all_orders[order_idx]
            items_needed = [int(t) for t in order.required]

            if not items_needed:
                order_idx += 1
                continue

            # Earliest any bot is free
            current_min_t = min(bot_free_at)
            if current_min_t >= self.num_rounds - 10:
                break

            # Create trips (groups of up to INV_CAP items)
            trips = []
            for i in range(0, len(items_needed), INV_CAP):
                trips.append(items_needed[i:i + INV_CAP])

            # Assign trips to bots
            assignments = self._assign_trips(trips, bot_pos, bot_free_at, current_min_t)
            if not assignments:
                order_idx += 1
                continue

            # Plan each assigned trip (sorted by earliest availability)
            assignments.sort(key=lambda a: bot_free_at[a[0]])
            latest_delivery_t = 0
            assigned_bots = set()

            for bot_id, trip_idx, trip_items in assignments:
                # Pad with waits until bot is free
                while len(bot_actions[bot_id]) < bot_free_at[bot_id]:
                    bot_actions[bot_id].append((ACT_WAIT, -1))

                trip_acts, end_pos, end_t = self._plan_trip(
                    bot_id, trip_items, bot_pos[bot_id],
                    bot_free_at[bot_id], res)

                bot_actions[bot_id].extend(trip_acts)
                bot_pos[bot_id] = end_pos
                bot_free_at[bot_id] = len(bot_actions[bot_id])
                latest_delivery_t = max(latest_delivery_t, bot_free_at[bot_id])
                assigned_bots.add(bot_id)

            if latest_delivery_t >= self.num_rounds:
                break

            # Synchronize: next order can only be delivered AFTER this
            # order's delivery completes. Ensure all bots are free at or
            # after the latest delivery time for this order.
            for bid in range(self.num_bots):
                if bot_free_at[bid] < latest_delivery_t:
                    # This bot should wait until current order is delivered
                    bot_free_at[bid] = latest_delivery_t

            # After delivery, move assigned bots AWAY from DZ to free it
            for bid in assigned_bots:
                if bot_pos[bid] in self.dz_set:
                    # Move one step away from DZ
                    while len(bot_actions[bid]) < bot_free_at[bid]:
                        bot_actions[bid].append((ACT_WAIT, -1))
                    for act, (nx, ny) in self._neighbors.get(bot_pos[bid], []):
                        if (nx, ny) not in self.dz_set:
                            nt = bot_free_at[bid] + 1
                            if res.is_free(nx, ny, nt, bid):
                                bot_actions[bid].append((act, -1))
                                res.reserve(nx, ny, nt, bid)
                                res.reserve(nx, ny, bot_free_at[bid], bid)
                                bot_pos[bid] = (nx, ny)
                                bot_free_at[bid] = len(bot_actions[bid])
                                break

            orders_planned += 1
            order_idx += 1

            if self.verbose and orders_planned % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Orders {orders_planned}: latest_t={latest_delivery_t}, "
                      f"elapsed={elapsed:.1f}s", file=sys.stderr)

            if time.time() - t0 > 120:
                if self.verbose:
                    print(f"  Planning time budget (120s) at order {orders_planned}",
                          file=sys.stderr)
                break

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  Planned {orders_planned} orders in {elapsed:.1f}s, "
                  f"latest_t={max(bot_free_at)}", file=sys.stderr)

        return self._to_combined(bot_actions)

    def plan_with_prefetch(self) -> list[list[tuple[int, int]]]:
        """Plan with cascade-aware prefetching.

        For each order batch, split bots into:
          - Delivery team: deliver active order items
          - Prefetch team: pick up next order items, stage at DZ for cascade
        """
        bot_pos = [self.spawn] * self.num_bots
        bot_free_at = [0] * self.num_bots
        bot_actions: list[list[tuple[int, int]]] = [[] for _ in range(self.num_bots)]

        res = ReservationTable(self.spawn)
        for bid in range(self.num_bots):
            res.reserve(self.spawn[0], self.spawn[1], 0, bid)

        order_idx = 0
        orders_planned = 0
        t0 = time.time()

        while order_idx < len(self.all_orders) - 1:
            current_min_t = min(bot_free_at)
            if current_min_t >= self.num_rounds - 10:
                break

            active_order = self.all_orders[order_idx]
            preview_order = self.all_orders[order_idx + 1] if order_idx + 1 < len(self.all_orders) else None

            active_items = [int(t) for t in active_order.required]
            preview_items = [int(t) for t in preview_order.required] if preview_order else []

            # Create trips for active order
            active_trips = []
            for i in range(0, len(active_items), INV_CAP):
                active_trips.append(active_items[i:i + INV_CAP])

            # Create trips for preview (prefetch - will stage at DZ for cascade)
            preview_trips = []
            for i in range(0, len(preview_items), INV_CAP):
                preview_trips.append(preview_items[i:i + INV_CAP])

            # Assign active trips to nearest bots
            all_trips = active_trips + preview_trips
            is_prefetch = [False] * len(active_trips) + [True] * len(preview_trips)

            # Build cost matrix
            n_trips = len(all_trips)
            cost = np.full((self.num_bots, n_trips), 1e6, dtype=np.float64)
            for bi in range(self.num_bots):
                for ti in range(n_trips):
                    trip = all_trips[ti]
                    wait_cost = max(0, bot_free_at[bi] - current_min_t)
                    p = bot_pos[bi]
                    travel = 0
                    for tid in trip:
                        info = self._find_nearest_item(tid, p)
                        if info:
                            _, _, adj = info
                            travel += self._dist(p, adj) + 1
                            p = adj
                    dz = self._nearest_dz(p)
                    travel += self._dist(p, dz) + 1
                    # Prefetch trips get slight penalty to prioritize active delivery
                    penalty = 5 if is_prefetch[ti] else 0
                    cost[bi, ti] = travel + wait_cost + penalty

            try:
                row_ind, col_ind = linear_sum_assignment(cost)
            except ValueError:
                break

            # Plan assigned trips
            latest_active_t = 0
            for bi, ti in zip(row_ind, col_ind):
                if cost[bi, ti] >= 1e5:
                    continue

                bid = bi
                trip_items = all_trips[ti]

                while len(bot_actions[bid]) < bot_free_at[bid]:
                    bot_actions[bid].append((ACT_WAIT, -1))

                # Prefetch bots still deliver (items go to active order via
                # game engine; if active order doesn't need them, they stay
                # in inventory for cascade auto-delivery when active completes)
                trip_acts, end_pos, end_t = self._plan_trip(
                    bid, trip_items, bot_pos[bid],
                    bot_free_at[bid], res, deliver=True)

                bot_actions[bid].extend(trip_acts)
                bot_pos[bid] = end_pos
                bot_free_at[bid] = len(bot_actions[bid])

                if not is_prefetch[ti]:
                    latest_active_t = max(latest_active_t, bot_free_at[bid])

            if max(bot_free_at) >= self.num_rounds:
                break

            # After active order, preview becomes active via cascade
            # We planned delivery for both, so advance by 1 (or 2 if cascade likely)
            order_idx += 1
            orders_planned += 1

            # If we planned prefetch too, skip the preview order
            if preview_trips and len(row_ind) > len(active_trips):
                # Check if ALL preview items were assigned
                preview_assigned = sum(1 for _, ti in zip(row_ind, col_ind)
                                       if ti < len(all_trips) and is_prefetch[ti]
                                       and cost[_, ti] < 1e5)
                if preview_assigned >= len(preview_trips):
                    order_idx += 1
                    orders_planned += 1

            if self.verbose and orders_planned % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Orders {orders_planned}: latest_t={max(bot_free_at)}, "
                      f"elapsed={elapsed:.1f}s", file=sys.stderr)

            if time.time() - t0 > 120:
                break

        if self.verbose:
            print(f"  Planned {orders_planned} orders (prefetch) in {time.time()-t0:.1f}s",
                  file=sys.stderr)

        return self._to_combined(bot_actions)

    def plan_fast_pipeline(self) -> list[list[tuple[int, int]]]:
        """Fast pipelined planning: maximize throughput with pre-positioning.

        Strategy:
        1. Assign bots to item types based on proximity (type specialists)
        2. Each bot runs pickup-deliver loops for its assigned types
        3. Bots pre-position near next items while waiting for order transitions
        """
        bot_pos = [self.spawn] * self.num_bots
        bot_free_at = [0] * self.num_bots
        bot_actions: list[list[tuple[int, int]]] = [[] for _ in range(self.num_bots)]

        res = ReservationTable(self.spawn)
        for bid in range(self.num_bots):
            res.reserve(self.spawn[0], self.spawn[1], 0, bid)

        order_idx = 0
        orders_planned = 0
        t0 = time.time()

        # Lookahead: plan delivery for current + pickup for next
        while order_idx < len(self.all_orders):
            current_min_t = min(bot_free_at)
            if current_min_t >= self.num_rounds - 5:
                break

            order = self.all_orders[order_idx]
            items_needed = [int(t) for t in order.required]

            if not items_needed:
                order_idx += 1
                continue

            # Smart trip creation: cluster items by geographic proximity
            trips = self._create_smart_trips(items_needed, bot_pos, bot_free_at)

            # Assign trips to bots
            assignments = self._assign_trips(trips, bot_pos, bot_free_at, current_min_t)
            if not assignments:
                order_idx += 1
                continue

            # Plan each trip
            for bot_id, trip_idx, trip_items in assignments:
                while len(bot_actions[bot_id]) < bot_free_at[bot_id]:
                    bot_actions[bot_id].append((ACT_WAIT, -1))

                trip_acts, end_pos, end_t = self._plan_trip(
                    bot_id, trip_items, bot_pos[bot_id],
                    bot_free_at[bot_id], res)

                bot_actions[bot_id].extend(trip_acts)
                bot_pos[bot_id] = end_pos
                bot_free_at[bot_id] = len(bot_actions[bot_id])

            # Pre-position idle bots toward next order items
            if order_idx + 1 < len(self.all_orders):
                next_items = [int(t) for t in self.all_orders[order_idx + 1].required]
                assigned_bots = {a[0] for a in assignments}

                for bid in range(self.num_bots):
                    if bid in assigned_bots:
                        continue
                    if bot_free_at[bid] > current_min_t + 5:
                        continue  # bot still busy

                    # Move toward nearest next-order item
                    for tid in next_items:
                        info = self._find_nearest_item(tid, bot_pos[bid])
                        if info and self._dist(bot_pos[bid], info[2]) > 2:
                            while len(bot_actions[bid]) < bot_free_at[bid]:
                                bot_actions[bid].append((ACT_WAIT, -1))

                            result = self._astar(
                                bot_pos[bid], info[2],
                                bot_free_at[bid], res, bid,
                                max_extra_wait=5)
                            if result:
                                path_acts, end_t = result
                                # Only pre-position for a few steps
                                limit = min(len(path_acts), 10)
                                p = bot_pos[bid]
                                for i in range(limit):
                                    act, _ = path_acts[i]
                                    if act == ACT_WAIT:
                                        np_ = p
                                    else:
                                        np_ = (p[0] + DX[act], p[1] + DY[act])
                                    res.reserve(np_[0], np_[1],
                                                bot_free_at[bid] + i + 1, bid)
                                    bot_actions[bid].append(path_acts[i])
                                    p = np_
                                bot_pos[bid] = p
                                bot_free_at[bid] = len(bot_actions[bid])
                            break  # one pre-position move per bot

            if max(bot_free_at) >= self.num_rounds:
                break

            orders_planned += 1
            order_idx += 1

            if self.verbose and orders_planned % 10 == 0:
                print(f"  Orders {orders_planned}: t={max(bot_free_at)}, "
                      f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)

            if time.time() - t0 > 120:
                break

        if self.verbose:
            print(f"  Planned {orders_planned} orders (pipeline) in "
                  f"{time.time()-t0:.1f}s", file=sys.stderr)

        return self._to_combined(bot_actions)

    # ------------------------------------------------------------------
    # Smart trip creation
    # ------------------------------------------------------------------

    def _create_smart_trips(self, items: list[int],
                            bot_pos: list[tuple[int, int]],
                            bot_free_at: list[int]) -> list[list[int]]:
        """Group items into trips of max 3, clustering by geographic proximity."""
        n = len(items)
        if n <= INV_CAP:
            return [items]

        # Get item positions
        item_positions = []
        for tid in items:
            info = self._find_nearest_item(tid, self.spawn)
            if info:
                item_positions.append(info[2])  # adj position
            else:
                item_positions.append(self.spawn)

        # Greedy clustering
        remaining = list(range(n))
        trips = []
        while remaining:
            if len(remaining) <= INV_CAP:
                trips.append([items[i] for i in remaining])
                break

            # Pick the item nearest to a DZ as seed
            best_seed = remaining[0]
            best_dz_dist = self._dist(item_positions[best_seed], self._nearest_dz(item_positions[best_seed]))
            for idx in remaining[1:]:
                d = self._dist(item_positions[idx], self._nearest_dz(item_positions[idx]))
                if d < best_dz_dist:
                    best_dz_dist = d
                    best_seed = idx

            # Add nearest items to seed
            cluster = [best_seed]
            remaining.remove(best_seed)

            while len(cluster) < INV_CAP and remaining:
                last_pos = item_positions[cluster[-1]]
                nearest = min(remaining, key=lambda i: self._dist(last_pos, item_positions[i]))
                cluster.append(nearest)
                remaining.remove(nearest)

            trips.append([items[i] for i in cluster])

        return trips

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    def _to_combined(self, bot_actions: list[list[tuple[int, int]]]
                     ) -> list[list[tuple[int, int]]]:
        """Convert per-bot action lists to per-round combined format."""
        combined = []
        for r in range(self.num_rounds):
            round_acts = []
            for bid in range(self.num_bots):
                if r < len(bot_actions[bid]):
                    round_acts.append(bot_actions[bid][r])
                else:
                    round_acts.append((ACT_WAIT, -1))
            combined.append(round_acts)
        return combined


# ---------------------------------------------------------------------------
# Multi-restart wrapper
# ---------------------------------------------------------------------------

def solve_global(map_state: MapState, all_orders: list[Order],
                 num_bots: int, num_rounds: int = 500,
                 restarts: int = 3, verbose: bool = False,
                 time_budget: float = 240.0
                 ) -> tuple[int, list[list[tuple[int, int]]]]:
    """Run global planner with multiple restarts, return best solution.

    Tries different planning strategies and random perturbations.
    Returns (best_score, best_combined_actions).
    """
    best_score = 0
    best_actions: list[list[tuple[int, int]]] = []
    t0 = time.time()

    strategies = ['basic', 'prefetch', 'pipeline']

    for strategy in strategies:
        if time.time() - t0 > time_budget:
            break

        for restart in range(restarts):
            if time.time() - t0 > time_budget:
                break

            rng_seed = restart * 1000 + hash(strategy) % 1000
            planner = GlobalPlanner(map_state, all_orders, num_bots, num_rounds,
                                    verbose=verbose, rng_seed=rng_seed)

            if verbose:
                print(f"\n--- Strategy: {strategy}, restart {restart} ---",
                      file=sys.stderr)

            if strategy == 'basic':
                combined = planner.plan()
            elif strategy == 'prefetch':
                combined = planner.plan_with_prefetch()
            else:
                combined = planner.plan_fast_pipeline()

            # Verify with game engine
            score, orders_comp, items_del = verify_plan(
                combined, map_state, all_orders, num_bots, num_rounds, verbose=False)

            if verbose:
                print(f"  Score: {score} (items={items_del}, orders={orders_comp})",
                      file=sys.stderr)

            if score > best_score:
                best_score = score
                best_actions = combined
                if verbose:
                    print(f"  *** NEW BEST: {score} ***", file=sys.stderr)

    if verbose:
        print(f"\nBest score: {best_score} in {time.time()-t0:.1f}s", file=sys.stderr)

    return best_score, best_actions


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_plan(combined_actions: list[list[tuple[int, int]]],
                map_state: MapState, all_orders: list[Order],
                num_bots: int, num_rounds: int,
                verbose: bool = True) -> tuple[int, int, int]:
    """Verify plan by running through game engine.

    Returns (score, orders_completed, items_delivered).
    """
    gs = GameState(map_state)
    gs.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
    gs.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
    for i in range(num_bots):
        gs.bot_positions[i] = [map_state.spawn[0], map_state.spawn[1]]

    # Set up orders
    orders_copy = [o.copy() for o in all_orders]
    gs.orders = [orders_copy[0], orders_copy[1]]
    gs.orders[0].status = 'active'
    gs.orders[1].status = 'preview'
    gs.next_order_idx = 2
    gs.active_idx = 0

    for r in range(min(num_rounds, len(combined_actions))):
        gs.round = r
        cpu_step(gs, combined_actions[r], orders_copy)

    if verbose:
        print(f"  Score: {gs.score} (items={gs.items_delivered}, "
              f"orders={gs.orders_completed})", file=sys.stderr)

    return gs.score, gs.orders_completed, gs.items_delivered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Global offline planner')
    parser.add_argument('--seeds', default='7005', help='Seeds to test')
    parser.add_argument('--difficulty', '-d', default='nightmare')
    parser.add_argument('--restarts', '-r', type=int, default=1)
    parser.add_argument('--strategy', '-s', default='all',
                        choices=['basic', 'prefetch', 'pipeline', 'all'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--time-budget', type=float, default=240.0)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    diff = args.difficulty
    num_rounds = DIFF_ROUNDS.get(diff, 300)
    cfg = CONFIGS[diff]

    scores = []
    for seed in seeds:
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Seed {seed}, {diff}, {cfg['bots']} bots, {num_rounds} rounds",
              file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        gs, all_orders = init_game(seed, diff)
        ms = gs.map_state

        if args.strategy == 'all':
            score, combined = solve_global(
                ms, all_orders, cfg['bots'], num_rounds,
                restarts=args.restarts, verbose=args.verbose,
                time_budget=args.time_budget)
        else:
            planner = GlobalPlanner(ms, all_orders, cfg['bots'], num_rounds,
                                    verbose=args.verbose)
            if args.strategy == 'basic':
                combined = planner.plan()
            elif args.strategy == 'prefetch':
                combined = planner.plan_with_prefetch()
            else:
                combined = planner.plan_fast_pipeline()

            score, _, _ = verify_plan(
                combined, ms, all_orders, cfg['bots'], num_rounds)

        scores.append(score)
        print(f"Score: {score}", file=sys.stderr)

    if len(scores) > 1:
        print(f"\nSummary: mean={np.mean(scores):.1f}, min={min(scores)}, "
              f"max={max(scores)}", file=sys.stderr)
        print(f"Scores: {scores}", file=sys.stderr)


if __name__ == '__main__':
    main()

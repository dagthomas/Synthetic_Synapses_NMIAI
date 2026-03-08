"""V10: Batch-Commit Chain Pipeline for nightmare mode.

Structural changes from V4 (302 baseline):
1. Zone separation: 1 dropoff for delivery, 2 for chain staging
2. All 20 bots always productive (no parking/fleeing)
3. Trip commitment: replan only on order change, not per-round
4. Chain-first: staging bots AT dropoff (not near it)
5. Deep pre-fetch: use all known future orders, not just preview
6. Fill-up: active carriers also carry preview items for chain bonus

Goal: 750+ (from 302). Leader: 1032.
"""
from __future__ import annotations

import time

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class Trip:
    """A committed multi-waypoint trip for one bot."""
    __slots__ = ['waypoints', 'wp_idx', 'role', 'stale']

    def __init__(self, waypoints: list, role: str):
        # waypoints: [(position, action_type, item_idx), ...]
        self.waypoints = waypoints
        self.wp_idx = 0
        self.role = role  # 'active', 'stage', 'future', 'deliver', 'park'
        self.stale = False

    @property
    def goal(self):
        if self.wp_idx < len(self.waypoints):
            return self.waypoints[self.wp_idx][0]
        return None

    @property
    def done(self):
        return self.wp_idx >= len(self.waypoints)


class V10Solver:
    """Batch-Commit Chain Pipeline: zone-separated delivery + staging."""

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = sorted([tuple(dz) for dz in ms.drop_off_zones],
                                 key=lambda d: d[0])
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.future_orders = future_orders or []

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)

        # Item type → [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # State
        self.trips: dict[int, Trip] = {}
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._last_active_id = -1
        self._last_orders_completed = -1
        self.chain_events: list[tuple[int, int]] = []

        # Zone assignment
        self._delivery_zone = self.drop_zones[1]  # middle by default
        self._staging_zones = [self.drop_zones[0], self.drop_zones[2]]

        # Corridor Y coordinates for parking
        self._corridor_ys = [1, ms.height // 2, ms.height - 3]

    # ------------------------------------------------------------------
    # Item lookup
    # ------------------------------------------------------------------

    def _find_item(self, pos, tid, claimed, preferred_drop=None):
        """Find best item of type tid. Returns (item_idx, adj_pos, cost)."""
        best_idx, best_adj, best_cost = None, None, 9999
        for item_idx, adj_cells in self.type_items.get(tid, []):
            if item_idx in claimed:
                continue
            for adj in adj_cells:
                d = self.tables.get_distance(pos, adj)
                if preferred_drop:
                    dd = self.tables.get_distance(adj, preferred_drop)
                else:
                    dd = min(self.tables.get_distance(adj, dz)
                             for dz in self.drop_zones)
                cost = d + dd * 0.3
                if cost < best_cost:
                    best_cost = cost
                    best_idx, best_adj = item_idx, adj
        return best_idx, best_adj, best_cost

    def _find_best_needed(self, pos, needs, claimed, preferred_drop=None):
        """Find nearest item matching any type in needs dict."""
        best_idx, best_adj, best_cost, best_tid = None, None, 9999, -1
        for tid, count in needs.items():
            if count <= 0:
                continue
            idx, adj, cost = self._find_item(pos, tid, claimed, preferred_drop)
            if cost < best_cost:
                best_idx, best_adj, best_cost, best_tid = idx, adj, cost, tid
        return best_idx, best_adj, best_tid

    # ------------------------------------------------------------------
    # Batch planning
    # ------------------------------------------------------------------

    def _plan_batch(self, bot_positions, bot_inventories,
                    active_order, preview_order, all_orders, state):
        """Assign trips to ALL bots. Called on order change."""
        self.trips.clear()
        num_bots = len(bot_positions)

        # --- Analyze orders ---
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Future orders (beyond preview)
        future = []
        if all_orders:
            for i in range(state.next_order_idx,
                           min(state.next_order_idx + 10, len(all_orders))):
                future.append(all_orders[i])

        # --- Classify bots ---
        active_carriers = []  # have active items
        preview_carriers = []  # have preview items
        dead_carriers = []  # have irrelevant items
        empty_bots = []

        for bid in range(num_bots):
            inv = bot_inventories.get(bid, [])
            if not inv:
                empty_bots.append(bid)
                continue
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            has_future = False
            if not has_active and not has_preview:
                for fo in future:
                    if any(fo.needs_type(t) for t in inv):
                        has_future = True
                        break
            if has_active:
                active_carriers.append(bid)
            elif has_preview:
                preview_carriers.append(bid)
            elif has_future:
                preview_carriers.append(bid)  # treat as stager
            else:
                dead_carriers.append(bid)

        # --- Choose delivery zone (nearest to active carriers) ---
        if active_carriers:
            zone_costs = {}
            for dz in self.drop_zones:
                zone_costs[dz] = sum(self.tables.get_distance(
                    bot_positions[bid], dz) for bid in active_carriers)
            self._delivery_zone = min(self.drop_zones, key=lambda d: zone_costs[d])
        self._staging_zones = [dz for dz in self.drop_zones
                               if dz != self._delivery_zone]

        # --- Active carriers → deliver ---
        for bid in active_carriers:
            self.trips[bid] = Trip(
                [(self._delivery_zone, ACT_DROPOFF, -1)],
                'deliver')

        # --- Preview/future carriers → stage at staging zones ---
        stager_counts = {dz: 0 for dz in self._staging_zones}
        for bid in preview_carriers:
            pos = bot_positions[bid]
            # Pick staging zone with least bots and closest
            sz = min(self._staging_zones,
                     key=lambda dz: self.tables.get_distance(pos, dz)
                                    + stager_counts.get(dz, 0) * 8)
            stager_counts[sz] = stager_counts.get(sz, 0) + 1
            self.trips[bid] = Trip([(sz, ACT_WAIT, -1)], 'stage')

        # --- Dead carriers → park in corridors ---
        occupied = set()
        for bid in dead_carriers:
            pos = bot_positions[bid]
            park = self._find_parking(pos, occupied)
            occupied.add(park)
            self.trips[bid] = Trip([(park, ACT_WAIT, -1)], 'park')

        # --- Empty bots → assign to active/preview/future items ---
        claimed = set()

        # Track what's already covered
        active_covered = {}
        for bid in active_carriers:
            for t in bot_inventories[bid]:
                if t in active_needs:
                    active_covered[t] = active_covered.get(t, 0) + 1

        preview_covered = {}
        for bid in preview_carriers:
            for t in bot_inventories[bid]:
                if t in preview_needs:
                    preview_covered[t] = preview_covered.get(t, 0) + 1

        active_short = {}
        for t, n in active_needs.items():
            s = n - active_covered.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_short = {}
        for t, n in preview_needs.items():
            s = n - preview_covered.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Sort empty bots by proximity to nearest needed item
        def _dist_to_any_need(bid):
            pos = bot_positions[bid]
            targets = active_short if active_short else preview_short
            if not targets:
                return 9999
            return min((self.tables.get_distance(pos, adj)
                        for tid in targets
                        for _, adj_cells in self.type_items.get(tid, [])
                        for adj in adj_cells), default=9999)

        empty_bots.sort(key=_dist_to_any_need)

        for bid in empty_bots:
            pos = bot_positions[bid]
            assigned = False

            # 1. Active items (go to delivery zone)
            if active_short:
                idx, adj, tid = self._find_best_needed(
                    pos, active_short, claimed, self._delivery_zone)
                if idx is not None:
                    self.trips[bid] = Trip(
                        [(adj, ACT_PICKUP, idx),
                         (self._delivery_zone, ACT_DROPOFF, -1)],
                        'active')
                    claimed.add(idx)
                    active_short[tid] -= 1
                    if active_short[tid] <= 0:
                        del active_short[tid]
                    assigned = True

            # 2. Preview items (go to staging zone)
            if not assigned and preview_short:
                sz = min(self._staging_zones,
                         key=lambda dz: self.tables.get_distance(pos, dz)
                                        + stager_counts.get(dz, 0) * 8)
                idx, adj, tid = self._find_best_needed(
                    pos, preview_short, claimed, sz)
                if idx is not None:
                    stager_counts[sz] = stager_counts.get(sz, 0) + 1
                    self.trips[bid] = Trip(
                        [(adj, ACT_PICKUP, idx),
                         (sz, ACT_WAIT, -1)],
                        'stage')
                    claimed.add(idx)
                    preview_short[tid] -= 1
                    if preview_short[tid] <= 0:
                        del preview_short[tid]
                    assigned = True

            # 3. Future order items
            if not assigned:
                for fi, fo in enumerate(future):
                    fo_needs = {}
                    for t in fo.needs():
                        fo_needs[t] = fo_needs.get(t, 0) + 1
                    idx, adj, tid = self._find_best_needed(
                        pos, fo_needs, claimed)
                    if idx is not None:
                        sz = min(self._staging_zones,
                                 key=lambda dz: self.tables.get_distance(adj, dz)
                                                + stager_counts.get(dz, 0) * 8)
                        stager_counts[sz] = stager_counts.get(sz, 0) + 1
                        self.trips[bid] = Trip(
                            [(adj, ACT_PICKUP, idx),
                             (sz, ACT_WAIT, -1)],
                            'stage')
                        claimed.add(idx)
                        assigned = True
                        break

            # 4. Park
            if not assigned:
                park = self._find_parking(pos, occupied)
                occupied.add(park)
                self.trips[bid] = Trip([(park, ACT_WAIT, -1)], 'park')

    # ------------------------------------------------------------------
    # Parking
    # ------------------------------------------------------------------

    def _find_parking(self, pos, occupied):
        best, best_d = self.spawn, 9999
        for cy in self._corridor_ys:
            for dx in range(15):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width:
                        cell = (cx, cy)
                        if (cell in self.tables.pos_to_idx
                                and cell not in self.drop_set
                                and cell not in occupied):
                            if any(self.tables.get_distance(cell, dz) <= 1
                                   for dz in self.drop_zones):
                                continue
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
        return best

    # ------------------------------------------------------------------
    # Escape
    # ------------------------------------------------------------------

    def _escape_action(self, bid, pos, rnd):
        dirs = list(MOVES)
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    # ------------------------------------------------------------------
    # Adjacent pickup (opportunistic)
    # ------------------------------------------------------------------

    def _check_adjacent_pickup(self, bid, pos, inv, active_needs, preview_needs):
        """Pick up adjacent active/preview items for free."""
        if len(inv) >= INV_CAP:
            return None
        bot_types = set(inv)
        for item_idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[item_idx])
            # Active items
            if tid in active_needs and active_needs[tid] > 0:
                if tid in bot_types and active_needs[tid] <= 1:
                    continue
                for adj in self.ms.item_adjacencies.get(item_idx, []):
                    if adj == pos:
                        return (ACT_PICKUP, item_idx)
            # Preview items (for chain)
            elif tid in preview_needs and preview_needs.get(tid, 0) > 0:
                if tid in bot_types:
                    continue
                for adj in self.ms.item_adjacencies.get(item_idx, []):
                    if adj == pos:
                        return (ACT_PICKUP, item_idx)
        return None

    # ------------------------------------------------------------------
    # Per-round action
    # ------------------------------------------------------------------

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        num_bots = len(state.bot_positions)

        # Extract state
        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall + congestion
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

        # --- Replan trigger: order change or too many done trips ---
        active_id = active_order.id if active_order else -1
        needs_replan = False
        if active_id != self._last_active_id:
            needs_replan = True
            self._last_active_id = active_id
        if state.orders_completed != self._last_orders_completed:
            needs_replan = True
            self._last_orders_completed = state.orders_completed

        # Also replan if many trips done or no trips exist
        if not self.trips:
            needs_replan = True
        else:
            done_or_stale = sum(1 for t in self.trips.values()
                                if t.done or t.stale)
            if done_or_stale >= num_bots // 2:
                needs_replan = True

        # Invalidate trips whose pickup types are no longer needed
        if not needs_replan:
            for bid, trip in self.trips.items():
                if trip.done or trip.stale:
                    continue
                if trip.role == 'active' and trip.wp_idx == 0:
                    # Check if the pickup type is still needed
                    wp = trip.waypoints[0]
                    if wp[1] == ACT_PICKUP and wp[2] >= 0:
                        tid = int(self.ms.item_types[wp[2]])
                        if active_order and not active_order.needs_type(tid):
                            trip.stale = True

        if needs_replan:
            self._plan_batch(bot_positions, bot_inventories,
                            active_order, preview_order, all_orders, state)

        # --- Compute active/preview needs for opportunistic pickup ---
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            # Subtract what's being carried
            for inv in bot_inventories.values():
                for t in inv:
                    if t in active_needs:
                        active_needs[t] = max(0, active_needs[t] - 1)
        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # --- Build goals for PIBT ---
        goals = {}
        goal_types = {}
        for bid in range(num_bots):
            trip = self.trips.get(bid)
            if trip and not trip.done and not trip.stale:
                goals[bid] = trip.goal
                goal_types[bid] = trip.role
            else:
                goals[bid] = bot_positions[bid]
                goal_types[bid] = 'park'

        # --- Urgency order ---
        prio = {'deliver': 0, 'active': 1, 'stage': 2, 'future': 3, 'park': 5}

        def _urgency(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals[bid])
            # Deliver bots at dropoff get highest priority
            if gt == 'deliver' and bot_positions[bid] in self.drop_set:
                return (-1, 0)
            return (prio.get(gt, 5), dist)

        urgency_order = sorted(range(num_bots), key=_urgency)

        # --- PIBT pathfinding ---
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # --- Build actions ---
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            trip = self.trips.get(bid)
            inv = bot_inventories[bid]

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                actions[bid] = (self._escape_action(bid, pos, rnd), -1)
                continue

            # --- At current waypoint: execute action ---
            if trip and not trip.done and not trip.stale:
                wp_pos, wp_act, wp_item = trip.waypoints[trip.wp_idx]
                if pos == wp_pos:
                    # Special: if deliver but no matching items, skip dropoff
                    if wp_act == ACT_DROPOFF:
                        if not inv:
                            trip.wp_idx += 1
                            if trip.wp_idx >= len(trip.waypoints):
                                trip.stale = True
                            actions[bid] = (ACT_WAIT, -1)
                            continue
                        # Only dropoff if at dropoff zone
                        if pos in self.drop_set:
                            actions[bid] = (ACT_DROPOFF, -1)
                            trip.wp_idx += 1
                            continue
                    elif wp_act == ACT_PICKUP:
                        actions[bid] = (ACT_PICKUP, wp_item)
                        trip.wp_idx += 1
                        continue
                    elif wp_act == ACT_WAIT:
                        # At staging position, check if we should deliver
                        if pos in self.drop_set and inv:
                            if active_order and any(active_order.needs_type(t) for t in inv):
                                actions[bid] = (ACT_DROPOFF, -1)
                                continue
                        actions[bid] = (ACT_WAIT, -1)
                        continue

            # --- At ANY dropoff with active items: deliver! ---
            if pos in self.drop_set and inv and active_order:
                if any(active_order.needs_type(t) for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # --- Opportunistic adjacent pickup ---
            opp = self._check_adjacent_pickup(
                bid, pos, inv, active_needs, preview_needs)
            if opp is not None:
                actions[bid] = opp
                continue

            # --- PIBT movement ---
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=200)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = V10Solver(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains, max_chain = 0, 0
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
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                drop_info = ""
                if c >= 1:
                    at = []
                    for b in range(len(state.bot_positions)):
                        bpos = (int(state.bot_positions[b, 0]),
                                int(state.bot_positions[b, 1]))
                        if bpos in solver.drop_set:
                            inv_b = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv_b}")
                    drop_info = f" Drop=[{','.join(at)}]"
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}"
                         if active else " DONE")
                      + extra + drop_info)

        elapsed = time.time() - t0
        if verbose:
            dead = sum(1 for b in range(len(state.bot_positions))
                       if state.bot_inv_list(b))
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"DeadBots={dead} "
                  f"Time={elapsed:.1f}s "
                  f"({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V10 Batch-Commit Chain Pipeline')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_v10, scores_v4 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - V10 Chain Pipeline")
        print(f"{'='*50}")
        score, _ = V10Solver.run_sim(seed, verbose=args.verbose)
        scores_v10.append(score)

        if args.compare:
            from nightmare_lmapf_solver import LMAPFSolver
            print(f"\n--- V4 ---")
            s4, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
            scores_v4.append(s4)
            print(f"\nV10={score} vs V4={s4} (delta={score - s4:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nV10: mean={statistics.mean(scores_v10):.1f} "
              f"max={max(scores_v10)} min={min(scores_v10)}")
        if scores_v4:
            print(f"V4: mean={statistics.mean(scores_v4):.1f}")

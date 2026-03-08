"""Batch Chain Reaction solver for nightmare mode - V3.

Rolling batch planner with chain reaction exploitation.
Each batch covers active + preview + 2-3 future orders.
Bots fetch 1-3 items each (active first), then deliver.
Chain reactions fire when active completes and staging bots are at dropoffs.

Key design:
  - 1 item per bot for active order (max parallelism, ~14 round delivery)
  - 3 staging bots at 3 dropoffs with preview items (chain reaction)
  - Remaining bots fetch future items (avoid dead inventory)
  - Rolling replan when batch is exhausted

Target: 750+ score (leader: 1032)
"""
from __future__ import annotations

import sys
import time
from itertools import permutations

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

NUM_ROUNDS = 500
NUM_BOTS = 20


class BotTrip:
    """A committed trip plan for one bot."""
    __slots__ = ['pickups', 'dropoff', 'phase', 'pickup_idx', 'role']

    def __init__(self):
        self.pickups: list[tuple[int, int, tuple[int, int], int]] = []  # (type_id, item_idx, adj_pos, order_idx)
        self.dropoff: tuple[int, int] = (0, 0)
        self.phase: str = 'idle'  # 'fetch', 'deliver', 'staged', 'idle', 'dead'
        self.pickup_idx: int = 0  # index into self.pickups for next pickup
        self.role: str = 'idle'  # 'active', 'staging', 'future', 'idle', 'dead'


class NightmareBatchSolverV3:
    """Rolling batch solver with chain reaction exploitation."""

    def __init__(self, ms: MapState, tables: PrecomputedTables | None = None,
                 captured_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = CONFIGS['nightmare']['bots']

        # Subsystems
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, self.tables,
                                               self.traffic, self.congestion)

        # Order sequence (full lookahead)
        self.captured_orders = captured_orders or []
        self._seq_pos = -1

        # Item lookup: type_id -> [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Zone mapping: which zone is each dropoff
        self.zone_drops = sorted(self.drop_zones, key=lambda d: d[0])

        # === Batch state ===
        self.batch_seq_start = -1
        self.batch_seq_end = -1
        self.bot_trips: dict[int, BotTrip] = {}
        self._last_active_req = None  # track active order changes

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Stats
        self.batches_planned = 0
        self.chain_events: list[tuple[int, int]] = []

    # ──────────────────────────────────────────────────────────────
    # Sequence matching
    # ──────────────────────────────────────────────────────────────

    def _find_seq_pos(self, active_order: Order | None) -> int:
        """Find position of active order in captured sequence."""
        if not active_order or not self.captured_orders:
            return -1
        active_req = tuple(sorted(active_order.required))
        start = max(0, self._seq_pos - 1)
        for i in range(start, len(self.captured_orders)):
            if tuple(sorted(self.captured_orders[i].required)) == active_req:
                self._seq_pos = i
                return i
        for i in range(0, start):
            if tuple(sorted(self.captured_orders[i].required)) == active_req:
                self._seq_pos = i
                return i
        return -1

    # ──────────────────────────────────────────────────────────────
    # Item finding
    # ──────────────────────────────────────────────────────────────

    def _find_best_item(self, pos: tuple[int, int], type_id: int,
                        claimed: set[int],
                        prefer_zone: int = -1) -> tuple[int | None, tuple[int, int] | None]:
        """Find nearest item of given type, return (item_idx, adj_pos)."""
        best_idx = None
        best_adj = None
        best_cost = 9999.0
        for item_idx, adj_cells in self.type_items.get(type_id, []):
            if item_idx in claimed:
                continue
            for adj in adj_cells:
                d = self.tables.get_distance(pos, adj)
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_idx = item_idx
                    best_adj = adj
        return best_idx, best_adj

    def _nearest_drop(self, pos: tuple[int, int]) -> tuple[int, int]:
        return min(self.drop_zones, key=lambda dz: self.tables.get_distance(pos, dz))

    def _tsp_sort_pickups(self, pickups, start_pos, end_pos):
        """Sort pickups to minimize total travel (try all permutations for <=3 items)."""
        if len(pickups) <= 1:
            return list(pickups)
        best_order = list(pickups)
        best_cost = 9999
        for perm in permutations(pickups):
            cost = self.tables.get_distance(start_pos, perm[0][2])  # adj_pos is at index 2
            for i in range(len(perm) - 1):
                cost += self.tables.get_distance(perm[i][2], perm[i + 1][2])
            cost += self.tables.get_distance(perm[-1][2], end_pos)
            if cost < best_cost:
                best_cost = cost
                best_order = list(perm)
        return best_order

    # ──────────────────────────────────────────────────────────────
    # Batch planning
    # ──────────────────────────────────────────────────────────────

    def _plan_batch(self, seq_pos: int,
                    bot_pos: dict[int, tuple[int, int]],
                    bot_inv: dict[int, list[int]],
                    active_order: Order | None = None):
        """Plan a rolling batch: active + preview + future orders.

        Assignment strategy:
        1. Active items: 1 bot per item (max parallelism)
        2. Preview items: assigned to bots at/near dropoffs, or bots with free slots
        3. Future items: remaining capacity
        """
        if seq_pos < 0 or seq_pos >= len(self.captured_orders):
            return

        # Determine active order needs
        if active_order:
            active_types = [int(t) for t in active_order.needs()]
        else:
            active_types = [int(t) for t in self.captured_orders[seq_pos].required]

        # Determine batch scope: active + up to 5 future orders
        batch_orders: list[list[int]] = [active_types]
        total_items = len(active_types)
        total_free_slots = sum(INV_CAP - len(bot_inv.get(b, [])) for b in range(self.num_bots))

        max_batch_items = min(total_free_slots, 40)  # cap batch size
        for i in range(seq_pos + 1, min(seq_pos + 6, len(self.captured_orders))):
            order_types = [int(t) for t in self.captured_orders[i].required]
            if total_items + len(order_types) > max_batch_items:
                break
            batch_orders.append(order_types)
            total_items += len(order_types)

        self.batch_seq_start = seq_pos
        self.batch_seq_end = seq_pos + len(batch_orders) - 1

        # Build all items needed, tagged by order index
        all_items: list[tuple[int, int]] = []  # (type_id, order_idx)
        for oi, order_types in enumerate(batch_orders):
            for tid in order_types:
                all_items.append((tid, oi))

        # Classify bots
        all_batch_types = set(tid for tid, _ in all_items)
        dead_bots = set()
        for bid in range(self.num_bots):
            inv = bot_inv.get(bid, [])
            if len(inv) >= INV_CAP and not any(t in all_batch_types for t in inv):
                dead_bots.add(bid)

        # Sort bots by free slots (most free first) and distance to nearest shelf
        bot_free = {}
        for bid in range(self.num_bots):
            if bid in dead_bots:
                bot_free[bid] = 0
            else:
                bot_free[bid] = INV_CAP - len(bot_inv.get(bid, []))

        # Greedy assignment: active items first (by rarity), then preview, then future
        claimed_items: set[int] = set()
        # (type_id, item_idx, adj_pos, order_idx)
        bot_assignments: dict[int, list[tuple[int, int, tuple[int, int], int]]] = {
            bid: [] for bid in range(self.num_bots)
        }
        bot_roles: dict[int, str] = {bid: 'idle' for bid in range(self.num_bots)}
        for bid in dead_bots:
            bot_roles[bid] = 'dead'

        # Phase 1: Assign active items (order_idx=0) — 1 item per bot for speed
        active_items = [(tid, oi) for tid, oi in all_items if oi == 0]
        for tid, oi in active_items:
            best_bid = None
            best_cost = 9999.0
            for bid in range(self.num_bots):
                if bot_free.get(bid, 0) <= 0:
                    continue
                if len(bot_assignments[bid]) >= 1 and oi == 0:
                    # For active items: prefer 1 per bot (max parallelism)
                    # Allow 2+ only if running out of bots
                    if any(aoi == 0 for _, _, _, aoi in bot_assignments[bid]):
                        continue
                pos = bot_pos.get(bid, self.spawn)
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is None:
                    continue
                dz = self._nearest_drop(adj)
                cost = self.tables.get_distance(pos, adj) + self.tables.get_distance(adj, dz) * 0.3
                if cost < best_cost:
                    best_cost = cost
                    best_bid = bid

            if best_bid is not None:
                pos = bot_pos.get(best_bid, self.spawn)
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is not None:
                    bot_assignments[best_bid].append((tid, item_idx, adj, oi))
                    claimed_items.add(item_idx)
                    bot_free[best_bid] -= 1
                    bot_roles[best_bid] = 'active'

        # Phase 2: Assign preview items (order_idx=1) — fill remaining slots
        preview_items = [(tid, oi) for tid, oi in all_items if oi == 1]
        for tid, oi in preview_items:
            best_bid = None
            best_cost = 9999.0
            for bid in range(self.num_bots):
                if bot_free.get(bid, 0) <= 0:
                    continue
                pos = bot_pos.get(bid, self.spawn)
                # Prefer bots that already have active items (piggybacking)
                role_bonus = -5 if bot_roles[bid] == 'active' else 0
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is None:
                    continue
                dz = self._nearest_drop(adj)
                cost = self.tables.get_distance(pos, adj) + self.tables.get_distance(adj, dz) * 0.3 + role_bonus
                if cost < best_cost:
                    best_cost = cost
                    best_bid = bid

            if best_bid is not None:
                pos = bot_pos.get(best_bid, self.spawn)
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is not None:
                    bot_assignments[best_bid].append((tid, item_idx, adj, oi))
                    claimed_items.add(item_idx)
                    bot_free[best_bid] -= 1
                    if bot_roles[best_bid] == 'idle':
                        bot_roles[best_bid] = 'staging'

        # Phase 3: Assign future items — fill remaining capacity
        future_items = [(tid, oi) for tid, oi in all_items if oi >= 2]
        for tid, oi in future_items:
            best_bid = None
            best_cost = 9999.0
            for bid in range(self.num_bots):
                if bot_free.get(bid, 0) <= 0:
                    continue
                pos = bot_pos.get(bid, self.spawn)
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is None:
                    continue
                dz = self._nearest_drop(adj)
                cost = self.tables.get_distance(pos, adj) + self.tables.get_distance(adj, dz) * 0.3
                if cost < best_cost:
                    best_cost = cost
                    best_bid = bid

            if best_bid is not None:
                pos = bot_pos.get(best_bid, self.spawn)
                item_idx, adj = self._find_best_item(pos, tid, claimed_items)
                if item_idx is not None:
                    bot_assignments[best_bid].append((tid, item_idx, adj, oi))
                    claimed_items.add(item_idx)
                    bot_free[best_bid] -= 1
                    if bot_roles[best_bid] == 'idle':
                        bot_roles[best_bid] = 'future'

        # Create trip plans
        self.bot_trips = {}
        for bid in range(self.num_bots):
            trip = BotTrip()
            trip.role = bot_roles[bid]

            if bot_roles[bid] == 'dead':
                trip.phase = 'dead'
                trip.dropoff = self.spawn
            elif not bot_assignments[bid]:
                trip.phase = 'idle'
                trip.dropoff = self.spawn
            else:
                pos = bot_pos.get(bid, self.spawn)
                dz = self._nearest_drop(pos)
                trip.dropoff = dz
                trip.pickups = self._tsp_sort_pickups(
                    bot_assignments[bid], pos, dz)
                trip.phase = 'fetch'
                trip.pickup_idx = 0

                # Skip pickups we already have in inventory
                inv = bot_inv.get(bid, [])
                while trip.pickup_idx < len(trip.pickups):
                    tid = trip.pickups[trip.pickup_idx][0]
                    if tid in inv:
                        inv = list(inv)
                        inv.remove(tid)
                        trip.pickup_idx += 1
                    else:
                        break
                if trip.pickup_idx >= len(trip.pickups):
                    trip.phase = 'deliver' if trip.role == 'active' else 'staged'

            self.bot_trips[bid] = trip

        self.batches_planned += 1
        na = sum(1 for t in self.bot_trips.values() if t.role == 'active')
        ns = sum(1 for t in self.bot_trips.values() if t.role == 'staging')
        nf = sum(1 for t in self.bot_trips.values() if t.role == 'future')
        nd = sum(1 for t in self.bot_trips.values() if t.role == 'dead')
        batch_size = len(batch_orders)
        print(f"  [batch #{self.batches_planned}] orders {seq_pos}-{self.batch_seq_end} "
              f"({batch_size} ord, {total_items} items) "
              f"active={na} staging={ns} future={nf} dead={nd}",
              file=sys.stderr)

    # ──────────────────────────────────────────────────────────────
    # Per-round goal computation
    # ──────────────────────────────────────────────────────────────

    def _needs_replan(self, active_order: Order | None, seq_pos: int,
                      bot_inv: dict[int, list[int]]) -> bool:
        """Check if we need to replan the batch."""
        if not self.bot_trips:
            return True

        # Replan if active order changed (new order in sequence)
        if active_order:
            active_req = tuple(sorted(active_order.required))
        else:
            active_req = None
        if active_req != self._last_active_req:
            # Order advanced — always replan to reassign bots
            return True

        # Replan if beyond batch
        if seq_pos >= 0 and seq_pos > self.batch_seq_end:
            return True

        # Replan if too many bots idle (wasting capacity)
        idle_count = sum(1 for t in self.bot_trips.values()
                         if t.phase in ('idle',) and t.role != 'dead')
        if idle_count >= 5:
            return True

        return False

    def _update_trips_on_order_change(self, bot_inv: dict[int, list[int]]):
        """When active order advances, update trip phases."""
        for bid, trip in self.bot_trips.items():
            if trip.role == 'active' and trip.phase in ('deliver', 'staged'):
                # Active bot delivered — check if still has items
                inv = bot_inv.get(bid, [])
                if inv:
                    trip.phase = 'staged'  # still has preview/future items
                    trip.role = 'staging'
                else:
                    trip.phase = 'idle'
            elif trip.role == 'staging':
                # Staging bot — might now be "active" for the new active order
                trip.role = 'staging'  # stays staging until delivered

    def _compute_goals(self, bot_pos: dict[int, tuple[int, int]],
                       bot_inv: dict[int, list[int]],
                       active_order: Order | None,
                       rnd: int
                       ) -> tuple[dict, dict, dict]:
        """Compute goals from committed trip plans."""
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}
        occupied_goals: set[tuple[int, int]] = set()

        # Count active items still needed
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Compute how many active items are "in flight" (in bot inventories)
        active_in_flight = 0
        for bid in range(self.num_bots):
            inv = bot_inv.get(bid, [])
            for t in inv:
                if t in active_needs:
                    active_in_flight += 1

        active_remaining = sum(active_needs.values()) - active_in_flight
        active_almost_done = active_remaining <= 0  # all items in transit or delivered

        for bid in range(self.num_bots):
            trip = self.bot_trips.get(bid)
            if trip is None:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            pos = bot_pos.get(bid, self.spawn)
            inv = bot_inv.get(bid, [])

            if trip.phase == 'dead':
                pk = self._find_parking(pos, occupied_goals)
                occupied_goals.add(pk)
                goals[bid] = pk
                goal_types[bid] = 'park'
                continue

            if trip.phase == 'idle':
                # Idle bots with free slots: try to fetch remaining active items
                if len(inv) < INV_CAP and active_needs:
                    # Find an active item type still needed
                    claimed = set(pickup_targets.values())
                    assigned_types = set()
                    for b2, t2 in self.bot_trips.items():
                        if t2.phase == 'fetch' and t2.pickup_idx < len(t2.pickups):
                            assigned_types.add(t2.pickups[t2.pickup_idx][0])
                    for t_need, count in active_needs.items():
                        item_idx, adj = self._find_best_item(pos, t_need, claimed)
                        if item_idx is not None:
                            goals[bid] = adj
                            goal_types[bid] = 'pickup'
                            pickup_targets[bid] = item_idx
                            trip.phase = 'fetch'
                            trip.role = 'active'
                            trip.pickups = [(t_need, item_idx, adj, 0)]
                            trip.pickup_idx = 0
                            trip.dropoff = self._nearest_drop(adj)
                            break
                    else:
                        pk = self._find_parking(pos, occupied_goals)
                        occupied_goals.add(pk)
                        goals[bid] = pk
                        goal_types[bid] = 'park'
                else:
                    pk = self._find_parking(pos, occupied_goals)
                    occupied_goals.add(pk)
                    goals[bid] = pk
                    goal_types[bid] = 'park'
                continue

            if trip.phase == 'fetch':
                # Check if current pickup is already in inventory
                while trip.pickup_idx < len(trip.pickups):
                    tid, item_idx, adj, _oi = trip.pickups[trip.pickup_idx]
                    if tid in inv:
                        # Already have this type, skip
                        inv = list(inv)
                        inv.remove(tid)
                        trip.pickup_idx += 1
                    else:
                        break

                if trip.pickup_idx < len(trip.pickups) and len(inv) < INV_CAP:
                    tid, item_idx, adj, _oi = trip.pickups[trip.pickup_idx]
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                else:
                    # Done fetching
                    if trip.role == 'active':
                        trip.phase = 'deliver'
                    else:
                        trip.phase = 'staged'
                    # Fall through to deliver/staged handling below

            if trip.phase == 'deliver':
                # Active bot: go to dropoff
                # Check if we have any items matching active order
                has_active = any(t in active_needs for t in inv)
                if has_active:
                    dz = trip.dropoff
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                else:
                    # No active items — became staging or idle
                    if inv:
                        trip.phase = 'staged'
                    else:
                        # Empty after delivery — try to fetch more active items
                        trip.phase = 'idle'
                        if active_needs:
                            claimed = set(pickup_targets.values())
                            for t_need, count in active_needs.items():
                                item_idx, adj = self._find_best_item(pos, t_need, claimed)
                                if item_idx is not None:
                                    goals[bid] = adj
                                    goal_types[bid] = 'pickup'
                                    pickup_targets[bid] = item_idx
                                    trip.phase = 'fetch'
                                    trip.role = 'active'
                                    trip.pickups = [(t_need, item_idx, adj, 0)]
                                    trip.pickup_idx = 0
                                    trip.dropoff = self._nearest_drop(adj)
                                    break
                        if trip.phase == 'idle':
                            pk = self._find_parking(pos, occupied_goals)
                            occupied_goals.add(pk)
                            goals[bid] = pk
                            goal_types[bid] = 'park'
                        continue

            if trip.phase == 'staged':
                if not inv:
                    trip.phase = 'idle'
                    pk = self._find_parking(pos, occupied_goals)
                    occupied_goals.add(pk)
                    goals[bid] = pk
                    goal_types[bid] = 'park'
                    continue

                # Staging bot with items: go to dropoff when active is almost done
                if active_almost_done or pos in self.drop_set:
                    dz = trip.dropoff
                    if dz not in occupied_goals:
                        goals[bid] = dz
                        occupied_goals.add(dz)
                    else:
                        # Dropoff occupied — find alternative dropoff
                        alt = None
                        for dz2 in self.drop_zones:
                            if dz2 not in occupied_goals:
                                alt = dz2
                                break
                        if alt:
                            goals[bid] = alt
                            occupied_goals.add(alt)
                        else:
                            # All dropoffs full — stage near dropoff
                            near = self._find_near_dropoff(pos, dz, occupied_goals)
                            goals[bid] = near
                            occupied_goals.add(near)
                    goal_types[bid] = 'stage'
                else:
                    # Not time yet — park near dropoff (within 5 steps)
                    dz = trip.dropoff
                    near = self._find_staging_wait(pos, dz, occupied_goals)
                    occupied_goals.add(near)
                    goals[bid] = near
                    goal_types[bid] = 'stage'

        return goals, goal_types, pickup_targets

    def _find_staging_wait(self, pos, dz, occupied):
        """Find a parking spot within 5 steps of dropoff (but not ON dropoff)."""
        best = pos
        best_d = 9999
        # Corridor cells near dropoff
        for dy in range(-3, 4):
            for dx in range(-5, 6):
                cell = (dz[0] + dx, dz[1] + dy)
                if cell in self.drop_set or cell in occupied:
                    continue
                if cell not in self.tables.pos_to_idx:
                    continue
                d_to_drop = self.tables.get_distance(cell, dz)
                if d_to_drop > 6 or d_to_drop < 2:
                    continue
                d_from_bot = self.tables.get_distance(pos, cell)
                if d_from_bot + d_to_drop < best_d:
                    best_d = d_from_bot + d_to_drop
                    best = cell
        return best

    def _find_near_dropoff(self, pos, dz, occupied):
        """Find a cell 1-2 steps from dropoff zone."""
        best = dz
        best_d = 9999
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = (dz[0] + dx, dz[1] + dy)
                if cell in self.drop_set or cell in occupied:
                    continue
                if cell not in self.tables.pos_to_idx:
                    continue
                d = self.tables.get_distance(cell, dz)
                if 0 < d <= 2:
                    score = self.tables.get_distance(pos, cell)
                    if score < best_d:
                        best_d = score
                        best = cell
        return best

    def _find_parking(self, pos, occupied):
        """Find corridor parking spot."""
        best = self.spawn
        best_d = 9999
        for cy in [1, self.ms.height // 2, self.ms.height - 3]:
            for dx in range(15):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width:
                        cell = (cx, cy)
                        if (cell in self.tables.pos_to_idx
                                and cell not in occupied
                                and cell not in self.drop_set):
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
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

    # ──────────────────────────────────────────────────────────────
    # SIM MODE
    # ──────────────────────────────────────────────────────────────

    def action(self, state: GameState, all_orders: list[Order], rnd: int
               ) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)

        bot_pos: dict[int, tuple[int, int]] = {}
        bot_inv: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_pos[bid] = (int(state.bot_positions[bid, 0]),
                            int(state.bot_positions[bid, 1]))
            bot_inv[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_pos.values()))
        for bid in range(num_bots):
            p = bot_pos[bid]
            if self.prev_positions.get(bid) == p:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = p

        active_order = state.get_active_order()
        seq_pos = self._find_seq_pos(active_order)

        # Batch management
        if self._needs_replan(active_order, seq_pos, bot_inv):
            if seq_pos >= 0:
                self._plan_batch(seq_pos, bot_pos, bot_inv, active_order)
                self._last_active_req = tuple(sorted(active_order.required)) if active_order else None

        if not self.bot_trips:
            return [(ACT_WAIT, -1)] * num_bots

        goals, goal_types, pickup_targets = self._compute_goals(
            bot_pos, bot_inv, active_order, rnd)

        # Urgency sort
        pmap = {'deliver': 0, 'pickup': 1, 'stage': 2, 'park': 5}
        urgency_order = sorted(range(num_bots), key=lambda bid: (
            pmap.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos.get(bid, self.spawn),
                                     goals.get(bid, self.spawn))))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency_order, goal_types=goal_types)

        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots
        for bid in range(num_bots):
            pos = bot_pos[bid]
            gt = goal_types.get(bid, 'park')
            inv = bot_inv[bid]
            trip = self.bot_trips.get(bid)

            # At dropoff: handle delivery/staging
            if pos in self.drop_set:
                if gt == 'deliver' and inv:
                    # Check if we have active items to deliver
                    if active_order:
                        active_types = set(int(t) for t in active_order.needs())
                        if any(t in active_types for t in inv):
                            actions[bid] = (ACT_DROPOFF, -1)
                            continue
                # Staging: wait for chain
                if gt == 'stage' and inv:
                    actions[bid] = (ACT_WAIT, -1)
                    continue
                # Empty at dropoff: flee
                if not inv:
                    act = self._escape_action(bid, pos, rnd)
                    actions[bid] = (act, -1)
                    continue

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # At pickup target
            if gt == 'pickup' and bid in pickup_targets:
                if pos == goals[bid]:
                    item_idx = pickup_targets[bid]
                    actions[bid] = (ACT_PICKUP, item_idx)
                    # Advance pickup index
                    if trip and trip.phase == 'fetch':
                        trip.pickup_idx += 1
                    continue

            actions[bid] = (path_actions.get(bid, ACT_WAIT), -1)

        return actions

    # ──────────────────────────────────────────────────────────────
    # WebSocket live mode
    # ──────────────────────────────────────────────────────────────

    def ws_action(self, live_bots: list[dict], data: dict,
                  map_state: MapState) -> list[dict]:
        ms = map_state or self.ms
        rnd = data.get('round', 0)

        active_order, preview_order = self._parse_ws_orders(data, ms)
        bot_pos, bot_inv = self._parse_ws_bots(live_bots, ms)

        self.congestion.update(list(bot_pos.values()))
        for bid, pos in bot_pos.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        seq_pos = self._find_seq_pos(active_order)

        if self._needs_replan(active_order, seq_pos, bot_inv):
            if seq_pos >= 0:
                self._plan_batch(seq_pos, bot_pos, bot_inv, active_order)
                self._last_active_req = tuple(sorted(active_order.required)) if active_order else None
            elif seq_pos < 0:
                return self._reactive_fallback(live_bots, data, ms)

        if not self.bot_trips:
            return self._reactive_fallback(live_bots, data, ms)

        goals, goal_types, pickup_targets = self._compute_goals(
            bot_pos, bot_inv, active_order, rnd)

        pmap = {'deliver': 0, 'pickup': 1, 'stage': 2, 'park': 5}
        all_bids = [b['id'] for b in live_bots]
        urgency_order = sorted(all_bids, key=lambda bid: (
            pmap.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos.get(bid, self.spawn),
                                     goals.get(bid, self.spawn))))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency_order, goal_types=goal_types)

        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left',
                        'move_right', 'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            inv_names = bot.get('inventory', [])
            inv = bot_inv.get(bid, [])
            trip = self.bot_trips.get(bid)

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            if pos in self.drop_set:
                if gt == 'deliver' and inv_names:
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    continue
                if gt == 'stage' and inv_names:
                    ws_actions.append({'bot': bid, 'action': 'wait'})
                    continue

            if gt == 'pickup' and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goals[bid] and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid, 'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id']
                    })
                    if trip and trip.phase == 'fetch':
                        trip.pickup_idx += 1
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _parse_ws_orders(self, data, ms):
        active_order = None
        preview_order = None
        for od in data.get('orders', []):
            req_ids = [ms.type_name_to_id.get(n, 0)
                       for n in od.get('items_required', [])]
            order = Order(0, req_ids, od.get('status', 'active'))
            for dn in od.get('items_delivered', []):
                tid = ms.type_name_to_id.get(dn, -1)
                if tid >= 0:
                    order.deliver_type(tid)
            if od.get('status') == 'active':
                active_order = order
            elif od.get('status') == 'preview':
                preview_order = order
        return active_order, preview_order

    def _parse_ws_bots(self, live_bots, ms):
        bot_pos = {}
        bot_inv = {}
        for bot in live_bots:
            bid = bot['id']
            bot_pos[bid] = tuple(bot['position'])
            inv = []
            for name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv[bid] = inv
        return bot_pos, bot_inv

    def _reactive_fallback(self, live_bots, data, ms):
        if not hasattr(self, '_v3'):
            from nightmare_solver_v2 import NightmareSolverV3
            self._v3 = NightmareSolverV3(ms, self.tables)
        return self._v3.ws_action(live_bots, data, ms)

    # ──────────────────────────────────────────────────────────────
    # Sim runner
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def run_sim(seed: int, verbose: bool = False,
                captured_orders: list[Order] | None = None) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=200)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)

        orders_to_use = captured_orders if captured_orders else all_orders
        solver = NightmareBatchSolverV3(ms, tables, captured_orders=orders_to_use)

        num_rounds = DIFF_ROUNDS['nightmare']
        action_log = []
        chains = 0
        max_chain = 0

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
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Batches={solver.batches_planned}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


# Keep V2 as alias for backwards compatibility
NightmareBatchSolverV2 = NightmareBatchSolverV3


def main():
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description='Batch Chain Solver V3')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    for seed in seeds:
        score, _ = NightmareBatchSolverV3.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f"Seed {seed}: {score}")
    if len(scores) > 1:
        print(f"\nMean: {np.mean(scores):.1f}  Max: {max(scores)}  Min: {min(scores)}")


if __name__ == '__main__':
    main()

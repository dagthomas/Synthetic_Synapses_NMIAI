"""Offline pipeline planner for nightmare mode.

Key insight: every bot carries BOTH active AND preview items. When a bot
drops off at DZ, active items deliver to current order. Preview items stay
in inventory for cascade auto-delivery when the order completes.

With 20 bots and 3 DZs, each order completion potentially triggers cascade
on bots at other DZs, completing the next order in 0 rounds.

This planner uses full order knowledge (deterministic per-day sequence) to:
1. Never create dead inventory (only pick up items that WILL be delivered)
2. Mix active + preview items in every trip for cascade potential
3. Keep all 20 bots continuously busy (no parking/fleeing)
4. Route bots to minimize total trip time

Usage:
    python nightmare_pipeline_planner.py --seeds 7005 -v
    python nightmare_pipeline_planner.py --seeds 1000-1009
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    build_map_from_capture, generate_all_orders,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TripWaypoint:
    """A destination with an action to perform there."""
    pos: tuple[int, int]
    action_type: int   # ACT_PICKUP, ACT_DROPOFF, or ACT_WAIT
    item_idx: int = -1  # For pickup: which item to pick up


@dataclass
class BotTrip:
    """A bot's multi-step trip: visit waypoints sequentially."""
    bot_id: int
    waypoints: list[TripWaypoint]
    for_order: int  # Primary order this trip serves (-1 = future prefetch)
    wp_idx: int = 0  # Current waypoint

    def current_goal(self) -> tuple[int, int] | None:
        if self.wp_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.wp_idx].pos

    def current_action(self) -> tuple[int, int] | None:
        if self.wp_idx >= len(self.waypoints):
            return None
        wp = self.waypoints[self.wp_idx]
        return (wp.action_type, wp.item_idx)

    def advance(self):
        self.wp_idx += 1

    def is_done(self) -> bool:
        return self.wp_idx >= len(self.waypoints)

    def goal_type_str(self) -> str:
        if self.is_done():
            return 'idle'
        wp = self.waypoints[self.wp_idx]
        if wp.action_type == ACT_PICKUP:
            return 'pickup'
        elif wp.action_type == ACT_DROPOFF:
            return 'deliver'
        return 'park'


def _order_needs(order: Order | None) -> dict[int, int]:
    """Return {type_id: count_still_needed}."""
    if not order:
        return {}
    needs: dict[int, int] = {}
    for t in order.needs():
        needs[t] = needs.get(t, 0) + 1
    return needs


# ---------------------------------------------------------------------------
# Pipeline Planner
# ---------------------------------------------------------------------------

class NightmarePipelinePlanner:
    """Offline planner: cascade-loaded delivery for nightmare mode."""

    # Max future orders to prefetch for (beyond preview)
    PREFETCH_DEPTH = 6

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 all_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.all_orders = all_orders or []
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = CONFIGS['nightmare']['bots']
        self.walkable = build_walkable(ms)

        # Subsystems
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)

        # Item lookup: type_id -> [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = [tuple(a) for a in ms.item_adjacencies.get(idx, [])]
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Position -> adjacent items for opportunistic pickup
        self.pos_adj_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            for adj in ms.item_adjacencies.get(idx, []):
                adj_t = (int(adj[0]), int(adj[1]))
                if adj_t not in self.pos_adj_items:
                    self.pos_adj_items[adj_t] = []
                self.pos_adj_items[adj_t].append((idx, tid))

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self.STALL_LIMIT = 3

        # Corridor rows for parking
        self._corridor_ys = [1, ms.height // 2, ms.height - 3]

    # ------------------------------------------------------------------
    # Trip assignment
    # ------------------------------------------------------------------

    def _assign_trips(self, idle_bots: list[int],
                      bot_pos: dict[int, tuple[int, int]],
                      bot_inv: dict[int, list[int]],
                      active_trips: dict[int, BotTrip],
                      state: GameState,
                      rnd: int):
        """Assign trips to idle bots. Core planning logic.

        Key constraints:
        - Reserve 2 empty bots for active order emergencies
        - Max 8 bots heading to DZ at once (prevent gridlock)
        - Only send bots to DZ if they carry active/preview items
        - Future-item bots park in corridors (don't clog DZ)
        """
        active_order = state.get_active_order()
        preview_order = state.get_preview_order()
        if not active_order:
            return

        active_needs = _order_needs(active_order)
        preview_needs = _order_needs(preview_order)

        # What's already being fetched/carried toward active?
        in_transit_active: dict[int, int] = defaultdict(int)
        in_transit_preview: dict[int, int] = defaultdict(int)

        for bid, trip in active_trips.items():
            inv = bot_inv.get(bid, [])
            for t in inv:
                if t in active_needs:
                    in_transit_active[t] += 1
                elif t in preview_needs:
                    in_transit_preview[t] += 1
            for wp in trip.waypoints[trip.wp_idx:]:
                if wp.action_type == ACT_PICKUP:
                    tid = int(self.ms.item_types[wp.item_idx])
                    if tid in active_needs:
                        in_transit_active[tid] += 1
                    elif tid in preview_needs:
                        in_transit_preview[tid] += 1

        # Also count items carried by idle bots
        for bid in idle_bots:
            for t in bot_inv.get(bid, []):
                if t in active_needs:
                    in_transit_active[t] += 1
                elif t in preview_needs:
                    in_transit_preview[t] += 1

        # Active shortfall
        active_short: dict[int, int] = {}
        for t, n in active_needs.items():
            s = n - in_transit_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Preview shortfall
        preview_short: dict[int, int] = {}
        for t, n in preview_needs.items():
            s = n - in_transit_preview.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Future orders for deeper prefetch
        future_needs: list[dict[int, int]] = []
        if self.all_orders:
            start = state.next_order_idx
            for i in range(start, min(start + self.PREFETCH_DEPTH, len(self.all_orders))):
                future_needs.append(_order_needs(self.all_orders[i]))

        active_type_set = set(active_needs.keys())
        preview_type_set = set(preview_needs.keys())
        all_future_types: set[int] = set()
        for fn in future_needs:
            all_future_types.update(fn.keys())

        # Classify idle bots
        active_carriers: list[int] = []
        preview_carriers: list[int] = []
        empty_bots: list[int] = []
        future_carriers: list[int] = []
        dead_bots: list[int] = []

        for bid in idle_bots:
            inv = bot_inv.get(bid, [])
            if not inv:
                empty_bots.append(bid)
            elif any(t in active_type_set for t in inv):
                active_carriers.append(bid)
            elif any(t in preview_type_set for t in inv):
                preview_carriers.append(bid)
            elif any(t in all_future_types for t in inv):
                future_carriers.append(bid)
            else:
                dead_bots.append(bid)

        claimed_items: set[int] = set()
        for bid, trip in active_trips.items():
            for wp in trip.waypoints[trip.wp_idx:]:
                if wp.action_type == ACT_PICKUP:
                    claimed_items.add(wp.item_idx)

        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}
        # Count bots currently heading to DZ
        dz_bound_count = 0
        for bid, trip in active_trips.items():
            for wp in trip.waypoints[trip.wp_idx:]:
                if wp.action_type == ACT_DROPOFF:
                    if wp.pos in dropoff_loads:
                        dropoff_loads[wp.pos] += 1
                    dz_bound_count += 1
                    break

        MAX_DZ_BOUND = 8  # Max bots heading to DZ simultaneously
        RESERVE_EMPTY = 2  # Keep N empty bots for active emergencies
        active_short_total = sum(active_short.values())

        occupied_parks: set[tuple[int, int]] = set()

        # === Phase 1: Active carriers → deliver to DZ ===
        for bid in active_carriers:
            if dz_bound_count >= MAX_DZ_BOUND:
                # DZ full — park near DZ instead
                park = self._corridor_parking(bot_pos[bid], occupied_parks)
                occupied_parks.add(park)
                active_trips[bid] = BotTrip(
                    bid, [TripWaypoint(park, ACT_WAIT)], -1)
                continue

            pos = bot_pos[bid]
            inv = bot_inv[bid]
            free_slots = INV_CAP - len(inv)
            waypoints: list[TripWaypoint] = []

            # Fill spare slots with preview items ON THE PATH to DZ
            if free_slots > 0 and preview_short and active_short_total == 0:
                items = self._select_items_on_path(
                    pos, preview_short, claimed_items,
                    max_items=free_slots, toward_dz=True)
                for item_idx, adj_pos, tid in items:
                    waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                    preview_short[tid] = preview_short.get(tid, 0) - 1
                    if preview_short[tid] <= 0:
                        del preview_short[tid]
                    claimed_items.add(item_idx)
                    pos = adj_pos

            dz = self._balanced_dz(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            dz_bound_count += 1
            waypoints.append(TripWaypoint(dz, ACT_DROPOFF))
            active_trips[bid] = BotTrip(bid, waypoints, 0)

        # === Phase 2: Empty bots → fetch active items (highest priority) ===
        remaining_empty = sorted(empty_bots, key=lambda bid:
            self._min_dist_to_types(bot_pos[bid], active_short)
            if active_short else 9999)

        assigned_to_active: set[int] = set()
        for bid in remaining_empty:
            if not active_short:
                break
            pos = bot_pos[bid]
            items = self._select_items(
                pos, active_short, claimed_items, max_items=INV_CAP)
            if not items:
                continue

            waypoints: list[TripWaypoint] = []
            for item_idx, adj_pos, tid in items:
                waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                active_short[tid] = active_short.get(tid, 0) - 1
                if active_short[tid] <= 0:
                    del active_short[tid]
                claimed_items.add(item_idx)
                pos = adj_pos

            # Fill remaining slots with preview items
            remaining_slots = INV_CAP - len(items)
            if remaining_slots > 0 and preview_short:
                more = self._select_items(
                    pos, preview_short, claimed_items,
                    max_items=remaining_slots)
                for item_idx, adj_pos, tid in more:
                    waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                    preview_short[tid] = preview_short.get(tid, 0) - 1
                    if preview_short[tid] <= 0:
                        del preview_short[tid]
                    claimed_items.add(item_idx)
                    pos = adj_pos

            if dz_bound_count < MAX_DZ_BOUND:
                dz = self._balanced_dz(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                dz_bound_count += 1
                waypoints.append(TripWaypoint(dz, ACT_DROPOFF))
            else:
                # Just fetch, will deliver when DZ frees up
                pass

            active_trips[bid] = BotTrip(bid, waypoints, 0)
            assigned_to_active.add(bid)

        # === Phase 2b: Redundant active fetchers (backup) ===
        if active_needs and not active_short:
            redundant_needs = dict(active_needs)
            redundant_count = 0
            for bid in remaining_empty:
                if bid in assigned_to_active or redundant_count >= 3:
                    break
                pos = bot_pos[bid]
                items = self._select_items(
                    pos, redundant_needs, claimed_items, max_items=INV_CAP)
                if items:
                    waypoints = []
                    for item_idx, adj_pos, tid in items:
                        waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                        claimed_items.add(item_idx)
                        pos = adj_pos
                    if dz_bound_count < MAX_DZ_BOUND:
                        dz = self._balanced_dz(pos, dropoff_loads)
                        dropoff_loads[dz] += 1
                        dz_bound_count += 1
                        waypoints.append(TripWaypoint(dz, ACT_DROPOFF))
                    active_trips[bid] = BotTrip(bid, waypoints, 0)
                    assigned_to_active.add(bid)
                    redundant_count += 1

        # === Phase 3: Preview carriers → stage at DZ (for cascade) ===
        # Only stage if active shortfall is low (nearly done)
        active_nearly_done = active_short_total <= 2
        max_staging = len(self.drop_zones)  # 1 per DZ max
        staging_count = 0

        for bid in preview_carriers:
            pos = bot_pos[bid]

            if (active_nearly_done and staging_count < max_staging
                    and dz_bound_count < MAX_DZ_BOUND):
                dz = self._balanced_dz(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                dz_bound_count += 1
                staging_count += 1
                active_trips[bid] = BotTrip(
                    bid, [TripWaypoint(dz, ACT_DROPOFF)], 1)
            else:
                # Park near DZ row, ready to stage when active is nearly done
                park = self._corridor_parking(pos, occupied_parks)
                occupied_parks.add(park)
                active_trips[bid] = BotTrip(
                    bid, [TripWaypoint(park, ACT_WAIT)], -1)

        # === Phase 4: Remaining empty bots → prefetch preview/future ===
        prefetch_empty = [bid for bid in remaining_empty
                          if bid not in assigned_to_active]
        # Reserve some empty bots
        max_prefetch = max(0, len(prefetch_empty) - RESERVE_EMPTY)
        prefetch_count = 0

        for bid in prefetch_empty:
            if bid in active_trips:
                continue

            if prefetch_count >= max_prefetch:
                # Reserve bot: park
                park = self._corridor_parking(bot_pos[bid], occupied_parks)
                occupied_parks.add(park)
                active_trips[bid] = BotTrip(
                    bid, [TripWaypoint(park, ACT_WAIT)], -1)
                continue

            pos = bot_pos[bid]
            waypoints: list[TripWaypoint] = []

            # Pick preview items first
            if preview_short:
                items = self._select_items(
                    pos, preview_short, claimed_items, max_items=INV_CAP)
                for item_idx, adj_pos, tid in items:
                    waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                    preview_short[tid] = preview_short.get(tid, 0) - 1
                    if preview_short[tid] <= 0:
                        del preview_short[tid]
                    claimed_items.add(item_idx)
                    pos = adj_pos

            # Fill remaining with future items
            remaining_slots = INV_CAP - len(waypoints)
            if remaining_slots > 0 and future_needs:
                for fi, fn in enumerate(future_needs):
                    if remaining_slots <= 0:
                        break
                    if not fn:
                        continue
                    items = self._select_items(
                        pos, fn, claimed_items, max_items=remaining_slots)
                    for item_idx, adj_pos, tid in items:
                        waypoints.append(TripWaypoint(adj_pos, ACT_PICKUP, item_idx))
                        fn[tid] = fn.get(tid, 0) - 1
                        if fn[tid] <= 0:
                            del fn[tid]
                        claimed_items.add(item_idx)
                        remaining_slots -= 1
                        pos = adj_pos

            if waypoints:
                # Prefetch bots DON'T go to DZ — just pick up and wait
                # They'll deliver when they become active/preview carriers
                active_trips[bid] = BotTrip(bid, waypoints, 2)
                prefetch_count += 1
            else:
                park = self._corridor_parking(pos, occupied_parks)
                occupied_parks.add(park)
                active_trips[bid] = BotTrip(
                    bid, [TripWaypoint(park, ACT_WAIT)], -1)

        # === Phase 5: Future carriers + dead bots → park ===
        for bid in future_carriers + dead_bots:
            if bid in active_trips:
                continue
            pos = bot_pos[bid]
            inv = bot_inv.get(bid, [])

            # Dead bots with items matching active → deliver
            if inv and any(t in active_type_set for t in inv):
                if dz_bound_count < MAX_DZ_BOUND:
                    dz = self._balanced_dz(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    dz_bound_count += 1
                    active_trips[bid] = BotTrip(
                        bid, [TripWaypoint(dz, ACT_DROPOFF)], 0)
                    continue

            # Otherwise park
            park = self._corridor_parking(pos, occupied_parks)
            occupied_parks.add(park)
            active_trips[bid] = BotTrip(
                bid, [TripWaypoint(park, ACT_WAIT)], -1)

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _generate_actions(self, bot_pos: dict[int, tuple[int, int]],
                          bot_inv: dict[int, list[int]],
                          bot_trips: dict[int, BotTrip],
                          state: GameState,
                          rnd: int) -> list[tuple[int, int]]:
        """Convert bot trips to per-round actions using pathfinder."""
        active_order = state.get_active_order()
        active_needs = _order_needs(active_order)

        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}

        for bid in range(self.num_bots):
            trip = bot_trips.get(bid)
            if trip is None or trip.is_done():
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            goal = trip.current_goal()
            if goal is None:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            goals[bid] = goal
            gt = trip.goal_type_str()
            goal_types[bid] = gt

        # Urgency: deliver > pickup > park
        priority_map = {'deliver': 0, 'pickup': 1, 'park': 5, 'idle': 5}
        urgency = sorted(range(self.num_bots), key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(
                bot_pos[bid], goals.get(bid, self.spawn)),
            (bid + rnd) % 100
        ))

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency,
            goal_types=goal_types, round_number=rnd)

        # Build final actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * self.num_bots

        for bid in range(self.num_bots):
            pos = bot_pos[bid]
            trip = bot_trips.get(bid)
            inv = bot_inv.get(bid, [])

            # Stall escape
            if self.stall_counts.get(bid, 0) >= self.STALL_LIMIT:
                goal = goals.get(bid, self.spawn)
                act = self._escape_action(bid, pos, rnd, goal)
                actions[bid] = (act, -1)
                continue

            # At DZ with items: ALWAYS drop off (enables cascade)
            if pos in self.drop_set and inv:
                # Drop off if active order could use any of our items
                # OR if we're on a delivery trip
                if any(t in active_needs for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                    if trip and not trip.is_done():
                        wp = trip.waypoints[trip.wp_idx]
                        if wp.action_type == ACT_DROPOFF and wp.pos == pos:
                            trip.advance()
                    continue
                # Delivery trip at DZ: drop off (might cascade later)
                if trip and not trip.is_done():
                    wp = trip.waypoints[trip.wp_idx]
                    if wp.action_type == ACT_DROPOFF and wp.pos == pos:
                        actions[bid] = (ACT_DROPOFF, -1)
                        trip.advance()
                        continue

            # At pickup waypoint: pick up
            if trip and not trip.is_done():
                wp = trip.waypoints[trip.wp_idx]
                if wp.action_type == ACT_PICKUP and pos == wp.pos:
                    if len(inv) < INV_CAP:
                        actions[bid] = (ACT_PICKUP, wp.item_idx)
                        trip.advance()
                        continue
                    else:
                        # Inventory full — skip this pickup
                        trip.advance()

            # At wait waypoint: wait
            if trip and not trip.is_done():
                wp = trip.waypoints[trip.wp_idx]
                if wp.action_type == ACT_WAIT and pos == wp.pos:
                    actions[bid] = (ACT_WAIT, -1)
                    trip.advance()
                    continue

            # Opportunistic adjacent pickup (free active items)
            if len(inv) < INV_CAP and active_needs:
                adj_items = self.pos_adj_items.get(pos, [])
                for item_idx, tid in adj_items:
                    if tid in active_needs:
                        actions[bid] = (ACT_PICKUP, item_idx)
                        break
                if actions[bid][0] == ACT_PICKUP:
                    continue

            # Navigate toward goal
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    # ------------------------------------------------------------------
    # Main simulation entry point
    # ------------------------------------------------------------------

    def run_sim(self, state: GameState, all_orders: list[Order],
                num_rounds: int = 500,
                verbose: bool = False) -> tuple[int, list]:
        """Run the planner through all rounds."""
        bot_trips: dict[int, BotTrip] = {}
        action_log = []
        chains = 0
        max_chain = 0

        # Stats
        goal_totals: dict[str, int] = defaultdict(int)
        stall_total = 0

        t0 = time.time()

        for rnd in range(num_rounds):
            state.round = rnd

            # Extract bot state
            bot_pos: dict[int, tuple[int, int]] = {}
            bot_inv: dict[int, list[int]] = {}
            for bid in range(self.num_bots):
                bot_pos[bid] = (int(state.bot_positions[bid, 0]),
                                int(state.bot_positions[bid, 1]))
                bot_inv[bid] = state.bot_inv_list(bid)

            # Update congestion + stalls
            self.congestion.update(list(bot_pos.values()))
            for bid in range(self.num_bots):
                if self.prev_positions.get(bid) == bot_pos[bid]:
                    self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
                else:
                    self.stall_counts[bid] = 0
                self.prev_positions[bid] = bot_pos[bid]

            # Clean up completed trips
            for bid in list(bot_trips.keys()):
                if bot_trips[bid].is_done():
                    del bot_trips[bid]

            # Assign trips to idle bots
            idle_bots = [bid for bid in range(self.num_bots)
                         if bid not in bot_trips]
            if idle_bots:
                self._assign_trips(
                    idle_bots, bot_pos, bot_inv,
                    bot_trips, state, rnd)

            # Generate and execute actions
            actions = self._generate_actions(
                bot_pos, bot_inv, bot_trips, state, rnd)
            action_log.append(actions)

            # Track stats
            for bid in range(self.num_bots):
                trip = bot_trips.get(bid)
                if trip:
                    gt = trip.goal_type_str()
                else:
                    gt = 'idle'
                goal_totals[gt] += 1
                if self.stall_counts.get(bid, 0) >= 1:
                    stall_total += 1

            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                cascade_str = f" CASCADE x{c}!" if c > 1 else ""
                busy = sum(1 for t in bot_trips.values() if not t.is_done())
                total_items = sum(len(inv) for inv in bot_inv.values())
                dz_count = sum(1 for bid in range(self.num_bots)
                               if bot_pos[bid] in self.drop_set
                               and bot_inv[bid])
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d} "
                      f"Busy={busy:2d}/20 Inv={total_items:2d} DZ={dz_count}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + cascade_str)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
            avg_per_rnd = {gt: cnt / num_rounds
                           for gt, cnt in sorted(goal_totals.items())}
            print(f"Avg/rnd: {' '.join(f'{gt}={v:.1f}' for gt, v in avg_per_rnd.items())}")
            print(f"Stalls: {stall_total} ({stall_total/num_rounds:.1f}/rnd)")

        return state.score, action_log

    # ------------------------------------------------------------------
    # WebSocket live action entry point
    # ------------------------------------------------------------------

    def ws_action(self, live_bots: list[dict], data: dict,
                  map_state: MapState) -> list[dict]:
        """Per-round entry for live_gpu_stream.py WebSocket format."""
        ms = map_state or self.ms
        num_bots = len(live_bots)

        # Build order objects from WS data
        orders_data = data.get('orders', [])
        active_order = None
        preview_order = None
        for od in orders_data:
            items_req = od.get('items_required', [])
            items_del = od.get('items_delivered', [])
            req_ids = [ms.type_name_to_id.get(n, 0) for n in items_req]
            order = Order(0, req_ids, od.get('status', 'active'))
            for dn in items_del:
                tid = ms.type_name_to_id.get(dn, -1)
                if tid >= 0:
                    order.deliver_type(tid)
            if od.get('status') == 'active':
                active_order = order
            elif od.get('status') == 'preview':
                preview_order = order

        # Build position/inventory dicts
        bot_pos: dict[int, tuple[int, int]] = {}
        bot_inv: dict[int, list[int]] = {}
        for bot in live_bots:
            bid = bot['id']
            bot_pos[bid] = tuple(bot['position'])
            inv = []
            for item_name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(item_name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv[bid] = inv

        rnd = data.get('round', 0)

        # Build a temporary GameState for _assign_trips
        temp_state = GameState(ms)
        temp_state.round = rnd
        temp_state.score = data.get('score', 0)
        temp_state.orders_completed = data.get('active_order_index', 0)
        temp_state.next_order_idx = temp_state.orders_completed + 2
        # Build orders list
        temp_state.orders = []
        if active_order:
            temp_state.orders.append(active_order)
        if preview_order:
            temp_state.orders.append(preview_order)
        temp_state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
        temp_state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
        for bid in range(num_bots):
            if bid in bot_pos:
                temp_state.bot_positions[bid] = list(bot_pos[bid])
            for i, t in enumerate(bot_inv.get(bid, [])):
                temp_state.bot_inventories[bid, i] = t

        # Update congestion + stalls
        self.congestion.update(list(bot_pos.values()))
        for bid, pos in bot_pos.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Initialize trip store if not exists
        if not hasattr(self, '_ws_trips'):
            self._ws_trips: dict[int, BotTrip] = {}

        # Clean up completed trips
        for bid in list(self._ws_trips.keys()):
            if self._ws_trips[bid].is_done():
                del self._ws_trips[bid]

        # Assign trips to idle bots
        idle_bots = [bid for bid in range(num_bots)
                     if bid not in self._ws_trips]
        if idle_bots:
            self._assign_trips(
                idle_bots, bot_pos, bot_inv,
                self._ws_trips, temp_state, rnd)

        # Generate actions
        active_needs = _order_needs(active_order)
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}

        for bid in range(num_bots):
            trip = self._ws_trips.get(bid)
            if trip is None or trip.is_done():
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
            else:
                goal = trip.current_goal()
                goals[bid] = goal if goal else self.spawn
                goal_types[bid] = trip.goal_type_str()

        priority_map = {'deliver': 0, 'pickup': 1, 'park': 5, 'idle': 5}
        all_bids = [bot['id'] for bot in live_bots]
        urgency = sorted(all_bids, key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(
                bot_pos.get(bid, self.spawn),
                goals.get(bid, self.spawn)),
            (bid + rnd) % 100
        ))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency,
            goal_types=goal_types, round_number=rnd)

        # Build WS actions
        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left',
                        'move_right', 'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            inv_names = bot.get('inventory', [])
            inv_types = [ms.type_name_to_id.get(n, -1) for n in inv_names]
            trip = self._ws_trips.get(bid)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= self.STALL_LIMIT:
                goal = goals.get(bid, self.spawn)
                act = self._escape_action(bid, pos, rnd, goal)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            # At DZ with items: always drop off
            if pos in self.drop_set and inv_names:
                if any(t in active_needs for t in inv_types if t >= 0):
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    if trip and not trip.is_done():
                        wp = trip.waypoints[trip.wp_idx]
                        if wp.action_type == ACT_DROPOFF and wp.pos == pos:
                            trip.advance()
                    continue
                if trip and not trip.is_done():
                    wp = trip.waypoints[trip.wp_idx]
                    if wp.action_type == ACT_DROPOFF and wp.pos == pos:
                        ws_actions.append({'bot': bid, 'action': 'drop_off'})
                        trip.advance()
                        continue

            # At pickup waypoint
            if trip and not trip.is_done():
                wp = trip.waypoints[trip.wp_idx]
                if wp.action_type == ACT_PICKUP and pos == wp.pos:
                    if len(inv_names) < INV_CAP:
                        ws_actions.append({
                            'bot': bid, 'action': 'pick_up',
                            'item_id': ms.items[wp.item_idx]['id'],
                        })
                        trip.advance()
                        continue
                    else:
                        trip.advance()

            # Opportunistic: adjacent active item
            if len(inv_names) < INV_CAP and active_needs:
                adj_items = self.pos_adj_items.get(pos, [])
                picked = False
                for item_idx, tid in adj_items:
                    if tid in active_needs:
                        ws_actions.append({
                            'bot': bid, 'action': 'pick_up',
                            'item_id': ms.items[item_idx]['id'],
                        })
                        picked = True
                        break
                if picked:
                    continue

            # Navigate
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    # ------------------------------------------------------------------
    # Item selection helpers
    # ------------------------------------------------------------------

    def _select_items(self, pos: tuple[int, int],
                      needs: dict[int, int],
                      claimed: set[int],
                      max_items: int = 3) -> list[tuple[int, tuple[int, int], int]]:
        """Select up to max_items to pick up, greedy nearest-first.

        Returns: [(item_idx, adj_pos, type_id), ...]
        """
        items = []
        remaining = dict(needs)
        current_pos = pos

        for _ in range(max_items):
            if not remaining:
                break
            best: tuple[int, tuple[int, int], int] | None = None
            best_cost = 9999.0

            for tid, count in remaining.items():
                if count <= 0:
                    continue
                for item_idx, adj_cells in self.type_items.get(tid, []):
                    if item_idx in claimed:
                        continue
                    for adj in adj_cells:
                        d = self.tables.get_distance(current_pos, adj)
                        # Slight preference for items close to DZ
                        drop_d = min(self.tables.get_distance(adj, dz)
                                     for dz in self.drop_zones)
                        cost = d + drop_d * 0.3
                        if cost < best_cost:
                            best_cost = cost
                            best = (item_idx, adj, tid)

            if best:
                items.append(best)
                claimed.add(best[0])
                current_pos = best[1]
                remaining[best[2]] = remaining.get(best[2], 0) - 1
                if remaining[best[2]] <= 0:
                    del remaining[best[2]]
            else:
                break

        return items

    def _select_items_on_path(self, pos: tuple[int, int],
                              needs: dict[int, int],
                              claimed: set[int],
                              max_items: int = 1,
                              toward_dz: bool = False
                              ) -> list[tuple[int, tuple[int, int], int]]:
        """Select items that are roughly on the path to DZ (low detour cost).

        Only picks up items where detour < 4 steps.
        """
        if not needs:
            return []

        dz_dist = min(self.tables.get_distance(pos, dz)
                      for dz in self.drop_zones)

        items = []
        for tid, count in needs.items():
            if count <= 0:
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d_to_item = self.tables.get_distance(pos, adj)
                    d_item_to_dz = min(self.tables.get_distance(adj, dz)
                                       for dz in self.drop_zones)
                    detour = (d_to_item + d_item_to_dz) - dz_dist
                    if detour <= 3:  # Max 3 extra steps
                        items.append((item_idx, adj, tid, detour))

        # Sort by detour cost
        items.sort(key=lambda x: x[3])

        result = []
        remaining = dict(needs)
        for item_idx, adj, tid, _ in items:
            if len(result) >= max_items:
                break
            if item_idx in claimed:
                continue
            if remaining.get(tid, 0) <= 0:
                continue
            result.append((item_idx, adj, tid))
            claimed.add(item_idx)
            remaining[tid] -= 1
            if remaining[tid] <= 0:
                del remaining[tid]

        return result

    def _min_dist_to_types(self, pos: tuple[int, int],
                           needs: dict[int, int]) -> int:
        """Minimum distance to any item of needed types."""
        best = 9999
        for tid in needs:
            for _, adj_cells in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

    def _balanced_dz(self, pos: tuple[int, int],
                     loads: dict[tuple[int, int], int]
                     ) -> tuple[int, int]:
        """Choose DZ balancing distance and current load."""
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            load_pen = loads.get(dz, 0) * 4
            score = d + load_pen
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _corridor_parking(self, pos: tuple[int, int],
                          occupied: set[tuple[int, int]]
                          ) -> tuple[int, int]:
        """Find a parking spot in a corridor."""
        best = self.spawn
        best_d = 9999
        for cy in self._corridor_ys:
            for dx in range(10):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width:
                        cell = (cx, cy)
                        if cell in self.walkable and cell not in occupied:
                            if any(self.tables.get_distance(cell, dz) <= 1
                                   for dz in self.drop_zones):
                                continue
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
        return best

    def _escape_action(self, bid: int, pos: tuple[int, int],
                       rnd: int, goal: tuple[int, int] | None = None) -> int:
        """Deterministic anti-stall movement."""
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_planner_sim(seed: int, verbose: bool = False,
                    live_map: MapState | None = None) -> tuple[int, list]:
    """Run full simulation with pipeline planner."""
    if live_map is not None:
        all_orders = generate_all_orders(
            seed, live_map, 'nightmare', count=150)
        num_bots = CONFIGS['nightmare']['bots']
        state = GameState(live_map)
        state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
        state.bot_inventories = np.full(
            (num_bots, INV_CAP), -1, dtype=np.int8)
        for i in range(num_bots):
            state.bot_positions[i] = [live_map.spawn[0], live_map.spawn[1]]
        state.orders = [all_orders[0].copy(), all_orders[1].copy()]
        state.orders[0].status = 'active'
        state.orders[1].status = 'preview'
        state.next_order_idx = 2
        state.active_idx = 0
        ms = live_map
    else:
        state, all_orders = init_game(seed, 'nightmare', num_orders=150)
        ms = state.map_state

    tables = PrecomputedTables.get(ms)
    planner = NightmarePipelinePlanner(ms, tables, all_orders)
    num_rounds = DIFF_ROUNDS['nightmare']

    return planner.run_sim(state, all_orders, num_rounds, verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no-live-map', action='store_true',
                        help='Use procedural map instead of live map')
    args = parser.parse_args()

    # Parse seeds
    seeds = []
    for part in args.seeds.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    # Load live map if available
    live_map = None
    if not args.no_live_map:
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap:
                live_map = build_map_from_capture(cap)
                print(f"Using live map (drop_zones={[tuple(dz) for dz in live_map.drop_off_zones]})")
        except Exception as e:
            print(f"Could not load live map: {e}")

    scores = []
    for seed in seeds:
        score, _ = run_planner_sim(seed, verbose=args.verbose, live_map=live_map)
        scores.append(score)
        if not args.verbose:
            print(f"Seed {seed}: {score}")

    if len(scores) > 1:
        print(f"\nMean={np.mean(scores):.1f} Min={min(scores)} Max={max(scores)}")
        print(f"Scores: {scores}")


if __name__ == '__main__':
    main()

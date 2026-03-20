"""Cascade solver for nightmare mode.

DZ-only cascade: when active order completes, only bots AT a dropoff
zone get inventories re-checked. Official docs say "any items in bot
inventories" but LIVE TESTING CONFIRMS DZ-only (2026-03-10, score 279,
zero cascade chains observed).

Strategy: maximize throughput (orders/round). Leader gets 1032 via
~100 orders in 500 rounds (5 rounds/order). Key is fast active order
completion + minimizing idle bots.

Bot roles:
  - Delivery team: complete active order at DZ ASAP
  - Prefetch team: pick items for future orders
  - Staging: full bots with preview items → park at DZ for small cascades
  - Redundant active fetchers: backup bots with active items for speed

Usage:
    python nightmare_cascade_solver.py --seeds 7005 -v
    python nightmare_cascade_solver.py --seeds 1000-1009 -v
    python nightmare_cascade_solver.py --seeds 7005 -v --no-live-map
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from game_engine import (
    init_game, step, GameState, Order, MapState,
    build_map_from_capture, generate_all_orders,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF, actions_to_ws_format,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Mission(NamedTuple):
    """Persistent bot mission that survives across rounds."""
    goal_type: str          # 'deliver', 'pickup', 'prefetch', 'stage', 'park', 'flee'
    goal: tuple[int, int]   # target position
    item_idx: int           # target item index (-1 if N/A)
    type_id: int            # target item type (-1 if N/A)
    order_idx: int          # future order index (-1 if active, >=0 if prefetch)
    assigned_round: int     # round when mission was assigned


def _order_needs_map(order: Order) -> dict[int, int]:
    """Return {type_id: count_still_needed} for an order."""
    needs: dict[int, int] = {}
    for t in order.needs():
        needs[t] = needs.get(t, 0) + 1
    return needs


def _inv_type_counts(inv: list[int]) -> dict[int, int]:
    """Count item types in an inventory list."""
    counts: dict[int, int] = {}
    for t in inv:
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Cascade solver
# ---------------------------------------------------------------------------

class CascadeSolver:
    """Exploit auto-delivery cascades: prefetch items for future orders.

    DZ-only cascade (CONFIRMED on live server 2026-03-11 + 2026-03-13):
    when active order completes, only bots AT a dropoff zone get inventories
    re-checked. MCP docs say "any items in bot inventories" but live server
    only checks bots at DZ positions.

    Strategy: maximize active order throughput. Use most bots for active
    delivery, small team for near-term prefetch, stage at DZ for cascades.
    """

    # How many future orders to look ahead for prefetching
    PREFETCH_DEPTH = 4
    # Max delivery-team size (bots actively completing the current order)
    MAX_DELIVERY = 10
    # Stall threshold before escape maneuver
    STALL_LIMIT = 3
    # Persistent mission reassignment cooldown (rounds)
    REASSIGN_COOLDOWN = 3
    # Max total prefetch items across fleet (prefetch gating)
    MAX_PREFETCH_CARRYING = 40

    def __init__(self, map_state: MapState,
                 precomputed_tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None,
                 **kwargs):
        self.ms = map_state
        self.tables = precomputed_tables or PrecomputedTables.get(map_state)
        self.walkable = build_walkable(map_state)
        self.drop_zones = [tuple(dz) for dz in map_state.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = map_state.spawn
        self.num_bots = CONFIGS['nightmare']['bots']

        # Subsystems
        self.traffic = TrafficRules(map_state)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            map_state, self.tables, self.traffic, self.congestion)

        # Pre-loaded future orders (full lookahead in sim, or from capture)
        self.future_orders = future_orders or []

        # Item index lookup: type_id -> [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            adj = map_state.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Precompute pos→adjacent items for fast opportunistic pickup
        # Keys must be (int, int) tuples to match bot position lookups
        self.pos_adj_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            for adj in map_state.item_adjacencies.get(idx, []):
                adj_t = (int(adj[0]), int(adj[1]))
                if adj_t not in self.pos_adj_items:
                    self.pos_adj_items[adj_t] = []
                self.pos_adj_items[adj_t].append((idx, tid))

        # Corridor rows for parking
        self._corridor_ys = [1, map_state.height // 2, map_state.height - 3]

        # Zone assignment: assign each bot to its nearest DZ zone
        # Bots prefer items near their zone's DZ for prefetch
        self.bot_zone: dict[int, int] = {}  # bid -> zone_idx
        self.zone_dz: dict[int, tuple[int, int]] = {}  # zone_idx -> dz
        for zi, dz in enumerate(self.drop_zones):
            self.zone_dz[zi] = dz
        bots_per_zone = self.num_bots // len(self.drop_zones)
        for bid in range(self.num_bots):
            zi = min(bid // bots_per_zone, len(self.drop_zones) - 1)
            self.bot_zone[bid] = zi

        # Stall detection
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Chain event log (for diagnostics)
        self.chain_events: list[tuple[int, int]] = []

        # Track last goal types for utilization stats
        self._last_goal_types: dict[int, str] = {}

        # Active order stuck detection
        self._active_stuck_rounds = 0
        self._last_orders_completed = 0

        # Persistent mission tracking (#3: reduce role churn)
        self._missions: dict[int, Mission] = {}
        self._last_active_order_id: int | None = None

        # Apply parameter overrides from kwargs
        for k in ('PREFETCH_DEPTH', 'MAX_DELIVERY', 'STALL_LIMIT',
                  'REASSIGN_COOLDOWN', 'MAX_PREFETCH_CARRYING'):
            if k in kwargs:
                setattr(self, k, kwargs[k])

    # ------------------------------------------------------------------
    # Future order management
    # ------------------------------------------------------------------

    def _get_future_orders(self, state: GameState,
                           all_orders: list[Order],
                           depth: int | None = None) -> list[Order]:
        """Get the next `depth` upcoming orders after the active one.

        Includes the preview order as future[0], then orders from all_orders
        or self.future_orders starting at next_order_idx.
        """
        if depth is None:
            depth = self.PREFETCH_DEPTH
        future: list[Order] = []

        # Preview is always future[0]
        preview = state.get_preview_order()
        if preview:
            future.append(preview)

        # Sim mode: full lookahead from all_orders
        if all_orders:
            start = state.next_order_idx
            for i in range(start, min(start + depth, len(all_orders))):
                future.append(all_orders[i])
        elif self.future_orders:
            # Live mode fallback
            start = state.orders_completed + 2
            for i in range(start, min(start + depth, len(self.future_orders))):
                future.append(self.future_orders[i])

        return future[:depth]

    # ------------------------------------------------------------------
    # Commitment tracking
    # ------------------------------------------------------------------

    def _build_commitments(self, bot_inventories: dict[int, list[int]],
                           active_order: Order | None,
                           future_orders: list[Order]
                           ) -> dict[int, dict[int, int]]:
        """Build order_commitments: which items are already covered.

        Returns: {future_order_index: {type_id: count_committed}}

        An item in a bot's inventory is "committed" to the earliest future
        order that still needs it (greedy first-fit). Active order items are
        handled separately by the delivery logic and excluded here.
        """
        # Collect all carried types across the fleet (excluding active-matching)
        active_needs = _order_needs_map(active_order) if active_order else {}
        # Pool of all carried items not needed for active
        pool: dict[int, int] = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                pool[t] = pool.get(t, 0) + 1
        # Remove active needs from pool (those items go to delivery, not prefetch)
        for t, n in active_needs.items():
            if t in pool:
                pool[t] = max(0, pool[t] - n)
                if pool[t] == 0:
                    del pool[t]

        commitments: dict[int, dict[int, int]] = {}
        remaining_pool = dict(pool)

        for fi, forder in enumerate(future_orders):
            needs = _order_needs_map(forder)
            committed: dict[int, int] = {}
            for t, n in needs.items():
                available = remaining_pool.get(t, 0)
                use = min(n, available)
                if use > 0:
                    committed[t] = use
                    remaining_pool[t] = available - use
                    if remaining_pool[t] == 0:
                        del remaining_pool[t]
            commitments[fi] = committed

        return commitments

    def _compute_prefetch_needs(self, future_orders: list[Order],
                                commitments: dict[int, dict[int, int]]
                                ) -> list[dict[int, int]]:
        """For each future order, compute {type: count_still_needed_to_prefetch}.

        Subtracts what's already committed (in bot inventories) from total needs.
        """
        prefetch_needs: list[dict[int, int]] = []
        for fi, forder in enumerate(future_orders):
            needs = _order_needs_map(forder)
            committed = commitments.get(fi, {})
            shortfall: dict[int, int] = {}
            for t, n in needs.items():
                s = n - committed.get(t, 0)
                if s > 0:
                    shortfall[t] = s
            prefetch_needs.append(shortfall)
        return prefetch_needs

    # ------------------------------------------------------------------
    # Persistent mission management (#3)
    # ------------------------------------------------------------------

    def _validate_missions(self, bot_positions: dict[int, tuple[int, int]],
                           bot_inventories: dict[int, list[int]],
                           active_order: Order | None,
                           active_short: dict[int, int],
                           future_orders: list[Order],
                           prefetch_needs: list[dict[int, int]],
                           rnd: int) -> dict[int, Mission]:
        """Validate persistent missions. Returns {bid: mission} for valid ones.

        A mission is invalidated if:
        - Bot completed it (at goal + performed action)
        - Active order changed (delivery/pickup missions for old order)
        - Item no longer needed (shortfall covered by other bots)
        - Cooldown expired AND bot is stalled
        """
        valid: dict[int, Mission] = {}
        active_id = id(active_order) if active_order else None
        order_changed = active_id != self._last_active_order_id
        self._last_active_order_id = active_id

        for bid, m in self._missions.items():
            pos = bot_positions.get(bid)
            inv = bot_inventories.get(bid, [])
            if pos is None:
                continue

            age = rnd - m.assigned_round

            # Deliver: valid if bot still has items and active didn't change
            if m.goal_type == 'deliver':
                if not inv or order_changed:
                    continue
                active_needs = _order_needs_map(active_order) if active_order else {}
                if not any(t in active_needs for t in inv):
                    continue
                # Completed if at DZ and will drop off
                if pos in self.drop_set:
                    continue
                valid[bid] = m
                continue

            # Pickup (active fetch): valid if type still in shortfall
            if m.goal_type == 'pickup':
                if order_changed or m.type_id not in active_short:
                    continue
                # Completed if at goal (will pick up)
                if pos == m.goal:
                    continue
                # Inventory full
                if len(inv) >= INV_CAP:
                    continue
                valid[bid] = m
                continue

            # Prefetch: valid if future order still needs this type
            if m.goal_type == 'prefetch':
                fi = m.order_idx
                if fi < 0 or fi >= len(prefetch_needs):
                    continue
                if m.type_id not in prefetch_needs[fi] or prefetch_needs[fi][m.type_id] <= 0:
                    continue
                if pos == m.goal:
                    continue
                if len(inv) >= INV_CAP:
                    continue
                valid[bid] = m
                continue

            # Stage: valid if bot has preview items and not yet at DZ
            if m.goal_type == 'stage':
                if pos in self.drop_set:
                    continue
                if not inv:
                    continue
                valid[bid] = m
                continue

            # Park/flee: re-evaluate every round (stale after cooldown)
            if m.goal_type in ('park', 'flee'):
                if age < self.REASSIGN_COOLDOWN:
                    valid[bid] = m

        return valid

    # ------------------------------------------------------------------
    # Hungarian assignment (#1)
    # ------------------------------------------------------------------

    def _hungarian_assign(self, bot_ids: list[int],
                          bot_positions: dict[int, tuple[int, int]],
                          needed: dict[int, int],
                          claimed_items: set[int],
                          all_bot_positions: dict[int, tuple[int, int]],
                          ) -> dict[int, tuple[int, tuple[int, int], int]]:
        """Globally optimal bot-to-item assignment via Hungarian algorithm.

        Args:
            bot_ids: Bot IDs to assign.
            bot_positions: Current bot positions.
            needed: {type_id: count_needed} items to pick up.
            claimed_items: Already-claimed item indices.
            all_bot_positions: All bot positions for congestion cost.

        Returns:
            {bid: (item_idx, adj_pos, type_id)} for assigned bots.
        """
        if not bot_ids or not needed:
            return {}

        # Build task list: (item_idx, adj_pos, type_id)
        tasks: list[tuple[int, tuple[int, int], int]] = []
        # Expand needed types into individual tasks (one per count)
        for tid, count in needed.items():
            items_for_type = self.type_items.get(tid, [])
            added = 0
            for item_idx, adj_cells in items_for_type:
                if item_idx in claimed_items:
                    continue
                if not adj_cells:
                    continue
                # Pick best adj cell (closest to nearest DZ)
                best_adj = min(adj_cells, key=lambda a: self._drop_dist(a))
                tasks.append((item_idx, best_adj, tid))
                added += 1
                if added >= count:
                    break

        if not tasks:
            return {}

        n_bots = len(bot_ids)
        n_tasks = len(tasks)
        size = max(n_bots, n_tasks)
        cost_matrix = np.full((size, size), 9999.0)

        for i, bid in enumerate(bot_ids):
            pos = bot_positions[bid]
            for j, (item_idx, adj, tid) in enumerate(tasks):
                dist = self.tables.get_distance(pos, adj)
                dz_dist = self._drop_dist(adj)

                # Soft congestion (#4): penalty for nearby bots
                nearby = 0
                for b, bp in all_bot_positions.items():
                    if b != bid:
                        md = abs(bp[0] - adj[0]) + abs(bp[1] - adj[1])
                        if md <= 2:
                            nearby += 1

                # Soft contention (#4): penalty for bots targeting same type nearby
                contention = 0
                for b, bp in all_bot_positions.items():
                    if b != bid:
                        md = abs(bp[0] - adj[0]) + abs(bp[1] - adj[1])
                        if md <= 4:
                            contention += 1

                cost = dist + dz_dist * 1.0 + nearby * 0.6 + contention * 0.5
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: dict[int, tuple[int, tuple[int, int], int]] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_bots and c < n_tasks and cost_matrix[r, c] < 9990:
                bid = bot_ids[r]
                assignments[bid] = tasks[c]

        return assignments

    # ------------------------------------------------------------------
    # Bot role assignment
    # ------------------------------------------------------------------

    def _assign_roles(self, bot_positions: dict[int, tuple[int, int]],
                      bot_inventories: dict[int, list[int]],
                      active_order: Order | None,
                      active_short: dict[int, int],
                      future_orders: list[Order],
                      prefetch_needs: list[dict[int, int]],
                      rnd: int, num_rounds: int,
                      ) -> tuple[dict[int, tuple[int, int]],   # goals
                                 dict[int, str],                # goal_types
                                 dict[int, int]]:               # pickup_targets
        """Assign every bot a goal and goal_type.

        Uses persistent missions (#3) to reduce role churn, Hungarian
        assignment (#1) for globally optimal bot-item matching, soft
        congestion costs (#4), and prefetch gating (#5).

        Phases:
        0. Retain valid persistent missions
        1. Active carriers -> deliver to nearest dropoff
        2. Active shortfall -> Hungarian assignment of empty bots to items
        3. Prefetch -> Hungarian assignment for future order items
        4. Full bots with useful items -> stage at DZ (DZ-only auto-delivery)
        5. Dead bots -> corridor parking
        """
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}
        occupied_goals: set[tuple[int, int]] = set()
        claimed_items: set[int] = set()

        # === Phase 0: Retain valid persistent missions (#3) ===
        valid_missions = self._validate_missions(
            bot_positions, bot_inventories, active_order, active_short,
            future_orders, prefetch_needs, rnd)

        for bid, m in valid_missions.items():
            goals[bid] = m.goal
            goal_types[bid] = m.goal_type
            if m.item_idx >= 0:
                pickup_targets[bid] = m.item_idx
                claimed_items.add(m.item_idx)

        # Classify bots (skip those with valid missions)
        active_needs = _order_needs_map(active_order) if active_order else {}
        active_carriers: list[int] = []
        prefetch_carriers: list[int] = []
        dead_bots: list[int] = []
        empty_bots: list[int] = []

        future_type_set: set[int] = set()
        for forder in future_orders:
            for t in forder.needs():
                future_type_set.add(t)

        for bid in range(self.num_bots):
            if bid in valid_missions:
                continue
            inv = bot_inventories.get(bid, [])
            if not inv:
                empty_bots.append(bid)
                continue
            has_active = any(t in active_needs for t in inv)
            if has_active:
                active_carriers.append(bid)
            elif any(t in future_type_set for t in inv):
                prefetch_carriers.append(bid)
            else:
                dead_bots.append(bid)

        # Track how many bots are assigned to each active type
        active_type_assigned: dict[int, int] = {}
        for bid in active_carriers:
            for t in bot_inventories[bid]:
                if t in active_needs:
                    active_type_assigned[t] = active_type_assigned.get(t, 0) + 1
        # Count mission-retained bots carrying active types
        for bid, m in valid_missions.items():
            if m.goal_type in ('deliver', 'pickup'):
                for t in bot_inventories.get(bid, []):
                    if t in active_needs:
                        active_type_assigned[t] = active_type_assigned.get(t, 0) + 1

        # Dropoff load balancing
        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}
        for bid, m in valid_missions.items():
            if m.goal_type == 'deliver' and m.goal in dropoff_loads:
                dropoff_loads[m.goal] += 1

        # === Phase 1: Active carriers deliver ===
        fill_up_bots: list[int] = []
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)
            total_short = sum(active_short.values())

            if free_slots == 0 or total_short == 0:
                dz = self._balanced_dropoff(pos, dropoff_loads, bid)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
                self._missions[bid] = Mission('deliver', dz, -1, -1, -1, rnd)
            else:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                min_item_dist = self._min_dist_to_types(pos, active_short.keys())
                if min_item_dist < drop_dist and min_item_dist < 8:
                    fill_up_bots.append(bid)
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads, bid)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                    self._missions[bid] = Mission('deliver', dz, -1, -1, -1, rnd)

        # === Phase 1b: Fill-up bots pick more active items ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            bot_types = set(bot_inventories[bid])
            filtered_short = {t: s for t, s in active_short.items()
                              if t not in bot_types or s > 1}
            assigned = False
            if filtered_short:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, filtered_short, active_type_assigned,
                    claimed_items, strict=False)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    active_type_assigned[tid] = active_type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    self._missions[bid] = Mission(
                        'pickup', adj_pos, item_idx, tid, -1, rnd)
                    assigned = True
            if not assigned:
                dz = self._balanced_dropoff(pos, dropoff_loads, bid)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
                self._missions[bid] = Mission('deliver', dz, -1, -1, -1, rnd)

        # === Phase 2: Empty bots fetch active shortfall (Hungarian) ===
        remaining_short: dict[int, int] = {}
        for t, need in active_short.items():
            s = need - active_type_assigned.get(t, 0)
            if s > 0:
                remaining_short[t] = s

        assigned_to_active: set[int] = set()

        if remaining_short and empty_bots:
            # Use Hungarian for globally optimal matching
            hungarian_result = self._hungarian_assign(
                empty_bots, bot_positions, remaining_short,
                claimed_items, bot_positions)

            for bid, (item_idx, adj_pos, tid) in hungarian_result.items():
                goals[bid] = adj_pos
                goal_types[bid] = 'pickup'
                pickup_targets[bid] = item_idx
                claimed_items.add(item_idx)
                assigned_to_active.add(bid)
                self._missions[bid] = Mission(
                    'pickup', adj_pos, item_idx, tid, -1, rnd)
                # Update remaining shortfall
                remaining_short[tid] = remaining_short.get(tid, 0) - 1
                if remaining_short.get(tid, 0) <= 0:
                    remaining_short.pop(tid, None)

        # === Phase 2b: Redundant active fetchers ===
        if active_needs and not remaining_short:
            redundant_needs = dict(active_needs)
            unassigned_empty = [b for b in empty_bots if b not in assigned_to_active]
            if unassigned_empty:
                max_redundant = min(3, len(unassigned_empty))
                redundant_bots = unassigned_empty[:max_redundant]
                hungarian_result = self._hungarian_assign(
                    redundant_bots, bot_positions, redundant_needs,
                    claimed_items, bot_positions)
                for bid, (item_idx, adj_pos, tid) in hungarian_result.items():
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    claimed_items.add(item_idx)
                    assigned_to_active.add(bid)
                    self._missions[bid] = Mission(
                        'pickup', adj_pos, item_idx, tid, -1, rnd)

        # Remaining empty bots
        remaining_empty = [bid for bid in empty_bots if bid not in assigned_to_active]

        # === Phase 3: Prefetch for future orders (Hungarian + gating) ===
        reserve = 2
        prefetch_assigned: list[dict[int, int]] = [{} for _ in future_orders]

        # Prefetch gating: count total prefetch items already carried
        total_prefetch_carrying = 0
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in future_type_set and t not in active_needs:
                    total_prefetch_carrying += 1

        # Dynamic prefetch cap based on rounds remaining
        rounds_left = max(1, num_rounds - rnd)
        dynamic_cap = self.MAX_PREFETCH_CARRYING
        if rounds_left < 100:
            dynamic_cap = max(10, int(dynamic_cap * rounds_left / 100))

        prefetch_allowed = total_prefetch_carrying < dynamic_cap
        active_under_control = sum(active_short.values()) <= self.MAX_DELIVERY // 2

        max_prefetch_empty = max(0, len(remaining_empty) - reserve)

        if prefetch_allowed and active_under_control and max_prefetch_empty > 0:
            # Build combined prefetch needs across all future orders
            combined_prefetch: dict[int, int] = {}
            for fi, needs in enumerate(prefetch_needs):
                for t, n in needs.items():
                    already = prefetch_assigned[fi].get(t, 0)
                    rem = n - already
                    if rem > 0:
                        combined_prefetch[t] = combined_prefetch.get(t, 0) + rem

            prefetch_bots = remaining_empty[:max_prefetch_empty]
            if combined_prefetch and prefetch_bots:
                hungarian_result = self._hungarian_assign(
                    prefetch_bots, bot_positions, combined_prefetch,
                    claimed_items, bot_positions)

                for bid, (item_idx, adj_pos, tid) in hungarian_result.items():
                    # Find which future order this type belongs to
                    fi_assigned = -1
                    for fi, needs in enumerate(prefetch_needs):
                        if tid in needs and needs[tid] > prefetch_assigned[fi].get(tid, 0):
                            fi_assigned = fi
                            break

                    goals[bid] = adj_pos
                    goal_types[bid] = 'prefetch'
                    pickup_targets[bid] = item_idx
                    claimed_items.add(item_idx)
                    if fi_assigned >= 0:
                        prefetch_assigned[fi_assigned][tid] = \
                            prefetch_assigned[fi_assigned].get(tid, 0) + 1
                    self._missions[bid] = Mission(
                        'prefetch', adj_pos, item_idx, tid, fi_assigned, rnd)

        # Park remaining unassigned empty bots
        for bid in remaining_empty:
            if bid not in goals:
                park = self._corridor_parking(bot_positions[bid], occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'
                self._missions[bid] = Mission('park', park, -1, -1, -1, rnd)

        # === Phase 4: Prefetch carriers -- fetch more or stage at DZ ===
        # DZ-only cascade: bots must be AT dropoff for auto-delivery.
        active_nearly_done = sum(active_short.values()) <= 3
        preview_needs_set: set[int] = set()
        if future_orders:
            for t in future_orders[0].needs():
                preview_needs_set.add(t)

        delivery_dzs: set[tuple[int, int]] = set()
        for bid2, gt2 in goal_types.items():
            if gt2 == 'deliver' and bid2 in goals:
                delivery_dzs.add(goals[bid2])

        max_staging = len(self.drop_zones)
        staging_count = 0

        for bid in prefetch_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            if free_slots > 0:
                assigned = False
                for fi, needs in enumerate(prefetch_needs):
                    effective_needs: dict[int, int] = {}
                    bot_types = set(inv)
                    for t, n in needs.items():
                        already = prefetch_assigned[fi].get(t, 0)
                        rem = n - already
                        if t in bot_types and rem <= 1:
                            continue
                        if rem > 0:
                            effective_needs[t] = rem

                    if not effective_needs:
                        continue

                    item_idx, adj_pos = self._assign_item(
                        bid, pos, effective_needs, prefetch_assigned[fi],
                        claimed_items, strict=True)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'prefetch'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        prefetch_assigned[fi][tid] = prefetch_assigned[fi].get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        self._missions[bid] = Mission(
                            'prefetch', adj_pos, item_idx, tid, fi, rnd)
                        assigned = True
                        break

                if not assigned:
                    park = self._corridor_parking(pos, occupied_goals)
                    occupied_goals.add(park)
                    goals[bid] = park
                    goal_types[bid] = 'park'
                    self._missions[bid] = Mission('park', park, -1, -1, -1, rnd)
            else:
                # Full inventory — check if items match preview (cascade-ready)
                preview_match = sum(1 for t in inv if t in preview_needs_set)
                all_preview = preview_match == len(inv)

                if all_preview and active_nearly_done and staging_count < max_staging:
                    dz = self._cascade_dropoff(pos, dropoff_loads, delivery_dzs)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'stage'
                    staging_count += 1
                    self._missions[bid] = Mission('stage', dz, -1, -1, -1, rnd)
                elif any(t in preview_needs_set for t in inv) and active_nearly_done and staging_count < max_staging:
                    dz = self._cascade_dropoff(pos, dropoff_loads, delivery_dzs)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'stage'
                    staging_count += 1
                    self._missions[bid] = Mission('stage', dz, -1, -1, -1, rnd)
                else:
                    park = self._corridor_parking(pos, occupied_goals)
                    occupied_goals.add(park)
                    goals[bid] = park
                    goal_types[bid] = 'park'
                    self._missions[bid] = Mission('park', park, -1, -1, -1, rnd)

        # === Phase 5: Dead bots -> corridor parking ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'
            self._missions[bid] = Mission('flee', park, -1, -1, -1, rnd)

        # Ensure all bots have goals
        for bid in range(self.num_bots):
            if bid not in goals:
                pos = bot_positions.get(bid, self.spawn)
                park = self._corridor_parking(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'
                self._missions[bid] = Mission('park', park, -1, -1, -1, rnd)

        self._last_goal_types = goal_types
        return goals, goal_types, pickup_targets

    # ------------------------------------------------------------------
    # Main action entry point
    # ------------------------------------------------------------------

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        """Per-round action decision. Returns [(action_type, item_idx)] per bot."""
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # Extract bot positions and inventories
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Update congestion
        self.congestion.update(list(bot_positions.values()))

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Track active order stuck duration
        if state.orders_completed > self._last_orders_completed:
            self._active_stuck_rounds = 0
            self._last_orders_completed = state.orders_completed
        else:
            self._active_stuck_rounds += 1

        # Get orders
        active_order = state.get_active_order()
        preview_order = state.get_preview_order()
        future = self._get_future_orders(state, all_orders)

        # Compute active shortfall
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

        # Commitment tracking and prefetch needs
        commitments = self._build_commitments(
            bot_inventories, active_order, future)
        prefetch_needs = self._compute_prefetch_needs(future, commitments)

        # Role assignment
        goals, goal_types, pickup_targets = self._assign_roles(
            bot_positions, bot_inventories,
            active_order, active_short,
            future, prefetch_needs,
            rnd, num_rounds)

        # Build urgency order with tiebreak rotation
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            rotation = (bid + rnd) % 100
            if gt == 'deliver':
                return (0, dist, rotation)
            elif gt == 'stage':
                return (1, dist, rotation)
            elif gt == 'pickup':
                return (2, dist, rotation)
            elif gt == 'prefetch':
                return (3, dist, rotation)
            elif gt == 'flee':
                return (4, dist, rotation)
            else:
                return (5, dist, rotation)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # Pathfinding with recursive PIBT
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order,
            goal_types=goal_types, round_number=rnd)

        # Build final actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape: override after STALL_LIMIT rounds stuck
            if self.stall_counts.get(bid, 0) >= self.STALL_LIMIT:
                act = self._escape_action(bid, pos, rnd, goal=goal)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF: deliver/stage bots drop off, others with active items too
            if pos in self.drop_set and bot_inventories[bid]:
                if gt in ('deliver', 'stage'):
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                # Any bot at DZ with active-matching items should deliver
                inv = bot_inventories[bid]
                if any(t in active_needs for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET: pick up item
            if gt in ('pickup', 'prefetch') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # OPPORTUNISTIC ADJACENT PICKUP (all bot types)
            if len(bot_inventories[bid]) < INV_CAP:
                opp = self._check_opportunistic_pickup(
                    bid, pos, gt, bot_inventories[bid],
                    active_order, active_short, future, prefetch_needs,
                    commitments)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Use pathfinder's action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    # ------------------------------------------------------------------
    # Opportunistic adjacent pickup
    # ------------------------------------------------------------------

    def _check_opportunistic_pickup(
            self, bid: int, pos: tuple[int, int],
            goal_type: str, bot_inv: list[int],
            active_order: Order | None,
            active_short: dict[int, int],
            future_orders: list[Order],
            prefetch_needs: list[dict[int, int]],
            commitments: dict[int, dict[int, int]]
    ) -> tuple[int, int] | None:
        """Check if any adjacent item is worth opportunistically picking up.

        Uses pos_adj_items for O(1) lookup instead of scanning all items.
        Priority:
        1. Active shortfall items (highest)
        2. Prefetch items for earliest future order with unmet needs
        """
        adj_items = self.pos_adj_items.get(pos, [])
        if not adj_items:
            return None

        bot_types = set(bot_inv)
        total_active_short = sum(active_short.values())

        # Guard: delivery bots with 2+ items should keep a slot for more
        # active items unless active is fully covered
        if goal_type == 'deliver' and len(bot_inv) >= 2 and total_active_short > 0:
            for item_idx, tid in adj_items:
                if tid not in active_short or active_short[tid] <= 0:
                    continue
                if tid in bot_types and active_short[tid] <= 1:
                    continue
                return (ACT_PICKUP, item_idx)
            return None

        # Priority 1: Active shortfall
        for item_idx, tid in adj_items:
            if tid not in active_short or active_short[tid] <= 0:
                continue
            if tid in bot_types and active_short[tid] <= 1:
                continue
            return (ACT_PICKUP, item_idx)

        # Priority 2: Future order prefetch (only if active is covered)
        if total_active_short > 0:
            return None

        for fi, needs in enumerate(prefetch_needs):
            if not needs:
                continue
            for item_idx, tid in adj_items:
                if tid not in needs or needs[tid] <= 0:
                    continue
                if tid in bot_types:
                    continue
                return (ACT_PICKUP, item_idx)

        return None

    # ------------------------------------------------------------------
    # Escape action (anti-stall)
    # ------------------------------------------------------------------

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int,
                       goal: tuple[int, int] | None = None) -> int:
        """Deterministic but varied direction to break stalls."""
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

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
        return min(self.tables.get_distance(pos, dz)
                   for dz in self.drop_zones)

    def _balanced_dropoff(self, pos: tuple[int, int],
                          loads: dict[tuple[int, int], int],
                          bot_id: int | None = None) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            load_pen = loads.get(dz, 0) * 5
            score = d + load_pen
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _cascade_dropoff(self, pos: tuple[int, int],
                         loads: dict[tuple[int, int], int],
                         delivery_dzs: set[tuple[int, int]]
                         ) -> tuple[int, int]:
        """Choose DZ for cascade staging: prefer DZs NOT used by delivery bots."""
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            load_pen = loads.get(dz, 0) * 5
            # Strong bonus for DZs without delivery bots
            delivery_pen = 15 if dz in delivery_dzs else 0
            score = d + load_pen + delivery_pen
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _assign_item(self, bot_id: int, bot_pos: tuple[int, int],
                     needed: dict[int, int],
                     assigned_counts: dict[int, int],
                     claimed: set[int],
                     strict: bool = False,
                     zone_dz: tuple[int, int] | None = None,
                     ) -> tuple[int | None, tuple[int, int] | None]:
        """Find the best item to pick up for a needed type.

        Returns (item_idx, adjacent_position) or (None, None).
        Cost = distance + drop_weight * drop_distance.
        Low drop_weight: all-bots cascade auto-delivers from anywhere.
        """
        best_idx = None
        best_adj = None
        best_cost = 9999.0
        drop_weight = 1.0  # DZ-only cascade: items near DZ deliver faster

        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            max_assign = need_count if strict else need_count + 1
            if assigned_counts.get(tid, 0) >= max_assign:
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    if zone_dz is not None:
                        drop_d = self.tables.get_distance(adj, zone_dz)
                    else:
                        drop_d = self._drop_dist(adj)
                    cost = d + drop_d * drop_weight
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj

        return best_idx, best_adj

    def _corridor_parking(self, pos: tuple[int, int],
                          occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find parking spot in corridor, away from dropoffs."""
        best = self.spawn
        best_d = 9999
        for cy in self._corridor_ys:
            for dx in range(15):
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

    def _min_dist_to_types(self, pos: tuple[int, int], types) -> int:
        """Minimum distance from pos to any item of the given types."""
        best = 9999
        for tid in types:
            for item_idx, adj_cells in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

    # ------------------------------------------------------------------
    # WebSocket live action (for live_gpu_stream.py)
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
        bot_pos_dict: dict[int, tuple[int, int]] = {}
        bot_inv_dict: dict[int, list[int]] = {}
        for bot in live_bots:
            bid = bot['id']
            bot_pos_dict[bid] = tuple(bot['position'])
            inv = []
            for item_name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(item_name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv_dict[bid] = inv

        rnd = data.get('round', 0)
        num_rounds = data.get('max_rounds', 500)

        # Track active stuck
        cur_score = data.get('score', 0)
        cur_completed = data.get('active_order_index', 0)
        if cur_completed > self._last_orders_completed:
            self._active_stuck_rounds = 0
            self._last_orders_completed = cur_completed
        else:
            self._active_stuck_rounds += 1

        # Update congestion and stall
        self.congestion.update(list(bot_pos_dict.values()))
        for bid, pos in bot_pos_dict.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Future orders from capture
        future: list[Order] = []
        if preview_order:
            future.append(preview_order)
        if self.future_orders:
            # Use active_order_index from server to correctly offset future orders
            active_idx = data.get('active_order_index', 0)
            start = active_idx + 2  # skip active + preview
            for i in range(min(start, len(self.future_orders)),
                           min(start + self.PREFETCH_DEPTH, len(self.future_orders))):
                future.append(self.future_orders[i])

        # Active shortfall
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for inv in bot_inv_dict.values():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Commitment + prefetch
        commitments = self._build_commitments(bot_inv_dict, active_order, future)
        prefetch_needs = self._compute_prefetch_needs(future, commitments)

        # Role assignment
        goals, goal_types, pickup_targets = self._assign_roles(
            bot_pos_dict, bot_inv_dict,
            active_order, active_short,
            future, prefetch_needs,
            rnd, num_rounds)

        # Urgency order
        priority_map = {'deliver': 0, 'stage': 1, 'pickup': 2,
                        'prefetch': 3, 'flee': 4, 'park': 5}
        all_bids = [bot['id'] for bot in live_bots]
        urgency_order = sorted(all_bids, key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(
                bot_pos_dict.get(bid, self.spawn),
                goals.get(bid, self.spawn)),
            (bid + rnd) % 100
        ))

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_pos_dict, goals, urgency_order,
            goal_types=goal_types, round_number=rnd)

        # Build WS actions
        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left',
                        'move_right', 'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            inv_names = bot.get('inventory', [])

            # Stall escape
            if self.stall_counts.get(bid, 0) >= self.STALL_LIMIT:
                act = self._escape_action(bid, pos, rnd, goal=goal)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            # At dropoff: deliver/stage bots drop off, others with active items too
            if pos in self.drop_set and inv_names:
                if gt in ('deliver', 'stage'):
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    continue
                # Any bot at DZ with active-matching items should deliver
                inv_types = [ms.type_name_to_id.get(n, -1) for n in inv_names]
                if any(t in active_needs for t in inv_types):
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    continue

            # At pickup target
            if gt in ('pickup', 'prefetch') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    })
                    continue

            # Opportunistic: active items
            if len(inv_names) < INV_CAP and active_short:
                opp = self._ws_active_adjacent(bid, pos, ms, active_short)
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # Opportunistic: prefetch items (when active is covered)
            if len(inv_names) < INV_CAP and not active_short:
                opp = self._ws_prefetch_adjacent(
                    bid, pos, ms, prefetch_needs,
                    set(bot_inv_dict.get(bid, [])))
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # Use pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_active_adjacent(self, bid: int, pos: tuple[int, int],
                            ms: MapState,
                            active_short: dict[int, int]) -> dict | None:
        """Pick up adjacent item if type is still needed by active order."""
        adj_items = self.pos_adj_items.get(pos, [])
        for item_idx, tid in adj_items:
            if tid in active_short:
                return {
                    'bot': bid,
                    'action': 'pick_up',
                    'item_id': ms.items[item_idx]['id'],
                }
        return None

    def _ws_prefetch_adjacent(self, bid: int, pos: tuple[int, int],
                              ms: MapState,
                              prefetch_needs: list[dict[int, int]],
                              bot_types: set) -> dict | None:
        """Pick up adjacent item matching any future order need."""
        adj_items = self.pos_adj_items.get(pos, [])
        if not adj_items:
            return None
        for fi, needs in enumerate(prefetch_needs):
            if not needs:
                continue
            for item_idx, tid in adj_items:
                if tid in needs and needs[tid] > 0 and tid not in bot_types:
                    return {
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    # ------------------------------------------------------------------
    # Simulation entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_sim(seed: int, verbose: bool = False,
                live_map: MapState | None = None,
                **kwargs) -> tuple[int, list]:
        """Run full simulation with cascade solver. Returns (score, action_log).

        If live_map is provided, uses the live server map layout with
        seed-based orders. Otherwise uses procedural map.
        Extra kwargs are passed to CascadeSolver (e.g. PREFETCH_DEPTH=10).
        """
        if live_map is not None:
            all_orders = generate_all_orders(
                seed, live_map, 'nightmare', count=200)
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
            state, all_orders = init_game(seed, 'nightmare', num_orders=200)
            ms = state.map_state

        num_bots = len(state.bot_positions)
        tables = PrecomputedTables.get(ms)
        solver = CascadeSolver(ms, tables, future_orders=all_orders, **kwargs)
        num_rounds = DIFF_ROUNDS['nightmare']

        chains = 0
        max_chain = 0
        action_log = []

        # Utilization tracking
        goal_totals: dict[str, int] = defaultdict(int)
        order_rounds: list[int] = []
        stall_total = 0
        escape_total = 0
        prefetch_total = 0

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd

            # PRE-STEP snapshot for chain diagnosis
            drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
            pre_at_drop: dict[int, tuple[tuple[int, int], list[int]]] = {}
            for b in range(len(state.bot_positions)):
                bp = (int(state.bot_positions[b, 0]),
                      int(state.bot_positions[b, 1]))
                if bp in drop_set:
                    inv = state.bot_inv_list(b)
                    if inv:
                        pre_at_drop[b] = (bp, inv)

            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)

            # Track utilization
            for gt in solver._last_goal_types.values():
                goal_totals[gt] += 1
                if gt == 'prefetch':
                    prefetch_total += 1

            # Track stalls
            for b in range(len(state.bot_positions)):
                if solver.stall_counts.get(b, 0) >= 1:
                    stall_total += 1
                if solver.stall_counts.get(b, 0) >= solver.STALL_LIMIT:
                    escape_total += 1

            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 0:
                order_rounds.append(rnd)
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CASCADE x{c}!" if c > 1 else ""
                # Inventory summary
                total_items = sum(len(state.bot_inv_list(b))
                                  for b in range(num_bots))
                # DZ staging info
                dz_bots_with_items = 0
                for b in range(num_bots):
                    bp = (int(state.bot_positions[b, 0]),
                          int(state.bot_positions[b, 1]))
                    if bp in drop_set and state.bot_inv_count(b) > 0:
                        dz_bots_with_items += 1
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d} "
                      f"Inv={total_items:2d}/60 DZ={dz_bots_with_items}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
            if solver.chain_events:
                print(f"Chain events: {solver.chain_events}")
            # Utilization summary
            avg_per_rnd = {gt: cnt / num_rounds
                           for gt, cnt in sorted(goal_totals.items())}
            working = (avg_per_rnd.get('deliver', 0)
                       + avg_per_rnd.get('pickup', 0)
                       + avg_per_rnd.get('prefetch', 0)
                       + avg_per_rnd.get('stage', 0))
            idle = avg_per_rnd.get('flee', 0) + avg_per_rnd.get('park', 0)
            print(f"Avg/rnd: {' '.join(f'{gt}={v:.1f}' for gt, v in avg_per_rnd.items())}")
            if working + idle > 0:
                print(f"Working={working:.1f} Idle={idle:.1f} "
                      f"({idle/(working+idle)*100:.0f}% idle)")
            print(f"Prefetch assignments: {prefetch_total} "
                  f"({prefetch_total/num_rounds:.1f}/rnd)")
            if len(order_rounds) > 1:
                gaps = [order_rounds[i+1] - order_rounds[i]
                        for i in range(len(order_rounds) - 1)]
                print(f"Order gaps: avg={np.mean(gaps):.1f} "
                      f"min={min(gaps)} max={max(gaps)}")
            print(f"Stalls: {stall_total} ({stall_total/num_rounds:.1f}/rnd) "
                  f"Escapes: {escape_total} ({escape_total/num_rounds:.2f}/rnd)")

        return state.score, action_log


# ---------------------------------------------------------------------------
# PostgreSQL recording (same pattern as nightmare_solver_v2.py)
# ---------------------------------------------------------------------------

DB_URL = "postgres://grocery:grocery123@localhost:5433/grocery_bot"


def record_to_pg(seed, score, orders_completed, items_delivered,
                 action_log, elapsed, live_map=None):
    """Record run to PostgreSQL."""
    import json
    import os
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("  psycopg2 not installed, skipping DB recording",
              file=sys.stderr)
        return None

    db_url = os.environ.get("GROCERY_DB_URL", DB_URL)
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        from game_engine import (build_map, CELL_WALL, CELL_SHELF,
                                 state_to_ws_format, actions_to_ws_format)

        if live_map is not None:
            ms = live_map
        else:
            ms = build_map('nightmare')
        cfg = CONFIGS['nightmare']

        walls = []
        shelves = []
        for y in range(ms.height):
            for x in range(ms.width):
                c = int(ms.grid[y, x])
                if c == CELL_WALL:
                    walls.append([x, y])
                elif c == CELL_SHELF:
                    shelves.append([x, y])

        items = [{"id": it["id"], "type": it["type"],
                  "position": list(it["position"])}
                 for it in ms.items]

        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height,
                              bot_count, item_types, order_size_min,
                              order_size_max, walls, shelves, items,
                              drop_off, spawn, final_score,
                              items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s)
            RETURNING id
        """, (
            seed, 'nightmare', ms.width, ms.height, cfg['bots'],
            ms.num_types, cfg['order_size'][0], cfg['order_size'][1],
            json.dumps(walls), json.dumps(shelves),
            json.dumps(items),
            json.dumps(list(ms.drop_off)),
            json.dumps(list(ms.spawn)),
            score, items_delivered, orders_completed,
            'synthetic',
        ))
        run_id = cur.fetchone()[0]

        # Insert round data
        if action_log:
            if live_map is not None:
                all_orders2 = generate_all_orders(
                    seed, live_map, 'nightmare', count=200)
                num_bots = cfg['bots']
                gs = GameState(live_map)
                gs.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
                gs.bot_inventories = np.full(
                    (num_bots, INV_CAP), -1, dtype=np.int8)
                for i in range(num_bots):
                    gs.bot_positions[i] = [live_map.spawn[0],
                                           live_map.spawn[1]]
                gs.orders = [all_orders2[0].copy(), all_orders2[1].copy()]
                gs.orders[0].status = 'active'
                gs.orders[1].status = 'preview'
                gs.next_order_idx = 2
                gs.active_idx = 0
            else:
                gs, all_orders2 = init_game(seed, 'nightmare', num_orders=200)

            round_tuples = []
            for rnd in range(min(len(action_log), 500)):
                gs.round = rnd
                ws_data = state_to_ws_format(gs, all_orders2)
                ws_acts = actions_to_ws_format(action_log[rnd], gs.map_state)
                bots = [{"id": b["id"], "position": b["position"],
                         "inventory": b.get("inventory", [])}
                        for b in ws_data["bots"]]
                orders = [{"id": o["id"],
                           "items_required": o["items_required"],
                           "items_delivered": o.get("items_delivered", []),
                           "status": o.get("status", "active")}
                          for o in ws_data.get("orders", [])]
                round_tuples.append((
                    run_id, rnd, json.dumps(bots), json.dumps(orders),
                    json.dumps(ws_acts), ws_data["score"], json.dumps([])
                ))
                step(gs, action_log[rnd], all_orders2)

            execute_values(cur, """
                INSERT INTO rounds (run_id, round_number, bots, orders,
                                    actions, score, events)
                VALUES %s
            """, round_tuples, page_size=100)

        conn.commit()
        conn.close()
        print(f"  Recorded to DB: run_id={run_id}", file=sys.stderr)
        return run_id
    except Exception as e:
        print(f"  DB error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Nightmare cascade solver -- exploit auto-delivery chains')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-record', action='store_true',
                        help='Skip PostgreSQL recording')
    parser.add_argument('--no-live-map', action='store_true',
                        help='Use procedural map instead of live server map')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    # Load live map from captured data (default for nightmare)
    live_map = None
    if not args.no_live_map:
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap and cap.get('grid'):
                live_map = build_map_from_capture(cap)
                print(f"Using live map: {live_map.width}x{live_map.height}, "
                      f"{live_map.num_items} items, "
                      f"{sum(1 for y in range(live_map.height) for x in range(live_map.width) if live_map.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF))} walkable",
                      file=sys.stderr)
            else:
                print("No capture data found, using procedural map",
                      file=sys.stderr)
        except Exception as e:
            print(f"Could not load live map: {e}, using procedural map",
                  file=sys.stderr)

    scores = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        score, action_log = CascadeSolver.run_sim(
            seed, verbose=args.verbose, live_map=live_map)
        elapsed = time.time() - st
        scores.append(score)
        print(f"Seed {seed}: {score}")

        if not args.no_record:
            # Replay for DB recording
            if live_map is not None:
                all_orders2 = generate_all_orders(
                    seed, live_map, 'nightmare', count=200)
                num_bots = CONFIGS['nightmare']['bots']
                state2 = GameState(live_map)
                state2.bot_positions = np.zeros(
                    (num_bots, 2), dtype=np.int16)
                state2.bot_inventories = np.full(
                    (num_bots, INV_CAP), -1, dtype=np.int8)
                for i in range(num_bots):
                    state2.bot_positions[i] = [live_map.spawn[0],
                                               live_map.spawn[1]]
                state2.orders = [all_orders2[0].copy(),
                                 all_orders2[1].copy()]
                state2.orders[0].status = 'active'
                state2.orders[1].status = 'preview'
                state2.next_order_idx = 2
                state2.active_idx = 0
            else:
                state2, all_orders2 = init_game(
                    seed, 'nightmare', num_orders=200)
            for rnd, acts in enumerate(action_log):
                state2.round = rnd
                step(state2, acts, all_orders2)
            record_to_pg(seed, score, state2.orders_completed,
                         state2.items_delivered, action_log, elapsed,
                         live_map=live_map)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()

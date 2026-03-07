"""Chain reaction planner for nightmare mode.

Computes optimal item assignments to maximize chain reactions.
When a TRIGGER bot completes the active order, all bots staged at dropoff
zones auto-deliver their matching items. If the new active order also completes,
the chain continues — one ACT_DROPOFF can cascade through 3-5+ orders.

Key algorithm:
1. Compute chain_value for each item type across future orders
2. Assign STAGING bots the 3 highest-value types covering consecutive orders
3. Simulate the chain to verify it won't break
4. Output: trigger assignments, staging assignments, fetch assignments
"""
from __future__ import annotations

from game_engine import Order, MapState, INV_CAP
from precompute import PrecomputedTables


class ChainPlan:
    """Result of chain planning."""
    __slots__ = [
        'trigger_bots', 'trigger_types',   # bids and types needed to complete active
        'stage_assignments',  # {bid: (dropoff_zone, [type_ids])}
        'fetch_assignments',  # {bid: type_id to pick up}
        'idle_bots',          # bids with dead inventory
        'expected_chain_len', # predicted chain length if trigger fires
        'future_type_values', # {type_id: chain_value}
    ]

    def __init__(self):
        self.trigger_bots: list[int] = []
        self.trigger_types: dict[int, list[int]] = {}  # bid -> [types to pick]
        self.stage_assignments: dict[int, tuple[tuple[int, int], list[int]]] = {}
        self.fetch_assignments: dict[int, int] = {}
        self.idle_bots: list[int] = []
        self.expected_chain_len = 0
        self.future_type_values: dict[int, float] = {}


class ChainPlanner:
    """Plans chain reactions across future orders."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        # Max staging bots per dropoff zone
        self.max_per_zone = 2

    def chain_value(self, type_id: int, future_orders: list[Order],
                    depth: int = 6) -> float:
        """How many of the next `depth` orders need this type?

        Higher = more valuable to carry to dropoff for chain reactions.
        Also weights earlier orders higher (exponential decay).
        """
        value = 0.0
        for i, order in enumerate(future_orders[:depth]):
            needs = order.needs()
            count = sum(1 for t in needs if t == type_id)
            if count > 0:
                # Earlier orders in the chain are worth more
                weight = 1.0 / (1.0 + i * 0.3)
                value += count * weight
        return value

    def compute_type_values(self, future_orders: list[Order],
                            depth: int = 6) -> dict[int, float]:
        """Compute chain value for all types across future orders."""
        values: dict[int, float] = {}
        all_types_seen: set[int] = set()
        for order in future_orders[:depth]:
            for t in order.needs():
                all_types_seen.add(t)
        for tid in all_types_seen:
            values[tid] = self.chain_value(tid, future_orders, depth)
        return values

    def optimal_staging_inventory(self, future_orders: list[Order],
                                  already_staged: dict[int, int],
                                  depth: int = 6) -> list[int]:
        """Compute the best 3-item inventory for a staging bot.

        Picks the 3 types with highest chain value that aren't already
        over-represented in staged inventories.

        Args:
            future_orders: upcoming orders
            already_staged: {type_id: count} of items already staged at dropoffs
            depth: how many future orders to consider

        Returns:
            List of up to 3 type_ids to carry.
        """
        values = self.compute_type_values(future_orders, depth)
        if not values:
            return []

        # Count how many of each type are needed across future orders
        type_needed: dict[int, int] = {}
        for order in future_orders[:depth]:
            for t in order.needs():
                type_needed[t] = type_needed.get(t, 0) + 1

        # Score each type: chain_value minus penalty for over-staging
        scored: list[tuple[float, int]] = []
        for tid, val in values.items():
            staged = already_staged.get(tid, 0)
            needed = type_needed.get(tid, 0)
            # Diminishing returns: each additional staged copy of same type is less valuable
            if staged >= needed:
                effective_val = val * 0.1  # already fully covered
            else:
                effective_val = val * (1.0 - staged / max(needed, 1) * 0.5)
            scored.append((effective_val, tid))

        scored.sort(reverse=True)
        return [tid for _, tid in scored[:INV_CAP]]

    def simulate_chain(self, future_orders: list[Order],
                       staged_inventories: dict[int, list[int]],
                       staged_zones: dict[int, tuple[int, int]]) -> int:
        """Simulate a chain reaction given staged bot inventories.

        Assumes the active order has just been completed by a trigger bot.
        Returns the number of additional orders completed by the chain.

        Args:
            future_orders: orders that will become active in sequence
            staged_inventories: {bid: [type_ids]} for bots at dropoff
            staged_zones: {bid: (x, y)} dropoff zone position

        Returns:
            Number of chain completions (0 = no chain)
        """
        if not future_orders:
            return 0

        # Copy inventories (we'll deplete them during simulation)
        inv_copy = {bid: list(types) for bid, types in staged_inventories.items()}
        chain_len = 0

        for order in future_orders:
            needs = list(order.needs())
            if not needs:
                chain_len += 1
                continue

            # Try auto-delivery from staged bots
            remaining = list(needs)
            for bid, inv in inv_copy.items():
                if not inv:
                    continue
                new_inv = []
                for t in inv:
                    if t in remaining:
                        remaining.remove(t)
                    else:
                        new_inv.append(t)
                inv_copy[bid] = new_inv

            if not remaining:
                chain_len += 1
            else:
                break  # Chain breaks

        return chain_len

    def plan_chain(self, active_order: Order | None,
                   future_orders: list[Order],
                   bot_positions: dict[int, tuple[int, int]],
                   bot_inventories: dict[int, list[int]]) -> ChainPlan:
        """Main planning entry point.

        Given current state + future order sequence, compute:
        - Which types each bot should carry
        - Which bots stage at which dropoff zone
        - Expected chain length

        Returns a ChainPlan with all assignments.
        """
        plan = ChainPlan()

        if not future_orders:
            plan.future_type_values = {}
            return plan

        # Compute chain values for all types
        plan.future_type_values = self.compute_type_values(future_orders)

        # Classify bots by current inventory
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Identify which bots are already carrying chain-valuable items
        # and which are at or near dropoff zones
        for bid, inv in bot_inventories.items():
            pos = bot_positions[bid]

            if not inv:
                # Empty bot — candidate for FETCH
                continue

            # Check if carrying active-needed items
            has_active = any(t in active_needs for t in inv)
            if has_active:
                continue  # Will be handled as TRIGGER/DELIVER

            # Check chain value of inventory
            inv_chain_value = sum(plan.future_type_values.get(t, 0) for t in inv)
            if inv_chain_value > 0:
                # Bot has chain-valuable items — candidate for STAGE
                pass
            else:
                plan.idle_bots.append(bid)

        # Estimate chain length with currently staged inventories
        # (actual staging assignments happen in the allocator)
        staged_inv: dict[int, list[int]] = {}
        staged_zones: dict[int, tuple[int, int]] = {}
        for bid, inv in bot_inventories.items():
            pos = bot_positions[bid]
            if pos in self.drop_set and inv:
                has_active = any(t in active_needs for t in inv)
                if not has_active:
                    staged_inv[bid] = inv
                    staged_zones[bid] = pos

        plan.expected_chain_len = self.simulate_chain(
            future_orders, staged_inv, staged_zones)

        return plan

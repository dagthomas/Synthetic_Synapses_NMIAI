"""Nightmare Cascade V2: Trip-committed cascade pipeline.

Key insight: V6 replans every round → staging bots flip-flop → no cascades.
V2 commits staging bots to multi-round trips (pick preview → stage at dropoff).
Active bots use V6's proven allocation. Only staging bots are committed.

Target: 90% cascade rate → 6.6 rounds/order → 76 orders → 800 pts.
"""
from __future__ import annotations
import sys, time, random, copy
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
from nightmare_solver_v6 import V6Allocator, NightmareSolverV6

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]

# Staging bot states
ST_IDLE = 0       # Available for assignment
ST_PICKUP = 1     # Going to pick up preview item
ST_FILL = 2       # Going to pick up more preview items (fill slots)
ST_TRAVEL = 3     # Going to dropoff
ST_STAGED = 4     # At dropoff, waiting for cascade


class StagingBot:
    """Tracks a committed staging bot's multi-round trip."""
    __slots__ = ['bid', 'state', 'target_items', 'target_dropoff',
                 'pickup_queue', 'current_target', 'current_item_idx',
                 'rounds_staged']

    def __init__(self, bid, dropoff):
        self.bid = bid
        self.state = ST_IDLE
        self.target_dropoff = dropoff
        self.target_items = []  # type IDs to pick up
        self.pickup_queue = []  # [(item_idx, adj_pos)] remaining pickups
        self.current_target = None  # current goal position
        self.current_item_idx = -1  # current item to pick up
        self.rounds_staged = 0


class NightmareCascadeV2:
    """Cascade pipeline solver: V6 for active + committed staging bots."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']
        self.future_orders = future_orders or []

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # V6 allocator for active bots
        self.allocator = V6Allocator(ms, tables, self.drop_zones,
                                     max_preview_pickers=99, drop_d_weight=0.8)

        # Type → [(item_idx, adj_positions, zone)]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            ix = int(ms.item_positions[idx, 0])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        # Staging management
        self.staging_bots: dict[int, StagingBot] = {}  # bid → StagingBot
        self._last_active_order_id = -1
        self._staging_claimed_items: set[int] = set()

        # V6 state tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._pos_history: dict[int, list[tuple[int, int]]] = {}

    def _find_preview_items(self, pos, preview_needs, claimed, max_items=3):
        """Find up to max_items preview items to pick up, ordered by cost."""
        results = []
        assigned_types = {}

        for _ in range(max_items):
            best_idx = None
            best_adj = None
            best_cost = 9999
            best_type = -1

            for tid, need in preview_needs.items():
                if assigned_types.get(tid, 0) >= need:
                    continue
                for item_idx, adj_cells, zone in self.type_items.get(tid, []):
                    if item_idx in claimed:
                        continue
                    for adj in adj_cells:
                        d = self.tables.get_distance(pos, adj)
                        cost = d
                        if cost < best_cost:
                            best_cost = cost
                            best_idx = item_idx
                            best_adj = adj
                            best_type = tid

            if best_idx is not None:
                results.append((best_idx, best_adj, best_type))
                claimed.add(best_idx)
                assigned_types[best_type] = assigned_types.get(best_type, 0) + 1
                pos = best_adj  # Next pickup starts from here
            else:
                break

        return results

    def _assign_staging_bots(self, bot_positions, bot_inventories,
                              preview_order, active_order):
        """Assign/update staging bots for cascade preparation."""
        if not preview_order:
            return

        preview_needs = {}
        for t in preview_order.needs():
            preview_needs[t] = preview_needs.get(t, 0) + 1

        # Remove staging bots that already have items matching ACTIVE order
        # (their items were cascade-delivered, they should rejoin active work)
        active_needs = set()
        if active_order:
            for t in active_order.needs():
                active_needs.add(t)

        for bid in list(self.staging_bots.keys()):
            sb = self.staging_bots[bid]
            inv = bot_inventories.get(bid, [])
            # If staging bot has active items, release it
            if any(t in active_needs for t in inv):
                del self.staging_bots[bid]
                continue
            # If staging bot has no preview items and is idle, release it
            if sb.state == ST_IDLE and not any(t in preview_needs for t in inv):
                del self.staging_bots[bid]

        # Count already staged items
        staged_items = {}
        for bid, sb in self.staging_bots.items():
            inv = bot_inventories.get(bid, [])
            for t in inv:
                if t in preview_needs:
                    staged_items[t] = staged_items.get(t, 0) + 1

        # How many items still needed for staging?
        staging_short = {}
        for t, need in preview_needs.items():
            s = need - staged_items.get(t, 0)
            if s > 0:
                staging_short[t] = s

        if not staging_short:
            return  # All preview items staged or being staged

        # Find dropoffs that need stagers
        stager_at_dropoff = {}
        for bid, sb in self.staging_bots.items():
            if sb.target_dropoff:
                stager_at_dropoff[sb.target_dropoff] = stager_at_dropoff.get(sb.target_dropoff, 0) + 1

        # Need stagers at free dropoffs
        free_dropoffs = [dz for dz in self.drop_zones
                         if stager_at_dropoff.get(dz, 0) < 1]

        if not free_dropoffs:
            return

        # Find available bots (empty, not already staging, not carrying active items)
        available = []
        for bid in range(self.num_bots):
            if bid in self.staging_bots:
                continue
            inv = bot_inventories.get(bid, [])
            if any(t in active_needs for t in inv):
                continue  # Active carrier, don't steal
            if len(inv) >= INV_CAP and not any(t in preview_needs for t in inv):
                continue  # Full with dead items
            available.append(bid)

        # Sort by proximity to preview items
        def proximity(bid):
            pos = bot_positions[bid]
            best = 999
            for tid in staging_short:
                for _, adj_cells, _ in self.type_items.get(tid, []):
                    for adj in adj_cells:
                        d = self.tables.get_distance(pos, adj)
                        if d < best:
                            best = d
            return best

        available.sort(key=proximity)

        # Assign up to len(free_dropoffs) new staging bots
        for i, bid in enumerate(available[:len(free_dropoffs)]):
            dz = free_dropoffs[i]
            pos = bot_positions[bid]
            inv = bot_inventories.get(bid, [])
            free_slots = INV_CAP - len(inv)

            if free_slots <= 0:
                continue

            # Find items to pick up
            pickups = self._find_preview_items(
                pos, staging_short, self._staging_claimed_items,
                max_items=min(free_slots, sum(staging_short.values())))

            if not pickups:
                continue

            sb = StagingBot(bid, dz)
            sb.pickup_queue = [(idx, adj) for idx, adj, _ in pickups]
            sb.target_items = [t for _, _, t in pickups]
            sb.current_item_idx = pickups[0][0]
            sb.current_target = pickups[0][1]
            sb.state = ST_PICKUP
            self.staging_bots[bid] = sb

            # Update staging_short
            for _, _, t in pickups:
                staging_short[t] = staging_short.get(t, 1) - 1
                if staging_short[t] <= 0:
                    del staging_short[t]

    def _update_staging_states(self, bot_positions, bot_inventories, preview_order):
        """Update staging bot states based on current positions."""
        preview_types = set()
        if preview_order:
            for t in preview_order.needs():
                preview_types.add(t)

        for bid, sb in list(self.staging_bots.items()):
            pos = bot_positions[bid]
            inv = bot_inventories.get(bid, [])

            if sb.state == ST_PICKUP:
                # At pickup target?
                if pos == sb.current_target:
                    # Will pick up this round, advance queue
                    sb.pickup_queue = sb.pickup_queue[1:]
                    if sb.pickup_queue:
                        sb.current_item_idx = sb.pickup_queue[0][0]
                        sb.current_target = sb.pickup_queue[0][1]
                        sb.state = ST_PICKUP  # Continue picking
                    else:
                        sb.state = ST_TRAVEL
                        sb.current_target = sb.target_dropoff

            elif sb.state == ST_TRAVEL:
                if pos == sb.target_dropoff:
                    sb.state = ST_STAGED
                    sb.rounds_staged = 0

            elif sb.state == ST_STAGED:
                sb.rounds_staged += 1
                # If staged too long (>30 rounds), something went wrong, release
                if sb.rounds_staged > 40:
                    del self.staging_bots[bid]

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        ms = self.ms
        num_bots = len(state.bot_positions)

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

        # Detect order change
        order_id = active_order.id if active_order else -1
        if order_id != self._last_active_order_id:
            # New order! Former staging bots with items matching NEW active should deliver
            for bid in list(self.staging_bots.keys()):
                sb = self.staging_bots[bid]
                inv = bot_inventories.get(bid, [])
                if active_order and any(active_order.needs_type(t) for t in inv):
                    # This bot has items for new active order → release for delivery
                    del self.staging_bots[bid]
            self._staging_claimed_items.clear()
            self._last_active_order_id = order_id

        # Update staging bot states
        self._update_staging_states(bot_positions, bot_inventories, preview_order)

        # Assign new staging bots if needed
        self._assign_staging_bots(bot_positions, bot_inventories,
                                   preview_order, active_order)

        # Build staging goals
        staging_goals = {}
        staging_goal_types = {}
        staging_pickup_targets = {}
        staging_bids = set(self.staging_bots.keys())

        for bid, sb in self.staging_bots.items():
            if sb.state == ST_PICKUP:
                staging_goals[bid] = sb.current_target
                staging_goal_types[bid] = 'pickup'
                staging_pickup_targets[bid] = sb.current_item_idx
            elif sb.state == ST_TRAVEL:
                staging_goals[bid] = sb.target_dropoff
                staging_goal_types[bid] = 'stage'
            elif sb.state == ST_STAGED:
                staging_goals[bid] = sb.target_dropoff
                staging_goal_types[bid] = 'stage'

        # Use V6 allocator for non-staging bots
        non_staging_positions = {bid: pos for bid, pos in bot_positions.items()
                                 if bid not in staging_bids}
        non_staging_inventories = {bid: inv for bid, inv in bot_inventories.items()
                                   if bid not in staging_bids}

        # Get V6 allocation for active bots
        active_needs = {}
        carrying_active = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid, inv in bot_inventories.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        future_orders = []
        if self.future_orders:
            idx = state.next_order_idx
            for i in range(8):
                if idx + i < len(self.future_orders):
                    future_orders.append(self.future_orders[idx + i])

        v6_goals, v6_goal_types, v6_pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, 500,
            future_orders=future_orders)

        # Merge: staging bots use staging goals, others use V6 goals
        goals = {}
        goal_types = {}
        pickup_targets = {}

        for bid in range(num_bots):
            if bid in staging_goals:
                goals[bid] = staging_goals[bid]
                goal_types[bid] = staging_goal_types[bid]
                if bid in staging_pickup_targets:
                    pickup_targets[bid] = staging_pickup_targets[bid]
            elif bid in v6_goals:
                goals[bid] = v6_goals[bid]
                goal_types[bid] = v6_goal_types.get(bid, 'park')
                if bid in v6_pickup_targets:
                    pickup_targets[bid] = v6_pickup_targets[bid]
            else:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif bid in staging_bids and gt == 'stage':
                return (1, dist)  # Staging bots get priority
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 4, dist)
            elif gt == 'pickup':
                return (2, dist)
            elif gt in ('stage', 'preview'):
                return (3, dist)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            is_staging = bid in staging_bids

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                dirs = MOVES[:]
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    if (nx, ny) in self.walkable:
                        actions[bid] = (a, -1)
                        break
                continue

            # At dropoff
            if pos in self.drop_set:
                if is_staging:
                    # Staging bot at dropoff: WAIT (cascade handles delivery)
                    actions[bid] = (ACT_WAIT, -1)
                    continue
                elif gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # At pickup target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup for active items (non-staging bots only)
            if not is_staging and len(bot_inventories[bid]) < INV_CAP and active_order:
                picked = False
                for item_idx in range(ms.num_items):
                    tid = int(ms.item_types[item_idx])
                    if tid not in active_short:
                        continue
                    for adj in ms.item_adjacencies.get(item_idx, []):
                        if adj == pos:
                            actions[bid] = (ACT_PICKUP, item_idx)
                            picked = True
                            break
                    if picked:
                        break
                if picked:
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareCascadeV2(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        action_log = []
        cascade_items = 0

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            s_before = state.score
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 0:
                bonus = 5 * c
                items = (state.score - s_before) - bonus
                if c > 1:
                    cascade_items += items

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                n_staging = len(solver.staging_bots)
                staged = sum(1 for sb in solver.staging_bots.values()
                            if sb.state == ST_STAGED)
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + f" stg={n_staging}({staged}at)" + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nCascade V2: {state.score} pts, {state.orders_completed} orders, "
                  f"{elapsed:.1f}s")

        return state.score, action_log


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7009
    score, actions = NightmareCascadeV2.run_sim(seed, verbose=True)
    print(f"\nFinal: {score}", file=sys.stderr)

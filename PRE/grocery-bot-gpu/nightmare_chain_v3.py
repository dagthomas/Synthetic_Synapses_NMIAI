#!/usr/bin/env python3
"""Nightmare Chain Cascade Solver V3.

Key insight: serial order ceiling is ~353 pts. Leader at 1032 requires chains.
Strategy: split 20 bots into waves that operate on different orders simultaneously.

Wave 0 (Active): 7 bots complete active order
Wave 1 (Preview): 7 bots pre-fetch preview items, stage at dropoffs
Wave 2 (Future): 6 bots pre-fetch N+2 items, stage at dropoffs

When Active wave delivers last item → chain fires:
- Preview items auto-deliver → preview order completes
- Future items auto-deliver → N+2 order completes (if fully staged)
- 3 orders in 1 trigger event!

After chain: rotate waves (Preview→Active, Future→Preview, freed bots→Future)
"""
from __future__ import annotations

import sys, time, copy
from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

sys.stdout.reconfigure(encoding='utf-8')

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
NUM_BOTS = CONFIGS['nightmare']['bots']
NUM_ROUNDS = DIFF_ROUNDS['nightmare']

# Wave assignments
WAVE_ACTIVE = 0
WAVE_PREVIEW = 1
WAVE_FUTURE = 2

# Bot task states
TASK_PICKUP = 'pickup'      # Going to pick up an item
TASK_DELIVER = 'deliver'    # Going to dropoff to deliver active items
TASK_STAGE = 'stage'        # Going to dropoff to stage (preview/future items)
TASK_STAGED = 'staged'      # At dropoff, waiting for chain
TASK_IDLE = 'idle'          # No task
TASK_HOLD = 'hold'          # At dropoff, holding delivery for chain timing


class BotTask:
    __slots__ = ['bid', 'wave', 'task', 'target_pos', 'item_idx',
                 'type_id', 'dropoff', 'items_to_pickup', 'rounds_waiting']

    def __init__(self, bid):
        self.bid = bid
        self.wave = WAVE_ACTIVE
        self.task = TASK_IDLE
        self.target_pos = None
        self.item_idx = -1
        self.type_id = -1
        self.dropoff = None
        self.items_to_pickup = []  # [(item_idx, adj_pos, type_id)]
        self.rounds_waiting = 0


class NightmareChainV3:
    """Chain cascade solver with wave-based bot management."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None,
                 active_bots: int = 8, preview_bots: int = 7,
                 max_hold: int = 5):
        self.ms = ms
        self.tables = tables
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.future_orders = future_orders or []
        self.n_active = active_bots
        self.n_preview = preview_bots
        self.n_future = NUM_BOTS - active_bots - preview_bots
        self.max_hold = max_hold

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, tables, self.traffic, self.congestion)

        # Item lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Bot management
        self.tasks: dict[int, BotTask] = {}
        for bid in range(NUM_BOTS):
            self.tasks[bid] = BotTask(bid)

        # Wave assignment
        self._assign_waves_initial()

        self._last_active_id = -1
        self._last_preview_id = -1
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

    def _assign_waves_initial(self):
        """Initial wave assignment by bot ID."""
        for bid in range(NUM_BOTS):
            if bid < self.n_active:
                self.tasks[bid].wave = WAVE_ACTIVE
            elif bid < self.n_active + self.n_preview:
                self.tasks[bid].wave = WAVE_PREVIEW
            else:
                self.tasks[bid].wave = WAVE_FUTURE

    def _wave_bots(self, wave: int) -> list[int]:
        return [bid for bid, t in self.tasks.items() if t.wave == wave]

    def _nearest_drop(self, pos):
        return min(self.drop_zones,
                   key=lambda dz: self.tables.get_distance(pos, dz))

    def _find_item(self, pos, type_id, claimed):
        """Find nearest item of type_id."""
        best_idx = -1
        best_adj = None
        best_cost = 9999
        for item_idx, adj_cells in self.type_items.get(type_id, []):
            if item_idx in claimed:
                continue
            for adj in adj_cells:
                d = self.tables.get_distance(pos, adj)
                if d < best_cost:
                    best_cost = d
                    best_idx = item_idx
                    best_adj = adj
        return best_idx, best_adj

    def _assign_wave_tasks(self, wave, order, bot_positions, bot_inventories,
                           claimed_items):
        """Assign tasks to bots in a wave for a given order."""
        if not order:
            return

        bots = self._wave_bots(wave)
        needs = list(order.needs())
        if not needs:
            return

        # Count what's already in inventory or being picked up
        need_count: dict[int, int] = {}
        for t in needs:
            need_count[t] = need_count.get(t, 0) + 1

        # Subtract items already in inventory
        for bid in bots:
            for t in bot_inventories.get(bid, []):
                if t in need_count and need_count[t] > 0:
                    need_count[t] -= 1

        # Subtract items being picked up
        for bid in bots:
            task = self.tasks[bid]
            if task.task == TASK_PICKUP and task.type_id >= 0:
                if task.type_id in need_count and need_count[task.type_id] > 0:
                    need_count[task.type_id] -= 1

        # Types still needed
        types_needed = []
        for t, c in need_count.items():
            for _ in range(c):
                types_needed.append(t)

        if not types_needed:
            # All items covered — assign delivery/staging
            is_active = (wave == WAVE_ACTIVE)
            for bid in bots:
                task = self.tasks[bid]
                inv = bot_inventories.get(bid, [])
                pos = bot_positions[bid]

                if task.task in (TASK_STAGED, TASK_HOLD):
                    continue  # Already at dropoff

                if inv:
                    has_order_items = any(order.needs_type(t) for t in inv)
                    if has_order_items:
                        dz = self._nearest_drop(pos)
                        task.target_pos = dz
                        task.dropoff = dz
                        task.task = TASK_DELIVER if is_active else TASK_STAGE
                    elif task.task == TASK_IDLE:
                        # Has items but not for this order — park
                        task.task = TASK_IDLE
                elif task.task not in (TASK_PICKUP,):
                    task.task = TASK_IDLE
            return

        # Assign pickups to available bots
        available = []
        for bid in bots:
            task = self.tasks[bid]
            inv = bot_inventories.get(bid, [])
            if len(inv) >= INV_CAP:
                continue  # Full
            if task.task == TASK_PICKUP:
                continue  # Already picking up
            if task.task in (TASK_STAGED, TASK_HOLD):
                continue  # At dropoff staging
            available.append(bid)

        # Sort by proximity to needed items
        def item_proximity(bid):
            pos = bot_positions[bid]
            best = 9999
            for tid in types_needed:
                for _, adj_cells in self.type_items.get(tid, []):
                    for adj in adj_cells:
                        d = self.tables.get_distance(pos, adj)
                        if d < best:
                            best = d
            return best

        available.sort(key=item_proximity)

        for bid in available:
            if not types_needed:
                break
            pos = bot_positions[bid]
            task = self.tasks[bid]

            # Find best item to pick up
            best_type = -1
            best_idx = -1
            best_adj = None
            best_cost = 9999

            for tid in set(types_needed):
                idx, adj = self._find_item(pos, tid, claimed_items)
                if idx >= 0:
                    dz = self._nearest_drop(adj)
                    cost = (self.tables.get_distance(pos, adj) +
                            self.tables.get_distance(adj, dz) * 0.4)
                    if cost < best_cost:
                        best_cost = cost
                        best_type = tid
                        best_idx = idx
                        best_adj = adj

            if best_idx >= 0:
                task.task = TASK_PICKUP
                task.target_pos = best_adj
                task.item_idx = best_idx
                task.type_id = best_type
                claimed_items.add(best_idx)
                types_needed.remove(best_type)

    def _count_staged(self, order, bot_positions, bot_inventories, wave):
        """Count how many items for this order are staged at dropoffs by this wave."""
        if not order:
            return 0, 0
        needs = list(order.needs())
        total = len(needs)
        if total == 0:
            return 0, 0

        need_count: dict[int, int] = {}
        for t in needs:
            need_count[t] = need_count.get(t, 0) + 1

        staged = 0
        bots = self._wave_bots(wave)
        for bid in bots:
            pos = bot_positions.get(bid)
            if pos not in self.drop_set:
                continue
            inv = bot_inventories.get(bid, [])
            for t in inv:
                if need_count.get(t, 0) > 0:
                    need_count[t] -= 1
                    staged += 1

        return staged, total

    def _rotate_waves(self, bot_positions):
        """After chain fires, rotate wave assignments."""
        # Preview bots become Active, Future bots become Preview
        # Old Active bots become Future
        old_active = self._wave_bots(WAVE_ACTIVE)
        old_preview = self._wave_bots(WAVE_PREVIEW)
        old_future = self._wave_bots(WAVE_FUTURE)

        for bid in old_preview:
            self.tasks[bid].wave = WAVE_ACTIVE
            self.tasks[bid].task = TASK_IDLE
        for bid in old_future:
            self.tasks[bid].wave = WAVE_PREVIEW
            self.tasks[bid].task = TASK_IDLE
        for bid in old_active:
            self.tasks[bid].wave = WAVE_FUTURE
            self.tasks[bid].task = TASK_IDLE

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)

        # Extract state
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
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

        # Get N+2 order
        future_order = None
        if all_orders and state.next_order_idx < len(all_orders):
            future_order = all_orders[state.next_order_idx]

        # Detect order change → rotate waves
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id and self._last_active_id >= 0:
            self._rotate_waves(bot_positions)
        self._last_active_id = active_id

        # Update task state based on PREVIOUS round's result
        # (inventory reflects what happened after last step())
        for bid in range(num_bots):
            task = self.tasks[bid]
            inv = bot_inventories[bid]

            if task.task == TASK_STAGED:
                task.rounds_waiting += 1
                if task.rounds_waiting > 50:
                    task.task = TASK_IDLE

        # Assign tasks for each wave
        claimed_items: set[int] = set()

        # Active wave
        self._assign_wave_tasks(WAVE_ACTIVE, active_order,
                                bot_positions, bot_inventories, claimed_items)

        # Preview wave
        self._assign_wave_tasks(WAVE_PREVIEW, preview_order,
                                bot_positions, bot_inventories, claimed_items)

        # Future wave
        self._assign_wave_tasks(WAVE_FUTURE, future_order,
                                bot_positions, bot_inventories, claimed_items)

        # Build goals for pathfinding
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}

        for bid in range(num_bots):
            task = self.tasks[bid]
            pos = bot_positions[bid]

            if task.target_pos:
                goals[bid] = task.target_pos
            else:
                goals[bid] = self.spawn
            goal_types[bid] = task.task if task.task != TASK_IDLE else 'park'

        # Chain timing: check if we should HOLD active delivery
        # Hold if preview staging is close to ready but not complete
        preview_staged, preview_total = self._count_staged(
            preview_order, bot_positions, bot_inventories, WAVE_PREVIEW)

        should_hold = False
        if active_order and preview_order and NUM_ROUNDS - rnd > 40:
            active_remaining = len(active_order.needs())
            if active_remaining <= 2 and preview_total > 0:
                if preview_staged > 0 and preview_staged < preview_total:
                    # Staging in progress — hold if won't take too long
                    should_hold = True

        # Urgency order
        def _urgency(bid):
            task = self.tasks[bid]
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            if task.task == TASK_DELIVER:
                return (0, dist)
            elif task.task == TASK_STAGE:
                return (1, dist)
            elif task.task == TASK_PICKUP:
                return (2, dist)
            elif task.task in (TASK_STAGED, TASK_HOLD):
                return (3, 0)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency)

        # PIBT pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            task = self.tasks[bid]
            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 5:
                for a in MOVES:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    if (nx, ny) in self.walkable:
                        actions[bid] = (a, -1)
                        break
                continue

            # At pickup target — issue pickup, then transition task
            if task.task == TASK_PICKUP and pos == task.target_pos:
                if task.item_idx >= 0 and len(inv) < INV_CAP:
                    actions[bid] = (ACT_PICKUP, task.item_idx)
                    # Transition for NEXT round: go deliver/stage
                    dz = self._nearest_drop(pos)
                    task.task = TASK_DELIVER if task.wave == WAVE_ACTIVE else TASK_STAGE
                    task.target_pos = dz
                    task.dropoff = dz
                    continue

            # At dropoff with items
            if pos in self.drop_set and inv:
                if task.task == TASK_DELIVER:
                    if should_hold and self._would_complete(
                            active_order, bid, bot_positions, bot_inventories):
                        actions[bid] = (ACT_WAIT, -1)
                        continue
                    actions[bid] = (ACT_DROPOFF, -1)
                    task.task = TASK_IDLE  # Transition after delivery
                    continue
                elif task.task == TASK_STAGE:
                    # Arrived at dropoff for staging — wait
                    task.task = TASK_STAGED
                    task.rounds_waiting = 0
                    actions[bid] = (ACT_WAIT, -1)
                    continue
                elif task.task == TASK_STAGED:
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # Opportunistic pickup (active items on adjacent cells)
            if task.wave == WAVE_ACTIVE and len(inv) < INV_CAP and active_order:
                opp = self._opportunistic_pickup(
                    bid, pos, active_order, inv, bot_inventories)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Follow pathfinder
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _would_complete(self, active, bid, bot_positions, bot_inventories):
        """Would this bot's delivery complete the active order?"""
        if not active:
            return False
        remaining = list(active.needs())
        # Remove what this bot would deliver
        for t in bot_inventories.get(bid, []):
            if t in remaining:
                remaining.remove(t)
        # Also count other bots at dropoff
        for bid2, inv2 in bot_inventories.items():
            if bid2 == bid:
                continue
            pos2 = bot_positions.get(bid2)
            if pos2 in self.drop_set:
                for t in inv2:
                    if t in remaining:
                        remaining.remove(t)
        return len(remaining) == 0

    def _opportunistic_pickup(self, bid, pos, order, inv, all_inv):
        """Pick up adjacent active items if helpful."""
        ms = self.ms
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if not order.needs_type(tid):
                continue
            if tid in inv:
                continue  # Already have this type
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False,
                active_bots: int = 8, preview_bots: int = 7,
                max_hold: int = 5) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareChainV3(ms, tables, future_orders=all_orders,
                                   active_bots=active_bots,
                                   preview_bots=preview_bots,
                                   max_hold=max_hold)
        action_log = []
        chains = 0
        max_chain = 0

        t0 = time.time()
        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            s_before = state.score
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                wave_counts = [0, 0, 0]
                for t in solver.tasks.values():
                    wave_counts[t.wave] += 1
                staged_p, total_p = solver._count_staged(
                    state.get_preview_order(),
                    {b: (int(state.bot_positions[b, 0]),
                         int(state.bot_positions[b, 1]))
                     for b in range(NUM_BOTS)},
                    {b: state.bot_inv_list(b) for b in range(NUM_BOTS)},
                    WAVE_PREVIEW)
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      f" W={wave_counts} Stg={staged_p}/{total_p}{extra}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: {state.score} pts, {state.orders_completed} orders, "
                  f"{elapsed:.1f}s, Chains={chains} MaxChain={max_chain}")

        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--active', type=int, default=8)
    parser.add_argument('--preview', type=int, default=7)
    parser.add_argument('--hold', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    score, actions = NightmareChainV3.run_sim(
        args.seed, verbose=True,
        active_bots=args.active, preview_bots=args.preview,
        max_hold=args.hold)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f"Saved: {saved}", file=sys.stderr)

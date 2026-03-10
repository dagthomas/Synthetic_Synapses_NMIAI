#!/usr/bin/env python3
"""Nightmare Committed Path Solver (CPS):

Key differences from V6:
1. Plans FULL paths (multi-round) using spacetime A*
2. Paths respect reservations from higher-priority bots
3. Replans only on order change, path completion, or stall
4. Over-assigns last items with flooding
5. Preview pre-fetching with committed paths

This should eliminate oscillation and reduce wasted movement.
"""
from __future__ import annotations
import sys, time, heapq, copy
import numpy as np
from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import build_walkable

sys.stdout.reconfigure(encoding='utf-8')

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
MAX_PATH_LEN = 60  # Max path search depth


class ReservationTable:
    """Spacetime reservation table for collision-free pathfinding."""

    def __init__(self, spawn):
        self.reservations: dict[tuple[int, int, int], int] = {}  # (x, y, round) → bot_id
        self.spawn = spawn

    def reserve(self, x, y, rnd, bot_id):
        if (x, y) == self.spawn:
            return  # Spawn is always free
        self.reservations[(x, y, rnd)] = bot_id

    def is_free(self, x, y, rnd, bot_id):
        if (x, y) == self.spawn:
            return True
        key = (x, y, rnd)
        if key in self.reservations:
            return self.reservations[key] == bot_id
        return True

    def clear_bot(self, bot_id):
        """Remove all reservations for a bot."""
        to_remove = [k for k, v in self.reservations.items() if v == bot_id]
        for k in to_remove:
            del self.reservations[k]

    def clear_from_round(self, rnd):
        """Remove all reservations from a round onward."""
        to_remove = [k for k in self.reservations if k[2] >= rnd]
        for k in to_remove:
            del self.reservations[k]


class SpacetimeAStar:
    """A* in spacetime (x, y, round) with reservation avoidance."""

    def __init__(self, ms: MapState, tables: PrecomputedTables):
        self.ms = ms
        self.tables = tables
        self.walkable = build_walkable(ms)
        self.spawn = ms.spawn

    def find_path(self, start, goal, start_round, bot_id, res_table,
                  max_depth=MAX_PATH_LEN):
        """Find path from start to goal avoiding reservations.

        Returns list of (action, target_pos) or None if no path.
        """
        if start == goal:
            return []

        h_dist = self.tables.get_distance(start, goal)
        # Priority queue: (f, g, x, y, round, parent_idx)
        open_set = [(h_dist, 0, start[0], start[1], start_round, -1)]
        visited = set()
        parents = []  # (x, y, round, action, parent_idx)

        while open_set:
            f, g, x, y, rnd, parent = heapq.heappop(open_set)

            if (x, y) == goal:
                # Reconstruct path
                path = []
                idx = len(parents) - 1
                if parent >= 0:
                    # We need to trace back through parents
                    # Actually, let me reconstruct differently
                    pass
                # Better reconstruction: store all states
                break

            state = (x, y, rnd)
            if state in visited:
                continue
            visited.add(state)

            if g >= max_depth:
                continue

            next_rnd = rnd + 1

            # Try all moves + wait
            for act in [ACT_WAIT] + MOVES:
                if act == ACT_WAIT:
                    nx, ny = x, y
                else:
                    nx = x + DX[act]
                    ny = y + DY[act]
                    if (nx, ny) not in self.walkable:
                        continue

                # Check reservation
                if not res_table.is_free(nx, ny, next_rnd, bot_id):
                    continue

                # Swap detection: check if the bot we're moving to was at our current pos
                # (prevents head-on collisions)
                if act != ACT_WAIT:
                    if not res_table.is_free(x, y, next_rnd, bot_id):
                        # Someone is planning to be at our current pos next round
                        # Check if they're currently at our target
                        occ = res_table.reservations.get((nx, ny, rnd))
                        if occ is not None and occ != bot_id:
                            # Potential swap — check if they're moving to our pos
                            if res_table.reservations.get((x, y, next_rnd)) == occ:
                                continue  # Swap detected, skip

                new_state = (nx, ny, next_rnd)
                if new_state in visited:
                    continue

                h = self.tables.get_distance((nx, ny), goal)
                new_g = g + 1
                new_f = new_g + h
                parent_idx = len(parents)
                parents.append((x, y, rnd, act, parent))
                heapq.heappush(open_set, (new_f, new_g, nx, ny, next_rnd, parent_idx))

        # Reconstruct path from parents
        if not parents:
            return None

        # Find the last parent that reaches goal
        goal_parents = [(i, p) for i, p in enumerate(parents)
                        if i > 0 or True]  # All parents

        # Actually, let me redo this with proper A* reconstruction
        return self._find_path_v2(start, goal, start_round, bot_id, res_table, max_depth)

    def _find_path_v2(self, start, goal, start_round, bot_id, res_table, max_depth):
        """Cleaner A* implementation."""
        if start == goal:
            return []

        h0 = self.tables.get_distance(start, goal)
        # (f, g, x, y, round, path_so_far)
        open_set = [(h0, 0, start[0], start[1], start_round, [])]
        visited = set()

        while open_set:
            f, g, x, y, rnd, path = heapq.heappop(open_set)

            state = (x, y, rnd)
            if state in visited:
                continue
            visited.add(state)

            if (x, y) == goal:
                return path

            if g >= max_depth:
                continue

            next_rnd = rnd + 1

            for act in MOVES + [ACT_WAIT]:
                if act == ACT_WAIT:
                    nx, ny = x, y
                else:
                    nx = x + DX[act]
                    ny = y + DY[act]
                    if (nx, ny) not in self.walkable:
                        continue

                if not res_table.is_free(nx, ny, next_rnd, bot_id):
                    continue

                new_state = (nx, ny, next_rnd)
                if new_state in visited:
                    continue

                h = self.tables.get_distance((nx, ny), goal)
                new_path = path + [(act, (nx, ny))]
                heapq.heappush(open_set, (g + 1 + h, g + 1, nx, ny, next_rnd, new_path))

        # No path found — return greedy BFS path (ignoring reservations)
        first_step = self.tables.get_first_step(start, goal)
        if first_step is not None:
            nx = start[0] + DX[first_step]
            ny = start[1] + DY[first_step]
            return [(first_step, (nx, ny))]
        return [(ACT_WAIT, start)]


class BotPlan:
    """A bot's committed plan: sequence of actions."""
    __slots__ = ['actions', 'step_idx', 'task_type', 'pickup_item', 'dropoff_target']

    def __init__(self, actions, task_type='pickup', pickup_item=None, dropoff_target=None):
        self.actions = actions  # [(action, target_pos), ...]
        self.step_idx = 0
        self.task_type = task_type
        self.pickup_item = pickup_item
        self.dropoff_target = dropoff_target

    @property
    def current_action(self):
        if self.step_idx < len(self.actions):
            return self.actions[self.step_idx]
        return None

    def advance(self):
        self.step_idx += 1

    @property
    def is_complete(self):
        return self.step_idx >= len(self.actions)


class NightmareCPS:
    """Committed Path Solver for nightmare mode."""

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

        self.astar = SpacetimeAStar(ms, tables)
        self.res_table = ReservationTable(self.spawn)

        # Type→items lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Pos→items
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            for adj in ms.item_adjacencies.get(idx, []):
                if adj not in self.pos_to_items:
                    self.pos_to_items[adj] = []
                self.pos_to_items[adj].append((idx, tid))

        # Bot state
        self.bot_plans: dict[int, BotPlan | None] = {b: None for b in range(self.num_bots)}
        self.stall_counts: dict[int, int] = {b: 0 for b in range(self.num_bots)}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._last_active_oid = -1
        self._claimed_items: set[int] = set()
        self._type_assigned: dict[int, int] = {}

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best, best_d

    def _plan_pickup_trip(self, bid, pos, item_idx, adj_cell, dropoff, rnd):
        """Plan: move to adj_cell → pickup → move to dropoff → dropoff action."""
        actions = []

        # Phase 1: Move to pickup position
        path_to_item = self.astar._find_path_v2(
            pos, adj_cell, rnd, bid, self.res_table, MAX_PATH_LEN)
        if path_to_item:
            for act, target in path_to_item:
                actions.append((act, target))

        # Phase 2: Pickup action
        arrival_round = rnd + len(actions)
        actions.append((ACT_PICKUP, adj_cell))

        # Phase 3: Move to dropoff
        path_to_drop = self.astar._find_path_v2(
            adj_cell, dropoff, arrival_round + 1, bid, self.res_table, MAX_PATH_LEN)
        if path_to_drop:
            for act, target in path_to_drop:
                actions.append((act, target))

        # Phase 4: Dropoff action
        actions.append((ACT_DROPOFF, dropoff))

        # Reserve the path
        current_pos = pos
        for i, (act, target) in enumerate(actions):
            step_rnd = rnd + i + 1
            if act in (ACT_PICKUP, ACT_DROPOFF):
                self.res_table.reserve(current_pos[0], current_pos[1], step_rnd, bid)
            else:
                if act != ACT_WAIT:
                    nx = current_pos[0] + DX[act]
                    ny = current_pos[1] + DY[act]
                    self.res_table.reserve(nx, ny, step_rnd, bid)
                    current_pos = (nx, ny)
                else:
                    self.res_table.reserve(current_pos[0], current_pos[1], step_rnd, bid)

        return BotPlan(actions, task_type='pickup', pickup_item=item_idx, dropoff_target=dropoff)

    def _plan_deliver_trip(self, bid, pos, dropoff, rnd):
        """Plan: move to dropoff → dropoff action."""
        path = self.astar._find_path_v2(
            pos, dropoff, rnd, bid, self.res_table, MAX_PATH_LEN)
        actions = []
        if path:
            actions = list(path)
        actions.append((ACT_DROPOFF, dropoff))
        return BotPlan(actions, task_type='deliver', dropoff_target=dropoff)

    def _plan_park(self, bid, pos, rnd):
        """Park: find a quiet spot and wait."""
        # Find corridor parking
        best = self.spawn
        best_d = 9999
        corridor_ys = [1, self.ms.height // 2, self.ms.height - 3]
        for cy in corridor_ys:
            for cx in range(self.ms.width):
                cell = (cx, cy)
                if cell in self.walkable and cell not in self.drop_set:
                    d = self.tables.get_distance(pos, cell)
                    if 0 < d < best_d:
                        best_d = d
                        best = cell

        path = self.astar._find_path_v2(
            pos, best, rnd, bid, self.res_table, 20)
        if path:
            return BotPlan(path, task_type='park')
        return BotPlan([(ACT_WAIT, pos)], task_type='park')

    def _assign_best_item(self, pos, needed, max_assign_bonus=1):
        """Find best item to pick up for needed types."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        total_needed = sum(needed.values())

        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            bonus = max_assign_bonus if total_needed <= 2 else 1
            max_assign = need_count + bonus
            if self._type_assigned.get(tid, 0) >= max_assign:
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in self._claimed_items:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    _, drop_d = self._nearest_drop(adj)
                    cost = d + drop_d * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        ms = self.ms
        num_bots = len(state.bot_positions)

        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] += 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        active_oid = active_order.id if active_order else -1

        # Detect order change → replan all
        if active_oid != self._last_active_oid:
            self._last_active_oid = active_oid
            # Invalidate all plans and reservations
            for bid in range(num_bots):
                self.bot_plans[bid] = None
            self.res_table.clear_from_round(rnd)
            self._claimed_items.clear()
            self._type_assigned.clear()

        # Compute needs
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        carrying_active = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Assign plans to bots that need them
        # Priority: bots carrying active items → deliver; empty bots → pickup; rest → park
        bots_needing_plan = []
        for bid in range(num_bots):
            plan = self.bot_plans[bid]
            if plan is not None and not plan.is_complete:
                # Check if stalled
                if self.stall_counts[bid] >= 8:
                    self.bot_plans[bid] = None
                    self.res_table.clear_bot(bid)
                    bots_needing_plan.append(bid)
                continue
            bots_needing_plan.append(bid)

        # Sort: carriers first (deliver), then empties (pickup), then full dead (park)
        def _sort_key(bid):
            inv = bot_inventories[bid]
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            pos = bot_positions[bid]
            if has_active:
                _, d = self._nearest_drop(pos)
                return (0, d)
            elif not inv:
                return (1, 0)
            elif has_preview:
                _, d = self._nearest_drop(pos)
                return (2, d)
            elif len(inv) < INV_CAP:
                return (3, 0)
            else:
                return (4, 0)

        bots_needing_plan.sort(key=_sort_key)

        for bid in bots_needing_plan:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)

            if has_active:
                # Deliver to nearest dropoff
                dropoff, _ = self._nearest_drop(pos)
                plan = self._plan_deliver_trip(bid, pos, dropoff, rnd)
                self.bot_plans[bid] = plan

            elif not inv or len(inv) < INV_CAP:
                # Try pickup
                if active_short:
                    item_idx, adj = self._assign_best_item(pos, active_short, max_assign_bonus=3)
                    if item_idx is not None:
                        dropoff, _ = self._nearest_drop(adj)
                        plan = self._plan_pickup_trip(bid, pos, item_idx, adj, dropoff, rnd)
                        self.bot_plans[bid] = plan
                        self._claimed_items.add(item_idx)
                        tid = int(ms.item_types[item_idx])
                        self._type_assigned[tid] = self._type_assigned.get(tid, 0) + 1
                        continue

                # Preview pickup
                remaining = sum(max(0, s - self._type_assigned.get(t, 0))
                                for t, s in active_short.items())
                if remaining == 0 and preview_needs:
                    item_idx, adj = self._assign_best_item(pos, preview_needs, max_assign_bonus=1)
                    if item_idx is not None:
                        dropoff, _ = self._nearest_drop(adj)
                        plan = self._plan_pickup_trip(bid, pos, item_idx, adj, dropoff, rnd)
                        self.bot_plans[bid] = plan
                        self._claimed_items.add(item_idx)
                        tid = int(ms.item_types[item_idx])
                        self._type_assigned[tid] = self._type_assigned.get(tid, 0) + 1
                        continue

                # Park
                plan = self._plan_park(bid, pos, rnd)
                self.bot_plans[bid] = plan

            elif has_preview and not active_short:
                dropoff, _ = self._nearest_drop(pos)
                plan = self._plan_deliver_trip(bid, pos, dropoff, rnd)
                self.bot_plans[bid] = plan

            else:
                # Dead inventory — park
                plan = self._plan_park(bid, pos, rnd)
                self.bot_plans[bid] = plan

        # Execute plans
        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            plan = self.bot_plans[bid]

            if plan is None or plan.is_complete:
                # No plan — opportunistic action
                opp = self._opportunistic_action(bid, pos, active_needs, preview_needs,
                                                  bot_inventories[bid], active_short)
                if opp:
                    actions[bid] = opp
                continue

            step_data = plan.current_action
            if step_data is None:
                continue

            act, target = step_data

            # Validate action
            if act == ACT_PICKUP:
                item_idx = plan.pickup_item
                if item_idx is not None and pos == target:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    plan.advance()
                else:
                    # Not at pickup position yet — move toward it
                    first_step = self.tables.get_first_step(pos, target)
                    if first_step is not None:
                        actions[bid] = (first_step, -1)
                    plan.advance()

            elif act == ACT_DROPOFF:
                if pos in self.drop_set and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    plan.advance()
                    self.bot_plans[bid] = None  # Trip complete
                elif pos != target:
                    first_step = self.tables.get_first_step(pos, target)
                    if first_step is not None:
                        actions[bid] = (first_step, -1)
                    plan.advance()
                else:
                    plan.advance()

            elif act == ACT_WAIT:
                actions[bid] = (ACT_WAIT, -1)
                plan.advance()

            else:
                # Movement action
                nx = pos[0] + DX[act]
                ny = pos[1] + DY[act]
                if (nx, ny) in self.walkable:
                    actions[bid] = (act, -1)
                else:
                    # Invalid move — use BFS fallback
                    first_step = self.tables.get_first_step(pos, target)
                    if first_step is not None:
                        actions[bid] = (first_step, -1)
                plan.advance()

            # Opportunistic adjacent pickup
            if actions[bid][0] not in (ACT_PICKUP, ACT_DROPOFF):
                if len(bot_inventories[bid]) < INV_CAP:
                    opp = self._opportunistic_action(bid, pos, active_needs, preview_needs,
                                                      bot_inventories[bid], active_short)
                    if opp and opp[0] == ACT_PICKUP:
                        actions[bid] = opp

        return actions

    def _opportunistic_action(self, bid, pos, active_needs, preview_needs,
                               bot_inv, active_short):
        """Pick up adjacent needed item if available."""
        adjacent = self.pos_to_items.get(pos, [])
        if not adjacent:
            return None
        bot_types = set(bot_inv)
        if len(bot_inv) >= INV_CAP:
            return None

        for item_idx, tid in adjacent:
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
                return (ACT_PICKUP, item_idx)
            elif not active_short and tid in preview_needs:
                if tid in bot_types:
                    continue
                return (ACT_PICKUP, item_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareCPS(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            step(state, actions, all_orders)

            if verbose and (rnd < 5 or rnd % 50 == 0):
                active = state.get_active_order()
                active_plans = sum(1 for p in solver.bot_plans.values() if p is not None)
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + f" Plans={active_plans}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Time={elapsed:.1f}s")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare CPS')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = NightmareCPS.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f"Seed {seed}: {score}")

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()

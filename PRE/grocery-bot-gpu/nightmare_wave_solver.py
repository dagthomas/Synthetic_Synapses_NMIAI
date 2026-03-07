"""Wave-based nightmare solver exploiting chain reactions.

Strategy: batch multiple orders, have all 20 bots fetch 3 items each in parallel,
deliver at dropoff zones, chain reaction auto-delivers items for future orders.

Key insight: bots carry items for FUTURE orders (beyond active+preview). When
the active order completes at dropoff, chain reaction auto-delivers future items
from ALL bots at dropoff zones, potentially completing multiple orders in 1 round.

Usage:
    python nightmare_wave_solver.py --seeds 7005 -v
    python nightmare_wave_solver.py --seeds 7001-7010
"""
from __future__ import annotations

import time
from itertools import permutations

import numpy as np

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


class NightmareWaveSolver:
    """Wave-based solver: batch fetch → deliver → chain reaction."""

    def __init__(self, ms: MapState, tables: PrecomputedTables):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']

        # Subsystems
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # type_id → [(item_idx, [adj_cells])]
        self.type_shelves: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if adj:
                self.type_shelves.setdefault(tid, []).append((idx, adj))

        # Wave state
        self.bot_fetch_queue: dict[int, list[int]] = {}  # bid → [type_ids to fetch]
        self.bot_current_target: dict[int, tuple[int, tuple[int, int]] | None] = {}
        self.last_completed = -1
        self.wave_start_round = 0

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        num_bots = self.num_bots

        # Extract positions and inventories
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Replan when orders completed changes
        if state.orders_completed != self.last_completed:
            self._plan_wave(state, all_orders, bot_positions, bot_inventories)
            self.last_completed = state.orders_completed
            self.wave_start_round = rnd

        # Wave timeout: replan if stuck for 60+ rounds
        if rnd - self.wave_start_round > 60:
            self._plan_wave(state, all_orders, bot_positions, bot_inventories)
            self.wave_start_round = rnd

        active = state.get_active_order()

        # Determine active shortfall (for opportunistic pickup priority)
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for inv in bot_inventories.values():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_active_short = sum(active_short.values())

        # Build goals
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        for bid in range(num_bots):
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            fetch_queue = self.bot_fetch_queue.get(bid, [])

            if fetch_queue and len(inv) < INV_CAP:
                # Still has items to fetch
                target = self.bot_current_target.get(bid)
                if target is None:
                    self._set_next_target(bid, pos)
                    target = self.bot_current_target.get(bid)

                if target:
                    item_idx, adj_cell = target
                    goals[bid] = adj_cell
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                else:
                    goals[bid] = self._nearest_drop(pos)
                    goal_types[bid] = 'deliver' if inv else 'park'
            elif inv:
                # Has items, heading to dropoff
                goals[bid] = self._nearest_drop(pos)
                goal_types[bid] = 'deliver'
            else:
                # Empty, no fetch → go to dropoff area (for next wave positioning)
                goals[bid] = self._nearest_drop(pos)
                goal_types[bid] = 'park'

        # Urgency: deliver > pickup > park
        urgency_order = sorted(range(num_bots), key=lambda bid: (
            0 if goal_types.get(bid) == 'deliver' else
            1 if goal_types.get(bid) == 'pickup' else 2,
            self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
        ))

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build final actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            gt = goal_types.get(bid, 'park')

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # At dropoff with items → deliver
            if pos in self.drop_set and inv and gt == 'deliver':
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            # At pickup target → pick up
            target = self.bot_current_target.get(bid)
            if target and pos == target[1] and gt == 'pickup':
                actions[bid] = (ACT_PICKUP, target[0])
                self._advance_fetch(bid, pos)
                continue

            # Opportunistic adjacent pickup (active items only, to avoid dead inventory)
            if len(inv) < INV_CAP and total_active_short > 0:
                opp = self._check_active_adjacent(bid, pos, active_short)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Use pathfinder
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _plan_wave(self, state: GameState, all_orders: list[Order],
                   bot_positions: dict[int, tuple[int, int]],
                   bot_inventories: dict[int, list[int]]):
        """Plan items for each bot based on upcoming orders."""
        self.bot_fetch_queue = {bid: [] for bid in range(self.num_bots)}
        self.bot_current_target = {}

        # Collect all items currently in bot inventories
        all_inv_items: list[int] = []
        for bid in range(self.num_bots):
            all_inv_items.extend(bot_inventories.get(bid, []))

        # Determine upcoming order needs
        upcoming_needs: list[list[int]] = []  # each element is a list of type_ids

        active = state.get_active_order()
        if active and not active.is_complete():
            upcoming_needs.append([int(t) for t in active.needs()])

        preview = state.get_preview_order()
        if preview and not preview.is_complete():
            upcoming_needs.append([int(t) for t in preview.needs()])

        for i in range(state.next_order_idx, len(all_orders)):
            upcoming_needs.append([int(t) for t in all_orders[i].required])

        if not upcoming_needs:
            return

        # Pack complete orders into wave, accounting for inventory
        available_inv = list(all_inv_items)
        total_cap = sum(INV_CAP - len(bot_inventories.get(bid, []))
                        for bid in range(self.num_bots))
        items_to_fetch: list[int] = []
        total_fetched = 0

        for needs in upcoming_needs:
            # Check which items are covered by existing inventory
            order_fetch: list[int] = []
            temp_inv = list(available_inv)

            for t in needs:
                if t in temp_inv:
                    temp_inv.remove(t)
                else:
                    order_fetch.append(t)

            # Can we fit ALL items for this order?
            if total_fetched + len(order_fetch) > total_cap:
                break

            items_to_fetch.extend(order_fetch)
            total_fetched += len(order_fetch)
            available_inv = temp_inv

        # Assign items to bots
        self._assign_items(items_to_fetch, bot_positions, bot_inventories)

    def _assign_items(self, items_to_fetch: list[int],
                      bot_positions: dict[int, tuple[int, int]],
                      bot_inventories: dict[int, list[int]]):
        """Assign item types to bots, minimizing total fetch time."""
        if not items_to_fetch:
            return

        bot_caps = {bid: INV_CAP - len(bot_inventories.get(bid, []))
                    for bid in range(self.num_bots)}

        # Greedy assignment: for each item, find cheapest bot
        remaining = list(items_to_fetch)

        while remaining:
            best_score = 99999.0
            best_bid = -1
            best_item_idx = -1

            for i, tid in enumerate(remaining):
                for bid in range(self.num_bots):
                    if len(self.bot_fetch_queue[bid]) >= bot_caps[bid]:
                        continue

                    # Estimate bot position after previous fetches
                    pos = bot_positions[bid]
                    for prev_t in self.bot_fetch_queue[bid]:
                        s = self._nearest_shelf_adj(pos, prev_t)
                        if s:
                            pos = s[1]

                    s = self._nearest_shelf_adj(pos, tid)
                    if s is None:
                        continue

                    d = self.tables.get_distance(pos, s[1])
                    drop_d = min(self.tables.get_distance(s[1], dz)
                                 for dz in self.drop_zones)
                    score = d + drop_d * 0.3

                    if score < best_score:
                        best_score = score
                        best_bid = bid
                        best_item_idx = i

            if best_bid < 0:
                break

            tid = remaining.pop(best_item_idx)
            self.bot_fetch_queue[best_bid].append(tid)

        # Optimize fetch order per bot (TSP for 3 items)
        for bid in range(self.num_bots):
            queue = self.bot_fetch_queue[bid]
            if len(queue) > 1:
                self.bot_fetch_queue[bid] = self._optimize_fetch_order(
                    bid, bot_positions[bid], queue)

        # Set initial targets
        for bid in range(self.num_bots):
            if self.bot_fetch_queue[bid]:
                self._set_next_target(bid, bot_positions[bid])

    def _optimize_fetch_order(self, bid: int, start_pos: tuple[int, int],
                               queue: list[int]) -> list[int]:
        """Optimize fetch order for minimum total travel (TSP with ≤3 items)."""
        if len(queue) <= 1:
            return queue

        best_order = list(queue)
        best_cost = 99999

        for perm in permutations(queue):
            cost = 0
            pos = start_pos
            for tid in perm:
                s = self._nearest_shelf_adj(pos, tid)
                if s is None:
                    cost = 99999
                    break
                cost += self.tables.get_distance(pos, s[1])
                pos = s[1]
            # Add cost to nearest dropoff
            cost += min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

            if cost < best_cost:
                best_cost = cost
                best_order = list(perm)

        return best_order

    def _set_next_target(self, bid: int, pos: tuple[int, int]):
        """Set next fetch target for bot."""
        queue = self.bot_fetch_queue.get(bid, [])
        if not queue:
            self.bot_current_target[bid] = None
            return

        tid = queue[0]
        shelf = self._nearest_shelf_adj(pos, tid)
        self.bot_current_target[bid] = shelf

    def _advance_fetch(self, bid: int, pos: tuple[int, int]):
        """After pickup, advance to next item in fetch queue."""
        queue = self.bot_fetch_queue.get(bid, [])
        if queue:
            queue.pop(0)

        if queue:
            self._set_next_target(bid, pos)
        else:
            self.bot_current_target[bid] = None

    def _nearest_shelf_adj(self, pos: tuple[int, int],
                           type_id: int) -> tuple[int, tuple[int, int]] | None:
        """Find nearest (item_idx, adj_cell) for type."""
        best = None
        best_d = 99999
        for item_idx, adj_cells in self.type_shelves.get(type_id, []):
            for adj in adj_cells:
                d = self.tables.get_distance(pos, adj)
                if d < best_d:
                    best_d = d
                    best = (item_idx, adj)
        return best

    def _nearest_drop(self, pos: tuple[int, int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _check_active_adjacent(self, bid: int, pos: tuple[int, int],
                                active_short: dict[int, int]) -> tuple[int, int] | None:
        """Pick up adjacent item if it's an active shortfall type."""
        ms = self.ms
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short or active_short[tid] <= 0:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int) -> int:
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        """Run full simulation. Returns (score, action_log)."""
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareWaveSolver(ms, tables)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
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

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN×{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


DB_URL = "postgres://grocery:grocery123@localhost:5433/grocery_bot"


def record_to_pg(seed, score, action_log, elapsed):
    """Record run to PostgreSQL."""
    import json
    import os
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        return None

    from game_engine import build_map, CELL_WALL, CELL_SHELF, state_to_ws_format, actions_to_ws_format

    db_url = os.environ.get("GROCERY_DB_URL", DB_URL)
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        ms = build_map('nightmare')
        cfg = CONFIGS['nightmare']

        walls, shelves_list = [], []
        for y in range(ms.height):
            for x in range(ms.width):
                c = int(ms.grid[y, x])
                if c == CELL_WALL:
                    walls.append([x, y])
                elif c == CELL_SHELF:
                    shelves_list.append([x, y])

        items = [{"id": it["id"], "type": it["type"], "position": list(it["position"])}
                 for it in ms.items]

        state2, all_orders2 = init_game(seed, 'nightmare', num_orders=100)
        for rnd, acts in enumerate(action_log):
            state2.round = rnd
            step(state2, acts, all_orders2)

        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, spawn,
                              final_score, items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            seed, 'nightmare', ms.width, ms.height, cfg['bots'],
            ms.num_types, cfg['order_size'][0], cfg['order_size'][1],
            json.dumps(walls), json.dumps(shelves_list),
            json.dumps(items), json.dumps(list(ms.drop_off)),
            json.dumps(list(ms.spawn)),
            state2.score, state2.items_delivered, state2.orders_completed,
            'wave_solver',
        ))
        run_id = cur.fetchone()[0]

        gs, ao = init_game(seed, 'nightmare', num_orders=100)
        round_tuples = []
        for rnd in range(min(len(action_log), 500)):
            gs.round = rnd
            ws_data = state_to_ws_format(gs, ao)
            ws_acts = actions_to_ws_format(action_log[rnd], gs.map_state)
            bots = [{"id": b["id"], "position": b["position"],
                     "inventory": b.get("inventory", [])} for b in ws_data["bots"]]
            orders = [{"id": o["id"], "items_required": o["items_required"],
                       "items_delivered": o.get("items_delivered", []),
                       "status": o.get("status", "active")}
                      for o in ws_data.get("orders", [])]
            round_tuples.append((
                run_id, rnd, json.dumps(bots), json.dumps(orders),
                json.dumps(ws_acts), ws_data["score"], json.dumps([])
            ))
            step(gs, action_log[rnd], ao)

        execute_values(cur, """
            INSERT INTO rounds (run_id, round_number, bots, orders, actions, score, events)
            VALUES %s
        """, round_tuples, page_size=100)

        conn.commit()
        conn.close()
        print(f"  Recorded to DB: run_id={run_id}", file=__import__('sys').stderr)
        return run_id
    except Exception as e:
        print(f"  DB error: {e}", file=__import__('sys').stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare wave solver (chain reaction)')
    parser.add_argument('--seeds', default='7001-7010')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        score, action_log = NightmareWaveSolver.run_sim(seed, verbose=args.verbose)
        elapsed = time.time() - st
        scores.append(score)
        print(f"Seed {seed}: {score}")

        if not args.no_record:
            record_to_pg(seed, score, action_log, elapsed)

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()

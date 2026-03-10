"""Chain-reaction pipeline solver for nightmare mode.

Strategy: pipeline active delivery with preview staging.
- TRIGGER bots: fetch remaining active items, deliver to trigger chain
- STAGE bots: carry preview items, wait at dropoff zones (3 max)
- PREFETCH bots: fetch items for order N+2, N+3 (next batch's preview)
- Pipeline: while current batch delivers, next batch items are being fetched

Target: 8-12 round cycles with chain depth 1 = 800-1300 score

Usage:
    python nightmare_chain_v4.py --seeds 7005 -v
    python nightmare_chain_v4.py --seeds 1000-1009
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState, build_map_from_capture,
    generate_all_orders,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap


class ChainPipelineSolver:
    """Chain-reaction pipeline solver for 20-bot nightmare mode."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.spawn = ms.spawn
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.num_bots = CONFIGS['nightmare']['bots']

        # Pathfinding
        traffic = TrafficRules(ms)
        congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, traffic, congestion)
        self.congestion = congestion

        # Orders
        self.future_orders = future_orders or []

        # Anti-stall
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Chain tracking
        self.chain_events: list[tuple[int, int]] = []
        self._last_goal_types: dict[int, str] = {}

    def _order_needs(self, order: Order) -> dict[int, int]:
        """Type_id -> count still needed for this order."""
        return dict(Counter(order.needs()))

    def _bot_has_types(self, inv: list[int], needs: dict[int, int]) -> bool:
        """Check if bot carries any type matching needs."""
        for tid in inv:
            if tid in needs and needs[tid] > 0:
                return True
        return False

    def _find_item(self, tid: int, pos: tuple[int, int],
                   claimed: set[int]) -> tuple[int, tuple[int, int], int] | None:
        """Find nearest unclaimed item of type tid. Returns (item_idx, adj_pos, distance)."""
        best = None
        best_d = 999
        for idx in range(self.ms.num_items):
            if idx in claimed:
                continue
            if int(self.ms.item_types[idx]) != tid:
                continue
            for adj in self.ms.item_adjacencies.get(idx, []):
                d = self.tables.get_distance(pos, adj)
                if d < best_d:
                    best_d = d
                    best = (idx, adj, d)
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

    def _balanced_drop(self, pos: tuple[int, int],
                       loads: dict[tuple[int, int], int]) -> tuple[int, int]:
        """Pick dropoff balancing distance and current load."""
        best = self.drop_zones[0]
        best_s = 999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            load = loads.get(dz, 0)
            s = d + load * 10
            if s < best_s:
                best_s = s
                best = dz
        return best

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int) -> int:
        h = hash((bid, rnd, pos)) % 4
        return [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT][h]

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS['nightmare']

        # Extract positions and inventories
        bot_pos: dict[int, tuple[int, int]] = {}
        bot_inv: dict[int, list[int]] = {}
        for b in range(num_bots):
            bot_pos[b] = (int(state.bot_positions[b, 0]),
                          int(state.bot_positions[b, 1]))
            bot_inv[b] = state.bot_inv_list(b)

        # Anti-stall
        self.congestion.update(list(bot_pos.values()))
        for b in range(num_bots):
            p = bot_pos[b]
            if p == self.prev_positions.get(b):
                self.stall_counts[b] = self.stall_counts.get(b, 0) + 1
            else:
                self.stall_counts[b] = 0
            self.prev_positions[b] = p

        active = state.get_active_order()
        preview = state.get_preview_order()

        if not active:
            return [(ACT_WAIT, -1)] * num_bots

        # What does active order still need?
        active_needs = self._order_needs(active)
        preview_needs = self._order_needs(preview) if preview else {}

        # === ROLE ASSIGNMENT ===
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}
        claimed_items: set[int] = set()

        # Track dropoff load for balancing
        drop_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        # Track what we've committed to for active order
        active_committed = Counter()  # tid -> count of bots carrying/fetching

        # --- Phase 1: DELIVER active items ---
        # Step 1a: Bots carrying active-matching items -> route to dropoff
        for b in range(num_bots):
            inv = bot_inv[b]
            if not inv:
                continue
            useful = False
            for tid in inv:
                remaining = active_needs.get(tid, 0) - active_committed.get(tid, 0)
                if remaining > 0:
                    useful = True
                    active_committed[tid] += 1
            if useful:
                dz = self._balanced_drop(bot_pos[b], drop_loads)
                goals[b] = dz
                goal_types[b] = 'deliver'
                drop_loads[dz] = drop_loads.get(dz, 0) + 1

        # Step 1b: Fetch remaining active items from shelves
        for tid, cnt in active_needs.items():
            remaining = cnt - active_committed.get(tid, 0)
            for _ in range(remaining):
                best_b = None
                best_result = None
                best_d = 999
                for b in range(num_bots):
                    if b in goals:
                        continue
                    if len(bot_inv[b]) >= INV_CAP:
                        continue
                    result = self._find_item(tid, bot_pos[b], claimed_items)
                    if result and result[2] < best_d:
                        best_d = result[2]
                        best_b = b
                        best_result = result
                if best_b is not None and best_result is not None:
                    idx, adj, _ = best_result
                    goals[best_b] = adj
                    goal_types[best_b] = 'pickup'
                    pickup_targets[best_b] = idx
                    claimed_items.add(idx)

        # --- Phase 2: STAGE preview items at dropoffs ---
        preview_committed = Counter()
        # Bots carrying preview-matching items -> route to dropoff
        for b in range(num_bots):
            if b in goals:
                continue
            inv = bot_inv[b]
            if not inv:
                continue
            has_preview = False
            for tid in inv:
                remaining = preview_needs.get(tid, 0) - preview_committed.get(tid, 0)
                if remaining > 0:
                    has_preview = True
                    preview_committed[tid] += 1
            if has_preview:
                dz = self._balanced_drop(bot_pos[b], drop_loads)
                goals[b] = dz
                goal_types[b] = 'stage'
                drop_loads[dz] = drop_loads.get(dz, 0) + 1

        # --- Phase 3: FETCH preview items from shelves ---
        max_preview_fetchers = 6
        preview_fetchers = 0
        for tid, cnt in preview_needs.items():
            remaining = cnt - preview_committed.get(tid, 0)
            for _ in range(remaining):
                if preview_fetchers >= max_preview_fetchers:
                    break
                best_b = None
                best_result = None
                best_d = 999
                for b in range(num_bots):
                    if b in goals:
                        continue
                    if len(bot_inv[b]) >= INV_CAP:
                        continue
                    result = self._find_item(tid, bot_pos[b], claimed_items)
                    if result and result[2] < best_d:
                        best_d = result[2]
                        best_b = b
                        best_result = result
                if best_b is not None and best_result is not None:
                    idx, adj, _ = best_result
                    goals[best_b] = adj
                    goal_types[best_b] = 'preview'
                    pickup_targets[best_b] = idx
                    claimed_items.add(idx)
                    preview_fetchers += 1

        # --- Phase 4: IDLE bots ---
        # Skip prefetching future orders (dead inventory risk too high)
        for b in range(num_bots):
            if b in goals:
                continue
            inv = bot_inv[b]
            if inv:
                # Has items -> go toward nearest dropoff (will auto-deliver if matches)
                dz = self._nearest_drop(bot_pos[b])
                goals[b] = dz
                goal_types[b] = 'deliver'
            else:
                goals[b] = bot_pos[b]
                goal_types[b] = 'park'

        self._last_goal_types = dict(goal_types)

        # === PATHFINDING ===
        priority_map = {'deliver': 0, 'pickup': 1, 'stage': 2, 'preview': 3,
                        'flee': 4, 'park': 5}
        rotation = rnd % 100
        urgency_order = sorted(range(num_bots), key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos.get(bid, self.spawn),
                                     goals.get(bid, self.spawn)),
            (bid + rotation) % 100
        ))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency_order,
            goal_types=goal_types, round_number=rnd)

        # === BUILD ACTIONS ===
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_pos[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # At dropoff with items -> ALWAYS deliver (highest priority)
            if pos in self.drop_set and bot_inv[bid]:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                actions[bid] = (self._escape_action(bid, pos, rnd), -1)
                continue

            # At pickup target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                    continue

            # Opportunistic adjacent pickup (any bot type)
            if len(bot_inv[bid]) < INV_CAP:
                opp = self._opp_pickup(bid, pos, active, preview)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Follow pathfinder
            actions[bid] = (path_actions.get(bid, ACT_WAIT), -1)

        return actions

    def _opp_pickup(self, bid: int, pos: tuple[int, int],
                    active: Order, preview: Order | None) -> tuple[int, int] | None:
        """Opportunistic: pick adjacent active/preview items."""
        active_needs = self._order_needs(active)
        preview_needs = self._order_needs(preview) if preview else {}

        best_idx = None
        best_val = -1
        for idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[idx])
            if tid in active_needs:
                val = 10.0
            elif tid in preview_needs:
                val = 5.0
            else:
                continue
            if val <= best_val:
                continue
            for adj in self.ms.item_adjacencies.get(idx, []):
                if adj == pos:
                    best_idx = idx
                    best_val = val
                    break
        if best_idx is not None:
            return (ACT_PICKUP, best_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False,
                live_map: MapState | None = None) -> tuple[int, list]:
        """Run full simulation."""
        if live_map is not None:
            all_orders = generate_all_orders(seed, live_map, 'nightmare', count=100)
            num_bots = CONFIGS['nightmare']['bots']
            state = GameState(live_map)
            state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
            state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
            for i in range(num_bots):
                state.bot_positions[i] = [live_map.spawn[0], live_map.spawn[1]]
            state.orders = [all_orders[0].copy(), all_orders[1].copy()]
            state.orders[0].status = 'active'
            state.orders[1].status = 'preview'
            state.next_order_idx = 2
            state.active_idx = 0
            ms = live_map
        else:
            state, all_orders = init_game(seed, 'nightmare', num_orders=100)
            ms = state.map_state

        tables = PrecomputedTables.get(ms)
        solver = ChainPipelineSolver(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
        action_log = []

        # Utilization tracking
        goal_totals: dict[str, int] = {}
        order_rounds = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)

            # Track utilization
            for gt in solver._last_goal_types.values():
                goal_totals[gt] = goal_totals.get(gt, 0) + 1

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
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
            if solver.chain_events:
                print(f"Chain events: {solver.chain_events}")
            # Utilization
            avg = {gt: cnt / num_rounds for gt, cnt in sorted(goal_totals.items())}
            working = sum(v for k, v in avg.items() if k not in ('park', 'flee'))
            idle = sum(v for k, v in avg.items() if k in ('park', 'flee'))
            print(f"Avg/rnd: {' '.join(f'{k}={v:.1f}' for k, v in avg.items())}")
            print(f"Working={working:.1f} Idle={idle:.1f} ({idle/(working+idle)*100:.0f}% idle)")
            if len(order_rounds) > 1:
                gaps = [order_rounds[i+1] - order_rounds[i]
                        for i in range(len(order_rounds) - 1)]
                print(f"Order gaps: avg={np.mean(gaps):.1f} min={min(gaps)} max={max(gaps)}")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare chain pipeline solver')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-live-map', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    live_map = None
    if not args.no_live_map:
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap and cap.get('grid'):
                live_map = build_map_from_capture(cap)
                print(f"Using live map: {live_map.width}x{live_map.height}, "
                      f"{live_map.num_items} items", file=sys.stderr)
        except Exception as e:
            print(f"Could not load live map: {e}", file=sys.stderr)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = ChainPipelineSolver.run_sim(
            seed, verbose=args.verbose, live_map=live_map)
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

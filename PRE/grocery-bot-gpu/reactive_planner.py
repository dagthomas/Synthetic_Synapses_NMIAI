"""Reactive per-round planner for multi-bot grocery game.

Instead of planning everything offline, this solver simulates the game
round by round, making optimal decisions at each step using precomputed
shortest paths and priority-based collision avoidance.

This avoids the offline/online mismatch that plagues space-time A* planners.

Usage:
    python reactive_planner.py --seeds 42 --difficulty hard -v
    python reactive_planner.py --seeds 7005 --difficulty nightmare -v
    python reactive_planner.py --seeds 1000-1009 --difficulty nightmare
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from game_engine import (
    init_game, step as cpu_step, simulate_game,
    GameState, MapState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import CONFIGS, DIFF_ROUNDS, parse_seeds
from precompute import PrecomputedTables


class ReactivePlanner:
    """Round-by-round reactive planner with task allocation."""

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 num_bots: int, num_rounds: int = 300,
                 verbose: bool = False):
        self.ms = map_state
        self.all_orders = all_orders
        self.num_bots = num_bots
        self.num_rounds = num_rounds
        self.verbose = verbose

        self.tables = PrecomputedTables.get(map_state)
        self.dz_list = [tuple(int(c) for c in dz) for dz in map_state.drop_off_zones]
        self.dz_set = set(self.dz_list)
        self.spawn = (int(map_state.spawn[0]), int(map_state.spawn[1]))

        # Item lookup: type_id -> [(item_idx, shelf_pos, [adj_positions])]
        self.type_to_items: dict[int, list[tuple[int, tuple, list]]] = defaultdict(list)
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            pos = (int(map_state.item_positions[idx, 0]),
                   int(map_state.item_positions[idx, 1]))
            adj = [(int(a[0]), int(a[1]))
                   for a in map_state.item_adjacencies.get(idx, [])]
            self.type_to_items[tid].append((idx, pos, adj))

        # Neighbor lookup
        self._walkable: set[tuple[int, int]] = set()
        for y in range(map_state.height):
            for x in range(map_state.width):
                if map_state.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    self._walkable.add((x, y))

        self._neighbors: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for pos in self._walkable:
            nbrs = []
            for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in self._walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

        # Bot mission tracking
        self._missions: dict[int, dict] = {}
        # 'pickup': go to item adj, pick up
        # 'deliver': go to DZ, drop off
        # 'park': move away from DZ

    def _dist(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        ai = self.tables.pos_to_idx.get(a)
        bi = self.tables.pos_to_idx.get(b)
        if ai is None or bi is None:
            return 9999
        return int(self.tables.dist_matrix[ai, bi])

    def _first_step(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Get first action (1-4) from a toward b using precomputed next_step."""
        ai = self.tables.pos_to_idx.get(a)
        bi = self.tables.pos_to_idx.get(b)
        if ai is None or bi is None:
            return ACT_WAIT
        return int(self.tables.next_step_matrix[ai, bi])

    def _nearest_dz(self, pos: tuple[int, int]) -> tuple[int, int]:
        best = self.dz_list[0]
        best_d = self._dist(pos, best)
        for dz in self.dz_list[1:]:
            d = self._dist(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _find_best_item(self, type_id: int, from_pos: tuple[int, int],
                        avoid_positions: set[tuple[int, int]] | None = None
                        ) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
        """Find nearest item of type, avoiding certain adjacent positions."""
        candidates = self.type_to_items.get(type_id, [])
        if not candidates:
            return None
        best = None
        best_d = 9999
        for item_idx, shelf_pos, adj_list in candidates:
            for adj in adj_list:
                if avoid_positions and adj in avoid_positions:
                    continue
                d = self._dist(from_pos, adj)
                if d < best_d:
                    best_d = d
                    best = (item_idx, shelf_pos, adj)
        return best

    def _assign_items_to_bots(self, state: GameState,
                              items_needed: list[int],
                              bots_available: list[int]
                              ) -> dict[int, tuple[int, int, tuple, tuple]]:
        """Assign needed item types to available bots using min-cost matching.

        Returns {bot_id: (type_id, item_idx, shelf_pos, adj_pos)}.
        """
        if not items_needed or not bots_available:
            return {}

        n_items = len(items_needed)
        n_bots = len(bots_available)

        # Cost matrix: [bots, items]
        cost = np.full((n_bots, n_items), 1e6, dtype=np.float64)
        item_info_cache = {}

        for bi, bid in enumerate(bots_available):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            for ii, tid in enumerate(items_needed):
                info = self._find_best_item(tid, pos)
                if info:
                    item_idx, shelf_pos, adj_pos = info
                    pickup_dist = self._dist(pos, adj_pos)
                    dz_dist = self._dist(adj_pos, self._nearest_dz(adj_pos))
                    cost[bi, ii] = pickup_dist + dz_dist + 1
                    item_info_cache[(bi, ii)] = (tid, item_idx, shelf_pos, adj_pos)

        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except ValueError:
            return {}

        assignments = {}
        for bi, ii in zip(row_ind, col_ind):
            if cost[bi, ii] < 1e5 and (bi, ii) in item_info_cache:
                bid = bots_available[bi]
                assignments[bid] = item_info_cache[(bi, ii)]

        return assignments

    def decide_round(self, state: GameState, all_orders: list[Order],
                     round_num: int) -> list[tuple[int, int]]:
        """Decide actions for all bots for this round."""
        active = state.get_active_order()
        preview = state.get_preview_order()
        active_needs = active.needs() if active else []

        actions = [(ACT_WAIT, -1)] * self.num_bots
        claimed_positions: set[tuple[int, int]] = set()

        # Track what's already being carried or pursued for active order
        # so we don't over-assign
        carried_for_active: dict[int, int] = defaultdict(int)  # tid -> count
        pursuing_active: dict[int, int] = defaultdict(int)  # tid -> count

        # Count items already in bots' inventories that match active needs
        active_need_counts: dict[int, int] = defaultdict(int)
        for tid in active_needs:
            active_need_counts[tid] += 1

        for bid in range(self.num_bots):
            inv = [int(state.bot_inventories[bid, i]) for i in range(INV_CAP)
                   if state.bot_inventories[bid, i] >= 0]
            for tid in inv:
                if tid in active_need_counts:
                    carried_for_active[tid] += 1

        # Phase 1: Validate existing missions
        for bid in range(self.num_bots):
            mission = self._missions.get(bid)
            if not mission:
                continue
            inv = [int(state.bot_inventories[bid, i]) for i in range(INV_CAP)
                   if state.bot_inventories[bid, i] >= 0]

            if mission['type'] == 'pickup':
                tid = mission['tid']
                # Already have this type? Or order no longer needs it?
                already_have = sum(1 for t in inv if t == tid)
                still_needed = active_need_counts.get(tid, 0) - carried_for_active.get(tid, 0)
                if already_have > 0 or still_needed <= 0:
                    # Mission complete or no longer needed
                    if inv and any(t in active_needs for t in inv):
                        pos = (int(state.bot_positions[bid, 0]),
                               int(state.bot_positions[bid, 1]))
                        self._missions[bid] = {'type': 'deliver',
                                               'target': self._nearest_dz(pos)}
                    else:
                        self._missions.pop(bid, None)
                else:
                    pursuing_active[tid] += 1

            elif mission['type'] == 'deliver':
                if not inv or not any(t in active_needs for t in inv):
                    self._missions.pop(bid, None)

        # Phase 2: Immediate actions
        for bid in range(self.num_bots):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = [int(state.bot_inventories[bid, i]) for i in range(INV_CAP)
                   if state.bot_inventories[bid, i] >= 0]

            # At DZ with matching items → drop off
            if pos in self.dz_set and inv and any(t in active_needs for t in inv):
                actions[bid] = (ACT_DROPOFF, -1)
                claimed_positions.add(pos)
                self._missions[bid] = {'type': 'deliver', 'target': pos}
                continue

            # At DZ with NO matching items → move away to free DZ
            if pos in self.dz_set and not any(t in active_needs for t in inv):
                for act, (nx, ny) in self._neighbors.get(pos, []):
                    if (nx, ny) not in claimed_positions and (nx, ny) not in self.dz_set:
                        actions[bid] = (act, -1)
                        claimed_positions.add((nx, ny))
                        self._missions.pop(bid, None)
                        break
                if actions[bid][0] != ACT_WAIT:
                    continue

            # At pickup target? Execute pickup
            mission = self._missions.get(bid)
            if (mission and mission['type'] == 'pickup'
                    and pos == mission.get('target')
                    and len(inv) < INV_CAP):
                item_idx = mission.get('item_idx', -1)
                tid = mission.get('tid', -1)
                if item_idx >= 0:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    claimed_positions.add(pos)
                    # After pickup: check if we should pick more or deliver
                    inv_after = inv + [tid]
                    if len(inv_after) >= INV_CAP:
                        # Inventory full → deliver
                        self._missions[bid] = {'type': 'deliver',
                                               'target': self._nearest_dz(pos)}
                    elif tid in active_needs:
                        # Check if more active items still unassigned
                        # (will be reassigned in Phase 3 next round)
                        self._missions.pop(bid, None)
                    else:
                        self._missions.pop(bid, None)
                    continue

        # Phase 3: Assign items to bots without missions
        idle_bots = []
        for bid in range(self.num_bots):
            if actions[bid][0] != ACT_WAIT:
                continue
            inv = [int(state.bot_inventories[bid, i]) for i in range(INV_CAP)
                   if state.bot_inventories[bid, i] >= 0]
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))

            # Has matching items for active → deliver (only if inv full or no more items to pick)
            if inv and any(t in active_needs for t in inv):
                if len(inv) >= INV_CAP:
                    # Full inventory → must deliver
                    if not self._missions.get(bid) or self._missions[bid]['type'] != 'deliver':
                        self._missions[bid] = {'type': 'deliver',
                                               'target': self._nearest_dz(pos)}
                    continue
                # Inventory not full: can we pick more active items?
                # Check remaining_needs (computed below)
                # For now, if no pickup mission, assign deliver
                if not self._missions.get(bid):
                    self._missions[bid] = {'type': 'deliver',
                                           'target': self._nearest_dz(pos)}
                continue

            # Has valid pickup mission → keep it
            mission = self._missions.get(bid)
            if mission and mission['type'] == 'pickup':
                continue

            idle_bots.append(bid)

        # What types still need a bot assigned?
        remaining_needs = []
        for tid, count in active_need_counts.items():
            already_covered = carried_for_active.get(tid, 0) + pursuing_active.get(tid, 0)
            for _ in range(max(0, count - already_covered)):
                remaining_needs.append(tid)

        if idle_bots and remaining_needs:
            assignments = self._assign_items_to_bots(
                state, remaining_needs, idle_bots)
            for bid, (tid, item_idx, shelf_pos, adj_pos) in assignments.items():
                self._missions[bid] = {
                    'type': 'pickup',
                    'tid': tid,
                    'target': adj_pos,
                    'item_idx': item_idx,
                }
                idle_bots.remove(bid)

        # Assign preview items to remaining idle bots (pickup only, no deliver)
        if idle_bots and preview:
            preview_needs_list = preview.needs()
            preview_remaining = []
            for tid in preview_needs_list:
                if pursuing_active.get(tid, 0) == 0:
                    preview_remaining.append(tid)
            if preview_remaining:
                assign2 = self._assign_items_to_bots(
                    state, preview_remaining[:len(idle_bots)], idle_bots)
                for bid, (tid, item_idx, shelf_pos, adj_pos) in assign2.items():
                    self._missions[bid] = {
                        'type': 'pickup',
                        'tid': tid,
                        'target': adj_pos,
                        'item_idx': item_idx,
                        'preview': True,  # flag: don't deliver until active
                    }

        # Phase 4: Generate movement actions
        for bid in range(self.num_bots):
            if actions[bid][0] != ACT_WAIT:
                # Already has an action from Phase 2.
                # For movement: bot is LEAVING current pos → claim destination
                # For pickup/dropoff: bot stays → claim current pos
                act = actions[bid][0]
                pos = (int(state.bot_positions[bid, 0]),
                       int(state.bot_positions[bid, 1]))
                if ACT_MOVE_UP <= act <= ACT_MOVE_RIGHT:
                    dest = (pos[0] + DX[act], pos[1] + DY[act])
                    claimed_positions.add(dest)
                else:
                    claimed_positions.add(pos)
                continue

            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            mission = self._missions.get(bid)

            if not mission:
                claimed_positions.add(pos)
                continue

            target = mission.get('target')
            if not target or pos == target:
                claimed_positions.add(pos)
                continue

            # Use precomputed next_step for optimal direction
            step_act = self._first_step(pos, target)
            if step_act != ACT_WAIT:
                nx, ny = pos[0] + DX[step_act], pos[1] + DY[step_act]
                if (nx, ny) not in claimed_positions or (nx, ny) == self.spawn:
                    actions[bid] = (step_act, -1)
                    claimed_positions.add((nx, ny))
                    continue

            # Optimal direction blocked; try alternatives
            best_act = ACT_WAIT
            best_dist = self._dist(pos, target)
            for act, (nx, ny) in self._neighbors.get(pos, []):
                if (nx, ny) in claimed_positions and (nx, ny) != self.spawn:
                    continue
                d = self._dist((nx, ny), target)
                if d < best_dist:
                    best_dist = d
                    best_act = act

            if best_act != ACT_WAIT:
                nx, ny = pos[0] + DX[best_act], pos[1] + DY[best_act]
                actions[bid] = (best_act, -1)
                claimed_positions.add((nx, ny))
            else:
                # Completely stuck, wait
                claimed_positions.add(pos)

        return actions

    def solve(self) -> tuple[int, list[list[tuple[int, int]]]]:
        """Run full game simulation with reactive planning."""
        gs, all_orders = init_game.__wrapped__(self) if False else (None, None)
        # Create fresh game state
        gs = GameState(self.ms)
        gs.bot_positions = np.zeros((self.num_bots, 2), dtype=np.int16)
        gs.bot_inventories = np.full((self.num_bots, INV_CAP), -1, dtype=np.int8)
        for i in range(self.num_bots):
            gs.bot_positions[i] = [self.spawn[0], self.spawn[1]]
        orders_copy = [o.copy() for o in self.all_orders]
        gs.orders = [orders_copy[0].copy(), orders_copy[1].copy()]
        gs.orders[0].status = 'active'
        gs.orders[1].status = 'preview'
        gs.next_order_idx = 2
        gs.active_idx = 0

        action_log = []
        for r in range(self.num_rounds):
            gs.round = r
            round_actions = self.decide_round(gs, orders_copy, r)
            action_log.append(round_actions)
            cpu_step(gs, round_actions, orders_copy)

            if self.verbose and r < 10:
                print(f"  R{r}: score={gs.score} orders={gs.orders_completed}",
                      file=sys.stderr)

        if self.verbose:
            print(f"  Final: score={gs.score} orders={gs.orders_completed} "
                  f"items={gs.items_delivered}", file=sys.stderr)

        return gs.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reactive planner')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('--difficulty', '-d', default='hard')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    diff = args.difficulty
    num_rounds = DIFF_ROUNDS.get(diff, 300)
    cfg = CONFIGS[diff]

    scores = []
    for seed in seeds:
        print(f"\n=== Seed {seed}, {diff} ===", file=sys.stderr)

        gs, all_orders = init_game(seed, diff)
        ms = gs.map_state

        planner = ReactivePlanner(ms, all_orders, cfg['bots'],
                                  num_rounds, verbose=args.verbose)
        score, action_log = planner.solve()
        scores.append(score)
        print(f"Score: {score}", file=sys.stderr)

    if len(scores) > 1:
        print(f"\nSummary: mean={np.mean(scores):.1f}, min={min(scores)}, "
              f"max={max(scores)}", file=sys.stderr)


if __name__ == '__main__':
    main()

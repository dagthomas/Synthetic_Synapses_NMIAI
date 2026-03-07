"""Nightmare solver V4: Maximum throughput pipelining.

Key design principles:
1. ALL 20 bots work at ALL times — zero parking
2. Pipelining: active pickup/deliver | preview pickup simultaneously
3. Every bot at dropoff with inventory delivers immediately (triggers chains)
4. Aggressive preview pre-fetching fills the pipeline
5. Smart adjacent pickup: only pick types that are actually needed (no duplicates)

Target: 800+ points (76+ orders in 500 rounds, ~6.5 rounds/order)
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables


MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class NightmareSolverV4:
    """V4: Maximum throughput pipeline solver."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = CONFIGS['nightmare']['bots']

        # Build walkable set
        self.walkable: set[tuple[int, int]] = set()
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    self.walkable.add((x, y))

        # Pre-loaded future orders
        self.future_orders = future_orders or []

        # Item lookup: type_id -> [(item_idx, adj_positions)]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Position to adjacent items lookup (for fast opportunistic pickup)
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}  # pos -> [(item_idx, type_id)]
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            for adj in ms.item_adjacencies.get(idx, []):
                if adj not in self.pos_to_items:
                    self.pos_to_items[adj] = []
                self.pos_to_items[adj].append((idx, tid))

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

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
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _balanced_drop(self, pos: tuple[int, int],
                        drop_loads: dict[tuple[int, int], int]) -> tuple[int, int]:
        """Pick dropoff balancing distance + current load."""
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + drop_loads.get(dz, 0) * 4
            if score < best_score:
                best_score = score
                best = dz
        return best

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        # Extract state
        bot_pos: dict[int, tuple[int, int]] = {}
        bot_inv: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_pos[bid] = (int(state.bot_positions[bid, 0]),
                           int(state.bot_positions[bid, 1]))
            bot_inv[bid] = state.bot_inv_list(bid)

        # Stall detection
        for bid in range(num_bots):
            if self.prev_positions.get(bid) == bot_pos[bid]:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = bot_pos[bid]

        # Orders
        active = state.get_active_order()
        preview = state.get_preview_order()

        # Compute needs
        active_needs: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs: dict[int, int] = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Count what's being carried
        active_carrying: dict[int, int] = {}
        preview_carrying: dict[int, int] = {}

        for bid in range(num_bots):
            inv = bot_inv[bid]
            for t in inv:
                if t in active_needs:
                    active_carrying[t] = active_carrying.get(t, 0) + 1
                elif t in preview_needs:
                    preview_carrying[t] = preview_carrying.get(t, 0) + 1

        # Shortfalls
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - active_carrying.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_short: dict[int, int] = {}
        for t, need in preview_needs.items():
            s = need - preview_carrying.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # ===== PHASE 1: Immediate actions at current position =====
        # Bots at dropoff: ALWAYS deliver (triggers chain reactions)
        # Bots adjacent to needed items: pick up
        phase1_done: set[int] = set()

        for bid in range(num_bots):
            pos = bot_pos[bid]
            inv = bot_inv[bid]

            # AT DROPOFF: deliver if carrying active-matching items
            if pos in self.drop_set and inv and active:
                has_active_match = any(active.needs_type(t) for t in inv)
                if has_active_match:
                    actions[bid] = (ACT_DROPOFF, -1)
                    phase1_done.add(bid)
                    continue
                # Non-matching items at dropoff: DO NOT deliver, leave the dropoff
                # (they would waste a round and not score)

            # ADJACENT to needed item: opportunistic pickup
            if len(inv) < INV_CAP and pos in self.pos_to_items:
                pickup = self._smart_adjacent_pickup(
                    pos, inv, active_needs, active_short, preview_needs, preview_short)
                if pickup is not None:
                    actions[bid] = (ACT_PICKUP, pickup)
                    phase1_done.add(bid)
                    continue

        # ===== PHASE 2: Assign goals to remaining bots =====
        # Track assignment counts to prevent over-assignment
        active_assigned: dict[int, int] = dict(active_carrying)
        preview_assigned: dict[int, int] = dict(preview_carrying)
        claimed_items: set[int] = set()
        drop_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        # Sort bots by distance to nearest needed item (closest first = most efficient)
        remaining_bots = [bid for bid in range(num_bots) if bid not in phase1_done]
        remaining_bots.sort(key=lambda bid: self._min_dist_to_types(
            bot_pos[bid], active_short.keys() if active_short else preview_needs.keys()))

        for bid in remaining_bots:
            pos = bot_pos[bid]
            inv = bot_inv[bid]
            free_slots = INV_CAP - len(inv)

            # CARRYING items: decide deliver or fill-up
            if inv:
                has_active = any(t in active_needs for t in inv)
                has_preview = any(t in preview_needs for t in inv)

                if has_active:
                    # Active carrier: fill up if close to more active items, else deliver
                    if free_slots > 0 and active_short:
                        remaining_active = {t: max(0, s - active_assigned.get(t, 0) + active_carrying.get(t, 0))
                                          for t, s in active_short.items()}
                        remaining_active = {t: s for t, s in remaining_active.items() if s > 0}
                        if remaining_active:
                            best = self._assign_pickup(bid, pos, remaining_active,
                                                       active_assigned, claimed_items)
                            if best:
                                item_idx, adj_pos = best
                                pick_dist = self.tables.get_distance(pos, adj_pos)
                                drop_dist = self._drop_dist(pos)
                                # Only fill up if item is closer than dropoff
                                if pick_dist < drop_dist:
                                    goals[bid] = adj_pos
                                    goal_types[bid] = 'pickup'
                                    pickup_targets[bid] = item_idx
                                    continue
                                else:
                                    # Undo assignment
                                    tid = int(ms.item_types[item_idx])
                                    active_assigned[tid] -= 1
                                    claimed_items.discard(item_idx)

                    # Deliver
                    dz = self._balanced_drop(pos, drop_loads)
                    drop_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'
                    continue

                elif has_preview:
                    # Preview carrier: fill up with more preview items if possible
                    if free_slots > 0 and preview_short:
                        best = self._assign_pickup(bid, pos, preview_short,
                                                   preview_assigned, claimed_items)
                        if best:
                            item_idx, adj_pos = best
                            pick_dist = self.tables.get_distance(pos, adj_pos)
                            drop_dist = self._drop_dist(pos)
                            if pick_dist < drop_dist:
                                goals[bid] = adj_pos
                                goal_types[bid] = 'pickup'
                                pickup_targets[bid] = item_idx
                                continue
                            else:
                                tid = int(ms.item_types[item_idx])
                                preview_assigned[tid] -= 1
                                claimed_items.discard(item_idx)

                    # Go NEAR dropoff (1-2 steps away, not ON it)
                    # This keeps the dropoff clear for active carriers while
                    # staying close enough for future chain reactions
                    dz = self._balanced_drop(pos, drop_loads)
                    near = self._near_drop_cell(dz, set(goals.values()))
                    goals[bid] = near
                    goal_types[bid] = 'stage_near'
                    continue
                else:
                    # Dead inventory: go near dropoff (items might become useful
                    # via chain reaction when active order changes)
                    dz = self._nearest_drop(pos)
                    near = self._near_drop_cell(dz, set(goals.values()))
                    goals[bid] = near
                    goal_types[bid] = 'stage_near'
                    continue

            # EMPTY: pick up items
            assigned = False

            # 1. Active shortfall
            if active_short and not assigned:
                best = self._assign_pickup(bid, pos, active_short,
                                           active_assigned, claimed_items)
                if best:
                    item_idx, adj_pos = best
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    assigned = True

            # 2. Preview shortfall
            if not assigned and preview_short:
                best = self._assign_pickup(bid, pos, preview_short,
                                           preview_assigned, claimed_items)
                if best:
                    item_idx, adj_pos = best
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    assigned = True

            # 3. Future pre-fetch
            if not assigned:
                future_list = self._get_future(state, all_orders)
                future_short: dict[int, int] = {}
                for order in future_list[:3]:
                    for t in order.needs():
                        future_short[t] = future_short.get(t, 0) + 1
                if future_short:
                    future_assigned: dict[int, int] = {}
                    best = self._assign_pickup(bid, pos, future_short,
                                               future_assigned, claimed_items,
                                               max_per_type=1)
                    if best:
                        item_idx, adj_pos = best
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        assigned = True

            # 4. Pre-position near center
            if not assigned:
                # Go to a spread position to be ready
                target_x = 4 + (bid * 22 // num_bots)
                goals[bid] = (target_x, 9)  # Middle corridor
                goal_types[bid] = 'idle'

        # ===== PHASE 3: Pathfinding =====
        priority_map = {'deliver': 0, 'pickup': 1, 'stage_near': 2, 'idle': 3}
        urgency_order = sorted(goals.keys(), key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'idle'), 3),
            self.tables.get_distance(bot_pos[bid], goals[bid])
        ))

        claims: dict[tuple[int, int], int] = {}
        planned_dest: dict[int, tuple[int, int]] = {}

        # Mark phase1 bots' positions as claimed (they're staying/acting)
        for bid in phase1_done:
            pos = bot_pos[bid]
            if pos not in claims or pos == self.spawn:
                claims[pos] = bid
            planned_dest[bid] = pos

        for bid in urgency_order:
            pos = bot_pos[bid]
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 4:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                self.stall_counts[bid] = 0
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                dest = (nx, ny) if (nx, ny) in self.walkable else pos
                if dest not in claims or dest == self.spawn:
                    claims[dest] = bid
                planned_dest[bid] = dest
                continue

            if pos == goal:
                # At goal — for pickup, do the pickup
                if goal_types.get(bid) == 'pickup' and bid in pickup_targets:
                    actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                else:
                    actions[bid] = (ACT_WAIT, -1)
                if pos not in claims or pos == self.spawn:
                    claims[pos] = bid
                planned_dest[bid] = pos
                continue

            # Navigate toward goal
            candidates = self._rank_moves(pos, goal)
            assigned = False
            for act, dest in candidates:
                if dest == self.spawn:
                    claims[dest] = bid
                    actions[bid] = (act, -1)
                    planned_dest[bid] = dest
                    assigned = True
                    break
                if dest not in claims:
                    # Swap detection
                    swap = False
                    for ob, op in bot_pos.items():
                        if op == dest and ob != bid and ob in planned_dest:
                            if planned_dest[ob] == pos:
                                swap = True
                                break
                    if not swap:
                        claims[dest] = bid
                        actions[bid] = (act, -1)
                        planned_dest[bid] = dest
                        assigned = True
                        break

            if not assigned:
                if pos not in claims or pos == self.spawn:
                    claims[pos] = bid
                actions[bid] = (ACT_WAIT, -1)
                planned_dest[bid] = pos

        return actions

    def _smart_adjacent_pickup(self, pos: tuple[int, int], inv: list[int],
                                active_needs: dict[int, int],
                                active_short: dict[int, int],
                                preview_needs: dict[int, int],
                                preview_short: dict[int, int]) -> int | None:
        """Pick up best adjacent item. Returns item_idx or None.

        Priority: active-needed > preview-needed.
        Never pick a type we already have unless order needs multiple copies.
        """
        bot_types = Counter(inv)
        best_idx = None
        best_priority = -1

        for item_idx, tid in self.pos_to_items.get(pos, []):
            # Active item shortfall
            if tid in active_short and active_short[tid] > 0:
                # Don't pick if we already carry this type AND shortfall is only 1
                if bot_types.get(tid, 0) > 0 and active_short[tid] <= 1:
                    continue
                priority = 10
            elif tid in preview_short and preview_short[tid] > 0:
                if bot_types.get(tid, 0) > 0 and preview_short[tid] <= 1:
                    continue
                priority = 5
            else:
                continue

            if priority > best_priority:
                best_priority = priority
                best_idx = item_idx

        return best_idx

    def _assign_pickup(self, bid: int, pos: tuple[int, int],
                        shortfall: dict[int, int],
                        assigned_counts: dict[int, int],
                        claimed: set[int],
                        max_per_type: int = 0) -> tuple[int, tuple[int, int]] | None:
        """Assign best item to pick up. Returns (item_idx, adj_pos) or None."""
        best_idx = None
        best_adj = None
        best_cost = 9999

        for tid, need in shortfall.items():
            if need <= 0:
                continue
            limit = need if max_per_type == 0 else min(need, max_per_type)
            if assigned_counts.get(tid, 0) >= limit:
                continue
            for item_idx, adj_list in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_list:
                    d = self.tables.get_distance(pos, adj)
                    # Add small drop distance factor to prefer items near dropoffs
                    drop_d = self._drop_dist(adj)
                    cost = d + drop_d * 0.2
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj

        if best_idx is not None:
            tid = int(self.ms.item_types[best_idx])
            assigned_counts[tid] = assigned_counts.get(tid, 0) + 1
            claimed.add(best_idx)
            return (best_idx, best_adj)
        return None

    def _near_drop_cell(self, dropoff: tuple[int, int],
                         occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find a walkable cell 1-2 steps from the dropoff, not the dropoff itself."""
        dx, dy = dropoff
        best = None
        best_d = 9999
        for nx, ny in [(dx-1, dy), (dx+1, dy), (dx, dy-1), (dx, dy+1),
                       (dx-2, dy), (dx+2, dy), (dx, dy-2), (dx, dy+2)]:
            cell = (nx, ny)
            if cell in self.drop_set:
                continue
            if cell not in self.walkable:
                continue
            if cell in occupied:
                continue
            d = abs(nx - dx) + abs(ny - dy)
            if d < best_d:
                best_d = d
                best = cell
        return best or dropoff  # Fallback to dropoff itself if no nearby cell

    def _min_dist_to_types(self, pos: tuple[int, int], types) -> int:
        best = 9999
        for tid in types:
            for item_idx, adj_list in self.type_items.get(tid, []):
                for adj in adj_list:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

    def _rank_moves(self, pos: tuple[int, int],
                     goal: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        candidates = []
        optimal = self.tables.get_first_step(pos, goal)
        for act in MOVES:
            nx, ny = pos[0] + DX[act], pos[1] + DY[act]
            dest = (nx, ny)
            if dest not in self.walkable:
                continue
            d = self.tables.get_distance(dest, goal)
            bonus = -0.5 if act == optimal else 0.0
            candidates.append((d + bonus, act, dest))
        candidates.sort()
        return [(act, dest) for _, act, dest in candidates]

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int) -> int:
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def _get_future(self, state: GameState, all_orders: list[Order]) -> list[Order]:
        future = []
        preview = state.get_preview_order()
        if preview:
            future.append(preview)
        if all_orders:
            for i in range(state.next_order_idx, min(state.next_order_idx + 6, len(all_orders))):
                future.append(all_orders[i])
        elif self.future_orders:
            start = state.orders_completed + 2
            for i in range(start, min(start + 6, len(self.future_orders))):
                future.append(self.future_orders[i])
        return future

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV4(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
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

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
                dropoff_info = ""
                if c >= 1:
                    at_drop = []
                    for b in range(len(state.bot_positions)):
                        bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                        if bp in drop_set:
                            binv = state.bot_inv_list(b)
                            at_drop.append(f"b{b}:{binv}")
                    if at_drop:
                        dropoff_info = f" AtDrop=[{', '.join(at_drop)}]"
                extra = f" CHAIN×{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra + dropoff_info)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7001-7010')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    for seed in seeds:
        score, _ = NightmareSolverV4.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f"Seed {seed}: {score}")

    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")


if __name__ == '__main__':
    main()

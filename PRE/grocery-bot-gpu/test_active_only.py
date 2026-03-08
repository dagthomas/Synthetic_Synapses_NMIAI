"""Quick test: active-only solver (no preview, no staging)."""
import time
import numpy as np
from game_engine import (
    init_game, step, ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN,
    ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


def run_active_only(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    walkable = build_walkable(ms)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
    drop_set = set(drop_zones)
    spawn = ms.spawn
    traffic = TrafficRules(ms)
    congestion = CongestionMap()
    pf = NightmarePathfinder(ms, tables, traffic, congestion)

    type_items = {}
    for idx in range(ms.num_items):
        tid = int(ms.item_types[idx])
        adj = ms.item_adjacencies.get(idx, [])
        type_items.setdefault(tid, []).append((idx, adj))

    num_rounds = DIFF_ROUNDS['nightmare']
    corridor_ys = [1, ms.height // 2, ms.height - 3]
    stall_counts = {}
    prev_pos = {}

    for rnd in range(num_rounds):
        state.round = rnd
        num_bots = len(state.bot_positions)
        positions = {}
        inventories = {}
        for bid in range(num_bots):
            positions[bid] = (int(state.bot_positions[bid, 0]),
                              int(state.bot_positions[bid, 1]))
            inventories[bid] = state.bot_inv_list(bid)

        congestion.update(list(positions.values()))
        for bid in range(num_bots):
            pos = positions[bid]
            if prev_pos.get(bid) == pos:
                stall_counts[bid] = stall_counts.get(bid, 0) + 1
            else:
                stall_counts[bid] = 0
            prev_pos[bid] = pos

        active = state.get_active_order()
        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        carrying = {}
        for bid in range(num_bots):
            for t in inventories[bid]:
                if t in active_needs:
                    carrying[t] = carrying.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying.get(t, 0)
            if s > 0:
                active_short[t] = s

        carriers = []
        empties = []
        deads = []
        for bid in range(num_bots):
            inv = inventories[bid]
            if any(t in active_needs for t in inv):
                carriers.append(bid)
            elif len(inv) < INV_CAP:
                empties.append(bid)
            else:
                deads.append(bid)

        goals = {}
        goal_types = {}
        pickup_targets = {}
        type_assigned = {}
        claimed = set()
        drop_loads = {dz: 0 for dz in drop_zones}
        occupied = set()

        def balanced_drop(p, loads):
            return min(drop_zones, key=lambda dz: tables.get_distance(p, dz) + loads.get(dz, 0) * 5)

        # Carriers deliver
        for bid in carriers:
            pos = positions[bid]
            inv = inventories[bid]
            free = INV_CAP - len(inv)
            # If has free slots and items nearby, fill up first
            if free > 0 and active_short:
                best_idx, best_adj, best_cost = None, None, 9999
                for tid, need in active_short.items():
                    if need - type_assigned.get(tid, 0) <= 0:
                        continue
                    for idx, adjs in type_items.get(tid, []):
                        if idx in claimed:
                            continue
                        for adj in adjs:
                            d = tables.get_distance(pos, adj)
                            dd = min(tables.get_distance(adj, dz) for dz in drop_zones)
                            cost = d + dd * 0.4
                            if cost < best_cost:
                                best_cost = cost
                                best_idx = idx
                                best_adj = adj
                dz = balanced_drop(pos, drop_loads)
                drop_d = tables.get_distance(pos, dz)
                if best_idx is not None and best_cost < drop_d and best_cost < 10:
                    goals[bid] = best_adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = best_idx
                    tid = int(ms.item_types[best_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed.add(best_idx)
                    continue

            dz = balanced_drop(pos, drop_loads)
            drop_loads[dz] += 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # Empties pick active items ONLY
        empties_sorted = sorted(empties, key=lambda bid:
            min((tables.get_distance(positions[bid], adj)
                 for tid in active_short
                 for idx, adjs in type_items.get(tid, [])
                 for adj in adjs), default=9999))

        for bid in empties_sorted:
            pos = positions[bid]
            if active_short:
                best_idx, best_adj, best_cost = None, None, 9999
                for tid, need in active_short.items():
                    if need - type_assigned.get(tid, 0) <= 0:
                        continue
                    for idx, adjs in type_items.get(tid, []):
                        if idx in claimed:
                            continue
                        for adj in adjs:
                            d = tables.get_distance(pos, adj)
                            dd = min(tables.get_distance(adj, dz) for dz in drop_zones)
                            cost = d + dd * 0.4
                            if cost < best_cost:
                                best_cost = cost
                                best_idx = idx
                                best_adj = adj
                if best_idx is not None:
                    goals[bid] = best_adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = best_idx
                    tid = int(ms.item_types[best_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed.add(best_idx)
                    continue

            # Park
            best_park = spawn
            best_d = 9999
            for cy in corridor_ys:
                for cx in range(1, ms.width - 1):
                    cell = (cx, cy)
                    if cell in tables.pos_to_idx and cell not in occupied:
                        if any(tables.get_distance(cell, dz) <= 1 for dz in drop_zones):
                            continue
                        d = tables.get_distance(pos, cell)
                        if 0 < d < best_d:
                            best_d = d
                            best_park = cell
            occupied.add(best_park)
            goals[bid] = best_park
            goal_types[bid] = 'park'

        for bid in deads:
            pos = positions[bid]
            best_park = spawn
            best_d = 9999
            for cy in corridor_ys:
                for cx in range(1, ms.width - 1):
                    cell = (cx, cy)
                    if cell in tables.pos_to_idx and cell not in occupied:
                        if any(tables.get_distance(cell, dz) <= 1 for dz in drop_zones):
                            continue
                        d = tables.get_distance(pos, cell)
                        if 0 < d < best_d:
                            best_d = d
                            best_park = cell
            occupied.add(best_park)
            goals[bid] = best_park
            goal_types[bid] = 'flee'

        pri = {'deliver': 0, 'flee': 1, 'pickup': 2, 'park': 5}
        urgency = sorted(range(num_bots), key=lambda bid: (
            pri.get(goal_types.get(bid, 'park'), 5),
            tables.get_distance(positions[bid], goals.get(bid, spawn))))
        path_actions = pf.plan_all(positions, goals, urgency, goal_types=goal_types)

        actions = [(ACT_WAIT, -1)] * num_bots
        for bid in range(num_bots):
            pos = positions[bid]
            gt = goal_types.get(bid, 'park')
            inv = inventories[bid]

            if stall_counts.get(bid, 0) >= 3:
                dirs = list(MOVES)
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    if (nx, ny) in walkable:
                        actions[bid] = (a, -1)
                        break
                continue

            if pos in drop_set and gt == 'deliver' and inv:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            if gt == 'pickup' and bid in pickup_targets and pos == goals[bid]:
                actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                continue

            # Adjacent active pickup
            if len(inv) < INV_CAP and active_short:
                bot_types = set(inv)
                found = False
                for item_idx in range(ms.num_items):
                    tid = int(ms.item_types[item_idx])
                    if tid not in active_short or active_short[tid] <= 0:
                        continue
                    if tid in bot_types and active_short[tid] <= 1:
                        continue
                    for adj in ms.item_adjacencies.get(item_idx, []):
                        if adj == pos:
                            actions[bid] = (ACT_PICKUP, item_idx)
                            active_short[tid] -= 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

            actions[bid] = (path_actions.get(bid, ACT_WAIT), -1)

        step(state, actions, all_orders)

    return state.score, state.orders_completed


if __name__ == '__main__':
    scores = []
    for seed in range(1000, 1010):
        score, orders = run_active_only(seed)
        scores.append(score)
        print(f"Seed {seed}: {score} ({orders} orders)")

    print(f"\nMean: {np.mean(scores):.1f}  Max: {max(scores)}  Min: {min(scores)}")

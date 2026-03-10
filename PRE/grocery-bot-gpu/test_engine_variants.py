"""Compare V3 on live map with 3 engine variants."""
import sys, numpy as np
from game_engine import (
    GameState, Order, MapState,
    ACT_WAIT, ACT_DROPOFF, ACT_PICKUP, INV_CAP,
    CELL_DROPOFF, CELL_FLOOR, DX, DY,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    build_map_from_capture, generate_all_orders,
)
from configs import CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v2 import NightmareSolverV3
from solution_store import load_capture


def _move(state, bid, act_type, occupied, ms):
    bx, by = int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])
    nx, ny = bx + DX[act_type], by + DY[act_type]
    if 0 <= nx < ms.width and 0 <= ny < ms.height:
        cell = ms.grid[ny, nx]
        if cell in (CELL_FLOOR, CELL_DROPOFF):
            if len(occupied.get((nx, ny), set())) == 0 or (nx, ny) == ms.spawn:
                occupied[(bx, by)].discard(bid)
                if not occupied[(bx, by)]:
                    del occupied[(bx, by)]
                state.bot_positions[bid] = [nx, ny]
                if (nx, ny) not in occupied:
                    occupied[(nx, ny)] = set()
                occupied[(nx, ny)].add(bid)


def _pickup(state, bid, item_idx, ms):
    bx, by = int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])
    if item_idx >= 0 and state.bot_inv_count(bid) < INV_CAP:
        ix = int(ms.item_positions[item_idx, 0])
        iy = int(ms.item_positions[item_idx, 1])
        if abs(bx - ix) + abs(by - iy) == 1:
            state.bot_inv_add(bid, int(ms.item_types[item_idx]))


def _chain(state, bid, all_orders, score_delta, auto_mode):
    """Handle order completion cascade. auto_mode: 'bid_only', 'dropoff', 'all'"""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
    active = state.get_active_order()
    while active and active.is_complete():
        active.complete = True
        score_delta += 5
        state.score += 5
        state.orders_completed += 1
        for o in state.orders:
            if not o.complete and o.status == 'preview':
                o.status = 'active'
                break
        if state.next_order_idx < len(all_orders):
            new_order = all_orders[state.next_order_idx].copy()
            new_order.status = 'preview'
            state.orders.append(new_order)
            state.next_order_idx += 1
        active = state.get_active_order()
        if active:
            if auto_mode == 'bid_only':
                d = state.bot_inv_remove_matching(bid, active)
                score_delta += d; state.score += d; state.items_delivered += d
            elif auto_mode == 'dropoff':
                for b2 in range(num_bots):
                    p = (int(state.bot_positions[b2, 0]), int(state.bot_positions[b2, 1]))
                    if p in drop_set:
                        d = state.bot_inv_remove_matching(b2, active)
                        score_delta += d; state.score += d; state.items_delivered += d
            else:  # all
                for b2 in range(num_bots):
                    d = state.bot_inv_remove_matching(b2, active)
                    score_delta += d; state.score += d; state.items_delivered += d
    return score_delta


def step_variant(state, actions, all_orders, auto_mode):
    ms = state.map_state
    num_bots = len(state.bot_positions)
    score_delta = 0
    occupied = {}
    for bid in range(num_bots):
        pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
        if pos not in occupied:
            occupied[pos] = set()
        occupied[pos].add(bid)

    for bid in range(num_bots):
        act_type, item_idx = actions[bid] if bid < len(actions) else (ACT_WAIT, -1)
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])

        if act_type in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
            _move(state, bid, act_type, occupied, ms)
        elif act_type == ACT_PICKUP:
            _pickup(state, bid, item_idx, ms)
        elif act_type == ACT_DROPOFF:
            if any(bx == dz[0] and by == dz[1] for dz in ms.drop_off_zones):
                if state.bot_inv_count(bid) > 0:
                    active = state.get_active_order()
                    if active:
                        d = state.bot_inv_remove_matching(bid, active)
                        score_delta += d
                        state.score += d
                        state.items_delivered += d
                        score_delta = _chain(state, bid, all_orders, score_delta, auto_mode)

    state.round += 1
    return score_delta


cap = load_capture('nightmare')
ms = build_map_from_capture(cap)
tables = PrecomputedTables.get(ms)
all_orders = generate_all_orders(7005, ms, 'nightmare', count=100)

for name, mode in [('delivering_bot_only', 'bid_only'),
                    ('dropoff_zone_bots', 'dropoff'),
                    ('all_bots', 'all')]:
    state = GameState(ms)
    num_bots = CONFIGS['nightmare']['bots']
    state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
    state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
    for i in range(num_bots):
        state.bot_positions[i] = [ms.spawn[0], ms.spawn[1]]
    state.orders = [all_orders[0].copy(), all_orders[1].copy()]
    state.orders[0].status = 'active'
    state.orders[1].status = 'preview'
    state.next_order_idx = 2
    state.active_idx = 0

    solver = NightmareSolverV3(ms, tables, future_orders=all_orders)
    for rnd in range(500):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        step_variant(state, actions, all_orders, mode)
    print(f"{name:25s}: Score={state.score:3d} Orders={state.orders_completed}")

print(f"{'live_server':25s}: Score=237 Orders=24")

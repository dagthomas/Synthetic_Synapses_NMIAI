#!/usr/bin/env python3
"""Local test for CaptureController — no server tokens needed.

Uses game_engine to simulate the game and feeds state_to_ws_format()
output into CaptureController.decide().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from game_engine import (
    init_game, step, state_to_ws_format,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF,
)
from configs import DIFF_ROUNDS
from capture_game import CaptureController, ST_PICKING, ST_DELIVERING, ST_IDLE

ACTION_MAP = {
    "wait": ACT_WAIT,
    "move_up": ACT_MOVE_UP,
    "move_down": ACT_MOVE_DOWN,
    "move_left": ACT_MOVE_LEFT,
    "move_right": ACT_MOVE_RIGHT,
    "pick_up": ACT_PICKUP,
    "drop_off": ACT_DROPOFF,
}


def run_local_test(seed=7005, difficulty="nightmare", verbose=False):
    state, all_orders = init_game(seed, difficulty)
    num_rounds = DIFF_ROUNDS.get(difficulty, 300)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    # Build item_id -> item_idx mapping for pick_up
    item_id_to_idx = {}
    for idx, item in enumerate(ms.items):
        item_id_to_idx[item['id']] = idx

    controller = CaptureController()
    seen_order_ids = set()
    captured_orders = []

    print(f"Local test: seed={seed}, difficulty={difficulty}")
    print(f"Map: {ms.width}x{ms.height}, {num_bots} bots, {len(ms.items)} items")
    print(f"Rounds: {num_rounds}")
    drops_info = ms.drop_off_zones
    print(f"Drop-off zones: {drops_info}")
    print(f"Spawn: {ms.spawn}")
    print()

    for rnd in range(num_rounds):
        state.round = rnd
        ws_data = state_to_ws_format(state, all_orders)

        # Track orders discovered
        for order in ws_data["orders"]:
            oid = order["id"]
            if oid not in seen_order_ids:
                seen_order_ids.add(oid)
                captured_orders.append({
                    'id': oid,
                    'items_required': list(order['items_required']),
                })

        # Get controller decisions (websocket-format actions)
        walls_set = set()
        for w in ws_data["grid"]["walls"]:
            walls_set.add((w[0], w[1]))
        for item in ws_data["items"]:
            walls_set.add((item["position"][0], item["position"][1]))

        ws_actions = controller.decide(ws_data, walls_set, ms.width, ms.height)

        # Convert WS actions to game_engine action tuples
        engine_actions = [(ACT_WAIT, -1)] * num_bots
        for wa in ws_actions:
            bid = wa["bot"]
            act_name = wa["action"]
            act_type = ACTION_MAP.get(act_name, ACT_WAIT)

            item_idx = -1
            if act_type == ACT_PICKUP and "item_id" in wa:
                item_idx = item_id_to_idx.get(wa["item_id"], -1)

            engine_actions[bid] = (act_type, item_idx)

        # Step the game
        score_before = state.score
        step(state, engine_actions, all_orders)
        score_delta = state.score - score_before

        # Detailed logging
        if verbose and (rnd < 20 or rnd % 25 == 0 or score_delta > 0 or rnd == num_rounds - 1):
            n_pick = sum(1 for cb in controller.bots.values() if cb.state == ST_PICKING)
            n_del = sum(1 for cb in controller.bots.values() if cb.state == ST_DELIVERING)
            n_idle = sum(1 for cb in controller.bots.values() if cb.state == ST_IDLE)
            stuck = [(cb.bid, cb.stuck_count) for cb in controller.bots.values()
                     if cb.stuck_count > 2 or cb.stuck_count < 0]
            print(f"  R{rnd:3d}: score={state.score:3d} (+{score_delta}) "
                  f"orders_done={state.orders_completed} seen={len(captured_orders)} "
                  f"pick={n_pick} del={n_del} idle={n_idle}"
                  + (f" stuck={stuck}" if stuck else ""))

            # Show active workers' positions and targets
            if rnd < 20 or rnd % 100 == 0:
                for cb in controller.bots.values():
                    if cb.state != ST_IDLE:
                        bot_data = next(b for b in ws_data["bots"] if b["id"] == cb.bid)
                        st_name = {0: "IDLE", 1: "PICK", 2: "DELV", 3: "WAIT"}[cb.state]
                        print(f"    Bot{cb.bid}: {st_name} pos={bot_data['position']} "
                              f"target={cb.target} inv={bot_data['inventory']} "
                              f"stuck={cb.stuck_count}")

        elif not verbose and (rnd < 5 or rnd % 50 == 0 or score_delta > 0 or rnd == num_rounds - 1):
            n_pick = sum(1 for cb in controller.bots.values() if cb.state == ST_PICKING)
            n_del = sum(1 for cb in controller.bots.values() if cb.state == ST_DELIVERING)
            n_idle = sum(1 for cb in controller.bots.values() if cb.state == ST_IDLE)
            print(f"  R{rnd:3d}: score={state.score:3d} orders_done={state.orders_completed} "
                  f"orders_seen={len(captured_orders)} "
                  f"pick={n_pick} del={n_del} idle={n_idle}")

    print()
    print(f"Final score: {state.score}")
    print(f"Orders completed: {state.orders_completed}")
    print(f"Orders discovered: {len(captured_orders)}")
    print(f"Items delivered: {state.items_delivered}")
    for i, o in enumerate(captured_orders):
        print(f"  Order {i}: {o['items_required']}")

    return state.score, captured_orders


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7005
    diff = sys.argv[2] if len(sys.argv) > 2 else "nightmare"
    verbose = "-v" in sys.argv
    run_local_test(seed, diff, verbose)

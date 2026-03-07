"""Quick live verification: do cracked orders match what the server sends?

Connects to a live game, plays with capture_solver, and checks every
order against our cracked predictions.
"""
import asyncio
import json
import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import Order, build_map_from_capture
from configs import CONFIGS
from capture_solver import solve_for_capture
from live_solver import detect_difficulty, ws_to_capture

SEED = 7004
RNG_SKIP = 509
SERVER_TYPE_LIST = [
    'bananas', 'apples', 'onions', 'oats', 'butter', 'flour',
    'cheese', 'cream', 'milk', 'pasta', 'yogurt', 'cereal',
    'eggs', 'rice', 'bread', 'tomatoes',
]


def generate_cracked_order_list(capture_data, num_orders=50):
    """Generate predicted orders from cracked seed."""
    ms = build_map_from_capture(capture_data)
    order_size = CONFIGS['expert']['order_size']

    item_counts = {}
    for item in capture_data['items']:
        t = item['type']
        item_counts[t] = item_counts.get(t, 0) + 1
    available_counts = {t: item_counts[t] for t in SERVER_TYPE_LIST}
    avail_types = list(available_counts.keys())

    rng = random.Random(SEED)
    for _ in range(RNG_SKIP):
        rng.random()

    orders = []
    for i in range(num_orders):
        n = rng.randint(order_size[0], order_size[1])
        temp_counts = dict(available_counts)
        required_names = []
        for _ in range(n):
            usable = [t for t in avail_types if temp_counts.get(t, 0) > 0]
            if not usable:
                usable = avail_types
            t = rng.choice(usable)
            required_names.append(t)
            if t in temp_counts:
                temp_counts[t] -= 1
        orders.append(required_names)

    return orders


async def verify_live(ws_url):
    import websockets

    print("Connecting to live game...", file=sys.stderr)

    # We'll build capture from first message, then generate cracked orders
    cracked_orders = None
    seen_order_ids = set()
    order_index = 0  # which cracked order to compare next
    matches = 0
    mismatches = 0

    # Simple greedy bot for playing
    from replay_solution import greedy_action, build_walkable, _ACT_NAMES

    map_state = None
    walkable = None

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                score = data.get('score', 0)
                print(f"\nGAME OVER - Score: {score}", file=sys.stderr)
                print(f"Order verification: {matches} matches, {mismatches} mismatches",
                      file=sys.stderr)
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]

            # First round: build capture and generate cracked orders
            if rnd == 0:
                capture = ws_to_capture(data)
                map_state = build_map_from_capture(capture)
                walkable = build_walkable(map_state)
                cracked_orders = generate_cracked_order_list(capture, 50)
                print(f"Generated 50 cracked order predictions", file=sys.stderr)
                print(f"Map: {map_state.width}x{map_state.height}, "
                      f"{len(data['bots'])} bots", file=sys.stderr)

            # Check orders against predictions
            for order in data.get('orders', []):
                oid = order.get('id', '')
                if oid in seen_order_ids:
                    continue
                seen_order_ids.add(oid)

                actual = order['items_required']
                if order_index < len(cracked_orders):
                    predicted = cracked_orders[order_index]
                    if actual == predicted:
                        matches += 1
                        status = "MATCH"
                    else:
                        mismatches += 1
                        status = "MISMATCH"
                    print(f"  Order {order_index}: {status}", file=sys.stderr)
                    print(f"    Actual:    {actual}", file=sys.stderr)
                    if status == "MISMATCH":
                        print(f"    Predicted: {predicted}", file=sys.stderr)
                    order_index += 1
                else:
                    print(f"  Order {order_index}: (no prediction)", file=sys.stderr)

            # Play with greedy bot
            live_bots = data.get('bots', [])
            ws_actions = []
            for bot in live_bots:
                ws_actions.append(
                    greedy_action(bot, data, map_state, walkable, live_bots))

            if rnd % 25 == 0 or rnd < 5:
                score = data.get('score', 0)
                print(f"R{rnd}: score={score} orders_seen={order_index} "
                      f"match={matches} mismatch={mismatches}", file=sys.stderr)

            await asyncio.sleep(0.005)
            await ws.send(json.dumps({"actions": ws_actions}))

    print(f"\nFinal: {matches}/{matches+mismatches} orders matched predictions",
          file=sys.stderr)
    return mismatches == 0


if __name__ == '__main__':
    ws_url = sys.argv[1]
    asyncio.run(verify_live(ws_url))

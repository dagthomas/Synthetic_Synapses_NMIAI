"""Analyze a game recording for per-order timing and issues."""
import json
import sys


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    result = data.get("result", {})
    print("=== GAME RESULT ===")
    print(f"  Score: {result.get('score', '?')}")
    print(f"  Rounds used: {result.get('rounds_used', '?')}")
    print(f"  Items delivered: {result.get('items_delivered', '?')}")
    print(f"  Orders completed: {result.get('orders_completed', '?')}")
    print()

    # Track order transitions
    seen_orders = {}
    order_list = []
    prev_active_id = None

    for rnd_data in data["rounds"]:
        state = rnd_data["state"]
        rnd = state["round"]

        active = next((o for o in state["orders"] if o.get("status") == "active"), None)
        if not active:
            continue

        aid = active["id"]
        if aid not in seen_orders:
            seen_orders[aid] = {
                "start": rnd,
                "required": active["items_required"],
            }
            order_list.append(aid)

        if aid != prev_active_id:
            if prev_active_id is not None:
                seen_orders[prev_active_id]["end"] = rnd
                seen_orders[prev_active_id]["rounds"] = rnd - seen_orders[prev_active_id]["start"]
            prev_active_id = aid

    if prev_active_id and "end" not in seen_orders[prev_active_id]:
        seen_orders[prev_active_id]["end"] = 300
        seen_orders[prev_active_id]["rounds"] = 300 - seen_orders[prev_active_id]["start"]

    print("=== PER-ORDER TIMING ===")
    for i, oid in enumerate(order_list):
        o = seen_orders[oid]
        rounds = o.get("rounds", "?")
        items = o["required"]
        print(f"  Order {i}: {len(items)} items {items} | {rounds} rounds (R{o['start']}-{o.get('end', '?')})")

    print()
    print("=== SLOW ORDERS (>25 rounds) ===")
    slow = [(i, seen_orders[oid]) for i, oid in enumerate(order_list) if seen_orders[oid].get("rounds", 0) > 25]
    if not slow:
        print("  None!")
    for i, o in slow:
        print(f"  Order {i}: {o['rounds']} rounds | {o['required']}")

    # Bot idle analysis
    print()
    print("=== BOT ACTIVITY ===")
    n_bots = len(data["rounds"][0]["state"]["bots"])
    idle_counts = [0] * n_bots
    total_rounds = len(data["rounds"])

    for rnd_data in data["rounds"]:
        actions = rnd_data["actions"]
        actions_by_bot = {a["bot"]: a for a in actions}
        for bid in range(n_bots):
            act = actions_by_bot.get(bid, {}).get("action", "wait")
            if act == "wait":
                idle_counts[bid] += 1

    for bid in range(n_bots):
        pct = idle_counts[bid] / total_rounds * 100
        print(f"  Bot {bid}: {idle_counts[bid]} idle rounds ({pct:.1f}%)")

    # Trace slow orders in detail
    if slow:
        print()
        worst_i, worst_o = max(slow, key=lambda x: x[1]["rounds"])
        worst_oid = order_list[worst_i]
        start, end = worst_o["start"], worst_o["end"]
        print(f"=== DETAILED TRACE: Order {worst_i} (R{start}-{end}, {worst_o['rounds']} rounds) ===")
        print(f"  Needs: {worst_o['required']}")

        for rnd_data in data["rounds"]:
            state = rnd_data["state"]
            rnd = state["round"]
            if rnd < start - 2 or rnd > end + 2:
                continue

            actions = rnd_data["actions"]
            actions_by_bot = {a["bot"]: a for a in actions}
            active = next((o for o in state["orders"] if o.get("status") == "active"), None)
            delivered = len(active["items_delivered"]) if active else "?"
            required = len(active["items_required"]) if active else "?"

            bot_info = []
            for bot in state["bots"]:
                bid = bot["id"]
                pos = tuple(bot["position"])
                inv = bot["inventory"]
                act = actions_by_bot.get(bid, {})
                action = act.get("action", "wait")
                item_id = act.get("item_id", "")
                extra = f" item={item_id}" if item_id else ""
                bot_info.append(f"B{bid}:{pos} inv={inv} -> {action}{extra}")

            print(f"  R{rnd:3d} [{delivered}/{required}] " + " | ".join(bot_info))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "simulation/medium/game_20260304_085552.json"
    analyze(path)

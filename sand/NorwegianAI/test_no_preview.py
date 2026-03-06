"""Test: what happens if we disable all preview item picking?"""
import copy
from simulator import load_game_data, LocalSimulator
from recorder import list_recordings


def decide_actions_no_preview(state, game_plan=None):
    """Simplified brain: only pick active order items, no preview."""
    from distance import DistanceMatrix
    from itertools import permutations

    # Quick access
    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = tuple(state["drop_off"])

    if not hasattr(decide_actions_no_preview, "_dm") or state["round"] == 0:
        decide_actions_no_preview._dm = DistanceMatrix(state)
    dm = decide_actions_no_preview._dm

    active = next((o for o in orders if o["status"] == "active" and not o["complete"]), None)
    if not active:
        return [{"bot": b["id"], "action": "wait"} for b in bots]

    actions = []
    for bot in bots:
        pos = (bot["position"][0], bot["position"][1])
        inv = bot["inventory"]
        bid = bot["id"]

        needed = {}
        for it in active["items_required"]:
            needed[it] = needed.get(it, 0) + 1
        for it in active["items_delivered"]:
            needed[it] = needed.get(it, 0) - 1
        needed = {k: v for k, v in needed.items() if v > 0}

        has_useful = any(needed.get(it, 0) > 0 for it in inv)

        # Drop off at drop zone
        if has_useful and pos == drop_off:
            actions.append({"bot": bid, "action": "drop_off"})
            continue

        # Full inventory → go deliver
        if has_useful and len(inv) >= 3:
            nxt = dm.next_step(pos, drop_off)
            if nxt and nxt != pos:
                dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
                act = {(1,0): "move_right", (-1,0): "move_left", (0,1): "move_down", (0,-1): "move_up"}.get((dx,dy), "wait")
                actions.append({"bot": bid, "action": act})
            else:
                actions.append({"bot": bid, "action": "wait"})
            continue

        # Account for items in inventory
        from_map = dict(needed)
        for it in inv:
            if it in from_map and from_map[it] > 0:
                from_map[it] -= 1
        from_map = {k: v for k, v in from_map.items() if v > 0}

        if from_map and len(inv) < 3:
            # Find closest needed item
            best_item = None
            best_cost = 999
            for item in items:
                if item["type"] in from_map and from_map[item["type"]] > 0:
                    shelf = (item["position"][0], item["position"][1])
                    cost = dm.trip_cost(pos, shelf)
                    if cost < best_cost:
                        best_cost = cost
                        best_item = item

            if best_item:
                shelf = (best_item["position"][0], best_item["position"][1])
                # Check if adjacent
                if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                    actions.append({"bot": bid, "action": "pick_up", "item_id": best_item["id"]})
                    continue
                # Navigate to adjacent
                adj, _ = dm.best_adjacent(pos, shelf)
                if adj:
                    nxt = dm.next_step(pos, adj)
                    if nxt and nxt != pos:
                        dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
                        act = {(1,0): "move_right", (-1,0): "move_left", (0,1): "move_down", (0,-1): "move_up"}.get((dx,dy), "wait")
                        actions.append({"bot": bid, "action": act})
                        continue

        # Go deliver if have useful
        if has_useful:
            nxt = dm.next_step(pos, drop_off)
            if nxt and nxt != pos:
                dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
                act = {(1,0): "move_right", (-1,0): "move_left", (0,1): "move_down", (0,-1): "move_up"}.get((dx,dy), "wait")
                actions.append({"bot": bid, "action": act})
            else:
                actions.append({"bot": bid, "action": "wait"})
            continue

        actions.append({"bot": bid, "action": "wait"})

    return actions


def main():
    recordings = list_recordings("easy")
    game_data = load_game_data(recordings[0])

    # Test with no preview
    sim = LocalSimulator(game_data)
    result = sim.run(decide_actions_no_preview, verbose=True)

    # Compare with normal brain
    from brain import decide_actions
    sim2 = LocalSimulator(game_data)
    print("\n--- Normal brain ---")
    result2 = sim2.run(decide_actions, verbose=True)


if __name__ == "__main__":
    main()

"""Analyze hard mode simulation for duplication, waste, and idle time.

Traces every round to find:
1. Duplicate pickup targets (2+ bots heading for same item type when only 1 needed)
2. Wasted deliveries (items delivered that don't match active order)
3. Idle/waiting bots when there's work to do
4. Total rounds spent on wasted activities
"""

import copy
import json
import sys
from collections import defaultdict

from recorder import list_recordings
from simulator import load_game_data, LocalSimulator
from brain import decide_actions, get_needed_items, _PARAMS
from distance import DistanceMatrix


def analyze_hard():
    recordings = list_recordings("hard")
    if not recordings:
        print("No hard recordings found.")
        return

    recording_path = recordings[0]
    print(f"Recording: {recording_path}")

    game_data = load_game_data(recording_path)
    order_sequence = game_data["order_sequence"]
    print(f"Orders: {len(order_sequence)}, Bots: {len(game_data['bots'])}, Max rounds: {game_data['max_rounds']}")

    sim = LocalSimulator(game_data)
    n_bots = len(sim.bots)

    # Tracking structures
    stats = {
        "total_rounds": 0,
        "total_bot_rounds": 0,
        "action_counts": defaultdict(int),

        # Idle
        "idle_rounds": 0,
        "idle_with_work": 0,
        "idle_no_work": 0,

        # Movement
        "move_rounds": 0,
        "move_to_pickup": 0,
        "move_to_deliver": 0,
        "move_clearing": 0,

        # Pickup
        "pickup_rounds": 0,
        "pickup_active_item": 0,
        "pickup_preview_item": 0,
        "pickup_speculative": 0,

        # Delivery
        "delivery_rounds": 0,
        "items_delivered_useful": 0,      # accepted by active order on dropoff
        "items_at_dropoff_not_needed": 0,  # carried to dropoff but not for active
        "items_preview_at_dropoff": 0,     # subset: match preview order
        "items_truly_wasted": 0,           # subset: match neither order

        # Duplicate targeting
        "duplicate_target_rounds": 0,
        "dup_events": [],

        # Congestion: rounds where a bot intended to move but didn't
        "blocked_move_rounds": 0,

        # Per-order stats
        "order_stats": [],
    }

    dm = None
    current_order_id = None
    order_start_round = 0

    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()

        if rnd == 0:
            dm = DistanceMatrix(state)

        active = next((o for o in state["orders"] if o.get("status") == "active" and not o["complete"]), None)
        preview = next((o for o in state["orders"] if o.get("status") == "preview" and not o["complete"]), None)

        active_needed = get_needed_items(active) if active else {}
        preview_needed = get_needed_items(preview) if preview else {}
        drop_off = tuple(state["drop_off"])

        # Track order transitions
        if active and active["id"] != current_order_id:
            if current_order_id is not None:
                stats["order_stats"].append({
                    "id": current_order_id,
                    "rounds_active": rnd - order_start_round,
                })
            current_order_id = active["id"]
            order_start_round = rnd

        # Items still needed from shelves (subtracting inventories)
        items_from_map = dict(active_needed)
        for bot in state["bots"]:
            for it in bot["inventory"]:
                if it in items_from_map and items_from_map[it] > 0:
                    items_from_map[it] -= 1
        items_from_map = {k: v for k, v in items_from_map.items() if v > 0}

        total_active_needed = sum(active_needed.values())

        # Get brain decisions
        actions = decide_actions(state, game_plan=order_sequence)
        action_map = {a["bot"]: a for a in actions}

        stats["total_rounds"] += 1
        stats["total_bot_rounds"] += n_bots

        # --- Bot-level analysis ---
        pickup_by_type_this_round = defaultdict(list)  # for duplicate detection

        for bot in state["bots"]:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = list(bot["inventory"])
            action = action_map.get(bid, {"action": "wait"})
            act = action.get("action", "wait")

            stats["action_counts"][act] += 1

            # --- WAIT ---
            if act == "wait":
                stats["idle_rounds"] += 1
                if total_active_needed > 0:
                    stats["idle_with_work"] += 1
                else:
                    stats["idle_no_work"] += 1

            # --- MOVE ---
            elif act.startswith("move_"):
                stats["move_rounds"] += 1
                dx_map = {"move_right": (1, 0), "move_left": (-1, 0),
                          "move_down": (0, 1), "move_up": (0, -1)}
                dx, dy = dx_map[act]
                new_pos = (pos[0] + dx, pos[1] + dy)

                d_before = dm.dist(pos, drop_off)
                d_after = dm.dist(new_pos, drop_off)
                if d_after >= 999:
                    d_after = d_before

                has_useful = any(active_needed.get(it, 0) > 0 for it in inv)

                if has_useful and d_after < d_before:
                    stats["move_to_deliver"] += 1
                elif not has_useful and d_after > d_before and d_before <= 3:
                    stats["move_clearing"] += 1
                else:
                    stats["move_to_pickup"] += 1

            # --- PICKUP ---
            elif act == "pick_up":
                stats["pickup_rounds"] += 1
                item_id = action.get("item_id")
                item_type = None
                if item_id:
                    for item in state["items"]:
                        if item["id"] == item_id:
                            item_type = item["type"]
                            break

                if item_type:
                    if active_needed.get(item_type, 0) > 0:
                        stats["pickup_active_item"] += 1
                        pickup_by_type_this_round[item_type].append(bid)
                    elif preview_needed.get(item_type, 0) > 0:
                        stats["pickup_preview_item"] += 1
                    else:
                        stats["pickup_speculative"] += 1

            # --- DROP_OFF ---
            elif act == "drop_off":
                stats["delivery_rounds"] += 1
                if active:
                    an = dict(active_needed)  # fresh copy per bot
                    for item_type in inv:
                        if an.get(item_type, 0) > 0:
                            stats["items_delivered_useful"] += 1
                            an[item_type] -= 1
                        else:
                            stats["items_at_dropoff_not_needed"] += 1
                            # Check if it matches preview
                            pn_check = dict(preview_needed)
                            if pn_check.get(item_type, 0) > 0:
                                stats["items_preview_at_dropoff"] += 1
                                pn_check[item_type] -= 1
                            else:
                                stats["items_truly_wasted"] += 1

        # --- Duplicate pickup detection ---
        for itype, bids in pickup_by_type_this_round.items():
            if len(bids) > 1:
                map_needed = items_from_map.get(itype, 0)
                if len(bids) > max(map_needed, 1):
                    excess = len(bids) - max(map_needed, 1)
                    stats["duplicate_target_rounds"] += excess
                    stats["dup_events"].append({
                        "round": rnd, "type": itype,
                        "bots": bids, "needed": map_needed, "excess": excess,
                    })

        # --- Collision detection ---
        # Record pre-positions
        pre_pos = {b["id"]: tuple(b["position"]) for b in state["bots"]}

        # Apply actions
        sim.apply_actions(actions)

        # Check post-positions
        post_state = sim.get_state()
        for b in post_state["bots"]:
            bid = b["id"]
            post = tuple(b["position"])
            act = action_map.get(bid, {}).get("action", "wait")
            if act.startswith("move_"):
                dx_map = {"move_right": (1, 0), "move_left": (-1, 0),
                          "move_down": (0, 1), "move_up": (0, -1)}
                dx, dy = dx_map[act]
                intended = (pre_pos[bid][0] + dx, pre_pos[bid][1] + dy)
                if post != intended:
                    stats["blocked_move_rounds"] += 1

    # Final order
    if current_order_id:
        stats["order_stats"].append({
            "id": current_order_id,
            "rounds_active": sim.max_rounds - order_start_round,
        })

    # =====================================================
    # PRINT RESULTS
    # =====================================================
    total = stats["total_bot_rounds"]
    print(f"\n{'='*60}")
    print(f"  HARD MODE WASTE ANALYSIS")
    print(f"  Score: {sim.score} | Orders: {sim.orders_completed} | Items: {sim.items_delivered}")
    print(f"{'='*60}")

    print(f"\n--- ACTION BREAKDOWN ({total} total bot-rounds) ---")
    for act, cnt in sorted(stats["action_counts"].items(), key=lambda x: -x[1]):
        print(f"  {act:15s}: {cnt:4d} ({100*cnt/total:.1f}%)")

    print(f"\n--- MOVEMENT ANALYSIS ({stats['move_rounds']} move rounds) ---")
    mr = max(stats["move_rounds"], 1)
    print(f"  Moving to pickup:  {stats['move_to_pickup']:4d} ({100*stats['move_to_pickup']/mr:.1f}%)")
    print(f"  Moving to deliver: {stats['move_to_deliver']:4d} ({100*stats['move_to_deliver']/mr:.1f}%)")
    print(f"  Clearing dropoff:  {stats['move_clearing']:4d} ({100*stats['move_clearing']/mr:.1f}%)")

    print(f"\n--- IDLE/WAIT ANALYSIS ({stats['idle_rounds']} wait rounds, {100*stats['idle_rounds']/total:.1f}% of total) ---")
    print(f"  Idle with work:    {stats['idle_with_work']:4d} (active order has items still needed)")
    print(f"  Idle no work:      {stats['idle_no_work']:4d} (all items covered/picked)")

    print(f"\n--- PICKUP ANALYSIS ({stats['pickup_rounds']} pickup rounds) ---")
    print(f"  Active order:      {stats['pickup_active_item']:4d}")
    print(f"  Preview order:     {stats['pickup_preview_item']:4d}")
    print(f"  Speculative:       {stats['pickup_speculative']:4d}")

    print(f"\n--- DELIVERY ANALYSIS ({stats['delivery_rounds']} drop_off actions) ---")
    total_items_at_dropoff = stats["items_delivered_useful"] + stats["items_at_dropoff_not_needed"]
    print(f"  Total items at dropoff:      {total_items_at_dropoff}")
    print(f"  Accepted by active order:    {stats['items_delivered_useful']:4d}")
    print(f"  NOT needed by active order:  {stats['items_at_dropoff_not_needed']:4d}")
    print(f"    of which preview items:    {stats['items_preview_at_dropoff']:4d} (strategic pre-positioning)")
    print(f"    of which truly wasted:     {stats['items_truly_wasted']:4d} (match no order)")

    print(f"\n--- DUPLICATE SAME-ROUND PICKUPS ---")
    print(f"  Excess bot-rounds: {stats['duplicate_target_rounds']}")
    if stats["dup_events"]:
        for ev in stats["dup_events"][:15]:
            print(f"    Round {ev['round']:3d}: {ev['type']:8s} x{ev['needed']} needed, {len(ev['bots'])} bots picking")
    else:
        print(f"  None detected (no two bots pick same active type same round when excess)")

    print(f"\n--- COLLISION/BLOCKING ---")
    print(f"  Blocked moves:     {stats['blocked_move_rounds']:4d} (move issued but bot didn't move)")
    print(f"  % of all moves:    {100*stats['blocked_move_rounds']/max(stats['move_rounds'],1):.1f}%")

    print(f"\n--- ORDER COMPLETION TIMES ---")
    for os_ in stats["order_stats"]:
        print(f"  Order {str(os_['id']):12s}: {os_['rounds_active']:3d} rounds")
    if stats["order_stats"]:
        times = [o["rounds_active"] for o in stats["order_stats"]]
        print(f"  Average: {sum(times)/len(times):.1f} rounds/order")
        print(f"  Fastest: {min(times)} | Slowest: {max(times)}")

    # ---- WASTE SUMMARY ----
    print(f"\n{'='*60}")
    print(f"  WASTE SUMMARY")
    print(f"{'='*60}")

    wasted_idle = stats["idle_with_work"]
    wasted_clearing = stats["move_clearing"]
    wasted_blocked = stats["blocked_move_rounds"]
    wasted_dup = stats["duplicate_target_rounds"]
    wasted_total = wasted_idle + wasted_clearing + wasted_blocked + wasted_dup

    print(f"  Idle when work exists:       {wasted_idle:4d} bot-rounds ({100*wasted_idle/total:.1f}%)")
    print(f"  Clearing dropoff area:       {wasted_clearing:4d} bot-rounds ({100*wasted_clearing/total:.1f}%)")
    print(f"  Blocked by collisions:       {wasted_blocked:4d} bot-rounds ({100*wasted_blocked/total:.1f}%)")
    print(f"  Duplicate pickup targets:    {wasted_dup:4d} bot-rounds")
    print(f"  Truly wasted deliveries:     {stats['items_truly_wasted']:4d} items (match no order)")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL WASTED BOT-ROUNDS:     {wasted_total:4d} / {total} ({100*wasted_total/total:.1f}%)")
    print()
    print(f"  Speculative pickups:         {stats['pickup_speculative']:4d} (not for active/preview)")
    print(f"  Preview items at dropoff:    {stats['items_preview_at_dropoff']:4d} (strategic, not waste)")

    # Theoretical analysis
    print(f"\n--- THEORETICAL EFFICIENCY ---")
    useful_rounds = stats["move_to_pickup"] + stats["move_to_deliver"] + stats["pickup_rounds"] + stats["delivery_rounds"]
    overhead = stats["move_clearing"] + stats["idle_rounds"] + stats["blocked_move_rounds"]
    print(f"  Productive rounds:  {useful_rounds:4d} ({100*useful_rounds/total:.1f}%)")
    print(f"  Overhead rounds:    {overhead:4d} ({100*overhead/total:.1f}%)")
    print(f"  Spec/preview work:  {stats['pickup_speculative']+stats['pickup_preview_item']:4d} pickups")


if __name__ == "__main__":
    analyze_hard()

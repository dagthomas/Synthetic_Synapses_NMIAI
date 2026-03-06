#!/usr/bin/env python3
"""Analyze a hard game recording JSON to identify where bots waste time.

Directly reads the recording file (no simulator needed).
"""

import json
import sys
from collections import defaultdict

RECORDING = "/Users/sandeeprc/Documents/Data/NorwegianAI/simulation/hard/game_20260305_223706.json"


def load_recording(path):
    with open(path) as f:
        return json.load(f)


def classify_action(action_str):
    if action_str == "wait":
        return "wait"
    elif action_str.startswith("move_"):
        return "move"
    elif action_str == "pick_up":
        return "pick"
    elif action_str == "drop_off":
        return "drop"
    else:
        return "other"


def inv_types(inv):
    """Extract item type strings from inventory list."""
    return [i["type"] if isinstance(i, dict) else str(i) for i in inv]


def main():
    data = load_recording(RECORDING)
    rounds = data["rounds"]
    num_rounds = len(rounds)
    num_bots = len(rounds[0]["state"]["bots"])

    print("=" * 80)
    print(f"HARD MODE RECORDING ANALYSIS: {RECORDING.split('/')[-1]}")
    print(f"Result: score={data['result']['score']}, orders={data['result']['orders_completed']}, items={data['result']['items_delivered']}")
    print(f"Rounds: {num_rounds}, Bots: {num_bots}")
    print(f"Drop-off: {rounds[0]['state']['drop_off']}")
    print("=" * 80)

    # =========================================================================
    # Build per-bot action history and position history
    # =========================================================================
    bot_actions = defaultdict(lambda: defaultdict(int))  # bot_id -> category -> count
    bot_action_list = defaultdict(list)  # bot_id -> [(round, action_str, category)]
    bot_positions = defaultdict(list)    # bot_id -> [(round, (x, y))]
    bot_inventories = defaultdict(list)  # bot_id -> [(round, [item_types])]

    for rnd_idx, rnd in enumerate(rounds):
        for bot in rnd["state"]["bots"]:
            bid = bot["id"]
            bot_positions[bid].append((rnd_idx, tuple(bot["position"])))
            bot_inventories[bid].append((rnd_idx, inv_types(bot["inventory"])))

        for act in rnd["actions"]:
            bid = act["bot"]
            action_str = act["action"]
            cat = classify_action(action_str)
            bot_actions[bid][cat] += 1
            bot_action_list[bid].append((rnd_idx, action_str, cat))

    # =========================================================================
    # SECTION 1: Per-bot action breakdown
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: PER-BOT ACTION BREAKDOWN")
    print("=" * 80)
    print(f"  {'Bot':>4} {'Move':>6} {'Pick':>6} {'Drop':>6} {'Wait':>6} {'Total':>6} {'Wait%':>6}")
    print("  " + "-" * 44)
    totals = defaultdict(int)
    for bid in sorted(bot_actions.keys()):
        counts = bot_actions[bid]
        total = sum(counts.values())
        wait_pct = counts["wait"] * 100 / total if total > 0 else 0
        print(f"  {bid:>4} {counts['move']:>6} {counts['pick']:>6} {counts['drop']:>6} {counts['wait']:>6} {total:>6} {wait_pct:>5.1f}%")
        for cat in counts:
            totals[cat] += counts[cat]
    total_all = sum(totals.values())
    print("  " + "-" * 44)
    print(f"  {'ALL':>4} {totals['move']:>6} {totals['pick']:>6} {totals['drop']:>6} {totals['wait']:>6} {total_all:>6}")
    print(f"  {'%':>4} {totals['move']*100/total_all:>5.1f}% {totals['pick']*100/total_all:>5.1f}% {totals['drop']*100/total_all:>5.1f}% {totals['wait']*100/total_all:>5.1f}%")

    # =========================================================================
    # SECTION 2: Idle stretches (consecutive waits >= 3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: IDLE STRETCHES (3+ consecutive waits per bot)")
    print("=" * 80)
    total_idle_rounds = 0
    all_stretches = []

    for bid in sorted(bot_action_list.keys()):
        actions = bot_action_list[bid]
        stretches = []
        streak_start = None
        streak_len = 0
        for rnd_idx, action_str, cat in actions:
            if cat == "wait":
                if streak_start is None:
                    streak_start = rnd_idx
                    streak_len = 1
                else:
                    streak_len += 1
            else:
                if streak_start is not None and streak_len >= 3:
                    stretches.append((streak_start, streak_start + streak_len - 1, streak_len))
                streak_start = None
                streak_len = 0
        if streak_start is not None and streak_len >= 3:
            stretches.append((streak_start, streak_start + streak_len - 1, streak_len))

        bot_idle = sum(s[2] for s in stretches)
        total_idle_rounds += bot_idle

        if stretches:
            print(f"\n  Bot {bid}: {len(stretches)} idle stretches, {bot_idle} total idle rounds")
            for start, end, length in stretches:
                pos = dict(bot_positions[bid]).get(start, "?")
                inv = dict(bot_inventories[bid]).get(start, [])
                inv_str = ", ".join(inv) if inv else "empty"
                print(f"    Rounds {start:>3}-{end:>3} ({length:>3} rounds) at pos {pos}, inv=[{inv_str}]")
                all_stretches.append((length, bid, start, end, pos, inv_str))
        else:
            print(f"\n  Bot {bid}: No idle stretches (3+ consecutive waits)")

    print(f"\n  TOTAL idle rounds (3+ consecutive waits): {total_idle_rounds} / {total_all} ({total_idle_rounds*100/total_all:.1f}%)")

    # Top 10 longest stretches
    all_stretches.sort(reverse=True)
    print(f"\n  Top 10 longest idle stretches across all bots:")
    print(f"  {'Bot':>4} {'Rounds':>12} {'Len':>5} {'Position':>12} {'Inventory'}")
    print("  " + "-" * 55)
    for length, bid, start, end, pos, inv_str in all_stretches[:10]:
        print(f"  {bid:>4} {start:>5}-{end:<5} {length:>5} {str(pos):>12} [{inv_str}]")

    # =========================================================================
    # SECTION 3: Order completion timeline
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: ORDER COMPLETION TIMELINE")
    print("=" * 80)

    order_events = []
    prev_active_id = None
    active_start = {}
    active_items = {}
    order_sequence = []

    for rnd_idx, rnd in enumerate(rounds):
        orders = rnd["state"]["orders"]
        if not orders:
            continue
        active = orders[0]
        active_id = active["id"]

        if active_id != prev_active_id:
            if prev_active_id is not None:
                order_events.append((prev_active_id, active_items[prev_active_id],
                                     active_start[prev_active_id], rnd_idx - 1))
            active_start[active_id] = rnd_idx
            active_items[active_id] = list(active["items_required"])
            order_sequence.append(active_id)
            prev_active_id = active_id

    # Handle last order
    if prev_active_id:
        last_orders = rounds[-1]["state"]["orders"]
        last_active = last_orders[0] if last_orders else None
        if last_active and last_active.get("complete"):
            order_events.append((prev_active_id, active_items[prev_active_id],
                                 active_start[prev_active_id], num_rounds - 1))
        else:
            order_events.append((prev_active_id, active_items[prev_active_id],
                                 active_start[prev_active_id], None))

    print(f"\n  {'Order':>10} {'#Items':>6} {'Start':>6} {'End':>6} {'Dur':>5} {'Status':>8}  Items Required")
    print("  " + "-" * 80)
    for order_id, items_req, start, end in order_events:
        dur = (end - start + 1) if end is not None else "---"
        if end is None:
            status = "INCOMPL"
        elif end >= num_rounds - 1 and order_id == order_events[-1][0]:
            status = "last"
        else:
            status = "done"
        end_str = str(end) if end is not None else "---"
        dur_str = str(dur) if isinstance(dur, int) else dur
        items_str = ", ".join(items_req)
        print(f"  {order_id:>10} {len(items_req):>6} {start:>6} {end_str:>6} {dur_str:>5} {status:>8}  {items_str}")

    completed = [e for e in order_events if e[3] is not None and (e[3] < num_rounds - 1 or e[0] != order_events[-1][0])]
    # Actually: an order is completed if the next order started
    completed = order_events[:-1]  # All but last (which may be incomplete)
    if completed:
        durations = [e[3] - e[2] + 1 for e in completed]
        print(f"\n  Completed orders: {len(completed)}")
        print(f"  Avg duration: {sum(durations)/len(durations):.1f} rounds")
        print(f"  Min duration: {min(durations)} rounds ({[e[0] for e in completed if e[3]-e[2]+1 == min(durations)]})")
        print(f"  Max duration: {max(durations)} rounds ({[e[0] for e in completed if e[3]-e[2]+1 == max(durations)]})")

        # Score contribution per order
        print(f"\n  Scoring: each completed order = 5 pts + items_delivered pts")
        print(f"  Average rounds per completed order: {sum(durations)/len(durations):.1f}")
        print(f"  Throughput: {len(completed)*300/sum(durations):.1f} orders per 300 rounds (if sustained)")

    # =========================================================================
    # SECTION 4: Per-order delivery details (which bots delivered for each order)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: PER-ORDER DELIVERY DETAILS (which bots delivered items)")
    print("=" * 80)

    # Build order -> round range
    order_ranges = {}
    for order_id, items_req, start, end in order_events:
        order_ranges[order_id] = (start, end if end is not None else num_rounds - 1)

    # Collect drop events
    drop_events = []
    for rnd_idx, rnd in enumerate(rounds):
        for act in rnd["actions"]:
            if act["action"] == "drop_off":
                bid = act["bot"]
                for bot in rnd["state"]["bots"]:
                    if bot["id"] == bid:
                        inv = inv_types(bot["inventory"])
                        break
                # Find which order is active
                orders = rnd["state"]["orders"]
                active_order = orders[0] if orders else None
                active_id = active_order["id"] if active_order else "?"
                needed = list(active_order["items_required"]) if active_order else []
                delivered_already = inv_types(active_order.get("items_delivered", [])) if active_order else []

                remaining = list(needed)
                for d in delivered_already:
                    if d in remaining:
                        remaining.remove(d)

                useful = []
                wasted = []
                for item in inv:
                    if item in remaining:
                        useful.append(item)
                        remaining.remove(item)
                    else:
                        wasted.append(item)

                drop_events.append((rnd_idx, bid, active_id, inv, useful, wasted))

    order_drops = defaultdict(list)
    for rnd_idx, bid, order_id, inv, useful, wasted in drop_events:
        order_drops[order_id].append((rnd_idx, bid, inv, useful, wasted))

    for order_id, items_req, start, end in order_events:
        drops = order_drops.get(order_id, [])
        print(f"\n  {order_id} (needs: {', '.join(items_req)}) [rounds {start}-{end if end else '?'}]")
        if not drops:
            print(f"    No drop events for this order!")
            continue
        bots_involved = set()
        total_useful = 0
        total_wasted = 0
        for rnd_idx, bid, inv, useful, wasted in drops:
            bots_involved.add(bid)
            total_useful += len(useful)
            total_wasted += len(wasted)
            useful_str = ", ".join(useful) if useful else "-"
            wasted_str = ", ".join(wasted) if wasted else "-"
            print(f"    Round {rnd_idx:>3}: Bot {bid} drops [{', '.join(inv)}] -> useful=[{useful_str}], wasted=[{wasted_str}]")
        print(f"    --> {len(bots_involved)} bots contributed ({sorted(bots_involved)}), {total_useful} useful items, {total_wasted} wasted drops")

    # =========================================================================
    # SECTION 5: Last 50 rounds (250-300) analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: LAST 50 ROUNDS (250-299) -- WHAT ARE BOTS DOING?")
    print("=" * 80)

    late_actions = defaultdict(lambda: defaultdict(int))
    for rnd_idx in range(250, min(300, num_rounds)):
        rnd = rounds[rnd_idx]
        for act in rnd["actions"]:
            bid = act["bot"]
            cat = classify_action(act["action"])
            late_actions[bid][cat] += 1

    print(f"\n  Action breakdown (rounds 250-299):")
    print(f"  {'Bot':>4} {'Move':>6} {'Pick':>6} {'Drop':>6} {'Wait':>6} {'Total':>6}")
    print("  " + "-" * 38)
    for bid in sorted(late_actions.keys()):
        counts = late_actions[bid]
        total = sum(counts.values())
        print(f"  {bid:>4} {counts.get('move',0):>6} {counts.get('pick',0):>6} {counts.get('drop',0):>6} {counts.get('wait',0):>6} {total:>6}")

    # Stuck/idle analysis
    print(f"\n  Per-bot stuck/idle analysis (rounds 250-299):")
    for bid in range(num_bots):
        waits_250 = []
        positions_250 = set()
        for rnd_idx in range(250, min(300, num_rounds)):
            rnd = rounds[rnd_idx]
            for act in rnd["actions"]:
                if act["bot"] == bid and act["action"] == "wait":
                    waits_250.append(rnd_idx)
            for bot in rnd["state"]["bots"]:
                if bot["id"] == bid:
                    positions_250.add(tuple(bot["position"]))

        # Find longest consecutive wait streak in this range
        streak = 0
        max_streak = 0
        max_streak_start = 0
        for rnd_idx in range(250, min(300, num_rounds)):
            is_wait = False
            for act in rounds[rnd_idx]["actions"]:
                if act["bot"] == bid and act["action"] == "wait":
                    is_wait = True
            if is_wait:
                if streak == 0:
                    streak_start_r = rnd_idx
                streak += 1
                if streak > max_streak:
                    max_streak = streak
                    max_streak_start = streak_start_r
            else:
                streak = 0

        print(f"    Bot {bid}: {len(waits_250)} waits, {len(positions_250)} unique positions", end="")
        if max_streak >= 3:
            pos_at = dict(bot_positions[bid]).get(max_streak_start, "?")
            print(f", longest idle streak={max_streak} rounds at {pos_at} (rounds {max_streak_start}-{max_streak_start+max_streak-1})")
        else:
            print(f", max idle streak={max_streak}")

    # Round-by-round detail for last 50 rounds
    print(f"\n  Round-by-round (last 50 rounds):")
    header = f"  {'Rnd':>4}"
    for b in range(num_bots):
        header += f"  {'Bot'+str(b):>12}"
    print(header)
    print("  " + "-" * (4 + 14 * num_bots))

    for rnd_idx in range(250, min(300, num_rounds)):
        rnd = rounds[rnd_idx]
        bot_info = {}
        for bot in rnd["state"]["bots"]:
            bot_info[bot["id"]] = {
                "pos": tuple(bot["position"]),
                "inv": len(bot["inventory"]),
                "inv_items": inv_types(bot["inventory"]),
            }
        for act in rnd["actions"]:
            bot_info[act["bot"]]["action"] = act["action"]

        line = f"  {rnd_idx:>4}"
        for b in range(num_bots):
            info = bot_info.get(b, {})
            pos = info.get("pos", (0, 0))
            inv = info.get("inv", 0)
            action = info.get("action", "?")
            short = action.replace("move_", "").replace("pick_up", "PICK").replace("drop_off", "DROP")
            line += f"  {pos[0]:>2},{pos[1]:<2}i{inv} {short:<5}"
        print(line)

    # =========================================================================
    # SECTION 6: Time waste breakdown per bot
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: TIME WASTE BREAKDOWN (per bot)")
    print("=" * 80)

    dropoff_pos = tuple(rounds[0]["state"]["drop_off"])

    for bid in range(num_bots):
        # First pick_up round
        first_pick = None
        for rnd_idx, action_str, cat in bot_action_list[bid]:
            if cat == "pick":
                first_pick = rnd_idx
                break

        # Last productive action (pick or drop)
        last_productive = 0
        for rnd_idx, action_str, cat in bot_action_list[bid]:
            if cat in ("pick", "drop"):
                last_productive = rnd_idx

        # Count by phase
        empty_move = 0
        loaded_move = 0
        wait_with_items = 0
        wait_empty = 0
        inv_lookup = dict(bot_inventories[bid])

        for rnd_idx, action_str, cat in bot_action_list[bid]:
            inv = inv_lookup.get(rnd_idx, [])
            has_items = len(inv) > 0
            if cat == "move":
                if has_items:
                    loaded_move += 1
                else:
                    empty_move += 1
            elif cat == "wait":
                if has_items:
                    wait_with_items += 1
                else:
                    wait_empty += 1

        startup = first_pick if first_pick is not None else num_rounds
        tail_idle = num_rounds - 1 - last_productive

        picks = bot_actions[bid]["pick"]
        drops = bot_actions[bid]["drop"]

        print(f"\n  Bot {bid}:")
        print(f"    Startup (rounds before first pick): {startup}")
        print(f"    Tail idle (rounds after last pick/drop): {tail_idle}")
        print(f"    Empty moves (no items): {empty_move}")
        print(f"    Loaded moves (carrying items): {loaded_move}")
        print(f"    Waits with items: {wait_with_items}")
        print(f"    Waits empty: {wait_empty}")
        print(f"    Picks: {picks}, Drops: {drops}")

    # =========================================================================
    # SECTION 7: Productivity analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: PRODUCTIVITY ANALYSIS")
    print("=" * 80)

    total_bot_rounds = num_bots * num_rounds
    total_moves = totals["move"]
    total_picks = totals["pick"]
    total_drops = totals["drop"]
    total_waits = totals["wait"]

    print(f"\n  Total bot-rounds available:     {total_bot_rounds}")
    print(f"  Moves:       {total_moves:>5} ({total_moves*100/total_bot_rounds:>5.1f}%)")
    print(f"  Picks:       {total_picks:>5} ({total_picks*100/total_bot_rounds:>5.1f}%)")
    print(f"  Drops:       {total_drops:>5} ({total_drops*100/total_bot_rounds:>5.1f}%)")
    print(f"  Waits:       {total_waits:>5} ({total_waits*100/total_bot_rounds:>5.1f}%)")
    print(f"  -----------------------------------------------")
    active_rounds = total_moves + total_picks + total_drops
    print(f"  Active (move+pick+drop): {active_rounds:>5} ({active_rounds*100/total_bot_rounds:>5.1f}%)")
    print(f"  Wasted (wait):           {total_waits:>5} ({total_waits*100/total_bot_rounds:>5.1f}%)")
    print(f"\n  Items delivered per 100 bot-rounds: {data['result']['items_delivered']*100/total_bot_rounds:.2f}")
    print(f"  Bot-rounds per item delivered:       {total_bot_rounds/max(data['result']['items_delivered'],1):.1f}")
    print(f"  Bot-rounds per order completed:      {total_bot_rounds/max(data['result']['orders_completed'],1):.1f}")
    print(f"  Items per drop-off action:           {data['result']['items_delivered']/max(total_drops,1):.2f}")

    # Movement direction analysis
    print(f"\n  Movement direction analysis:")
    direction_counts = defaultdict(int)
    for bid in range(num_bots):
        for rnd_idx, action_str, cat in bot_action_list[bid]:
            if cat == "move":
                direction_counts[action_str] += 1

    for d in ["move_up", "move_down", "move_left", "move_right"]:
        cnt = direction_counts.get(d, 0)
        print(f"    {d:>12}: {cnt:>5} ({cnt*100/max(total_moves,1):.1f}%)")

    # Oscillation detection
    opp = {"move_up": "move_down", "move_down": "move_up",
           "move_left": "move_right", "move_right": "move_left"}
    oscillations = 0
    for bid in range(num_bots):
        prev_move = None
        for rnd_idx, action_str, cat in bot_action_list[bid]:
            if cat == "move":
                if prev_move and prev_move in opp and opp[prev_move] == action_str:
                    oscillations += 1
                prev_move = action_str
            else:
                prev_move = None
    print(f"\n  Direction oscillations (move A then opposite): {oscillations} ({oscillations*100/max(total_moves,1):.1f}% of moves)")
    print(f"  (Likely collision avoidance / deadlock recovery)")

    # =========================================================================
    # SECTION 8: Position heatmap (most occupied cells)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: POSITION HEATMAP (where bots spend the most time)")
    print("=" * 80)

    pos_counts = defaultdict(int)
    per_bot_pos = defaultdict(lambda: defaultdict(int))
    for bid in range(num_bots):
        for rnd_idx, pos in bot_positions[bid]:
            pos_counts[pos] += 1
            per_bot_pos[bid][pos] += 1

    top_positions = sorted(pos_counts.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 most-occupied positions (all bots combined):")
    print(f"  {'Position':>12} {'Rounds':>7} {'%':>6}  Bots present")
    print("  " + "-" * 50)
    for pos, count in top_positions:
        bots_there = []
        for bid in range(num_bots):
            if per_bot_pos[bid][pos] > 5:
                bots_there.append(f"B{bid}:{per_bot_pos[bid][pos]}")
        print(f"  {str(pos):>12} {count:>7} {count*100/total_bot_rounds:>5.1f}%  {', '.join(bots_there)}")

    # =========================================================================
    # SECTION 9: Inventory utilization
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: INVENTORY UTILIZATION")
    print("=" * 80)

    inv_empty = 0
    inv_partial = 0
    inv_full = 0
    for bid in range(num_bots):
        for rnd_idx, inv in bot_inventories[bid]:
            n = len(inv)
            if n == 0:
                inv_empty += 1
            elif n >= 3:
                inv_full += 1
            else:
                inv_partial += 1

    print(f"\n  Empty inventory (0 items):   {inv_empty:>5} ({inv_empty*100/total_bot_rounds:>5.1f}%)")
    print(f"  Partial (1-2 items):         {inv_partial:>5} ({inv_partial*100/total_bot_rounds:>5.1f}%)")
    print(f"  Full (3 items):              {inv_full:>5} ({inv_full*100/total_bot_rounds:>5.1f}%)")

    # =========================================================================
    # SECTION 10: Delivery gap analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 10: DELIVERY GAP ANALYSIS")
    print("=" * 80)

    delivery_rounds = []
    for rnd_idx, rnd in enumerate(rounds):
        for act in rnd["actions"]:
            if act["action"] == "drop_off":
                delivery_rounds.append(rnd_idx)
                break  # count once per round even if multiple bots drop

    if delivery_rounds:
        print(f"\n  Total rounds with a delivery: {len(delivery_rounds)}")
        print(f"  First delivery: round {delivery_rounds[0]}")
        print(f"  Last delivery: round {delivery_rounds[-1]}")
        print(f"  Idle after last delivery: {num_rounds - 1 - delivery_rounds[-1]} rounds")

        gaps = [delivery_rounds[i + 1] - delivery_rounds[i] for i in range(len(delivery_rounds) - 1)]
        if gaps:
            print(f"\n  Gaps between deliveries:")
            print(f"    Average gap: {sum(gaps)/len(gaps):.1f} rounds")
            print(f"    Max gap: {max(gaps)} rounds", end="")
            idx = gaps.index(max(gaps))
            print(f" (between rounds {delivery_rounds[idx]} and {delivery_rounds[idx+1]})")
            print(f"    Min gap: {min(gaps)} rounds")

            # Distribution of gaps
            gap_dist = defaultdict(int)
            for g in gaps:
                bucket = g // 5 * 5  # 5-round buckets
                gap_dist[bucket] += 1
            print(f"\n    Gap distribution:")
            for bucket in sorted(gap_dist.keys()):
                print(f"      {bucket:>2}-{bucket+4:<2} rounds: {gap_dist[bucket]:>3} occurrences {'#' * gap_dist[bucket]}")

    # =========================================================================
    # SECTION 11: Score progression
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 11: SCORE PROGRESSION")
    print("=" * 80)

    print(f"\n  {'Round':>6} {'Score':>6} {'Items':>6} {'Orders':>7}")
    print("  " + "-" * 30)
    for rnd_idx in [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299]:
        if rnd_idx < num_rounds:
            s = rounds[rnd_idx]["state"]
            score = s.get("score", 0)
            # Count items delivered from orders
            orders = s["orders"]
            active = orders[0] if orders else None
            items_del = len(active.get("items_delivered", [])) if active else 0
            orders_comp = s.get("active_order_index", 0)
            print(f"  {rnd_idx:>6} {score:>6} {'-':>6} {'-':>7}")

    # Actually, let's track score from the result or derive it
    # The state has a 'score' field
    print(f"\n  Score at key rounds:")
    for rnd_idx in [0, 50, 100, 150, 200, 250, 299]:
        if rnd_idx < num_rounds:
            score = rounds[rnd_idx]["state"].get("score", 0)
            idx = rounds[rnd_idx]["state"].get("active_order_index", 0)
            print(f"    Round {rnd_idx:>3}: score={score:>4}, active_order_index={idx}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 80)

    print(f"""
  GAME RESULT: {data['result']['score']} points ({data['result']['items_delivered']} items + {data['result']['orders_completed']} orders x 5)

  TOTAL BOT-ROUNDS: {total_bot_rounds}
    Active (move+pick+drop): {active_rounds} ({active_rounds*100/total_bot_rounds:.1f}%)
    Wasted (wait):           {total_waits} ({total_waits*100/total_bot_rounds:.1f}%)

  EFFICIENCY METRICS:
    Bot-rounds per item delivered: {total_bot_rounds/max(data['result']['items_delivered'],1):.1f}
    Bot-rounds per order completed: {total_bot_rounds/max(data['result']['orders_completed'],1):.1f}
    Items per drop-off action: {data['result']['items_delivered']/max(total_drops,1):.2f}

  BIGGEST TIME SINKS:""")

    # Identify biggest wait offenders
    for bid in sorted(bot_actions.keys()):
        w = bot_actions[bid]["wait"]
        if w > 20:
            print(f"    Bot {bid}: {w} waits ({w*100/num_rounds:.0f}% of rounds)")

    print(f"\n  IDLE STRETCHES: {total_idle_rounds} total idle rounds in {len(all_stretches)} stretches")
    if all_stretches:
        print(f"    Longest: Bot {all_stretches[0][1]}, {all_stretches[0][0]} rounds (rounds {all_stretches[0][2]}-{all_stretches[0][3]})")

    # Last 50 rounds summary
    late_waits = sum(late_actions[bid].get("wait", 0) for bid in range(num_bots))
    late_total = 50 * num_bots
    print(f"\n  LAST 50 ROUNDS: {late_waits} waits out of {late_total} bot-rounds ({late_waits*100/late_total:.1f}% idle)")


if __name__ == "__main__":
    main()

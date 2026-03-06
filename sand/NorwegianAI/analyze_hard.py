#!/usr/bin/env python3
"""Detailed analysis of hard mode simulation.

Instruments the brain + simulator to collect per-round, per-bot statistics,
then prints a comprehensive summary of bottlenecks and improvement opportunities.
"""

import copy
import json
import sys
from collections import defaultdict

from simulator import load_game_data, LocalSimulator, SyntheticOrderGenerator
from brain import decide_actions, get_needed_items
from distance import DistanceMatrix
from pathfinding import reset_shelf_cache


RECORDING_PATH = "simulation/hard/game_20260304_103357.json"


def classify_action(action_dict):
    """Classify an action into a category."""
    act = action_dict.get("action", "wait")
    if act.startswith("move_"):
        return "move"
    elif act == "pick_up":
        return "pickup"
    elif act == "drop_off":
        return "dropoff"
    else:
        return "wait"


def run_instrumented_simulation():
    """Run sim with full instrumentation."""
    reset_shelf_cache()

    game_data = load_game_data(RECORDING_PATH)
    sim = LocalSimulator(game_data)

    # Pre-compute distance matrix for analysis
    state0 = sim.get_state()
    dm = DistanceMatrix(state0)

    n_bots = len(sim.bots)
    bot_ids = [b["id"] for b in sim.bots]
    drop_off = tuple(sim.drop_off)

    # ---- Per-round, per-bot tracking ----
    round_data = []           # list of dicts per round
    bot_stats = {bid: {
        "moves": 0, "pickups": 0, "dropoffs": 0, "waits": 0,
        "items_picked": 0, "items_delivered": 0,
        "blocked_moves": 0,  # wanted to move but couldn't (wait due to collision)
    } for bid in bot_ids}

    # ---- Order tracking ----
    order_log = []            # [{id, items_required, activated_round, completed_round, contributing_bots, trips}]
    current_order_id = None
    current_order_activated = None
    current_order_bots = set()
    current_order_trips = 0   # number of dropoff actions for this order
    prev_orders_completed = 0
    prev_items_delivered = 0

    # ---- Bot position tracking for congestion ----
    bot_positions_log = []    # per round: {bid: (x,y)}

    # ---- Delivery event log ----
    delivery_events = []      # (round, bot_id, items_count)

    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()

        # Track active order transitions
        active = next((o for o in state["orders"] if o.get("status") == "active"), None)
        if active and active["id"] != current_order_id:
            # New order activated
            if current_order_id is not None:
                # Close previous order
                order_log.append({
                    "id": current_order_id,
                    "activated_round": current_order_activated,
                    "completed_round": rnd - 1 if prev_orders_completed > 0 else None,
                    "contributing_bots": list(current_order_bots),
                    "trips": current_order_trips,
                })
            current_order_id = active["id"]
            current_order_activated = rnd
            current_order_bots = set()
            current_order_trips = 0

        # Get actions from the brain
        actions = decide_actions(state)
        actions_by_bot = {a["bot"]: a for a in actions}

        # Record per-bot data
        round_info = {"round": rnd, "bots": {}}
        bot_positions = {}
        for bot in state["bots"]:
            bid = bot["id"]
            pos = tuple(bot["position"])
            bot_positions[bid] = pos
            action = actions_by_bot.get(bid, {"action": "wait"})
            cat = classify_action(action)

            round_info["bots"][bid] = {
                "pos": pos,
                "action": action.get("action", "wait"),
                "category": cat,
                "inventory": list(bot["inventory"]),
                "inv_size": len(bot["inventory"]),
            }

            bot_stats[bid][f"{cat}s"] += 1
            if cat == "pickup":
                bot_stats[bid]["items_picked"] += 1
                current_order_bots.add(bid)
            elif cat == "dropoff":
                current_order_bots.add(bid)
                current_order_trips += 1

        bot_positions_log.append(bot_positions)

        # Apply actions and detect deliveries/completions
        old_items = sim.items_delivered
        old_orders = sim.orders_completed
        sim.apply_actions(actions)

        new_items = sim.items_delivered - old_items
        new_orders = sim.orders_completed - old_orders

        if new_items > 0:
            # Find which bot(s) dropped off
            for bot in state["bots"]:
                bid = bot["id"]
                action = actions_by_bot.get(bid, {"action": "wait"})
                if action.get("action") == "drop_off":
                    delivery_events.append((rnd, bid, new_items))
                    bot_stats[bid]["items_delivered"] += new_items

        if new_orders > 0:
            if order_log and order_log[-1]["completed_round"] is None:
                order_log[-1]["completed_round"] = rnd

        round_info["score"] = sim.score
        round_info["items_delivered"] = sim.items_delivered
        round_info["orders_completed"] = sim.orders_completed
        round_data.append(round_info)

        prev_orders_completed = sim.orders_completed
        prev_items_delivered = sim.items_delivered

    # Close final order
    if current_order_id is not None:
        order_log.append({
            "id": current_order_id,
            "activated_round": current_order_activated,
            "completed_round": sim.max_rounds if sim.orders_completed > prev_orders_completed else None,
            "contributing_bots": list(current_order_bots),
            "trips": current_order_trips,
        })

    return {
        "sim": sim,
        "dm": dm,
        "round_data": round_data,
        "bot_stats": bot_stats,
        "order_log": order_log,
        "bot_positions_log": bot_positions_log,
        "delivery_events": delivery_events,
        "bot_ids": bot_ids,
        "drop_off": drop_off,
        "n_bots": n_bots,
        "game_data": game_data,
    }


def analyze_order_timing(results):
    """Analyze per-order timing and identify bottlenecks."""
    order_log = results["order_log"]
    round_data = results["round_data"]

    print("\n" + "=" * 80)
    print("  ORDER TIMING ANALYSIS")
    print("=" * 80)

    # Reconstruct order info from game data
    game_data = results["game_data"]
    order_items = {}
    for od in game_data["order_sequence"]:
        order_items[od["id"]] = od["items_required"]

    completed_orders = []
    prev_completion = 0

    for entry in order_log:
        oid = entry["id"]
        activated = entry["activated_round"]
        completed = entry["completed_round"]
        items = order_items.get(oid, ["?"])
        n_items = len(items)
        bots = entry["contributing_bots"]
        trips = entry["trips"]

        if completed is not None:
            duration = completed - activated
            gap = activated - prev_completion if prev_completion > 0 else 0
            completed_orders.append({
                "id": oid, "items": items, "n_items": n_items,
                "activated": activated, "completed": completed,
                "duration": duration, "gap": gap,
                "bots": bots, "trips": trips,
            })
            prev_completion = completed
        else:
            print(f"  [INCOMPLETE] {oid}: {items} ({n_items} items) activated@{activated}, "
                  f"bots={bots}, trips={trips}")

    print(f"\n  {'Order':<12} {'Items':<5} {'Activated':>9} {'Completed':>9} {'Duration':>8} "
          f"{'Gap':>5} {'Bots':>5} {'Trips':>5}  Items")
    print("  " + "-" * 78)

    total_duration = 0
    for o in completed_orders:
        print(f"  {o['id']:<12} {o['n_items']:<5} {o['activated']:>9} {o['completed']:>9} "
              f"{o['duration']:>8} {o['gap']:>5} {len(o['bots']):>5} {o['trips']:>5}  "
              f"{', '.join(o['items'])}")
        total_duration += o["duration"]

    n_completed = len(completed_orders)
    if n_completed:
        avg_duration = total_duration / n_completed
        print(f"\n  Average order duration: {avg_duration:.1f} rounds")
        print(f"  Orders completed: {n_completed}")

    # Longest gaps
    if completed_orders:
        by_duration = sorted(completed_orders, key=lambda x: x["duration"], reverse=True)
        print(f"\n  Top 5 SLOWEST orders:")
        for o in by_duration[:5]:
            print(f"    {o['id']}: {o['duration']} rounds ({o['n_items']} items: {', '.join(o['items'])})")


def analyze_bot_efficiency(results):
    """Per-bot action breakdown and efficiency metrics."""
    bot_stats = results["bot_stats"]
    bot_ids = results["bot_ids"]
    n_bots = results["n_bots"]
    round_data = results["round_data"]
    max_rounds = 300

    print("\n" + "=" * 80)
    print("  PER-BOT EFFICIENCY")
    print("=" * 80)

    print(f"\n  {'Bot':<6} {'Moves':>7} {'Pickups':>8} {'Dropoffs':>9} {'Waits':>7} "
          f"{'Items Picked':>13} {'Items Delivered':>15} {'Wait%':>7}")
    print("  " + "-" * 78)

    total_waits = 0
    total_moves = 0
    total_pickups = 0
    total_dropoffs = 0

    for bid in bot_ids:
        s = bot_stats[bid]
        total = s["moves"] + s["pickups"] + s["dropoffs"] + s["waits"]
        wait_pct = 100.0 * s["waits"] / total if total > 0 else 0
        print(f"  {bid:<6} {s['moves']:>7} {s['pickups']:>8} {s['dropoffs']:>9} {s['waits']:>7} "
              f"{s['items_picked']:>13} {s['items_delivered']:>15} {wait_pct:>6.1f}%")
        total_waits += s["waits"]
        total_moves += s["moves"]
        total_pickups += s["pickups"]
        total_dropoffs += s["dropoffs"]

    total_bot_rounds = max_rounds * n_bots
    print(f"\n  Total bot-rounds: {total_bot_rounds}")
    print(f"  Moves:    {total_moves:>5} ({100*total_moves/total_bot_rounds:.1f}%)")
    print(f"  Pickups:  {total_pickups:>5} ({100*total_pickups/total_bot_rounds:.1f}%)")
    print(f"  Dropoffs: {total_dropoffs:>5} ({100*total_dropoffs/total_bot_rounds:.1f}%)")
    print(f"  Waits:    {total_waits:>5} ({100*total_waits/total_bot_rounds:.1f}%)")


def analyze_wait_patterns(results):
    """Identify when and why bots wait."""
    round_data = results["round_data"]
    bot_ids = results["bot_ids"]

    print("\n" + "=" * 80)
    print("  WAIT PATTERN ANALYSIS")
    print("=" * 80)

    # Find stretches of consecutive waits per bot
    wait_streaks = {bid: [] for bid in bot_ids}
    current_streak = {bid: 0 for bid in bot_ids}
    streak_start = {bid: None for bid in bot_ids}

    for rd in round_data:
        rnd = rd["round"]
        for bid in bot_ids:
            if rd["bots"][bid]["category"] == "wait":
                if current_streak[bid] == 0:
                    streak_start[bid] = rnd
                current_streak[bid] += 1
            else:
                if current_streak[bid] > 0:
                    wait_streaks[bid].append({
                        "start": streak_start[bid],
                        "length": current_streak[bid],
                        "pos": rd["bots"][bid]["pos"],
                        "inv": rd["bots"][bid]["inventory"],
                    })
                current_streak[bid] = 0

    # Close any open streaks
    for bid in bot_ids:
        if current_streak[bid] > 0:
            last_rd = round_data[-1]
            wait_streaks[bid].append({
                "start": streak_start[bid],
                "length": current_streak[bid],
                "pos": last_rd["bots"][bid]["pos"],
                "inv": last_rd["bots"][bid]["inventory"],
            })

    # Show longest wait streaks
    all_streaks = []
    for bid in bot_ids:
        for ws in wait_streaks[bid]:
            all_streaks.append((ws["length"], bid, ws["start"], ws["pos"], ws["inv"]))

    all_streaks.sort(reverse=True)
    print(f"\n  Top 20 longest wait streaks:")
    print(f"  {'Bot':<6} {'Start':>6} {'Length':>7} {'Position':<12} {'Inventory'}")
    print("  " + "-" * 60)
    for length, bid, start, pos, inv in all_streaks[:20]:
        print(f"  {bid:<6} {start:>6} {length:>7} {str(pos):<12} {inv}")

    # Count rounds where ALL bots are waiting
    all_wait_rounds = 0
    for rd in round_data:
        if all(rd["bots"][bid]["category"] == "wait" for bid in bot_ids):
            all_wait_rounds += 1
    print(f"\n  Rounds where ALL bots waited: {all_wait_rounds}")

    # Count rounds where >= 3 bots wait
    many_wait_rounds = 0
    for rd in round_data:
        n_waiting = sum(1 for bid in bot_ids if rd["bots"][bid]["category"] == "wait")
        if n_waiting >= 3:
            many_wait_rounds += 1
    print(f"  Rounds where >= 3 bots waited: {many_wait_rounds}")


def analyze_congestion(results):
    """Analyze where congestion happens (bots blocking each other)."""
    round_data = results["round_data"]
    bot_ids = results["bot_ids"]
    drop_off = results["drop_off"]
    bot_positions_log = results["bot_positions_log"]

    print("\n" + "=" * 80)
    print("  CONGESTION ANALYSIS")
    print("=" * 80)

    # Count how often bots are near the dropoff
    dropoff_proximity = defaultdict(int)  # distance -> count
    dropoff_crowding = defaultdict(int)   # round -> n_bots within dist 3

    for rnd_idx, positions in enumerate(bot_positions_log):
        near_count = 0
        for bid, pos in positions.items():
            dist = abs(pos[0] - drop_off[0]) + abs(pos[1] - drop_off[1])
            if dist <= 3:
                near_count += 1
        if near_count >= 2:
            dropoff_crowding[near_count] += 1

    print(f"\n  Rounds with N bots within 3 Manhattan distance of dropoff:")
    for n in sorted(dropoff_crowding.keys()):
        print(f"    {n} bots nearby: {dropoff_crowding[n]} rounds")

    # Count bots on same row as dropoff (y=12)
    same_row = 0
    for rnd_idx, positions in enumerate(bot_positions_log):
        n_on_row = sum(1 for pos in positions.values() if pos[1] == drop_off[1])
        if n_on_row >= 2:
            same_row += 1
    print(f"  Rounds with >= 2 bots on dropoff row (y={drop_off[1]}): {same_row}")


def analyze_travel_distances(results):
    """Analyze how far bots travel for items."""
    dm = results["dm"]
    drop_off = results["drop_off"]
    game_data = results["game_data"]
    items = game_data["items"]

    print("\n" + "=" * 80)
    print("  SHELF DISTANCE ANALYSIS (from drop-off)")
    print("=" * 80)

    # Compute trip cost for each item type
    type_costs = defaultdict(list)
    for item in items:
        itype = item["type"]
        shelf = (item["position"][0], item["position"][1])
        cost = dm.trip_cost(drop_off, shelf)
        type_costs[itype].append((cost, shelf))

    print(f"\n  {'Item Type':<12} {'Min Cost':>8} {'Max Cost':>8} {'Avg Cost':>8} {'Shelves':>7}")
    print("  " + "-" * 50)
    for itype in sorted(type_costs.keys()):
        costs = [c for c, _ in type_costs[itype]]
        print(f"  {itype:<12} {min(costs):>8} {max(costs):>8} {sum(costs)/len(costs):>8.1f} {len(costs):>7}")

    # All bots start at (20,12) — how far from dropoff?
    start_to_dropoff = dm.dist((20, 12), drop_off)
    print(f"\n  Bot start position (20,12) to dropoff (1,12): {start_to_dropoff} steps")


def analyze_score_trajectory(results):
    """Score over time."""
    round_data = results["round_data"]

    print("\n" + "=" * 80)
    print("  SCORE TRAJECTORY")
    print("=" * 80)

    # Score at key intervals
    print(f"\n  {'Round':>6} {'Score':>6} {'Items':>6} {'Orders':>7}")
    print("  " + "-" * 30)
    for rnd in [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299]:
        if rnd < len(round_data):
            rd = round_data[rnd]
            print(f"  {rnd:>6} {rd['score']:>6} {rd['items_delivered']:>6} {rd['orders_completed']:>7}")


def analyze_theoretical_throughput(results):
    """Estimate theoretical max throughput for this grid."""
    dm = results["dm"]
    drop_off = results["drop_off"]
    game_data = results["game_data"]
    items = game_data["items"]
    n_bots = results["n_bots"]

    print("\n" + "=" * 80)
    print("  THEORETICAL THROUGHPUT ANALYSIS")
    print("=" * 80)

    # Average trip cost across all item types
    type_min_costs = {}
    for item in items:
        itype = item["type"]
        shelf = (item["position"][0], item["position"][1])
        cost = dm.trip_cost(drop_off, shelf)
        if itype not in type_min_costs or cost < type_min_costs[itype]:
            type_min_costs[itype] = cost

    avg_min_trip = sum(type_min_costs.values()) / len(type_min_costs)
    print(f"\n  Average min trip cost per item type: {avg_min_trip:.1f}")

    # For a 3-item batch picked from nearby shelves:
    # Estimate ~12-18 rounds per 3-item trip (travel + 3 pickups + return + dropoff)
    # With 5 bots, theoretically could complete ~5 * 300 / avg_trip_cost items
    print(f"\n  Rough theoretical estimates (5 bots, 300 rounds):")

    for trip_cost in [10, 15, 20, 25]:
        items_per_trip = 3
        trips = (300 * n_bots) / trip_cost
        total_items = trips * items_per_trip
        # Orders average ~4 items for hard
        orders = total_items / 4
        score = total_items + 5 * orders
        print(f"    If avg trip cost = {trip_cost}: ~{total_items:.0f} items, ~{orders:.0f} orders, "
              f"score ~{score:.0f}")

    # Current utilization
    sim = results["sim"]
    print(f"\n  Actual: {sim.items_delivered} items, {sim.orders_completed} orders, "
          f"score {sim.score}")
    print(f"  Bot-rounds used: {300 * n_bots}")
    print(f"  Rounds per item delivered: {300 * n_bots / max(sim.items_delivered, 1):.1f}")


def analyze_bot_movement_heatmap(results):
    """Show where bots spend their time."""
    bot_positions_log = results["bot_positions_log"]
    bot_ids = results["bot_ids"]

    print("\n" + "=" * 80)
    print("  BOT POSITION HEATMAP (where bots spend time)")
    print("=" * 80)

    # Count time at each position, per bot and total
    pos_counts = defaultdict(int)
    per_bot_pos = {bid: defaultdict(int) for bid in bot_ids}

    for positions in bot_positions_log:
        for bid, pos in positions.items():
            pos_counts[pos] += 1
            per_bot_pos[bid][pos] += 1

    # Top positions overall
    top_positions = sorted(pos_counts.items(), key=lambda x: -x[1])[:15]
    print(f"\n  Top 15 most-occupied positions (all bots combined):")
    print(f"  {'Position':<12} {'Rounds':>7} {'% of 1500':>10}")
    print("  " + "-" * 32)
    for pos, count in top_positions:
        print(f"  {str(pos):<12} {count:>7} {100*count/1500:>9.1f}%")

    # Per-bot dominant positions
    print(f"\n  Per-bot top 3 positions:")
    for bid in bot_ids:
        tops = sorted(per_bot_pos[bid].items(), key=lambda x: -x[1])[:3]
        pos_str = ", ".join(f"{pos}:{count}" for pos, count in tops)
        print(f"    Bot {bid}: {pos_str}")


def analyze_round_by_round_detail(results):
    """Print first N rounds in detail."""
    round_data = results["round_data"]
    bot_ids = results["bot_ids"]

    print("\n" + "=" * 80)
    print("  FIRST 60 ROUNDS DETAIL (all bots)")
    print("=" * 80)

    print(f"\n  {'Rnd':>4}", end="")
    for bid in bot_ids:
        print(f"  Bot{bid:<3} {'Pos':<8} {'Inv':>3}", end="")
    print(f"  {'Score':>5} {'Items':>5} {'Orders':>6}")
    print("  " + "-" * 110)

    for rd in round_data[:60]:
        rnd = rd["round"]
        line = f"  {rnd:>4}"
        for bid in bot_ids:
            bd = rd["bots"][bid]
            act_short = {
                "move_up": "U", "move_down": "D", "move_left": "L", "move_right": "R",
                "pick_up": "P", "drop_off": "X", "wait": "W"
            }.get(bd["action"], "?")
            line += f"  {act_short:<6} {str(bd['pos']):<8} {bd['inv_size']:>3}"
        line += f"  {rd['score']:>5} {rd['items_delivered']:>5} {rd['orders_completed']:>6}"
        print(line)


def analyze_idle_inventory(results):
    """Analyze bots carrying items that aren't useful for the current order."""
    round_data = results["round_data"]
    bot_ids = results["bot_ids"]

    print("\n" + "=" * 80)
    print("  INVENTORY UTILIZATION")
    print("=" * 80)

    total_bot_rounds = len(round_data) * len(bot_ids)
    full_inv = 0
    empty_inv = 0
    partial_inv = 0

    for rd in round_data:
        for bid in bot_ids:
            inv_size = rd["bots"][bid]["inv_size"]
            if inv_size == 0:
                empty_inv += 1
            elif inv_size >= 3:
                full_inv += 1
            else:
                partial_inv += 1

    print(f"\n  Empty inventory (0 items):   {empty_inv:>5} ({100*empty_inv/total_bot_rounds:.1f}%)")
    print(f"  Partial inventory (1-2):     {partial_inv:>5} ({100*partial_inv/total_bot_rounds:.1f}%)")
    print(f"  Full inventory (3 items):    {full_inv:>5} ({100*full_inv/total_bot_rounds:.1f}%)")


def analyze_delivery_gaps(results):
    """Find gaps between deliveries."""
    delivery_events = results["delivery_events"]

    print("\n" + "=" * 80)
    print("  DELIVERY GAP ANALYSIS")
    print("=" * 80)

    if not delivery_events:
        print("  No delivery events!")
        return

    print(f"\n  Total delivery events: {len(delivery_events)}")
    print(f"\n  {'Round':>6} {'Bot':>4} {'Items':>6}")
    print("  " + "-" * 20)
    for rnd, bid, n in delivery_events:
        print(f"  {rnd:>6} {bid:>4} {n:>6}")

    # Gaps between consecutive deliveries
    rounds = [e[0] for e in delivery_events]
    gaps = [rounds[i+1] - rounds[i] for i in range(len(rounds)-1)]
    if gaps:
        print(f"\n  Gap between deliveries:")
        print(f"    Average: {sum(gaps)/len(gaps):.1f} rounds")
        print(f"    Max gap: {max(gaps)} rounds (between rounds {rounds[gaps.index(max(gaps))]} "
              f"and {rounds[gaps.index(max(gaps))+1]})")
        print(f"    Min gap: {min(gaps)} rounds")

    # First delivery
    print(f"\n  First delivery at round {rounds[0]}")
    print(f"  Last delivery at round {rounds[-1]}")
    print(f"  Idle rounds after last delivery: {299 - rounds[-1]}")


def analyze_shelf_usage(results):
    """Analyze which shelves are actually used vs optimal choices."""
    dm = results["dm"]
    drop_off = results["drop_off"]
    game_data = results["game_data"]
    items = game_data["items"]

    print("\n" + "=" * 80)
    print("  SHELF USAGE: CHEAPEST vs ACTUAL")
    print("=" * 80)

    # Show cheapest shelf per type and second cheapest
    print(f"\n  {'Type':<10} {'Cheapest':<12} {'Cost':>5}  {'2nd Best':<12} {'Cost':>5}  {'Saving':>6}")
    print("  " + "-" * 60)
    for itype in sorted(set(it["type"] for it in items)):
        type_items = [(dm.trip_cost(drop_off, tuple(it["position"])), tuple(it["position"]))
                      for it in items if it["type"] == itype]
        type_items.sort()
        best = type_items[0]
        second = type_items[1] if len(type_items) > 1 else (999, None)
        saving = second[0] - best[0]
        print(f"  {itype:<10} {str(best[1]):<12} {best[0]:>5}  {str(second[1]):<12} {second[0]:>5}  {saving:>6}")

    # Show aisle breakdown
    print(f"\n  Aisle distances to dropoff (walkable cells):")
    for ax in [4, 8, 12, 16]:
        for section, y in [("lower", 11), ("mid", 7), ("upper", 1)]:
            d = dm.dist((ax, y), drop_off)
            print(f"    Aisle {ax} {section:>5} (y={y}): {d:>3} steps to dropoff")


def analyze_order_vs_theoretical(results):
    """Compare actual order times vs theoretical minimums."""
    dm = results["dm"]
    drop_off = results["drop_off"]
    game_data = results["game_data"]
    items = game_data["items"]
    order_log = results["order_log"]

    print("\n" + "=" * 80)
    print("  ORDER EFFICIENCY: ACTUAL vs THEORETICAL MINIMUM")
    print("=" * 80)

    order_items_map = {}
    for od in game_data["order_sequence"]:
        order_items_map[od["id"]] = od["items_required"]

    print(f"\n  {'Order':<10} {'Actual':>6} {'MinMake':>7} {'Delta':>6} {'Items':>5}  Types")
    print("  " + "-" * 65)

    for entry in order_log:
        oid = entry["id"]
        activated = entry["activated_round"]
        completed = entry["completed_round"]
        if completed is None:
            continue
        actual = completed - activated
        req_items = order_items_map.get(oid, [])

        # Compute theoretical min makespan with 5 bots
        item_costs = []
        for itype in req_items:
            best = float("inf")
            for it in items:
                if it["type"] == itype:
                    cost = dm.trip_cost(drop_off, tuple(it["position"]))
                    if cost < best:
                        best = cost
            item_costs.append(best)

        sorted_costs = sorted(item_costs, reverse=True)
        theoretical_min = sorted_costs[0] if len(req_items) <= 5 else sorted_costs[0]

        delta = actual - theoretical_min
        sign = "+" if delta >= 0 else ""
        print(f"  {oid:<10} {actual:>6} {theoretical_min:>7} {sign}{delta:>5} {len(req_items):>5}  "
              f"{', '.join(req_items)}")

    print(f"\n  Negative delta = preview pre-picking saved time (items already in inventory)")
    print(f"  Positive delta = overhead from travel, congestion, or bot coordination")


def main():
    print("=" * 80)
    print("  HARD MODE SIMULATION ANALYSIS")
    print(f"  Recording: {RECORDING_PATH}")
    print("=" * 80)

    results = run_instrumented_simulation()

    sim = results["sim"]
    print(f"\n  FINAL SCORE: {sim.score}")
    print(f"  Items delivered: {sim.items_delivered}")
    print(f"  Orders completed: {sim.orders_completed}")
    print(f"  Grid: {sim.width}x{sim.height}, {results['n_bots']} bots")
    print(f"  Drop-off: {results['drop_off']}")

    analyze_score_trajectory(results)
    analyze_bot_efficiency(results)
    analyze_wait_patterns(results)
    analyze_congestion(results)
    analyze_travel_distances(results)
    analyze_theoretical_throughput(results)
    analyze_bot_movement_heatmap(results)
    analyze_idle_inventory(results)
    analyze_delivery_gaps(results)
    analyze_order_timing(results)
    analyze_round_by_round_detail(results)

    analyze_shelf_usage(results)
    analyze_order_vs_theoretical(results)

    # Final summary
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE IMPROVEMENT ANALYSIS")
    print("=" * 80)

    bot_stats = results["bot_stats"]
    total_waits = sum(s["waits"] for s in bot_stats.values())
    total_rounds = 300 * results["n_bots"]

    print(f"""
  CURRENT PERFORMANCE
  ===================
  Score:           {sim.score} ({sim.items_delivered} items + {sim.orders_completed} orders x 5)
  Orders completed: {sim.orders_completed} of 15+ available
  Avg order time:  21.2 rounds
  Bot-rounds:      {total_rounds} (waits: {total_waits} = {100*total_waits/total_rounds:.1f}%)
  Rounds per item: {total_rounds / max(sim.items_delivered, 1):.1f}
  Items per trip:  {sim.items_delivered / max(sum(s["dropoffs"] for s in bot_stats.values()), 1):.1f}

  GRID CHARACTERISTICS
  ====================
  22x14 grid, 4 aisles (x=4,8,12,16), 64 shelves, 12 item types
  Drop-off at (1,12) -- bottom-left corner
  Bot start at (20,12) -- bottom-right corner, 19 steps from dropoff!
  Lower shelves (y=8-10) are 7-23 cost from dropoff (CHEAP)
  Upper shelves (y=2-6) are 25-51 cost from dropoff (EXPENSIVE)
  Bread (cost 27) and milk (cost 27) dominate as bottleneck items

  BOTTLENECK #1: STARTUP PENALTY (cost: ~40 bot-rounds x 5 = 200 bot-rounds)
  ============================================================================
  All 5 bots start stacked at (20,12). They must traverse 19+ cells before
  reaching any useful shelf. First delivery happens at round 41 (zero score
  for first 41 rounds). That is 205 bot-rounds wasted = 13.7% of total.

  FIX: Route bots through different aisles on the way to dropoff area,
  picking items en route. Bot starting near aisle 16 (x=16) should pick
  preview items there, not travel all the way to aisle 4 empty-handed.

  BOTTLENECK #2: MASSIVE IDLE TIME FOR BOT 4 (83 waits = 27.7%)
  ==============================================================
  Bot 4 spends 79 of 300 rounds parked at (4,1) doing nothing. Bot 2
  spends 42 rounds at (12,1). These are their "home aisle" park positions.
  Combined: 121 bot-rounds = 8.1% of total completely wasted.

  FIX: Idle bots should always be pre-picking preview items or positioning
  for the next order. Never park at y=1 when useful work exists.

  BOTTLENECK #3: LOW ITEMS PER DROPOFF TRIP ({sim.items_delivered / max(sum(s["dropoffs"] for s in bot_stats.values()), 1):.1f} avg)
  ==============================================================
  32 dropoff events for 51 items = 1.6 items/trip. Each return trip costs
  4-16 rounds of travel. With shelf revisiting, should get 2.5-3 items/trip.

  FIX: Batch items more aggressively. When a bot is near cheap shelves
  (lower section), pick up 3 items before returning. Use shelf revisiting
  (pick same type multiple times from one shelf with zero movement cost).

  BOTTLENECK #4: EXPENSIVE ITEMS (bread=27, milk=27, cereal=25)
  =============================================================
  8 of 15 orders contain bread or milk. Each round-trip costs 27 rounds
  for a SINGLE item. These items only exist in the upper grid section.
  With 5 bots, 1-2 should be dedicated "far-item runners."

  FIX: Pre-dispatch bots to upper shelves for bread/milk/cereal as soon
  as the preview order is visible. When a bot finishes cheap items, send
  it to pre-pick expensive items for the next order.

  BOTTLENECK #5: AISLE CONVERGENCE AND CONGESTION
  ================================================
  The heatmap shows (4,1) as the #1 most-occupied position at 94 rounds.
  All bots converge on aisle 4 because it has the cheapest shelves (lower
  section: cream=7, oats=11, cheese=13). This causes congestion and waits.
  Aisles 8 and 12 have equally cheap items but are underused.

  FIX: Distribute bots across aisles 4, 8, and 12 for lower-section items.
  Use yogurt(7,10) cost=15, eggs(9,10) cost=19 from aisle 8 instead of
  always sending all bots to aisle 4.

  BOTTLENECK #6: PREVIEW PICKING NOT AGGRESSIVE ENOUGH
  =====================================================
  Many orders complete in LESS time than their theoretical min (negative
  overhead), proving preview picking works. But Bot 4 and Bot 2 still
  idle for 131 rounds combined while they could be pre-picking.

  FIX: When preview order is visible, immediately assign idle bots to pick
  its items. Even expensive items (bread, milk) should be pre-picked if
  the bot is otherwise idle.

  ESTIMATED IMPROVEMENT POTENTIAL
  ================================
  Fix startup (-10 rounds on order 0):                  +8  points
  Reduce Bot 4/2 idle time (83+48=131 -> ~30 waits):   +15 points
  Increase items/trip (1.6 -> 2.5):                     +20 points
  Aggressive preview pre-picking:                        +15 points
  Better aisle distribution:                             +10 points
  Total estimated improvement:                           +68 points

  Projected score: ~180-185 (from 116)
  Theoretical max with perfect play: ~200-225
""")


if __name__ == "__main__":
    main()

"""Analyze a nightmare grocery-bot replay JSON for performance issues.

Scoring model: score = items_delivered + (orders_completed * 5)
Each item delivered to drop-off = +1 point.
Completing an entire order gives +5 bonus.
"""

import json
import sys
from collections import defaultdict


def load_replay(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(data: dict) -> None:
    run = data["run"]
    rounds = data["rounds"]
    total_rounds = len(rounds)
    final_round = rounds[-1]["round_number"]
    final_score = rounds[-1]["score"]
    num_bots = run["bot_count"]

    print("=" * 80)
    print(f"  NIGHTMARE REPLAY ANALYSE - Seed {run['seed']}")
    print(f"  {run['grid_width']}x{run['grid_height']} grid, {num_bots} bots, "
          f"{run['item_types']} item-typer, {total_rounds} runder")
    print(f"  Sluttpoeng: {final_score}  |  Items levert: {run['items_delivered']}  "
          f"|  Ordrer fullfort: {run['orders_completed']}")
    print(f"  Scoring: {run['items_delivered']} items + {run['orders_completed']}*5 bonus "
          f"= {run['items_delivered'] + run['orders_completed']*5}")
    print("=" * 80)

    # Build round lookup by round_number
    round_by_num: dict[int, dict] = {}
    for rd in rounds:
        round_by_num[rd["round_number"]] = rd

    # Build action lookup: actions are in rd["actions"] as list of {bot, action}
    def get_bot_action(rd: dict, bot_id: int) -> str:
        for a in rd.get("actions", []):
            if a["bot"] == bot_id:
                return a["action"]
        return "unknown"

    def get_action_counts(rd: dict) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for a in rd.get("actions", []):
            act = a["action"]
            if act == "wait":
                counts["wait"] += 1
            elif act.startswith("move"):
                counts["move"] += 1
            elif act.startswith("pick"):
                counts["pick"] += 1
            elif act.startswith("drop"):
                counts["drop"] += 1
            else:
                counts["other"] += 1
        return counts

    # =========================================================================
    # 1. SCORE VED HVER 50. RUNDE
    # =========================================================================
    print("\n" + "=" * 80)
    print("  1. SCORE-UTVIKLING (hver 50. runde)")
    print("=" * 80)

    score_at: dict[int, int] = {}
    for rd in rounds:
        score_at[rd["round_number"]] = rd["score"]

    prev_score = 0
    for milestone in range(50, final_round + 1, 50):
        rn = milestone if milestone in score_at else max(r for r in score_at if r <= milestone)
        s = score_at[rn]
        delta = s - prev_score
        bar = "#" * (delta // 2) if delta > 0 else "-"
        print(f"  Runde {milestone:>4}: score {s:>4}  (delta +{delta:>3})  {bar}")
        prev_score = s

    print(f"\n  Runde {final_round:>4}: score {final_score:>4}  (SLUTT)")

    # =========================================================================
    # 2. ORDER TRACKING
    # =========================================================================
    print("\n" + "=" * 80)
    print("  2. ORDRE-ANALYSE")
    print("=" * 80)

    # Track order lifecycle
    all_orders: dict[str, dict] = {}
    for rd in rounds:
        rn = rd["round_number"]
        for o in rd["orders"]:
            oid = o["id"]
            if oid not in all_orders:
                all_orders[oid] = {
                    "first_seen": rn,
                    "first_status": o["status"],
                    "items_required": list(o["items_required"]),
                    "n_required": len(o["items_required"]),
                }
            all_orders[oid]["last_seen"] = rn
            all_orders[oid]["last_status"] = o["status"]
            all_orders[oid]["items_delivered"] = list(o.get("items_delivered", []))

    # Determine active_from (when status became "active") and completion
    for oid, info in all_orders.items():
        # Find first round where status = "active"
        info["active_from"] = None
        for rd in rounds:
            for o in rd["orders"]:
                if o["id"] == oid and o["status"] == "active":
                    info["active_from"] = rd["round_number"]
                    break
            if info["active_from"] is not None:
                break

        # Completed = disappeared before game end
        info["completed"] = info["last_seen"] < final_round
        if info["completed"] and info["active_from"] is not None:
            info["duration"] = info["last_seen"] - info["active_from"] + 1
            info["completed_round"] = info["last_seen"] + 1
        else:
            info["duration"] = (final_round - info["active_from"] + 1) if info["active_from"] is not None else 0

    completed_orders = {k: v for k, v in all_orders.items() if v["completed"]}
    uncompleted = {k: v for k, v in all_orders.items() if not v["completed"]}

    if completed_orders:
        durations = [v["duration"] for v in completed_orders.values()]
        avg_dur = sum(durations) / len(durations)
        max_dur = max(durations)
        min_dur = min(durations)
    else:
        avg_dur = max_dur = min_dur = 0

    stalled = {k: v for k, v in completed_orders.items() if v["duration"] > 40}

    print(f"\n  Totalt ordrer sett:           {len(all_orders)}")
    print(f"  Fullforte ordrer:             {len(completed_orders)}")
    print(f"  Ufullforte ordrer:            {len(uncompleted)}")
    print(f"  Gjennomsnittlig varighet:     {avg_dur:.1f} runder")
    print(f"  Raskeste ordre:               {min_dur} runder")
    print(f"  Tregeste ordre:               {max_dur} runder")
    print(f"  Ordrer som stalte (>40 rdr):  {len(stalled)}")

    print(f"\n  --- ALLE ORDRER ---")
    print(f"  {'Ordre':>10} | {'Aktiv fra':>10} | {'Ferdig':>7} | {'Varighet':>9} | {'Items':>6} | {'Status':>10} | {'Items needed'}")
    print("  " + "-" * 95)
    for oid in sorted(all_orders.keys(), key=lambda x: int(x.split("_")[1])):
        info = all_orders[oid]
        af = info["active_from"] if info["active_from"] is not None else "-"
        cr = info.get("completed_round", "-")
        dur = info["duration"]
        n_req = info["n_required"]
        n_del = len(info["items_delivered"])
        status = "FULLFORT" if info["completed"] else ("UFERDIG" if info["active_from"] else "PREVIEW")
        marker = " *** STALL" if oid in stalled else ""
        print(f"  {oid:>10} | {str(af):>10} | {str(cr):>7} | {dur:>6} rdr | {n_req:>6} | {status:>10} | "
              f"{info['items_required']}{marker}")

    # =========================================================================
    # 3. ITEM-TRACKING FOR TREGESTE ORDRER
    # =========================================================================
    print("\n" + "=" * 80)
    print("  3. DETALJERT ITEM-TRACKING FOR TREGESTE ORDRER")
    print("=" * 80)

    # Sort by duration, show top 8
    sorted_orders = sorted(all_orders.items(),
                           key=lambda x: x[1]["duration"], reverse=True)

    for oid, info in sorted_orders[:8]:
        if info["active_from"] is None:
            continue
        print(f"\n  {oid} ({info['duration']} runder, {info['n_required']} items): "
              f"{info['items_required']}")

        # Track deliveries round by round
        prev_delivered = []
        for rd in rounds:
            rn = rd["round_number"]
            for o in rd["orders"]:
                if o["id"] == oid and o["status"] == "active":
                    cur_del = o.get("items_delivered", [])
                    if len(cur_del) > len(prev_delivered):
                        # Find new items
                        new_items = list(cur_del)
                        for item in prev_delivered:
                            if item in new_items:
                                new_items.remove(item)
                        remaining = info["n_required"] - len(cur_del)
                        rounds_since_active = rn - info["active_from"]
                        print(f"    Runde {rn:>3} (+{rounds_since_active:>3}): levert {new_items}, "
                              f"{remaining} gjenstar")
                        prev_delivered = list(cur_del)

        if info["completed"]:
            print(f"    Runde {info['completed_round']:>3}: ORDRE FULLFORT (+5 bonus)")
        else:
            remaining_items = list(info["items_required"])
            for item in info["items_delivered"]:
                if item in remaining_items:
                    remaining_items.remove(item)
            print(f"    UFERDIG: mangler {remaining_items}")

    # =========================================================================
    # 4. INVENTORY WASTE
    # =========================================================================
    print("\n" + "=" * 80)
    print("  4. INVENTORY WASTE (bots med unyttige items)")
    print("=" * 80)

    print(f"\n  {'Runde':>6} | {'Bots m/inv':>11} | {'Waste bots':>11} | "
          f"{'Waste items':>12} | {'Total carried':>14} | {'Waste %':>8}")
    print("  " + "-" * 80)

    for milestone in list(range(50, final_round + 1, 50)) + [final_round]:
        rn = milestone if milestone in round_by_num else max(r for r in round_by_num if r <= milestone)
        rd = round_by_num[rn]

        # Collect all needed items across active orders
        needed_items: list[str] = []
        for order in rd["orders"]:
            if order["status"] == "active":
                required = list(order["items_required"])
                delivered = list(order.get("items_delivered", []))
                for item in delivered:
                    if item in required:
                        required.remove(item)
                needed_items.extend(required)

        needed_counter: dict[str, int] = defaultdict(int)
        for item in needed_items:
            needed_counter[item] += 1

        bots_with_inv = 0
        waste_bots = 0
        waste_items_count = 0
        total_carried = 0

        for bot in rd["bots"]:
            inv = bot.get("inventory", [])
            if inv:
                bots_with_inv += 1
                total_carried += len(inv)
                remaining_needed = dict(needed_counter)
                useful = 0
                for item in inv:
                    if remaining_needed.get(item, 0) > 0:
                        remaining_needed[item] -= 1
                        useful += 1
                wasted = len(inv) - useful
                if wasted > 0:
                    waste_bots += 1
                    waste_items_count += wasted

        waste_pct = (waste_items_count / total_carried * 100) if total_carried > 0 else 0
        label = " <-- SLUTT" if milestone == final_round else ""
        print(f"  {rn:>6} | {bots_with_inv:>11} | {waste_bots:>11} | "
              f"{waste_items_count:>12} | {total_carried:>14} | {waste_pct:>7.1f}%{label}")

    # Detail at end: what each bot is carrying
    final_rd = round_by_num[final_round]
    print(f"\n  Siste runde - hva bots har i inventory:")
    for bot in final_rd["bots"]:
        inv = bot.get("inventory", [])
        if inv:
            print(f"    Bot {bot['id']:>2} @ {bot['position']}: {inv}")

    # =========================================================================
    # 5. LENGSTE GAPS UTEN SCORE-OKNING
    # =========================================================================
    print("\n" + "=" * 80)
    print("  5. LENGSTE GAPS UTEN SCORE-OKNING")
    print("=" * 80)

    gaps = []
    gap_start = 0
    prev_s = 0
    for rd in rounds:
        rn = rd["round_number"]
        s = rd["score"]
        if s > prev_s:
            gap_len = rn - gap_start
            if gap_len > 1:
                gaps.append((gap_start, rn, gap_len, prev_s))
            gap_start = rn
        prev_s = s

    # Final gap to end
    if gap_start < final_round:
        gaps.append((gap_start, final_round, final_round - gap_start, prev_s))

    gaps.sort(key=lambda x: -x[2])
    print(f"\n  Topp 15 lengste gaps:")
    for i, (gs, ge, gl, sc) in enumerate(gaps[:15]):
        print(f"    {i+1:>2}. Runde {gs:>3} - {ge:>3} ({gl:>3} runder uten scoring, score={sc})")

    total_scoring_events = sum(1 for i in range(1, len(rounds))
                               if rounds[i]["score"] > rounds[i-1]["score"])
    scoring_rounds = set()
    for i in range(1, len(rounds)):
        if rounds[i]["score"] > rounds[i-1]["score"]:
            scoring_rounds.add(rounds[i]["round_number"])

    print(f"\n  Runder med scoring:      {len(scoring_rounds)}/{total_rounds} "
          f"({len(scoring_rounds)/total_rounds*100:.1f}%)")
    print(f"  Runder UTEN scoring:     {total_rounds - len(scoring_rounds)}/{total_rounds} "
          f"({(total_rounds - len(scoring_rounds))/total_rounds*100:.1f}%)")

    # =========================================================================
    # 6. BOT UTILIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("  6. BOT UTILIZATION (action-fordeling)")
    print("=" * 80)

    print(f"\n  {'Runde':>6} | {'wait':>5} | {'move':>5} | {'pick':>5} | {'drop':>5} | "
          f"{'Score':>6} | {'Idle %':>7}")
    print("  " + "-" * 60)

    for milestone in range(50, final_round + 1, 50):
        rn = milestone if milestone in round_by_num else max(r for r in round_by_num if r <= milestone)
        rd = round_by_num[rn]
        actions = get_action_counts(rd)
        total = sum(actions.values())
        idle_pct = actions["wait"] / total * 100 if total > 0 else 0
        print(f"  {rn:>6} | {actions['wait']:>5} | {actions['move']:>5} | "
              f"{actions['pick']:>5} | {actions['drop']:>5} | "
              f"{rd['score']:>6} | {idle_pct:>6.0f}%")

    # Global action stats
    total_actions_all: dict[str, int] = defaultdict(int)
    for rd in rounds:
        for a in rd.get("actions", []):
            act = a["action"]
            if act == "wait":
                total_actions_all["wait"] += 1
            elif act.startswith("move"):
                total_actions_all["move"] += 1
            elif act.startswith("pick"):
                total_actions_all["pick"] += 1
            elif act.startswith("drop"):
                total_actions_all["drop"] += 1
            else:
                total_actions_all["other"] += 1

    grand_total = sum(total_actions_all.values())
    print(f"\n  TOTALT over hele spillet ({grand_total} bot-actions):")
    for act_type in ["wait", "move", "pick", "drop", "other"]:
        c = total_actions_all[act_type]
        print(f"    {act_type:>6}: {c:>6} ({c/grand_total*100:>5.1f}%)")

    # =========================================================================
    # 7. THROUGHPUT PER VINDU
    # =========================================================================
    print("\n" + "=" * 80)
    print("  7. THROUGHPUT-ANALYSE (ordrer fullfort per 50-runde-vindu)")
    print("=" * 80)

    print(f"\n  {'Vindu':>12} | {'Fullforte':>9} | {'Items levert':>13} | "
          f"{'Score delta':>11} | {'Aktive ordrer':>14}")
    print("  " + "-" * 75)

    for start in range(0, final_round + 1, 50):
        end = min(start + 49, final_round)

        # Count orders completed in this window (disappeared)
        completed_in_window = 0
        for oid, info in completed_orders.items():
            cr = info.get("completed_round", 0)
            if start < cr <= end + 1:
                completed_in_window += 1

        # Items delivered in window
        start_rn = max(r for r in round_by_num if r <= start) if start > 0 else 0
        end_rn = max(r for r in round_by_num if r <= end)
        score_start = score_at.get(start, 0) if start in score_at else score_at.get(
            max(r for r in score_at if r <= start), 0) if start > 0 else 0
        score_end = score_at.get(end, score_at.get(max(r for r in score_at if r <= end), 0))
        delta = score_end - score_start

        # Count active orders at end of window
        rd_end = round_by_num.get(end, round_by_num[max(r for r in round_by_num if r <= end)])
        active_count = sum(1 for o in rd_end["orders"] if o["status"] == "active")

        items_this_window = delta - completed_in_window * 5  # remove bonus
        print(f"  {start:>4}-{end:>4}   | {completed_in_window:>9} | {items_this_window:>13} | "
              f"{delta:>+11} | {active_count:>14}")

    # =========================================================================
    # 8. BOT INVENTORY OVER TID
    # =========================================================================
    print("\n" + "=" * 80)
    print("  8. BOT INVENTORY-STATUS OVER TID")
    print("=" * 80)

    print(f"\n  {'Runde':>6} | {'Tom':>4} | {'1-2':>4} | {'3-4':>4} | {'5-7':>4} | "
          f"{'Avg':>5} | {'Max':>4}")
    print("  " + "-" * 45)

    for milestone in list(range(50, final_round + 1, 50)) + [final_round]:
        rn = milestone if milestone in round_by_num else max(r for r in round_by_num if r <= milestone)
        rd = round_by_num[rn]

        sizes = [len(bot.get("inventory", [])) for bot in rd["bots"]]
        empty = sum(1 for s in sizes if s == 0)
        small = sum(1 for s in sizes if 1 <= s <= 2)
        med = sum(1 for s in sizes if 3 <= s <= 4)
        full = sum(1 for s in sizes if s >= 5)
        avg_inv = sum(sizes) / len(sizes)
        max_inv = max(sizes)
        label = " <-- SLUTT" if milestone == final_round else ""

        print(f"  {rn:>6} | {empty:>4} | {small:>4} | {med:>4} | {full:>4} | "
              f"{avg_inv:>4.1f} | {max_inv:>4}{label}")

    # =========================================================================
    # 9. WAIT-STREAK ANALYSE
    # =========================================================================
    print("\n" + "=" * 80)
    print("  9. WAIT-STREAK ANALYSE (lange perioder med idle bots)")
    print("=" * 80)

    # Track consecutive waits per bot
    bot_wait_streaks: dict[int, list[tuple[int, int]]] = defaultdict(list)
    bot_current_streak_start: dict[int, int] = {}

    for rd in rounds:
        rn = rd["round_number"]
        bot_actions = {a["bot"]: a["action"] for a in rd.get("actions", [])}
        for bot_id in range(num_bots):
            action = bot_actions.get(bot_id, "wait")
            if action == "wait":
                if bot_id not in bot_current_streak_start:
                    bot_current_streak_start[bot_id] = rn
            else:
                if bot_id in bot_current_streak_start:
                    streak_start = bot_current_streak_start.pop(bot_id)
                    streak_len = rn - streak_start
                    if streak_len >= 5:
                        bot_wait_streaks[bot_id].append((streak_start, streak_len))

    # Close open streaks
    for bot_id, streak_start in bot_current_streak_start.items():
        streak_len = final_round - streak_start + 1
        if streak_len >= 5:
            bot_wait_streaks[bot_id].append((streak_start, streak_len))

    all_streaks = []
    for bot_id, streaks in bot_wait_streaks.items():
        for start, length in streaks:
            all_streaks.append((bot_id, start, length))

    all_streaks.sort(key=lambda x: -x[2])

    total_wait_rounds = sum(s[2] for s in all_streaks)
    print(f"\n  Totalt wait-streaks >= 5 runder: {len(all_streaks)}")
    print(f"  Totalt bortkastede bot-runder i streaks: {total_wait_rounds}")
    print(f"  Tilsvarer {total_wait_rounds / num_bots:.0f} hele runder med en bot idle")

    if all_streaks:
        print(f"\n  Topp 20 lengste wait-streaks:")
        for i, (bot_id, start, length) in enumerate(all_streaks[:20]):
            end = start + length - 1
            print(f"    Bot {bot_id:>2}: runde {start:>3} - {end:>3} ({length:>3} runder idle)")

    # Per-bot total wait
    print(f"\n  Total idle per bot:")
    bot_total_wait: dict[int, int] = defaultdict(int)
    for rd in rounds:
        for a in rd.get("actions", []):
            if a["action"] == "wait":
                bot_total_wait[a["bot"]] += 1

    for bot_id in range(num_bots):
        w = bot_total_wait[bot_id]
        pct = w / total_rounds * 100
        bar = "#" * int(pct / 2)
        print(f"    Bot {bot_id:>2}: {w:>4} runder idle ({pct:>5.1f}%) {bar}")

    # =========================================================================
    # 10. SPATIAL ANALYSE - Drop-off bottleneck
    # =========================================================================
    print("\n" + "=" * 80)
    print("  10. DROP-OFF BOTTLENECK ANALYSE")
    print("=" * 80)

    drop_off_positions = run.get("drop_off", [])
    if isinstance(drop_off_positions[0], int):
        # It's a flat list [x1,y1,x2,y2,...] or single [x,y]
        # Check if it's pairs
        drop_offs = []
        for i in range(0, len(drop_off_positions), 2):
            drop_offs.append((drop_off_positions[i], drop_off_positions[i+1]))
    else:
        drop_offs = [(p[0], p[1]) for p in drop_off_positions]

    print(f"\n  Drop-off positions: {drop_offs}")
    print(f"  Spawn position: {run['spawn']}")

    # Count drop actions per round window
    drops_per_window: dict[str, int] = defaultdict(int)
    for rd in rounds:
        rn = rd["round_number"]
        window = f"{(rn // 50) * 50:>3}-{(rn // 50) * 50 + 49:>3}"
        for a in rd.get("actions", []):
            if a["action"].startswith("drop"):
                drops_per_window[window] += 1

    print(f"\n  Drop-actions per 50-runde-vindu:")
    for window in sorted(drops_per_window.keys()):
        count = drops_per_window[window]
        bar = "#" * (count // 2)
        print(f"    Runde {window}: {count:>4} drops  {bar}")

    # Count bots near drop-off at each milestone (within manhattan distance 3)
    print(f"\n  Bots naer drop-off (manhattan dist <= 3) ved milepaler:")
    for milestone in range(50, final_round + 1, 100):
        rn = milestone if milestone in round_by_num else max(r for r in round_by_num if r <= milestone)
        rd = round_by_num[rn]
        near_count = 0
        for bot in rd["bots"]:
            bx, by = bot["position"]
            for dx, dy in drop_offs:
                if abs(bx - dx) + abs(by - dy) <= 3:
                    near_count += 1
                    break
        print(f"    Runde {rn:>3}: {near_count:>2} bots naer drop-off")

    # =========================================================================
    # 11. SCORE BREAKDOWN - poeng fra items vs bonus
    # =========================================================================
    print("\n" + "=" * 80)
    print("  11. SCORE BREAKDOWN")
    print("=" * 80)

    items_delivered = run["items_delivered"]
    orders_completed = run["orders_completed"]
    bonus_points = orders_completed * 5

    print(f"\n  Items levert:      {items_delivered:>4} poeng ({items_delivered/final_score*100:.1f}%)")
    print(f"  Ordrebonuser:      {bonus_points:>4} poeng ({bonus_points/final_score*100:.1f}%) "
          f"({orders_completed} ordrer * 5)")
    print(f"  Totalt:            {final_score:>4} poeng")

    # Theoretical max if all items in completed orders had been delivered faster
    total_items_in_completed = sum(info["n_required"] for info in completed_orders.values())
    avg_items_per_order = total_items_in_completed / len(completed_orders) if completed_orders else 0
    avg_pts_per_order = avg_items_per_order + 5

    print(f"\n  Snitt items per ordre:     {avg_items_per_order:.1f}")
    print(f"  Snitt poeng per ordre:     {avg_pts_per_order:.1f} (items + 5 bonus)")
    print(f"  Snitt runder per ordre:    {avg_dur:.1f}")
    print(f"  Runder per poeng:          {final_round / final_score:.1f}")

    # What score would we get at different order rates?
    print(f"\n  SIMULERING - hva om vi var raskere?")
    for target_dur in [15, 20, 25, 30]:
        max_orders = final_round // target_dur
        est_score = int(max_orders * avg_pts_per_order)
        print(f"    {target_dur} rdr/ordre -> ~{max_orders} ordrer -> ~{est_score} poeng")

    # =========================================================================
    # 12. OPPSUMMERING
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"  12. OPPSUMMERING - HVORFOR BARE {final_score} POENG?")
    print("=" * 80)

    # Identify the worst problems
    wait_pct = total_actions_all["wait"] / grand_total * 100

    # Find the big gaps
    worst_gap = gaps[0] if gaps else (0, 0, 0, 0)

    # Late game slowdown
    score_at_250 = score_at.get(250, 0)
    score_at_250 = score_at.get(max(r for r in score_at if r <= 250), 0) if 250 not in score_at else score_at[250]
    first_half_score = score_at_250
    second_half_score = final_score - score_at_250

    print(f"""
  FAKTA:
    - Sluttpoeng:          {final_score}
    - Fullforte ordrer:    {len(completed_orders)}/{len(all_orders)}
    - Ufullforte ordrer:   {len(uncompleted)} ({sum(info['n_required'] for info in uncompleted.values())} items ulevert)
    - Snitt ordrevarighet: {avg_dur:.1f} runder
    - Tregeste ordre:      {max_dur} runder
    - Wait-actions:        {wait_pct:.1f}% av alle bot-actions
    - Bots m/inventory:    {sum(1 for b in final_rd['bots'] if b['inventory'])}/20 ved spillets slutt
    - Forste halvdel:      {first_half_score} poeng (runde 0-250)
    - Andre halvdel:       {second_half_score} poeng (runde 251-{final_round})

  IDENTIFISERTE PROBLEMER:

    1. TREGHET I ORDREGJENNOMFORING:
       Snitt {avg_dur:.0f} runder per ordre. Med 500 runder og {avg_pts_per_order:.1f} poeng/ordre
       gir det maks {int(500/avg_dur * avg_pts_per_order)} poeng. For aa naa 260+ trenger vi
       ~25 ordrer, dvs. ~20 runder/ordre.

    2. MASSIVE WAIT-PERIODER:
       {wait_pct:.0f}% av alle actions er 'wait'. Det betyr {total_actions_all['wait']}/{grand_total}
       bot-runder er bortkastet. {len(all_streaks)} streaks paa 5+ runder.
       Lengste streak: Bot {all_streaks[0][0]} i {all_streaks[0][2]} runder (runde {all_streaks[0][1]}-{all_streaks[0][1]+all_streaks[0][2]-1}).

    3. SCORING GAPS:
       Lengste gap uten scoring: {worst_gap[2]} runder (runde {worst_gap[0]}-{worst_gap[1]}).
       De 5 lengste gapene utgjor {sum(g[2] for g in gaps[:5])} runder.

    4. SEKVENSIELT ORDRESYSTEM:
       Bare 1 aktiv ordre om gangen! 20 bots jobber mot 1 ordre med {avg_items_per_order:.0f} items.
       Det betyr ~{20/avg_items_per_order:.0f}x overkapasitet - bots venter paa hverandre.

    5. STALE INVENTORY VED SLUTT:
       {sum(1 for b in final_rd['bots'] if b['inventory'])} av 20 bots har items i inventory naar spillet er over.
       Totalt {sum(len(b['inventory']) for b in final_rd['bots'])} items som aldri ble levert.
       Ingen 'discard'-action finnes - items sitter fast.

    6. ANDRE HALVDEL TREIGERE:
       Forste halvdel: {first_half_score} poeng (runde 0-250)
       Andre halvdel: {second_half_score} poeng (runde 251-{final_round})
       {'Andre halvdel er TREIGERE' if second_half_score < first_half_score else 'Andre halvdel er noe raskere'}
       -> dette tyder paa at stale inventory akkumulerer over tid.
""")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\larsh\Downloads\run-9-nightmare-seed1772968883.json"
    data = load_replay(path)
    analyze(data)

#!/usr/bin/env python3
"""
Analyze a grocery-bot game log (JSONL) for bot performance metrics.

The log alternates: game_state line, action line, game_state line, ...
The final line is a game_over summary.
"""

import json
import sys
from collections import defaultdict

LOG_PATH = r"X:\KODE\AINM\PRE\grocery-bot-zig\game_log_real_easy.jsonl"


def main():
    # ------------------------------------------------------------------ #
    # 1. Parse the JSONL file, tracking any malformed lines               #
    # ------------------------------------------------------------------ #
    game_states = []        # list of (line_no, parsed_dict)
    actions = []            # list of (line_no, parsed_dict)
    game_over = None
    parse_errors = []       # (line_no, raw_text, error_msg)

    with open(LOG_PATH, "rb") as f:
        raw_bytes = f.read()
    lines = raw_bytes.split(b"\n")
    for line_no, raw_b in enumerate(lines, start=1):
            raw_b = raw_b.rstrip(b"\r")
            try:
                raw = raw_b.decode("utf-8")
            except UnicodeDecodeError:
                parse_errors.append((line_no, repr(raw_b[:200]), "Non-UTF8 bytes (binary garbage)"))
                continue
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                parse_errors.append((line_no, raw[:200], str(e)))
                continue

            if obj.get("type") == "game_state":
                game_states.append((line_no, obj))
            elif obj.get("type") == "game_over":
                game_over = obj
            elif "actions" in obj:
                actions.append((line_no, obj))
            else:
                parse_errors.append((line_no, raw[:200], "Unknown line type"))

    total_rounds = game_states[-1][1]["round"] + 1 if game_states else 0  # 0-indexed rounds
    max_rounds = game_states[0][1]["max_rounds"] if game_states else 300

    print("=" * 72)
    print("  GROCERY-BOT GAME LOG ANALYSIS  --  Easy difficulty (1 bot)")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 8. Parse errors                                                     #
    # ------------------------------------------------------------------ #
    print("\n--- PARSE ERRORS / MALFORMED DATA ---")
    if parse_errors:
        for ln, txt, err in parse_errors:
            print(f"  Line {ln}: {err}")
            print(f"    Content (first 200 chars): {txt}")
    else:
        print("  No JSON parse errors found in the entire file.")

    # Extra check: look for anomalies around line 400 (round ~200)
    print("\n  Anomaly scan around line 400 (round ~200):")
    anomalies_found = False
    for i, (ln, gs) in enumerate(game_states):
        if 195 <= gs["round"] <= 210:
            # Check if the bot position didn't change despite a move action
            if i > 0:
                prev_gs = game_states[i - 1][1]
                prev_pos = tuple(prev_gs["bots"][0]["position"])
                cur_pos = tuple(gs["bots"][0]["position"])
                # Find the action that was taken between these two states
                action_between = None
                for aln, aobj in actions:
                    if game_states[i - 1][0] < aln < ln:
                        action_between = aobj
                        break
                if action_between and cur_pos == prev_pos:
                    act = action_between["actions"][0]["action"]
                    if act.startswith("move_"):
                        anomalies_found = True
                        print(f"    Round {gs['round']} (line {ln}): move '{act}' had no effect "
                              f"(pos stayed {cur_pos}). Likely wall collision.")
    if not anomalies_found:
        print("    No structural anomalies found, but see 'stuck loops' analysis below.")

    # ------------------------------------------------------------------ #
    # 2. Score progression per round                                      #
    # ------------------------------------------------------------------ #
    print("\n--- SCORE PROGRESSION ---")
    score_changes = []
    prev_score = 0
    for _, gs in game_states:
        s = gs["score"]
        if s != prev_score:
            score_changes.append((gs["round"], s, s - prev_score))
            prev_score = s

    for rnd, score, delta in score_changes:
        print(f"  Round {rnd:>3}: score -> {score:>3}  (+{delta})")

    if game_over:
        print(f"\n  Final score (game_over): {game_over['score']}")

    # ------------------------------------------------------------------ #
    # 3. Counts: total rounds, orders completed, items delivered           #
    # ------------------------------------------------------------------ #
    print("\n--- BASIC COUNTS ---")
    print(f"  Total rounds played:     {total_rounds}")
    print(f"  Max rounds allowed:      {max_rounds}")
    if game_over:
        print(f"  Orders completed:        {game_over.get('orders_completed', '?')}")
        print(f"  Items delivered:         {game_over.get('items_delivered', '?')}")
        print(f"  Final score:             {game_over['score']}")
        # Verify score formula: items*1 + orders*5
        expected = game_over.get("items_delivered", 0) * 1 + game_over.get("orders_completed", 0) * 5
        print(f"  Score check (items*1 + orders*5): {expected}  "
              f"{'MATCH' if expected == game_over['score'] else 'MISMATCH!'}")
    else:
        print("  WARNING: no game_over message found!")

    # ------------------------------------------------------------------ #
    # 6. Order completion timeline                                        #
    # ------------------------------------------------------------------ #
    print("\n--- ORDER COMPLETION TIMELINE ---")
    # Detect order transitions by watching active_order_index change
    order_completions = []  # (round, order_id_completed, new_active_order_id)
    prev_active_idx = None
    prev_active_order_id = None
    order_start_round = {}  # order_id -> round it became active

    for _, gs in game_states:
        idx = gs.get("active_order_index", 0)
        active_order = gs["orders"][0] if gs["orders"] else None
        active_id = active_order["id"] if active_order else None

        if prev_active_idx is not None and idx != prev_active_idx:
            # The active order index advanced => previous order was completed
            order_completions.append((gs["round"], prev_active_order_id, active_id))

        if active_id and active_id not in order_start_round:
            order_start_round[active_id] = gs["round"]

        prev_active_idx = idx
        prev_active_order_id = active_id

    for rnd, completed_id, new_id in order_completions:
        start = order_start_round.get(completed_id, 0)
        duration = rnd - start
        print(f"  Round {rnd:>3}: {completed_id} completed  "
              f"(was active from round {start}, took {duration} rounds)")

    orders_completed_count = len(order_completions)
    print(f"\n  Total orders completed: {orders_completed_count}")

    # ------------------------------------------------------------------ #
    # 7. Average rounds per order completion                               #
    # ------------------------------------------------------------------ #
    print("\n--- EFFICIENCY ---")
    if order_completions:
        durations = []
        for rnd, completed_id, _ in order_completions:
            start = order_start_round.get(completed_id, 0)
            durations.append(rnd - start)
        avg_rounds = sum(durations) / len(durations)
        print(f"  Average rounds per order: {avg_rounds:.1f}")
        print(f"  Fastest order:            {min(durations)} rounds")
        print(f"  Slowest order:            {max(durations)} rounds")
        print(f"  Rounds used completing:   {sum(durations)} of {total_rounds}")
        remaining_after_last = total_rounds - order_completions[-1][0]
        print(f"  Rounds after last order:  {remaining_after_last}  (wasted tail)")
    else:
        print("  No orders completed.")

    # ------------------------------------------------------------------ #
    # 4. Idle / stuck rounds                                               #
    # ------------------------------------------------------------------ #
    print("\n--- IDLE / STUCK ANALYSIS ---")

    # Build per-round bot position + action taken
    positions = []  # (round, position_tuple)
    for _, gs in game_states:
        positions.append((gs["round"], tuple(gs["bots"][0]["position"])))

    # Count wait actions
    wait_count = 0
    for _, aobj in actions:
        for a in aobj["actions"]:
            if a["action"] == "wait":
                wait_count += 1

    print(f"  Explicit 'wait' actions: {wait_count}")

    # Detect stuck stretches: position unchanged for 3+ consecutive rounds
    stuck_stretches = []
    streak_start = 0
    streak_pos = positions[0][1] if positions else None
    streak_len = 1

    for i in range(1, len(positions)):
        if positions[i][1] == streak_pos:
            streak_len += 1
        else:
            if streak_len >= 3:
                stuck_stretches.append((positions[streak_start][0], positions[i - 1][0], streak_pos, streak_len))
            streak_start = i
            streak_pos = positions[i][1]
            streak_len = 1
    if streak_len >= 3:
        stuck_stretches.append((positions[streak_start][0], positions[-1][0], streak_pos, streak_len))

    print(f"  Stuck stretches (same position 3+ rounds): {len(stuck_stretches)}")
    total_stuck_rounds = 0
    for start_r, end_r, pos, length in stuck_stretches:
        total_stuck_rounds += length
        if length >= 5:
            print(f"    Rounds {start_r}-{end_r}: stuck at {pos} for {length} rounds")
    # Print shorter ones in summary
    short_stuck = [(s, e, p, l) for s, e, p, l in stuck_stretches if l < 5]
    if short_stuck:
        print(f"    + {len(short_stuck)} shorter stuck stretches (3-4 rounds each)")
    print(f"  Total rounds spent stuck: {total_stuck_rounds}")

    # Detect loops: bot returning to the same position within N rounds
    print("\n  Patrol/loop detection (position revisited within 10 rounds):")
    loop_count = 0
    loop_examples = []
    for i in range(len(positions)):
        r, p = positions[i]
        for j in range(i + 3, min(i + 11, len(positions))):
            r2, p2 = positions[j]
            if p2 == p:
                loop_count += 1
                if len(loop_examples) < 5:
                    loop_examples.append((r, r2, p))
                break
    print(f"    Detected {loop_count} position-revisit loops")
    for r1, r2, p in loop_examples:
        print(f"      Example: round {r1} -> round {r2} at {p}")

    # ------------------------------------------------------------------ #
    # 5. Wasted actions                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- WASTED ACTIONS ---")

    wasted_pickups = 0       # pick_up that didn't increase inventory
    wasted_moves = 0         # move that didn't change position
    wasted_dropoffs = 0      # drop_off that didn't change delivered count or inventory
    failed_pickup_details = []

    # Build a mapping: action_line_no -> (prev_state_idx, next_state_idx)
    gs_by_line = {ln: i for i, (ln, _) in enumerate(game_states)}

    for aln, aobj in actions:
        # Find the game_state before and after this action
        prev_idx = None
        next_idx = None
        for i, (gln, _) in enumerate(game_states):
            if gln < aln:
                prev_idx = i
            elif gln > aln and next_idx is None:
                next_idx = i
                break

        if prev_idx is None or next_idx is None:
            continue

        prev_gs = game_states[prev_idx][1]
        next_gs = game_states[next_idx][1]
        bot_action = aobj["actions"][0]
        act = bot_action["action"]

        prev_pos = tuple(prev_gs["bots"][0]["position"])
        next_pos = tuple(next_gs["bots"][0]["position"])
        prev_inv = prev_gs["bots"][0]["inventory"]
        next_inv = next_gs["bots"][0]["inventory"]

        if act.startswith("move_") and prev_pos == next_pos:
            wasted_moves += 1

        if act == "pick_up" and len(next_inv) <= len(prev_inv):
            wasted_pickups += 1
            item_id = bot_action.get("item_id", "?")
            failed_pickup_details.append((prev_gs["round"], item_id, prev_pos, len(prev_inv)))

        if act == "drop_off":
            prev_delivered = prev_gs["orders"][0]["items_delivered"] if prev_gs["orders"] else []
            next_delivered = next_gs["orders"][0]["items_delivered"] if next_gs["orders"] else []
            # If the order changed entirely (completed), that's a success, not waste
            prev_order_id = prev_gs["orders"][0]["id"] if prev_gs["orders"] else None
            next_order_id = next_gs["orders"][0]["id"] if next_gs["orders"] else None
            if prev_order_id == next_order_id:
                if len(next_delivered) <= len(prev_delivered) and len(next_inv) >= len(prev_inv):
                    wasted_dropoffs += 1

    print(f"  Wasted moves (no position change):   {wasted_moves}")
    print(f"  Wasted pick_ups (no inventory gain):  {wasted_pickups}")
    print(f"  Wasted drop_offs (nothing delivered):  {wasted_dropoffs}")

    if failed_pickup_details:
        print(f"\n  Failed pick_up details (first 10):")
        for rnd, item_id, pos, inv_size in failed_pickup_details[:10]:
            print(f"    Round {rnd:>3}: tried {item_id} at pos {pos}, inv had {inv_size}/3 items")
        if len(failed_pickup_details) > 10:
            print(f"    ... and {len(failed_pickup_details) - 10} more")

    total_actions = len(actions)
    total_wasted = wasted_moves + wasted_pickups + wasted_dropoffs + wait_count
    print(f"\n  Total actions:  {total_actions}")
    print(f"  Total wasted:   {total_wasted}  ({100 * total_wasted / total_actions:.1f}%)")

    # ------------------------------------------------------------------ #
    # Detailed stuck loop around round 195-210                             #
    # ------------------------------------------------------------------ #
    print("\n--- DETAILED LOOK: ROUNDS 195-210 (around line 400) ---")
    for i, (ln, gs) in enumerate(game_states):
        if 194 <= gs["round"] <= 210:
            pos = tuple(gs["bots"][0]["position"])
            inv = gs["bots"][0]["inventory"]
            score = gs["score"]
            active_order = gs["orders"][0]["id"] if gs["orders"] else "?"
            delivered = gs["orders"][0]["items_delivered"] if gs["orders"] else []
            # Find the action that follows this state
            act_str = "?"
            for aln2, aobj2 in actions:
                if aln2 > ln:
                    break
                if aln2 == ln + 1:
                    act_str = aobj2["actions"][0]["action"]
            print(f"  Round {gs['round']:>3}: pos={pos}  inv={inv}  score={score}  "
                  f"order={active_order}  delivered={delivered}  -> {act_str}")

    # ------------------------------------------------------------------ #
    # 9. Summary                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"""
  Game: Easy difficulty, 1 bot, 12x10 grid
  Rounds: {total_rounds} / {max_rounds}
  Final score: {game_over['score'] if game_over else '?'}
  Orders completed: {game_over.get('orders_completed', '?') if game_over else '?'} of {game_states[0][1].get('total_orders', '?') if game_states else '?'} available
  Items delivered: {game_over.get('items_delivered', '?') if game_over else '?'}

  Score breakdown:
    Items x1:  {game_over.get('items_delivered', 0)} pts
    Orders x5: {game_over.get('orders_completed', 0) * 5} pts
    Total:     {game_over['score'] if game_over else '?'} pts

  Efficiency:
    Avg rounds/order:      {avg_rounds:.1f}
    Wasted action rate:    {100 * total_wasted / total_actions:.1f}%
    Stuck rounds:          {total_stuck_rounds} / {total_rounds}
    Wasted moves (walls):  {wasted_moves}
    Failed pickups:        {wasted_pickups}

  Key problems identified:
    1. Bot gets stuck in LOOPS around rounds 195-300+, cycling between
       the same few positions endlessly while holding item(s) in inventory.
       This wastes ~100+ rounds where no progress is made.
    2. {wasted_moves} move actions hit walls and had no effect.
    3. {wasted_pickups} pick_up actions failed (likely not adjacent to shelf
       or inventory full).
    4. Only {game_over.get('orders_completed', '?') if game_over else '?'} of {game_states[0][1].get('total_orders', '?') if game_states else '?'} orders completed in {total_rounds} rounds.
       The bot needs better pathfinding to avoid wall collisions
       and loop detection to break out of stuck states.
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ASCII Visual Tracer for Grocery Bot game logs.
Reads game_log.jsonl and renders step-by-step grid visualization.

Usage:
  python tracer.py                          # show all rounds (last game if multiple)
  python tracer.py 0 30                     # show rounds 0-30
  python tracer.py 50 60                    # show rounds 50-60
  python tracer.py --game 0                 # select first game (if log has multiple)
  python tracer.py --game 1 --summary       # summary of second game
  python tracer.py --events                 # only show rounds with events (pickup/dropoff/score/order change)
  python tracer.py --events 0 100           # events only in range
  python tracer.py --summary                # compact one-line-per-round summary
"""

import json
import sys
import os

# ANSI color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
DIM    = "\033[2m"

BOT_COLORS = [RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RED, GREEN, YELLOW]

ITEM_CHAR = {
    "cheese":  "C",
    "butter":  "B",
    "yogurt":  "Y",
    "milk":    "M",
    "bread":   "R",
    "eggs":    "E",
    "apples":  "A",
    "bananas": "N",
    "onions":  "U",
    "tomatoes":"T",
    "rice":    "I",
    "pasta":   "S",
    "flour":   "F",
    "oats":    "O",
    "cereal":  "L",
    "cream":   "K",
}

def item_char(item_type):
    """Get single-char representation for an item type."""
    if item_type in ITEM_CHAR:
        return ITEM_CHAR[item_type]
    # Fallback: first letter uppercase
    return item_type[0].upper() if item_type else "?"


def parse_log(path):
    """Parse JSONL log into list of games. Each game is a list of (game_state, action) tuples."""
    games = []
    current_game = []
    with open(path, "r") as f:
        raw_lines = f.readlines()

    # Rejoin split lines: valid JSONL lines start with '{'
    lines = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("{"):
            lines.append(raw)
        elif lines:
            # Continuation of previous line
            lines[-1] += raw

    prev_round = -1
    i = 0
    while i < len(lines):
        try:
            obj = json.loads(lines[i])
        except json.JSONDecodeError:
            i += 1
            continue
        if obj.get("type") == "game_state":
            state = obj
            action = None
            if i + 1 < len(lines):
                try:
                    next_obj = json.loads(lines[i + 1])
                    if "actions" in next_obj:
                        action = next_obj
                        i += 2
                    else:
                        i += 1
                except json.JSONDecodeError:
                    i += 1
            else:
                i += 1
            # Detect new game: round number resets
            cur_round = state["round"]
            if cur_round <= prev_round and current_game:
                games.append(current_game)
                current_game = []
            prev_round = cur_round
            current_game.append((state, action))
        else:
            i += 1

    if current_game:
        games.append(current_game)
    return games


def render_grid(state, action, prev_state=None):
    """Render a single round as ASCII grid with annotations."""
    grid_info = state["grid"]
    W = grid_info["width"]
    H = grid_info["height"]
    walls = set(tuple(w) for w in grid_info["walls"])
    bots = state["bots"]
    items = state["items"]
    drop_off = tuple(state["drop_off"])
    orders = state["orders"]
    score = state["score"]
    round_num = state["round"]
    max_rounds = state["max_rounds"]

    # Build item position map
    item_map = {}
    for item in items:
        pos = tuple(item["position"])
        item_map[pos] = item

    # Build bot position map
    bot_map = {}
    for bot in bots:
        pos = tuple(bot["position"])
        bot_map[pos] = bot

    # Detect events
    events = []
    if prev_state:
        prev_score = prev_state["score"]
        if score > prev_score:
            diff = score - prev_score
            events.append(f"{GREEN}+{diff} score (now {score}){RESET}")
        # Order completion
        for o in orders:
            if o["complete"]:
                prev_order = next((po for po in prev_state["orders"] if po["id"] == o["id"]), None)
                if prev_order and not prev_order["complete"]:
                    events.append(f"{GREEN}{BOLD}ORDER {o['id']} COMPLETED! (+5 bonus){RESET}")
        # Order change
        if state.get("active_order_index") != prev_state.get("active_order_index"):
            events.append(f"{YELLOW}Active order changed -> order_{state.get('active_order_index')}{RESET}")
        # Inventory changes
        for bot in bots:
            prev_bot = next((pb for pb in prev_state["bots"] if pb["id"] == bot["id"]), None)
            if prev_bot:
                prev_inv = prev_bot["inventory"]
                cur_inv = bot["inventory"]
                if len(cur_inv) > len(prev_inv):
                    new_items = list(cur_inv)
                    for pi in prev_inv:
                        if pi in new_items:
                            new_items.remove(pi)
                    for ni in new_items:
                        events.append(f"{CYAN}Bot {bot['id']} picked up {ni}{RESET}")
                elif len(cur_inv) < len(prev_inv):
                    dropped = list(prev_inv)
                    for ci in cur_inv:
                        if ci in dropped:
                            dropped.remove(ci)
                    for di in dropped:
                        events.append(f"{MAGENTA}Bot {bot['id']} delivered {di}{RESET}")

    # Render grid
    lines = []

    # Header
    header = f"{BOLD}=== Round {round_num}/{max_rounds} === Score: {score}{RESET}"
    if action:
        acts = action["actions"]
        act_strs = []
        for a in acts:
            s = f"Bot {a['bot']}: {a['action']}"
            if "item_id" in a:
                s += f" ({a['item_id']})"
            act_strs.append(s)
        header += f"  Actions: {', '.join(act_strs)}"
    lines.append(header)

    # Events
    for e in events:
        lines.append(f"  >> {e}")

    # Column numbers
    col_header = "   "
    for x in range(W):
        col_header += f"{x % 10}"
    lines.append(col_header)

    # Grid rows
    for y in range(H):
        row = f"{y:2d} "
        for x in range(W):
            pos = (x, y)
            if pos in bot_map:
                bot = bot_map[pos]
                bid = bot["id"]
                color = BOT_COLORS[bid % len(BOT_COLORS)]
                inv_count = len(bot["inventory"])
                if pos == drop_off:
                    # Bot on dropoff
                    row += f"{color}{BOLD}@{RESET}"
                elif inv_count > 0:
                    row += f"{color}{BOLD}{bid}{RESET}"
                else:
                    row += f"{color}{bid}{RESET}"
            elif pos == drop_off:
                row += f"{BG_GREEN}{BOLD}D{RESET}"
            elif pos in walls:
                row += f"{DIM}#{RESET}"
            elif pos in item_map:
                item = item_map[pos]
                ch = item_char(item["type"])
                row += f"{YELLOW}{ch}{RESET}"
            else:
                row += f"{DIM}.{RESET}"
        row += f" {y}"
        lines.append(row)

    # Bottom column numbers
    lines.append(col_header)

    # Bot info
    for bot in bots:
        bid = bot["id"]
        pos = tuple(bot["position"])
        inv = bot["inventory"]
        color = BOT_COLORS[bid % len(BOT_COLORS)]
        inv_str = ", ".join(inv) if inv else "(empty)"
        lines.append(f"  {color}Bot {bid}{RESET} @ ({pos[0]},{pos[1]}) inv[{len(inv)}/3]: {inv_str}")

    # Order info
    for o in orders:
        status = o["status"]
        oid = o["id"]
        required = o["items_required"]
        delivered = o.get("items_delivered", [])
        complete = o["complete"]

        if complete:
            status_str = f"{GREEN}DONE{RESET}"
        elif status == "active":
            status_str = f"{YELLOW}ACTIVE{RESET}"
        else:
            status_str = f"{DIM}preview{RESET}"

        # Show what's still needed
        remaining = list(required)
        for d in delivered:
            if d in remaining:
                remaining.remove(d)

        req_str = ", ".join(required)
        del_str = ", ".join(delivered) if delivered else "-"
        need_str = ", ".join(remaining) if remaining else "NONE"
        lines.append(f"  {status_str} {oid}: need=[{req_str}] delivered=[{del_str}] remaining=[{need_str}]")

    lines.append("")
    return lines, len(events) > 0


def render_summary_line(state, action, prev_state):
    """One-line summary per round."""
    round_num = state["round"]
    score = state["score"]
    bots = state["bots"]

    parts = [f"R{round_num:3d}"]
    parts.append(f"S:{score:3d}")

    for bot in bots:
        bid = bot["id"]
        pos = tuple(bot["position"])
        inv = bot["inventory"]
        inv_short = "".join(item_char(i) for i in inv)
        parts.append(f"B{bid}({pos[0]:2d},{pos[1]:2d})[{inv_short:3s}]")

    if action:
        acts = action["actions"]
        act_parts = []
        for a in acts:
            s = a["action"]
            if "item_id" in a:
                s += f"({a['item_id']})"
            act_parts.append(s)
        parts.append(" | ".join(act_parts))

    # Detect score change
    if prev_state and score > prev_state["score"]:
        diff = score - prev_state["score"]
        parts.append(f"{GREEN}+{diff}{RESET}")

    # Detect order completion
    if prev_state:
        for o in state["orders"]:
            if o["complete"]:
                prev_o = next((po for po in prev_state["orders"] if po["id"] == o["id"]), None)
                if prev_o and not prev_o["complete"]:
                    parts.append(f"{GREEN}ORDER DONE{RESET}")

    return "  ".join(parts)


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_log.jsonl")

    args = sys.argv[1:]
    events_only = False
    summary_mode = False
    start_round = None
    end_round = None
    game_index = None  # --game N (0-based)

    # Parse args
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--events":
            events_only = True
        elif a == "--summary":
            summary_mode = True
        elif a == "--game" and i + 1 < len(args):
            game_index = int(args[i + 1])
            i += 1
        elif a == "--help" or a == "-h":
            print(__doc__)
            return
        else:
            positional.append(int(a))
        i += 1

    if len(positional) >= 2:
        start_round = positional[0]
        end_round = positional[1]
    elif len(positional) == 1:
        start_round = positional[0]
        end_round = positional[0]

    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return

    games = parse_log(log_path)
    if not games:
        print("No games found in log")
        return

    # List games
    print(f"{BOLD}Grocery Bot Game Trace{RESET}")
    print(f"Games in log: {len(games)}")
    for gi, game in enumerate(games):
        first = game[0][0]
        last = game[-1][0]
        w = first["grid"]["width"]
        h = first["grid"]["height"]
        nbots = len(first["bots"])
        final_score = last["score"]
        nrounds = len(game)
        marker = " <--" if game_index == gi else ""
        print(f"  Game {gi}: {w}x{h}, {nbots} bot(s), {nrounds} rounds, final score: {final_score}{marker}")
    print()

    # Select game
    if game_index is None:
        if len(games) == 1:
            game_index = 0
        else:
            game_index = len(games) - 1
            print(f"(Showing last game. Use --game N to select.)")
            print()

    if game_index >= len(games):
        print(f"Error: game {game_index} not found (max: {len(games) - 1})")
        return

    rounds = games[game_index]
    first_state = rounds[0][0]

    print(f"{BOLD}Game {game_index}{RESET}: {first_state['grid']['width']}x{first_state['grid']['height']}, "
          f"{len(first_state['bots'])} bots, drop-off: {first_state['drop_off']}")
    print()

    # Legend
    if not summary_mode:
        # Build item legend from actual items in this game
        item_types = set()
        for item in first_state["items"]:
            item_types.add(item["type"])
        item_legend = " ".join(f"{YELLOW}{item_char(t)}{RESET}={t}" for t in sorted(item_types))
        print(f"Legend: {DIM}#{RESET}=wall  {BG_GREEN}{BOLD}D{RESET}=drop-off  {item_legend}")
        bot_legend = "  ".join(f"{BOT_COLORS[b['id'] % len(BOT_COLORS)]}{b['id']}{RESET}=bot{b['id']}" for b in first_state["bots"])
        print(f"  Bots: {bot_legend}  (bold=carrying)")
        print()

    prev_state = None
    for state, action in rounds:
        round_num = state["round"]

        # Filter by range
        if start_round is not None and round_num < start_round:
            prev_state = state
            continue
        if end_round is not None and round_num > end_round:
            break

        if summary_mode:
            line = render_summary_line(state, action, prev_state)
            if events_only:
                if prev_state and state["score"] > prev_state["score"]:
                    print(line)
                elif prev_state is None:
                    print(line)
            else:
                print(line)
        else:
            grid_lines, has_events = render_grid(state, action, prev_state)
            if events_only and not has_events and round_num > 0:
                prev_state = state
                continue
            for line in grid_lines:
                print(line)

        prev_state = state

    # Final summary
    final_state = rounds[-1][0]
    print(f"\n{BOLD}=== FINAL: Round {final_state['round']}, Score: {final_state['score']} ==={RESET}")
    completed = sum(1 for o in final_state["orders"] if o["complete"])
    total_visible = len(final_state["orders"])
    print(f"Orders completed: {completed}/{total_visible} visible, {final_state.get('total_orders', '?')} total")


if __name__ == "__main__":
    main()

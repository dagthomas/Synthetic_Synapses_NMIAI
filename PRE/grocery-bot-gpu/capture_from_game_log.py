"""Extract capture data from a Zig bot game_log_*.jsonl file.

Parses all game_state lines to accumulate the full order list revealed
across 300 rounds, then saves capture.json for GPU DP optimization.

Usage:
    python capture_from_game_log.py <game_log_path> <difficulty>
    python capture_from_game_log.py --latest <difficulty>
"""
import json
import sys
import os
import glob
import argparse

from solution_store import merge_capture, save_solution
from game_engine import (build_map_from_capture,
                         ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN,
                         ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_PICKUP, ACT_DROPOFF)
from configs import detect_difficulty


def find_latest_game_log(search_dirs=None):
    """Find the most recent game_log_*.jsonl across search directories."""
    if search_dirs is None:
        here = os.path.dirname(os.path.abspath(__file__))
        zig_dir = os.path.join(os.path.dirname(here), 'grocery-bot-zig')
        search_dirs = [here, zig_dir]

    best = None
    best_mtime = 0
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, 'game_log_*.jsonl')):
            mtime = os.path.getmtime(f)
            size = os.path.getsize(f)
            if size > 0 and mtime > best_mtime:
                best = f
                best_mtime = mtime
    return best


def extract_capture(game_log_path, difficulty=None):
    """Parse a game_log JSONL and extract capture data.

    Returns (capture_dict, final_score, difficulty) or raises on error.
    """
    grid = None
    items = None
    drop_off = None
    num_bots = 0
    width = height = 0
    seen_order_ids = set()
    orders_in_order = []
    final_score = 0

    with open(game_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # JSONL log may contain non-JSON lines (stderr, partial writes)

            # Skip action lines
            if 'actions' in data and 'type' not in data:
                continue

            if data.get('type') == 'game_over':
                final_score = data.get('score', final_score)
                continue

            if data.get('type') != 'game_state':
                continue

            # Round 0: extract static map data
            if data['round'] == 0:
                grid = data['grid']
                items = data['items']
                drop_off = data['drop_off']
                drop_off_zones = data.get('drop_off_zones')
                num_bots = len(data['bots'])
                width = grid['width']
                height = grid['height']

            final_score = data.get('score', final_score)

            # Accumulate unique orders across all rounds
            for order in data.get('orders', []):
                oid = order['id']
                if oid not in seen_order_ids:
                    seen_order_ids.add(oid)
                    orders_in_order.append({
                        'items_required': list(order['items_required']),
                    })

    if grid is None:
        raise ValueError(f"No valid game_state found in {game_log_path}")

    # Auto-detect difficulty if not provided
    if difficulty is None:
        difficulty = detect_difficulty(num_bots, width=width, height=height)
        if difficulty is None:
            raise ValueError(
                f"Cannot auto-detect difficulty for {width}x{height} with {num_bots} bots")

    capture = {
        'grid': grid,
        'items': items,
        'drop_off': drop_off,
        'num_bots': num_bots,
        'difficulty': difficulty,
        'orders': orders_in_order,
    }
    if drop_off_zones:
        capture['drop_off_zones'] = drop_off_zones

    return capture, final_score, difficulty


WS_ACTION_TO_ACT = {
    'wait': ACT_WAIT,
    'move_up': ACT_MOVE_UP,
    'move_down': ACT_MOVE_DOWN,
    'move_left': ACT_MOVE_LEFT,
    'move_right': ACT_MOVE_RIGHT,
    'pick_up': ACT_PICKUP,
    'drop_off': ACT_DROPOFF,
}


def extract_game_actions(game_log_path, map_state):
    """Extract the action sequence from a game log as internal (act_type, item_idx) tuples.

    The game log alternates: game_state line, then action-response line.
    Action lines have {"actions": [{"bot": 0, "action": "move_up", ...}, ...]}.

    Returns list of round_actions (one per round), each is [(act, item_idx)] * num_bots,
    or None on failure.
    """
    # Build item_id -> index lookup from map_state
    item_id_to_idx = {}
    for idx, item in enumerate(map_state.items):
        item_id_to_idx[item['id']] = idx

    num_bots = None
    all_round_actions = []

    with open(game_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # JSONL log may contain non-JSON lines (stderr, partial writes)

            # Detect num_bots from first game_state
            if data.get('type') == 'game_state' and num_bots is None:
                num_bots = len(data.get('bots', []))

            # Action lines have 'actions' key but no 'type'
            if 'actions' not in data or 'type' in data:
                continue

            if num_bots is None:
                continue

            ws_actions = data['actions']
            # Build per-bot action list (indexed by bot id)
            round_acts = [(ACT_WAIT, -1)] * num_bots
            for wa in ws_actions:
                bid = wa.get('bot', 0)
                act_name = wa.get('action', 'wait')
                act_type = WS_ACTION_TO_ACT.get(act_name, ACT_WAIT)
                item_idx = -1
                if act_type == ACT_PICKUP:
                    item_id = wa.get('item_id', '')
                    item_idx = item_id_to_idx.get(item_id, -1)
                    if item_idx < 0:
                        act_type = ACT_WAIT  # unknown item -> wait
                if 0 <= bid < num_bots:
                    round_acts[bid] = (act_type, item_idx)
            all_round_actions.append(round_acts)

    if not all_round_actions:
        return None

    # Pad to 300 rounds if needed
    while len(all_round_actions) < 300:
        all_round_actions.append([(ACT_WAIT, -1)] * (num_bots or 1))

    return all_round_actions[:300]


def main():
    parser = argparse.ArgumentParser(
        description='Extract capture from Zig bot game_log')
    parser.add_argument('game_log', nargs='?',
                        help='Path to game_log_*.jsonl (or --latest)')
    parser.add_argument('difficulty', nargs='?',
                        choices=['easy', 'medium', 'hard', 'expert', 'nightmare'],
                        help='Difficulty (auto-detected if omitted)')
    parser.add_argument('--latest', action='store_true',
                        help='Use most recent game_log file')
    args = parser.parse_args()

    if args.latest or args.game_log is None:
        log_path = find_latest_game_log()
        if log_path is None:
            print("ERROR: No game_log_*.jsonl files found", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest: {log_path}", file=sys.stderr)
    else:
        log_path = args.game_log

    if not os.path.exists(log_path):
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    capture, score, difficulty = extract_capture(log_path, args.difficulty)

    merged, num_new, total = merge_capture(difficulty, capture)

    if num_new > 0:
        print(f"Merged {num_new} new orders (total: {total}, was: {total - num_new})", file=sys.stderr)
    else:
        print(f"No new orders (total: {total})", file=sys.stderr)

    # Extract game actions and save as warm-start solution
    if score > 0:
        try:
            map_state = build_map_from_capture(merged)
            game_actions = extract_game_actions(log_path, map_state)
            if game_actions:
                saved = save_solution(difficulty, score, game_actions)
                if saved:
                    print(f"Saved game actions as solution (score={score})", file=sys.stderr)
                else:
                    print(f"Existing solution is better (game score={score})", file=sys.stderr)
            else:
                print(f"No actions extracted from game log", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not extract game actions: {e}", file=sys.stderr)

    # Import game log to PostgreSQL in background
    try:
        import subprocess as _subprocess  # nosec B404
        _import_script = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'replay', 'import_logs.py',
        ))
        if os.path.exists(_import_script):
            _subprocess.Popen(  # nosec B603 B607
                ['python', _import_script, log_path, '--run-type', 'live'],
                stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
            )
    except Exception as e:
        print(f"Warning: background DB import failed: {e}", file=sys.stderr)

    # Output summary as JSON for pipeline consumption
    result = {
        'type': 'capture_done',
        'difficulty': difficulty,
        'orders': total,
        'new_orders': num_new,
        'items': len(merged['items']),
        'num_bots': merged['num_bots'],
        'grid': f"{merged['grid']['width']}x{merged['grid']['height']}",
        'probe_score': score,
        'storage': 'postgres',
    }
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

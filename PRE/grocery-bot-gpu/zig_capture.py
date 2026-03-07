"""Run Zig bot on live server and convert game log to capture format.

Usage:
    python zig_capture.py <wss://...token> <difficulty>
"""
import json
import os
import sys
import time

ZIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grocery-bot-zig')


def run_zig_bot(ws_url, difficulty):
    """Run Zig bot, return stdout lines (game log JSONL)."""
    exe = os.path.join(ZIG_DIR, 'zig-out', 'bin', f'grocery-bot-{difficulty}.exe')
    if not os.path.exists(exe):
        exe = os.path.join(ZIG_DIR, 'zig-out', 'bin', 'grocery-bot.exe')

    if not os.path.exists(exe):
        print(f"ERROR: No Zig bot executable found at {exe}", file=sys.stderr)
        return None

    print(f"Running Zig bot: {os.path.basename(exe)}", file=sys.stderr)

    from subprocess_helpers import run_bot_game

    return_code, stderr_output, stdout_output = run_bot_game(exe, ws_url, cwd=ZIG_DIR, timeout=180)

    print(stderr_output, file=sys.stderr)

    return stdout_output


def parse_game_log_lines(lines):
    """Parse Zig bot game log lines (strings) into capture format."""
    grid = None
    items = None
    drop_off = None
    num_bots = 0
    orders = []
    seen_order_ids = set()
    final_score = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get('type') == 'game_over':
            final_score = data.get('score', 0)
            continue

        if data.get('type') != 'game_state':
            continue

        rnd = data.get('round', -1)

        if rnd == 0:
            grid = data.get('grid')
            items = data.get('items')
            drop_off = data.get('drop_off')
            num_bots = len(data.get('bots', []))

        for order in data.get('orders', []):
            oid = order.get('id', '')
            if oid and oid not in seen_order_ids:
                seen_order_ids.add(oid)
                orders.append({
                    'id': oid,
                    'items_required': list(order['items_required']),
                    'items_delivered': [],
                    'status': 'future',
                })

    if orders:
        orders[0]['status'] = 'active'
        if len(orders) > 1:
            orders[1]['status'] = 'preview'

    capture = {
        'difficulty': None,
        'grid': grid,
        'items': items,
        'drop_off': drop_off,
        'num_bots': num_bots,
        'orders': orders,
        'probe_score': final_score,
    }
    return capture


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python zig_capture.py <wss://...> <difficulty>")
        sys.exit(1)

    ws_url = sys.argv[1]
    difficulty = sys.argv[2]

    stdout_output = run_zig_bot(ws_url, difficulty)
    if not stdout_output:
        print("ERROR: No game log produced", file=sys.stderr)
        sys.exit(1)

    log_lines = stdout_output.splitlines()
    print(f"Parsing game log: {len(log_lines)} lines from stdout", file=sys.stderr)
    capture = parse_game_log_lines(log_lines)
    capture['difficulty'] = difficulty

    print(f"Score: {capture['probe_score']}", file=sys.stderr)
    print(f"Orders captured: {len(capture['orders'])}", file=sys.stderr)

    # Merge with existing capture
    from solution_store import save_capture, load_capture
    existing = load_capture(difficulty)
    if existing:
        existing_ids = set(o.get('id', '') for o in existing.get('orders', []))
        new_count = 0
        for o in capture['orders']:
            if o['id'] not in existing_ids:
                existing['orders'].append(o)
                existing_ids.add(o['id'])
                new_count += 1
        existing['probe_score'] = max(existing.get('probe_score', 0), capture['probe_score'])
        save_capture(difficulty, existing)
        print(f"Merged: +{new_count} new orders ({len(existing['orders'])} total)",
              file=sys.stderr)
    else:
        save_capture(difficulty, capture)
        print(f"Capture saved to DB ({difficulty})", file=sys.stderr)

    # Import directly to PostgreSQL
    try:
        _replay_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'replay'))
        sys.path.insert(0, _replay_dir)
        from import_logs import parse_log_lines, save_to_db
        record = parse_log_lines(log_lines, pseudo_seed=int(time.time()))
        if record:
            run_id = save_to_db(
                os.environ.get("GROCERY_DB_URL",
                               "postgres://grocery:grocery123@localhost:5433/grocery_bot"),
                record, run_type='live')
            print(f"  [db] Saved to PostgreSQL run_id={run_id}", file=sys.stderr)
    except Exception as e:
        print(f"  [db] Direct DB import failed: {e}", file=sys.stderr)

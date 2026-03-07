"""Run Zig bot on live server and convert game log to capture format.

Usage:
    python zig_capture.py <wss://...token> <difficulty>
"""
import json
import os
import sys

ZIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'grocery-bot-zig')


def run_zig_bot(ws_url, difficulty):
    """Run Zig bot, return path to game log."""
    exe = os.path.join(ZIG_DIR, 'zig-out', 'bin', f'grocery-bot-{difficulty}.exe')
    if not os.path.exists(exe):
        exe = os.path.join(ZIG_DIR, 'zig-out', 'bin', 'grocery-bot.exe')

    if not os.path.exists(exe):
        print(f"ERROR: No Zig bot executable found at {exe}", file=sys.stderr)
        return None

    print(f"Running Zig bot: {os.path.basename(exe)}", file=sys.stderr)

    from subprocess_helpers import run_bot_game

    return_code, stderr_output, log_path = run_bot_game(exe, ws_url, cwd=ZIG_DIR, timeout=180)

    print(stderr_output, file=sys.stderr)

    return log_path


def parse_game_log(log_path):
    """Parse Zig bot game log into capture format."""
    grid = None
    items = None
    drop_off = None
    num_bots = 0
    orders = []
    seen_order_ids = set()
    final_score = 0

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # JSONL log may contain non-JSON lines (stderr, partial writes)

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

            # Capture orders
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
        'difficulty': None,  # Set by caller
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

    log_path = run_zig_bot(ws_url, difficulty)
    if not log_path:
        print("ERROR: No game log produced", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing game log: {log_path}", file=sys.stderr)
    capture = parse_game_log(log_path)
    capture['difficulty'] = difficulty

    print(f"Score: {capture['probe_score']}", file=sys.stderr)
    print(f"Orders captured: {len(capture['orders'])}", file=sys.stderr)

    # Merge with existing capture (don't lose previously captured orders)
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
        # Keep existing grid/items/drop_off (first capture is authoritative)
        existing['probe_score'] = max(existing.get('probe_score', 0), capture['probe_score'])
        save_capture(difficulty, existing)
        print(f"Merged: +{new_count} new orders ({len(existing['orders'])} total)",
              file=sys.stderr)
    else:
        save_capture(difficulty, capture)
        print(f"Capture saved to DB ({difficulty})", file=sys.stderr)

    # Auto-import to PostgreSQL as a 'live' run
    try:
        import subprocess  # nosec B404
        import_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'grocery-bot-zig', 'replay', 'import_logs.py')
        result = subprocess.run(  # nosec B603 B607
            [sys.executable, import_script, '--run-type', 'live', log_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  DB import: {result.stdout.strip()}", file=sys.stderr)
        else:
            print(f"  DB import failed: {result.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"  DB import error: {e}", file=sys.stderr)

"""Reactive GPU replay: plays pre-computed actions, re-solves when encountering unknown orders.

Two-game approach:
  Game 1 (probe): Already done via capture_and_solve_stream.py (captures orders + GPU DP solves)
  Game 2 (this): Replays optimal actions. When server shows an order not in the plan,
                 re-solves remaining rounds with GPU DP as a subprocess.

Uses file-based IPC: worker writes result JSON when done, main loop polls file existence.

Usage:
    python reactive_replay.py <ws_url> <difficulty>
"""
import asyncio
import json
import os
import sys
import time
import subprocess  # nosec B404
import tempfile

from game_engine import (
    build_map_from_capture,
    actions_to_ws_format, MAX_ROUNDS,
)
from solution_store import load_capture, load_solution, load_meta


def orders_match(a, b):
    """Check if two orders have the same items_required (order-insensitive)."""
    return sorted(a) == sorted(b)


async def reactive_replay(ws_url, difficulty, verbose=True):
    """Replay saved solution with reactive GPU re-solving on unknown orders."""
    import websockets

    capture = load_capture(difficulty)
    if not capture:
        print(f"No capture found for {difficulty}", file=sys.stderr)
        return 0

    planned_actions = load_solution(difficulty)
    if not planned_actions:
        print(f"No solution found for {difficulty}", file=sys.stderr)
        return 0

    ms = build_map_from_capture(capture)
    meta = load_meta(difficulty)
    expected_score = meta.get('score', 0) if meta else 0

    captured_orders = list(capture['orders'])
    num_captured = len(captured_orders)

    if verbose:
        print(f"Loaded: {difficulty}, {num_captured} captured orders, "
              f"expected score={expected_score}", file=sys.stderr)

    current_plan = planned_actions
    seen_order_items = [o['items_required'] for o in captured_orders]

    # Re-solve state — uses Popen (non-blocking) + file-based result
    solve_proc = None
    solve_result_path = os.path.join(tempfile.gettempdir(),
                                     f'reactive_result_{difficulty}.json')
    solve_capture_path = os.path.join(tempfile.gettempdir(),
                                      f'reactive_capture_{difficulty}.json')
    re_solve_count = 0

    # Delete stale result file
    try:
        os.remove(solve_result_path)
    except FileNotFoundError:
        pass  # No stale result file to clean up

    def check_solve_done():
        """Check if solver subprocess finished by testing result file existence."""
        nonlocal solve_proc
        if solve_proc is None:
            return None

        # Check result file directly — more reliable than poll() on Windows
        if not os.path.exists(solve_result_path):
            return None

        # Result file exists — worker is done. Read it.
        try:
            with open(solve_result_path) as f:
                result = json.load(f)
            # Wait for process to fully exit
            try:
                solve_proc.wait(timeout=1)
            except Exception:
                pass  # Process may have already exited; proceed to cleanup
            solve_proc = None
            # Remove result file so we don't re-read it
            try:
                os.remove(solve_result_path)
            except Exception:
                pass  # Best-effort cleanup; file may already be removed
            return [[(a, i) for a, i in r] for r in result]
        except (json.JSONDecodeError, ValueError):
            return None  # File exists but still being written
        except Exception as e:
            if verbose:
                print(f"  Result read error: {e}", file=sys.stderr)
            return None

    def start_solve(all_orders):
        """Launch GPU DP solver as subprocess."""
        nonlocal solve_proc, re_solve_count

        # Kill existing
        if solve_proc is not None and solve_proc.poll() is None:
            solve_proc.kill()
            solve_proc = None

        re_solve_count += 1

        # Delete old result
        try:
            os.remove(solve_result_path)
        except FileNotFoundError:
            pass  # No old result file to clean up

        # Write updated capture
        updated = dict(capture)
        updated['orders'] = all_orders
        with open(solve_capture_path, 'w') as f:
            json.dump(updated, f)

        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_reactive_solve_worker.py')

        solve_proc = subprocess.Popen(  # nosec B603 B607
            [sys.executable, worker, solve_capture_path, difficulty, solve_result_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,  # Worker logs to file, not pipes
        )

        if verbose:
            print(f"  Re-solve #{re_solve_count} started (PID {solve_proc.pid})",
                  file=sys.stderr)

    # Game log
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'game_log_{int(time.time())}.jsonl')
    log_file = open(log_path, 'w')
    final_score = 0

    if verbose:
        print(f"Connecting...", file=sys.stderr)

    async with websockets.connect(ws_url) as ws:
        for round_num in range(MAX_ROUNDS + 1):
            msg = await ws.recv()
            data = json.loads(msg)
            log_file.write(json.dumps(data) + '\n')
            log_file.flush()

            if data.get("type") == "game_over":
                final_score = data.get("score", 0)
                if verbose:
                    print(f"\nGAME_OVER Score:{final_score} "
                          f"(expected={expected_score}, re-solves={re_solve_count})",
                          file=sys.stderr)
                break

            if data.get("type") != "game_state":
                continue

            rnd = data["round"]
            score = data.get("score", 0)

            # Detect new orders
            new_order = False
            for order in data.get("orders", []):
                items = order.get("items_required", [])
                if not any(orders_match(items, k) for k in seen_order_items):
                    new_order = True
                    seen_order_items.append(items)
                    if verbose:
                        print(f"\n  R{rnd}: NEW ORDER! items={items} "
                              f"({len(seen_order_items)} total)", file=sys.stderr)

            # Check solve completion (non-blocking)
            new_plan = check_solve_done()
            if new_plan is not None:
                current_plan = new_plan
                if verbose:
                    print(f"  R{rnd}: SWITCHED to re-solved plan!", file=sys.stderr)
            elif solve_proc is not None and rnd % 10 == 0:
                res_exists = os.path.exists(solve_result_path)
                if verbose and res_exists:
                    print(f"  R{rnd}: result file ready!", file=sys.stderr)

            # Start re-solve on new orders
            if new_order and (solve_proc is None or solve_proc.poll() is not None):
                all_orders = [{'id': f'o{i}', 'items_required': items}
                              for i, items in enumerate(seen_order_items)]
                start_solve(all_orders)

            # Send action
            if 0 <= rnd < len(current_plan):
                ws_actions = actions_to_ws_format(current_plan[rnd], ms)
            else:
                ws_actions = [{"bot": b['id'], "action": "wait"}
                              for b in data['bots']]

            response = {"actions": ws_actions}
            log_file.write(json.dumps(response) + '\n')
            log_file.flush()
            await ws.send(json.dumps(response))

            if verbose and (rnd < 10 or rnd % 25 == 0 or rnd >= 295):
                solving = solve_proc is not None and solve_proc.poll() is None
                print(f"  R{rnd:3d}: score={score:3d} "
                      f"[{'re-solving' if solving else 'plan'}]", file=sys.stderr)

    log_file.close()

    # Cleanup
    if solve_proc is not None and solve_proc.poll() is None:
        solve_proc.kill()

    if verbose:
        print(f"Log: {log_path}", file=sys.stderr)
    if final_score > expected_score:
        print(f"NEW BEST: {final_score} (was {expected_score})", file=sys.stderr)

    return final_score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Reactive GPU replay (re-solves on unknown orders)')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    args = parser.parse_args()

    score = asyncio.run(reactive_replay(args.ws_url, args.difficulty))
    print(f"Final score: {score}")

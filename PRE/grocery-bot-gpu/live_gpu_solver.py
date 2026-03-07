"""Live GPU DP solver — single-connection capture + solve + replay.

Connects to WebSocket, captures round 0 data, runs GPU DP search
(~3.4s for Easy), then replays the optimal actions. Waits during
the first few rounds while DP computes.

Usage:
    python live_gpu_solver.py <wss://...token> [--difficulty auto]
"""
import argparse
import asyncio
import json
import os
import sys
import time
import threading

import torch

from game_engine import (
    init_game_from_capture, build_map_from_capture,
    actions_to_ws_format, ACT_WAIT,
)
from gpu_beam_search import GPUBeamSearcher
from configs import detect_difficulty
from live_solver import ws_to_capture


async def live_gpu_dp(ws_url, difficulty=None, log_dir=None):
    """Connect, capture, GPU DP solve, and replay in one connection."""
    import websockets

    log_buffer = []  # in-memory log buffer

    # Shared state between main loop and solver thread
    dp_result = [None]  # (score, actions) when done
    solve_error = [None]

    print(f"Connecting to server...", file=sys.stderr)

    async with websockets.connect(ws_url) as ws:
        solver_thread = None
        map_state = None
        rounds_waited = 0
        final_score = 0
        dp_actions = None
        dp_score = None
        t_start = None

        async for message in ws:
            data = json.loads(message)
            log_buffer.append(data)

            if data["type"] == "game_over":
                final_score = data.get('score', 0)
                print(f"\nGAME_OVER Score:{final_score}", file=sys.stderr)
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]
            score = data.get("score", 0)
            max_rounds = data.get("max_rounds", 300)

            if rnd == 0:
                t_start = time.time()
                # Detect difficulty
                num_bots = len(data['bots'])
                if difficulty is None or difficulty == 'auto':
                    difficulty = detect_difficulty(num_bots)

                print(f"  Difficulty: {difficulty}, bots: {num_bots}", file=sys.stderr)

                if num_bots > 1:
                    print(f"  WARNING: GPU DP only supports single-bot (Easy).",
                          file=sys.stderr)
                    print(f"  Falling back to wait actions.", file=sys.stderr)

                # Capture game data
                capture = ws_to_capture(data)
                gs, all_orders = init_game_from_capture(capture)
                map_state = gs.map_state

                # Start GPU DP in background thread
                def solve():
                    try:
                        searcher = GPUBeamSearcher(
                            gs.map_state, all_orders, device='cuda')
                        dp_result[0] = searcher.dp_search(
                            gs, max_states=500000, verbose=True)
                    except Exception as e:
                        solve_error[0] = e
                        print(f"  SOLVE ERROR: {e}", file=sys.stderr)

                solver_thread = threading.Thread(target=solve, daemon=True)
                solver_thread.start()

                # Respond with wait immediately (while DP computes)
                response = {"actions": [
                    {"bot": b['id'], "action": "wait"} for b in data['bots']
                ]}
                rounds_waited += 1
                log_buffer.append(response)
                await ws.send(json.dumps(response))

                print(f"  R{rnd:3d}: wait (DP computing...)", file=sys.stderr)
                continue

            # Check if DP is done
            if dp_actions is None and dp_result[0] is not None:
                dp_score, dp_actions = dp_result[0]
                dt = time.time() - t_start
                print(f"\n  DP READY at round {rnd}: optimal_score={dp_score}, "
                      f"solve_time={dt:.1f}s, waited={rounds_waited} rounds",
                      file=sys.stderr)

            if dp_actions is not None and map_state is not None:
                # Use DP action, offset by rounds waited
                dp_round = rnd - rounds_waited
                if 0 <= dp_round < len(dp_actions):
                    ws_actions = actions_to_ws_format(
                        dp_actions[dp_round], map_state)
                else:
                    # Past the end of DP plan, wait
                    ws_actions = [
                        {"bot": b['id'], "action": "wait"}
                        for b in data['bots']
                    ]
            else:
                # Still computing, send wait
                ws_actions = [
                    {"bot": b['id'], "action": "wait"} for b in data['bots']
                ]
                rounds_waited += 1

            response = {"actions": ws_actions}
            log_file.write(json.dumps(response) + '\n')
            log_file.flush()
            await ws.send(json.dumps(response))

            if rnd % 25 == 0 or rnd == max_rounds - 1:
                status = "DP" if dp_actions is not None else "wait"
                print(f"  R{rnd:3d}: score={score:3d} [{status}]", file=sys.stderr)

        # Wait for solver thread to finish
        if solver_thread and solver_thread.is_alive():
            solver_thread.join(timeout=5)

    print(f"\nGame log: {len(log_buffer)} entries in memory", file=sys.stderr)

    # Import directly to PostgreSQL
    try:
        _replay_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'replay'))
        sys.path.insert(0, _replay_dir)
        from import_logs import parse_log_lines, save_to_db
        timestamp = int(time.time())
        record = parse_log_lines(log_buffer, pseudo_seed=timestamp)
        if record:
            run_id = save_to_db(
                os.environ.get("GROCERY_DB_URL",
                               "postgres://grocery:grocery123@localhost:5433/grocery_bot"),
                record, run_type='live')
            print(f"  [db] Saved to PostgreSQL run_id={run_id}", file=sys.stderr)
    except Exception as _e:
        print(f"  [db] Direct DB import failed: {_e}", file=sys.stderr)
    if dp_score:
        print(f"DP optimal: {dp_score}, Live: {final_score}, "
              f"Lost rounds: {rounds_waited}", file=sys.stderr)
    return final_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Live GPU DP solver (capture + solve + replay)')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('--difficulty', default='auto',
                        help='Difficulty (auto-detects from bot count)')
    parser.add_argument('--log-dir', default=None, help='Log directory')
    args = parser.parse_args()

    diff = args.difficulty if args.difficulty != 'auto' else None
    asyncio.run(live_gpu_dp(args.ws_url, difficulty=diff, log_dir=args.log_dir))

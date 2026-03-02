"""Full competition pipeline: capture -> solve -> replay.

Usage:
    python pipeline.py <difficulty> <ws_url_probe> <ws_url_replay> [--time <solve_seconds>]

Steps:
    1. Check if today's capture exists for this difficulty (skip probe if so)
    2. If not, play a probe game to capture item positions and orders
    3. Solve offline using captured state (multi-strategy + optimizer)
    4. Replay optimized solution on the server for max score

The capture is saved by date, so subsequent runs today reuse the same capture.
"""
import asyncio
import sys
import time
import json
from datetime import datetime, timezone

from capture_game import capture_and_play, load_capture, get_capture_path
from solve_from_capture import solve_from_capture
from ws_client import replay, load_actions
from game_engine import build_map_from_capture


async def run_pipeline(difficulty, ws_url_probe, ws_url_replay, solve_time=300.0):
    """Run the full capture -> solve -> replay pipeline."""
    t0 = time.time()
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    print(f"{'='*60}")
    print(f"PIPELINE: {difficulty} ({today})")
    print(f"{'='*60}\n")

    # Phase 1: Capture (or load existing)
    print(f"--- Phase 1: Capture ---")
    capture = load_capture(difficulty)
    if capture:
        print(f"Reusing today's capture ({get_capture_path(difficulty)})")
        print(f"  Items: {len(capture['items'])}")
        print(f"  Orders: {len(capture['orders'])}")
        print(f"  Probe score: {capture.get('probe_score', 'N/A')}")
    else:
        print(f"No capture for today, running probe game...")
        capture = await capture_and_play(ws_url_probe, difficulty)
        print(f"  Waiting 10s cooldown...")
        await asyncio.sleep(10)

    print(f"\n--- Phase 2: Solve ({solve_time:.0f}s) ---")
    score, solution_file = solve_from_capture(capture, time_limit=solve_time)

    print(f"\n--- Phase 3: Replay ---")
    actions = load_actions(solution_file)
    map_state = build_map_from_capture(capture)

    replay_score = await replay(ws_url_replay, actions, map_state, verbose=True)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {difficulty}")
    print(f"  Probe score:    {capture.get('probe_score', 0)}")
    print(f"  Offline score:  {score}")
    print(f"  Replay score:   {replay_score}")
    print(f"  Total time:     {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    return replay_score


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <difficulty> <ws_url_probe> <ws_url_replay> [--time <secs>]")
        print()
        print("Example:")
        print("  python pipeline.py hard 'wss://game.ainm.no/ws?token=PROBE' 'wss://game.ainm.no/ws?token=REPLAY'")
        print()
        print("For separate steps:")
        print("  python capture_game.py <ws_url> <difficulty>")
        print("  python solve_from_capture.py <difficulty> --time 300")
        print("  python ws_client.py <ws_url> solution_<diff>_captured.json --capture captures/<diff>_<date>.json")
        sys.exit(1)

    difficulty = sys.argv[1]
    ws_probe = sys.argv[2]
    ws_replay = sys.argv[3]
    solve_time = 300.0

    for i, arg in enumerate(sys.argv):
        if arg == '--time' and i + 1 < len(sys.argv):
            solve_time = float(sys.argv[i + 1])

    asyncio.run(run_pipeline(difficulty, ws_probe, ws_replay, solve_time))

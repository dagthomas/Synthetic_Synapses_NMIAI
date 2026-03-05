"""Verify game_engine.py matches sim_server.py exactly.

Runs the beam search solver's action log through both the in-process engine
and the sim_server WebSocket, comparing scores.
"""
import asyncio
import json
import sys
import subprocess  # nosec B404
import time
import os

from game_engine import (
    build_map, init_game, step, simulate_game,
    actions_to_ws_format, ACT_WAIT,
)
from beam_search import beam_search
from pathfinding import precompute_all_distances


async def verify_vs_sim_server(seed, difficulty, action_log, port=9990):
    """Run action_log through sim_server and compare scores."""
    import websockets

    ms = build_map(difficulty)
    ws_url = f"ws://localhost:{port}"

    # Start sim_server
    sim_dir = os.path.join(os.path.dirname(__file__), '..', 'grocery-bot-zig')
    sim_script = os.path.join(sim_dir, 'sim_server.py')
    proc = subprocess.Popen(  # nosec B603 B607
        [sys.executable, sim_script, str(port), difficulty, '--seed', str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    await asyncio.sleep(1.0)  # wait for server to start

    try:
        async with websockets.connect(ws_url, ping_interval=None, close_timeout=5) as ws:
            for rnd in range(len(action_log)):
                msg = await ws.recv()
                data = json.loads(msg)

                if data.get("type") == "game_over":
                    return data.get("score", 0), rnd

                # Send actions
                ws_actions = actions_to_ws_format(action_log[rnd], ms)
                await ws.send(json.dumps({"actions": ws_actions}))

            # Final game_over
            msg = await ws.recv()
            data = json.loads(msg)
            return data.get("score", 0), len(action_log)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def verify_engine_standalone(seed, difficulty):
    """Run a simple greedy game through the engine and print score."""
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    dist_maps = precompute_all_distances(ms)

    from action_gen import generate_joint_actions
    from beam_search import eval_state

    for rnd in range(300):
        joint_actions = generate_joint_actions(
            state, dist_maps, all_orders, max_per_bot=3, max_joint=500
        )
        best_ev = -999999999
        best_state = None
        for actions in joint_actions:
            new_state = state.copy()
            new_state.round = rnd
            step(new_state, actions, all_orders)
            ev = eval_state(new_state, all_orders, dist_maps)
            if ev > best_ev:
                best_ev = ev
                best_state = new_state
        state = best_state

    return state.score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', '-d', default='easy')
    parser.add_argument('--seed', '-s', type=int, default=7001)
    parser.add_argument('--beam-width', '-b', type=int, default=1)
    parser.add_argument('--verify-ws', action='store_true',
                        help='Also verify against sim_server.py via WebSocket')
    parser.add_argument('--port', type=int, default=9990)
    args = parser.parse_args()

    print(f"Verifying: {args.difficulty} seed={args.seed} beam_width={args.beam_width}")

    # Run beam search
    score, action_log, stats = beam_search(
        args.seed, args.difficulty, beam_width=args.beam_width, verbose=True
    )
    print(f"\nEngine score: {score}")

    if args.verify_ws:
        print("\nVerifying against sim_server.py...")
        ws_score, ws_rounds = asyncio.run(
            verify_vs_sim_server(args.seed, args.difficulty, action_log, args.port)
        )
        print(f"Sim server score: {ws_score} (rounds: {ws_rounds})")
        if ws_score == score:
            print("PASS: Scores match!")
        else:
            print(f"FAIL: Engine={score} vs SimServer={ws_score}")


if __name__ == '__main__':
    main()

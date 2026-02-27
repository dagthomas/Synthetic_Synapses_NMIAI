#!/usr/bin/env python3
"""Grocery Bot - Python implementation (optimized)."""
from __future__ import annotations

import asyncio
import json
import sys

from parser import GameState, parse_game_state
import strategy


def run_replay(log_path: str):
    print(f"=== REPLAY MODE: {log_path} ===", file=sys.stderr)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Cannot open {log_path}: {e}", file=sys.stderr)
        return

    state = GameState()
    total_rounds = 0
    total_waits = 0
    total_moves = 0
    total_pickups = 0
    total_dropoffs = 0
    final_score = 0
    last_score = 0
    score_gain_rounds = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('{"actions":'):
            continue

        try:
            is_running = parse_game_state(line, state)
        except Exception as e:
            print(f"Parse error: {e}", file=sys.stderr)
            continue
        if not is_running:
            break

        final_score = state.score
        if state.score > last_score:
            score_gain_rounds += 1
            last_score = state.score
        total_rounds += 1

        if state.round == 0:
            strategy.init_pbots()
            strategy.expected_count = 0

        try:
            response = strategy.decide_actions(state)
        except Exception as e:
            print(f"R{state.round} Decision error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

        # Count actions
        for act in json.loads(response).get("actions", []):
            a = act.get("action", "")
            if a == "wait":
                total_waits += 1
            elif a.startswith("move_"):
                total_moves += 1
            elif a == "pick_up":
                total_pickups += 1
            elif a == "drop_off":
                total_dropoffs += 1

    print(f"\n=== REPLAY RESULTS ===", file=sys.stderr)
    print(f"Log file score: {final_score}", file=sys.stderr)
    print(f"Rounds processed: {total_rounds}", file=sys.stderr)
    print(f"Rounds where score increased: {score_gain_rounds}", file=sys.stderr)
    total_actions = total_moves + total_pickups + total_dropoffs + total_waits
    print(f"Bot-actions: {total_moves} moves, {total_pickups} pickups, {total_dropoffs} dropoffs, {total_waits} waits", file=sys.stderr)
    if total_actions > 0:
        pct = (total_moves + total_pickups + total_dropoffs) * 100 // total_actions
        print(f"Useful action rate: {pct}%", file=sys.stderr)
    print(f"======================", file=sys.stderr)


async def run_live(url: str):
    try:
        import websockets
    except ImportError:
        print("Error: pip install websockets", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {url}", file=sys.stderr)
    state = GameState()
    last_responded_round = -1

    try:
        log_file = open("game_log.jsonl", "w", encoding="utf-8")
    except OSError:
        log_file = None

    try:
        async with websockets.connect(url, max_size=2**20) as ws:
            async for message in ws:
                data = message if isinstance(message, str) else message.decode("utf-8")

                if log_file:
                    log_file.write(data)
                    log_file.write("\n")

                try:
                    is_running = parse_game_state(data, state)
                except Exception as e:
                    print(f"Parse error: {e}", file=sys.stderr)
                    continue
                if not is_running:
                    break

                if state.round == 0:
                    strategy.init_pbots()
                    strategy.expected_count = 0

                if state.round % 50 == 0:
                    print(f"R{state.round}/{state.max_rounds} Score:{state.score}", file=sys.stderr)

                # Detect round gap
                current_round = state.round
                if last_responded_round >= 0 and current_round > last_responded_round + 1:
                    print(f"R{state.round} DESYNC: skip to re-sync", file=sys.stderr)
                    try:
                        strategy.decide_actions(state)
                    except Exception:
                        pass
                    last_responded_round = current_round
                    continue

                try:
                    response = strategy.decide_actions(state)
                except Exception as e:
                    print(f"Decision error: {e}", file=sys.stderr)
                    continue

                if log_file:
                    log_file.write(response)
                    log_file.write("\n")

                await ws.send(response)
                last_responded_round = current_round

    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
    finally:
        if log_file:
            log_file.close()


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage:\n  python main.py <ws://...>    Live\n  python main.py --replay <log>  Replay", file=sys.stderr)
        sys.exit(1)

    if args[0] == "--replay":
        if len(args) < 2:
            sys.exit(1)
        run_replay(args[1])
    else:
        asyncio.run(run_live(args[0]))


if __name__ == "__main__":
    main()

"""GPU DP solver with JSON streaming output for the Matrix dashboard.

Outputs JSON lines to stdout for SSE consumption by SvelteKit.
Runs GPU DP search and saves improved solution.

Usage:
    python gpu_solve_stream.py <difficulty> [--seed SEED]
"""
import json
import sys
import time
import argparse

import torch

from game_engine import ACT_WAIT


def emit(data):
    """Output a JSON line to stdout for SSE streaming."""
    print(json.dumps(data), flush=True)


def solve(difficulty, seed=None):
    """Run GPU DP solver with streaming progress output."""
    from solution_store import load_capture, load_meta, save_solution
    from game_engine import init_game, init_game_from_capture
    from gpu_beam_search import GPUBeamSearcher

    t0 = time.time()

    # Load capture or create game
    capture = load_capture(difficulty)
    if capture:
        gs, all_orders = init_game_from_capture(capture)
        emit({"type": "source", "source": "capture"})
    elif seed:
        gs, all_orders = init_game(seed, difficulty)
        emit({"type": "source", "source": "seed", "seed": seed})
    else:
        emit({"type": "error", "msg": "No capture found and no seed provided"})
        return

    ms = gs.map_state
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    emit({
        "type": "init",
        "difficulty": difficulty,
        "items": ms.num_items,
        "types": ms.num_types,
        "width": ms.width,
        "height": ms.height,
        "cells": int((ms.grid == 0).sum() + (ms.grid == 3).sum()),
        "orders": len(all_orders),
        "device": device,
        "vram_total": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if device == 'cuda' else 0,
        "gpu_name": torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A',
    })

    # Initialize searcher
    num_bots = len(gs.bot_positions)
    searcher = GPUBeamSearcher(ms, all_orders, device=device, num_bots=num_bots)
    emit({"type": "gpu_ready", "time": round(time.time() - t0, 3),
          "actions_per_state": searcher.num_actions})

    # Verify
    ok = searcher.verify_against_cpu(gs.copy(), all_orders, num_rounds=100)
    emit({"type": "verify", "ok": ok,
          "time": round(time.time() - t0, 3)})

    if not ok:
        emit({"type": "error", "msg": "GPU verification failed!"})
        return

    # Get previous best
    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    emit({"type": "prev_best", "score": prev_score})

    # Progress callback
    def on_round(rnd, best_score, unique, expanded, elapsed):
        emit({
            "type": "round",
            "r": rnd,
            "score": best_score,
            "unique": unique,
            "expanded": expanded,
            "time": round(elapsed, 3),
        })

    # Run GPU DP search
    emit({"type": "solving", "msg": "Starting GPU DP search..."})
    score, bot_actions = searcher.dp_search(
        gs.copy(), max_states=500000, verbose=False, on_round=on_round)

    # Wrap single-bot actions into per-round multi-bot format
    wait_pad = [(ACT_WAIT, -1)] * (num_bots - 1)
    actions = [[(a, i)] + wait_pad for a, i in bot_actions]

    total_time = time.time() - t0

    emit({
        "type": "result",
        "score": score,
        "time": round(total_time, 3),
        "rounds": len(actions),
        "optimal": True,  # No pruning needed means provably optimal
    })

    # Save if improved
    if score > prev_score:
        save_solution(difficulty, score, actions)
        meta = load_meta(difficulty)
        emit({
            "type": "improved",
            "old_score": prev_score,
            "new_score": score,
            "delta": score - prev_score,
        })
    else:
        emit({
            "type": "no_improvement",
            "score": score,
            "prev": prev_score,
        })

    emit({"type": "done", "score": score, "time": round(total_time, 3)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU DP solver (streaming JSON)')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    solve(args.difficulty, seed=args.seed)

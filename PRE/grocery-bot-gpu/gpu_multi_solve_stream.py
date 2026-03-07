"""Sequential per-bot GPU DP solver with JSON streaming output for dashboard.

Outputs JSON lines to stdout for SSE consumption by SvelteKit.
Handles both single-bot (Easy) and multi-bot (Medium/Hard/Expert).

Usage:
    python gpu_multi_solve_stream.py <difficulty> [--seed SEED] [--max-states N]
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


def solve(difficulty, seed=None, max_states=100000):
    """Run sequential per-bot GPU DP with streaming progress output."""
    # Disable Zig FFI - causes illegal instruction crash on this machine
    import gpu_sequential_solver
    gpu_sequential_solver._ZIG_AVAILABLE = False

    from solution_store import load_capture, load_meta, save_solution
    from game_engine import init_game, init_game_from_capture
    from gpu_sequential_solver import solve_sequential
    from configs import CONFIGS

    t0 = time.time()

    # Load capture or create game
    capture = load_capture(difficulty)
    if capture:
        emit({"type": "source", "source": "capture"})
    elif seed:
        emit({"type": "source", "source": "seed", "seed": seed})
    else:
        emit({"type": "error", "msg": "No capture found and no seed provided"})
        return

    # Get game info
    if capture:
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture)
        num_bots = capture.get('num_bots', CONFIGS[difficulty]['bots'])
    else:
        from game_engine import build_map
        ms = build_map(difficulty)
        num_bots = CONFIGS[difficulty]['bots']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import numpy as np
    walkable = int((ms.grid == 0).sum() + (ms.grid == 3).sum())

    emit({
        "type": "init",
        "difficulty": difficulty,
        "items": ms.num_items,
        "types": ms.num_types,
        "width": ms.width,
        "height": ms.height,
        "num_bots": num_bots,
        "cells": walkable,
        "device": device,
        "vram_total": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if device == 'cuda' else 0,
        "gpu_name": torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A',
    })

    # Get previous best
    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    emit({"type": "prev_best", "score": prev_score})

    # Progress callbacks
    def on_bot_progress(bot_id, num_bots, score, elapsed):
        emit({
            "type": "bot_done",
            "bot_id": bot_id,
            "total_bots": num_bots,
            "score": score,
            "time": round(elapsed, 3),
        })

    def on_round(bot_id, rnd, score, unique, expanded, elapsed):
        # Emit bot_start on first round
        if rnd == 0:
            emit({
                "type": "bot_start",
                "bot_id": bot_id,
                "total_bots": num_bots,
            })
        emit({
            "type": "round",
            "bot_id": bot_id,
            "r": rnd,
            "score": score,
            "unique": unique,
            "expanded": expanded,
            "time": round(elapsed, 3),
        })

    def on_phase(phase_name, iteration, cpu_score):
        emit({
            "type": "phase",
            "phase": phase_name,
            "iteration": iteration,
            "cpu_score": cpu_score,
        })

    # Run solver
    emit({"type": "solving", "msg": f"Starting sequential GPU DP ({num_bots} bots)..."})

    score, actions = solve_sequential(
        capture_data=capture,
        seed=seed,
        difficulty=difficulty,
        device=device,
        max_states=max_states,
        verbose=False,
        on_bot_progress=on_bot_progress,
        on_round=on_round,
        on_phase=on_phase,
    )

    total_time = time.time() - t0

    emit({
        "type": "result",
        "score": score,
        "time": round(total_time, 3),
        "rounds": len(actions),
        "num_bots": num_bots,
        "optimal": num_bots == 1,
    })

    # Save if improved
    if score > prev_score:
        save_solution(difficulty, score, actions)
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
    parser = argparse.ArgumentParser(description='Sequential GPU DP solver (streaming JSON)')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-states', type=int, default=100000)
    args = parser.parse_args()

    solve(args.difficulty, seed=args.seed, max_states=args.max_states)

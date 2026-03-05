"""DEPRECATED: Main entry point: solve one seed at one difficulty.

This module is not part of the active production pipeline. Kept for reference.

Usage:
    python solve.py easy 7001 --beam-width 100
    python solve.py expert 7001 --beam-width 1000 --replay ws://localhost:9880
"""
import argparse
import asyncio
import json
import os
import time

from game_engine import build_map, init_game, simulate_game, MAX_ROUNDS
from beam_search import beam_search, greedy_search
from ws_client import replay, save_actions, load_actions


def main():
    parser = argparse.ArgumentParser(description='Grocery Bot GPU Beam Search Solver')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('seed', type=int)
    parser.add_argument('--beam-width', type=int, default=100, help='Beam width (default 100)')
    parser.add_argument('--max-per-bot', type=int, default=3, help='Max candidates per bot')
    parser.add_argument('--max-joint', type=int, default=500, help='Max joint actions per state')
    parser.add_argument('--replay', type=str, default=None,
                        help='WebSocket URL to replay solution (e.g., ws://localhost:9850)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save actions to JSON file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load actions from JSON file (skip solving)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy search (beam_width=1)')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.load:
        # Load pre-computed actions
        actions = load_actions(args.load)
        score = None
        if verbose:
            print(f"Loaded {len(actions)} rounds from {args.load}")
    else:
        # Solve
        if args.greedy:
            score, actions, stats = greedy_search(args.seed, args.difficulty, verbose=verbose)
        else:
            score, actions, stats = beam_search(
                args.seed, args.difficulty,
                beam_width=args.beam_width,
                max_per_bot=args.max_per_bot,
                max_joint=args.max_joint,
                verbose=verbose,
            )
        if verbose:
            print(f"\nSolved: {args.difficulty} seed={args.seed} score={score}")

    # Save if requested
    if args.save:
        save_actions(actions, args.save)

    # Auto-save to solutions dir
    sol_dir = os.path.join(os.path.dirname(__file__), 'solutions')
    os.makedirs(sol_dir, exist_ok=True)
    sol_path = os.path.join(sol_dir, f'{args.difficulty}_{args.seed}.json')
    save_actions(actions, sol_path)

    # Replay if requested
    if args.replay:
        ms = build_map(args.difficulty)
        ws_score = asyncio.run(replay(args.replay, actions, ms, verbose=verbose))
        if score is not None:
            if ws_score == score:
                print(f"REPLAY VERIFIED: {ws_score} == {score}")
            else:
                print(f"REPLAY MISMATCH: {ws_score} != {score} (solver)")

    return score


if __name__ == '__main__':
    main()

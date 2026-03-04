"""Benchmark AnytimeGPUStream against in-process Python simulator.

No WebSocket needed. Feeds game_engine states directly into the solver's
synchronous decision methods, runs multiple seeds, reports statistics.

Usage:
    python benchmark_stream.py easy --seeds 7001-7005
    python benchmark_stream.py medium --seed 7001
    python benchmark_stream.py hard --seeds 5          # seeds 7001-7005
    python benchmark_stream.py easy --cpu --no-refine  # quick CPU-only test

The script also prints a greedy baseline (no background threads) for comparison.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import (
    init_game, step, state_to_ws_format,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, MAX_ROUNDS,
)
from live_gpu_stream import AnytimeGPUStream

_ACTION_STR_MAP = {
    'wait': ACT_WAIT,
    'move_up': ACT_MOVE_UP,
    'move_down': ACT_MOVE_DOWN,
    'move_left': ACT_MOVE_LEFT,
    'move_right': ACT_MOVE_RIGHT,
    'pick_up': ACT_PICKUP,
    'drop_off': ACT_DROPOFF,
}


def ws_action_to_internal(ws_act, ms):
    """Convert a WS action dict back to (act, item_idx) for game_engine.step()."""
    act = _ACTION_STR_MAP.get(ws_act.get('action', 'wait'), ACT_WAIT)
    item_idx = -1
    if act == ACT_PICKUP:
        iid = ws_act.get('item_id', '')
        for idx, item in enumerate(ms.items):
            if item['id'] == iid:
                item_idx = idx
                break
    return (act, item_idx)


def run_seed(difficulty, seed, device='cuda', max_states=None, no_refine=False,
            verbose=True, round_delay=0.0, record=False):
    """Run one seed via in-process sim. Returns (final_score, elapsed_s).

    round_delay: seconds to sleep between rounds, simulating live game timing.
    Real game is 0.4s/round. Use 0.1-0.4 to let background GPU threads contribute.
    record: if True, write a JSONL log and import to PostgreSQL as 'synthetic'.
    """
    import json as _json
    import subprocess as _subprocess

    solver = AnytimeGPUStream(
        ws_url='sim://in-process',
        save=False,
        max_states=max_states,
        no_refine=no_refine,
        device=device,
    )

    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    t0 = time.time()
    last_source = 'none'

    log_lines = [] if record else None

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        data = state_to_ws_format(state, all_orders)

        if rnd == 0:
            solver._init_round0(data)
        else:
            solver._check_new_orders(data)

        if round_delay > 0:
            time.sleep(round_delay)

        ws_actions, source = solver._get_actions(rnd, data)

        # Source-change logging
        if verbose and (source != last_source or rnd < 3 or rnd % 50 == 0 or rnd >= MAX_ROUNDS - 3):
            plan_sc = solver._plan.score
            marker = f'[{last_source}→{source}]' if source != last_source else f'[{source}]'
            print(f"  R{rnd:3d} Score={state.score:4d} plan={plan_sc:4d} {marker}",
                  file=sys.stderr)
            last_source = source

        # Capture JSONL data for DB import
        if record:
            log_lines.append(_json.dumps(data))
            log_lines.append(_json.dumps({'actions': ws_actions}))

        # Convert WS actions → internal (act, item_idx)
        actions_by_bot = {wa['bot']: ws_action_to_internal(wa, ms) for wa in ws_actions}
        num_bots = len(state.bot_positions)
        internal_actions = [actions_by_bot.get(bid, (ACT_WAIT, -1)) for bid in range(num_bots)]

        step(state, internal_actions, all_orders)

    elapsed = time.time() - t0

    # Write JSONL and import to DB
    if record and log_lines:
        import tempfile
        log_lines.append(_json.dumps({'type': 'game_over', 'score': state.score}))
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'game_log_syn_{difficulty}_{seed}_{int(t0)}.jsonl')
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines) + '\n')
        _import_script = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'grocery-bot-zig', 'replay', 'import_logs.py',
        ))
        if os.path.exists(_import_script):
            _subprocess.Popen(
                ['python', _import_script, log_path, '--run-type', 'synthetic'],
                stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
            )
            print(f"  [db] Importing synthetic log to PostgreSQL (seed={seed})", file=sys.stderr)

    return state.score, elapsed


def run_greedy_baseline(difficulty, seed):
    """Run pure-greedy (no background solvers) for comparison."""
    solver = AnytimeGPUStream(
        ws_url='sim://greedy',
        save=False,
        max_states=1,   # 1 state = effectively instant/noop GPU pass
        no_refine=True,
        device='cpu',
    )
    # Disable background threads by not calling _init_round0
    # Instead manually run just greedy decisions

    from live_gpu_stream import ACT_WAIT
    from replay_solution import build_walkable
    from game_engine import build_map_from_capture
    from live_solver import ws_to_capture

    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        data = state_to_ws_format(state, all_orders)

        if rnd == 0:
            # Init solver state (walkable, map) without starting background threads
            cap = ws_to_capture(data)
            solver._capture = cap
            solver._difficulty = cap['difficulty']
            solver._num_bots = cap['num_bots']
            from replay_solution import build_walkable
            solver._map_state = build_map_from_capture(cap)
            solver._walkable = build_walkable(solver._map_state)
            from precompute import PrecomputedTables
            try:
                solver._tables = PrecomputedTables.get(solver._map_state)
            except Exception:
                solver._tables = None
            for order in data.get('orders', []):
                oid = order.get('id', f'order_{len(solver._seen_order_ids)}')
                solver._seen_order_ids.add(oid)

        live_bots = data.get('bots', [])
        occupied = {tuple(b['position']) for b in live_bots}
        ws_actions = solver._greedy_all(live_bots, data, occupied)

        actions_by_bot = {wa['bot']: ws_action_to_internal(wa, ms) for wa in ws_actions}
        num_bots = len(state.bot_positions)
        internal_actions = [actions_by_bot.get(bid, (ACT_WAIT, -1)) for bid in range(num_bots)]
        step(state, internal_actions, all_orders)

    return state.score


def parse_seeds(seed_arg, seed_single):
    if seed_single is not None:
        return [seed_single]
    s = str(seed_arg)
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    try:
        n = int(s)
        if n < 1000:
            # treat as count from 7001
            return list(range(7001, 7001 + n))
        return [n]
    except ValueError:
        return [7001]


def main():
    parser = argparse.ArgumentParser(description='Benchmark AnytimeGPUStream in-process')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--seed', type=int, default=None, help='Single seed')
    parser.add_argument('--seeds', type=str, default='7001-7003',
                        help='Seed range e.g. 7001-7005 or count e.g. 5 (=7001-7005)')
    parser.add_argument('--max-states', type=int, default=None,
                        help='Override GPU pass state budget')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip refinement in GPU passes')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (no GPU)')
    parser.add_argument('--round-delay', type=float, default=0.0,
                        help='Sleep N seconds between rounds to simulate live timing '
                             '(real game = 0.4s/round, so --round-delay 0.4 = realistic)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip greedy baseline comparison')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-round output')
    parser.add_argument('--record', action='store_true',
                        help='Write JSONL log per seed and import to PostgreSQL as synthetic')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'
    seeds = parse_seeds(args.seeds, args.seed)

    print(f"\nBenchmark: {args.difficulty}, device={device}, "
          f"seeds={seeds[0]}-{seeds[-1]} ({len(seeds)} total)", file=sys.stderr)
    if args.max_states:
        print(f"  max_states override: {args.max_states:,}", file=sys.stderr)

    # Greedy baseline
    baseline_scores = []
    if not args.no_baseline:
        print(f"\n--- Greedy baseline ---", file=sys.stderr)
        for seed in seeds:
            sc = run_greedy_baseline(args.difficulty, seed)
            baseline_scores.append(sc)
            print(f"  Seed {seed}: greedy={sc}", file=sys.stderr)
        print(f"  Baseline mean={sum(baseline_scores)/len(baseline_scores):.1f} "
              f"max={max(baseline_scores)} min={min(baseline_scores)}", file=sys.stderr)

    # Anytime solver
    print(f"\n--- AnytimeGPUStream ---", file=sys.stderr)
    scores = []
    times = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===", file=sys.stderr)
        score, elapsed = run_seed(
            difficulty=args.difficulty,
            seed=seed,
            device=device,
            max_states=args.max_states,
            no_refine=args.no_refine,
            verbose=not args.quiet,
            round_delay=args.round_delay,
            record=args.record,
        )
        scores.append(score)
        times.append(elapsed)
        print(f"Seed {seed}: score={score} ({elapsed:.1f}s)", file=sys.stderr)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Results: {args.difficulty}, {len(seeds)} seeds", file=sys.stderr)
    if baseline_scores:
        print(f"  Greedy: mean={sum(baseline_scores)/len(baseline_scores):.1f} "
              f"max={max(baseline_scores)} min={min(baseline_scores)}", file=sys.stderr)
    print(f"  Anytime: mean={sum(scores)/len(scores):.1f} "
          f"max={max(scores)} min={min(scores)}", file=sys.stderr)
    if baseline_scores:
        delta = sum(scores) / len(scores) - sum(baseline_scores) / len(baseline_scores)
        print(f"  Delta vs greedy: {delta:+.1f}", file=sys.stderr)
    print(f"  Avg time/game: {sum(times)/len(times):.1f}s", file=sys.stderr)
    print(f"  Scores: {scores}", file=sys.stderr)


if __name__ == '__main__':
    main()

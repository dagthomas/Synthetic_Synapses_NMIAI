"""Profile GPU beam search to identify bottlenecks.

Runs one search() call with torch.profiler to measure:
- Kernel launch count and GPU utilization
- Memory bandwidth usage
- Top time-consuming operations

Usage:
    python profile_gpu.py [difficulty] [--beam N] [--seed N] [--trace]
"""
import argparse
import sys
import time
import torch
from torch.profiler import profile, ProfilerActivity, schedule

from game_engine import init_game
from gpu_beam_search import GPUBeamSearcher


def main():
    parser = argparse.ArgumentParser(description='Profile GPU beam search')
    parser.add_argument('difficulty', nargs='?', default='easy',
                        choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beam', type=int, default=50000)
    parser.add_argument('--trace', action='store_true',
                        help='Save Chrome trace to profile_trace.json')
    args = parser.parse_args()

    print(f"Profiling: {args.difficulty}, beam={args.beam}, seed={args.seed}",
          file=sys.stderr)

    # Setup
    gs, all_orders = init_game(args.seed, args.difficulty)
    device = 'cuda'
    num_bots = len(gs.bot_positions)
    searcher = GPUBeamSearcher(gs.map_state, all_orders, device=device,
                               num_bots=num_bots)

    # Warm up compiled functions (first call triggers compilation)
    gs_warmup = gs.copy()
    print("Warming up (first 10 rounds)...", file=sys.stderr)
    state = searcher._from_game_state(gs_warmup)
    for rnd in range(10):
        B = state['bot_x'].shape[0]
        expanded, actions, action_items, valid_mask, C = \
            searcher._smart_expand(state, max_cands=20)
        new_state = searcher._step(expanded, actions, action_items, round_num=rnd)
        evals = searcher._eval(new_state, round_num=rnd)
        k = min(args.beam, new_state['bot_x'].shape[0])
        _, topk_idx = torch.topk(evals, k)
        state = {key: val[topk_idx] for key, val in new_state.items()}
    torch.cuda.synchronize()
    print("Warmup done.", file=sys.stderr)

    # Profile the actual search
    gs2 = gs.copy()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    trace_path = 'profile_trace.json' if args.trace else None

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        t0 = time.time()
        score, actions = searcher.dp_search(gs2, max_states=args.beam)
        torch.cuda.synchronize()
        dt = time.time() - t0

    print(f"\nScore: {score}, Time: {dt:.2f}s\n", file=sys.stderr)

    # Summary
    print("=== Top 30 CUDA operations (by total time) ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30))

    print("\n=== Top 20 CPU operations (by total time) ===")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=20))

    # Kernel launch stats
    events = prof.key_averages()
    total_cuda_time = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)
    total_cpu_time = sum(e.cpu_time_total for e in events if e.cpu_time_total > 0)
    num_kernels = sum(e.count for e in events if e.cuda_time_total > 0)
    print(f"\n=== Summary ===")
    print(f"Total CUDA time: {total_cuda_time/1e6:.2f}s")
    print(f"Total CPU time:  {total_cpu_time/1e6:.2f}s")
    print(f"Kernel launches: {num_kernels}")
    print(f"Wall time:       {dt:.2f}s")
    if dt > 0:
        print(f"GPU utilization:  ~{total_cuda_time/1e6/dt*100:.0f}%")

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace saved to: {trace_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")


if __name__ == '__main__':
    main()

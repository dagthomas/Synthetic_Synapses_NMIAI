"""Deep offline training CLI — hours-long optimization without live tokens.

Three-phase training:
  Phase 1: Exploration (30%) — try many pass-1 orderings at explore_states
  Phase 2: Intensification (50%) — deep refinement with squad joint DP
  Phase 3: LNS (20%) — destroy-repair cycles to escape local optima

Checkpoints every 10 minutes. Score-safe (never overwrites better solutions).

Usage:
    python deep_optimize.py hard --budget 7200 --max-states 500000
    python deep_optimize.py expert --budget 14400 --max-states 200000 --orderings 50
    python deep_optimize.py hard --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import _shared  # noqa: F401
from b200_config import get_params, detect_gpu, print_gpu_info
from squad_solver import solve_squads, SquadConfig
from solution_store import save_solution, load_capture, load_solution, load_meta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')


def _checkpoint_path(difficulty: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f'deep_{difficulty}.json')


def _save_checkpoint(difficulty: str, data: dict):
    path = _checkpoint_path(difficulty)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_checkpoint(difficulty: str) -> dict | None:
    path = _checkpoint_path(difficulty)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def deep_optimize(difficulty: str, budget_s: float = 7200,
                  max_states: int | None = None,
                  joint_states: int | None = None,
                  orderings: int | None = None,
                  gpu: str = 'auto',
                  resume: bool = False):
    """Run deep offline optimization for a difficulty.

    Args:
        budget_s: Total time budget in seconds.
        max_states: Override per-bot beam width.
        joint_states: Override joint DP beam width.
        orderings: Override number of pass-1 orderings.
        gpu: GPU profile ('auto', 'b200', '5090').
        resume: Resume from checkpoint.
    """
    capture = load_capture(difficulty)
    if not capture:
        print(f"No capture data for {difficulty}. Run a game first.", file=sys.stderr)
        return

    gpu_type = gpu if gpu != 'auto' else detect_gpu()
    params = get_params(difficulty, gpu_type)

    if max_states:
        params.max_states = max_states
    if joint_states:
        params.joint_states = joint_states
    if orderings:
        params.pass1_orderings = orderings

    existing_meta = load_meta(difficulty)
    existing_score = existing_meta.get('score', 0) if existing_meta else 0

    checkpoint = _load_checkpoint(difficulty) if resume else None
    phase_start = 0
    if checkpoint and resume:
        phase_start = checkpoint.get('last_phase', 0) + 1
        print(f"Resuming from phase {phase_start}, "
              f"best score so far: {checkpoint.get('best_score', 0)}", file=sys.stderr)

    t0 = time.time()
    deadline = t0 + budget_s

    print(f"\nDeep Optimize: {difficulty}", file=sys.stderr)
    print(f"  Budget: {budget_s:.0f}s ({budget_s/3600:.1f}h)", file=sys.stderr)
    print(f"  GPU: {gpu_type}", file=sys.stderr)
    print(f"  max_states={params.max_states:,}, joint_states={params.joint_states:,}",
          file=sys.stderr)
    print(f"  orderings={params.pass1_orderings}, refine={params.refine_iters}, "
          f"lns={params.lns_rounds}", file=sys.stderr)
    print(f"  Existing score: {existing_score}", file=sys.stderr)

    best_score = existing_score

    # === Phase 1: Exploration (30% of budget) ===
    phase1_budget = budget_s * 0.3
    if phase_start <= 0 and (time.time() - t0) < phase1_budget:
        print(f"\n--- Phase 1: Exploration ({phase1_budget:.0f}s) ---", file=sys.stderr)

        explore_config = SquadConfig(
            difficulty=difficulty,
            max_states=params.explore_states,
            joint_states=0,  # no joint DP in exploration
            joint_squad_size=1,
            explore_states=params.explore_states,
            pass1_orderings=params.pass1_orderings,
            refine_iters=0,
            lns_rounds=0,
            max_dp_bots=params.max_dp_bots,
            max_time_s=phase1_budget,
            speed_bonus=params.speed_bonus,
            device='cuda',
        )

        score, actions = solve_squads(capture, explore_config)

        if score > best_score:
            best_score = score
            saved = save_solution(difficulty, score, actions)
            if saved:
                print(f"  Phase 1: NEW BEST {score}", file=sys.stderr)

        _save_checkpoint(difficulty, {
            'last_phase': 0,
            'best_score': best_score,
            'phase1_score': score,
            'timestamp': time.time(),
        })

    # === Phase 2: Intensification (50% of budget) ===
    phase2_budget = budget_s * 0.5
    remaining = deadline - time.time()
    phase2_actual = min(phase2_budget, remaining * 0.7)

    if phase_start <= 1 and phase2_actual > 30:
        print(f"\n--- Phase 2: Intensification ({phase2_actual:.0f}s) ---", file=sys.stderr)

        intense_config = SquadConfig(
            difficulty=difficulty,
            max_states=params.max_states,
            joint_states=params.joint_states,
            joint_squad_size=params.joint_squad_size,
            explore_states=params.explore_states,
            pass1_orderings=1,  # already explored in Phase 1
            refine_iters=params.refine_iters,
            lns_rounds=0,  # LNS in Phase 3
            max_dp_bots=params.max_dp_bots,
            max_time_s=phase2_actual,
            speed_bonus=params.speed_bonus,
            device='cuda',
        )

        score, actions = solve_squads(capture, intense_config)

        if score > best_score:
            best_score = score
            saved = save_solution(difficulty, score, actions)
            if saved:
                print(f"  Phase 2: NEW BEST {score}", file=sys.stderr)

        _save_checkpoint(difficulty, {
            'last_phase': 1,
            'best_score': best_score,
            'phase2_score': score,
            'timestamp': time.time(),
        })

    # === Phase 3: LNS (20% of budget) ===
    remaining = deadline - time.time()
    phase3_actual = remaining * 0.9  # leave 10% margin

    if phase_start <= 2 and phase3_actual > 30:
        print(f"\n--- Phase 3: LNS ({phase3_actual:.0f}s) ---", file=sys.stderr)

        lns_config = SquadConfig(
            difficulty=difficulty,
            max_states=params.max_states,
            joint_states=params.joint_states,
            joint_squad_size=params.joint_squad_size,
            explore_states=params.explore_states,
            pass1_orderings=1,
            refine_iters=0,
            lns_rounds=params.lns_rounds,
            max_dp_bots=params.max_dp_bots,
            max_time_s=phase3_actual,
            speed_bonus=params.speed_bonus,
            device='cuda',
        )

        score, actions = solve_squads(capture, lns_config)

        if score > best_score:
            best_score = score
            saved = save_solution(difficulty, score, actions)
            if saved:
                print(f"  Phase 3: NEW BEST {score}", file=sys.stderr)

        _save_checkpoint(difficulty, {
            'last_phase': 2,
            'best_score': best_score,
            'phase3_score': score,
            'timestamp': time.time(),
        })

    total_time = time.time() - t0
    print(f"\nDeep Optimize complete: {difficulty}", file=sys.stderr)
    print(f"  Final score: {best_score} (started at {existing_score})", file=sys.stderr)
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep offline optimization')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--budget', type=float, default=7200,
                        help='Time budget in seconds (default: 7200 = 2h)')
    parser.add_argument('--max-states', type=int, default=None)
    parser.add_argument('--joint-states', type=int, default=None)
    parser.add_argument('--orderings', type=int, default=None)
    parser.add_argument('--gpu', default='auto', choices=['auto', 'b200', '5090', 'generic'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    print_gpu_info()
    deep_optimize(
        args.difficulty,
        budget_s=args.budget,
        max_states=args.max_states,
        joint_states=args.joint_states,
        orderings=args.orderings,
        gpu=args.gpu,
        resume=args.resume,
    )

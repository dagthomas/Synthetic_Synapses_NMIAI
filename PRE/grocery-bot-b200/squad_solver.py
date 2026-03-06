"""Squad-based multi-bot orchestrator with LNS.

Replaces sequential per-bot DP with squad-based joint planning:
  1. All-permutation Pass-1: try many bot orderings at explore_states
  2. Congestion-based squad assignment: group bots that interact most
  3. Joint DP per squad via JointBeamSearcher
  4. LNS destroy-repair: escape local optima with congestion-targeted destroy

On B200: 5-bot joint for Hard, 4-bot squads for Expert.
On 5090: 2-bot squads (same as existing GPUBeamSearcher2Bot but generalized).
"""
from __future__ import annotations

import itertools
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

import _shared  # noqa: F401
from game_engine import (
    GameState, MapState, Order, CaptureData,
    init_game_from_capture,
    MAX_ROUNDS, INV_CAP, ACT_WAIT,
    step as cpu_step,
)
from gpu_sequential_solver import (
    solve_sequential, refine_from_solution, pre_simulate_locked,
    cpu_verify, _make_combined, compute_bot_contributions,
    greedy_plan_bots, SolveConfig, compute_type_assignments,
)
from solution_store import save_solution, load_capture, load_solution, load_meta
from configs import CONFIGS
from b200_config import get_params, B200Params
from joint_beam_search import JointBeamSearcher


@dataclass
class SquadConfig:
    """Configuration for squad-based solving."""
    difficulty: str = 'hard'
    max_states: int = 500_000
    joint_states: int = 5_000_000
    joint_squad_size: int = 3
    explore_states: int = 100_000
    pass1_orderings: int = 120
    refine_iters: int = 100
    lns_rounds: int = 40
    max_dp_bots: int = 5
    max_time_s: float | None = None
    speed_bonus: float = 100.0
    speed_decay: float = 0.5
    no_filler: bool = True
    no_compile: bool = False
    device: str = 'cuda'
    verbose: bool = True


def compute_congestion(capture_data: CaptureData, all_orders: list[Order],
                       combined_actions: list[list[tuple[int, int]]],
                       num_bots: int) -> np.ndarray:
    """Compute pairwise congestion matrix by simulating the solution.

    congestion[i][j] = number of rounds where manhattan(bot_i, bot_j) <= 3.

    Returns [num_bots, num_bots] float matrix.
    """
    gs, _ = init_game_from_capture(capture_data,
                                   num_orders=len(capture_data.get('orders', [])))
    congestion = np.zeros((num_bots, num_bots), dtype=np.float32)

    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)

        # Record pairwise distances
        for i in range(num_bots):
            xi, yi = int(gs.bot_positions[i, 0]), int(gs.bot_positions[i, 1])
            for j in range(i + 1, num_bots):
                xj, yj = int(gs.bot_positions[j, 0]), int(gs.bot_positions[j, 1])
                dist = abs(xi - xj) + abs(yi - yj)
                if dist <= 3:
                    congestion[i, j] += 1
                    congestion[j, i] += 1

    return congestion


def assign_squads(num_bots: int, congestion_matrix: np.ndarray,
                  max_squad_size: int = 3) -> list[tuple[int, ...]]:
    """Assign bots to squads by greedy pairing of most-congested bots.

    Returns list of tuples, each containing bot IDs in one squad.
    """
    assigned = set()
    squads = []

    # Sort all pairs by congestion descending
    pairs = []
    for i in range(num_bots):
        for j in range(i + 1, num_bots):
            pairs.append((congestion_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for _, i, j in pairs:
        if i in assigned or j in assigned:
            continue
        squad = [i, j]
        assigned.add(i)
        assigned.add(j)

        # Try to extend to triple
        if max_squad_size >= 3:
            best_k = -1
            best_cong = 0
            for k in range(num_bots):
                if k in assigned:
                    continue
                # k must be congested with BOTH i and j
                cong_k = min(congestion_matrix[i, k], congestion_matrix[j, k])
                if cong_k > best_cong:
                    best_cong = cong_k
                    best_k = k
            if best_k >= 0 and best_cong > 10:  # threshold: at least 10 rounds near both
                squad.append(best_k)
                assigned.add(best_k)

                # Try to extend to quad
                if max_squad_size >= 4:
                    best_m = -1
                    best_cong_m = 0
                    for m in range(num_bots):
                        if m in assigned:
                            continue
                        cong_m = min(congestion_matrix[s, m] for s in squad)
                        if cong_m > best_cong_m:
                            best_cong_m = cong_m
                            best_m = m
                    if best_m >= 0 and best_cong_m > 10:
                        squad.append(best_m)
                        assigned.add(best_m)

        squads.append(tuple(sorted(squad)))

    # Solo bots
    for i in range(num_bots):
        if i not in assigned:
            squads.append((i,))

    return squads


def all_permutation_pass1(capture_data: CaptureData, config: SquadConfig,
                          max_orderings: int = 120,
                          ) -> list[tuple[int, list[list[tuple[int, int]]]]]:
    """Try multiple bot orderings at explore_states, return top-3.

    Returns: sorted list of (score, combined_actions) tuples, best first.
    """
    num_bots = capture_data.get('num_bots', CONFIGS[config.difficulty]['bots'])
    dp_bots = min(config.max_dp_bots, num_bots)
    bot_ids = list(range(dp_bots))

    # Generate orderings
    if dp_bots <= 5:
        # All permutations
        all_perms = list(itertools.permutations(bot_ids))
    else:
        # Random sample
        rng = random.Random(42)
        all_perms = set()
        all_perms.add(tuple(bot_ids))
        all_perms.add(tuple(reversed(bot_ids)))
        while len(all_perms) < max_orderings:
            perm = list(bot_ids)
            rng.shuffle(perm)
            all_perms.add(tuple(perm))
        all_perms = list(all_perms)

    all_perms = all_perms[:max_orderings]

    results = []
    t0 = time.time()

    for idx, perm in enumerate(all_perms):
        if config.max_time_s and (time.time() - t0) > config.max_time_s * 0.3:
            if config.verbose:
                print(f"  Pass-1 time limit reached after {idx}/{len(all_perms)} orderings",
                      file=sys.stderr)
            break

        solve_config = SolveConfig(
            max_states=config.explore_states,
            no_filler=config.no_filler,
            no_compile=config.no_compile,
            bot_order=list(perm),
            speed_bonus=config.speed_bonus,
            max_dp_bots=config.max_dp_bots,
            max_refine_iters=0,  # no refinement in exploration
            num_pass1_orderings=1,
        )

        score, actions = solve_sequential(
            capture_data=capture_data,
            difficulty=config.difficulty,
            device=config.device,
            config=solve_config,
            verbose=False,
        )

        results.append((score, actions))

        if config.verbose and (idx < 3 or idx % 10 == 0):
            elapsed = time.time() - t0
            perm_str = ','.join(str(b) for b in perm)
            print(f"  Pass-1 [{idx+1}/{len(all_perms)}] order=({perm_str}): "
                  f"score={score}, elapsed={elapsed:.0f}s", file=sys.stderr)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

    results.sort(key=lambda x: -x[0])
    return results[:3]


def solve_squads(capture_data: CaptureData, config: SquadConfig,
                 ) -> tuple[int, list[list[tuple[int, int]]]]:
    """Main squad-based solver with all phases.

    Phase 0: Initial sequential DP (warm-up)
    Phase 1: All-permutation Pass-1
    Phase 2: Squad joint DP
    Phase 3: Iterative refinement (weakest-first)
    Phase 4: LNS destroy-repair

    Returns (score, combined_actions).
    """
    t0 = time.time()
    difficulty = config.difficulty
    num_bots = capture_data.get('num_bots', CONFIGS[difficulty]['bots'])
    num_orders = len(capture_data.get('orders', []))

    gs, all_orders = init_game_from_capture(capture_data, num_orders=num_orders)
    ms = gs.map_state

    if config.verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Squad Solver: {difficulty}, {num_bots} bots, {num_orders} orders",
              file=sys.stderr)
        print(f"  max_states={config.max_states:,}, joint_states={config.joint_states:,}, "
              f"squad_size={config.joint_squad_size}", file=sys.stderr)
        print(f"  orderings={config.pass1_orderings}, refine={config.refine_iters}, "
              f"lns={config.lns_rounds}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    # === Phase 0: Quick sequential warm-up ===
    if config.verbose:
        print(f"\n--- Phase 0: Sequential warm-up ---", file=sys.stderr)

    warmup_config = SolveConfig(
        max_states=config.explore_states,
        no_filler=config.no_filler,
        no_compile=config.no_compile,
        speed_bonus=config.speed_bonus,
        max_dp_bots=config.max_dp_bots,
        max_refine_iters=0,
        num_pass1_orderings=1,
    )
    best_score, best_actions = solve_sequential(
        capture_data=capture_data,
        difficulty=difficulty,
        device=config.device,
        config=warmup_config,
        verbose=config.verbose,
    )

    if config.verbose:
        print(f"  Phase 0 score: {best_score}", file=sys.stderr)

    if config.device == 'cuda':
        torch.cuda.empty_cache()

    # === Phase 1: All-permutation Pass-1 ===
    if config.pass1_orderings > 1:
        if config.verbose:
            print(f"\n--- Phase 1: All-permutation Pass-1 "
                  f"({config.pass1_orderings} orderings) ---", file=sys.stderr)

        top_results = all_permutation_pass1(capture_data, config,
                                            max_orderings=config.pass1_orderings)

        for score, actions in top_results:
            if score > best_score:
                best_score = score
                best_actions = actions

        if config.verbose:
            scores_str = ', '.join(str(s) for s, _ in top_results[:5])
            print(f"  Top Pass-1 scores: {scores_str}", file=sys.stderr)
            print(f"  Best after Phase 1: {best_score}", file=sys.stderr)

    def _time_remaining():
        if config.max_time_s is None:
            return float('inf')
        return config.max_time_s - (time.time() - t0)

    # === Phase 2: Squad joint DP ===
    if config.joint_squad_size >= 2 and config.joint_states > 0 and _time_remaining() > 30:
        if config.verbose:
            print(f"\n--- Phase 2: Squad joint DP ---", file=sys.stderr)

        # Convert combined_actions to bot_actions dict
        bot_actions = {}
        for bid in range(num_bots):
            bot_actions[bid] = [(best_actions[r][bid][0], best_actions[r][bid][1])
                                for r in range(MAX_ROUNDS)]

        # Compute congestion
        congestion = compute_congestion(capture_data, all_orders, best_actions, num_bots)

        if config.verbose:
            # Show top congestion pairs
            top_pairs = []
            for i in range(num_bots):
                for j in range(i + 1, num_bots):
                    top_pairs.append((congestion[i, j], i, j))
            top_pairs.sort(reverse=True)
            pair_str = ', '.join(f"({i},{j}):{c:.0f}" for c, i, j in top_pairs[:5])
            print(f"  Top congestion pairs: {pair_str}", file=sys.stderr)

        # Assign squads
        squads = assign_squads(num_bots, congestion,
                               max_squad_size=config.joint_squad_size)

        if config.verbose:
            squad_str = ', '.join(str(s) for s in squads)
            print(f"  Squads: {squad_str}", file=sys.stderr)

        # Run joint DP per squad
        for squad in squads:
            if len(squad) < 2:
                continue  # solo bots keep their sequential plan
            if _time_remaining() < 20:
                break

            if config.verbose:
                print(f"\n  Joint DP for squad {squad}...", file=sys.stderr)

            # Lock all bots NOT in this squad
            locked_ids = sorted(bid for bid in range(num_bots) if bid not in squad)
            locked = None
            if locked_ids:
                locked = pre_simulate_locked(
                    gs.copy(), all_orders, bot_actions, locked_ids)

            searcher = JointBeamSearcher(
                ms, all_orders,
                n_candidates=len(squad),
                candidate_bot_ids=squad,
                num_bots=num_bots,
                device=config.device,
                locked_trajectories=locked,
                no_compile=config.no_compile,
                speed_bonus=config.speed_bonus,
                chunk_size=config.joint_states // 2,
            )

            joint_score, joint_acts = searcher.dp_search_nbot(
                gs.copy(),
                max_states=config.joint_states,
                verbose=config.verbose,
                max_combos=50,
            )

            del searcher
            if config.device == 'cuda':
                torch.cuda.empty_cache()

            # Verify: does joint plan beat sequential?
            test_actions = dict(bot_actions)
            for bid, acts in joint_acts.items():
                test_actions[bid] = acts
            combined = _make_combined(test_actions, num_bots)

            gs_v = gs.copy()
            verified_score = cpu_verify(gs_v, all_orders, combined, num_bots)

            if verified_score > best_score:
                if config.verbose:
                    print(f"    Squad {squad}: {best_score} -> {verified_score} "
                          f"(+{verified_score - best_score})", file=sys.stderr)
                best_score = verified_score
                best_actions = combined
                for bid, acts in joint_acts.items():
                    bot_actions[bid] = acts
            else:
                if config.verbose:
                    print(f"    Squad {squad}: no improvement "
                          f"(joint={verified_score}, current={best_score})", file=sys.stderr)

    # === Phase 3: Iterative refinement ===
    if config.refine_iters > 0 and _time_remaining() > 30:
        if config.verbose:
            print(f"\n--- Phase 3: Refinement ({config.refine_iters} iters) ---",
                  file=sys.stderr)

        refine_budget = _time_remaining() * 0.6  # 60% of remaining time
        ref_score, ref_actions = refine_from_solution(
            best_actions,
            capture_data=capture_data,
            difficulty=difficulty,
            device=config.device,
            max_states=config.max_states,
            max_refine_iters=config.refine_iters,
            max_time_s=refine_budget,
            no_filler=config.no_filler,
            speed_bonus=config.speed_bonus,
            max_dp_bots=config.max_dp_bots,
        )

        if ref_score > best_score:
            if config.verbose:
                print(f"  Refinement: {best_score} -> {ref_score} "
                      f"(+{ref_score - best_score})", file=sys.stderr)
            best_score = ref_score
            best_actions = ref_actions

        if config.device == 'cuda':
            torch.cuda.empty_cache()

    # === Phase 4: LNS destroy-repair ===
    if config.lns_rounds > 0 and _time_remaining() > 30:
        if config.verbose:
            print(f"\n--- Phase 4: LNS ({config.lns_rounds} rounds) ---",
                  file=sys.stderr)

        best_score, best_actions = _lns_loop(
            capture_data, config, gs, all_orders, ms,
            best_score, best_actions, num_bots, t0)

    total_time = time.time() - t0
    if config.verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Squad Solver final: score={best_score}, time={total_time:.0f}s",
              file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    return best_score, best_actions


def _lns_loop(capture_data, config, gs, all_orders, ms,
              best_score, best_actions, num_bots, t0):
    """LNS destroy-repair loop."""
    rng = random.Random(42)
    dp_bots = min(config.max_dp_bots, num_bots)

    # Convert to bot_actions dict
    bot_actions = {}
    for bid in range(num_bots):
        bot_actions[bid] = [(best_actions[r][bid][0], best_actions[r][bid][1])
                            for r in range(MAX_ROUNDS)]

    strategies = ['full', 'window', 'type']

    for lns_round in range(config.lns_rounds):
        if config.max_time_s and (time.time() - t0) > config.max_time_s * 0.95:
            break

        strategy = strategies[lns_round % len(strategies)]

        # Choose bots to destroy
        if strategy == 'full':
            # Full destroy: reset 2-3 bots completely
            n_destroy = min(config.joint_squad_size, dp_bots)
            # Pick weakest bots
            contribs = compute_bot_contributions(
                gs.copy(), all_orders, bot_actions, num_bots,
                list(range(dp_bots)),
            )
            destroy_bids = sorted(range(dp_bots), key=lambda b: contribs.get(b, 0))[:n_destroy]

        elif strategy == 'window':
            # Window destroy: find congestion hotspot, reset bots in that window
            combined = _make_combined(bot_actions, num_bots)
            congestion = compute_congestion(capture_data, all_orders, combined, num_bots)
            # Find most congested pair
            best_pair = (0, 1)
            best_cong = 0
            for i in range(dp_bots):
                for j in range(i + 1, dp_bots):
                    if congestion[i, j] > best_cong:
                        best_cong = congestion[i, j]
                        best_pair = (i, j)
            destroy_bids = list(best_pair)

        elif strategy == 'type':
            # Type destroy: reset bots handling specific item types + reshuffle
            n_destroy = min(2, dp_bots)
            destroy_bids = rng.sample(range(dp_bots), n_destroy)

        destroy_bids = sorted(destroy_bids)

        if config.verbose and lns_round < 5:
            print(f"  LNS [{lns_round+1}/{config.lns_rounds}] "
                  f"strategy={strategy}, destroy={destroy_bids}", file=sys.stderr)

        # Destroy: reset selected bots
        test_actions = dict(bot_actions)
        wait_plan = [(ACT_WAIT, -1)] * MAX_ROUNDS
        for bid in destroy_bids:
            test_actions[bid] = wait_plan

        # Repair: re-plan destroyed bots with joint DP if possible
        if len(destroy_bids) >= 2 and config.joint_states > 0:
            locked_ids = sorted(bid for bid in range(num_bots) if bid not in destroy_bids)
            locked = None
            if locked_ids:
                locked = pre_simulate_locked(gs.copy(), all_orders, test_actions, locked_ids)

            searcher = JointBeamSearcher(
                ms, all_orders,
                n_candidates=len(destroy_bids),
                candidate_bot_ids=tuple(destroy_bids),
                num_bots=num_bots,
                device=config.device,
                locked_trajectories=locked,
                no_compile=config.no_compile,
                speed_bonus=config.speed_bonus,
            )

            lns_states = min(config.joint_states, config.max_states)
            _, joint_acts = searcher.dp_search_nbot(
                gs.copy(), max_states=lns_states, verbose=False, max_combos=30)

            del searcher
            if config.device == 'cuda':
                torch.cuda.empty_cache()

            for bid, acts in joint_acts.items():
                test_actions[bid] = acts
        else:
            # Single-bot repair for each destroyed bot
            for bid in destroy_bids:
                locked_ids = sorted(b for b in range(num_bots) if b != bid and b not in destroy_bids)
                locked_ids.extend(sorted(b for b in destroy_bids if b != bid and b in test_actions and test_actions[b] != wait_plan))
                locked_ids = sorted(set(locked_ids))
                locked = None
                if locked_ids:
                    locked = pre_simulate_locked(gs.copy(), all_orders, test_actions, locked_ids)

                from gpu_beam_search import GPUBeamSearcher
                searcher = GPUBeamSearcher(
                    ms, all_orders, device=config.device, num_bots=num_bots,
                    locked_trajectories=locked, no_compile=config.no_compile,
                    speed_bonus=config.speed_bonus)

                _, bot_acts = searcher.dp_search(
                    gs.copy(), max_states=config.max_states, verbose=False, bot_id=bid)
                test_actions[bid] = bot_acts

                del searcher
                if config.device == 'cuda':
                    torch.cuda.empty_cache()

        # Verify
        combined = _make_combined(test_actions, num_bots)
        verified = cpu_verify(gs.copy(), all_orders, combined, num_bots)

        if verified > best_score:
            if config.verbose:
                print(f"    LNS improved: {best_score} -> {verified} "
                      f"(+{verified - best_score})", file=sys.stderr)
            best_score = verified
            best_actions = combined
            bot_actions = dict(test_actions)
        else:
            if config.verbose and lns_round < 5:
                print(f"    LNS no improvement ({verified} vs {best_score})",
                      file=sys.stderr)

    return best_score, best_actions


def solve_from_capture(difficulty: str, gpu: str = 'auto',
                       max_time_s: float | None = None,
                       ) -> tuple[int, list]:
    """Convenience: load capture and solve with auto-detected GPU params."""
    capture = load_capture(difficulty)
    if not capture:
        print(f"No capture data for {difficulty}", file=sys.stderr)
        return 0, []

    params = get_params(difficulty, gpu)

    config = SquadConfig(
        difficulty=difficulty,
        max_states=params.max_states,
        joint_states=params.joint_states,
        joint_squad_size=params.joint_squad_size,
        explore_states=params.explore_states,
        pass1_orderings=params.pass1_orderings,
        refine_iters=params.refine_iters,
        lns_rounds=params.lns_rounds,
        max_dp_bots=params.max_dp_bots,
        max_time_s=max_time_s,
        speed_bonus=params.speed_bonus,
    )

    score, actions = solve_squads(capture, config)

    if score > 0:
        saved = save_solution(difficulty, score, actions)
        if saved:
            print(f"Saved: {difficulty} score={score}", file=sys.stderr)

    return score, actions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='B200 Squad Solver')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--gpu', default='auto', choices=['auto', 'b200', '5090', 'generic'])
    parser.add_argument('--max-time', type=float, default=None)
    parser.add_argument('--max-states', type=int, default=None)
    parser.add_argument('--joint-states', type=int, default=None)
    parser.add_argument('--orderings', type=int, default=None)
    args = parser.parse_args()

    params = get_params(args.difficulty, args.gpu)
    if args.max_states:
        params.max_states = args.max_states
    if args.joint_states:
        params.joint_states = args.joint_states
    if args.orderings:
        params.pass1_orderings = args.orderings

    score, actions = solve_from_capture(
        args.difficulty, gpu=args.gpu, max_time_s=args.max_time)
    print(f"Final score: {score}")

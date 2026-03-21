"""CMA-ES optimizer for Astar Island prediction pipeline.

Uses Covariance Matrix Adaptation Evolution Strategy instead of
Metropolis-Hastings perturbation. CMA-ES learns parameter correlations
and navigates narrow valleys in the fitness landscape much better.

Usage:
    python autoloop_cmaes.py              # Run until Ctrl+C
    python autoloop_cmaes.py --sigma 0.3  # Initial step size
    python autoloop_cmaes.py --robust 0.5 # Variance penalty: Mean - λ*StdDev
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cma
import numpy as np

from autoloop import DEFAULT_PARAMS, PARAM_SPACE, ExperimentLog
from autoloop_fast import FastHarness, ROUND_NAMES, BOOM_ROUNDS, compute_score

LOG_PATH = Path(__file__).parent / "data" / "autoloop_cmaes_log.jsonl"

# CMA-ES only optimizes continuous params. Categorical/int params use best known values.
# We select the most impactful continuous parameters.
CMA_PARAMS = [
    "fk_prior_weight",
    "fk_max_strength",
    "mult_power",
    "mult_power_sett",
    "mult_power_port",
    "mult_smooth",
    "mult_sett_lo",
    "mult_sett_hi",
    "mult_empty_lo",
    "mult_empty_hi",
    "floor_nonzero",
    "temp_low",
    "temp_high",
    "temp_ent_lo",
    "temp_ent_hi",
    "smooth_alpha",
    "dist_exp_damp",
    "regime_prior_scale",
    "cluster_sett_boost",
    "cal_fine_base",
    "cal_fine_divisor",
    "cal_fine_max",
    "cal_coarse_base",
    "cal_coarse_divisor",
    "cal_coarse_max",
    "cal_base_base",
    "cal_base_divisor",
    "cal_base_max",
    "cal_global_weight",
]


def get_bounds():
    """Get lower/upper bounds for CMA params."""
    lo = []
    hi = []
    for key in CMA_PARAMS:
        spec = PARAM_SPACE[key]
        lo.append(spec["lo"])
        hi.append(spec["hi"])
    return lo, hi


def params_to_vector(params: dict) -> np.ndarray:
    """Convert param dict to CMA-ES vector."""
    return np.array([params.get(k, DEFAULT_PARAMS[k]) for k in CMA_PARAMS])


def vector_to_params(vec: np.ndarray, base_params: dict) -> dict:
    """Convert CMA-ES vector back to param dict."""
    params = dict(base_params)
    lo, hi = get_bounds()
    for i, key in enumerate(CMA_PARAMS):
        params[key] = float(np.clip(vec[i], lo[i], hi[i]))
    return params


def evaluate_with_variance(harness: FastHarness, params: dict, robust_lambda: float = 0.0):
    """Evaluate params, return (neg_score, scores_dict, per_round_scores).

    CMA-ES minimizes, so we negate the score.
    If robust_lambda > 0, use Mean - λ*StdDev as objective.
    """
    scores = harness.evaluate(params)
    per_round = [scores[r] for r in ROUND_NAMES]

    if robust_lambda > 0:
        mean_score = np.mean(per_round)
        std_score = np.std(per_round)
        objective = mean_score - robust_lambda * std_score
    else:
        objective = scores["avg"]

    return -objective, scores, per_round


def main():
    parser = argparse.ArgumentParser(description="CMA-ES optimizer for Astar Island")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Initial step size (fraction of param range)")
    parser.add_argument("--robust", type=float, default=0.0,
                        help="Variance penalty lambda: Score = Mean - λ*StdDev (0=disabled)")
    parser.add_argument("--popsize", type=int, default=0,
                        help="Population size (0=auto)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per round (1=fast, 5=accurate)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best params in MH log")
    args = parser.parse_args()

    print("Loading harness...")
    harness = FastHarness(seeds_per_round=args.seeds)

    # Start from best known params (from MH autoloop)
    mh_log_path = Path(__file__).parent / "data" / "autoloop_fast_log.jsonl"
    base_params = dict(DEFAULT_PARAMS)

    if args.resume and mh_log_path.exists():
        mh_log = ExperimentLog(mh_log_path)
        if mh_log.best_score > 0:
            base_params = dict(mh_log.best_params)
            print(f"Resuming from MH best: {mh_log.best_score:.3f}")

    # Also check best_params.json
    bp_path = Path(__file__).parent / "best_params.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text())
        # Map production keys back to autoloop keys
        prod_to_al = {
            "prior_w": "fk_prior_weight",
            "emp_max": "fk_max_strength",
            "exp_damp": "dist_exp_damp",
            "base_power": "mult_power",
            "T_high": "temp_high",
            "smooth_alpha": "smooth_alpha",
            "floor": "floor_nonzero",
        }
        for prod_key, al_key in prod_to_al.items():
            if prod_key in bp:
                base_params[al_key] = bp[prod_key]

    # Evaluate baseline
    print("Evaluating baseline...")
    neg_base, base_scores, _ = evaluate_with_variance(harness, base_params, args.robust)
    baseline_score = -neg_base
    print(f"Baseline: avg={base_scores['avg']:.3f} boom={base_scores['boom_avg']:.3f} "
          f"nonboom={base_scores['nonboom_avg']:.3f}")
    if args.robust > 0:
        per_round = [base_scores[r] for r in ROUND_NAMES]
        print(f"  Robust objective (λ={args.robust}): {baseline_score:.3f} "
              f"(mean={np.mean(per_round):.3f}, std={np.std(per_round):.3f})")

    # Set up CMA-ES
    x0 = params_to_vector(base_params)
    lo, hi = get_bounds()

    # Scale sigma relative to parameter ranges
    ranges = np.array(hi) - np.array(lo)
    sigma0 = args.sigma  # fraction of range

    opts = {
        'bounds': [lo, hi],
        'maxiter': 100000,
        'verbose': -1,  # quiet; we do our own logging
        'tolfun': 1e-6,
        'CMA_stds': ranges * sigma0,  # per-parameter initial stds
    }
    if args.popsize > 0:
        opts['popsize'] = args.popsize

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    log = ExperimentLog(LOG_PATH)
    best_score = base_scores["avg"]
    best_params = dict(base_params)
    iteration = log.count()
    start_time = time.time()

    # Log baseline
    if iteration == 0:
        log.append({
            "id": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": "cmaes_baseline",
            "params": {},
            "scores_quick": {k: round(v, 3) for k, v in base_scores.items()},
            "scores_full": {k: round(v, 3) for k, v in base_scores.items()},
            "accepted": True,
            "baseline_avg": 0.0,
            "elapsed": 0.0,
        })
        iteration = 1

    print(f"\nStarting CMA-ES optimization ({len(CMA_PARAMS)} params, "
          f"popsize={es.sp.popsize}, sigma={sigma0})")
    if args.robust > 0:
        print(f"Robust mode: Score = Mean - {args.robust}*StdDev")
    print()

    try:
        generation = 0
        while not es.stop():
            generation += 1
            solutions = es.ask()
            fitnesses = []

            gen_start = time.time()
            for sol in solutions:
                params = vector_to_params(sol, base_params)
                try:
                    neg_score, scores, per_round = evaluate_with_variance(
                        harness, params, args.robust)
                    fitnesses.append(neg_score)

                    avg = scores["avg"]
                    accepted = avg > best_score

                    if accepted:
                        best_score = avg
                        best_params = dict(params)

                        # Write best_params.json
                        try:
                            param_map = {
                                "fk_prior_weight": "prior_w",
                                "fk_max_strength": "emp_max",
                                "dist_exp_damp": "exp_damp",
                                "mult_power": "base_power",
                                "temp_high": "T_high",
                                "smooth_alpha": "smooth_alpha",
                                "floor_nonzero": "floor",
                            }
                            bp = {}
                            if bp_path.exists():
                                bp = json.loads(bp_path.read_text())
                            for al_key, prod_key in param_map.items():
                                if al_key in params:
                                    bp[prod_key] = round(params[al_key], 4)
                            bp["score_avg"] = round(avg, 3)
                            bp["score_boom"] = round(scores.get("boom_avg", 0), 3)
                            bp["score_nonboom"] = round(scores.get("nonboom_avg", 0), 3)
                            bp["updated_at"] = datetime.now(timezone.utc).isoformat()
                            bp["source"] = "cmaes"
                            bp["experiment_id"] = iteration
                            bp_path.write_text(json.dumps(bp, indent=2))
                        except Exception:
                            pass

                        print(f"  [{iteration:5d}] ***BEST {best_score:.3f}*** "
                              f"boom={scores['boom_avg']:.1f} nonboom={scores['nonboom_avg']:.1f}")

                    # Log every eval
                    changed = {k: round(params[k], 4) for k in CMA_PARAMS
                               if abs(params[k] - DEFAULT_PARAMS.get(k, 0)) > 1e-6}
                    log.append({
                        "id": iteration,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "name": f"cmaes_gen{generation}",
                        "params": changed,
                        "scores_quick": {k: round(v, 3) for k, v in scores.items()},
                        "scores_full": {k: round(v, 3) for k, v in scores.items()},
                        "accepted": accepted,
                        "baseline_avg": round(best_score, 3),
                        "elapsed": 0.0,
                    })
                    iteration += 1

                except Exception as e:
                    fitnesses.append(0.0)  # worst possible (CMA minimizes)
                    iteration += 1

            es.tell(solutions, fitnesses)

            gen_elapsed = time.time() - gen_start
            total_elapsed = time.time() - start_time
            rate = iteration / max(total_elapsed, 1) * 3600

            # Per-generation summary
            best_gen_fitness = min(fitnesses)
            best_gen_score = -best_gen_fitness
            print(f"[Gen {generation:4d}] best_gen={best_gen_score:.3f} "
                  f"best_ever={best_score:.3f} sigma={es.sigma:.4f} "
                  f"({gen_elapsed:.1f}s, {rate:.0f}/hr)")

    except KeyboardInterrupt:
        pass

    total_elapsed = time.time() - start_time
    print(f"\n\nDone: {iteration} evaluations in {total_elapsed/60:.1f}min")
    print(f"Best score: {best_score:.3f}")
    print(f"\nBest params diff from default:")
    for k in CMA_PARAMS:
        if abs(best_params[k] - DEFAULT_PARAMS.get(k, 0)) > 1e-6:
            print(f"  {k}: {DEFAULT_PARAMS[k]} -> {best_params[k]:.4f}")

    es.result_pretty()


if __name__ == "__main__":
    main()

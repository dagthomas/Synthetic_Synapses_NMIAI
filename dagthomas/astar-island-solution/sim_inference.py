"""CMA-ES parameter inference for the parametric simulator — v2.

Multi-start CMA-ES with regime-specific warm starts.
Fits the hidden parameters per round against GT or observations.

Usage:
    python sim_inference.py                    # Fit all rounds against GT
    python sim_inference.py --round round5     # Fit single round
    python sim_inference.py --quick            # Fewer evaluations
"""
import argparse
import json
import math
import time
from pathlib import Path

import cma
import numpy as np

from sim_data import RoundData, load_round, ALL_ROUNDS, load_observations, terrain_to_class
from sim_model import (
    Simulator, compute_score, PARAM_SPEC, PARAM_NAMES, N_PARAMS,
    params_to_vec, vec_to_params, default_params,
)

RESULTS_DIR = Path(__file__).parent / "data" / "sim_params"

# Regime-specific starting points (learned from fitting 15 rounds)
WARM_STARTS = {
    "collapse": {
        "base_survival": -3.0, "expansion_str": 0.3, "expansion_scale": 1.5,
        "decay_power": 3.0, "max_reach": 3.0, "coastal_mod": -0.5,
        "food_coeff": 0.5, "cluster_pen": -0.3, "cluster_optimal": 2.0,
        "cluster_quad": -0.2, "ruin_rate": 0.1, "port_factor": 0.1,
        "forest_resist": 0.3, "forest_clear": 0.5, "forest_reclaim": 0.1,
        "exp_death": 0.7,
    },
    "moderate": {
        "base_survival": -0.5, "expansion_str": 0.5, "expansion_scale": 2.5,
        "decay_power": 2.5, "max_reach": 5.0, "coastal_mod": -0.3,
        "food_coeff": 0.5, "cluster_pen": -0.3, "cluster_optimal": 2.0,
        "cluster_quad": -0.2, "ruin_rate": 0.4, "port_factor": 0.25,
        "forest_resist": 0.2, "forest_clear": 0.3, "forest_reclaim": 0.05,
        "exp_death": 0.4,
    },
    "boom": {
        "base_survival": 0.0, "expansion_str": 0.8, "expansion_scale": 4.0,
        "decay_power": 1.5, "max_reach": 8.0, "coastal_mod": -0.3,
        "food_coeff": 0.5, "cluster_pen": -0.3, "cluster_optimal": 2.0,
        "cluster_quad": -0.2, "ruin_rate": 0.2, "port_factor": 0.3,
        "forest_resist": 0.1, "forest_clear": 0.4, "forest_reclaim": 0.03,
        "exp_death": 0.3,
    },
}


def get_bounds():
    lo = [PARAM_SPEC[k][1] for k in PARAM_NAMES]
    hi = [PARAM_SPEC[k][2] for k in PARAM_NAMES]
    return lo, hi


def detect_regime(rd: RoundData) -> str:
    """Detect regime from GT settlement density."""
    if rd.ground_truth is not None:
        sett_prob = rd.ground_truth[:, :, 1].mean()
        if sett_prob < 0.03:
            return "collapse"
        elif sett_prob > 0.15:
            return "boom"
    return "moderate"


def detect_regime_from_obs(observations: list[dict], terrain: np.ndarray) -> str:
    """Detect regime from observations (for live rounds)."""
    if not observations:
        return "moderate"
    sett_count = 0
    total_count = 0
    H, W = terrain.shape
    for obs in observations:
        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < H and 0 <= mx < W:
                    code = grid[row][col]
                    if code not in (10, 5):  # skip ocean/mountain
                        total_count += 1
                        if terrain_to_class(code) == 1:
                            sett_count += 1
    if total_count == 0:
        return "moderate"
    sett_rate = sett_count / total_count
    if sett_rate < 0.03:
        return "collapse"
    elif sett_rate > 0.15:
        return "boom"
    return "moderate"


def _run_cma(objective, x0, lo, hi, max_evals, sigma=0.5, cma_seed=42):
    """Run a single CMA-ES optimization. Returns (best_vec, best_score)."""
    opts = cma.CMAOptions()
    opts["bounds"] = [lo, hi]
    opts["maxfevals"] = max_evals
    opts["verbose"] = -9
    opts["seed"] = cma_seed
    opts["tolfun"] = 0.01
    opts["tolx"] = 1e-4

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

    return es.result.xbest, -es.result.fbest


def fit_to_gt(rd: RoundData, n_sims: int = 500, sigma: float = 0.5,
              max_evals: int = 300, verbose: bool = True,
              multi_start: bool = True, use_gpu: bool = False) -> tuple[dict, float]:
    """Multi-start CMA-ES fitting against ground truth.

    Runs 3 starts (collapse/moderate/boom warm starts) and keeps the best.
    use_gpu: Use PyTorch GPU simulator (much faster for high n_sims).
    """
    if use_gpu:
        try:
            from sim_model_gpu import GPUSimulator
            sim = GPUSimulator(rd)
        except Exception:
            sim = Simulator(rd)
    else:
        sim = Simulator(rd)
    gt = rd.ground_truth
    lo, hi = get_bounds()

    def objective(vec):
        params = vec_to_params(vec)
        pred = sim.run(params, n_sims=n_sims, seed=42)
        score = compute_score(gt, pred)
        return -score

    if multi_start:
        # Try all 3 regime-specific starts
        best_vec, best_score = None, -1.0
        regimes = ["collapse", "moderate", "boom"]
        detected = detect_regime(rd)
        # Put detected regime first (gets more budget)
        regimes = [detected] + [r for r in regimes if r != detected]
        budgets = [max_evals // 2, max_evals // 4, max_evals // 4]

        for regime, budget in zip(regimes, budgets):
            x0 = params_to_vec(WARM_STARTS[regime])
            vec, score = _run_cma(objective, x0, lo, hi, budget,
                                  sigma=sigma, cma_seed=42 + hash(regime) % 1000)
            if score > best_score:
                best_vec, best_score = vec, score
                if verbose:
                    print(f"    {regime}: score={score:.2f}")

        # Polish: refine best with smaller sigma
        if best_vec is not None:
            vec, score = _run_cma(objective, best_vec.copy(), lo, hi,
                                  max_evals // 4, sigma=sigma * 0.3,
                                  cma_seed=123)
            if score > best_score:
                best_vec, best_score = vec, score
                if verbose:
                    print(f"    polish: score={score:.2f}")
    else:
        x0 = params_to_vec(default_params())
        best_vec, best_score = _run_cma(objective, x0, lo, hi, max_evals,
                                         sigma=sigma)

    best_params = vec_to_params(best_vec)
    if verbose:
        print(f"    Final: score={best_score:.2f}")

    return best_params, best_score


def _load_transfer_data():
    """Load GT-fitted params from all historical rounds."""
    path = Path(__file__).parent / "data" / "sim_params" / "transfer_data.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _knn_warm_start(obs_features: dict, transfer_data: list, k: int = 3) -> dict:
    """Find k nearest historical rounds by observable features, average their params.

    obs_features: dict with keys 'survival_rate', 'sett_rate', 'ruin_rate'
    Returns averaged params from k nearest rounds.
    """
    def feat_vec(record):
        f = record["features"]
        return np.array([f["gt_survival"], f["gt_sett_rate"], f["gt_ruin_rate"]])

    target = np.array([
        obs_features.get("survival_rate", 0.3),
        obs_features.get("sett_rate", 0.1),
        obs_features.get("ruin_rate", 0.01),
    ])

    dists = [(np.linalg.norm(target - feat_vec(r)), r) for r in transfer_data]
    dists.sort(key=lambda x: x[0])

    weights = [1.0 / (d + 0.01) for d, _ in dists[:k]]
    total_w = sum(weights)

    params = {}
    for pname in PARAM_NAMES:
        val = sum(w * dists[i][1]["params"][pname] for i, w in enumerate(weights))
        val /= total_w
        lo, hi = PARAM_SPEC[pname][1], PARAM_SPEC[pname][2]
        params[pname] = float(np.clip(val, lo, hi))

    neighbors = [dists[i][1]["round"] for i in range(k)]
    return params, neighbors


def _compute_obs_features(observations: list[dict], terrain: np.ndarray) -> dict:
    """Compute observable features from API observations for KNN matching."""
    H, W = terrain.shape
    sett_count = 0
    ruin_count = 0
    dynamic_count = 0
    initial_sett_alive = 0
    initial_sett_total = 0

    initial_sett_cells = set()
    for y in range(H):
        for x in range(W):
            if terrain[y, x] in (1, 2):
                initial_sett_cells.add((y, x))

    for obs in observations:
        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < H and 0 <= mx < W:
                    code = grid[row][col]
                    if code not in (10, 5):  # skip ocean/mountain
                        dynamic_count += 1
                        cls = terrain_to_class(code)
                        if cls == 1:
                            sett_count += 1
                        elif cls == 3:
                            ruin_count += 1
                        if (my, mx) in initial_sett_cells:
                            initial_sett_total += 1
                            if cls in (1, 2):
                                initial_sett_alive += 1

    return {
        "survival_rate": initial_sett_alive / max(initial_sett_total, 1),
        "sett_rate": sett_count / max(dynamic_count, 1),
        "ruin_rate": ruin_count / max(dynamic_count, 1),
    }


def fit_to_observations(rd: RoundData, observations: list[dict],
                        n_sims: int = 300, sigma: float = 0.5,
                        max_evals: int = 200, verbose: bool = True,
                        use_gpu: bool = False) -> tuple[dict, float]:
    """Fit from API observations using KNN warm-start + CMA-ES refinement.

    1. Compute observable features (survival rate, settlement rate, ruin rate)
    2. Find 3 nearest historical rounds by these features
    3. Average their GT-fitted params as warm start
    4. Refine with short CMA-ES
    """
    if use_gpu:
        try:
            from sim_model_gpu import GPUSimulator
            sim = GPUSimulator(rd)
        except Exception:
            sim = Simulator(rd)
    else:
        sim = Simulator(rd)
    H, W = rd.terrain.shape

    obs_cells = []
    for obs in observations:
        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < H and 0 <= mx < W:
                    cls = terrain_to_class(grid[row][col])
                    obs_cells.append((my, mx, cls))

    if not obs_cells:
        if verbose:
            print("  No observations available")
        return default_params(), 0.0

    obs_y = np.array([c[0] for c in obs_cells])
    obs_x = np.array([c[1] for c in obs_cells])
    obs_cls = np.array([c[2] for c in obs_cells])

    if verbose:
        print(f"  {len(obs_cells)} observed cells")

    lo, hi = get_bounds()

    def objective(vec):
        params = vec_to_params(vec)
        pred = sim.run(params, n_sims=n_sims, seed=42)
        pred_safe = np.maximum(pred, 1e-6)
        probs = pred_safe[obs_y, obs_x, obs_cls]
        ll = np.mean(np.log(probs))
        return -ll

    # Try KNN warm-start from historical data
    transfer_data = _load_transfer_data()
    knn_start = None
    if transfer_data:
        obs_features = _compute_obs_features(observations, rd.terrain)
        knn_params, neighbors = _knn_warm_start(obs_features, transfer_data, k=3)
        knn_start = params_to_vec(knn_params)
        if verbose:
            print(f"  KNN neighbors: {neighbors}")
            print(f"  Obs features: surv={obs_features['survival_rate']:.2f} "
                  f"sett={obs_features['sett_rate']:.3f} ruin={obs_features['ruin_rate']:.3f}")

    # Strategy: KNN warm-start (if available) + regime warm-start, pick best
    regime = detect_regime_from_obs(observations, rd.terrain)
    starts = []
    if knn_start is not None:
        starts.append(("knn", knn_start, max_evals * 2 // 3, 0.2))  # tighter sigma
    starts.append((regime, params_to_vec(WARM_STARTS[regime]), max_evals // 3, 0.5))

    best_vec, best_ll = None, float("inf")
    for name, x0, budget, sig in starts:
        try:
            opts = cma.CMAOptions()
            opts["bounds"] = [lo, hi]
            opts["maxfevals"] = budget
            opts["verbose"] = -9
            opts["seed"] = 42 + hash(name) % 1000
            opts["tolfun"] = 0.001
            es = cma.CMAEvolutionStrategy(x0, sig, opts)
            while not es.stop():
                solutions = es.ask()
                fitnesses = [objective(s) for s in solutions]
                es.tell(solutions, fitnesses)
            if es.result.fbest < best_ll:
                best_vec = es.result.xbest
                best_ll = es.result.fbest
                if verbose:
                    print(f"    {name}: ll={-best_ll:.4f}")
        except Exception:
            continue

    best_params = vec_to_params(best_vec) if best_vec is not None else default_params()

    # Compute actual score if GT available
    score = 0.0
    if rd.ground_truth is not None:
        pred = sim.run(best_params, n_sims=500, seed=42)
        score = compute_score(rd.ground_truth, pred)

    if verbose:
        print(f"    Converged: score={score:.2f}")

    return best_params, score


def fit_all_rounds(n_sims: int = 500, max_evals: int = 400):
    """Fit all rounds against GT with multi-start CMA-ES. Save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_scores = {}

    for rname in ALL_ROUNDS:
        rd = load_round(rname, 0)
        if rd is None or rd.ground_truth is None:
            continue
        print(f"\n{rname}:")
        params, score = fit_to_gt(rd, n_sims=n_sims, max_evals=max_evals)
        all_scores[rname] = score

        result_path = RESULTS_DIR / f"{rname}.json"
        result_path.write_text(json.dumps({
            "params": params, "score": score, "n_sims": n_sims,
        }, indent=2))

    if all_scores:
        scores = list(all_scores.values())
        print(f"\n{'='*50}")
        print(f"Average: {np.mean(scores):.2f}  Min: {min(scores):.2f}  Max: {max(scores):.2f}")
        for rname, score in sorted(all_scores.items()):
            print(f"  {rname}: {score:.2f}")

    return all_scores


# Adaptive alpha: learned from backtest data (updated 2026-03-22)
# Boom: sim captures spatial dynamics well from obs
# Collapse: obs-fitting is unreliable, keep alpha low
REGIME_ALPHAS = {
    "collapse": 0.15,
    "moderate": 0.30,
    "boom": 0.65,
}


def get_adaptive_alpha(regime: str) -> float:
    """Get blend alpha based on detected regime."""
    return REGIME_ALPHAS.get(regime, 0.20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--loo", action="store_true")
    args = parser.parse_args()

    n_sims = 300 if args.quick else 500
    max_evals = 150 if args.quick else 400

    if args.all:
        fit_all_rounds(n_sims=n_sims, max_evals=max_evals)
    elif args.round:
        rd = load_round(args.round, 0)
        if rd is None:
            print(f"Could not load {args.round}")
            exit(1)
        t0 = time.perf_counter()
        params, score = fit_to_gt(rd, n_sims=n_sims, max_evals=max_evals)
        t1 = time.perf_counter()
        print(f"\nBest: score={score:.2f} ({t1-t0:.1f}s)")
        for k, v in params.items():
            print(f"  {k}: {v:.4f}")
    else:
        rd = load_round("round5", 0)
        print("Fitting round5...")
        params, score = fit_to_gt(rd, n_sims=n_sims, max_evals=max_evals)
        print(f"Score: {score:.2f}")

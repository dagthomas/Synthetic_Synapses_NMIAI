"""Analyze: Can we estimate the spatial decay rate (lambda) from observations?

The simulator uses: P(expand | d) = expansion_str * exp(-(d/scale)^power)
We approximate this as: P(sett | d) = a * exp(-lambda * d)

This script:
1. Fits lambda from ground truth (all 40x40x5 seeds)
2. Fits lambda from observations (50 queries per round, 15x15 viewports)
3. Tests convergence: how many queries to get good lambda estimate?
4. Tests: can lambda be estimated from initial state alone (no queries)?
"""

import json
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_cdt

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"
SIM_PARAMS = Path(__file__).parent / "data" / "sim_params" / "transfer_data.json"

# Map round UUID -> round number
ROUND_MAP = {}
for d in OBS_DIR.iterdir():
    if d.is_dir() and (d / "initial_states.json").exists():
        info = json.loads((d / "initial_states.json").read_text())
        rnum = info.get("round_number")
        if rnum:
            ROUND_MAP[f"round{rnum}"] = d.name

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def terrain_to_class(code):
    return TERRAIN_TO_CLASS.get(code, 0)


def exp_decay(d, a, lam):
    return a * np.exp(-lam * d)


def fit_lambda(distances, rates, counts, min_count=10, min_rate=0.0005):
    """Fit exponential decay to distance-rate data. Returns (a, lam, r2) or None."""
    distances = np.asarray(distances, dtype=float)
    rates = np.asarray(rates, dtype=float)
    counts = np.asarray(counts, dtype=float)

    mask = (distances >= 1) & (counts >= min_count) & (rates > min_rate)
    if mask.sum() < 3:
        return None

    try:
        popt, _ = curve_fit(
            exp_decay, distances[mask], rates[mask],
            p0=[0.3, 0.3],
            bounds=([0.001, 0.01], [1.0, 5.0]),
            sigma=1.0 / np.sqrt(counts[mask]),
            maxfev=5000,
        )
        a, lam = popt
        pred = exp_decay(distances[mask], a, lam)
        ss_res = np.sum((rates[mask] - pred) ** 2)
        ss_tot = np.sum((rates[mask] - rates[mask].mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return a, lam, r2
    except Exception:
        return None


def get_gt_lambda(round_name):
    """Fit lambda from ground truth data (calibration analysis files)."""
    round_dir = DATA_DIR / round_name
    if not round_dir.exists():
        return None

    detail = json.loads((round_dir / "round_detail.json").read_text())
    dist_bins = defaultdict(list)

    for si in range(detail["seeds_count"]):
        analysis_path = round_dir / f"analysis_seed_{si}.json"
        if not analysis_path.exists():
            continue

        analysis = json.loads(analysis_path.read_text())
        terrain = np.array(analysis["initial_grid"], dtype=int)
        gt = np.array(analysis["ground_truth"])  # (40, 40, 6)

        # Initial settlements
        is_sett = (terrain == 1) | (terrain == 2)
        if not is_sett.any():
            continue

        dist_map = distance_transform_cdt(~is_sett, metric="taxicab")

        h, w = terrain.shape
        for y in range(h):
            for x in range(w):
                if terrain[y, x] in (10, 5):  # skip ocean/mountain
                    continue
                d = int(dist_map[y, x])
                if d == 0:
                    continue
                sett_prob = float(gt[y, x, 1]) + float(gt[y, x, 2])  # settlement + port
                dist_bins[d].append(sett_prob)

    if not dist_bins:
        return None

    distances = sorted(dist_bins.keys())
    rates = [np.mean(dist_bins[d]) for d in distances]
    counts = [len(dist_bins[d]) for d in distances]

    result = fit_lambda(distances, rates, counts)
    if result is None:
        return None

    a, lam, r2 = result
    return {
        "round": round_name,
        "a_gt": a, "lambda_gt": lam, "r2_gt": r2,
        "dist_bins": {d: (np.mean(dist_bins[d]), len(dist_bins[d])) for d in distances},
    }


def get_obs_lambda(round_name, max_queries=None, seed_filter=None):
    """Fit lambda from observation queries.

    max_queries: only use first N queries (by query number)
    seed_filter: only use observations from this seed index
    """
    if round_name not in ROUND_MAP:
        return None

    rid = ROUND_MAP[round_name]
    obs_dir = OBS_DIR / rid
    if not obs_dir.exists():
        return None

    # Load initial states for terrain grids
    info = json.loads((obs_dir / "initial_states.json").read_text())

    # Load all observation files
    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    if not obs_files:
        return None

    # Parse and sort by query number
    obs_data = []
    for op in obs_files:
        obs = json.loads(op.read_text())
        obs_data.append(obs)

    # Sort by query number
    obs_data.sort(key=lambda o: o["query_num"])

    # Filter by max queries
    if max_queries is not None:
        obs_data = obs_data[:max_queries]

    # Filter by seed
    if seed_filter is not None:
        obs_data = [o for o in obs_data if o["seed_index"] == seed_filter]

    dist_bins = defaultdict(lambda: [0, 0])  # distance -> [settlement_count, total_count]

    for obs in obs_data:
        sid = obs["seed_index"]
        terrain = np.array(info["initial_states"][sid]["grid"], dtype=int)

        # Find initial settlements
        is_sett = (terrain == 1) | (terrain == 2)
        if not is_sett.any():
            continue

        dist_map = distance_transform_cdt(~is_sett, metric="taxicab")

        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < 40 and 0 <= mx < 40:
                    if terrain[my, mx] in (10, 5):  # skip ocean/mountain
                        continue
                    d = int(dist_map[my, mx])
                    if d == 0:
                        continue
                    cls = terrain_to_class(grid[row][col])
                    is_s = 1 if cls in (1, 2) else 0  # settlement or port
                    dist_bins[d][0] += is_s
                    dist_bins[d][1] += 1

    if not dist_bins:
        return None

    distances = sorted(dist_bins.keys())
    rates = [dist_bins[d][0] / max(dist_bins[d][1], 1) for d in distances]
    counts = [dist_bins[d][1] for d in distances]

    result = fit_lambda(distances, rates, counts, min_count=5, min_rate=0.0001)
    if result is None:
        return None

    a, lam, r2 = result
    total_cells = sum(c for c in counts)
    total_sett = sum(dist_bins[d][0] for d in distances)
    return {
        "round": round_name,
        "n_queries": len(obs_data) if seed_filter is None else len(obs_data),
        "a_obs": a, "lambda_obs": lam, "r2_obs": r2,
        "total_cells": total_cells, "total_settlements": total_sett,
    }


def get_initial_state_features(round_name):
    """Extract features from initial state that might correlate with lambda."""
    if round_name not in ROUND_MAP:
        round_dir = DATA_DIR / round_name
        if not round_dir.exists():
            return None
        detail = json.loads((round_dir / "round_detail.json").read_text())
        grids = []
        for si in range(detail["seeds_count"]):
            analysis_path = round_dir / f"analysis_seed_{si}.json"
            if analysis_path.exists():
                a = json.loads(analysis_path.read_text())
                grids.append(np.array(a["initial_grid"], dtype=int))
        if not grids:
            return None
    else:
        rid = ROUND_MAP[round_name]
        obs_dir = OBS_DIR / rid
        info = json.loads((obs_dir / "initial_states.json").read_text())
        grids = [np.array(s["grid"], dtype=int) for s in info["initial_states"]]

    features = []
    for terrain in grids:
        n_sett = np.sum((terrain == 1) | (terrain == 2))
        n_forest = np.sum(terrain == 4)
        n_ocean = np.sum(terrain == 10)
        n_mountain = np.sum(terrain == 5)
        n_buildable = np.sum((terrain != 10) & (terrain != 5))

        # Settlement density on buildable land
        sett_density = n_sett / max(n_buildable, 1)
        forest_frac = n_forest / max(n_buildable, 1)

        # Average distance between settlements
        is_sett = (terrain == 1) | (terrain == 2)
        if is_sett.any():
            dist_map = distance_transform_cdt(~is_sett, metric="taxicab")
            sett_locs = np.argwhere(is_sett)
            # Mean inter-settlement distance (using nearest neighbor)
            if len(sett_locs) > 1:
                from scipy.spatial.distance import cdist
                dists = cdist(sett_locs, sett_locs, metric="cityblock")
                np.fill_diagonal(dists, 999)
                mean_nn_dist = np.mean(np.min(dists, axis=1))
            else:
                mean_nn_dist = 20.0

            # Coastal settlement fraction
            coastal_count = 0
            for sy, sx in sett_locs:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < 40 and 0 <= nx < 40 and terrain[ny, nx] == 10:
                        coastal_count += 1
                        break
            coastal_frac = coastal_count / max(len(sett_locs), 1)
        else:
            mean_nn_dist = 20.0
            coastal_frac = 0.0

        features.append({
            "n_sett": int(n_sett),
            "sett_density": sett_density,
            "forest_frac": forest_frac,
            "ocean_frac": n_ocean / 1600,
            "mountain_frac": n_mountain / 1600,
            "mean_nn_dist": mean_nn_dist,
            "coastal_frac": coastal_frac,
        })

    # Average across seeds
    avg = {}
    for k in features[0]:
        avg[k] = np.mean([f[k] for f in features])
    return avg


def load_sim_params():
    """Load fitted sim parameters per round."""
    if not SIM_PARAMS.exists():
        return {}
    data = json.loads(SIM_PARAMS.read_text())
    params = {}
    for entry in data:
        params[entry["round"]] = entry["params"]
    return params


def main():
    # Get all rounds that have calibration data
    all_rounds = sorted(
        [d.name for d in DATA_DIR.iterdir() if d.is_dir()],
        key=lambda r: int(r.replace("round", ""))
    )

    sim_params = load_sim_params()

    print("=" * 80)
    print("PART 1: GROUND TRUTH LAMBDA (from full 40x40 ground truth)")
    print("=" * 80)

    gt_results = {}
    for rn in all_rounds:
        r = get_gt_lambda(rn)
        if r:
            gt_results[rn] = r

    print(f"\n{'Round':<10} {'a_GT':>8} {'lam_GT':>8} {'R2':>6}  {'sim_scale':>10} {'sim_power':>10} {'sim_str':>8} {'sim_reach':>10}")
    print("-" * 90)
    for rn in all_rounds:
        if rn not in gt_results:
            print(f"{rn:<10} {'no GT':>8}")
            continue
        r = gt_results[rn]
        sp = sim_params.get(rn, {})
        sim_scale = sp.get("expansion_scale", 0)
        sim_power = sp.get("decay_power", 0)
        sim_str = sp.get("expansion_str", 0)
        sim_reach = sp.get("max_reach", 0)
        print(f"{rn:<10} {r['a_gt']:8.4f} {r['lambda_gt']:8.4f} {r['r2_gt']:6.3f}"
              f"  {sim_scale:10.3f} {sim_power:10.3f} {sim_str:8.4f} {sim_reach:10.3f}")

    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 2: OBSERVATION-BASED LAMBDA (50 queries)")
    print("=" * 80)

    obs_results = {}
    for rn in all_rounds:
        r = get_obs_lambda(rn, max_queries=50)
        if r:
            obs_results[rn] = r

    print(f"\n{'Round':<10} {'lam_GT':>8} {'lam_obs':>8} {'Error%':>8} {'R2_obs':>7} {'Cells':>7} {'Setts':>6}")
    print("-" * 65)
    errors = []
    for rn in all_rounds:
        gt = gt_results.get(rn)
        obs = obs_results.get(rn)
        if gt and obs:
            err = abs(obs["lambda_obs"] - gt["lambda_gt"]) / max(gt["lambda_gt"], 0.01) * 100
            errors.append(err)
            print(f"{rn:<10} {gt['lambda_gt']:8.4f} {obs['lambda_obs']:8.4f} {err:7.1f}% "
                  f"{obs['r2_obs']:7.3f} {obs['total_cells']:7d} {obs['total_settlements']:6d}")
        elif gt:
            print(f"{rn:<10} {gt['lambda_gt']:8.4f} {'no obs':>8}")
        elif obs:
            print(f"{rn:<10} {'no GT':>8} {obs['lambda_obs']:8.4f}")

    if errors:
        print(f"\nMean absolute error: {np.mean(errors):.1f}%")
        print(f"Median absolute error: {np.median(errors):.1f}%")

    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 3: CONVERGENCE — How many queries for good lambda?")
    print("=" * 80)

    query_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # For each round, fit lambda at different query counts
    convergence = {}  # round -> {n_queries -> lambda}
    for rn in all_rounds:
        if rn not in gt_results or rn not in ROUND_MAP:
            continue
        convergence[rn] = {}
        for nq in query_counts:
            r = get_obs_lambda(rn, max_queries=nq)
            if r:
                convergence[rn][nq] = r["lambda_obs"]
            else:
                convergence[rn][nq] = None

    # Print convergence table
    print(f"\n{'Round':<10} {'lam_GT':>7}", end="")
    for nq in query_counts:
        print(f" {'q=' + str(nq):>7}", end="")
    print()
    print("-" * (18 + 8 * len(query_counts)))

    for rn in sorted(convergence.keys(), key=lambda r: int(r.replace("round", ""))):
        gt_lam = gt_results[rn]["lambda_gt"]
        print(f"{rn:<10} {gt_lam:7.3f}", end="")
        for nq in query_counts:
            lam = convergence[rn].get(nq)
            if lam is not None:
                print(f" {lam:7.3f}", end="")
            else:
                print(f" {'---':>7}", end="")
        print()

    # Error at 25 vs 50 queries
    print(f"\n{'Round':<10} {'lam_GT':>8} {'lam@25':>8} {'Err@25':>8} {'lam@50':>8} {'Err@50':>8}")
    print("-" * 55)
    errs_25 = []
    errs_50 = []
    for rn in sorted(convergence.keys(), key=lambda r: int(r.replace("round", ""))):
        gt_lam = gt_results[rn]["lambda_gt"]
        lam_25 = convergence[rn].get(25)
        lam_50 = convergence[rn].get(50)
        e25 = abs(lam_25 - gt_lam) / max(gt_lam, 0.01) * 100 if lam_25 else None
        e50 = abs(lam_50 - gt_lam) / max(gt_lam, 0.01) * 100 if lam_50 else None
        if e25 is not None: errs_25.append(e25)
        if e50 is not None: errs_50.append(e50)
        print(f"{rn:<10} {gt_lam:8.4f} "
              f"{lam_25:8.4f} {e25:7.1f}% " if lam_25 else f"{rn:<10} {gt_lam:8.4f} {'---':>8} {'---':>8} ",
              end="")
        if lam_50:
            print(f"{lam_50:8.4f} {e50:7.1f}%")
        else:
            print(f"{'---':>8} {'---':>8}")

    if errs_25:
        print(f"\nAt 25 queries (Phase 1): mean error = {np.mean(errs_25):.1f}%, median = {np.median(errs_25):.1f}%")
    if errs_50:
        print(f"At 50 queries (all):     mean error = {np.mean(errs_50):.1f}%, median = {np.median(errs_50):.1f}%")

    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 4: LAMBDA FROM INITIAL STATE ALONE (no queries)")
    print("=" * 80)
    print("\nCan initial-state features predict lambda?")

    feat_data = []
    for rn in all_rounds:
        if rn not in gt_results:
            continue
        feats = get_initial_state_features(rn)
        if feats:
            feats["round"] = rn
            feats["lambda_gt"] = gt_results[rn]["lambda_gt"]
            feats["a_gt"] = gt_results[rn]["a_gt"]
            feat_data.append(feats)

    if feat_data:
        print(f"\n{'Round':<10} {'lam_GT':>8} {'n_sett':>7} {'density':>8} {'forest':>8} {'nn_dist':>8} {'coastal':>8}")
        print("-" * 65)
        for f in feat_data:
            print(f"{f['round']:<10} {f['lambda_gt']:8.4f} {f['n_sett']:7.1f} "
                  f"{f['sett_density']:8.4f} {f['forest_frac']:8.4f} "
                  f"{f['mean_nn_dist']:8.2f} {f['coastal_frac']:8.3f}")

        # Compute correlations
        lams = [f["lambda_gt"] for f in feat_data]
        feature_names = ["n_sett", "sett_density", "forest_frac", "ocean_frac",
                         "mountain_frac", "mean_nn_dist", "coastal_frac"]

        print(f"\nCorrelation with lambda_GT:")
        print("-" * 40)
        for fn in feature_names:
            vals = [f[fn] for f in feat_data]
            if np.std(vals) > 0 and np.std(lams) > 0:
                corr = np.corrcoef(vals, lams)[0, 1]
                print(f"  {fn:<20} r = {corr:+.3f}")
            else:
                print(f"  {fn:<20} constant")

    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 5: DISTANCE-PROBABILITY CURVES (GT vs Obs)")
    print("=" * 80)

    # For a few rounds, show the actual P(sett|d) curves
    sample_rounds = [rn for rn in ["round2", "round5", "round7", "round10", "round14"]
                     if rn in gt_results and rn in obs_results]

    for rn in sample_rounds:
        gt = gt_results[rn]
        print(f"\n--- {rn} (lam_GT={gt['lambda_gt']:.3f}, a_GT={gt['a_gt']:.3f}) ---")
        print(f"{'Dist':>5} {'P_GT':>8} {'P_fit':>8} {'P_obs':>8} {'N_GT':>6} {'N_obs':>6}")

        # Get obs distance data
        obs_bins = _get_obs_distance_bins(rn)

        for d in range(1, 20):
            gt_rate, gt_count = gt["dist_bins"].get(d, (0, 0))
            gt_fit = exp_decay(d, gt["a_gt"], gt["lambda_gt"])

            if obs_bins and d in obs_bins:
                s, t = obs_bins[d]
                obs_rate = s / max(t, 1)
                obs_n = t
            else:
                obs_rate = float("nan")
                obs_n = 0

            print(f"{d:5d} {gt_rate:8.4f} {gt_fit:8.4f} {obs_rate:8.4f} {gt_count:6d} {obs_n:6d}")


def _get_obs_distance_bins(round_name):
    """Helper to get raw distance bins from observations."""
    if round_name not in ROUND_MAP:
        return None

    rid = ROUND_MAP[round_name]
    obs_dir = OBS_DIR / rid
    info = json.loads((obs_dir / "initial_states.json").read_text())

    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    dist_bins = defaultdict(lambda: [0, 0])

    for op in obs_files:
        obs = json.loads(op.read_text())
        sid = obs["seed_index"]
        terrain = np.array(info["initial_states"][sid]["grid"], dtype=int)

        is_sett = (terrain == 1) | (terrain == 2)
        if not is_sett.any():
            continue
        dist_map = distance_transform_cdt(~is_sett, metric="taxicab")

        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < 40 and 0 <= mx < 40:
                    if terrain[my, mx] in (10, 5):
                        continue
                    d = int(dist_map[my, mx])
                    if d == 0:
                        continue
                    cls = terrain_to_class(grid[row][col])
                    is_s = 1 if cls in (1, 2) else 0
                    dist_bins[d][0] += is_s
                    dist_bins[d][1] += 1

    return dict(dist_bins)


if __name__ == "__main__":
    main()

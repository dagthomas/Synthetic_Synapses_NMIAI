"""Test: Fit per-round spatial decay lam from observations and ground truth.

The hypothesis: the Norse simulator uses P(sett|d) = vigor * exp(-lam*d),
and lam varies per round. Our current pipeline uses distance BUCKETS which
average over all rounds' lam values. Fitting lam per-round should improve
boom predictions where lam is low (settlements spread far).

This script:
1. Extracts the actual P(sett|d) curve from each round's ground truth
2. Fits vigor and lam for each round
3. Shows how much lam varies between rounds
4. Tests if lam can be estimated from 50 mid-sim observations
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_cdt

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
    "round9": "2a341ace-0f57-4309-9b89-e59fe0f09179",
    "round10": "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    "round11": "324fde07-1670-4202-b199-7aa92ecb40ee",
    "round12": "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
}
ROUND_NAMES = ["round1", "round2", "round3", "round4", "round5", "round6",
               "round7", "round9", "round10", "round11", "round12"]


def exp_decay(d, vigor, lam):
    """P(settlement | distance d) = vigor * exp(-lam * d)"""
    return vigor * np.exp(-lam * d)


def analyze_round(round_name):
    """Extract P(settlement|distance) from ground truth and fit exponential kernel."""
    round_dir = DATA_DIR / round_name
    if not round_dir.exists():
        return None

    detail = json.loads((round_dir / "round_detail.json").read_text())

    # Collect (distance, P(settlement)) for all dynamic cells across seeds
    dist_sett_pairs = []  # (distance, settlement_prob)
    dist_bins = {}  # distance -> list of settlement probs

    for si in range(detail["seeds_count"]):
        analysis_path = round_dir / f"analysis_seed_{si}.json"
        if not analysis_path.exists():
            continue

        analysis = json.loads(analysis_path.read_text())
        terrain = np.array(analysis["initial_grid"], dtype=int)
        gt = np.array(analysis["ground_truth"])  # (40, 40, 6)

        # Find settlements
        is_sett = (terrain == 1) | (terrain == 2)
        if not is_sett.any():
            continue

        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')

        h, w = terrain.shape
        for y in range(h):
            for x in range(w):
                if terrain[y, x] in (10, 5):  # skip ocean/mountain
                    continue
                d = int(dist_map[y, x])
                if d == 0:  # skip settlement cells themselves
                    continue
                sett_prob = float(gt[y, x, 1])  # P(settlement)
                if d not in dist_bins:
                    dist_bins[d] = []
                dist_bins[d].append(sett_prob)

    if not dist_bins:
        return None

    # Compute mean P(settlement) at each distance
    distances = sorted(dist_bins.keys())
    mean_sett = []
    counts = []
    for d in distances:
        mean_sett.append(np.mean(dist_bins[d]))
        counts.append(len(dist_bins[d]))

    distances = np.array(distances, dtype=float)
    mean_sett = np.array(mean_sett)
    counts = np.array(counts)

    # Fit exponential: P(sett|d) = vigor * exp(-lambda * d)
    # Only fit on d >= 1 with sufficient data
    mask = (distances >= 1) & (counts >= 10) & (mean_sett > 0.001)
    if mask.sum() < 3:
        return {"round": round_name, "fit": False, "distances": distances,
                "mean_sett": mean_sett, "counts": counts}

    try:
        popt, pcov = curve_fit(exp_decay, distances[mask], mean_sett[mask],
                               p0=[0.3, 0.3], bounds=([0.001, 0.01], [1.0, 5.0]),
                               sigma=1.0 / np.sqrt(counts[mask]),  # weight by sample size
                               maxfev=5000)
        vigor, lam = popt
        # Compute R² on fit range
        pred = exp_decay(distances[mask], vigor, lam)
        ss_res = np.sum((mean_sett[mask] - pred) ** 2)
        ss_tot = np.sum((mean_sett[mask] - mean_sett[mask].mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
    except Exception as e:
        return {"round": round_name, "fit": False, "error": str(e),
                "distances": distances, "mean_sett": mean_sett, "counts": counts}

    return {
        "round": round_name,
        "fit": True,
        "vigor": vigor,
        "lambda": lam,
        "r2": r2,
        "distances": distances,
        "mean_sett": mean_sett,
        "counts": counts,
    }


def analyze_observations(round_name):
    """Try to estimate lam from mid-sim observations (50 queries)."""
    round_dir = DATA_DIR / round_name
    if not round_dir.exists() or round_name not in ROUND_IDS:
        return None

    detail = json.loads((round_dir / "round_detail.json").read_text())
    rid = ROUND_IDS[round_name]
    obs_dir = OBS_DIR / rid
    if not obs_dir.exists():
        return None

    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    if not obs_files:
        return None

    from utils import terrain_to_class

    # Collect observed (distance, is_settlement) pairs
    dist_obs = {}  # distance -> (settlement_count, total_count)

    for op in obs_files:
        obs = json.loads(op.read_text())
        sid = obs["seed_index"]
        state = detail["initial_states"][sid]
        terrain = np.array(state["grid"], dtype=int)

        is_sett = (terrain == 1) | (terrain == 2)
        if not is_sett.any():
            continue
        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')

        vp, grid = obs["viewport"], obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < 40 and 0 <= mx < 40:
                    if terrain[my, mx] in (10, 5):
                        continue
                    d = int(dist_map[my, mx])
                    if d == 0:
                        continue
                    oc = terrain_to_class(grid[row][col])
                    is_s = 1 if oc == 1 else 0
                    if d not in dist_obs:
                        dist_obs[d] = [0, 0]
                    dist_obs[d][0] += is_s
                    dist_obs[d][1] += 1

    if not dist_obs:
        return None

    # Fit exponential from observations
    distances = sorted(dist_obs.keys())
    obs_rates = []
    obs_counts = []
    for d in distances:
        s, t = dist_obs[d]
        obs_rates.append(s / max(t, 1))
        obs_counts.append(t)

    distances = np.array(distances, dtype=float)
    obs_rates = np.array(obs_rates)
    obs_counts = np.array(obs_counts)

    mask = (distances >= 1) & (obs_counts >= 20) & (obs_rates > 0.001)
    if mask.sum() < 3:
        return {"round": round_name, "obs_fit": False}

    try:
        popt, pcov = curve_fit(exp_decay, distances[mask], obs_rates[mask],
                               p0=[0.3, 0.3], bounds=([0.001, 0.01], [1.0, 5.0]),
                               sigma=1.0 / np.sqrt(obs_counts[mask]),
                               maxfev=5000)
        vigor_obs, lam_obs = popt
    except Exception:
        return {"round": round_name, "obs_fit": False}

    return {
        "round": round_name,
        "obs_fit": True,
        "vigor_obs": vigor_obs,
        "lambda_obs": lam_obs,
    }


def main():
    print("=" * 70)
    print("FITTING P(settlement | distance) = vigor * exp(-lam * d) PER ROUND")
    print("=" * 70)

    results = []
    for rn in ROUND_NAMES:
        r = analyze_round(rn)
        if r:
            results.append(r)

    # Print fitted parameters
    print(f"\n{'Round':<10} {'Vigor':>8} {'Lambda':>8} {'R²':>8} {'Regime':>15}")
    print("-" * 55)
    for r in results:
        if r["fit"]:
            # Classify regime
            if r["vigor"] < 0.05:
                regime = "Collapse"
            elif r["vigor"] < 0.15:
                regime = "Moderate"
            elif r["vigor"] < 0.30:
                regime = "Thriving"
            else:
                regime = "BOOM"
            print(f"{r['round']:<10} {r['vigor']:8.4f} {r['lambda']:8.4f} {r['r2']:8.3f} {regime:>15}")
        else:
            print(f"{r['round']:<10} {'FAILED':>8}")

    # Show the distance-probability curves
    print(f"\n{'':>10}", end="")
    for d in range(1, 16):
        print(f"  d={d:2d}", end="")
    print()
    print("-" * (10 + 15 * 6))

    for r in results:
        if not r["fit"]:
            continue
        print(f"{r['round']:<10}", end="")
        for d in range(1, 16):
            idx = np.where(r["distances"] == d)[0]
            if len(idx) > 0:
                print(f" {r['mean_sett'][idx[0]]:5.3f}", end="")
            else:
                print(f"     -", end="")
        print()

    # Lambda variation analysis
    fitted = [r for r in results if r["fit"]]
    if fitted:
        lambdas = [r["lambda"] for r in fitted]
        vigors = [r["vigor"] for r in fitted]
        print(f"\nLambda range: {min(lambdas):.3f} — {max(lambdas):.3f} "
              f"(ratio: {max(lambdas)/max(min(lambdas), 0.001):.1f}x)")
        print(f"Vigor range:  {min(vigors):.4f} — {max(vigors):.4f} "
              f"(ratio: {max(vigors)/max(min(vigors), 0.0001):.0f}x)")

    # Now test: can we estimate lam from observations?
    print("\n" + "=" * 70)
    print("ESTIMATING lam FROM MID-SIM OBSERVATIONS (50 queries)")
    print("=" * 70)

    print(f"\n{'Round':<10} {'lam_GT':>8} {'lam_obs':>8} {'Error%':>8}")
    print("-" * 40)
    for r in fitted:
        obs = analyze_observations(r["round"])
        if obs and obs.get("obs_fit"):
            err = abs(obs["lambda_obs"] - r["lambda"]) / max(r["lambda"], 0.01) * 100
            print(f"{r['round']:<10} {r['lambda']:8.3f} {obs['lambda_obs']:8.3f} {err:7.1f}%")
        else:
            print(f"{r['round']:<10} {r['lambda']:8.3f} {'N/A':>8}")


if __name__ == "__main__":
    main()

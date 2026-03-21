"""Test: Reaction-diffusion model for settlement probability prediction.

Instead of using distance buckets (which ignore terrain barriers and
settlement geometry), solve the diffusion equation on the terrain grid:

    dP/dt = D * laplacian(P) + source(settlements) - decay * P

This naturally:
- Routes around mountains and oceans
- Creates corridors between settlements
- Captures terrain barrier effects (~25% reduction behind mountains)
- Produces a continuous probability field

Compare: diffusion field correlation vs distance-based correlation with GT.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

from autoloop_fast import ROUND_NAMES, BOOM_ROUNDS, compute_score
from autoloop import DEFAULT_PARAMS
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask, _build_feature_key_index,
    build_calibration_lookup, build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
from autoloop_fast import ROUND_IDS

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"


def solve_diffusion(terrain, settlements, vigor=0.3, diffusion_coeff=1.0,
                    decay=0.1, n_steps=50, dt=0.1):
    """Solve reaction-diffusion on the terrain grid.

    P represents "settlement influence" — higher = more likely to become settlement.

    The equation:
        dP/dt = D * laplacian(P) + source(settlements) * vigor - decay * P

    Terrain effects:
    - Ocean (10), Mountain (5): P = 0 (barriers, no diffusion through them)
    - Forest (4): D *= 0.7 (partial barrier)
    - Plains (11), Empty (0): D *= 1.0 (full diffusion)

    Returns: (H, W) probability field
    """
    h, w = terrain.shape
    P = np.zeros((h, w), dtype=float)

    # Barriers: ocean and mountain block diffusion
    barrier = (terrain == 10) | (terrain == 5)

    # Terrain conductivity (how easily diffusion flows through)
    conductivity = np.ones((h, w), dtype=float)
    conductivity[terrain == 4] = 0.7   # forest slows diffusion
    conductivity[barrier] = 0.0        # ocean/mountain blocks

    # Source: settlement cells
    sett_points = set()
    for s in settlements:
        sx, sy = int(s["x"]), int(s["y"])
        if 0 <= sy < h and 0 <= sx < w:
            sett_points.add((sy, sx))
            P[sy, sx] = vigor

    # Iterative diffusion
    for step in range(n_steps):
        # Laplacian with conductivity-weighted diffusion
        lap = np.zeros_like(P)
        # 4-connected neighbors
        lap[1:, :]  += conductivity[:-1, :] * P[:-1, :]   # from above
        lap[:-1, :] += conductivity[1:, :]  * P[1:, :]    # from below
        lap[:, 1:]  += conductivity[:, :-1] * P[:, :-1]   # from left
        lap[:, :-1] += conductivity[:, 1:]  * P[:, 1:]    # from right
        lap -= 4.0 * conductivity * P

        P += dt * (diffusion_coeff * lap - decay * P)

        # Re-apply source and barriers
        for (sy, sx) in sett_points:
            P[sy, sx] = vigor
        P[barrier] = 0.0

        # Clamp
        P = np.clip(P, 0.0, 1.0)

    return P


def test_diffusion_correlation():
    """Compare diffusion field vs distance with ground truth P(settlement)."""
    print("Testing diffusion field correlation with P(settlement)...")

    all_diffusion = []
    all_distance = []
    all_psett = []

    for rn in ROUND_NAMES:
        round_dir = DATA_DIR / rn
        if not round_dir.exists():
            continue

        detail = json.loads((round_dir / "round_detail.json").read_text())

        for si in range(detail["seeds_count"]):
            analysis_path = round_dir / f"analysis_seed_{si}.json"
            if not analysis_path.exists():
                continue

            analysis = json.loads(analysis_path.read_text())
            terrain = np.array(analysis["initial_grid"], dtype=int)
            gt = np.array(analysis["ground_truth"])
            settlements = detail["initial_states"][si]["settlements"]

            if not settlements:
                continue

            # Diffusion field
            diff_field = solve_diffusion(terrain, settlements, vigor=0.5,
                                         diffusion_coeff=1.5, decay=0.08, n_steps=80)

            # Distance field
            is_sett = (terrain == 1) | (terrain == 2)
            if not is_sett.any():
                continue
            dist_map = distance_transform_cdt(~is_sett, metric='taxicab')

            h, w = terrain.shape
            for y in range(h):
                for x in range(w):
                    if terrain[y, x] in (10, 5):
                        continue
                    d = int(dist_map[y, x])
                    if d == 0:
                        continue
                    psett = float(gt[y, x, 1])
                    all_diffusion.append(diff_field[y, x])
                    all_distance.append(d)
                    all_psett.append(psett)

    diffusion = np.array(all_diffusion)
    distance = np.array(all_distance)
    psett = np.array(all_psett)

    r_diff = np.corrcoef(diffusion, psett)[0, 1]
    r_dist = np.corrcoef(1.0 / np.maximum(distance, 0.5), psett)[0, 1]
    r_log_diff = np.corrcoef(np.log(np.maximum(diffusion, 1e-10)), psett)[0, 1]

    print(f"\nCorrelation with P(settlement):")
    print(f"  diffusion field:         r = {r_diff:.4f}")
    print(f"  log(diffusion):          r = {r_log_diff:.4f}")
    print(f"  1/distance:              r = {r_dist:.4f}")
    print(f"  Diffusion advantage:       {r_diff - r_dist:+.4f}")

    # Within-distance-bucket discrimination
    print(f"\nDiffusion discrimination WITHIN distance buckets:")
    print(f"{'Distance':>10} {'Low diff P(s)':>14} {'High diff P(s)':>14} {'Delta':>8}")
    print("-" * 50)
    for d_lo, d_hi, label in [(1, 2, "d=1"), (2, 4, "d=2-3"), (4, 6, "d=4-5"),
                               (6, 9, "d=6-8"), (9, 20, "d=9+")]:
        d_mask = (distance >= d_lo) & (distance < d_hi)
        if not d_mask.any():
            continue
        diff_median = np.median(diffusion[d_mask])
        low_mask = d_mask & (diffusion <= diff_median)
        high_mask = d_mask & (diffusion > diff_median)
        if low_mask.any() and high_mask.any():
            low_p = psett[low_mask].mean()
            high_p = psett[high_mask].mean()
            print(f"{label:>10} {low_p:14.4f} {high_p:14.4f} {high_p-low_p:+8.4f}")

    return r_diff, r_dist


def test_diffusion_hyperparams():
    """Grid search diffusion hyperparameters."""
    print("\n\nGrid searching diffusion hyperparameters...")

    # Just use 3 rounds for speed
    test_rounds = ["round2", "round6", "round11"]

    # Collect data for these rounds
    data = []
    for rn in test_rounds:
        round_dir = DATA_DIR / rn
        if not round_dir.exists():
            continue
        detail = json.loads((round_dir / "round_detail.json").read_text())
        for si in range(min(2, detail["seeds_count"])):  # 2 seeds for speed
            analysis = json.loads((round_dir / f"analysis_seed_{si}.json").read_text())
            terrain = np.array(analysis["initial_grid"], dtype=int)
            gt = np.array(analysis["ground_truth"])
            settlements = detail["initial_states"][si]["settlements"]
            if settlements:
                is_sett = (terrain == 1) | (terrain == 2)
                if is_sett.any():
                    dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
                    data.append((terrain, gt, settlements, dist_map))

    print(f"Testing on {len(data)} seed/round pairs")
    print(f"{'D':>6} {'decay':>6} {'steps':>6} {'r(diff)':>8} {'r(1/d)':>8} {'delta':>8}")
    print("-" * 48)

    best_r = 0
    best_params = {}

    for D in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for decay in [0.03, 0.05, 0.08, 0.12, 0.2]:
            for n_steps in [40, 80]:
                all_diff = []
                all_dist = []
                all_ps = []

                for terrain, gt, setts, dist_map in data:
                    df = solve_diffusion(terrain, setts, vigor=0.5,
                                        diffusion_coeff=D, decay=decay,
                                        n_steps=n_steps)
                    h, w = terrain.shape
                    for y in range(h):
                        for x in range(w):
                            if terrain[y, x] in (10, 5) or dist_map[y, x] == 0:
                                continue
                            all_diff.append(df[y, x])
                            all_dist.append(dist_map[y, x])
                            all_ps.append(float(gt[y, x, 1]))

                diff = np.array(all_diff)
                dist = np.array(all_dist)
                ps = np.array(all_ps)

                r_d = np.corrcoef(diff, ps)[0, 1]
                r_n = np.corrcoef(1.0 / np.maximum(dist, 0.5), ps)[0, 1]

                if r_d > best_r:
                    best_r = r_d
                    best_params = {"D": D, "decay": decay, "n_steps": n_steps}
                    print(f"{D:6.1f} {decay:6.2f} {n_steps:6d} {r_d:8.4f} {r_n:8.4f} {r_d-r_n:+8.4f} ***")

    print(f"\nBest: r={best_r:.4f}, params={best_params}")


if __name__ == "__main__":
    r_d, r_n = test_diffusion_correlation()
    test_diffusion_hyperparams()

    print(f"\n{'='*50}")
    if r_d > r_n:
        print(f"DIFFUSION WINS: r={r_d:.4f} vs 1/distance r={r_n:.4f}")
    else:
        print(f"1/DISTANCE WINS: r={r_n:.4f} vs diffusion r={r_d:.4f}")

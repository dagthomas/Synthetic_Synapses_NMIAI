"""Test: Gravity model vs nearest-distance for settlement prediction.

Current pipeline uses min(manhattan_distance) to nearest settlement.
Gravity model uses sum of contributions from ALL settlements:
  gravity(cell) = sum_i exp(-lam * d_i) for all settlements i

This captures:
- Cells between two settlements get boosted (corridor effect)
- Dense clusters generate higher expansion pressure
- Not just "how far from nearest" but "how much total settlement pressure"

We test if replacing distance with gravity in the FK improves predictions.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.optimize import curve_fit

from autoloop_fast import ROUND_NAMES, BOOM_ROUNDS, ROUND_IDS, compute_score
from calibration import CalibrationModel, build_feature_keys, _dist_bucket
from config import MAP_H, MAP_W, NUM_CLASSES

DATA_DIR = Path(__file__).parent / "data" / "calibration"


def compute_gravity(terrain, settlements, lam=0.3):
    """Compute gravity field: sum of exp(-lam*d) for all settlements."""
    h, w = terrain.shape
    sett_points = [(int(s["x"]), int(s["y"])) for s in settlements]
    gravity = np.zeros((h, w), dtype=float)
    for sx, sy in sett_points:
        for y in range(h):
            for x in range(w):
                d = abs(x - sx) + abs(y - sy)
                gravity[y, x] += math.exp(-lam * d)
    return gravity


def gravity_bucket(g):
    """Bucket gravity into 5 levels."""
    if g < 0.01:
        return 0  # no influence
    if g < 0.1:
        return 1  # weak
    if g < 0.5:
        return 2  # moderate
    if g < 1.5:
        return 3  # strong
    return 4  # very strong (dense cluster)


def analyze_gravity_correlation():
    """Check if gravity correlates better with P(sett) than nearest distance."""
    print("Analyzing gravity vs nearest-distance correlation with P(settlement)...")

    all_gravity = []
    all_nearest = []
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

            sett_points = [(int(s["x"]), int(s["y"])) for s in settlements]
            if not sett_points:
                continue

            # Gravity with multiple lambda values
            gravity_03 = compute_gravity(terrain, settlements, lam=0.3)

            # Nearest distance
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
                    all_gravity.append(gravity_03[y, x])
                    all_nearest.append(d)
                    all_psett.append(psett)

    gravity = np.array(all_gravity)
    nearest = np.array(all_nearest)
    psett = np.array(all_psett)

    # Correlation analysis
    from numpy import corrcoef
    r_gravity = corrcoef(gravity, psett)[0, 1]
    r_nearest = corrcoef(1.0 / np.maximum(nearest, 0.5), psett)[0, 1]
    r_log_grav = corrcoef(np.log(np.maximum(gravity, 1e-10)), psett)[0, 1]

    print(f"\nCorrelation with P(settlement):")
    print(f"  gravity (sum exp(-0.3*d)):     r = {r_gravity:.4f}")
    print(f"  log(gravity):                  r = {r_log_grav:.4f}")
    print(f"  1/nearest_distance:            r = {r_nearest:.4f}")

    # Per-gravity-bucket analysis
    print(f"\n{'Gravity bucket':>15} {'Mean P(sett)':>12} {'Count':>8}")
    print("-" * 38)
    for lo, hi, label in [(0, 0.01, "0: none"), (0.01, 0.1, "1: weak"),
                           (0.1, 0.5, "2: moderate"), (0.5, 1.5, "3: strong"),
                           (1.5, 100, "4: very strong")]:
        mask = (gravity >= lo) & (gravity < hi)
        if mask.any():
            print(f"{label:>15} {psett[mask].mean():12.4f} {mask.sum():8d}")

    # Per-distance-bucket analysis for comparison
    print(f"\n{'Dist bucket':>15} {'Mean P(sett)':>12} {'Count':>8}")
    print("-" * 38)
    for lo, hi, label in [(1, 2, "d=1"), (2, 3, "d=2"), (3, 4, "d=3"),
                           (4, 6, "d=4-5"), (6, 9, "d=6-8"), (9, 100, "d=9+")]:
        mask = (nearest >= lo) & (nearest < hi)
        if mask.any():
            print(f"{label:>15} {psett[mask].mean():12.4f} {mask.sum():8d}")

    # Key test: does gravity help BEYOND distance?
    # For cells at the same distance bucket, does gravity add info?
    print(f"\nGravity discrimination WITHIN distance buckets:")
    print(f"{'Distance':>10} {'Low grav P(s)':>14} {'High grav P(s)':>14} {'Delta':>8}")
    print("-" * 50)
    for d_lo, d_hi, label in [(1, 2, "d=1"), (2, 4, "d=2-3"), (4, 6, "d=4-5"),
                               (6, 9, "d=6-8")]:
        d_mask = (nearest >= d_lo) & (nearest < d_hi)
        if not d_mask.any():
            continue
        grav_median = np.median(gravity[d_mask])
        low_mask = d_mask & (gravity <= grav_median)
        high_mask = d_mask & (gravity > grav_median)
        if low_mask.any() and high_mask.any():
            low_p = psett[low_mask].mean()
            high_p = psett[high_mask].mean()
            print(f"{label:>10} {low_p:14.4f} {high_p:14.4f} {high_p-low_p:+8.4f}")

    return r_gravity, r_nearest


if __name__ == "__main__":
    r_g, r_n = analyze_gravity_correlation()

    # Summary
    print(f"\n{'='*50}")
    print(f"VERDICT: gravity r={r_g:.4f} vs 1/distance r={r_n:.4f}")
    if r_g > r_n:
        print(f"Gravity wins by {r_g - r_n:.4f} correlation")
    else:
        print(f"1/distance wins by {r_n - r_g:.4f} correlation")

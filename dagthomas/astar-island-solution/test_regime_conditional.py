"""Test: Regime-conditional calibration model.

The key insight: each round has different (vigor, lambda) parameters.
The current calibration model AVERAGES over all rounds, losing this info.

This model:
1. Groups historical rounds by vigor level (estimated from GT settlement %)
2. Builds separate calibration priors for low/medium/high vigor
3. At prediction time, uses estimated vigor to interpolate between priors
4. This gives per-feature-key distributions that are REGIME-AWARE

Expected impact: cells at d=4-5 in a boom round should predict much higher
settlement probability than the round-averaged calibration.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt, uniform_filter

from autoloop import DEFAULT_PARAMS
from autoloop_fast import ROUND_NAMES, BOOM_ROUNDS, ROUND_IDS, compute_score
from calibration import CalibrationModel, build_feature_keys, _dist_bucket
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask, _build_feature_key_index,
    build_calibration_lookup, build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"


def compute_round_vigor(round_dir):
    """Compute vigor (avg settlement probability) for a round from GT."""
    detail = json.loads((round_dir / "round_detail.json").read_text())
    sett_probs = []
    for si in range(detail["seeds_count"]):
        ap = round_dir / f"analysis_seed_{si}.json"
        if not ap.exists():
            continue
        analysis = json.loads(ap.read_text())
        terrain = np.array(analysis["initial_grid"], dtype=int)
        gt = np.array(analysis["ground_truth"])
        dynamic = (terrain != 10) & (terrain != 5)
        if dynamic.any():
            sett_probs.append(gt[dynamic, 1].mean())
    return np.mean(sett_probs) if sett_probs else 0.0


def build_vigor_conditional_cal(train_rounds, round_vigors, target_vigor):
    """Build calibration model weighted by proximity to target vigor.

    Rounds with vigor close to target get higher weight.
    This creates a regime-specific prior.
    """
    # Weight each round by exp(-|vigor - target|^2 / sigma^2)
    sigma = 0.08  # bandwidth — how much vigor difference matters
    cal = CalibrationModel()

    for rname in train_rounds:
        round_dir = DATA_DIR / rname
        if not round_dir.exists():
            continue

        vig = round_vigors.get(rname, 0.1)
        weight = math.exp(-((vig - target_vigor) ** 2) / (2 * sigma ** 2))

        if weight < 0.01:
            continue  # Skip very dissimilar rounds

        # Load round with weight applied to GT probabilities
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue

        detail = json.loads(detail_path.read_text())
        for seed_idx in range(detail["seeds_count"]):
            ap = round_dir / f"analysis_seed_{seed_idx}.json"
            if not ap.exists():
                continue
            analysis = json.loads(ap.read_text())
            terrain = np.asarray(analysis["initial_grid"], dtype=int)
            ground_truth = np.asarray(analysis["ground_truth"], dtype=float)
            settlements = detail["initial_states"][seed_idx]["settlements"]

            fkeys = build_feature_keys(terrain, settlements)
            h, w = terrain.shape
            for y in range(h):
                for x in range(w):
                    fine_key = fkeys[y][x]
                    coarse_key = (fine_key[0], fine_key[1], fine_key[2], fine_key[4])
                    gt_probs = ground_truth[y, x] * weight  # weighted GT

                    cal.fine_sums[fine_key] += gt_probs
                    cal.fine_counts[fine_key] += weight

                    cal.coarse_sums[coarse_key] += gt_probs
                    cal.coarse_counts[coarse_key] += weight

                    cal.base_sums[fine_key[0]] += gt_probs
                    cal.base_counts[fine_key[0]] += weight

                    cal.global_sum += gt_probs
                    cal.global_count += weight
                    cal.total_cells += 1

        cal.rounds_loaded += 1

    if cal.global_count > 0:
        from calibration import _floor_and_renormalize
        cal.global_probs = _floor_and_renormalize(cal.global_sum / cal.global_count)

    return cal


def estimate_vigor_from_obs(obs_files, detail, seeds_per_round=5):
    """Estimate vigor from mid-sim observations."""
    sett_count = 0
    total_count = 0

    for op in obs_files:
        obs = json.loads(op.read_text())
        sid = obs["seed_index"]
        if sid >= seeds_per_round:
            continue
        terrain = np.array(detail["initial_states"][sid]["grid"], dtype=int)
        vp, grid = obs["viewport"], obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                    if terrain[my, mx] in (10, 5):
                        continue
                    oc = terrain_to_class(grid[row][col])
                    if oc == 1:
                        sett_count += 1
                    total_count += 1

    if total_count == 0:
        return 0.1
    return sett_count / total_count


def evaluate_regime_conditional(params, sigma=0.08):
    """LOO evaluation with regime-conditional calibration."""
    results = {}

    # Pre-compute vigors for all rounds
    round_vigors = {}
    for rn in ROUND_NAMES + ["round1"]:
        rd = DATA_DIR / rn
        if rd.exists():
            round_vigors[rn] = compute_round_vigor(rd)

    for test_round in ROUND_NAMES:
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]
        detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
        rid = ROUND_IDS[test_round]
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

        # Estimate vigor from observations
        est_vigor = estimate_vigor_from_obs(obs_files, detail)

        # Build regime-conditional calibration
        cal = build_vigor_conditional_cal(train_rounds, round_vigors, est_vigor)

        # Build standard calibration for comparison
        cal_std = CalibrationModel()
        for tr in train_rounds:
            cal_std.add_round(DATA_DIR / tr)

        # Build seeds
        seeds = []
        for si in range(5):
            state = detail["initial_states"][si]
            terrain = np.array(state["grid"], dtype=int)
            gt = np.array(
                json.loads((DATA_DIR / test_round / f"analysis_seed_{si}.json").read_text())[
                    "ground_truth"
                ]
            )
            fkeys = build_feature_keys(terrain, state["settlements"])
            idx_grid, unique_keys = _build_feature_key_index(fkeys)
            coastal = _build_coastal_mask(terrain)
            static_mask = (terrain == 10) | (terrain == 5)
            dynamic_mask = ~static_mask
            inland_dynamic = dynamic_mask & ~coastal
            seeds.append({
                "terrain": terrain, "fkeys": fkeys, "idx_grid": idx_grid,
                "unique_keys": unique_keys, "coastal": coastal,
                "static_mask": static_mask, "dynamic_mask": dynamic_mask,
                "inland_dynamic": inland_dynamic, "gt": gt, "state": state,
            })

        # Build FK buckets
        gm = GlobalMultipliers()
        fk = FeatureKeyBuckets()
        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= 5:
                continue
            vp, grid = obs["viewport"], obs["grid"]
            fkeys_s = seeds[sid]["fkeys"]
            for row in range(len(grid)):
                for col in range(len(grid[0]) if grid else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(grid[row][col])
                        gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        fk.add_observation(fkeys_s[my][mx], oc)

        # Multiplier
        if gm.observed.sum() > 0:
            smooth = params.get("mult_smooth", 5.0) * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            raw_ratio = (gm.observed + smooth) / np.maximum(gm.expected + smooth, 1e-6)
            mult = np.power(raw_ratio, params.get("mult_power", 0.4))
            mult[0] = np.clip(mult[0], 0.75, 1.25)
            mult[5] = np.clip(mult[5], 0.85, 1.15)
            for c in [1, 2, 3]:
                mult[c] = np.clip(mult[c], 0.15, 2.0)
            mult[4] = np.clip(mult[4], 0.5, 1.8)
        else:
            mult = np.ones(NUM_CLASSES)

        scores = []
        for seed_data in seeds:
            terrain = seed_data["terrain"]
            idx_grid = seed_data["idx_grid"]
            unique_keys = seed_data["unique_keys"]
            coastal = seed_data["coastal"]
            dynamic_mask = seed_data["dynamic_mask"]
            inland_dynamic = seed_data["inland_dynamic"]
            gt = seed_data["gt"]

            # Use regime-conditional calibration
            cal_priors = build_calibration_lookup(cal, unique_keys, params)
            raw_cal_grid = cal_priors[idx_grid]

            fk_min = params.get("fk_min_count", 5)
            fk_emp, fk_cnt = build_fk_empirical_lookup(fk, unique_keys, fk_min)

            pred = cal_priors[idx_grid]
            emp_grid = fk_emp[idx_grid]
            cnt_grid = fk_cnt[idx_grid]
            has_fk = cnt_grid >= fk_min

            pw = max(0.5, params.get("fk_prior_weight", 5.0))
            ms = params.get("fk_max_strength", 8.0)
            strengths = np.minimum(ms, np.sqrt(cnt_grid))
            blended = pred * pw + emp_grid * strengths[:, :, np.newaxis]
            blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
            pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

            # Distance-aware multiplier
            dist_aware = params.get("dist_aware_mult", False)
            if dist_aware and gm.observed.sum() > 0:
                is_sett = (terrain == 1) | (terrain == 2)
                if is_sett.any():
                    dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
                else:
                    dist_map = np.full_like(terrain, 99, dtype=int)
                exp_damp = params.get("dist_exp_damp", 0.7)
                mult_exp = mult.copy()
                for c in [1, 2, 3]:
                    mult_exp[c] = 1.0 + (mult[c] - 1.0) * exp_damp
                sett_mask = dist_map == 0
                pred[sett_mask] *= mult[np.newaxis, :]
                pred[~sett_mask] *= mult_exp[np.newaxis, :]
            else:
                pred *= mult[np.newaxis, np.newaxis, :]
            pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            # Smoothing
            sa = params.get("smooth_alpha", 0.0)
            if sa > 0:
                for k in [1, 3]:
                    smoothed = uniform_filter(pred[:, :, k], size=3, mode='reflect')
                    pred[:, :, k] = pred[:, :, k] * (1 - sa) + smoothed * sa
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            # Temperature
            T_low = params.get("temp_low", 1.0)
            T_high = params.get("temp_high", 1.0)
            if T_low != 1.0 or T_high != 1.0:
                ent_lo = params.get("temp_ent_lo", 0.2)
                ent_hi = params.get("temp_ent_hi", 1.0)
                cal_entropy = -np.sum(raw_cal_grid * np.log(np.maximum(raw_cal_grid, 1e-10)), axis=-1)
                t_frac = np.clip((cal_entropy - ent_lo) / max(ent_hi - ent_lo, 1e-6), 0.0, 1.0)
                T_grid = T_low + t_frac * (T_high - T_low)
                T_grid_3d = np.maximum(T_grid[:, :, np.newaxis], 0.1)
                pred = np.where(pred > 0, np.power(np.maximum(pred, 1e-30), 1.0 / T_grid_3d), 0.0)
                np.nan_to_num(pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            # Structural zeros
            pred[dynamic_mask, 5] = 0.0
            pred[inland_dynamic, 2] = 0.0

            # Floor
            floor = params.get("floor_nonzero", 0.005)
            dp = pred[dynamic_mask]
            nz = dp > 0
            dp = np.where(nz, np.maximum(dp, floor), 0.0)
            dp /= np.maximum(dp.sum(axis=-1, keepdims=True), 1e-10)
            pred[dynamic_mask] = dp

            pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
            pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

            scores.append(compute_score(gt, pred))

        results[test_round] = float(np.mean(scores))

    results["avg"] = float(np.mean([results[r] for r in ROUND_NAMES]))
    boom = [results[r] for r in ROUND_NAMES if r in BOOM_ROUNDS]
    nonboom = [results[r] for r in ROUND_NAMES if r not in BOOM_ROUNDS]
    results["boom_avg"] = float(np.mean(boom)) if boom else 0.0
    results["nonboom_avg"] = float(np.mean(nonboom)) if nonboom else 0.0
    return results, round_vigors


def main():
    params = dict(DEFAULT_PARAMS)

    # Baseline
    print("Running BASELINE (standard calibration)...")
    from autoloop_fast import FastHarness
    harness = FastHarness(seeds_per_round=5)
    base = harness.evaluate(params)
    print(f"  Baseline: avg={base['avg']:.3f} boom={base['boom_avg']:.3f} "
          f"nonboom={base['nonboom_avg']:.3f}")

    # Regime-conditional
    print("\nGrid searching sigma (bandwidth)...")
    print(f"{'sigma':>8} {'avg':>8} {'boom':>8} {'nonboom':>8} {'delta':>8}")
    print("-" * 44)

    for sigma_val in [0.03, 0.05, 0.08, 0.12, 0.20, 0.50, 1.0]:
        # Monkey-patch sigma in the build function
        import test_regime_conditional as trc
        original_build = trc.build_vigor_conditional_cal

        def patched_build(train_rounds, round_vigors, target_vigor, _s=sigma_val):
            # Override sigma
            import calibration
            cal = CalibrationModel()
            for rname in train_rounds:
                round_dir = DATA_DIR / rname
                if not round_dir.exists():
                    continue
                vig = round_vigors.get(rname, 0.1)
                weight = math.exp(-((vig - target_vigor) ** 2) / (2 * _s ** 2))
                if weight < 0.01:
                    continue
                detail_path = round_dir / "round_detail.json"
                if not detail_path.exists():
                    continue
                detail = json.loads(detail_path.read_text())
                for seed_idx in range(detail["seeds_count"]):
                    ap = round_dir / f"analysis_seed_{seed_idx}.json"
                    if not ap.exists():
                        continue
                    analysis = json.loads(ap.read_text())
                    terrain = np.asarray(analysis["initial_grid"], dtype=int)
                    ground_truth = np.asarray(analysis["ground_truth"], dtype=float)
                    settlements = detail["initial_states"][seed_idx]["settlements"]
                    fkeys = build_feature_keys(terrain, settlements)
                    h, w = terrain.shape
                    for y in range(h):
                        for x in range(w):
                            fine_key = fkeys[y][x]
                            coarse_key = (fine_key[0], fine_key[1], fine_key[2], fine_key[4])
                            gt_probs = ground_truth[y, x] * weight
                            cal.fine_sums[fine_key] += gt_probs
                            cal.fine_counts[fine_key] += weight
                            cal.coarse_sums[coarse_key] += gt_probs
                            cal.coarse_counts[coarse_key] += weight
                            cal.base_sums[fine_key[0]] += gt_probs
                            cal.base_counts[fine_key[0]] += weight
                            cal.global_sum += gt_probs
                            cal.global_count += weight
                            cal.total_cells += 1
                    cal.rounds_loaded += 1
                if cal.global_count > 0:
                    from calibration import _floor_and_renormalize
                    cal.global_probs = _floor_and_renormalize(cal.global_sum / cal.global_count)
                return cal
            return cal

        trc.build_vigor_conditional_cal = patched_build
        scores, vigors = evaluate_regime_conditional(params, sigma=sigma_val)
        trc.build_vigor_conditional_cal = original_build

        delta = scores["avg"] - base["avg"]
        print(f"{sigma_val:8.2f} {scores['avg']:8.3f} {scores['boom_avg']:8.3f} "
              f"{scores['nonboom_avg']:8.3f} {delta:+8.3f}")

    # Show vigors
    print("\nRound vigors (GT settlement %):")
    vigors_list = []
    for rn in ROUND_NAMES + ["round1"]:
        rd = DATA_DIR / rn
        if rd.exists():
            v = compute_round_vigor(rd)
            vigors_list.append((rn, v))
            print(f"  {rn}: {v:.4f}")


if __name__ == "__main__":
    main()

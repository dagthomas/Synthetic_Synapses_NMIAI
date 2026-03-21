"""Backtest: Lambda-kernel settlement prediction blended with existing pipeline.

Approach:
1. From observations, fit P(sett|d) = vigor * exp(-lam * d)
2. Use this as a per-round distance-decay correction on top of the FK pipeline
3. Compare LOO scores: baseline vs lambda-blended

The lambda kernel captures the per-round "spread rate" which varies 11.6x between
rounds but is averaged out by distance-bucket calibration.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt, uniform_filter
from scipy.optimize import curve_fit

from autoloop import DEFAULT_PARAMS
from autoloop_fast import ROUND_NAMES, BOOM_ROUNDS, ROUND_IDS, compute_score
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"


def exp_decay(d, vigor, lam):
    return vigor * np.exp(-lam * d)


def fit_lambda_from_obs(obs_files, detail, seeds_per_round=5):
    """Fit (vigor, lambda) from mid-sim observations.

    Returns: (vigor, lam) or None if insufficient data.
    Also returns per-distance (sett_count, total_count) dict.
    """
    dist_obs = {}  # distance -> [sett_count, total_count]

    for op in obs_files:
        obs = json.loads(op.read_text())
        sid = obs["seed_index"]
        if sid >= seeds_per_round:
            continue
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
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
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
        return None, dist_obs

    distances = sorted(dist_obs.keys())
    rates = []
    counts = []
    for d in distances:
        s, t = dist_obs[d]
        rates.append(s / max(t, 1))
        counts.append(t)

    distances = np.array(distances, dtype=float)
    rates = np.array(rates)
    counts = np.array(counts)

    mask = (distances >= 1) & (counts >= 15) & (rates >= 0.0)
    if mask.sum() < 3:
        return None, dist_obs

    try:
        popt, _ = curve_fit(exp_decay, distances[mask], rates[mask],
                            p0=[0.2, 0.3], bounds=([0.001, 0.01], [1.0, 5.0]),
                            sigma=1.0 / np.sqrt(np.maximum(counts[mask], 1)),
                            maxfev=5000)
        return (float(popt[0]), float(popt[1])), dist_obs
    except Exception:
        return None, dist_obs


def evaluate_with_lambda(params, blend_alpha=0.3):
    """Evaluate with lambda-kernel blending.

    blend_alpha: how much to blend the lambda kernel prediction with the FK prediction.
    0.0 = pure FK (baseline), 1.0 = pure lambda kernel.
    """
    results = {}
    lambda_fits = {}

    for test_round in ROUND_NAMES:
        # Build calibration (LOO)
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]
        cal = CalibrationModel()
        for tr in train_rounds:
            cal.add_round(DATA_DIR / tr)

        detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
        rid = ROUND_IDS[test_round]
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

        # Fit lambda from observations
        lam_fit, dist_obs = fit_lambda_from_obs(obs_files, detail)
        lambda_fits[test_round] = lam_fit

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

        # Build FK buckets and global multipliers from observations
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

        # Compute multiplier
        if gm.observed.sum() > 0:
            smooth_val = params.get("mult_smooth", 5.0)
            smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            raw_ratio = (gm.observed + smooth) / np.maximum(gm.expected + smooth, 1e-6)
            base_power = params.get("mult_power", 0.4)
            ratio = np.power(raw_ratio, base_power)
            ratio[0] = np.clip(ratio[0], params.get("mult_empty_lo", 0.75),
                               params.get("mult_empty_hi", 1.25))
            ratio[5] = np.clip(ratio[5], 0.85, 1.15)
            ratio[1] = np.clip(ratio[1], params.get("mult_sett_lo", 0.15),
                               params.get("mult_sett_hi", 2.0))
            ratio[2] = np.clip(ratio[2], params.get("mult_port_lo", 0.15),
                               params.get("mult_port_hi", 2.0))
            ratio[3] = np.clip(ratio[3], params.get("mult_sett_lo", 0.15),
                               params.get("mult_sett_hi", 2.0))
            ratio[4] = np.clip(ratio[4], params.get("mult_forest_lo", 0.5),
                               params.get("mult_forest_hi", 1.8))
            mult = ratio
        else:
            mult = np.ones(NUM_CLASSES)

        scores = []
        for seed_data in seeds:
            terrain = seed_data["terrain"]
            idx_grid = seed_data["idx_grid"]
            unique_keys = seed_data["unique_keys"]
            coastal = seed_data["coastal"]
            static_mask = seed_data["static_mask"]
            dynamic_mask = seed_data["dynamic_mask"]
            inland_dynamic = seed_data["inland_dynamic"]
            gt = seed_data["gt"]

            # === Standard FK pipeline ===
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

            # === LAMBDA KERNEL BLENDING ===
            if lam_fit is not None and blend_alpha > 0:
                vigor, lam = lam_fit
                is_sett = (terrain == 1) | (terrain == 2)
                if is_sett.any():
                    dist_map = distance_transform_cdt(~is_sett, metric='taxicab').astype(float)
                else:
                    dist_map = np.full_like(terrain, 99.0, dtype=float)

                # Lambda kernel: predicted P(settlement) at each distance
                lam_sett = vigor * np.exp(-lam * dist_map)
                lam_sett[dist_map == 0] = vigor  # at settlement cells
                lam_sett = np.clip(lam_sett, 0.0, 0.95)

                # Build a full 6-class prediction from lambda kernel
                # Settlement = lam_sett, rest redistributed proportionally from FK pred
                lam_pred = pred.copy()
                # Only adjust on dynamic non-settlement cells
                adjust_mask = dynamic_mask & (dist_map > 0)

                # Blend settlement channel: FK pred vs lambda kernel
                fk_sett = pred[:, :, 1]
                blended_sett = fk_sett * (1 - blend_alpha) + lam_sett * blend_alpha
                # Compute the delta we're adding/removing from settlement
                delta_sett = blended_sett - fk_sett

                lam_pred[:, :, 1] = blended_sett
                # Compensate from empty (primary) proportionally
                lam_pred[:, :, 0] -= delta_sett * 0.7
                lam_pred[:, :, 4] -= delta_sett * 0.2  # forest
                lam_pred[:, :, 3] -= delta_sett * 0.1  # ruin

                lam_pred = np.maximum(lam_pred, 1e-10)
                lam_pred /= np.maximum(lam_pred.sum(axis=-1, keepdims=True), 1e-10)

                pred = np.where(adjust_mask[:, :, np.newaxis], lam_pred, pred)

            # === Rest of standard pipeline ===
            # Distance-aware multiplier
            dist_aware = params.get("dist_aware_mult", False)
            if dist_aware and gm.observed.sum() > 0:
                is_sett = (terrain == 1) | (terrain == 2)
                if is_sett.any():
                    dist_map2 = distance_transform_cdt(~is_sett, metric='taxicab')
                else:
                    dist_map2 = np.full_like(terrain, 99, dtype=int)
                exp_damp = params.get("dist_exp_damp", 0.7)
                mult_exp = mult.copy()
                for c in [1, 2, 3]:
                    mult_exp[c] = 1.0 + (mult[c] - 1.0) * exp_damp
                sett_mask = dist_map2 == 0
                pred[sett_mask] *= mult[np.newaxis, :]
                pred[~sett_mask] *= mult_exp[np.newaxis, :]
            else:
                pred *= mult[np.newaxis, np.newaxis, :]
            pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            # Smoothing
            smooth_alpha = params.get("smooth_alpha", 0.0)
            if smooth_alpha > 0:
                for k in [1, 3]:
                    smoothed = uniform_filter(pred[:, :, k], size=3, mode='reflect')
                    pred[:, :, k] = pred[:, :, k] * (1 - smooth_alpha) + smoothed * smooth_alpha
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            # Temperature
            T_low = params.get("temp_low", 1.0)
            T_high = params.get("temp_high", 1.0)
            if T_low != 1.0 or T_high != 1.0:
                ent_lo = params.get("temp_ent_lo", 0.2)
                ent_hi = params.get("temp_ent_hi", 1.0)
                cal_entropy = -np.sum(
                    raw_cal_grid * np.log(np.maximum(raw_cal_grid, 1e-10)), axis=-1)
                t_frac = np.clip((cal_entropy - ent_lo) / max(ent_hi - ent_lo, 1e-6), 0.0, 1.0)
                T_grid = T_low + t_frac * (T_high - T_low)
                boom_boost = 0.10 * math.sqrt(min(float(mult[1]), 1.0))
                is_sett = (terrain == 1) | (terrain == 2)
                if is_sett.any():
                    dm = distance_transform_cdt(~is_sett, metric='taxicab')
                    sr = 2 + int(3.0 * min(float(mult[1]), 1.2))
                    T_grid[dm <= sr] += boom_boost
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

            # Lock static
            pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
            pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

            scores.append(compute_score(gt, pred))

        results[test_round] = float(np.mean(scores))

    results["avg"] = float(np.mean([results[r] for r in ROUND_NAMES]))
    boom_scores = [results[r] for r in ROUND_NAMES if r in BOOM_ROUNDS]
    nonboom_scores = [results[r] for r in ROUND_NAMES if r not in BOOM_ROUNDS]
    results["boom_avg"] = float(np.mean(boom_scores)) if boom_scores else 0.0
    results["nonboom_avg"] = float(np.mean(nonboom_scores)) if nonboom_scores else 0.0
    return results, lambda_fits


def main():
    params = dict(DEFAULT_PARAMS)

    # First run baseline
    print("Running BASELINE (no lambda kernel)...")
    t0 = time.time()
    base, _ = evaluate_with_lambda(params, blend_alpha=0.0)
    print(f"  Baseline: avg={base['avg']:.3f} boom={base['boom_avg']:.3f} "
          f"nonboom={base['nonboom_avg']:.3f} ({time.time()-t0:.1f}s)")

    # Grid search blend_alpha
    print(f"\n{'alpha':>6} {'avg':>8} {'boom':>8} {'nonboom':>8} {'delta':>8} | Lambda fits")
    print("-" * 70)

    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]:
        scores, fits = evaluate_with_lambda(params, blend_alpha=alpha)
        delta = scores["avg"] - base["avg"]
        fit_str = " ".join(f"{r[-2:]}={f[1]:.2f}" if f else f"{r[-2:]}=N/A"
                           for r, f in sorted(fits.items())[:4])
        print(f"{alpha:6.2f} {scores['avg']:8.3f} {scores['boom_avg']:8.3f} "
              f"{scores['nonboom_avg']:8.3f} {delta:+8.3f} | {fit_str}")

    # Show per-round detail for best alpha
    best_alpha = 0.15  # adjust after seeing results
    print(f"\n--- Per-round detail at alpha={best_alpha} ---")
    scores, fits = evaluate_with_lambda(params, blend_alpha=best_alpha)
    print(f"{'Round':<10} {'Score':>8} {'Base':>8} {'Delta':>8} {'vigor':>8} {'lam':>8}")
    print("-" * 55)
    for r in ROUND_NAMES:
        f = fits.get(r)
        tag = " *" if r in BOOM_ROUNDS else ""
        v_str = f"{f[0]:.3f}" if f else "N/A"
        l_str = f"{f[1]:.3f}" if f else "N/A"
        delta = scores[r] - base[r]
        print(f"{r:<10} {scores[r]:8.2f} {base[r]:8.2f} {delta:+8.2f} {v_str:>8} {l_str:>8}{tag}")
    print("-" * 55)
    print(f"{'AVG':<10} {scores['avg']:8.3f} {base['avg']:8.3f} "
          f"{scores['avg'] - base['avg']:+8.3f}")
    print(f"{'BOOM':<10} {scores['boom_avg']:8.3f} {base['boom_avg']:8.3f} "
          f"{scores['boom_avg'] - base['boom_avg']:+8.3f}")


if __name__ == "__main__":
    main()

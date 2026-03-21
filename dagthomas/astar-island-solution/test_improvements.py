"""A/B test: spatial clustering in FK and CMA-ES optimizer.

Compares the LOO harness score with and without spatial clustering,
using the same parameter set.

Usage:
    python test_improvements.py              # A/B test clustering
    python test_improvements.py --defaults   # Use DEFAULT_PARAMS instead of best
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from autoloop import DEFAULT_PARAMS, PARAM_SPACE
from autoloop_fast import ROUND_NAMES, BOOM_ROUNDS, compute_score
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from scipy.ndimage import distance_transform_cdt, uniform_filter
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

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


def build_harness_for_variant(use_cluster: bool, seeds_per_round: int = 5):
    """Build a harness variant with or without spatial clustering."""
    rounds = {}

    for test_round in ROUND_NAMES:
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]

        cal = CalibrationModel()
        for tr in train_rounds:
            cal.add_round(DATA_DIR / tr, use_cluster=use_cluster)

        detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
        rid = ROUND_IDS[test_round]
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

        seeds = []
        for si in range(min(seeds_per_round, 5)):
            state = detail["initial_states"][si]
            terrain = np.array(state["grid"], dtype=int)
            gt = np.array(
                json.loads((DATA_DIR / test_round / f"analysis_seed_{si}.json").read_text())[
                    "ground_truth"
                ]
            )
            fkeys = build_feature_keys(terrain, state["settlements"], use_cluster=use_cluster)
            idx_grid, unique_keys = _build_feature_key_index(fkeys)
            coastal = _build_coastal_mask(terrain)
            static_mask = (terrain == 10) | (terrain == 5)
            dynamic_mask = ~static_mask
            inland_dynamic = dynamic_mask & ~coastal

            seeds.append({
                "terrain": terrain,
                "fkeys": fkeys,
                "idx_grid": idx_grid,
                "unique_keys": unique_keys,
                "coastal": coastal,
                "static_mask": static_mask,
                "dynamic_mask": dynamic_mask,
                "inland_dynamic": inland_dynamic,
                "gt": gt,
                "state": state,
            })

        gm = GlobalMultipliers()
        fk = FeatureKeyBuckets()

        for si in range(min(seeds_per_round, 5)):
            seed_data = seeds[si]

        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= seeds_per_round:
                continue
            vp, grid = obs["viewport"], obs["grid"]
            seed_data = seeds[sid]
            fkeys_s = seed_data["fkeys"]
            for row in range(len(grid)):
                for col in range(len(grid[0]) if grid else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(grid[row][col])
                        gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        fk.add_observation(fkeys_s[my][mx], oc)

        rounds[test_round] = {"cal": cal, "seeds": seeds, "gm": gm, "fk": fk}

    return rounds


def evaluate_variant(rounds, params):
    """Evaluate params on a harness variant. Same logic as FastHarness.evaluate."""
    results = {}

    for test_round in ROUND_NAMES:
        rd = rounds[test_round]
        cal = rd["cal"]
        gm = rd["gm"]
        fk = rd["fk"]

        if gm.observed.sum() > 0:
            smooth_val = params.get("mult_smooth", 5.0)
            smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            raw_ratio = (gm.observed + smooth) / np.maximum(gm.expected + smooth, 1e-6)
            base_power = params.get("mult_power", 0.4)
            ratio = np.power(raw_ratio, base_power)
            sett_power = params.get("mult_power_sett", base_power)
            port_power = params.get("mult_power_port", base_power)
            if sett_power != base_power:
                ratio[1] = np.power(raw_ratio[1], sett_power)
            if port_power != base_power:
                ratio[2] = np.power(raw_ratio[2], port_power)
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

        sett_ratio_raw = raw_ratio[1] if gm.observed.sum() > 0 else 1.0
        regime_pw_scale = params.get("regime_prior_scale", 0.0)
        regime_pw_adj = 0.0
        if regime_pw_scale > 0 and gm.observed.sum() > 0:
            if sett_ratio_raw > 1.0:
                regime_pw_adj = -regime_pw_scale
            elif sett_ratio_raw < 0.1:
                regime_pw_adj = regime_pw_scale * 0.5

        scores = []
        for seed_data in rd["seeds"]:
            terrain = seed_data["terrain"]
            idx_grid = seed_data["idx_grid"]
            unique_keys = seed_data["unique_keys"]
            coastal = seed_data["coastal"]
            static_mask = seed_data["static_mask"]
            dynamic_mask = seed_data["dynamic_mask"]
            inland_dynamic = seed_data["inland_dynamic"]
            gt = seed_data["gt"]

            cal_priors = build_calibration_lookup(cal, unique_keys, params)
            raw_cal_grid = cal_priors[idx_grid]

            fk_min = params.get("fk_min_count", 5)
            fk_emp, fk_cnt = build_fk_empirical_lookup(fk, unique_keys, fk_min)

            pred = cal_priors[idx_grid]
            emp_grid = fk_emp[idx_grid]
            cnt_grid = fk_cnt[idx_grid]
            has_fk = cnt_grid >= fk_min

            pw = max(0.5, params.get("fk_prior_weight", 5.0) + regime_pw_adj)
            ms = params.get("fk_max_strength", 8.0)
            sfn = params.get("fk_strength_fn", "sqrt")

            if sfn == "sqrt":
                strengths = np.minimum(ms, np.sqrt(cnt_grid))
            elif sfn == "log":
                strengths = np.minimum(ms, np.log1p(cnt_grid) * 2)
            else:
                strengths = np.minimum(ms, cnt_grid * 0.1)

            blended = pred * pw + emp_grid * strengths[:, :, np.newaxis]
            blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
            pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

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

            smooth_alpha = params.get("smooth_alpha", 0.0)
            if smooth_alpha > 0:
                for k in [1, 3]:
                    smoothed = uniform_filter(pred[:, :, k], size=3, mode='reflect')
                    pred[:, :, k] = pred[:, :, k] * (1 - smooth_alpha) + smoothed * smooth_alpha
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

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
                    dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
                    sett_radius = 2 + int(3.0 * min(float(mult[1]), 1.2))
                    T_grid[dist_map <= sett_radius] += boom_boost
                T_grid_3d = np.maximum(T_grid[:, :, np.newaxis], 0.1)
                exponent = 1.0 / T_grid_3d
                pred = np.where(pred > 0, np.power(np.maximum(pred, 1e-30), exponent), 0.0)
                np.nan_to_num(pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

            use_prop_redist = params.get("prop_redist", False)
            if use_prop_redist:
                mountain_mass = np.where(dynamic_mask, pred[:, :, 5], 0.0)
                port_mass = np.where(inland_dynamic, pred[:, :, 2], 0.0)
                freed_mass = mountain_mass + port_mass
                pred[dynamic_mask, 5] = 0.0
                pred[inland_dynamic, 2] = 0.0
                redist_w = raw_cal_grid.copy()
                redist_w[dynamic_mask, 5] = 0.0
                redist_w[inland_dynamic, 2] = 0.0
                redist_sum = redist_w.sum(axis=-1, keepdims=True)
                redist_w = redist_w / np.maximum(redist_sum, 1e-10)
                pred += freed_mass[:, :, np.newaxis] * redist_w
            else:
                pred[dynamic_mask, 5] = 0.0
                pred[inland_dynamic, 2] = 0.0

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
    boom_scores = [results[r] for r in ROUND_NAMES if r in BOOM_ROUNDS]
    nonboom_scores = [results[r] for r in ROUND_NAMES if r not in BOOM_ROUNDS]
    results["boom_avg"] = float(np.mean(boom_scores)) if boom_scores else 0.0
    results["nonboom_avg"] = float(np.mean(nonboom_scores)) if nonboom_scores else 0.0
    return results


def print_comparison(label_a, scores_a, label_b, scores_b, key_counts_a=None, key_counts_b=None):
    """Print side-by-side comparison."""
    print(f"\n{'Round':<10} {label_a:>12} {label_b:>12} {'Delta':>8}")
    print("-" * 44)
    for r in ROUND_NAMES:
        tag = " *" if r in BOOM_ROUNDS else ""
        delta = scores_b[r] - scores_a[r]
        print(f"{r:<10} {scores_a[r]:12.2f} {scores_b[r]:12.2f} {delta:+8.2f}{tag}")
    print("-" * 44)
    for metric in ["avg", "boom_avg", "nonboom_avg"]:
        delta = scores_b[metric] - scores_a[metric]
        print(f"{metric:<10} {scores_a[metric]:12.3f} {scores_b[metric]:12.3f} {delta:+8.3f}")

    if key_counts_a and key_counts_b:
        print(f"\nFK key counts (first round shown):")
        r0 = ROUND_NAMES[0]
        print(f"  {label_a}: {key_counts_a[r0]} keys")
        print(f"  {label_b}: {key_counts_b[r0]} keys")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--defaults", action="store_true",
                        help="Use DEFAULT_PARAMS instead of best_params.json")
    args = parser.parse_args()

    # Load params
    if args.defaults:
        params = dict(DEFAULT_PARAMS)
        print("Using DEFAULT_PARAMS")
    else:
        params = dict(DEFAULT_PARAMS)
        bp_path = Path(__file__).parent / "best_params.json"
        if bp_path.exists():
            bp = json.loads(bp_path.read_text())
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
                    params[al_key] = bp[prod_key]
        print(f"Using best_params.json: prior_w={params['fk_prior_weight']}, "
              f"T_high={params['temp_high']}")

    # Build both variants
    print("\n--- Building CONTROL (no cluster) ---")
    t0 = time.time()
    rounds_ctrl = build_harness_for_variant(use_cluster=False, seeds_per_round=args.seeds)
    t1 = time.time()
    print(f"Control built in {t1-t0:.1f}s")

    print("\n--- Building TEST (with cluster) ---")
    rounds_test = build_harness_for_variant(use_cluster=True, seeds_per_round=args.seeds)
    t2 = time.time()
    print(f"Test built in {t2-t1:.1f}s")

    # Count FK keys
    key_counts_ctrl = {}
    key_counts_test = {}
    for r in ROUND_NAMES:
        key_counts_ctrl[r] = len(rounds_ctrl[r]["seeds"][0]["unique_keys"])
        key_counts_test[r] = len(rounds_test[r]["seeds"][0]["unique_keys"])

    # Evaluate both
    print("\n--- Evaluating CONTROL ---")
    scores_ctrl = evaluate_variant(rounds_ctrl, params)

    print("--- Evaluating TEST ---")
    scores_test = evaluate_variant(rounds_test, params)

    print_comparison("NO_CLUSTER", scores_ctrl, "CLUSTER", scores_test,
                     key_counts_ctrl, key_counts_test)

    # StdDev comparison
    ctrl_per_round = [scores_ctrl[r] for r in ROUND_NAMES]
    test_per_round = [scores_test[r] for r in ROUND_NAMES]
    print(f"\nStdDev: control={np.std(ctrl_per_round):.2f}, test={np.std(test_per_round):.2f}")


if __name__ == "__main__":
    main()

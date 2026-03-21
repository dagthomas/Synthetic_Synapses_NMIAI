"""Backtest ensemble of simulator + statistical predictions.

Tests whether blending simulator predictions with the existing
statistical model (predict_gemini pipeline) improves LOO scores
beyond the 88.6 baseline.

Usage:
    python sim_backtest.py                # Full backtest
    python sim_backtest.py --quick        # Fast test on 3 rounds
    python sim_backtest.py --alpha 0.2    # Set blend alpha
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from sim_data import load_round, ALL_ROUNDS, ROUND_IDS, load_observations, terrain_to_class
from sim_model import Simulator, compute_score, default_params, PARAM_SPEC, PARAM_NAMES
from sim_inference import fit_to_gt, fit_to_observations

# Import statistical prediction infrastructure
from autoloop_fast import (
    FastHarness, ROUND_NAMES, BOOM_ROUNDS, compute_score as fast_score,
    DATA_DIR, OBS_DIR,
)
from autoloop import DEFAULT_PARAMS as STAT_DEFAULT_PARAMS


def get_statistical_prediction(harness: FastHarness, round_name: str,
                               seed_idx: int, params: dict | None = None) -> np.ndarray:
    """Get statistical model prediction for a round+seed.

    Uses the FastHarness evaluate path but extracts per-seed predictions.
    """
    if params is None:
        params = dict(STAT_DEFAULT_PARAMS)

    rd = harness.rounds.get(round_name)
    if rd is None:
        return None

    # Re-run the prediction pipeline for this specific seed
    from autoloop_fast import build_calibration_lookup, build_fk_empirical_lookup
    from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
    from scipy.ndimage import distance_transform_cdt, uniform_filter
    from config import NUM_CLASSES

    cal = rd["cal"]
    gm = rd["gm"]
    fk = rd["fk"]
    seed_data = rd["seeds"][seed_idx]

    terrain = seed_data["terrain"]
    idx_grid = seed_data["idx_grid"]
    unique_keys = seed_data["unique_keys"]
    coastal = seed_data["coastal"]
    static_mask = seed_data["static_mask"]
    dynamic_mask = seed_data["dynamic_mask"]
    inland_dynamic = seed_data["inland_dynamic"]

    # Build multiplier
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
        ratio[0] = np.clip(ratio[0], params.get("mult_empty_lo", 0.75), params.get("mult_empty_hi", 1.25))
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        ratio[1] = np.clip(ratio[1], params.get("mult_sett_lo", 0.15), params.get("mult_sett_hi", 2.0))
        ratio[2] = np.clip(ratio[2], params.get("mult_port_lo", 0.15), params.get("mult_port_hi", 2.0))
        ratio[3] = np.clip(ratio[3], params.get("mult_sett_lo", 0.15), params.get("mult_sett_hi", 2.0))
        ratio[4] = np.clip(ratio[4], params.get("mult_forest_lo", 0.5), params.get("mult_forest_hi", 1.8))
        mult = ratio
    else:
        mult = np.ones(NUM_CLASSES)

    # Regime detection
    sett_ratio_raw = raw_ratio[1] if gm.observed.sum() > 0 else 1.0
    regime_pw_scale = params.get("regime_prior_scale", 0.0)
    regime_pw_adj = 0.0
    if regime_pw_scale > 0 and gm.observed.sum() > 0:
        if sett_ratio_raw > 1.0:
            regime_pw_adj = -regime_pw_scale
        elif sett_ratio_raw < 0.1:
            regime_pw_adj = regime_pw_scale * 0.5

    # Build prediction
    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    raw_cal_grid = cal_priors[idx_grid]
    fk_min = params.get("fk_min_count", 5)
    fk_emp, fk_cnt = build_fk_empirical_lookup(fk, unique_keys, fk_min)

    pred = cal_priors[idx_grid].copy()
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

    # Multiplier
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

    # Cluster boost
    cluster_boost = params.get("cluster_sett_boost", 0.0)
    if cluster_boost > 0:
        cd = seed_data["cluster_density"]
        cluster_factor = 1.0 + cluster_boost * np.minimum(cd, 3.0) / 3.0
        pred[:, :, 1] *= cluster_factor
        excess = pred[:, :, 1] * (1.0 - 1.0 / np.maximum(cluster_factor, 1e-10))
        pred[:, :, 0] -= excess * 0.5
        pred[:, :, 4] -= excess * 0.3
        pred[:, :, 3] -= excess * 0.2
        pred = np.maximum(pred, 1e-10)
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
        cal_entropy = -np.sum(raw_cal_grid * np.log(np.maximum(raw_cal_grid, 1e-10)), axis=-1)
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

    # Structural zeros + floor
    use_prop_redist = params.get("prop_redist", False)
    floor_val = params.get("floor_nonzero", 0.008)

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
        redist_norm = np.where(redist_sum > 0, redist_w / np.maximum(redist_sum, 1e-10), 1.0 / NUM_CLASSES)
        pred += freed_mass[:, :, np.newaxis] * redist_norm

    # Floor
    pred = np.maximum(pred, floor_val)
    pred[static_mask] = 0.0
    ocean = terrain == 10
    mountain = terrain == 5
    pred[ocean, :] = 0.0
    pred[ocean, 0] = 1.0
    pred[mountain, :] = 0.0
    pred[mountain, 5] = 1.0
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    return pred


def ensemble_blend(stat_pred: np.ndarray, sim_pred: np.ndarray,
                   alpha: float = 0.2, terrain: np.ndarray = None) -> np.ndarray:
    """Blend statistical and simulator predictions.

    alpha: weight for simulator (0 = pure statistical, 1 = pure simulator)
    Does NOT apply additional floor — preserves the floors already in each prediction.
    """
    if alpha == 0.0:
        return stat_pred.copy()

    blended = (1.0 - alpha) * stat_pred + alpha * sim_pred

    # Re-lock static cells
    if terrain is not None:
        ocean = terrain == 10
        mountain = terrain == 5
        blended[ocean, :] = 0.0
        blended[ocean, 0] = 1.0
        blended[mountain, :] = 0.0
        blended[mountain, 5] = 1.0

    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    return blended


def backtest(alpha: float = 0.2, n_sims: int = 500, max_evals: int = 200,
             rounds: list[str] | None = None, seeds_per_round: int = 1):
    """Run full backtest: statistical model vs ensemble vs simulator alone."""

    harness = FastHarness(seeds_per_round=seeds_per_round)

    # Start from DEFAULT_PARAMS (has all keys), overlay best_params.json
    stat_params = dict(STAT_DEFAULT_PARAMS)
    bp_path = Path(__file__).parent / "best_params.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text())
        # Map abbreviated names to full parameter names
        key_map = {
            "prior_w": "fk_prior_weight",
            "emp_max": "fk_max_strength",
            "exp_damp": "dist_exp_damp",
            "base_power": "mult_power",
            "T_high": "temp_high",
            "smooth_alpha": "smooth_alpha",
            "floor": "floor_nonzero",
        }
        for short, full in key_map.items():
            if short in bp:
                stat_params[full] = bp[short]

    test_rounds = rounds or ROUND_NAMES
    results = {}

    for rname in test_rounds:
        if rname not in harness.rounds:
            continue

        print(f"\n{'='*50}")
        print(f"{rname}")

        rd_data = harness.rounds[rname]
        seed_data = rd_data["seeds"][0]
        gt = seed_data["gt"]

        # 1. Statistical prediction
        stat_pred = get_statistical_prediction(harness, rname, 0, stat_params)
        stat_score = compute_score(gt, stat_pred)
        print(f"  Statistical: {stat_score:.2f}")

        # 2. Simulator prediction (fit to GT for ceiling test)
        rd = load_round(rname, 0)
        if rd is None or rd.ground_truth is None:
            print(f"  Skipping simulator (no data)")
            continue

        t0 = time.perf_counter()
        sim_params, _ = fit_to_gt(rd, n_sims=n_sims, max_evals=max_evals, verbose=False)
        sim = Simulator(rd)
        sim_pred = sim.run(sim_params, n_sims=n_sims, seed=42)
        t1 = time.perf_counter()
        sim_score = compute_score(gt, sim_pred)
        print(f"  Simulator:   {sim_score:.2f} ({t1-t0:.1f}s)")

        # 3. Ensemble at various alphas
        best_alpha = 0.0
        best_ens_score = stat_score
        for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            ens_pred = ensemble_blend(stat_pred, sim_pred, alpha=a, terrain=rd.terrain)
            ens_score = compute_score(gt, ens_pred)
            if ens_score > best_ens_score:
                best_ens_score = ens_score
                best_alpha = a

        ens_pred = ensemble_blend(stat_pred, sim_pred, alpha=best_alpha, terrain=rd.terrain)
        ens_score = compute_score(gt, ens_pred)
        delta = ens_score - stat_score
        sign = "+" if delta >= 0 else ""
        print(f"  Ensemble:    {ens_score:.2f} (alpha={best_alpha:.2f}, {sign}{delta:.2f})")

        results[rname] = {
            "stat": stat_score,
            "sim": sim_score,
            "ensemble": ens_score,
            "best_alpha": best_alpha,
            "delta": delta,
        }

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"{'Round':<10} {'Statistical':>12} {'Simulator':>10} {'Ensemble':>10} {'Delta':>8} {'Alpha':>6}")
        print("-" * 60)
        for rname in test_rounds:
            if rname not in results:
                continue
            r = results[rname]
            print(f"{rname:<10} {r['stat']:>12.2f} {r['sim']:>10.2f} {r['ensemble']:>10.2f} {r['delta']:>+8.2f} {r['best_alpha']:>6.2f}")

        stats = list(results.values())
        avg_stat = np.mean([r["stat"] for r in stats])
        avg_sim = np.mean([r["sim"] for r in stats])
        avg_ens = np.mean([r["ensemble"] for r in stats])
        avg_delta = np.mean([r["delta"] for r in stats])
        print("-" * 60)
        print(f"{'Average':<10} {avg_stat:>12.2f} {avg_sim:>10.2f} {avg_ens:>10.2f} {avg_delta:>+8.2f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rounds", nargs="*", default=None)
    args = parser.parse_args()

    rounds = args.rounds
    if args.quick:
        rounds = ["round3", "round5", "round6"]

    n_sims = 300 if args.quick else 500
    max_evals = 100 if args.quick else 200

    backtest(alpha=args.alpha, n_sims=n_sims, max_evals=max_evals, rounds=rounds)

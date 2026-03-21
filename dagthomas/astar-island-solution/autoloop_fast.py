"""Fast autonomous experiment loop — vectorized predictions.

~5x faster than autoloop.py. Uses fast_predict for vectorized numpy operations.

Usage:
    python autoloop_fast.py              # Run until Ctrl+C
    python autoloop_fast.py --summary    # Print summary
"""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from calibration import CalibrationModel, build_cluster_density, build_feature_keys

# Lightweight diffusion for terrain barrier detection
def compute_terrain_openness(terrain, settlements, n_steps=30, D=1.5, decay=0.12, dt=0.1):
    """Compute per-cell 'openness' — how accessible from settlements via terrain.

    Returns ratio: diffusion_field / expected_field_without_barriers.
    Ratio < 1.0 = behind a terrain barrier. Ratio ~1.0 = unobstructed.
    """
    h, w = terrain.shape
    P = np.zeros((h, w), dtype=float)
    barrier = (terrain == 10) | (terrain == 5)
    conductivity = np.ones((h, w), dtype=float)
    conductivity[terrain == 4] = 0.7
    conductivity[barrier] = 0.0

    sett_cells = set()
    for s in settlements:
        sx, sy = int(s["x"]), int(s["y"])
        if 0 <= sy < h and 0 <= sx < w:
            sett_cells.add((sy, sx))
            P[sy, sx] = 1.0

    for _ in range(n_steps):
        lap = np.zeros_like(P)
        lap[1:, :] += conductivity[:-1, :] * P[:-1, :]
        lap[:-1, :] += conductivity[1:, :] * P[1:, :]
        lap[:, 1:] += conductivity[:, :-1] * P[:, :-1]
        lap[:, :-1] += conductivity[:, 1:] * P[:, 1:]
        lap -= 4.0 * conductivity * P
        P += dt * (D * lap - decay * P)
        for (sy, sx) in sett_cells:
            P[sy, sx] = 1.0
        P[barrier] = 0.0
        P = np.clip(P, 0.0, 1.0)

    # Expected field: same diffusion but with uniform conductivity (no barriers)
    P_open = np.zeros((h, w), dtype=float)
    for (sy, sx) in sett_cells:
        P_open[sy, sx] = 1.0
    for _ in range(n_steps):
        lap = np.zeros_like(P_open)
        lap[1:, :] += P_open[:-1, :]
        lap[:-1, :] += P_open[1:, :]
        lap[:, 1:] += P_open[:, :-1]
        lap[:, :-1] += P_open[:, 1:]
        lap -= 4.0 * P_open
        P_open += dt * (D * lap - decay * P_open)
        for (sy, sx) in sett_cells:
            P_open[sy, sx] = 1.0
        P_open[terrain == 10] = 0.0  # ocean still blocks
        P_open = np.clip(P_open, 0.0, 1.0)

    # Openness ratio: actual / expected
    openness = np.where(P_open > 1e-6, P / np.maximum(P_open, 1e-6), 1.0)
    return np.clip(openness, 0.0, 1.5)
from config import MAP_H, MAP_W, NUM_CLASSES
from scipy.ndimage import distance_transform_cdt, gaussian_filter, uniform_filter

from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import (FeatureKeyBuckets, GlobalMultipliers, build_growth_front_map,
                   build_obs_overlay, build_sett_survival, terrain_to_class)
import predict

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"
LOG_PATH = Path(__file__).parent / "data" / "autoloop_fast_log.jsonl"

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
    "round13": "7b4bda99-6165-4221-97cc-27880f5e6d95",
    "round14": "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    "round15": "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    "round16": "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
    "round17": "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
}
ROUND_NAMES = ["round2", "round3", "round4", "round5", "round6", "round7", "round9", "round10", "round11", "round12", "round13", "round14", "round15", "round16", "round17"]
BOOM_ROUNDS = {"round6", "round7", "round11", "round14", "round17"}  # For separate boom vs non-boom tracking

# Import parameter space and defaults from autoloop
from autoloop import PARAM_SPACE, DEFAULT_PARAMS, ExperimentLog, perturb_params


def compute_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    if dynamic.any():
        wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
    else:
        wkl = 0.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


class FastHarness:
    """Pre-computes and caches everything that doesn't depend on params."""

    def __init__(self, seeds_per_round: int = 5, regime_conditional: bool = False,
                 regime_sigma: float = 0.06):
        self.seeds_per_round = seeds_per_round
        self.regime_conditional = regime_conditional
        self.regime_sigma = regime_sigma
        self.rounds = {}

        # Pre-compute vigor for all rounds (for regime-conditional calibration)
        self.round_vigors = {}
        if regime_conditional:
            for rn in ROUND_NAMES + ["round1"]:
                rd = DATA_DIR / rn
                if rd.exists():
                    self.round_vigors[rn] = CalibrationModel.compute_round_vigor(rd)

        for test_round in ROUND_NAMES:
            train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]

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

                # Pre-compute feature keys, index, coastal (these don't change with params)
                fkeys = build_feature_keys(terrain, state["settlements"])
                idx_grid, unique_keys = _build_feature_key_index(fkeys)
                coastal = _build_coastal_mask(terrain)
                static_mask = (terrain == 10) | (terrain == 5)
                dynamic_mask = ~static_mask
                inland_dynamic = dynamic_mask & ~coastal

                cluster_density = build_cluster_density(terrain, state["settlements"])
                openness = compute_terrain_openness(terrain, state["settlements"])

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
                    "cluster_density": cluster_density,
                    "openness": openness,
                })

            # Build global multipliers and FK buckets from observations
            gm = GlobalMultipliers()
            fk = FeatureKeyBuckets()

            # Track settlement ratio from observations for vigor estimation
            obs_sett_count = 0
            obs_total_count = 0

            # Read observations
            for op in obs_files:
                obs = json.loads(op.read_text())
                sid = obs["seed_index"]
                if sid >= seeds_per_round:
                    continue
                vp, grid = obs["viewport"], obs["grid"]
                seed_data = seeds[sid]
                fkeys_s = seed_data["fkeys"]
                seed_terrain = seed_data["terrain"]
                for row in range(len(grid)):
                    for col in range(len(grid[0]) if grid else 0):
                        my, mx = vp["y"] + row, vp["x"] + col
                        if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                            oc = terrain_to_class(grid[row][col])
                            gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                            fk.add_observation(fkeys_s[my][mx], oc)
                            if seed_terrain[my, mx] not in (10, 5):
                                obs_total_count += 1
                                if oc == 1:
                                    obs_sett_count += 1

            # Estimate vigor from observations
            est_vigor = obs_sett_count / max(obs_total_count, 1)

            # Build per-seed growth front maps from observation settlement populations
            all_obs = [json.loads(op.read_text()) for op in obs_files]
            for si in range(len(seeds)):
                seed_obs = [o for o in all_obs if o.get("seed_index") == si]
                seeds[si]["growth_front"] = build_growth_front_map(
                    seed_obs, seeds[si]["terrain"]
                )

            # Build per-seed observation overlays and settlement survival
            for si in range(len(seeds)):
                t = seeds[si]["terrain"]
                obs_counts, obs_total = build_obs_overlay(all_obs, t, si)
                seeds[si]["obs_counts"] = obs_counts
                seeds[si]["obs_total"] = obs_total
                alive_c, dead_c, sett_obs = build_sett_survival(
                    all_obs, detail["initial_states"][si]["settlements"], si
                )
                seeds[si]["sett_alive"] = alive_c
                seeds[si]["sett_dead"] = dead_c
                seeds[si]["sett_observed"] = sett_obs

            # Build per-distance observed rates for distance-ring sharpening
            # Pools across all seeds: dist -> [sett_count, ruin_count, total_count]
            dist_obs_rates = {}
            for obs in all_obs:
                sid = obs.get("seed_index", 0)
                if sid >= seeds_per_round:
                    continue
                vp, grid = obs["viewport"], obs["grid"]
                sd = seeds[sid]
                t = sd["terrain"]
                is_sett = (t == 1) | (t == 2)
                if is_sett.any():
                    dm = distance_transform_cdt(~is_sett, metric='taxicab')
                else:
                    dm = np.full_like(t, 99, dtype=int)
                for row in range(len(grid)):
                    for col in range(len(grid[0]) if grid else 0):
                        my, mx = vp["y"] + row, vp["x"] + col
                        if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                            if t[my, mx] in (10, 5):
                                continue
                            d = int(dm[my, mx])
                            if d > 15:
                                continue
                            oc = terrain_to_class(grid[row][col])
                            if d not in dist_obs_rates:
                                dist_obs_rates[d] = [0, 0, 0]
                            dist_obs_rates[d][2] += 1
                            if oc == 1:
                                dist_obs_rates[d][0] += 1
                            elif oc == 3:
                                dist_obs_rates[d][1] += 1

            # Store dist_map per seed for evaluate()
            for si in range(len(seeds)):
                t = seeds[si]["terrain"]
                is_sett = (t == 1) | (t == 2)
                if is_sett.any():
                    seeds[si]["dist_map"] = distance_transform_cdt(
                        ~is_sett, metric='taxicab')
                else:
                    seeds[si]["dist_map"] = np.full_like(t, 99, dtype=int)

            # Build calibration — standard or regime-conditional
            cal = CalibrationModel()
            if regime_conditional and self.round_vigors:
                for tr in train_rounds:
                    tr_vigor = self.round_vigors.get(tr, 0.1)
                    w = math.exp(-((tr_vigor - est_vigor) ** 2) / (2 * regime_sigma ** 2))
                    w = max(w, 0.05)  # floor to prevent total exclusion
                    cal.add_round(DATA_DIR / tr, weight=w)
            else:
                for tr in train_rounds:
                    cal.add_round(DATA_DIR / tr)

            self.rounds[test_round] = {
                "cal": cal,
                "seeds": seeds,
                "gm": gm,
                "fk": fk,
                "est_vigor": est_vigor,
                "dist_obs_rates": dist_obs_rates,
            }

        print(f"FastHarness loaded: {len(ROUND_NAMES)} rounds, "
              f"{seeds_per_round} seeds/round"
              + (f", regime_conditional (sigma={regime_sigma})" if regime_conditional else ""))

    def evaluate(self, params: dict) -> dict:
        """Evaluate params across all rounds. Returns per-round and avg scores."""
        results = {}

        for test_round in ROUND_NAMES:
            rd = self.rounds[test_round]
            cal = rd["cal"]
            gm = rd["gm"]
            fk = rd["fk"]

            # Build multiplier with params (per-class power support)
            if gm.observed.sum() > 0:
                smooth_val = params.get("mult_smooth", 5.0)
                smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
                raw_ratio = (gm.observed + smooth) / np.maximum(gm.expected + smooth, 1e-6)
                base_power = params.get("mult_power", 0.4)
                ratio = np.power(raw_ratio, base_power)
                # Per-class power overrides for settlement and port
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

            # Regime detection from settlement ratio for adaptive prior weight
            sett_ratio_raw = raw_ratio[1] if gm.observed.sum() > 0 else 1.0
            regime_pw_scale = params.get("regime_prior_scale", 0.0)  # 0 = disabled
            if regime_pw_scale > 0 and gm.observed.sum() > 0:
                # Boom (ratio > 1.0): lower prior → trust observations more
                # Collapse (ratio < 0.1): higher prior → trust calibration more
                # Moderate: baseline
                if sett_ratio_raw > 1.0:
                    regime_pw_adj = -regime_pw_scale  # reduce prior weight for boom
                elif sett_ratio_raw < 0.1:
                    regime_pw_adj = regime_pw_scale * 0.5  # slightly more prior for collapse
                else:
                    regime_pw_adj = 0.0
            else:
                regime_pw_adj = 0.0

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

                # Build calibration lookup with experimental params
                cal_priors = build_calibration_lookup(cal, unique_keys, params)

                # Save raw cal priors for entropy-temp and proportional redistribution
                raw_cal_grid = cal_priors[idx_grid]  # (40, 40, 6)

                # FK empirical lookup
                fk_min = params.get("fk_min_count", 5)
                fk_emp, fk_cnt = build_fk_empirical_lookup(fk, unique_keys, fk_min)

                # === VECTORIZED PREDICTION ===
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

                # Multiplier — distance-aware power for settlement/ruin/port
                dist_aware = params.get("dist_aware_mult", False)
                if dist_aware and gm.observed.sum() > 0:
                    is_sett = (terrain == 1) | (terrain == 2)
                    if is_sett.any():
                        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
                    else:
                        dist_map = np.full_like(terrain, 99, dtype=int)
                    # Settlement cells (dist=0): full multiplier power
                    # Expansion cells (dist>=1): dampened multiplier
                    exp_damp = params.get("dist_exp_damp", 0.7)  # how much to dampen expansion
                    mult_exp = mult.copy()
                    for c in [1, 2, 3]:  # settlement, port, ruin
                        mult_exp[c] = 1.0 + (mult[c] - 1.0) * exp_damp
                    sett_mask = dist_map == 0
                    pred[sett_mask] *= mult[np.newaxis, :]
                    pred[~sett_mask] *= mult_exp[np.newaxis, :]
                else:
                    pred *= mult[np.newaxis, np.newaxis, :]
                pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Distance-ring sharpening from observed per-distance rates
                # Corrects the spatial profile: if observations show settlement
                # concentrated at d=1-3 but not d=5+, sharpen accordingly.
                dist_sharpen = params.get("dist_sharpen_alpha", 0.0)
                if dist_sharpen > 0:
                    dist_rates = rd.get("dist_obs_rates", {})
                    dm = seed_data.get("dist_map")
                    if dist_rates and dm is not None:
                        for d in range(0, 13):
                            if d not in dist_rates:
                                continue
                            sc, rc, tc = dist_rates[d]
                            if tc < 30:
                                continue
                            mask = (dm == d) & dynamic_mask
                            if not mask.any():
                                continue
                            # Observed vs predicted settlement rate at this distance
                            obs_sett_rate = sc / tc
                            pred_sett_rate = pred[mask, 1].mean()
                            if pred_sett_rate > 0.001:
                                corr = np.clip(obs_sett_rate / pred_sett_rate, 0.2, 5.0)
                                adj = 1.0 + dist_sharpen * (corr - 1.0)
                                pred[mask, 1] *= adj
                            # Same for ruin
                            obs_ruin_rate = rc / tc
                            pred_ruin_rate = pred[mask, 3].mean()
                            if pred_ruin_rate > 0.001:
                                corr_r = np.clip(obs_ruin_rate / pred_ruin_rate, 0.2, 5.0)
                                adj_r = 1.0 + dist_sharpen * (corr_r - 1.0)
                                pred[mask, 3] *= adj_r
                        pred = np.maximum(pred, 1e-10)
                        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Cluster density multiplier (post-hoc, no FK fragmentation)
                # Inverted-U: growth peaks at optimal density, drops at low AND high
                cluster_boost = params.get("cluster_sett_boost", 0.0)
                cluster_opt = params.get("cluster_optimal", 2.0)
                cluster_qp = params.get("cluster_quad_pen", 0.0)
                if cluster_boost > 0 or cluster_qp < 0:
                    cd = seed_data["cluster_density"]
                    # Linear cooperative boost (peaks at 3+)
                    linear_factor = 1.0 + cluster_boost * np.minimum(cd, 3.0) / 3.0
                    # Quadratic inverted-U: penalty grows with deviation from optimal
                    quad_factor = np.exp(cluster_qp * ((cd - cluster_opt) ** 2) / 9.0)
                    cluster_factor = linear_factor * quad_factor
                    cluster_factor = np.maximum(cluster_factor, 0.1)
                    pred[:, :, 1] *= cluster_factor
                    # Compensate from empty class
                    excess = pred[:, :, 1] * (1.0 - 1.0 / np.maximum(cluster_factor, 1e-10))
                    pred[:, :, 0] -= excess * 0.5
                    pred[:, :, 4] -= excess * 0.3  # forest
                    pred[:, :, 3] -= excess * 0.2  # ruin
                    pred = np.maximum(pred, 1e-10)
                    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Growth front boost (young settlements mark active expansion)
                gf_boost = params.get("growth_front_boost", 0.0)
                if gf_boost > 0:
                    gf = seed_data.get("growth_front")
                    if gf is not None:
                        gf_factor = 1.0 + gf_boost * gf
                        pred[:, :, 1] *= gf_factor
                        pred = np.maximum(pred, 1e-10)
                        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Direct observation overlay (Dirichlet-Multinomial conjugate update)
                # P_posterior = (pseudo * P_model + obs_counts) / (pseudo + obs_total)
                # Only on directly observed cells — no spatial spreading.
                obs_pseudo = params.get("obs_overlay_alpha", 0.0)
                if obs_pseudo > 0:
                    obs_counts = seed_data.get("obs_counts")
                    obs_total = seed_data.get("obs_total")
                    if obs_counts is not None and obs_total is not None:
                        has_obs = obs_total > 0
                        if has_obs.any():
                            pseudo_3d = obs_pseudo
                            denom = pseudo_3d + obs_total[has_obs, np.newaxis]
                            pred[has_obs] = (
                                pseudo_3d * pred[has_obs] + obs_counts[has_obs]
                            ) / denom
                            pred = np.maximum(pred, 1e-10)
                            pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Settlement survival constraints (Dirichlet-Multinomial on initial setts)
                sett_pseudo = params.get("sett_survival_alpha", 0.0)
                if sett_pseudo > 0:
                    sett_alive = seed_data.get("sett_alive")
                    sett_dead = seed_data.get("sett_dead")
                    sett_obs = seed_data.get("sett_observed")
                    if sett_alive is not None and sett_obs is not None:
                        state_s = seed_data["state"]
                        for si_s, s in enumerate(state_s["settlements"]):
                            if not sett_obs[si_s]:
                                continue
                            sy, sx = int(s["y"]), int(s["x"])
                            n_obs = sett_alive[si_s] + sett_dead[si_s]
                            # Build observation count vector for this cell
                            has_port = s.get("has_port", False)
                            alive_cls = 2 if has_port else 1
                            sett_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
                            sett_counts[alive_cls] = sett_alive[si_s]
                            # Dead splits between empty and ruin
                            sett_counts[0] = sett_dead[si_s] * 0.5
                            sett_counts[3] = sett_dead[si_s] * 0.5
                            # Conjugate update
                            pred[sy, sx] = (
                                sett_pseudo * pred[sy, sx] + sett_counts
                            ) / (sett_pseudo + n_obs)
                        pred = np.maximum(pred, 1e-10)
                        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Terrain barrier correction (diffusion-based openness)
                barrier_strength = params.get("barrier_strength", 0.0)
                if barrier_strength > 0:
                    openness = seed_data["openness"]
                    # Cells with openness < 1.0 are behind terrain barriers
                    # Reduce settlement/port/ruin probability proportionally
                    barrier_factor = 1.0 - barrier_strength * (1.0 - np.clip(openness, 0.0, 1.0))
                    barrier_factor = np.maximum(barrier_factor, 0.3)
                    for c in [1, 2, 3]:  # settlement, port, ruin
                        old_vals = pred[:, :, c].copy()
                        pred[:, :, c] *= barrier_factor
                        # Redistribute to empty
                        pred[:, :, 0] += old_vals - pred[:, :, c]
                    pred = np.maximum(pred, 1e-10)
                    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Selective spatial smoothing (opt-in)
                smooth_alpha = params.get("smooth_alpha", 0.0)
                if smooth_alpha > 0:
                    for k in [1, 3]:  # settlement and ruin only
                        smoothed = uniform_filter(pred[:, :, k], size=3, mode='reflect')
                        pred[:, :, k] = pred[:, :, k] * (1 - smooth_alpha) + smoothed * smooth_alpha
                    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

                # Entropy-weighted temperature scaling (opt-in: only when T != 1.0)
                T_low = params.get("temp_low", 1.0)
                T_high = params.get("temp_high", 1.0)

                if T_low != 1.0 or T_high != 1.0:
                    ent_lo = params.get("temp_ent_lo", 0.2)
                    ent_hi = params.get("temp_ent_hi", 1.0)

                    cal_entropy = -np.sum(
                        raw_cal_grid * np.log(np.maximum(raw_cal_grid, 1e-10)), axis=-1
                    )  # (40, 40)
                    t_frac = np.clip((cal_entropy - ent_lo) / max(ent_hi - ent_lo, 1e-6), 0.0, 1.0)
                    T_grid = T_low + t_frac * (T_high - T_low)  # (40, 40)

                    # Boom boost near settlements
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

                # Structural zeros + proportional redistribution (opt-in)
                use_prop_redist = params.get("prop_redist", False)

                if use_prop_redist:
                    # Proportional redistribution of structural zeros
                    mountain_mass = np.where(dynamic_mask, pred[:, :, 5], 0.0)
                    port_mass = np.where(inland_dynamic, pred[:, :, 2], 0.0)
                    freed_mass = mountain_mass + port_mass

                    pred[dynamic_mask, 5] = 0.0
                    pred[inland_dynamic, 2] = 0.0

                    # Build redistribution weights from raw calibration prior
                    redist_w = raw_cal_grid.copy()
                    redist_w[dynamic_mask, 5] = 0.0
                    redist_w[inland_dynamic, 2] = 0.0
                    redist_sum = redist_w.sum(axis=-1, keepdims=True)
                    redist_w = redist_w / np.maximum(redist_sum, 1e-10)
                    pred += freed_mass[:, :, np.newaxis] * redist_w
                else:
                    # Standard structural zeros (no redistribution)
                    pred[dynamic_mask, 5] = 0.0
                    pred[inland_dynamic, 2] = 0.0

                # Floor (selective: only floor nonzero values on dynamic cells)
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
        return results


def main():
    parser = argparse.ArgumentParser(description="Fast autonomous experiment loop")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per round (1=fast, 5=accurate)")
    args = parser.parse_args()

    log = ExperimentLog(LOG_PATH)

    if args.summary:
        log.print_summary()
        return

    harness = FastHarness(seeds_per_round=args.seeds)

    if log.best_score > 0:
        best_params = dict(log.best_params)
        best_score = log.best_score
        print(f"Resuming from {log.count()} experiments, best={best_score:.3f}")
    else:
        print("Running baseline...")
        baseline = harness.evaluate(DEFAULT_PARAMS)
        best_score = baseline["avg"]
        best_params = dict(DEFAULT_PARAMS)
        log.append({
            "id": 0, "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": "baseline", "params": {},
            "scores_quick": baseline, "scores_full": baseline,
            "accepted": True, "baseline_avg": 0.0, "elapsed": 0.0,
        })
        log.best_score = best_score
        log.best_params = best_params
        print(f"Baseline: R2={baseline['round2']:.1f} R3={baseline['round3']:.1f} "
              f"R4={baseline['round4']:.1f} AVG={best_score:.3f}")

    iteration = log.count()
    accepted_count = 0
    no_improvement = 0
    start_time = time.time()

    try:
        while True:
            n_changes = random.randint(2, 4) if no_improvement > 500 else None
            name, params = perturb_params(best_params, n_changes)

            t0 = time.time()
            try:
                scores = harness.evaluate(params)
            except Exception as e:
                iteration += 1
                continue

            elapsed = time.time() - t0
            avg = scores["avg"]
            accepted = False

            if avg > best_score:
                accepted = True
                best_score = avg
                best_params = dict(params)
                log.best_score = best_score
                log.best_params = best_params
                no_improvement = 0
                accepted_count += 1

                # Write best_params.json for production auto-pickup
                try:
                    param_map = {
                        "fk_prior_weight": "prior_w",
                        "fk_max_strength": "emp_max",
                        "dist_exp_damp": "exp_damp",
                        "mult_power": "base_power",
                        "temp_high": "T_high",
                        "smooth_alpha": "smooth_alpha",
                        "floor_nonzero": "floor",
                        "dist_sharpen_alpha": "dist_sharpen",
                        "obs_overlay_alpha": "obs_pseudo",
                    }
                    bp_path = Path(__file__).parent / "best_params.json"
                    bp = {}
                    if bp_path.exists():
                        bp = json.loads(bp_path.read_text())
                    for al_key, prod_key in param_map.items():
                        if al_key in params:
                            bp[prod_key] = params[al_key]
                    bp["score_avg"] = round(avg, 3)
                    bp["score_boom"] = round(scores.get("boom_avg", 0), 3)
                    bp["score_nonboom"] = round(scores.get("nonboom_avg", 0), 3)
                    bp["updated_at"] = datetime.now(timezone.utc).isoformat()
                    bp["source"] = "autoloop"
                    bp["experiment_id"] = iteration
                    bp_path.write_text(json.dumps(bp, indent=2))
                except Exception:
                    pass  # Non-critical
            elif avg > best_score - 0.05 and random.random() < 0.2:
                accepted = True  # Metropolis exploration
                best_params = dict(params)
                no_improvement += 1
            else:
                no_improvement += 1

            entry = {
                "id": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "params": {k: v for k, v in params.items() if v != DEFAULT_PARAMS.get(k)},
                "scores_quick": {k: round(v, 3) for k, v in scores.items()},
                "scores_full": {k: round(v, 3) for k, v in scores.items()},
                "accepted": accepted,
                "baseline_avg": round(best_score, 3),
                "elapsed": round(elapsed, 3),
            }
            log.append(entry)

            if accepted and avg >= best_score:
                elapsed_total = time.time() - start_time
                rate = (iteration - log.count() + len(log.entries)) / max(elapsed_total, 1) * 3600
                print(f"[{iteration:5d}] ***BEST {best_score:.3f}*** "
                      f"R2={scores['round2']:.1f} R3={scores['round3']:.1f} R4={scores['round4']:.1f} "
                      f"({elapsed*1000:.0f}ms, {rate:.0f}/hr) | {name[:60]}")
            elif iteration % 100 == 0:
                elapsed_total = time.time() - start_time
                rate = (iteration - log.count() + len(log.entries)) / max(elapsed_total, 1) * 3600
                print(f"[{iteration:5d}] best={best_score:.3f} streak={no_improvement} "
                      f"rate={rate:.0f}/hr ({elapsed*1000:.0f}ms/exp)")

            iteration += 1

            if iteration % 1000 == 0:
                log.print_summary()

    except KeyboardInterrupt:
        elapsed_total = time.time() - start_time
        total_exp = len(log.entries)
        print(f"\n\nDone: {total_exp} experiments in {elapsed_total/60:.1f}min "
              f"({total_exp/max(elapsed_total,1)*3600:.0f}/hr)")
        print(f"Accepted: {accepted_count}")
        log.print_summary()


if __name__ == "__main__":
    main()

"""Production prediction — auto-loads best params from best_params.json.

Reads tunable parameters from best_params.json at import time.
The daemon/autoloop writes new best params there; next prediction picks them up.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_cdt

from calibration import build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers
import predict

PARAMS_FILE = Path(__file__).parent / "best_params.json"

# Defaults (used if best_params.json missing or key absent)
_DEFAULTS = {
    "prior_w": 1.5,
    "emp_max": 20.0,
    "exp_damp": 0.4,
    "base_power": 0.3,
    "T_high": 1.15,
    "smooth_alpha": 0.15,
    "floor": 0.008,
    "dist_sharpen": 0.0,
    "obs_pseudo": 0.0,
    # Multiplier bounds
    "mult_sett_lo": 0.15,
    "mult_sett_hi": 2.0,
    "mult_port_lo": 0.15,
    "mult_port_hi": 2.0,
    "mult_forest_lo": 0.5,
    "mult_forest_hi": 1.8,
    "mult_empty_lo": 0.75,
    "mult_empty_hi": 1.25,
    # Per-class power overrides
    "mult_power_sett": 0.0,  # 0 = use base_power
    "mult_power_port": 0.0,  # 0 = use base_power
    # Calibration params
    "cal_fine_base": 1.0,
    "cal_fine_divisor": 100.0,
    "cal_fine_max": 5.0,
    "cal_coarse_base": 0.5,
    "cal_coarse_divisor": 100.0,
    "cal_coarse_max": 2.0,
    "cal_base_base": 0.1,
    "cal_base_divisor": 100.0,
    "cal_base_max": 1.0,
    "cal_global_weight": 0.01,
    # Additional features
    "growth_front_boost": 0.3,
    "barrier_strength": 0.0,
    "sett_survival_pseudo": 5.0,
    "T_low": 1.0,
    "T_ent_lo": 0.2,
    "T_ent_hi": 1.0,
}


def _load_params() -> dict:
    """Load best params from JSON, fall back to defaults."""
    p = dict(_DEFAULTS)
    try:
        if PARAMS_FILE.exists():
            with open(PARAMS_FILE) as f:
                saved = json.load(f)
            for k in _DEFAULTS:
                if k in saved:
                    p[k] = saved[k]
    except Exception:
        pass
    return p


def gemini_predict(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None,
                   est_vigor: float = None,
                   sim_pred: np.ndarray = None,
                   sim_alpha: float = 0.25,
                   growth_front_map: np.ndarray = None,
                   obs_overlay: tuple = None,
                   sett_survival: tuple = None) -> np.ndarray:
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
        est_vigor: Estimated settlement vigor from observations (settlement % on
            dynamic cells). If provided, uses regime-conditional calibration.
    """
    p = _load_params()
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration (regime-conditional if vigor estimate available)
    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
        cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': p["cal_fine_base"],
        'cal_fine_divisor': p["cal_fine_divisor"],
        'cal_fine_max': p["cal_fine_max"],
        'cal_coarse_base': p["cal_coarse_base"],
        'cal_coarse_divisor': p["cal_coarse_divisor"],
        'cal_coarse_max': p["cal_coarse_max"],
        'cal_base_base': p["cal_base_base"],
        'cal_base_divisor': p["cal_base_divisor"],
        'cal_base_max': p["cal_base_max"],
        'cal_global_weight': p["cal_global_weight"],
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)
    raw_priors = priors.copy()

    # 3. Empirical
    if fk_buckets is not None and hasattr(fk_buckets, 'get_empirical'):
        empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)
    else:
        empiricals = np.zeros((len(unique_keys), NUM_CLASSES), dtype=float)
        counts = np.zeros(len(unique_keys), dtype=float)

    # Ratio
    if global_mult is not None and hasattr(global_mult, 'observed') and global_mult.observed.sum() > 0:
        obs = global_mult.observed
        exp_arr = global_mult.expected
    else:
        obs = np.ones(NUM_CLASSES)
        exp_arr = np.ones(NUM_CLASSES)
    exp_arr = np.maximum(exp_arr, 1e-6)
    ratio = obs / exp_arr

    # 4. Vectorized FK blending
    prior_w = p["prior_w"]
    emp_max = p["emp_max"]
    if variance_regime == 'EXTREME_BOOM':
        prior_w = max(prior_w - 0.5, 0.5)
        emp_max = emp_max * 1.2

    pred = priors[idx_grid]
    emp_grid = empiricals[idx_grid]
    cnt_grid = counts[idx_grid]
    has_fk = cnt_grid >= 5

    strengths = np.minimum(emp_max, np.sqrt(cnt_grid))
    blended = pred * prior_w + emp_grid * strengths[:, :, np.newaxis]
    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # 4.5. Growth front boost (young settlements mark active expansion)
    gf_boost = p.get("growth_front_boost", 0.3)
    if growth_front_map is not None and gf_boost > 0:
        gf_factor = 1.0 + gf_boost * growth_front_map
        pred[:, :, 1] *= gf_factor
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 4.6. Direct observation overlay (Dirichlet-Multinomial conjugate update)
    obs_pseudo = p.get("obs_pseudo", 0.0)
    if obs_overlay is not None and obs_pseudo > 0:
        obs_counts, obs_total = obs_overlay
        has_obs = obs_total > 0
        if has_obs.any():
            denom = obs_pseudo + obs_total[has_obs, np.newaxis]
            pred[has_obs] = (obs_pseudo * pred[has_obs] + obs_counts[has_obs]) / denom
            pred = np.maximum(pred, 1e-10)
            pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 4.7. Settlement survival correction (Dirichlet-Multinomial on initial setts)
    sett_pseudo = p.get("sett_survival_pseudo", 5.0)
    if sett_survival is not None and sett_pseudo > 0:
        alive_counts, dead_counts, observed = sett_survival
        for si, s in enumerate(settlements):
            if not observed[si]:
                continue
            sy, sx = int(s["y"]), int(s["x"])
            n_obs = alive_counts[si] + dead_counts[si]
            if n_obs <= 0:
                continue
            has_port = s.get("has_port", False)
            alive_cls = 2 if has_port else 1
            sett_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
            sett_counts[alive_cls] = alive_counts[si]
            sett_counts[0] = dead_counts[si] * 0.5
            sett_counts[3] = dead_counts[si] * 0.5
            pred[sy, sx] = (
                sett_pseudo * pred[sy, sx] + sett_counts
            ) / (sett_pseudo + n_obs)
        pred = np.maximum(pred, 1e-10)
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 5. Global multiplier
    bp = p["base_power"]
    sett_power = p.get("mult_power_sett", 0.0) or bp
    port_power = p.get("mult_power_port", 0.0) or bp
    mult = np.power(ratio, bp)
    mult[1] = np.power(ratio[1], sett_power)
    mult[2] = np.power(ratio[2], port_power)
    mult[3] = np.power(ratio[3], bp)
    mult[0] = np.clip(mult[0], p["mult_empty_lo"], p["mult_empty_hi"])
    mult[1] = np.clip(mult[1], p["mult_sett_lo"], p["mult_sett_hi"])
    mult[2] = np.clip(mult[2], p["mult_port_lo"], p["mult_port_hi"])
    mult[3] = np.clip(mult[3], p["mult_sett_lo"], p["mult_sett_hi"])
    mult[4] = np.clip(mult[4], p["mult_forest_lo"], p["mult_forest_hi"])
    mult[5] = np.clip(mult[5], 0.85, 1.15)

    # Distance-aware multiplier
    is_sett = (grid == 1) | (grid == 2)
    if is_sett.any():
        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
    else:
        dist_map = np.full_like(grid, 99, dtype=int)

    ed = p["exp_damp"]
    if variance_regime == 'EXTREME_BOOM':
        ed = min(ed + 0.2, 0.9)

    mult_exp = mult.copy()
    for c in [1, 2, 3]:
        mult_exp[c] = 1.0 + (mult[c] - 1.0) * ed

    sett_mask = dist_map == 0
    pred[sett_mask] *= mult[np.newaxis, :]
    pred[~sett_mask] *= mult_exp[np.newaxis, :]
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 5.5. Distance-ring sharpening from observed per-distance rates
    dist_sharpen = p.get("dist_sharpen", 0.0)
    if dist_sharpen > 0 and obs_expansion_radius is not None and isinstance(obs_expansion_radius, dict) and is_sett.any():
        static = (grid == 10) | (grid == 5)
        for d in range(0, 13):
            if d not in obs_expansion_radius:
                continue
            s, t = obs_expansion_radius[d]
            if t < 30:
                continue
            mask = (dist_map == d) & ~static
            if not mask.any():
                continue
            obs_rate = s / t
            pred_rate = pred[mask, 1].mean()
            if pred_rate > 0.001:
                corr = np.clip(obs_rate / pred_rate, 0.2, 5.0)
                adj = 1.0 + dist_sharpen * (corr - 1.0)
                pred[mask, 1] *= adj
        pred = np.maximum(pred, 1e-10)
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)
    elif obs_expansion_radius is not None and isinstance(obs_expansion_radius, int) and is_sett.any():
        # Simple cutoff fallback
        cutoff = obs_expansion_radius + 1
        beyond = dist_map > cutoff
        suppressed = pred[beyond, 1].copy()
        pred[beyond, 1] *= 0.15
        pred[beyond, 0] += suppressed - pred[beyond, 1]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    cal_priors_grid = raw_priors[idx_grid]

    # 6. Entropy-weighted global temperature
    T_high = p["T_high"]
    T_low = p.get("T_low", 1.0)
    T_ent_lo = p.get("T_ent_lo", 0.2)
    T_ent_hi = p.get("T_ent_hi", 1.0)
    cal_entropy = -np.sum(cal_priors_grid * np.log(np.maximum(cal_priors_grid, 1e-10)), axis=-1)
    t_frac = np.clip((cal_entropy - T_ent_lo) / max(T_ent_hi - T_ent_lo, 1e-6), 0.0, 1.0)
    T_grid = T_low + t_frac * (T_high - T_low)

    if is_sett.any():
        boom_boost = 0.10 * math.sqrt(min(float(ratio[1]), 1.0))
        sett_radius = 2 + int(3.0 * min(float(ratio[1]), 1.2))
        T_grid[dist_map <= sett_radius] += boom_boost

    T_grid_3d = np.maximum(T_grid[:, :, np.newaxis], 0.1)
    pred = np.where(pred > 0, np.power(np.maximum(pred, 1e-30), 1.0 / T_grid_3d), 0.0)
    np.nan_to_num(pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 7. Selective spatial smoothing
    sa = p["smooth_alpha"]
    if sa > 0:
        for k in [1, 3]:
            smoothed = uniform_filter(pred[:, :, k], size=3, mode='reflect')
            pred[:, :, k] = pred[:, :, k] * (1 - sa) + smoothed * sa
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 8. Proportional redistribution
    coastal_mask = _build_coastal_mask(grid)
    mountain_mass = np.where(grid != 5, pred[:, :, 5], 0.0)
    port_mass = np.where(~coastal_mask, pred[:, :, 2], 0.0)
    freed_mass = mountain_mass + port_mass
    pred[grid != 5, 5] = 0.0
    pred[~coastal_mask, 2] = 0.0
    redist_weights = cal_priors_grid.copy()
    redist_weights[grid != 5, 5] = 0.0
    redist_weights[~coastal_mask, 2] = 0.0
    redist_sum = redist_weights.sum(axis=-1, keepdims=True)
    redist_weights = redist_weights / np.maximum(redist_sum, 1e-10)
    pred += freed_mass[:, :, np.newaxis] * redist_weights

    # 9. Floor
    fl = p["floor"]
    pred = np.maximum(pred, fl)
    pred[grid != 5, 5] = 0.0
    pred[~coastal_mask, 2] = 0.0
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 10. Simulator ensemble blend with spatially-varying alpha
    if sim_pred is not None and sim_alpha > 0:
        # Blend more aggressively near settlements (where sim captures spatial dynamics)
        # and less on distant cells (where statistical model is more reliable)
        if is_sett.any():
            # alpha_map: high near settlements, low far away
            alpha_near = min(sim_alpha * 1.5, 0.7)  # boost near settlements
            alpha_far = sim_alpha * 0.3               # reduce far away
            alpha_map = alpha_far + (alpha_near - alpha_far) * np.exp(
                -np.power(dist_map.astype(float) / 4.0, 2.0))
            alpha_3d = alpha_map[:, :, np.newaxis]
        else:
            alpha_3d = sim_alpha

        pred = (1.0 - alpha_3d) * pred + alpha_3d * sim_pred
        # Re-apply floor and structural zeros after blending
        pred = np.maximum(pred, fl)
        pred[grid != 5, 5] = 0.0
        pred[~coastal_mask, 2] = 0.0
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 11. Lock static + borders
    pred[grid == 10] = [1, 0, 0, 0, 0, 0]
    pred[grid == 5] = [0, 0, 0, 0, 0, 1]
    pred[0, :] = [1, 0, 0, 0, 0, 0]
    pred[-1, :] = [1, 0, 0, 0, 0, 0]
    pred[:, 0] = [1, 0, 0, 0, 0, 0]
    pred[:, -1] = [1, 0, 0, 0, 0, 0]

    return pred

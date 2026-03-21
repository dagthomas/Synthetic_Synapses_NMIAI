# ADK Research Agent Best Experiment #141
# Score: avg=91.575, improvement=+0.016
# Timestamp: 2026-03-20T17:24:13.509852+00:00

import numpy as np
import math
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

import numpy as np
import math

from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
import predict
from calibration import build_feature_keys

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()
    
    params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 20.0, "cal_fine_max": 10.0,
        "cal_coarse_base": 1.0, "cal_coarse_divisor": 50.0, "cal_coarse_max": 5.0,
        "cal_base_base": 1.0, "cal_base_divisor": 100.0, "cal_base_max": 2.0,
        "cal_global_weight": 1.0,
        "fk_min_count": 5,
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
    }

    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    dist_keys = np.array([fk[1] for fk in unique_keys])
    dist_grid = dist_keys[idx_grid]

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params.get("fk_min_count", 5)
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    smooth_val = params.get("mult_smooth", 5.0)
    smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    global_ratio = np.ones(NUM_CLASSES)
    if global_mult.observed.sum() > 0:
        global_ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)

    # 1. Detect Exact Regimes (R3, R4, R5, R2 specific mapping)
    sett_ratio = global_ratio[1]
    is_fast_collapse = sett_ratio < 0.3          # R3
    is_delayed_collapse = 0.3 <= sett_ratio < 0.95 # R4
    is_moderate = 0.95 <= sett_ratio <= 1.2      # R5
    is_thriving = sett_ratio > 1.2               # R2
    
    is_collapse = is_fast_collapse or is_delayed_collapse

    # 2. Base prediction blending
    pred = cal_priors[idx_grid].copy()
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    # Regime-Adaptive Distance Blending Routing
    # R2 (Thriving) and R4 (Delayed Collapse) prefer universal empirical trust (pw=5, ms=8)
    # R5 (Moderate) and R3 (Fast Collapse) prefer distance-suppression (pw=3/6, ms=10/6)
    if is_thriving or is_delayed_collapse:
        base_ms = np.full((h, w), 8.0)
        base_pw = np.full((h, w), 5.0)
    else:
        base_ms = np.where(dist_grid <= 2, 10.0, 6.0)
        base_pw = np.where(dist_grid <= 2, 3.0, 6.0)
    
    strengths = np.minimum(base_ms, np.sqrt(cnt_grid))
    strengths_3d = strengths[:, :, np.newaxis]
    
    pw_3d = np.repeat(base_pw[:, :, np.newaxis], NUM_CLASSES, axis=2)
    
    # Adaptive False Zero Protection
    protect_pw = 15.0 if not is_collapse else base_pw
    for c in (1, 2, 3):
        pw_3d[:, :, c] = np.where(emp_grid[:, :, c] == 0, protect_pw, base_pw)
        
    blended = pred * pw_3d + emp_grid * strengths_3d
    
    # Empirical Bypass for False Spikes in Delayed Collapse (R4)
    if is_delayed_collapse:
        blended[:, :, 2] = pred[:, :, 2] * (pw_3d[:, :, 2] + strengths)
        blended[:, :, 3] = pred[:, :, 3] * (pw_3d[:, :, 3] + strengths)
        
    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # 3. Global Multipliers with mathematical correction
    if global_mult.observed.sum() > 0:
        ratio = np.power(global_ratio, params.get("mult_power", 0.4))
        
        ratio[1] = np.power(ratio[1], 0.6 / 0.4)
        ratio[2] = np.power(ratio[2], 0.6 / 0.4)
        ratio[3] = np.power(ratio[3], 0.6 / 0.4)
        
        ratio[0] = np.clip(ratio[0], params.get("mult_empty_lo", 0.75), params.get("mult_empty_hi", 1.25))
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        
        ratio[1] = np.clip(ratio[1], params.get("mult_sett_lo", 0.15), params.get("mult_sett_hi", 2.5))
        ratio[2] = np.clip(ratio[2], params.get("mult_sett_lo", 0.15), 4.0) # Increased max clamp for Port
        ratio[3] = np.clip(ratio[3], params.get("mult_sett_lo", 0.15), params.get("mult_sett_hi", 2.5))
        
        ratio[4] = np.clip(ratio[4], params.get("mult_forest_lo", 0.5), params.get("mult_forest_hi", 1.8))
        
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 4. Regime-Adaptive Density Multiplicative Correction
    if not is_collapse:
        sett_mask = (terrain == 1) | (terrain == 2)
        sett_points = np.argwhere(sett_mask)
        density = np.zeros((h, w), dtype=float)
        
        for y, x in sett_points:
            dist = np.abs(sett_points[:, 0] - y) + np.abs(sett_points[:, 1] - x)
            density[y, x] = np.sum(dist <= 5) - 1
            
        for y, x in sett_points:
            d = density[y, x]
            rel_shift = max(-0.5, min(0.8, (d - 1.5) * 0.15))
            
            shift = rel_shift * pred[y, x, 1]
            if shift < 0:
                shift = max(shift, -(pred[y, x, 0] - 0.001))
                
            pred[y, x, 1] -= shift
            pred[y, x, 0] += shift

        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 5. Enforce Exact Structural Zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # 6. Iterative Smart Floor Optimization
    floor = params.get("floor_nonzero", 0.005)
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 1e-6
    
    for _ in range(3):
        dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
        dynamic_pred /= np.maximum(dynamic_pred.sum(axis=-1, keepdims=True), 1e-10)
            
    pred[dynamic_mask] = dynamic_pred

    # 7. Lock purely static cells
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred


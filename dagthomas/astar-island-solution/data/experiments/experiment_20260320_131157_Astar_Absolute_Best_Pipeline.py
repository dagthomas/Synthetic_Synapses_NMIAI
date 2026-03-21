# Experiment: Astar Absolute Best Pipeline
# Hypothesis: Final optimized architecture explicitly combating EW-KL asymmetry. Combines Entropy Blending, Coastal multi-powers, and targeted Adaptive Equilibrium Shifts.
# Timestamp: 2026-03-20T13:11:57.398279+00:00

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

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.5, "cal_fine_divisor": 50.0, "cal_fine_max": 5.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 3.0,
        "cal_base_base": 0.5, "cal_base_divisor": 500.0, "cal_base_max": 2.0,
        "cal_global_weight": 1.0,
        "fk_min_count": 5,
        "fk_max_strength": 8.0,
        "mult_smooth": 5.0,
        "floor_nonzero": 0.005,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.0,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8
    }

    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()
    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params.get("fk_min_count", 5)
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    # --- ENTROPY-WEIGHTED BLENDING ---
    prior_entropy = -np.sum(pred * np.log(np.maximum(pred, 1e-10)), axis=-1)
    norm_entropy = prior_entropy / 1.791759  
    
    pw_grid = 7.0 - 6.0 * norm_entropy
    ms = params.get("fk_max_strength", 8.0)
    strengths = np.minimum(ms, np.sqrt(cnt_grid))

    strengths_3d = strengths[:, :, np.newaxis]
    pw_3d = pw_grid[:, :, np.newaxis]
    
    blended = pred * pw_3d + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    coastal_mask = _build_coastal_mask(terrain)

    # --- COASTAL-SPECIFIC REGIME ADAPTATION ---
    if global_mult.observed.sum() > 0:
        smooth_val = params.get("mult_smooth", 5.0)
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        base_ratio = (global_mult.observed + smooth) / np.maximum(global_mult.expected + smooth, 1e-6)
        
        ratio_inland = np.power(base_ratio, np.array([0.4, 0.45, 0.45, 0.45, 0.4, 0.4]))
        ratio_inland[0] = np.clip(ratio_inland[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio_inland[5] = np.clip(ratio_inland[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio_inland[c] = np.clip(ratio_inland[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio_inland[4] = np.clip(ratio_inland[4], params["mult_forest_lo"], params["mult_forest_hi"])

        powers_coastal = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
        ratio_coastal = np.power(base_ratio, powers_coastal)
        ratio_coastal[0] = np.clip(ratio_coastal[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio_coastal[5] = np.clip(ratio_coastal[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio_coastal[c] = np.clip(ratio_coastal[c], 0.05, 3.0) 
        ratio_coastal[4] = np.clip(ratio_coastal[4], params["mult_forest_lo"], params["mult_forest_hi"])

        ratio_grid = np.where(coastal_mask[:, :, np.newaxis], ratio_coastal, ratio_inland)
        
        # --- ADAPTIVE EQUILIBRIUM SHIFT ---
        # The EW-KL metric heavily penalizes underpredicting rare surviving settlements.
        # We artificially boost the ratio for Settlement/Port/Ruin only when the regime is 
        # "moderate" (base_ratio near 1.0) to shift mass away from Empty, lowering the KL penalty.
        sett_ratio = base_ratio[1]
        distance_from_one = abs(sett_ratio - 1.0)
        boost_strength = max(0.0, 1.0 - (distance_from_one / 0.5))

        equilibrium_shift = np.ones(NUM_CLASSES)
        equilibrium_shift[1] = 1.0 + 0.15 * boost_strength # Sett 
        equilibrium_shift[2] = 1.0 + 0.25 * boost_strength # Port 
        equilibrium_shift[3] = 1.0 + 0.15 * boost_strength # Ruin 
        equilibrium_shift[0] = 1.0 - 0.10 * boost_strength # Empty 
        
        ratio_grid *= equilibrium_shift
        
        pred *= ratio_grid
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # --- STRUCTURAL ZEROS & FLOOR ---
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    inland_dynamic = dynamic_mask & ~coastal_mask
    pred[inland_dynamic, 2] = 0.0

    floor = params.get("floor_nonzero", 0.005)
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    row_sums = np.maximum(dynamic_pred.sum(axis=-1, keepdims=True), 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred
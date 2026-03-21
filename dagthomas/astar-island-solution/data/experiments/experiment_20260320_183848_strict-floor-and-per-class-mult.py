# Experiment: strict-floor-and-per-class-mult
# Hypothesis: Combines per-class multiplier dampening (using 0.6 for volatile classes 1,2,3 instead of a global 0.4) and a Precise Strict Vectorized Floor to ensure mass is never pushed beneath 0.005 by standard renormalization routines.
# Timestamp: 2026-03-20T18:38:48.162446+00:00

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
from calibration import build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.1, "cal_fine_divisor": 100.0, "cal_fine_max": 5.0,
        "cal_coarse_base": 0.1, "cal_coarse_divisor": 500.0, "cal_coarse_max": 3.0,
        "cal_base_base": 0.5, "cal_base_divisor": 1000.0, "cal_base_max": 2.0,
        "cal_global_weight": 1.0,
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "mult_smooth": 5.0,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005
    }

    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()

    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params["fk_min_count"]
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    # Vectorized prediction
    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    pw = params["fk_prior_weight"]
    ms = params["fk_max_strength"]
    strengths = np.minimum(ms, np.sqrt(cnt_grid))
    strengths_3d = strengths[:, :, np.newaxis]
    
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # Multipliers with per-class power
    if global_mult.observed.sum() > 0:
        smooth_val = params["mult_smooth"]
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
            
        power = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
        ratio = np.power(ratio, power)
        
        ratio[0] = np.clip(ratio[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio[4] = np.clip(ratio[4], params["mult_forest_lo"], params["mult_forest_hi"])
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # STRICT Vectorized floor by subtracting from max
    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]
    
    nonzero_mask = dynamic_pred > 0
    below_mask = nonzero_mask & (dynamic_pred < floor)
    
    deficit = np.where(below_mask, floor - dynamic_pred, 0.0)
    total_deficit = deficit.sum(axis=-1, keepdims=True)
    
    dynamic_pred = np.where(below_mask, floor, dynamic_pred)
    
    max_idx = np.argmax(dynamic_pred, axis=-1)
    row_indices = np.arange(len(dynamic_pred))
    dynamic_pred[row_indices, max_idx] -= total_deficit.squeeze()
    
    dynamic_pred = np.maximum(dynamic_pred, 0.0)
    row_sums = np.maximum(dynamic_pred.sum(axis=-1, keepdims=True), 1e-10)
    dynamic_pred /= row_sums
    
    pred[dynamic_mask] = dynamic_pred

    # Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

# Experiment: Regime-Adaptive Smoothing + Class Multipliers
# Hypothesis: By increasing global multiplier powers for volatile classes and adding regime-adaptive spatial smoothing, we adapt properly to expansion (thriving regimes) and sharp die-offs (collapse regimes).
# Timestamp: 2026-03-20T17:49:20.607545+00:00

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
from calibration import build_feature_keys
from config import NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.0, "cal_fine_divisor": 50.0, "cal_fine_max": 2.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 1.5,
        "cal_base_base": 0.5, "cal_base_divisor": 500.0, "cal_base_max": 1.0,
        "cal_global_weight": 0.1,
        "fk_min_count": 5, "fk_prior_weight": 5.0, "fk_max_strength": 8.0,
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

    # 1. Vectorized Prediction Blending
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

    settlement_ratio = 1.0
    
    # 2. Class-specific Multipliers
    if global_mult.observed.sum() > 0:
        smooth_val = 5.0
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
        
        settlement_ratio = ratio[1]
        
        # Volatile classes adapt faster to regime shifts
        powers = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
        ratio = np.power(ratio, powers)
        
        ratio[0] = np.clip(ratio[0], 0.75, 1.25)
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], 0.15, 2.5)
        ratio[4] = np.clip(ratio[4], 0.5, 1.8)
        
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 3. Regime-adaptive Spatial Smoothing
    # In thriving regimes (ratio > 1), we apply a cross-kernel smooth to handle 
    # coherent settlement expansion. In collapse regimes (ratio < 1), we disable 
    # smoothing to preserve sharp probabilities and avoid bleeding mass into empty cells.
    if settlement_ratio > 1.0:
        alpha = 0.90
        smoothed_pred = pred.copy()
        smoothed_pred[1:-1, 1:-1] = (
            pred[1:-1, 1:-1] * 4 +
            pred[:-2, 1:-1] + pred[2:, 1:-1] +
            pred[1:-1, :-2] + pred[1:-1, 2:]
        ) / 8.0
        
        pred = alpha * pred + (1 - alpha) * smoothed_pred
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 4. Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # 5. Vectorized floor
    floor = 0.005
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # 6. Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

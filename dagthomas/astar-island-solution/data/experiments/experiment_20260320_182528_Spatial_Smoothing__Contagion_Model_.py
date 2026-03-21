# Experiment: Spatial Smoothing (Contagion Model)
# Hypothesis: Apply a 3x3 blur to settlement, port, and ruin channels with 0.85 center weight. This captures multi-source spatial contagion, spreading probability organically where settlements cluster together.
# Timestamp: 2026-03-20T18:25:28.416333+00:00

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
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
import predict

def blur_channel(channel: np.ndarray, center_weight=0.85) -> np.ndarray:
    """Simple 3x3 blur using shift operations."""
    side_weight = (1.0 - center_weight) / 8.0
    padded = np.pad(channel, pad_width=1, mode='edge')
    blurred = (
        padded[1:-1, 1:-1] * center_weight +
        padded[0:-2, 1:-1] * side_weight +  # Top
        padded[2:, 1:-1] * side_weight +    # Bottom
        padded[1:-1, 0:-2] * side_weight +  # Left
        padded[1:-1, 2:] * side_weight +    # Right
        padded[0:-2, 0:-2] * side_weight +  # Top-Left
        padded[0:-2, 2:] * side_weight +    # Top-Right
        padded[2:, 0:-2] * side_weight +    # Bottom-Left
        padded[2:, 2:] * side_weight        # Bottom-Right
    )
    return blurred

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 50.0, "cal_fine_max": 5.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 2.0,
        "cal_base_base": 0.1, "cal_base_divisor": 500.0, "cal_base_max": 1.0,
        "cal_global_weight": 0.05,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_min_count": 5,
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_power_sett": 0.6,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
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

    # Multipliers
    if global_mult.observed.sum() > 0:
        smooth_val = params["mult_smooth"]
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
            
        ratio_powered = np.power(ratio, params["mult_power"])
        for c in (1, 2, 3):
            ratio_powered[c] = np.power(ratio[c], params["mult_power_sett"])
        ratio = ratio_powered

        ratio[0] = np.clip(ratio[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio[4] = np.clip(ratio[4], params["mult_forest_lo"], params["mult_forest_hi"])
        
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # ==== SPATIAL SMOOTHING ====
    blur_weight = 0.85
    pred[:, :, 1] = blur_channel(pred[:, :, 1], center_weight=blur_weight) # Settlement
    pred[:, :, 2] = blur_channel(pred[:, :, 2], center_weight=blur_weight) # Port
    pred[:, :, 3] = blur_channel(pred[:, :, 3], center_weight=blur_weight) # Ruin
    # Renormalize after blur
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # Vectorized floor: set all zeros to 0, nonzeros to max(val, floor)
    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]  
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

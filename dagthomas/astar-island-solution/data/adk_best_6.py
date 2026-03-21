# ADK Research Agent Best Experiment #6
# Score: avg=90.969, improvement=+0.189
# Timestamp: 2026-03-20T17:44:18.536548+00:00

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
from config import NUM_CLASSES
from fast_predict import (
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
    _build_coastal_mask
)
from calibration import build_feature_keys
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()

    params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 50.0, "cal_fine_max": 10.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 5.0,
        "cal_base_base": 0.1, "cal_base_divisor": 500.0, "cal_base_max": 2.0,
        "cal_global_weight": 0.1,
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_strength_fn": "sqrt",
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.0,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
    }

    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, params["fk_min_count"])

    # Base Prior
    pred = cal_priors[idx_grid]

    # Step 2: FK bucket blending
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= params["fk_min_count"]

    pw = params["fk_prior_weight"]
    ms = params["fk_max_strength"]
    strengths = np.minimum(ms, np.sqrt(cnt_grid))

    strengths_3d = strengths[:, :, np.newaxis]
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # Step 3: USE THE CORRECT GLOBAL MULTIPLIERS METHOD!
    if global_mult.observed.sum() > 0:
        ratio = global_mult.get_multipliers(cal.global_probs)
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # Step 4: Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # Step 5: Vectorized floor
    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # Step 6: Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred


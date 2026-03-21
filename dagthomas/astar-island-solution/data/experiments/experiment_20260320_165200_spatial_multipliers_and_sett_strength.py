# Experiment: spatial_multipliers_and_sett_strength
# Hypothesis: Replaces global multipliers with spatial feature-specific multipliers (grouped by Coastal status and Terrain type). Coastal settlements often have distinct survival rates from inland ones, so adapting the multipliers spatially should fix the R5 underprediction on coastal ports/settlements. Also fixes a bug where power=0.6 wasn't applied to volatile classes in the vectorized function, and increases max empirical strength for settlement cells to 15.0 to trust observations more on volatile survival predictions.
# Timestamp: 2026-03-20T16:52:00.141216+00:00

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

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()
    
    # Default parameters, enhanced with our new logic params
    params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 20.0, "cal_fine_max": 10.0,
        "cal_coarse_base": 1.0, "cal_coarse_divisor": 50.0, "cal_coarse_max": 5.0,
        "cal_base_base": 1.0, "cal_base_divisor": 100.0, "cal_base_max": 2.0,
        "cal_global_weight": 1.0,
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_max_strength_sett": 15.0, # Increased max strength for settlements
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.0,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
        "spatial_mult_global_weight": 50.0,
    }

    # 1. Feature keys and lookup tables
    from calibration import build_feature_keys
    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, params["fk_min_count"])

    # 2. Vectorized base prediction
    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= params["fk_min_count"]

    # Blending logic with higher max strength for settlements
    pw = params["fk_prior_weight"]
    ms_grid = np.full((h, w), params["fk_max_strength"])
    sett_mask = (terrain == 1) | (terrain == 2)
    ms_grid[sett_mask] = params["fk_max_strength_sett"]
    
    strengths = np.minimum(ms_grid, np.sqrt(cnt_grid))
    strengths_3d = strengths[:, :, np.newaxis]
    
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # 3. Spatial Multipliers (Grouped by Coastal & Terrain Type)
    if global_mult.observed.sum() > 0:
        num_groups = 6
        obs_g = np.zeros((num_groups, NUM_CLASSES))
        exp_g = np.zeros((num_groups, NUM_CLASSES))
        
        key_to_group = np.zeros(len(unique_keys), dtype=int)
        
        for i, fk in enumerate(unique_keys):
            t_code = fk[0]
            coastal = fk[2]
            
            if t_code in (1, 2):
                t_group = 2
            elif t_code == 4:
                t_group = 1
            else:
                t_group = 0
                
            g = t_group * 2 + (1 if coastal else 0)
            key_to_group[i] = g
            
            # Aggregate counts for the multiplier
            if fk in fk_buckets.totals and fk_buckets.totals[fk] > 0:
                count = fk_buckets.totals[fk]
                obs_sum = fk_buckets.counts[fk]
                obs_g[g] += obs_sum
                exp_g[g] += count * cal_priors[i]
                
        # Blend with global counts to stabilize
        obs_global = global_mult.observed
        exp_global = global_mult.expected
        
        global_weight = params["spatial_mult_global_weight"]
        global_O_rate = obs_global / max(1.0, obs_global.sum())
        global_E_rate = exp_global / max(1.0, exp_global.sum())
        
        smooth = params["mult_smooth"] * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        
        O_blended = obs_g + global_weight * global_O_rate[np.newaxis, :] + smooth
        E_blended = exp_g + global_weight * global_E_rate[np.newaxis, :] + smooth
        
        ratio = O_blended / np.maximum(E_blended, 1e-6)
        
        # Apply powers: 0.6 for settlement/port/ruin (classes 1,2,3), 0.4 for rest
        power = params["mult_power"]
        ratio_pow = np.zeros_like(ratio)
        for c in range(NUM_CLASSES):
            p = 0.6 if c in (1, 2, 3) else power
            ratio_pow[:, c] = np.power(ratio[:, c], p)
            
        # Clipping
        ratio_pow[:, 0] = np.clip(ratio_pow[:, 0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio_pow[:, 5] = np.clip(ratio_pow[:, 5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio_pow[:, c] = np.clip(ratio_pow[:, c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio_pow[:, 4] = np.clip(ratio_pow[:, 4], params["mult_forest_lo"], params["mult_forest_hi"])
        
        # Apply to grid
        group_grid = key_to_group[idx_grid]
        cell_ratios = ratio_pow[group_grid]
        
        pred *= cell_ratios
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 4. Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # 5. Smart Floor (vectorized)
    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    row_sums = np.maximum(dynamic_pred.sum(axis=-1, keepdims=True), 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # 6. Lock static cells
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

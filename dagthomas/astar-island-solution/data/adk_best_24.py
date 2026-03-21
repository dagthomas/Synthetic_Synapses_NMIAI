# ADK Research Agent Best Experiment #24
# Score: avg=91.533, improvement=+0.085
# Timestamp: 2026-03-20T10:25:41.756263+00:00

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
from scipy.ndimage import uniform_filter

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = np.array(state['grid'])
    settlements = state['settlements']
    
    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)
    
    # 2. Calibration
    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0,
        'cal_fine_divisor': 100.0,
        'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5,
        'cal_coarse_divisor': 100.0,
        'cal_coarse_max': 2.0,
        'cal_base_base': 0.1,
        'cal_base_divisor': 100.0,
        'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)
    
    # 3. Empirical
    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)
    
    # 4. Blending
    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        prior = priors[i]
        emp = empiricals[i]
        c = counts[i]
        
        prior_w = 5.0
        emp_w = min(math.sqrt(c), 8.0)
        
        if c >= 5:
            blend = (prior * prior_w + emp * emp_w) / (prior_w + emp_w)
        else:
            blend = prior
            
        lookup[i] = blend
        
    probs = lookup[idx_grid]
    
    # 5. Global Multipliers
    obs = global_mult.observed
    exp = global_mult.expected
    exp = np.maximum(exp, 1e-6)
    ratio = obs / exp
    
    power = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
    mults = np.power(ratio, power)
    
    mults[1] = np.clip(mults[1], 0.15, 2.5)
    mults[2] = np.clip(mults[2], 0.15, 2.5)
    mults[3] = np.clip(mults[3], 0.15, 2.5)
    
    probs = probs * mults
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    # Dynamic Softening using T
    # Base T = 1.0, scales up to 1.08 based on survival ratio
    # If ratio[1] is 0 (R3), T = 1.0 (no softening)
    # If ratio[1] is > 1.0 (R2), T ~ 1.08
    T = 1.0 + 0.08 * min(ratio[1], 1.0)
    probs = np.power(probs, 1.0/T)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    # Spatial smoothing post-processing
    alpha = 0.75
    for k in [1, 2, 3]:
        smoothed = uniform_filter(probs[:, :, k], size=3, mode='reflect')
        probs[:, :, k] = probs[:, :, k] * alpha + smoothed * (1 - alpha)
        
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    # 6. Structural zeros
    coastal_mask = _build_coastal_mask(grid)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0
    
    # 7. Floor nonzero classes
    probs = np.maximum(probs, 0.005)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    # 8. Lock static cells
    probs[grid == 10] = [1, 0, 0, 0, 0, 0]
    probs[grid == 5] = [0, 0, 0, 0, 0, 1]
    
    probs[0, :] = [1, 0, 0, 0, 0, 0]
    probs[-1, :] = [1, 0, 0, 0, 0, 0]
    probs[:, 0] = [1, 0, 0, 0, 0, 0]
    probs[:, -1] = [1, 0, 0, 0, 0, 0]
    
    return probs


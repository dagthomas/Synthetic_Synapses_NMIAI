"""Sigmoid distance-from-settlement prediction function.

Replaces step-function spatial multipliers with sigmoid-decayed multipliers:
- Settlement: smooth boost near settlements, suppression far away (kills 3 errors)
- Empty/Forest: inverse sigmoid suppression near settlements
- Port: coastal Gaussian (unchanged)
- Coastal penalty on settlement class (settlements die 2x more on coast)
"""
import math

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


def experimental_pred_fn(state, global_mult, fk_buckets):
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.3,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)

    # 3. Empirical
    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    obs = global_mult.observed
    exp = np.maximum(global_mult.expected, 1e-6)
    ratio = obs / exp

    # 4. Blending with regime-adaptive empirical trust
    emp_max_weight = np.clip(12.0 - 4.0 * ratio[1], 6.0, 12.0)
    # Coastal/inland power split: coastal settlements die 2× more
    # Coastal: low power (0.40) — empirical already captures high mortality, avoid over-correction
    # Inland: high power (0.75) — global ratio signal is more informative here
    power_sett_coastal = np.array([0.4, 0.40, 0.75, 0.75, 0.4, 0.4])
    power_sett_inland  = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
    power_exp_coastal  = np.array([0.4, 0.40, 0.60, 0.50, 0.4, 0.4])
    power_exp_inland   = np.array([0.4, 0.50, 0.60, 0.50, 0.4, 0.4])

    regime_raw = float(np.clip(ratio[1], 0.3, 1.5))
    sett_prior_boost = 1.30 + 0.17 * (regime_raw - 0.3) / 1.2

    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)

    for i in range(N):
        prior = priors[i].copy()
        emp = empiricals[i]
        c = counts[i]
        dist_bucket = unique_keys[i][1]
        is_coastal = unique_keys[i][2]

        if dist_bucket <= 3:
            prior[1] *= sett_prior_boost
            prior = prior / prior.sum()

        prior_w = 5.0
        emp_w = min(math.sqrt(c), emp_max_weight)

        if c >= 5:
            blend = (prior * prior_w + emp * emp_w) / max(prior_w, emp_w)
        else:
            blend = prior

        if dist_bucket == 0:
            mults = np.power(ratio, power_sett_coastal if is_coastal else power_sett_inland)
        else:
            mults = np.power(ratio, power_exp_coastal if is_coastal else power_exp_inland)

        mults[1] = np.clip(mults[1], 0.08, 2.0)
        mults[2] = np.clip(mults[2], 0.08, 2.0)
        mults[3] = np.clip(mults[3], 0.08, 2.0)

        blend = blend * mults
        blend = blend / blend.sum()
        lookup[i] = blend

    probs = lookup[idx_grid]

    # 5. Sigmoid distance-from-settlement spatial multipliers
    is_sett = (grid == 1) | (grid == 2)
    land_mask = (grid != 10) & (grid != 5)

    if is_sett.any():
        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
    else:
        dist_map = np.full((MAP_H, MAP_W), 99, dtype=np.int32)

    regime = float(np.clip(ratio[1], 0.3, 1.5))
    coastal_mask = _build_coastal_mask(grid)
    dist_float = dist_map.astype(np.float32)

    # Sigmoid parameters
    beta = 0.8    # decay sharpness
    gamma = 3.5   # influence radius (transition center)

    # sigmoid_decay: ~0.94 at dist=0, 0.5 at dist=gamma, ~0.06 at dist=7
    sigmoid_decay = 1.0 / (1.0 + np.exp(beta * (dist_float - gamma)))
    # sigmoid_rise: inverse — ~0.06 at dist=0, 0.5 at dist=gamma, ~0.94 at dist=7
    sigmoid_rise = 1.0 - sigmoid_decay

    # 5a. Settlement class: sigmoid-decayed multiplier
    # Near settlements (clustered): boost up to ~1.4x
    # Far from settlements (isolated): suppress to ~0.85x
    # Fixes: spurious settlements in isolation, Empty/Forest bleed near settlements
    alpha_sett = 0.35 + 0.10 * regime       # 0.38-0.50 boost magnitude
    base_sett = 0.85 + 0.03 * regime        # 0.86-0.90 far-distance suppression
    sett_mult = base_sett + (1.0 + alpha_sett - base_sett) * sigmoid_decay
    # Coastal penalty: settlements on coast die 2x more (0-36% vs 30-64% survival)
    coastal_penalty = 0.92 - 0.03 * regime  # 0.88-0.91 on coast
    sett_mult = np.where(coastal_mask & land_mask, sett_mult * coastal_penalty, sett_mult)
    sett_mult[~land_mask] = 1.0

    # 5b. Empty class: inverse sigmoid — suppress near settlements, recover far
    # Near settlements mass should go to settlement class, not empty
    empty_near = 0.82 + 0.02 * regime       # 0.83-0.85 near-settlement suppression
    empty_mult = empty_near + (1.0 - empty_near) * sigmoid_rise
    empty_mult[~land_mask] = 1.0

    # 5c. Port class: coastal proximity Gaussian (unchanged — already smooth)
    if coastal_mask.any():
        dist_to_coast = distance_transform_cdt(~coastal_mask, metric='taxicab').astype(np.float32)
    else:
        dist_to_coast = np.full((MAP_H, MAP_W), 99, dtype=np.float32)
    port_prior_boost = 1.25  # +25% to address coastal underprediction (was 1.17)
    port_mult = port_prior_boost * (1.0 + 1.5 * np.exp(-dist_to_coast**2 / 4.5))

    # 5d. Forest class: inverse sigmoid suppression near settlements
    # Forest near settlements gets cleared for expansion
    forest_near = 0.83 + 0.02 * regime      # 0.84-0.86 near-settlement suppression
    gamma_forest = gamma * 0.85              # forest clears slightly closer than settlement radius
    sigmoid_rise_forest = 1.0 / (1.0 + np.exp(-beta * (dist_float - gamma_forest)))
    forest_mult = forest_near + (1.0 - forest_near) * sigmoid_rise_forest
    forest_mult[~land_mask] = 1.0

    # Apply all spatial multipliers
    probs[:, :, 1] *= sett_mult
    probs[:, :, 0] *= empty_mult
    probs[:, :, 2] *= port_mult
    probs[:, :, 4] *= forest_mult

    # 5e. Posterior-based settlement density/cluster adjustment
    # Two effects: (1) penalize dense existing clusters (survival drops with density),
    # (2) boost expansion zone where neighbor posteriors support cluster growth.
    from scipy.ndimage import convolve
    sett_post = probs[:, :, 1].copy()
    kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32) / 8.0
    neighbor_sett = convolve(sett_post, kernel_8, mode='constant', cval=0.0)

    if is_sett.any():
        sett_float = is_sett.astype(np.float32)
        kernel_m3 = np.zeros((7, 7), dtype=np.float32)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if abs(dy) + abs(dx) <= 3:
                    kernel_m3[dy + 3, dx + 3] = 1.0
        kernel_m3[3, 3] = 0.0
        nearby_count = convolve(sett_float, kernel_m3, mode='constant', cval=0.0)
        # Existing settlements: sparse survive 47-77%, dense only 11-49%
        dense_penalty = 1.0 / (1.0 + 0.25 * nearby_count)
        # Expansion zone: regime-scaled posterior coherence boost
        exp_alpha = 0.4 * regime  # ~0.12 collapse, ~0.60 thriving
        expansion_boost = 1.0 + exp_alpha * neighbor_sett
        density_mult = np.where(dist_map == 0, dense_penalty,
                       np.where((dist_map >= 1) & (dist_map <= 3), expansion_boost, 1.0))
        density_mult[~land_mask] = 1.0
        probs[:, :, 1] *= density_mult

    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 6. Distance-aware temperature softening
    radius = 1.8 + int(3.0 * min(ratio[1], 1.2))
    T_max = 1.0 + 0.10 * math.sqrt(min(ratio[1], 1.0))
    T_grid = np.ones((MAP_H, MAP_W, 1), dtype=np.float32)
    T_grid[dist_map <= radius] = T_max

    probs = np.power(probs, 1.0 / T_grid)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 7. Selective spatial smoothing (settlement/ruin only, NOT port)
    alpha_smooth = 0.75
    for k in [1, 3]:
        smoothed = uniform_filter(probs[:, :, k], size=5, mode='reflect')
        probs[:, :, k] = probs[:, :, k] * alpha_smooth + smoothed * (1 - alpha_smooth)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 7b. Coastal edge boost for ports — bottom 3 rows / rightmost 3 cols
    edge_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
    edge_mask[-3:, :] = True
    edge_mask[:, -3:] = True
    probs[edge_mask & coastal_mask, 2] *= 1.08

    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 8. Structural zeros
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0

    # 9. Floor nonzero classes at 0.005 (1/200 = min GT granularity)
    probs = np.maximum(probs, 0.005)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 10. Lock static cells + borders
    probs[grid == 10] = [1, 0, 0, 0, 0, 0]
    probs[grid == 5] = [0, 0, 0, 0, 0, 1]
    probs[0, :] = [1, 0, 0, 0, 0, 0]
    probs[-1, :] = [1, 0, 0, 0, 0, 0]
    probs[:, 0] = [1, 0, 0, 0, 0, 0]
    probs[:, -1] = [1, 0, 0, 0, 0, 0]

    return probs

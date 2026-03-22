# FAILED: Compilation failed: '[' was never closed (<string>, line 109)
# Direction: Increase the `settlement` class `clamp_max` from 2.5 to 2.75 to allow the Bayesian prior to better c

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
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
    try:
        p = _load_params()
    except NameError:
        p = {"prior_w": 1.0, "emp_max": 2.0}
        
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
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.0,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
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
    prior_w = p.get("prior_w", 1.0)
    emp_max = p.get("emp_max", 2.0)
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

    # 5. Apply Global Multiplier Ratio
    # Modification: Increase settlement (class 3) clamp_max to 2.75
    clamp_max = np.array([2.5, 2.5, 2.5, 2.75, 2.5, 2.5])
    clamp_min = 0.1
    clamped_ratio = np.clip(ratio, clamp_min, clamp_max)
    pred *= clamped_ratio

    # 6. Spatial Smoothing
    pred = uniform_filter(pred, size=(3, 3, 1))

    # 7. Simulation Blending
    if sim_pred is not None:
        pred = (1.0 - sim_alpha) * pred + sim_alpha * sim_pred

    # 8. Constraints & Masks
    coastal = _build_coastal_mask(grid)
    is_mountain = (grid == 5)

    pred[~coastal, 2] = 0.0
    pred[~is_mountain, 5] = 0.0
    pred[is_mountain, 5] = 1.0
    pred[is_mountain, :5] = 0.0

    if obs_expansion_radius is not None and len(settlements) > 0:
        sett_y = [s[0] for s in settlements]
        sett_x = [s[1] for s in settlements]
        dist_grid = np.ones((MAP_H, MAP_W))
        dist_grid[sett_y, sett_x] = 0
        dist_map = distance_transform_cdt(dist_grid, metric='taxicab')
        mask = dist_map > obs_expansion_radius
        pred[mask, 3] *= 0.1

    # 9. Normalize and Floor
    pred = np.maximum(pred, 0.005)
    
    # Re-apply strict constraints after floor
    pred[~coastal, 2] = 0.0
    pred[~is_m
    return probs
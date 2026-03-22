# Score: avg=85.51
# Direction: Decrease the `settlement` class `clamp_m` from 1.0 to 0.97 to reduce high-entropy over-prediction in
# Error: None

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
    prior_w = 2.0
    emp_max = 5.0
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

    # 5. Global Multipliers
    mults = np.ones(NUM_CLASSES)
    mults[1] = np.clip(ratio[1], 0.2, 0.97)  # Decreased clamp_m from 1.0 to 0.97
    mults[2] = np.clip(ratio[2], 0.5, 2.0)
    mults[3] = np.clip(ratio[3], 0.5, 2.0)
    mults[4] = np.clip(ratio[4], 0.5, 2.0)
    
    pred *= mults

    if sim_pred is not None:
        pred = pred * (1 - sim_alpha) + sim_pred * sim_alpha

    # 6. Constraints & Normalization
    coastal = _build_coastal_mask(grid)
    for r in range(MAP_H):
        for c in range(MAP_W):
            cell_type = terrain_to_class(grid[r, c])
            if cell_type == 5:
                pred[r, c, :] = 0.0
                pred[r, c, 5] = 1.0
            else:
                pred[r, c, 5] = 0.0
                if not coastal[r, c]:
                    pred[r, c, 2] = 0.0
            
            s = 0.0
            for i in range(NUM_CLASSES):
                if pred[r, c, i] > 0:
                    pred[r, c, i] = max(0.005, pred[r, c, i])
                    s += pred[r, c, i]
            
            if s > 0:
                pred[r, c, :] /= s
            else:
                pred[r, c, cell_type] = 1.0
                
    probs = pred
    return probs
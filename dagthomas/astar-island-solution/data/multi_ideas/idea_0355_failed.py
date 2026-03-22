# FAILED: Compilation failed: '(' was never closed (<string>, line 99)
# Direction: Adjust the `settlement` class `clamp_range` from [0.15, 2.5] to [0.12, 2.8] to allow for higher peak

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
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
        est_vigor: Estimated settlement vigor from observations (settlement % on
            dynamic cells). If provided, uses regime-conditional calibration.
    """
    p = {"prior_w": 2.0, "emp_max": 5.0}
    grid = np.array(state['grid'])
    settlements = np.array(state['settlements'])

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
    prior_w = p["prior_w"]
    emp_max = p["emp_max"]
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

    pred *= ratio

    clamp_min = np.array([0.01, 0.12, 0.01, 0.01, 0.01, 0.01])
    clamp_max = np.array([10.0, 2.8, 5.0, 5.0, 5.0, 5.0])
    pred = np.clip(pred, clamp_min, clamp_max)

    if sim_pred is not None:
        pred = pred * (1 - sim_alpha) + sim_pred * sim_alpha

    if obs_expansion_radius is not None and np.any(settlements):
        dist = distance_transform_cdt(settlements == 0, metric='chessboard')
        pred[:, :, 1] = np.where(dist <= obs_expansion_radius, pred[:, :, 1], 0.0)

    is_mountain = (grid == 3)
    pred[:, :, 5] = np.where(is_mountain, pred[:, :, 5], 0.0)
    for c in range(5):
        pred[:, :, c] = np.where(is_mountain, 0.0, pred[:, :, c])

    coastal = _build_coastal_mask(grid)
    pred[:, :, 2] = np.where(coastal, pred[:, :, 2], 0.0)

    pred = np.maximum(pred, 0.005)
    pred[:, :, 5] = np.where(is_mountain, pred[:, :, 5], 0.0)
    for c in range(
    return probs
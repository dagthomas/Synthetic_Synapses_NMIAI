# FAILED: Compilation failed: '(' was never closed (<string>, line 103)
# Direction: Adjust the `settlement` class `clamp_range` upper bound from 2.5 to 2.75 to allow for higher density

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

    # 5. Global Multipliers
    pred *= ratio

    # Clamp ranges - Settlement max adjusted to 2.75
    clamp_mins = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    clamp_maxs = np.array([5.0, 2.75, 5.0, 5.0, 5.0, 5.0])
    rp = raw_priors[idx_grid]
    pred = np.clip(pred, rp * clamp_mins, rp * clamp_maxs)

    # 6. Spatial constraints
    if obs_expansion_radius is not None:
        init_sett = (grid == 1)
        if init_sett.any():
            dist = distance_transform_cdt(~init_sett, metric='chessboard')
            mask = dist > obs_expansion_radius
            pred[mask, 1] = 0.0

    if growth_front_map is not None:
        pred[:, :, 1] *= np.maximum(growth_front_map, 0.1)

    if obs_overlay is not None:
        o_mask, o_class = obs_overlay
        for c in range(
    return probs
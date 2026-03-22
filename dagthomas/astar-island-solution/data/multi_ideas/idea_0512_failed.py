# FAILED: Crashed on round2 seed 0: name 'probs' is not defined
# Direction: Increase the `settlement` class `clamp_max` from 2.5 to 3.0 to allow for higher peak probabilities i

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
    p = {"prior_w": 2.0, "emp_max": 3.0}
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

    # 5. Apply Multipliers
    clamp_max = np.array([2.0, 2.0, 2.0, 3.0, 2.0, 2.0])
    adj_ratio = np.clip(ratio, 0.1, clamp_max)
    pred = pred * adj_ratio
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 6. Apply Simulation if available
    if sim_pred is not None:
        pred = pred * (1 - sim_alpha) + sim_pred * sim_alpha
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 7. Apply obs_expansion_radius logic
    if obs_expansion_radius is not None and settlements:
        sett_mask = np.zeros((40, 40), dtype=bool)
        for r, c in settlements:
            if 0 <= r < 40 and 0 <= c < 40:
                sett_mask[r, c] = True
        dist = distance_transform_cdt(~sett_mask, metric='chessboard')
        beyond_radius = dist > obs_expansion_radius
        pred[beyond_radius, 3] *= 0.1
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 8. Constraints (mountain, port)
    coastal = _build_coastal_mask(grid)
    for r in range(40):
        for c in range(40):
            terr = grid[r, c]
            if terr != 5:
                pred[r, c, 5] = 0.0
            if not coastal[r, c]:
                pred[r, c, 2] = 0.0

    # 9. Floor and Normalize
    pred = np.maximum(pred, 0.005)
    for r in range(40):
        for c in range(40):
            terr = grid[r, c]
    return probs
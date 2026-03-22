# FAILED: Compilation failed: invalid syntax (<string>, line 105)
# Direction: Change the FK blend normalization formula from `(pw + ew)` to `max(pw, ew)` to prevent high-confiden

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
    def _load_params():
        if hasattr(predict, '_load_params'):
            try:
                return predict._load_params()
            except Exception:
                pass
        return {"prior_w": 2.0, "emp_max": 10.0, "ratio_damp": 0.5, "smooth_size": 3, "smooth_w": 0.1}
    
    p = _load_params()
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

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

    if fk_buckets is not None and hasattr(fk_buckets, 'get_empirical'):
        empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)
    else:
        empiricals = np.zeros((len(unique_keys), NUM_CLASSES), dtype=float)
        counts = np.zeros(len(unique_keys), dtype=float)

    if global_mult is not None and hasattr(global_mult, 'observed') and global_mult.observed.sum() > 0:
        obs = global_mult.observed
        exp_arr = global_mult.expected
    else:
        obs = np.ones(NUM_CLASSES)
        exp_arr = np.ones(NUM_CLASSES)
    exp_arr = np.maximum(exp_arr, 1e-6)
    ratio = obs / exp_arr

    prior_w = p.get("prior_w", 2.0)
    emp_max = p.get("emp_max", 10.0)
    if variance_regime == 'EXTREME_BOOM':
        prior_w = max(prior_w - 0.5, 0.5)
        emp_max = emp_max * 1.2

    pred = priors[idx_grid]
    emp_grid = empiricals[idx_grid]
    cnt_grid = counts[idx_grid]
    has_fk = cnt_grid >= 5

    ew = np.minimum(emp_max, np.sqrt(cnt_grid))[:, :, np.newaxis]
    pw = prior_w
    
    norm_weight = np.maximum(pw, ew)
    blended = (pred * pw + emp_grid * ew) / np.maximum(norm_weight, 1e-10)
    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    ratio_damp = p.get("ratio_damp", 0.5)
    ratio_damped = 1.0 + (ratio - 1.0) * ratio_damp
    pred = pred * ratio_damped
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    smooth_size = p.get("smooth_size", 3)
    smooth_w = p.get("smooth_w", 0.1)
    smoothed = uniform_filter(pred, size=(smooth_size, smooth_size, 1))
    pred = pred * (1 - smooth_w) + smoothed * smooth_w

    if sim_pred is not None:
        pred = pred * (1 - sim_alpha) + sim_pred * sim_alpha

    coastal = _build_coastal_mask(grid)
    for r in range(MAP_H):
        for c in range(MAP_W):
            terrain = grid[r, c]
            base_cls = terrain_to_class(terrain)
            
            if base_cls == 5:
                pred[r, c, :] = 0.0
                pred[r, c, 5] = 1.0
            else:
                pred[r, c, 5] = 0.0
                if not coastal[r, c]:
                    pred[r, c, 2] = 0.0
                
                mask = pred[r, c] > 0
                if np.any(mask):
                    pred[r, c, mask] = np.maximum(pred[r, c, mask], 0.005)
                    pred[r, c] /= pred[r, c].sum()

    if obs_overlay is
    return probs
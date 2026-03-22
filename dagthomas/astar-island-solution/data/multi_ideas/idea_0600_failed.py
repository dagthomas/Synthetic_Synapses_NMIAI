# FAILED: Crashed on round2 seed 0: 'function' object is not subscriptable
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
    ew = strengths[:, :, np.newaxis]
    
    norm_factors = np.maximum(prior_w, ew)
    blended = (pred * prior_w + emp_grid * ew) / np.maximum(norm_factors, 1e-10)
    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    pred *= ratio
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    if sim_pred is not None:
        pred = pred * (1 - sim_alpha) + sim_pred * sim_alpha
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    coast_mask = _build_coastal_mask(grid)
    probs = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
    
    for r in range(MAP_H):
        for c in range(MAP_W):
            terrain = grid[r, c]
            cell_probs = pred[r, c].copy()
            
            if terrain == 0:
                cell_probs[:] = 0.0
                cell_probs[5] = 1.0
            else:
                cell_probs[5] = 0.0
                if not coast_mask[r, c]:
                    cell_probs[2] = 0.0
                    
            if cell_probs.sum() == 0:
                cell_probs[terrain_to_class[terrain]] = 1.0
                
            cell_probs /= cell_probs.sum()
            
            for i in range(NUM_CLASSES):
                if cell_probs[i] > 0:
                    cell_probs[i] = max(cell_probs[i], 0.005)
            
            probs[r, c] = cell_probs / cell_probs.sum()

    return probs
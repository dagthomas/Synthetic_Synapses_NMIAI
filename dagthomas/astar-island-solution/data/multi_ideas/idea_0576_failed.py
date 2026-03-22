# FAILED: Compilation failed: expected an indented block after 'if' statement on line 28 (<string>, line 29)
# Direction: Change the FK blend normalization from `(pw + ew)` to `max(pw, ew)` to prevent over-smoothing in hig

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
    
    if '_load_params' in globals():
        p = globals()['_load_params']()
    elif hasattr(predict, '_load_params'):
        p = predict._load_params()
    else:
        p = {"prior_w": 2.0, "emp_max": 5.0, "ratio_w": 1.0, "spatial_w": 0.5}

    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration (regime-conditional if vigor estimate available)
    if est_vigor is not None:
    return probs
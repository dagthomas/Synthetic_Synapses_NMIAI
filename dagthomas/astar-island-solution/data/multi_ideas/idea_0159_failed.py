# FAILED: Crashed on round2 seed 0: name 'probs' is not defined
# Direction: Increase the `settlement` class `coarse_max` calibration weight from 2.0 to 2.35 to better scale the

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
    p = globals().get('_load_params', lambda: {"prior_w": 1.0, "emp_max": 5.0})()
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
        
    coarse_max_arr = [2.0] * NUM_CLASSES
    coarse_max_arr
    return probs
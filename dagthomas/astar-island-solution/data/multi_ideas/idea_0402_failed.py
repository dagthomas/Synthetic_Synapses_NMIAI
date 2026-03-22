# FAILED: Compilation failed: '{' was never closed (<string>, line 33)
# Direction: Adjust the `settlement` class `clamp_range` upper bound from 2.5 to 2.75 to reduce peak probability 

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
    p = {"prior_w": 1.0, "emp_max": 3.0}
    if hasattr(predict, '_load_params'):
        try:
            p = predict._load_params()
        except Exception:
            pass

    grid = np.array(state['grid'])
    settlements = state.get('settlements', [])

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
        if hasattr(predict, 'get_calibration'):
            cal = predict.get_calibration()
        else:
            cal = {}

    cal_params = {
    return probs
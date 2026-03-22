# FAILED: Crashed on round2 seed 0: name 'probs' is not defined
# Direction: Change the FK blend normalization from `(pw + ew)` to `max(pw, ew)` to prevent probability dilution 

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

    if hasattr(predict, '_load_params'):
        p = predict._load_params()
        prior_w = p.get("prior_w", 1.0)
        emp_max = p.get("emp_max", 5.0)
    else:
        prior_w = 1.0
        emp_max = 5.0

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    return probs
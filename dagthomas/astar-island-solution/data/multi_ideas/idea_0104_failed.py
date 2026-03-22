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
    
    def _load_params():
        return {
            "prior_w": 2.0,
            "emp_max": 10.0,
            "ratio_w": 0.5,
            "smooth_sigma": 1.0,
            "floor": 0.005
        }
        
    p = _load_params()
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
        cal = predict.get_calibration()
    return probs
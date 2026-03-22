# FAILED: Crashed on round2 seed 0: name 'idx_grid' is not defined
# Direction: Modify the FK blend normalization from `(pw + ew)` to `max(pw, ew)` to prevent strong empirical sign

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
    
    p = {"prior_w": 2.0, "emp_max": 10.0, "smooth_w": 0.2}
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique
    return probs
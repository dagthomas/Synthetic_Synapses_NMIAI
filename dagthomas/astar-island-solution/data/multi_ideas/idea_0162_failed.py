# FAILED: Compilation failed: unterminated string literal (detected at line 22) (<string>, line 22)
# Direction: Increase the `settlement` class calibration weight `coarse_max` from 2.0 to 2.2 to better leverage t

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
    """
    if '_load_params' in globals():
        p = globals()['_load_params']()
    elif hasattr(predict, '_load_params'):
        p = predict._load_params()
    else:
        p = {
            "prior_w": 1.5,
            "emp_max": 2.0,
            "base_power": 1.0,
            "dist_2
    return probs
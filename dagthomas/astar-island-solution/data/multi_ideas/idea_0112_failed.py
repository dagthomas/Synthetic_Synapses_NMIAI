# FAILED: Compilation failed: '(' was never closed (<string>, line 22)
# Direction: Increase the settlement class upper clamp from 2.5 to 2.8 to allow the Bayesian update higher confid

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
    try:
        p = _load_params()
    except NameError:
        p = {"prior_w": 1.0, "emp_max": 2.5, "base_power": 0.5}
        
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(
    return probs
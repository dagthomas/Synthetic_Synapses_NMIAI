# FAILED: Crashed on round2 seed 0: name 'probs' is not defined
# Direction: Change the `settlement` class clamp range from [0.15, 2.5] to [0.12, 2.8] to better capture the vari

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
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
        est_vigor: Estimated settlement vigor from observations (settlement % on
            dynamic cells). If provided, uses regime-conditional calibration.
    """
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
        cal = predict.get_calibration()
    return probs
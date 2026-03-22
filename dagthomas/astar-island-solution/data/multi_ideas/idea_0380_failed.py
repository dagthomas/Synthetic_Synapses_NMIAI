# FAILED: Crashed on round2 seed 0: experimental_pred_fn() missing 5 required positional arguments: 'p', 'variance_regime', 'ratio', 'grid', and '_build_coastal
# Direction: Adjust the `settlement` class `clamp_range` lower bound from 0.15 to 0.13 to allow the Bayesian upda

def experimental_pred_fn(pred, has_fk, blended, p, variance_regime, ratio, grid, _build_coastal):
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # Apply global ratio
    ratio_w = p.get("ratio_w", 0.5)
    if variance_regime == 'EXTREME_BOOM':
        ratio_w = min(ratio_w * 1.5, 1.0)
    pred = pred * ((1.0 - ratio_w) + ratio_w * ratio)

    # Apply clamping - MODIFICATION: adjusted lower bound from 0.15 to 0.13
    pred[..., 1] = np.clip(pred[..., 1], 0.13, 0.99)

    # Spatial smoothing
    smooth_w = p.get("smooth_w", 0.0)
    if smooth_w > 0:
        smoothed = uniform_filter(pred, size=3)
        pred = pred * (1.0 - smooth_w) + smoothed * smooth_w

    # Floor >= 0.005 for nonzero classes
    pred = np.maximum(pred, 0.005)

    # Constraints & Masks
    is_mountain = (grid == 3) | (grid == 4)
    pred[~is_mountain, 5] = 0.0

    is_coastal = _build_coastal
    return pred
    return probs
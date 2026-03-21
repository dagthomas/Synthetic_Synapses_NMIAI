# FAILED: Crashed on round2 seed 0: unsupported operand type(s) for *: 'dict' and 'FeatureKeyBuckets'
# Direction: Increase the settlement class upper clamp from 2.5 to 3.0. This allows the Bayesian system to assign

def experimental_pred_fn(pred, grid, mult):
    pred = pred * mult
    pred[:, :, 4] = np.clip(pred[:, :, 4], 0.0, 3.0)
    pred = np.maximum(pred, 0.005)
    
    coastal = _build_coastal_mask(grid)
    pred[~coastal, 2] = 0.0
    pred[grid != 5, 5] = 0.0
    
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred
    return probs
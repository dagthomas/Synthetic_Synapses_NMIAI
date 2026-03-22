# FAILED: Crashed on round2 seed 0: experimental_pred_fn() takes 2 positional arguments but 3 were given
# Direction: Increase the `settlement` class `coarse_max` calibration weight from 2.0 to 2.1 to address its 4.5% 

def experimental_pred_fn(pred, NUM_CLASSES):
    sums = pred.sum(axis=-1, keepdims=True)
    pred = np.where(sums > 0, pred / sums, 1.0 / NUM_CLASSES)
    
    # Floor >= 0.005 for nonzero classes
    mask = pred > 0
    pred[mask] = np.maximum(pred[mask], 0.005)
    
    # Re-normalize
    sums = pred.sum(axis=-1, keepdims=True)
    probs = np.where(sums > 0, pred / sums, 1.0 / NUM_CLASSES)
    return probs
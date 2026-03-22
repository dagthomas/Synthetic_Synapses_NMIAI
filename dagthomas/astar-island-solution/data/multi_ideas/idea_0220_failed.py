# FAILED: Crashed on round2 seed 0: experimental_pred_fn() missing 7 required positional arguments: 'pred', 'ratio', 'multi_store', 'NUM_CLASSES', 'MAP_H', 'MAP
# Direction: Change the feature-key (FK) blend normalization from `(pw + ew)` to `max(pw, ew)` to prevent strong 

def experimental_pred_fn(grid, has_fk, blended, pred, ratio, multi_store, NUM_CLASSES, MAP_H, MAP_W, terrain_to_class):
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # 5. Apply Ratio
    pred *= ratio

    # 6. Spatial smoothing
    smoothed = np.zeros_like(pred)
    for c in range(NUM_CLASSES):
        smoothed[:, :, c] = uniform_filter(pred[:, :, c], size=3)
    pred = smoothed

    # 7. Apply multipliers if multi_store is provided
    if multi_store is not None:
        for c in range(NUM_CLASSES):
            pred[:, :, c] *= multi_store.get_multiplier_map(c)

    # 8. Constraints (Mountain, Port)
    terrain_classes = np.zeros((MAP_H, MAP_W), dtype=int)
    for r in range(MAP_H):
        for c in range(MAP_W):
            terrain_classes[r, c] = terrain_to_class(grid[r, c])
            
    is_mountain = (terrain_classes == 1)
    is_port = (terrain_classes == 2)
    
    return pred
    return probs
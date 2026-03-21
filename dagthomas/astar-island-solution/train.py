"""ML training pipeline for Astar Island (Phase 2 — after collecting ground truth).

Usage:
    python train.py --collect           # Collect ground truth from completed rounds
    python train.py --train             # Train model on collected data
    python train.py --evaluate          # Evaluate model on held-out data
"""
import argparse
import json
from pathlib import Path

import numpy as np

from client import AstarIslandClient
from config import CLASS_NAMES, MAP_H, MAP_W, NUM_CLASSES
from utils import apply_floor, initial_grid_to_classes

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def collect_ground_truth(client: AstarIslandClient):
    """Pull ground truth from all completed rounds."""
    gt_dir = DATA_DIR / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    rounds = client.get_my_rounds()
    completed = [r for r in rounds if r["status"] == "completed" and r.get("round_score") is not None]

    print(f"Found {len(completed)} completed rounds with scores")

    for r in completed:
        round_id = r["id"]
        round_dir = gt_dir / round_id
        if round_dir.exists():
            print(f"  Round {r['round_number']}: already collected")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        detail = client.get_round_detail(round_id)

        for seed_idx in range(detail["seeds_count"]):
            try:
                analysis = client.get_analysis(round_id, seed_idx)
                fname = f"seed_{seed_idx}.json"
                with open(round_dir / fname, "w") as f:
                    json.dump(analysis, f)
                print(f"  Round {r['round_number']}, seed {seed_idx}: collected "
                      f"(score: {analysis.get('score', 'N/A')})")
            except Exception as e:
                print(f"  Round {r['round_number']}, seed {seed_idx}: {e}")

    print(f"\nGround truth saved to {gt_dir}")


def build_features(initial_grid: list[list[int]], settlements: list[dict] = None) -> np.ndarray:
    """Build per-cell feature vectors from initial state.

    Returns (H, W, F) feature array.
    """
    classes = initial_grid_to_classes(initial_grid)
    h, w = classes.shape

    features = []

    # One-hot terrain type (6 features)
    one_hot = np.zeros((h, w, NUM_CLASSES))
    for cls in range(NUM_CLASSES):
        one_hot[:, :, cls] = (classes == cls).astype(float)
    features.append(one_hot)

    # Position features (2)
    ys = np.arange(h)[:, None].repeat(w, axis=1) / h
    xs = np.arange(w)[None, :].repeat(h, axis=0) / w
    features.append(np.stack([ys, xs], axis=-1))

    # Distance to edges (4)
    d_top = np.arange(h)[:, None].repeat(w, axis=1).astype(float) / h
    d_bottom = (h - 1 - np.arange(h))[:, None].repeat(w, axis=1).astype(float) / h
    d_left = np.arange(w)[None, :].repeat(h, axis=0).astype(float) / w
    d_right = (w - 1 - np.arange(w))[None, :].repeat(h, axis=0).astype(float) / w
    features.append(np.stack([d_top, d_bottom, d_left, d_right], axis=-1))

    # Distance to center (1)
    cy, cx = h / 2, w / 2
    dist_center = np.sqrt((ys * h - cy) ** 2 + (xs * w - cx) ** 2) / max(h, w)
    features.append(dist_center[:, :, np.newaxis])

    # Settlement proximity features
    settlement_mask = (classes == 1) | (classes == 2)
    # Distance to nearest settlement (BFS-like using distance transform)
    from scipy.ndimage import distance_transform_edt
    if settlement_mask.any():
        dist_settlement = distance_transform_edt(~settlement_mask)
    else:
        dist_settlement = np.full((h, w), max(h, w), dtype=float)
    features.append((dist_settlement / max(h, w))[:, :, np.newaxis])

    # Settlements within radius 3, 5, 10
    for radius in [3, 5, 10]:
        count = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                y_lo, y_hi = max(0, y - radius), min(h, y + radius + 1)
                x_lo, x_hi = max(0, x - radius), min(w, x + radius + 1)
                count[y, x] = settlement_mask[y_lo:y_hi, x_lo:x_hi].sum()
        features.append(count[:, :, np.newaxis] / (2 * radius + 1) ** 2)

    # Ocean distance
    ocean_mask = classes == 0
    if ocean_mask.any():
        dist_ocean = distance_transform_edt(~ocean_mask)
    else:
        dist_ocean = np.full((h, w), max(h, w), dtype=float)
    features.append((dist_ocean / max(h, w))[:, :, np.newaxis])

    # Mountain distance
    mountain_mask = classes == 5
    if mountain_mask.any():
        dist_mountain = distance_transform_edt(~mountain_mask)
    else:
        dist_mountain = np.full((h, w), max(h, w), dtype=float)
    features.append((dist_mountain / max(h, w))[:, :, np.newaxis])

    # Forest distance
    forest_mask = classes == 4
    if forest_mask.any():
        dist_forest = distance_transform_edt(~forest_mask)
    else:
        dist_forest = np.full((h, w), max(h, w), dtype=float)
    features.append((dist_forest / max(h, w))[:, :, np.newaxis])

    # Adjacent forest count (1)
    adj_forest = np.zeros((h, w))
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and classes[ny, nx] == 4:
                    adj_forest[y, x] += 1
    features.append(adj_forest[:, :, np.newaxis] / 8)

    # Coastal flag (adjacent to ocean) (1)
    coastal = np.zeros((h, w))
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and classes[ny, nx] == 0:
                    # Check if it's actually ocean (code 10)
                    if initial_grid[ny][nx] == 10:
                        coastal[y, x] = 1
    features.append(coastal[:, :, np.newaxis])

    return np.concatenate(features, axis=-1)


def load_training_data() -> tuple:
    """Load all ground truth data and build training set.

    Returns:
        X: (N, F) feature array
        y: (N, 6) target probability array
    """
    gt_dir = DATA_DIR / "ground_truth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"No ground truth data at {gt_dir}. Run --collect first.")

    X_all, y_all = [], []

    for round_dir in sorted(gt_dir.iterdir()):
        if not round_dir.is_dir():
            continue

        for seed_file in sorted(round_dir.glob("seed_*.json")):
            with open(seed_file) as f:
                data = json.load(f)

            if not data.get("ground_truth") or not data.get("initial_grid"):
                continue

            initial_grid = data["initial_grid"]
            ground_truth = np.array(data["ground_truth"])  # (H, W, 6)

            features = build_features(initial_grid)  # (H, W, F)
            h, w = features.shape[:2]

            X_all.append(features.reshape(-1, features.shape[-1]))
            y_all.append(ground_truth.reshape(-1, NUM_CLASSES))

    if not X_all:
        raise ValueError("No training data found")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"Training data: {X.shape[0]} cells, {X.shape[1]} features")
    return X, y


def train_gradient_boosting(X: np.ndarray, y: np.ndarray):
    """Train gradient boosting model (per-class regression)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train one regressor per class
    models = []
    for cls_idx in range(NUM_CLASSES):
        print(f"\nTraining class {cls_idx} ({CLASS_NAMES[cls_idx]})...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train[:, cls_idx])

        train_score = model.score(X_train, y_train[:, cls_idx])
        val_score = model.score(X_val, y_val[:, cls_idx])
        print(f"  R² train: {train_score:.4f}, val: {val_score:.4f}")
        models.append(model)

    # Evaluate on validation set
    y_pred = np.column_stack([m.predict(X_val) for m in models])
    y_pred = apply_floor(y_pred)

    # KL divergence
    y_val_safe = np.maximum(y_val, 1e-10)
    y_pred_safe = np.maximum(y_pred, 1e-10)
    kl = np.sum(y_val_safe * np.log(y_val_safe / y_pred_safe), axis=-1)
    print(f"\nValidation mean KL divergence: {kl.mean():.4f}")

    # Entropy-weighted KL
    entropy = -np.sum(y_val_safe * np.log(y_val_safe), axis=-1)
    dynamic_mask = entropy > 0.01
    if dynamic_mask.any():
        weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / entropy[dynamic_mask].sum()
        score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
        print(f"Estimated score: {score:.1f}")

    # Save models
    import pickle
    model_path = MODEL_DIR / "gbm_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to {model_path}")

    return models


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Astar Island")
    parser.add_argument("--collect", action="store_true", help="Collect ground truth data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    args = parser.parse_args()

    if args.collect:
        client = AstarIslandClient()
        collect_ground_truth(client)
    elif args.train:
        X, y = load_training_data()
        train_gradient_boosting(X, y)
    elif args.evaluate:
        print("Evaluation not yet implemented. Use --train for now.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

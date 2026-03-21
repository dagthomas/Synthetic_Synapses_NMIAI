"""Dirichlet Full Matrix post-hoc calibration.

Learns a 6x6 weight matrix W and 6-dim bias b in log-probability space:
    log(p_calibrated) = log(p_raw) @ W^T + b  →  softmax  →  renormalize

42 parameters total (36 matrix + 6 bias), fitted by minimizing cross-entropy
on historical (prediction, ground_truth) dynamic cell pairs.

Usage:
    from dirichlet_cal import DirichletCalibrator
    cal = DirichletCalibrator.load()          # load fitted params
    pred_cal = cal.transform(pred_40x40x6)    # apply calibration
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from config import NUM_CLASSES, MAP_H, MAP_W

PARAMS_FILE = Path(__file__).parent / "data" / "dirichlet_params.json"


class DirichletCalibrator:
    """Full-matrix Dirichlet calibration in log-probability space."""

    def __init__(self):
        self.W = np.eye(NUM_CLASSES, dtype=np.float64)
        self.b = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.fitted = False

    def fit(self, pred_flat: np.ndarray, gt_flat: np.ndarray, max_iter: int = 300):
        """Fit W and b from (N, 6) prediction/ground-truth pairs.

        Only pass dynamic cells (entropy > 0) — static cells add noise.
        """
        log_p = np.log(np.maximum(pred_flat, 1e-30))
        x0 = np.concatenate([np.eye(NUM_CLASSES).flatten(), np.zeros(NUM_CLASSES)])

        def loss(x):
            W = x[:36].reshape(NUM_CLASSES, NUM_CLASSES)
            b = x[36:]
            logits = log_p @ W.T + b
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            probs = np.maximum(probs, 1e-30)
            return -np.sum(gt_flat * np.log(probs)) / len(gt_flat)

        result = minimize(loss, x0, method="L-BFGS-B",
                          options={"maxiter": max_iter, "ftol": 1e-8})
        self.W = result.x[:36].reshape(NUM_CLASSES, NUM_CLASSES)
        self.b = result.x[36:]
        self.fitted = True
        return result

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """Apply calibration to (H, W, 6) or (N, 6) prediction array."""
        shape = pred.shape
        flat = np.log(np.maximum(pred.reshape(-1, NUM_CLASSES), 1e-30))
        logits = flat @ self.W.T + self.b
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.maximum(probs.reshape(shape), 1e-10)

    def save(self, path: Path = None):
        """Save fitted W and b to JSON."""
        path = path or PARAMS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "W_diag": np.diag(self.W).tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path = None) -> "DirichletCalibrator":
        """Load fitted params from JSON. Returns unfitted instance if file missing."""
        path = path or PARAMS_FILE
        cal = cls()
        try:
            if path.exists():
                data = json.loads(path.read_text())
                cal.W = np.array(data["W"], dtype=np.float64)
                cal.b = np.array(data["b"], dtype=np.float64)
                cal.fitted = True
        except Exception:
            pass
        return cal

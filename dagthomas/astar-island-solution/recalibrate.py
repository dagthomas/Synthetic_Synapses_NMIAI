"""Post-hoc recalibration of predictions using historical ground truth.

Implements:
  1. Histogram binning — simple, non-parametric, per-class
  2. Isotonic regression — monotonic, per-class, per-feature-key-group
  3. Dirichlet-style matrix scaling — learns cross-class interactions in log space

All methods use LOO: when calibrating round N, they train on all OTHER rounds.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask, _build_feature_key_index,
    build_calibration_lookup, build_fk_empirical_lookup,
)
from predict_gemini import gemini_predict
from utils import GlobalMultipliers, FeatureKeyBuckets, terrain_to_class
import predict

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
    "round9": "2a341ace-0f57-4309-9b89-e59fe0f09179",
    "round10": "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    "round11": "324fde07-1670-4202-b199-7aa92ecb40ee",
    "round12": "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
}
ROUND_NAMES = list(ROUND_IDS.keys())
BOOM_ROUNDS = {"round6", "round7", "round11"}


def compute_score(gt, pred):
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    if dynamic.any():
        wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
    else:
        wkl = 0.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


def load_round_data():
    """Pre-load all round data for LOO evaluation."""
    data = {}
    for rn in ROUND_NAMES:
        detail = json.loads((DATA_DIR / rn / "round_detail.json").read_text())
        rid = ROUND_IDS[rn]
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

        gm = GlobalMultipliers()
        fk = FeatureKeyBuckets()
        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= 5:
                continue
            vp, g = obs["viewport"], obs["grid"]
            state = detail["initial_states"][sid]
            terrain = np.array(state["grid"], dtype=int)
            fkeys = build_feature_keys(terrain, state["settlements"])
            for row in range(len(g)):
                for col in range(len(g[0]) if g else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(g[row][col])
                        gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        fk.add_observation(fkeys[my][mx], oc)

        seeds = []
        for si in range(5):
            gt = np.array(json.loads(
                (DATA_DIR / rn / f"analysis_seed_{si}.json").read_text()
            )["ground_truth"])
            seeds.append((detail["initial_states"][si], gt))

        data[rn] = {"detail": detail, "gm": gm, "fk": fk, "seeds": seeds}
    return data


def get_dynamic_mask(grid):
    """Get mask of dynamic cells (not ocean, mountain, or border)."""
    terrain = np.array(grid, dtype=int)
    static = (terrain == 10) | (terrain == 5)
    dynamic = ~static
    dynamic[0, :] = False
    dynamic[-1, :] = False
    dynamic[:, 0] = False
    dynamic[:, -1] = False
    return dynamic


# ================================================================
# Method 1: Histogram Binning
# ================================================================

class HistogramBinner:
    """Per-class histogram binning recalibration."""

    def __init__(self, n_bins=20):
        self.n_bins = n_bins
        self.bin_edges = {}  # class -> edges
        self.bin_values = {}  # class -> calibrated values

    def fit(self, pred_flat, gt_flat):
        """Fit from arrays of shape (N, 6)."""
        for c in range(NUM_CLASSES):
            p = pred_flat[:, c]
            g = gt_flat[:, c]
            # Create equal-frequency bins
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(p, percentiles)
            edges = np.unique(edges)  # remove duplicates
            if len(edges) < 3:
                self.bin_edges[c] = np.array([0.0, 1.0])
                self.bin_values[c] = np.array([g.mean()])
                continue

            bin_idx = np.digitize(p, edges[1:-1])
            values = []
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if mask.sum() > 0:
                    values.append(g[mask].mean())
                else:
                    values.append(edges[b])
            self.bin_edges[c] = edges
            self.bin_values[c] = np.array(values)

    def transform(self, pred):
        """Apply calibration to (40, 40, 6) prediction."""
        result = pred.copy()
        for c in range(NUM_CLASSES):
            if c not in self.bin_edges:
                continue
            p = pred[:, :, c].flatten()
            edges = self.bin_edges[c]
            values = self.bin_values[c]
            bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(values) - 1)
            result[:, :, c] = values[bin_idx].reshape(MAP_H, MAP_W)

        # Renormalize
        result = np.maximum(result, 1e-10)
        result /= result.sum(axis=-1, keepdims=True)
        return result


# ================================================================
# Method 2: Isotonic Regression (per-class)
# ================================================================

class IsotonicCalibrator:
    """Per-class isotonic regression."""

    def __init__(self):
        self.calibrators = {}

    def fit(self, pred_flat, gt_flat):
        from sklearn.isotonic import IsotonicRegression
        for c in range(NUM_CLASSES):
            ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            ir.fit(pred_flat[:, c], gt_flat[:, c])
            self.calibrators[c] = ir

    def transform(self, pred):
        result = pred.copy()
        for c in range(NUM_CLASSES):
            if c not in self.calibrators:
                continue
            p = pred[:, :, c].flatten()
            result[:, :, c] = self.calibrators[c].transform(p).reshape(MAP_H, MAP_W)

        result = np.maximum(result, 1e-10)
        result /= result.sum(axis=-1, keepdims=True)
        return result


# ================================================================
# Method 3: Matrix Scaling (Dirichlet-style)
# ================================================================

class MatrixScaler:
    """Learn W matrix + bias in log-probability space.

    log(p_cal) = W @ log(p_raw) + b, then softmax.
    W is diagonal (6 params) + bias (6 params) = 12 params.
    Generalizes temperature scaling (which is W=I/T, b=0).
    """

    def __init__(self, diagonal_only=True):
        self.diagonal_only = diagonal_only
        self.W = np.eye(NUM_CLASSES)
        self.b = np.zeros(NUM_CLASSES)

    def fit(self, pred_flat, gt_flat, max_iter=200):
        """Fit using cross-entropy (= KL + const) minimization."""
        log_pred = np.log(np.maximum(pred_flat, 1e-30))

        if self.diagonal_only:
            # 12 params: 6 diagonal + 6 bias
            x0 = np.concatenate([np.ones(NUM_CLASSES), np.zeros(NUM_CLASSES)])

            def loss(x):
                w = x[:NUM_CLASSES]
                b = x[NUM_CLASSES:]
                logits = log_pred * w[np.newaxis, :] + b[np.newaxis, :]
                # Softmax
                logits -= logits.max(axis=-1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                probs = np.maximum(probs, 1e-30)
                # Cross-entropy
                ce = -np.sum(gt_flat * np.log(probs)) / len(gt_flat)
                return ce

            result = minimize(loss, x0, method="L-BFGS-B",
                              options={"maxiter": max_iter, "ftol": 1e-8})
            self.W = np.diag(result.x[:NUM_CLASSES])
            self.b = result.x[NUM_CLASSES:]
        else:
            # Full 6x6 matrix = 42 params
            x0 = np.concatenate([np.eye(NUM_CLASSES).flatten(), np.zeros(NUM_CLASSES)])

            def loss(x):
                W = x[:36].reshape(NUM_CLASSES, NUM_CLASSES)
                b = x[36:]
                logits = log_pred @ W.T + b[np.newaxis, :]
                logits -= logits.max(axis=-1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                probs = np.maximum(probs, 1e-30)
                ce = -np.sum(gt_flat * np.log(probs)) / len(gt_flat)
                return ce

            result = minimize(loss, x0, method="L-BFGS-B",
                              options={"maxiter": max_iter, "ftol": 1e-8})
            self.W = result.x[:36].reshape(NUM_CLASSES, NUM_CLASSES)
            self.b = result.x[36:]

    def transform(self, pred):
        log_pred = np.log(np.maximum(pred, 1e-30))
        # Apply matrix in log space
        shape = log_pred.shape
        flat = log_pred.reshape(-1, NUM_CLASSES)
        logits = flat @ self.W.T + self.b[np.newaxis, :]
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        result = probs.reshape(shape)
        result = np.maximum(result, 1e-10)
        return result


# ================================================================
# LOO Evaluation
# ================================================================

def evaluate_method(method_name, round_data, calibrator_factory):
    """Evaluate a recalibration method using nested LOO.

    Outer LOO: leave one round out for testing.
    Inner data: all other rounds provide (pred, gt) pairs for fitting calibrator.
    """
    results = {}

    for test_round in ROUND_NAMES:
        # Build calibration model (LOO)
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]
        cal = CalibrationModel()
        for tr in train_rounds:
            cal.add_round(DATA_DIR / tr)
        predict._calibration_model = cal

        rd = round_data[test_round]

        # Collect training data for calibrator from OTHER rounds
        train_preds = []
        train_gts = []
        for tr in ROUND_NAMES:
            if tr == test_round:
                continue
            tr_data = round_data[tr]
            # Build cal for this training round (LOO within training)
            tr_train = [r for r in ROUND_NAMES + ["round1"] if r != tr and r != test_round]
            tr_cal = CalibrationModel()
            for ttr in tr_train:
                tr_cal.add_round(DATA_DIR / ttr)
            predict._calibration_model = tr_cal

            for state, gt in tr_data["seeds"]:
                pred = gemini_predict(state, tr_data["gm"], tr_data["fk"])
                grid = np.array(state["grid"])
                dynamic = get_dynamic_mask(grid)
                train_preds.append(pred[dynamic])
                train_gts.append(gt[dynamic])

        train_preds = np.concatenate(train_preds)
        train_gts = np.concatenate(train_gts)

        # Fit calibrator
        calibrator = calibrator_factory()
        calibrator.fit(train_preds, train_gts)

        # Evaluate on test round
        predict._calibration_model = cal
        scores_raw = []
        scores_cal = []
        for state, gt in rd["seeds"]:
            pred = gemini_predict(state, rd["gm"], rd["fk"])
            pred_cal = calibrator.transform(pred)

            # Re-lock static cells after calibration
            grid = np.array(state["grid"])
            pred_cal[grid == 10] = [1, 0, 0, 0, 0, 0]
            pred_cal[grid == 5] = [0, 0, 0, 0, 0, 1]
            pred_cal[0, :] = [1, 0, 0, 0, 0, 0]
            pred_cal[-1, :] = [1, 0, 0, 0, 0, 0]
            pred_cal[:, 0] = [1, 0, 0, 0, 0, 0]
            pred_cal[:, -1] = [1, 0, 0, 0, 0, 0]

            scores_raw.append(compute_score(gt, pred))
            scores_cal.append(compute_score(gt, pred_cal))

        results[test_round] = {
            "raw": float(np.mean(scores_raw)),
            "calibrated": float(np.mean(scores_cal)),
        }

    return results


def print_results(method_name, results):
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"{'Round':<10} {'Raw':>8} {'Calibrated':>12} {'Delta':>8}")
    print("-" * 42)

    raw_vals = []
    cal_vals = []
    for rn in ROUND_NAMES:
        r = results[rn]
        d = r["calibrated"] - r["raw"]
        tag = " *" if rn in BOOM_ROUNDS else ""
        print(f"{rn:<10} {r['raw']:8.2f} {r['calibrated']:12.2f} {d:+8.2f}{tag}")
        raw_vals.append(r["raw"])
        cal_vals.append(r["calibrated"])

    raw_avg = np.mean(raw_vals)
    cal_avg = np.mean(cal_vals)
    raw_boom = np.mean([results[r]["raw"] for r in BOOM_ROUNDS])
    cal_boom = np.mean([results[r]["calibrated"] for r in BOOM_ROUNDS])
    print("-" * 42)
    print(f"{'AVG':<10} {raw_avg:8.2f} {cal_avg:12.2f} {cal_avg-raw_avg:+8.2f}")
    print(f"{'BOOM':<10} {raw_boom:8.2f} {cal_boom:12.2f} {cal_boom-raw_boom:+8.2f}")


if __name__ == "__main__":
    import sys
    import time

    print("Loading round data...")
    t0 = time.time()
    round_data = load_round_data()
    print(f"Loaded {len(round_data)} rounds in {time.time()-t0:.1f}s\n")

    # Test each method
    methods = {
        "Histogram Binning (20 bins)": lambda: HistogramBinner(n_bins=20),
        "Histogram Binning (50 bins)": lambda: HistogramBinner(n_bins=50),
        "Dirichlet Diagonal": lambda: MatrixScaler(diagonal_only=True),
        "Dirichlet Full Matrix": lambda: MatrixScaler(diagonal_only=False),
    }

    # Check if sklearn available for isotonic
    try:
        from sklearn.isotonic import IsotonicRegression
        methods["Isotonic Regression"] = lambda: IsotonicCalibrator()
    except ImportError:
        print("sklearn not available — skipping isotonic regression\n")

    for name, factory in methods.items():
        print(f"\nTesting {name}...")
        t0 = time.time()
        results = evaluate_method(name, round_data, factory)
        elapsed = time.time() - t0
        print_results(f"{name} ({elapsed:.0f}s)", results)

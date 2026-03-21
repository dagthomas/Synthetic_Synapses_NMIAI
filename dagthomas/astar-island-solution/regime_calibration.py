"""Regime-weighted CalibrationModel.

Instead of equally weighting all historical rounds, weight them by
similarity to the current round's regime (detected from observations).

Regime detection: observed settlement % from FK buckets
  - COLLAPSE: < 5% settlement in observations
  - MODERATE: 5-18% settlement
  - BOOM: > 18% settlement

Each historical round gets a weight based on how similar its regime is:
  - Same regime: weight = 3.0
  - Adjacent regime: weight = 1.0
  - Opposite regime: weight = 0.3
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibration import CalibrationModel, FeatureKey, build_feature_keys, _floor_and_renormalize
from config import NUM_CLASSES, PROB_FLOOR

DATA_DIR = Path(__file__).parent / "data" / "calibration"

# Regime classification for each round (based on GT analysis)
ROUND_REGIMES = {
    "round1": "MODERATE",   # sett_prob=0.161
    "round2": "BOOM",       # sett_prob=0.205
    "round3": "COLLAPSE",   # sett_prob=0.009
    "round4": "MODERATE",   # sett_prob=0.092 (near-collapse but moderate dynamic range)
    "round5": "MODERATE",   # sett_prob=0.146
    "round6": "BOOM",       # sett_prob=0.244
    "round7": "BOOM",       # sett_prob=0.256
}

# Regime similarity weights
REGIME_WEIGHTS = {
    ("COLLAPSE", "COLLAPSE"): 3.0,
    ("COLLAPSE", "MODERATE"): 1.0,
    ("COLLAPSE", "BOOM"): 0.3,
    ("MODERATE", "COLLAPSE"): 1.0,
    ("MODERATE", "MODERATE"): 3.0,
    ("MODERATE", "BOOM"): 1.0,
    ("BOOM", "COLLAPSE"): 0.3,
    ("BOOM", "MODERATE"): 1.0,
    ("BOOM", "BOOM"): 3.0,
}


def detect_regime(observed_sett_pct: float) -> str:
    """Detect regime from observed settlement percentage."""
    if observed_sett_pct < 0.05:
        return "COLLAPSE"
    elif observed_sett_pct < 0.18:
        return "MODERATE"
    else:
        return "BOOM"


class RegimeCalibrationModel:
    """CalibrationModel that weights rounds by regime similarity."""

    def __init__(self):
        self.fine_sums: dict[FeatureKey, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float))
        self.fine_counts: dict[FeatureKey, int] = defaultdict(int)
        self.coarse_sums: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float))
        self.coarse_counts: dict[tuple, int] = defaultdict(int)
        self.base_sums: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float))
        self.base_counts: dict[int, int] = defaultdict(int)
        self.global_sum = np.zeros(NUM_CLASSES, dtype=float)
        self.global_count = 0
        self.global_probs = _floor_and_renormalize(np.ones(NUM_CLASSES, dtype=float))
        self.rounds_loaded = 0
        self.total_cells = 0

    @classmethod
    def build(cls, exclude_round: str, current_regime: str) -> "RegimeCalibrationModel":
        """Build a regime-weighted calibration model.

        Args:
            exclude_round: Round to exclude (leave-one-out)
            current_regime: Detected regime for weighting
        """
        model = cls()
        if not DATA_DIR.exists():
            return model

        for round_dir in sorted(DATA_DIR.iterdir()):
            round_name = round_dir.name
            if not round_dir.is_dir() or not (round_dir / "round_detail.json").exists():
                continue
            if round_name == exclude_round:
                continue

            # Get regime weight
            round_regime = ROUND_REGIMES.get(round_name, "MODERATE")
            weight = REGIME_WEIGHTS.get((current_regime, round_regime), 1.0)

            try:
                detail = json.loads((round_dir / "round_detail.json").read_text())
                for seed_idx in range(detail["seeds_count"]):
                    analysis_path = round_dir / f"analysis_seed_{seed_idx}.json"
                    if not analysis_path.exists():
                        continue
                    analysis = json.loads(analysis_path.read_text())
                    terrain = np.asarray(analysis["initial_grid"], dtype=int)
                    ground_truth = np.asarray(analysis["ground_truth"], dtype=float)
                    settlements = detail["initial_states"][seed_idx]["settlements"]
                    feature_keys = build_feature_keys(terrain, settlements)
                    height, width = terrain.shape

                    for y in range(height):
                        for x in range(width):
                            fine_key = feature_keys[y][x]
                            coarse_key = (fine_key[0], fine_key[1], fine_key[2], fine_key[4])
                            gt_probs = ground_truth[y, x] * weight

                            model.fine_sums[fine_key] += gt_probs
                            model.fine_counts[fine_key] += weight
                            model.coarse_sums[coarse_key] += gt_probs
                            model.coarse_counts[coarse_key] += weight
                            model.base_sums[fine_key[0]] += gt_probs
                            model.base_counts[fine_key[0]] += weight
                            model.global_sum += gt_probs
                            model.global_count += weight
                            model.total_cells += 1

                model.rounds_loaded += 1
            except Exception as e:
                print(f"  Warning: {round_dir}: {e}")
                continue

        if model.global_count > 0:
            model.global_probs = _floor_and_renormalize(
                model.global_sum / model.global_count)
        return model

    def prior_for(self, feature_key: FeatureKey) -> np.ndarray:
        """Same interface as CalibrationModel.prior_for."""
        coarse_key = (feature_key[0], feature_key[1], feature_key[2], feature_key[4])
        vector = np.zeros(NUM_CLASSES, dtype=float)
        total_weight = 0.0

        fine_count = self.fine_counts.get(feature_key, 0)
        if fine_count > 0:
            fine_weight = min(5.0, 1.0 + fine_count / 100.0)
            fine_dist = self.fine_sums[feature_key] / self.fine_sums[feature_key].sum()
            vector += fine_weight * fine_dist
            total_weight += fine_weight

        coarse_count = self.coarse_counts.get(coarse_key, 0)
        if coarse_count > 0:
            coarse_weight = min(2.0, 0.5 + coarse_count / 100.0)
            coarse_dist = self.coarse_sums[coarse_key] / self.coarse_sums[coarse_key].sum()
            vector += coarse_weight * coarse_dist
            total_weight += coarse_weight

        base_count = self.base_counts.get(feature_key[0], 0)
        if base_count > 0:
            base_weight = min(1.0, 0.1 + base_count / 100.0)
            base_dist = self.base_sums[feature_key[0]] / self.base_sums[feature_key[0]].sum()
            vector += base_weight * base_dist
            total_weight += base_weight

        if total_weight == 0.0:
            return _floor_and_renormalize(self.global_probs)

        vector += 0.01 * self.global_probs
        total_weight += 0.01
        return _floor_and_renormalize(vector / total_weight)

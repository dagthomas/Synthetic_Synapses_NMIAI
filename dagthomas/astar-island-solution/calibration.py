"""Calibration model built from historical round ground truth data.

Builds hierarchical feature-key-based priors:
  Fine:   (terrain, dist_bucket, coastal, forest_neighbors, has_port, cluster) → prob[6]
  Coarse: (terrain, dist_bucket, coastal, has_port) → prob[6]
  Base:   (terrain,) → prob[6]
  Global: overall class distribution

Each level is weighted by observation count. Result is a blended prior
far more accurate than hardcoded R1 averages.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import NUM_CLASSES, PROB_FLOOR

# Feature key: (terrain_code, dist_bucket, coastal, forest_neighbors, has_port_flag, cluster_bucket)
# cluster_bucket is always 0 unless use_cluster=True in build_feature_keys
FeatureKey = tuple[int, int, bool, int, int, int]


def build_cluster_density(terrain: np.ndarray, settlements: list[dict],
                          radius: int = 5) -> np.ndarray:
    """Compute per-cell settlement cluster density (settlements within Manhattan radius).

    Returns: (H, W) array of settlement counts within the radius.
    Use this for post-hoc spatial correction without FK fragmentation.
    """
    height, width = terrain.shape
    sett_points = [(int(s["x"]), int(s["y"])) for s in settlements]
    cluster_counts = np.zeros((height, width), dtype=int)
    for sx, sy in sett_points:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) > radius:
                    continue
                cy, cx = sy + dy, sx + dx
                if 0 <= cy < height and 0 <= cx < width:
                    cluster_counts[cy, cx] += 1
    return cluster_counts

DATA_DIR = Path(__file__).parent / "data" / "calibration"


def _dist_bucket(distance: int) -> int:
    """Distance buckets: 0, 1, 2, 3, 4-5, 6-8, 9+.

    Finer granularity at d=4-8 where the biggest prediction errors occur
    (cells at d=4-5 behave very differently from d=8+ but were lumped together).
    """
    if distance <= 0:
        return 0
    if distance <= 3:
        return distance
    if distance <= 5:
        return 4  # d=4-5: nearby expansion zone
    if distance <= 8:
        return 5  # d=6-8: moderate distance
    return 6  # d=9+: far from settlements


def _floor_and_renormalize(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    probs = np.maximum(np.asarray(probs, dtype=float), floor)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return probs


def _cluster_bucket(count: int) -> int:
    """Cluster density bucket: 0 (isolated), 1 (sparse), 2 (dense).

    Coarse bucketing to avoid FK fragmentation while capturing
    the key signal: dense settlement clusters behave differently.
    """
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    return 2


def build_feature_keys(terrain: np.ndarray, settlements: list[dict],
                       use_cluster: bool = False) -> list[list[FeatureKey]]:
    """Build per-cell feature keys for an entire grid.

    Args:
        use_cluster: If True, include settlement cluster density in the FK.
            Adds a 6th element (0=isolated, 1=sparse, 2=dense) based on
            settlement count within Manhattan distance 5.
    """
    height, width = terrain.shape
    sett_points = [(int(s["x"]), int(s["y"])) for s in settlements]
    has_port_lookup = {
        (int(s["x"]), int(s["y"])): 1 if s.get("has_port") else 0
        for s in settlements
    }

    # Pre-compute per-cell settlement cluster density (settlements within r=5)
    cluster_counts = None
    if use_cluster:
        cluster_radius = 5
        cluster_counts = np.zeros((height, width), dtype=int)
        for sx, sy in sett_points:
            for dy in range(-cluster_radius, cluster_radius + 1):
                for dx in range(-cluster_radius, cluster_radius + 1):
                    if abs(dy) + abs(dx) > cluster_radius:
                        continue  # Manhattan distance
                    cy, cx = sy + dy, sx + dx
                    if 0 <= cy < height and 0 <= cx < width:
                        cluster_counts[cy, cx] += 1

    keys: list[list[FeatureKey]] = []
    for y in range(height):
        row: list[FeatureKey] = []
        for x in range(width):
            coastal = False
            forest_neighbors = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if int(terrain[ny, nx]) == 10:
                        coastal = True
                    if int(terrain[ny, nx]) == 4:
                        forest_neighbors += 1

            if sett_points:
                distance = min(abs(x - sx) + abs(y - sy) for sx, sy in sett_points)
            else:
                distance = 99

            has_port_flag = has_port_lookup.get((x, y), -1)
            cluster = _cluster_bucket(cluster_counts[y, x]) if use_cluster else 0
            row.append((
                int(terrain[y, x]),
                _dist_bucket(distance),
                coastal,
                min(forest_neighbors, 3),
                has_port_flag,
                cluster,
            ))
        keys.append(row)
    return keys


class CalibrationModel:
    """Hierarchical prior model trained on historical ground truth."""

    def __init__(self):
        # Fine level: exact feature key
        self.fine_sums: dict[FeatureKey, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float)
        )
        self.fine_counts: dict[FeatureKey, int] = defaultdict(int)

        # Coarse level: drop forest_neighbors
        self.coarse_sums: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float)
        )
        self.coarse_counts: dict[tuple, int] = defaultdict(int)

        # Base level: terrain only
        self.base_sums: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=float)
        )
        self.base_counts: dict[int, int] = defaultdict(int)

        # Global
        self.global_sum = np.zeros(NUM_CLASSES, dtype=float)
        self.global_count = 0
        self.global_probs = _floor_and_renormalize(np.ones(NUM_CLASSES, dtype=float))

        self.rounds_loaded = 0
        self.total_cells = 0

    def add_round(self, round_dir: Path, use_cluster: bool = False,
                  weight: float = 1.0) -> bool:
        """Load one round's ground truth into the model.

        Args:
            weight: Multiplier for this round's contribution. Use for
                regime-conditional calibration (weight rounds by vigor similarity).
        """
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            return False

        try:
            detail = json.loads(detail_path.read_text())
            seeds_count = detail["seeds_count"]

            for seed_idx in range(seeds_count):
                analysis_path = round_dir / f"analysis_seed_{seed_idx}.json"
                if not analysis_path.exists():
                    continue

                analysis = json.loads(analysis_path.read_text())
                terrain = np.asarray(analysis["initial_grid"], dtype=int)
                ground_truth = np.asarray(analysis["ground_truth"], dtype=float)
                settlements = detail["initial_states"][seed_idx]["settlements"]

                feature_keys = build_feature_keys(terrain, settlements, use_cluster=use_cluster)
                height, width = terrain.shape

                for y in range(height):
                    for x in range(width):
                        fine_key = feature_keys[y][x]
                        # Coarse: drop forest_neighbors (idx 3) and cluster (idx 5)
                        coarse_key = (fine_key[0], fine_key[1], fine_key[2], fine_key[4])
                        gt_probs = ground_truth[y, x] * weight

                        self.fine_sums[fine_key] += gt_probs
                        self.fine_counts[fine_key] += weight

                        self.coarse_sums[coarse_key] += gt_probs
                        self.coarse_counts[coarse_key] += weight

                        self.base_sums[fine_key[0]] += gt_probs
                        self.base_counts[fine_key[0]] += weight

                        self.global_sum += gt_probs
                        self.global_count += weight
                        self.total_cells += 1

            if self.global_count > 0:
                self.global_probs = _floor_and_renormalize(
                    self.global_sum / self.global_count
                )
            self.rounds_loaded += 1
            return True
        except Exception as e:
            print(f"  Warning: failed to load {round_dir}: {e}")
            return False

    @staticmethod
    def compute_round_vigor(round_dir: Path) -> float:
        """Compute vigor (mean settlement probability on dynamic cells) from GT."""
        detail = json.loads((round_dir / "round_detail.json").read_text())
        sett_probs = []
        for si in range(detail["seeds_count"]):
            ap = round_dir / f"analysis_seed_{si}.json"
            if not ap.exists():
                continue
            analysis = json.loads(ap.read_text())
            terrain = np.asarray(analysis["initial_grid"], dtype=int)
            gt = np.asarray(analysis["ground_truth"], dtype=float)
            dynamic = (terrain != 10) & (terrain != 5)
            if dynamic.any():
                sett_probs.append(float(gt[dynamic, 1].mean()))
        return float(np.mean(sett_probs)) if sett_probs else 0.0

    @classmethod
    def from_all_rounds(cls) -> "CalibrationModel":
        """Load calibration from all available round data."""
        model = cls()
        if not DATA_DIR.exists():
            print("No calibration data directory found")
            return model

        for round_dir in sorted(DATA_DIR.iterdir()):
            if round_dir.is_dir() and (round_dir / "round_detail.json").exists():
                if model.add_round(round_dir):
                    print(f"  Loaded {round_dir.name}: {model.total_cells} total cells")

        if model.rounds_loaded > 0:
            print(f"  CalibrationModel: {model.rounds_loaded} rounds, "
                  f"{model.total_cells} cells, "
                  f"{len(model.fine_sums)} fine keys, "
                  f"{len(model.coarse_sums)} coarse keys")
        return model

    def prior_for(self, feature_key: FeatureKey) -> np.ndarray:
        """Get calibrated prior for a feature key.

        Blends fine, coarse, base, and global levels weighted by
        observation count at each level.
        """
        coarse_key = (feature_key[0], feature_key[1], feature_key[2], feature_key[4])

        vector = np.zeros(NUM_CLASSES, dtype=float)
        total_weight = 0.0

        # Fine level
        fine_count = self.fine_counts.get(feature_key, 0)
        if fine_count > 0:
            fine_weight = min(4.0, 1.0 + fine_count / 120.0)
            fine_dist = self.fine_sums[feature_key] / self.fine_sums[feature_key].sum()
            vector += fine_weight * fine_dist
            total_weight += fine_weight

        # Coarse level
        coarse_count = self.coarse_counts.get(coarse_key, 0)
        if coarse_count > 0:
            coarse_weight = min(3.0, 0.75 + coarse_count / 200.0)
            coarse_dist = self.coarse_sums[coarse_key] / self.coarse_sums[coarse_key].sum()
            vector += coarse_weight * coarse_dist
            total_weight += coarse_weight

        # Base level
        base_count = self.base_counts.get(feature_key[0], 0)
        if base_count > 0:
            base_weight = min(1.5, 0.5 + base_count / 1000.0)
            base_dist = self.base_sums[feature_key[0]] / self.base_sums[feature_key[0]].sum()
            vector += base_weight * base_dist
            total_weight += base_weight

        if total_weight == 0.0:
            # No data at all — fall back to global
            return _floor_and_renormalize(self.global_probs)

        # Add global as regularizer
        vector += 0.4 * self.global_probs
        total_weight += 0.4

        blended = vector / total_weight
        return _floor_and_renormalize(blended)

    def get_stats(self) -> dict:
        return {
            "rounds_loaded": self.rounds_loaded,
            "total_cells": self.total_cells,
            "fine_keys": len(self.fine_sums),
            "coarse_keys": len(self.coarse_sums),
            "base_keys": len(self.base_sums),
            "global_probs": [round(float(p), 4) for p in self.global_probs],
        }

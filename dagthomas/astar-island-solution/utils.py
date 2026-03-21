import numpy as np
from config import TERRAIN_TO_CLASS, MAP_H, MAP_W, NUM_CLASSES, PROB_FLOOR


def terrain_to_class(code: int) -> int:
    """Map raw terrain code to prediction class index."""
    return TERRAIN_TO_CLASS.get(code, 0)


def apply_floor(prediction: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Clamp minimum probability and renormalize so rows sum to 1.

    Iterates to ensure floor is maintained after normalization.
    """
    prediction = prediction.copy()
    for _ in range(5):
        prediction = np.maximum(prediction, floor)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)
        if prediction.min() >= floor - 1e-10:
            break
    return prediction


def initial_grid_to_classes(grid: list[list[int]]) -> np.ndarray:
    """Convert raw terrain grid to class index grid. Returns (H, W) array."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    result = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            result[y, x] = terrain_to_class(grid[y][x])
    return result


def classify_cells(grid: list[list[int]]) -> dict:
    """Classify cells into static vs dynamic based on terrain type.

    Returns dict with boolean masks (H, W):
      - 'ocean': True for ocean cells
      - 'mountain': True for mountain cells
      - 'static': True for cells unlikely to change (ocean, mountain)
      - 'forest': True for forest cells
      - 'dynamic': True for cells near settlements (within radius 5)
    """
    classes = initial_grid_to_classes(grid)
    h, w = classes.shape

    ocean = classes == 0
    mountain = classes == 5
    forest = classes == 4
    static = ocean | mountain

    # Find settlement/port positions
    settlement_positions = []
    for y in range(h):
        for x in range(w):
            if classes[y, x] in (1, 2):  # Settlement or Port
                settlement_positions.append((y, x))

    # Dynamic = within radius 5 of any settlement
    dynamic = np.zeros((h, w), dtype=bool)
    radius = 5
    for sy, sx in settlement_positions:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if abs(dy) + abs(dx) <= radius:
                        dynamic[ny, nx] = True

    # Don't mark truly static cells as dynamic
    dynamic = dynamic & ~static

    return {
        "ocean": ocean,
        "mountain": mountain,
        "static": static,
        "forest": forest,
        "dynamic": dynamic,
        "settlement_positions": settlement_positions,
    }


def dynamism_heatmap(grid: list[list[int]], settlements: list[dict]) -> np.ndarray:
    """Compute per-cell dynamism score based on settlement proximity.

    Higher score = more likely to change = higher priority for querying.
    Returns (H, W) float array.
    """
    classes = initial_grid_to_classes(grid)
    h, w = classes.shape
    heatmap = np.zeros((h, w), dtype=float)

    # Settlement positions with weights
    for s in settlements:
        sx, sy = s["x"], s["y"]
        weight = 2.0 if s.get("has_port", False) else 1.0

        for dy in range(-10, 11):
            for dx in range(-10, 11):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dist = max(abs(dy), abs(dx))
                    if dist <= 10:
                        heatmap[ny, nx] += weight / (1 + dist)

    # Zero out truly static cells
    heatmap[classes == 0] = 0  # Ocean/Plains/Empty — but plains CAN change
    # Actually, only ocean (code 10) and mountain (code 5) are truly static
    # We need raw codes for this
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 10:  # Ocean
                heatmap[y, x] = 0
            elif grid[y][x] == 5:  # Mountain
                heatmap[y, x] = 0

    return heatmap


def build_obs_overlay(observations: list, terrain: np.ndarray,
                      seed_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Build per-cell empirical distribution from direct observations.

    Data assimilation: observed cells get a zero-variance update that
    overrides model drift.  Unobserved cells keep counts=0.

    Args:
        observations: List of observation dicts (seed_index, viewport, grid).
        terrain: (H, W) initial terrain grid.
        seed_idx: Which seed to extract observations for.

    Returns:
        obs_counts: (H, W, 6) class counts per cell.
        obs_total:  (H, W) total observation count per cell.
    """
    H, W = terrain.shape
    obs_counts = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)
    obs_total = np.zeros((H, W), dtype=np.float32)

    for obs in observations:
        if obs.get("seed_index") != seed_idx:
            continue
        vp, grid = obs["viewport"], obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < H and 0 <= mx < W:
                    cls = terrain_to_class(grid[row][col])
                    obs_counts[my, mx, cls] += 1.0
                    obs_total[my, mx] += 1.0

    return obs_counts, obs_total


def build_sett_survival(observations: list, settlements: list,
                        seed_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-settlement alive/dead evidence from observations.

    For each initial settlement, tracks how often it was observed alive
    vs dead.  This is strong Bayesian evidence for that specific cell.

    Args:
        observations: List of observation dicts.
        settlements: Initial settlement list with x, y positions.
        seed_idx: Which seed to extract.

    Returns:
        alive_counts: (n_sett,) times observed alive.
        dead_counts:  (n_sett,) times observed dead.
        observed:     (n_sett,) bool — was this settlement seen at all?
    """
    n_sett = len(settlements)
    alive_counts = np.zeros(n_sett, dtype=np.float32)
    dead_counts = np.zeros(n_sett, dtype=np.float32)

    # Map settlement positions to indices
    sett_pos = {}  # (y, x) -> index
    for i, s in enumerate(settlements):
        sett_pos[(int(s["y"]), int(s["x"]))] = i

    for obs in observations:
        if obs.get("seed_index") != seed_idx:
            continue
        vp, grid = obs["viewport"], obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                key = (my, mx)
                if key in sett_pos:
                    idx = sett_pos[key]
                    cls = terrain_to_class(grid[row][col])
                    if cls in (1, 2):  # settlement or port = alive
                        alive_counts[idx] += 1.0
                    else:  # empty, ruin, forest = dead
                        dead_counts[idx] += 1.0

    observed = (alive_counts + dead_counts) > 0
    return alive_counts, dead_counts, observed


def build_growth_front_map(observations: list, terrain: np.ndarray) -> np.ndarray:
    """Build 40x40 heat map of settlement youth/activity from observations.

    Young settlements (population < 1.0) mark active expansion fronts.
    Their influence spreads over Manhattan r=3 with distance decay.

    Args:
        observations: List of observation dicts with 'settlements' field.
            Each settlement has 'x', 'y', 'population'.
        terrain: (H, W) terrain grid for bounds.

    Returns:
        (H, W) float array — higher = more active growth front.
    """
    H, W = terrain.shape
    front_map = np.zeros((H, W), dtype=float)

    for obs in observations:
        for s in obs.get("settlements", []):
            pop = s.get("population", 2.0)
            if pop >= 1.0:
                continue  # Only young settlements
            # Youth score: lower pop = more active front
            youth = 1.0 - pop  # 0.0 (pop=1.0) to 1.0 (pop=0.0)
            sy, sx = int(s.get("y", -1)), int(s.get("x", -1))
            if sy < 0 or sx < 0:
                continue
            # Spread influence with Manhattan distance decay
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        d = abs(dy) + abs(dx)
                        if d <= 3:
                            front_map[ny, nx] += youth / (1.0 + d)

    # Normalize to [0, 1]
    mx = front_map.max()
    if mx > 0:
        front_map /= mx
    return front_map


class ObservationAccumulator:
    """Accumulates simulation observations to build empirical distributions."""

    def __init__(self, height: int = MAP_H, width: int = MAP_W, num_classes: int = NUM_CLASSES):
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.counts = np.zeros((height, width, num_classes), dtype=float)
        self.n_obs = np.zeros((height, width), dtype=int)

    def add_observation(self, grid: list[list[int]], viewport: dict):
        """Merge a viewport observation into the accumulator.

        Args:
            grid: The viewport grid (viewport_h x viewport_w) with terrain codes
            viewport: Dict with x, y, w, h defining viewport bounds
        """
        vx, vy = viewport["x"], viewport["y"]
        vh, vw = len(grid), len(grid[0]) if grid else 0

        for row in range(vh):
            for col in range(vw):
                my, mx = vy + row, vx + col
                if 0 <= my < self.height and 0 <= mx < self.width:
                    cls = terrain_to_class(grid[row][col])
                    self.counts[my, mx, cls] += 1
                    self.n_obs[my, mx] += 1

    def get_distribution(self) -> np.ndarray:
        """Get empirical probability distribution (H, W, 6).

        For cells with observations, returns normalized counts.
        For unobserved cells, returns uniform distribution.
        """
        dist = np.full((self.height, self.width, self.num_classes), 1.0 / self.num_classes)
        observed = self.n_obs > 0
        dist[observed] = self.counts[observed] / self.n_obs[observed, np.newaxis]
        return dist

    def get_observation_count(self) -> np.ndarray:
        """Returns (H, W) array of observation counts per cell."""
        return self.n_obs.copy()


class GlobalTransitionMatrix:
    """Pools terrain transitions across ALL seeds to estimate global parameters.

    Since hidden parameters are shared across seeds, observing a settlement
    collapse on seed 3 is evidence for all seeds.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        # Transition counts: initial_class -> observed_class, bucketed by distance
        # distance buckets: 0-1, 2-3, 4-5, 6-7, 8+
        self.dist_buckets = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 999)]
        self.counts = {}  # (initial_class, bucket_idx) -> array of shape (num_classes,)
        self.totals = {}  # (initial_class, bucket_idx) -> int
        # Coastal variants
        self.coastal_counts = {}
        self.coastal_totals = {}

    def _bucket(self, dist: float) -> int:
        for i, (lo, hi) in enumerate(self.dist_buckets):
            if lo <= dist <= hi:
                return i
        return len(self.dist_buckets) - 1

    def add_observation(self, initial_grid: list[list[int]], observed_grid: list[list[int]],
                        viewport: dict, settlements: list[dict]):
        """Add one viewport observation, comparing observed terrain to initial.

        Args:
            initial_grid: Full 40x40 initial terrain grid for this seed
            observed_grid: Viewport-sized observed terrain grid (post-simulation)
            viewport: {x, y, w, h}
            settlements: Initial settlement list for distance computation
        """
        vx, vy = viewport["x"], viewport["y"]
        vh = len(observed_grid)
        vw = len(observed_grid[0]) if vh > 0 else 0

        # Precompute distances to initial settlements
        sett_positions = [(s["y"], s["x"]) for s in settlements if s.get("alive", True)]

        # Precompute ocean adjacency for initial grid
        h_full = len(initial_grid)
        w_full = len(initial_grid[0]) if h_full > 0 else 0

        for row in range(vh):
            for col in range(vw):
                my, mx = vy + row, vx + col
                if my >= h_full or mx >= w_full:
                    continue

                initial_code = initial_grid[my][mx]
                observed_code = observed_grid[row][col]
                initial_cls = terrain_to_class(initial_code)
                observed_cls = terrain_to_class(observed_code)

                # Distance to nearest initial settlement
                min_dist = 999.0
                for sy, sx in sett_positions:
                    d = ((my - sy) ** 2 + (mx - sx) ** 2) ** 0.5
                    min_dist = min(min_dist, d)

                bucket = self._bucket(min_dist)
                key = (initial_cls, bucket)

                if key not in self.counts:
                    self.counts[key] = np.zeros(self.num_classes)
                    self.totals[key] = 0
                self.counts[key][observed_cls] += 1
                self.totals[key] += 1

                # Coastal variant
                is_coastal = False
                if initial_code != 10:  # Not ocean itself
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = my + dy, mx + dx
                        if 0 <= ny < h_full and 0 <= nx < w_full and initial_grid[ny][nx] == 10:
                            is_coastal = True
                            break

                if is_coastal:
                    ckey = (initial_cls, bucket, "coastal")
                    if ckey not in self.coastal_counts:
                        self.coastal_counts[ckey] = np.zeros(self.num_classes)
                        self.coastal_totals[ckey] = 0
                    self.coastal_counts[ckey][observed_cls] += 1
                    self.coastal_totals[ckey] += 1

    def get_transition_prob(self, initial_class: int, distance: float,
                            coastal: bool = False) -> np.ndarray | None:
        """Get pooled transition probability for a given initial class and distance.

        Returns (6,) array or None if insufficient data.
        """
        bucket = self._bucket(distance)

        if coastal:
            ckey = (initial_class, bucket, "coastal")
            if ckey in self.coastal_counts and self.coastal_totals[ckey] >= 5:
                return self.coastal_counts[ckey] / self.coastal_totals[ckey]

        key = (initial_class, bucket)
        if key in self.counts and self.totals[key] >= 5:
            return self.counts[key] / self.totals[key]

        return None

    def get_stats(self) -> dict:
        """Return summary statistics."""
        total_obs = sum(self.totals.values())
        keys_with_data = len([k for k, v in self.totals.items() if v >= 5])
        return {
            "total_observations": total_obs,
            "buckets_with_data": keys_with_data,
            "total_buckets": len(self.counts),
        }


class GlobalMultipliers:
    """Track observed vs expected class distributions to detect regime shifts.

    Computes per-class multipliers: if we observe way fewer settlements than
    our prior expects, the settlement multiplier drops. Works continuously
    instead of binary collapse detection.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.observed = np.zeros(num_classes, dtype=float)
        self.expected = np.zeros(num_classes, dtype=float)

    def add_observation(self, observed_class: int, prior_probs: np.ndarray):
        """Record one cell observation and its prior expectation."""
        self.observed[observed_class] += 1.0
        self.expected += prior_probs

    def get_multipliers(self, global_prior: np.ndarray = None) -> np.ndarray:
        """Compute per-class multiplier from observed/expected ratio.

        Uses smoothing to avoid extreme values with few observations.
        Clamps to safe ranges per class.
        """
        if self.observed.sum() == 0:
            return np.ones(self.num_classes, dtype=float)

        # Smooth with global prior to avoid division by zero
        if global_prior is None:
            global_prior = np.full(self.num_classes, 1.0 / self.num_classes)
        smooth = 5.0 * global_prior

        ratio = (self.observed + smooth) / np.maximum(self.expected + smooth, 1e-6)
        # Per-class dampening: settlement/port need MORE dampening (0.6)
        # to avoid overreacting. Base classes use 0.4.
        base_ratio = np.power(ratio, 0.4)
        base_ratio[1] = np.power(ratio[1], 0.6)  # settlement
        base_ratio[2] = np.power(ratio[2], 0.6)  # port
        base_ratio[3] = np.power(ratio[3], 0.6)  # ruin
        ratio = base_ratio

        # Per-class clamping
        ratio[0] = np.clip(ratio[0], 0.75, 1.25)
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for cls in (1, 2, 3):
            ratio[cls] = np.clip(ratio[cls], 0.15, 2.5)
        ratio[4] = np.clip(ratio[4], 0.5, 1.8)

        return ratio

    def get_summary(self) -> dict:
        """Return human-readable summary."""
        mult = self.get_multipliers()
        class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
        return {
            "total_cells_observed": int(self.observed.sum()),
            "multipliers": {name: round(float(m), 4) for name, m in zip(class_names, mult)},
            "observed_counts": {name: int(c) for name, c in zip(class_names, self.observed)},
            "expected_mass": {name: round(float(e), 1) for name, e in zip(class_names, self.expected)},
        }


class FeatureKeyBuckets:
    """Pool observations by feature key for reliable empirical distributions.

    With 50 queries × 225 cells = 11,250 observations across ~120 feature keys,
    each bucket gets ~94 observations on average — far more reliable than
    per-cell counts (1-2 observations).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.counts = {}   # feature_key -> np.ndarray of shape (num_classes,)
        self.totals = {}   # feature_key -> int

    def add_observation(self, feature_key: tuple, observed_class: int):
        """Record one observation for a feature key."""
        if feature_key not in self.counts:
            self.counts[feature_key] = np.zeros(self.num_classes, dtype=float)
            self.totals[feature_key] = 0
        self.counts[feature_key][observed_class] += 1.0
        self.totals[feature_key] += 1

    def get_empirical(self, feature_key: tuple) -> tuple[np.ndarray | None, int]:
        """Get empirical distribution for a feature key.

        Returns (distribution, count) or (None, 0) if no data.
        """
        if feature_key not in self.counts or self.totals[feature_key] == 0:
            return None, 0
        count = self.totals[feature_key]
        dist = self.counts[feature_key] / count
        return dist, count

    def get_stats(self) -> dict:
        total_obs = sum(self.totals.values())
        keys_with_data = len([k for k, v in self.totals.items() if v > 0])
        avg_count = total_obs / max(keys_with_data, 1)
        return {
            "total_observations": total_obs,
            "keys_with_data": keys_with_data,
            "avg_per_key": round(avg_count, 1),
        }


class MultiSampleStore:
    """Stores per-viewport per-seed raw observations for variance analysis.

    When the API is stochastic, querying the same (seed, viewport) multiple times
    yields different rollouts. This class stores all raw grids for later analysis
    of settlement variance — the key signal for distinguishing R7 (extreme boom)
    from R5 (moderate).
    """

    def __init__(self):
        self.samples = {}  # (seed_idx, vp_x, vp_y) -> list[list[list[int]]]
        self.feature_keys_by_seed = {}  # seed_idx -> list[list[tuple]]

    def set_feature_keys(self, seed_idx: int, fkeys: list):
        """Store pre-computed feature keys for a seed."""
        self.feature_keys_by_seed[seed_idx] = fkeys

    def add_sample(self, seed_idx: int, vp_x: int, vp_y: int, grid: list):
        """Add a raw observation grid for a (seed, viewport) combination."""
        key = (seed_idx, vp_x, vp_y)
        self.samples.setdefault(key, []).append(grid)

    def get_samples(self, seed_idx: int, vp_x: int, vp_y: int) -> list:
        """Get all raw grids for a (seed, viewport)."""
        return self.samples.get((seed_idx, vp_x, vp_y), [])

    def get_num_samples(self, seed_idx: int, vp_x: int, vp_y: int) -> int:
        return len(self.samples.get((seed_idx, vp_x, vp_y), []))

    def get_per_fk_variance(self) -> dict:
        """Compute variance of settlement rate across samples per feature key.

        Returns dict: {feature_key: {mean_sett_pct, variance_sett, max_sett_pct, n_samples}}
        Only includes keys with at least 2 multi-sample viewports.
        """
        from collections import defaultdict
        fk_sett_rates = defaultdict(list)  # fk -> list of per-sample settlement rates

        for (seed_idx, vp_x, vp_y), grids in self.samples.items():
            if len(grids) < 2:
                continue
            fkeys = self.feature_keys_by_seed.get(seed_idx)
            if fkeys is None:
                continue

            # For each sample, compute per-FK settlement counts
            for grid in grids:
                fk_counts_this = defaultdict(lambda: [0, 0])  # fk -> [sett_count, total]
                for row in range(len(grid)):
                    for col in range(len(grid[0]) if grid else 0):
                        my, mx = vp_y + row, vp_x + col
                        if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                            fk = fkeys[my][mx]
                            obs_cls = terrain_to_class(grid[row][col])
                            fk_counts_this[fk][1] += 1
                            if obs_cls in (1, 2):  # settlement or port
                                fk_counts_this[fk][0] += 1

                for fk, (sc, tc) in fk_counts_this.items():
                    if tc > 0:
                        fk_sett_rates[fk].append(sc / tc)

        result = {}
        for fk, rates in fk_sett_rates.items():
            if len(rates) >= 2:
                arr = np.array(rates)
                result[fk] = {
                    'mean_sett_pct': float(arr.mean()),
                    'variance_sett': float(arr.var()),
                    'max_sett_pct': float(arr.max()),
                    'n_samples': len(rates),
                }
        return result

    def get_overall_variance(self) -> dict:
        """Compute aggregate settlement variance across all viewports.

        Returns dict with: avg_variance, max_variance, avg_sett_pct, max_sett_pct, n_viewports
        """
        per_viewport = []  # list of (mean_sett_pct, variance_sett) across all viewports

        for (seed_idx, vp_x, vp_y), grids in self.samples.items():
            if len(grids) < 2:
                continue
            # Per-sample settlement percentage for this viewport
            sett_pcts = []
            for grid in grids:
                total = 0
                sett = 0
                for row in grid:
                    for code in row:
                        cls = terrain_to_class(code)
                        total += 1
                        if cls in (1, 2):
                            sett += 1
                if total > 0:
                    sett_pcts.append(sett / total)

            if len(sett_pcts) >= 2:
                arr = np.array(sett_pcts)
                per_viewport.append({
                    'mean': float(arr.mean()),
                    'var': float(arr.var()),
                    'max': float(arr.max()),
                })

        if not per_viewport:
            return {
                'avg_variance': 0.0, 'max_variance': 0.0,
                'avg_sett_pct': 0.0, 'max_sett_pct': 0.0,
                'n_viewports': 0,
            }

        return {
            'avg_variance': float(np.mean([v['var'] for v in per_viewport])),
            'max_variance': float(max(v['var'] for v in per_viewport)),
            'avg_sett_pct': float(np.mean([v['mean'] for v in per_viewport])),
            'max_sett_pct': float(max(v['max'] for v in per_viewport)),
            'n_viewports': len(per_viewport),
        }

    def get_stats(self) -> dict:
        total_keys = len(self.samples)
        multi = sum(1 for grids in self.samples.values() if len(grids) >= 2)
        total_grids = sum(len(grids) for grids in self.samples.values())
        return {
            'total_viewports': total_keys,
            'multi_sample_viewports': multi,
            'total_grids': total_grids,
        }

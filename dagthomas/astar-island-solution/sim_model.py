"""Parametric transition simulator for Astar Island — v2.

Major improvements over v1:
  - Gaussian distance decay (exp(-(d/scale)^power)) for sharp expansion cutoff
  - Hard max_reach parameter for zero expansion beyond threshold
  - Nearest-distance model (reverted from additive — avoids far-field leakage)
  - Per-settlement survival with food radius, border vulnerability
  - Separate port distance model

Hidden parameters (14 per round):
  base_survival   - logit for overall settlement survival
  expansion_str   - peak expansion probability (at d=0)
  expansion_scale - distance scale for Gaussian decay
  decay_power     - exponent on distance: exp(-(d/scale)^power), >1 = sharper
  max_reach       - hard cutoff: zero expansion beyond this distance
  coastal_mod     - coastal survival logit adjustment
  food_coeff      - food adjacency survival boost
  cluster_pen     - cluster density penalty on survival
  ruin_rate       - dead settlement → ruin probability
  port_factor     - coastal expansion → port probability
  forest_resist   - forest resistance to expansion
  forest_clear    - forest clearing rate near settlements
  forest_reclaim  - empty → forest reclamation rate
  exp_death       - probability expanded settlement dies
"""
import math

import numpy as np

from sim_data import RoundData, load_round

# Parameter specification: name -> (default, lo, hi)
PARAM_SPEC = {
    "base_survival": (-0.5, -6.0, 3.0),
    "expansion_str": (0.35, 0.005, 0.95),
    "expansion_scale": (2.0, 0.5, 8.0),
    "decay_power": (2.0, 1.0, 4.0),
    "max_reach": (5.0, 1.5, 15.0),
    "coastal_mod": (-0.3, -3.0, 1.0),
    "food_coeff": (0.5, 0.0, 3.0),
    "cluster_pen": (-0.3, -2.0, 0.5),
    "cluster_optimal": (2.0, 0.5, 5.0),   # inverted-U peak density
    "cluster_quad": (-0.2, -2.0, 0.0),     # quadratic penalty strength
    "ruin_rate": (0.5, 0.01, 0.99),
    "port_factor": (0.25, 0.01, 1.0),
    "forest_resist": (0.3, 0.0, 0.95),
    "forest_clear": (0.2, 0.0, 0.8),
    "forest_reclaim": (0.05, 0.0, 0.5),
    "exp_death": (0.3, 0.0, 0.9),
}

PARAM_NAMES = list(PARAM_SPEC.keys())
N_PARAMS = len(PARAM_NAMES)


def params_to_vec(params: dict) -> np.ndarray:
    return np.array([params.get(k, PARAM_SPEC[k][0]) for k in PARAM_NAMES])


def vec_to_params(vec: np.ndarray) -> dict:
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_SPEC[name][1], PARAM_SPEC[name][2]
        params[name] = float(np.clip(vec[i], lo, hi))
    return params


def default_params() -> dict:
    return {k: v[0] for k, v in PARAM_SPEC.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class Simulator:
    """Monte Carlo simulator for a single round+seed — v2."""

    def __init__(self, rd: RoundData):
        self.rd = rd
        self.terrain = rd.terrain
        self.H, self.W = rd.terrain.shape
        self.n_sett = len(rd.settlements)

        sf = rd.sett_features
        self.sett_positions = sf["positions"]
        self.sett_coastal = sf["is_coastal"]
        self.sett_food = sf["food_adj"]
        self.sett_cluster = sf["cluster_r3"]
        self.sett_cluster_r5 = sf["cluster_r5"]
        self.sett_has_port = sf["has_port"]
        self.sett_dist_maps = sf["dist_maps"]  # (n_sett, H, W)

        cf = rd.cell_features
        self.cell_coastal = cf["is_coastal"]
        self.cell_is_ocean = cf["is_ocean"]
        self.cell_is_mountain = cf["is_mountain"]
        self.cell_is_forest = cf["is_forest"]
        self.cell_is_settlement = cf["is_settlement"]
        self.cell_buildable = cf["is_buildable"]
        self.cell_min_sett_dist = cf["min_sett_dist"]

        self.expand_mask = self.cell_buildable & ~self.cell_is_settlement

        self.terrain_class = np.zeros((self.H, self.W), dtype=np.int8)
        for code, cls in [(10, 0), (11, 0), (0, 0), (1, 1), (2, 2),
                          (3, 3), (4, 4), (5, 5)]:
            self.terrain_class[self.terrain == code] = cls

        self.sett_y = self.sett_positions[:, 1]
        self.sett_x = self.sett_positions[:, 0]

        # Per-settlement: food within radius 2
        self.sett_food_r2 = np.zeros(self.n_sett, dtype=float)
        for i in range(self.n_sett):
            sx, sy = self.sett_positions[i]
            food = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) > 2:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if self.terrain[ny, nx] in (4, 11):
                            food += 1
            self.sett_food_r2[i] = food

        # Per-settlement border distance (distance to map edge)
        self.sett_border_dist = np.minimum(
            np.minimum(self.sett_positions[:, 0], self.W - 1 - self.sett_positions[:, 0]),
            np.minimum(self.sett_positions[:, 1], self.H - 1 - self.sett_positions[:, 1])
        ).astype(float)

    def compute_survival_probs(self, params: dict) -> np.ndarray:
        """Per-settlement survival probability with rich spatial features."""
        food_norm = self.sett_food_r2 / 12.0
        coastal = self.sett_coastal.astype(float)
        cluster_norm = self.sett_cluster / 3.0

        logits = (params["base_survival"]
                  + params["food_coeff"] * food_norm
                  + params["coastal_mod"] * coastal
                  + params["cluster_pen"] * cluster_norm
                  + params["cluster_quad"] * ((self.sett_cluster - params["cluster_optimal"]) ** 2) / 9.0)
        return sigmoid(logits)

    def _compute_nearest_alive_dist(self, alive: np.ndarray) -> np.ndarray:
        """alive: (bs, n_sett) → (bs, H, W)."""
        bs = alive.shape[0]
        nearest = np.full((bs, self.H, self.W), 999.0, dtype=np.float32)
        for si in range(self.n_sett):
            d = self.sett_dist_maps[si]
            mask = alive[:, si:si + 1, None]
            nearest = np.where(mask & (d[None, :, :] < nearest),
                               d[None, :, :], nearest)
        return nearest

    def _expansion_prob(self, dist: np.ndarray, params: dict) -> np.ndarray:
        """Gaussian-power decay with hard cutoff.

        P(expand | d) = str * exp(-(d/scale)^power) if d <= max_reach, else 0
        """
        exp_str = params["expansion_str"]
        scale = params["expansion_scale"]
        power = params["decay_power"]
        max_reach = params["max_reach"]

        normalized_d = dist / scale
        prob = exp_str * np.exp(-np.power(normalized_d, power))
        prob = np.where(dist <= max_reach, prob, 0.0)
        return prob

    def run(self, params: dict, n_sims: int = 2000, batch_size: int = 200,
            seed: int | None = None) -> np.ndarray:
        """Run Monte Carlo simulations, return (H, W, 6) probability tensor."""
        rng = np.random.default_rng(seed)

        H, W = self.H, self.W
        n_sett = self.n_sett

        surv_probs = self.compute_survival_probs(params)
        port_factor = params["port_factor"]
        forest_resist = params["forest_resist"]
        ruin_rate = params["ruin_rate"]
        forest_clear = params["forest_clear"]
        forest_reclaim = params["forest_reclaim"]
        exp_death = params["exp_death"]

        counts = np.zeros((H, W, 6), dtype=np.float64)
        n_batches = (n_sims + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            bs = min(batch_size, n_sims - batch_idx * batch_size)

            # 1. Sample settlement survival
            alive = rng.random((bs, n_sett)) < surv_probs[None, :]

            # Initialize grids
            grids = np.broadcast_to(self.terrain_class[None, :, :], (bs, H, W)).copy()

            # Settlement outcomes
            ruin_rolls = rng.random((bs, n_sett))
            for si in range(n_sett):
                sy, sx = self.sett_y[si], self.sett_x[si]
                alive_si = alive[:, si]
                grids[alive_si, sy, sx] = 2 if self.sett_has_port[si] else 1
                dead = ~alive_si
                grids[dead & (ruin_rolls[:, si] < ruin_rate), sy, sx] = 3
                grids[dead & (ruin_rolls[:, si] >= ruin_rate), sy, sx] = 0

            # 2. Nearest alive distance
            nearest_dist = self._compute_nearest_alive_dist(alive)

            # 3. Expansion with Gaussian-power decay + hard cutoff
            exp_prob = self._expansion_prob(nearest_dist, params)
            exp_prob[:, self.cell_is_forest] *= (1.0 - forest_resist)
            exp_prob[:, ~self.expand_mask] = 0.0

            expanded = rng.random((bs, H, W)) < exp_prob
            is_port_exp = expanded & self.cell_coastal[None, :, :] & (rng.random((bs, H, W)) < port_factor)
            is_sett_exp = expanded & ~is_port_exp

            grids[is_sett_exp] = 1
            grids[is_port_exp] = 2

            # 4. Expansion death → ruin
            exp_dies = expanded & (rng.random((bs, H, W)) < exp_death)
            exp_ruin = exp_dies & (rng.random((bs, H, W)) < ruin_rate)
            grids[exp_ruin] = 3
            grids[exp_dies & ~exp_ruin] = 0

            # 5. Forest clearing near active settlements (Gaussian decay)
            clear_prob = forest_clear * np.exp(-np.power(nearest_dist / 3.0, 2.0))
            forest_cells = (grids == 4)
            forest_cleared = forest_cells & (rng.random((bs, H, W)) < clear_prob)
            grids[forest_cleared] = 0

            # 6. Forest reclamation on distant empty cells
            empty_cells = (grids == 0)
            reclaim_prob = np.where(
                self.cell_is_forest[None, :, :],
                forest_reclaim * 2.0,
                forest_reclaim * 0.5
            )
            reclaim_prob = reclaim_prob * np.clip(nearest_dist / 5.0, 0.0, 1.0)
            reclaimed = empty_cells & (rng.random((bs, H, W)) < reclaim_prob)
            grids[reclaimed] = 4

            # Accumulate counts
            for cls in range(6):
                counts[:, :, cls] += (grids == cls).sum(axis=0)

        probs = counts / n_sims
        return probs


def compute_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Entropy-weighted KL divergence score (competition formula)."""
    gt_safe = np.maximum(gt, 1e-10)
    pred_safe = np.maximum(pred, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    if not dynamic.any():
        return 100.0
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


if __name__ == "__main__":
    import time

    rd = load_round("round5", 0)
    sim = Simulator(rd)
    params = default_params()

    print("Running Monte Carlo (2000 sims, batch=200)...")
    t0 = time.perf_counter()
    pred = sim.run(params, n_sims=2000, seed=42)
    t1 = time.perf_counter()
    print(f"  Time: {t1 - t0:.3f}s")
    print(f"  Score: {compute_score(rd.ground_truth, pred):.2f}")

"""GPU-accelerated Monte Carlo simulator for Astar Island.

Uses PyTorch CUDA for ~100x speedup over CPU numpy version.
Drop-in replacement for sim_model.Simulator — same interface, GPU backend.

Usage:
    from sim_model_gpu import GPUSimulator
    sim = GPUSimulator(rd, device='cuda')
    pred = sim.run(params, n_sims=10000)  # (H, W, 6) numpy array
"""
import numpy as np
import torch

from sim_data import RoundData
from sim_model import PARAM_SPEC, PARAM_NAMES, compute_score  # re-export score


class GPUSimulator:
    """Monte Carlo simulator on GPU — PyTorch backend."""

    def __init__(self, rd: RoundData, device: str = 'cuda'):
        self.device = torch.device(device)
        self.H, self.W = rd.terrain.shape
        self.n_sett = len(rd.settlements)

        # Terrain data on GPU
        self.terrain = torch.tensor(rd.terrain, dtype=torch.int8, device=self.device)

        # Terrain class mapping: ocean/land/empty=0, sett=1, port=2, ruin=3, forest=4, mountain=5
        tc = np.zeros((self.H, self.W), dtype=np.int8)
        for code, cls in [(10, 0), (11, 0), (0, 0), (1, 1), (2, 2),
                          (3, 3), (4, 4), (5, 5)]:
            tc[rd.terrain == code] = cls
        self.terrain_class = torch.tensor(tc, dtype=torch.int8, device=self.device)

        # Settlement features
        sf = rd.sett_features
        self.sett_coastal = torch.tensor(sf["is_coastal"], dtype=torch.float32, device=self.device)
        self.sett_has_port = torch.tensor(sf["has_port"], dtype=torch.bool, device=self.device)
        self.sett_cluster = torch.tensor(sf["cluster_r3"], dtype=torch.float32, device=self.device)

        # Pre-compute distance maps on GPU: (n_sett, H, W)
        self.dist_maps = torch.tensor(sf["dist_maps"], dtype=torch.float32, device=self.device)

        # Settlement positions
        pos = sf["positions"]
        self.sett_y = torch.tensor(pos[:, 1], dtype=torch.long, device=self.device)
        self.sett_x = torch.tensor(pos[:, 0], dtype=torch.long, device=self.device)

        # Food within radius 2 (pre-computed on CPU, moved to GPU)
        sett_food_r2 = np.zeros(self.n_sett, dtype=np.float32)
        for i in range(self.n_sett):
            sx, sy = int(pos[i, 0]), int(pos[i, 1])
            food = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) > 2:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if rd.terrain[ny, nx] in (4, 11):
                            food += 1
            sett_food_r2[i] = food
        self.sett_food_r2 = torch.tensor(sett_food_r2, dtype=torch.float32, device=self.device)

        # Cell masks
        cf = rd.cell_features
        self.cell_coastal = torch.tensor(cf["is_coastal"], dtype=torch.bool, device=self.device)
        self.cell_is_forest = torch.tensor(cf["is_forest"], dtype=torch.bool, device=self.device)
        self.expand_mask = torch.tensor(
            cf["is_buildable"] & ~cf["is_settlement"],
            dtype=torch.bool, device=self.device
        )

    def run(self, params: dict, n_sims: int = 5000, batch_size: int = 2500,
            seed: int | None = None) -> np.ndarray:
        """Run Monte Carlo simulations on GPU. Returns (H, W, 6) numpy array."""
        if seed is not None:
            torch.manual_seed(seed)

        H, W = self.H, self.W
        n_sett = self.n_sett
        dev = self.device

        # Extract params
        base_survival = params["base_survival"]
        expansion_str = params["expansion_str"]
        expansion_scale = params["expansion_scale"]
        decay_power = params["decay_power"]
        max_reach = params["max_reach"]
        coastal_mod = params["coastal_mod"]
        food_coeff = params["food_coeff"]
        cluster_pen = params["cluster_pen"]
        cluster_optimal = params["cluster_optimal"]
        cluster_quad = params["cluster_quad"]
        ruin_rate = params["ruin_rate"]
        port_factor = params["port_factor"]
        forest_resist = params["forest_resist"]
        forest_clear = params["forest_clear"]
        forest_reclaim = params["forest_reclaim"]
        exp_death = params["exp_death"]

        # Per-settlement survival probability
        food_norm = self.sett_food_r2 / 12.0
        cluster_norm = self.sett_cluster / 3.0
        logits = (base_survival
                  + food_coeff * food_norm
                  + coastal_mod * self.sett_coastal
                  + cluster_pen * cluster_norm
                  + cluster_quad * ((self.sett_cluster - cluster_optimal) ** 2) / 9.0)
        surv_probs = torch.sigmoid(logits)  # (n_sett,)

        # Accumulate counts
        counts = torch.zeros(H, W, 6, dtype=torch.float64, device=dev)
        n_batches = (n_sims + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            bs = min(batch_size, n_sims - batch_idx * batch_size)

            # 1. Sample settlement survival: (bs, n_sett)
            alive = torch.rand(bs, n_sett, device=dev) < surv_probs.unsqueeze(0)

            # Initialize grids from terrain class
            grids = self.terrain_class.unsqueeze(0).expand(bs, -1, -1).clone()  # (bs, H, W)

            # Settlement outcomes
            ruin_rolls = torch.rand(bs, n_sett, device=dev)
            for si in range(n_sett):
                sy, sx = self.sett_y[si], self.sett_x[si]
                alive_si = alive[:, si]
                dead_si = ~alive_si
                cls = 2 if self.sett_has_port[si] else 1
                grids[alive_si, sy, sx] = cls
                is_ruin = dead_si & (ruin_rolls[:, si] < ruin_rate)
                grids[is_ruin, sy, sx] = 3
                grids[dead_si & ~is_ruin, sy, sx] = 0

            # 2. Nearest alive distance: (bs, H, W)
            nearest = torch.full((bs, H, W), 999.0, device=dev)
            # Process settlements in chunks to balance memory vs speed
            chunk = min(n_sett, 10)
            for ci in range(0, n_sett, chunk):
                ce = min(ci + chunk, n_sett)
                # (1, chunk, H, W)
                d_chunk = self.dist_maps[ci:ce].unsqueeze(0)
                # (bs, chunk, 1, 1)
                a_chunk = alive[:, ci:ce].unsqueeze(-1).unsqueeze(-1)
                masked = torch.where(a_chunk, d_chunk, torch.tensor(999.0, device=dev))
                chunk_min = masked.min(dim=1).values  # (bs, H, W)
                nearest = torch.minimum(nearest, chunk_min)

            # 3. Expansion probability: Gaussian-power decay with hard cutoff
            normalized_d = nearest / expansion_scale
            exp_prob = expansion_str * torch.exp(-torch.pow(normalized_d, decay_power))
            exp_prob = torch.where(nearest <= max_reach, exp_prob, torch.zeros_like(exp_prob))
            # Forest resistance
            exp_prob[:, self.cell_is_forest] *= (1.0 - forest_resist)
            # Only expand on buildable, non-settlement cells
            exp_prob[:, ~self.expand_mask] = 0.0

            # Sample expansion
            expanded = torch.rand(bs, H, W, device=dev) < exp_prob
            is_port_exp = expanded & self.cell_coastal.unsqueeze(0) & (torch.rand(bs, H, W, device=dev) < port_factor)
            is_sett_exp = expanded & ~is_port_exp
            grids[is_sett_exp] = 1
            grids[is_port_exp] = 2

            # 4. Expansion death -> ruin
            exp_dies = expanded & (torch.rand(bs, H, W, device=dev) < exp_death)
            exp_ruin = exp_dies & (torch.rand(bs, H, W, device=dev) < ruin_rate)
            grids[exp_ruin] = 3
            grids[exp_dies & ~exp_ruin] = 0

            # 5. Forest clearing near active settlements
            clear_prob = forest_clear * torch.exp(-torch.pow(nearest / 3.0, 2.0))
            forest_cells = (grids == 4)
            forest_cleared = forest_cells & (torch.rand(bs, H, W, device=dev) < clear_prob)
            grids[forest_cleared] = 0

            # 6. Forest reclamation on distant empty cells
            empty_cells = (grids == 0)
            reclaim_base = torch.where(
                self.cell_is_forest.unsqueeze(0).expand(bs, -1, -1),
                torch.tensor(forest_reclaim * 2.0, device=dev),
                torch.tensor(forest_reclaim * 0.5, device=dev),
            )
            reclaim_p = reclaim_base * torch.clamp(nearest / 5.0, 0.0, 1.0)
            reclaimed = empty_cells & (torch.rand(bs, H, W, device=dev) < reclaim_p)
            grids[reclaimed] = 4

            # Accumulate counts
            for cls in range(6):
                counts[:, :, cls] += (grids == cls).sum(dim=0).double()

        probs = (counts / n_sims).cpu().numpy()
        return probs


def fit_cma_gpu(rd: RoundData, target: np.ndarray, n_sims: int = 1000,
                max_evals: int = 300, sigma: float = 0.5,
                warm_start: dict = None, verbose: bool = False) -> tuple[dict, float]:
    """CMA-ES fitting on GPU. Fits simulator params to match target distribution.

    Args:
        rd: Round data (terrain, settlements, etc.)
        target: (H, W, 6) ground truth or observation-derived target
        n_sims: Monte Carlo samples per evaluation
        max_evals: CMA-ES budget
        sigma: Initial step size
        warm_start: Starting parameter dict (default: moderate regime)
        verbose: Print progress

    Returns:
        (best_params, best_score)
    """
    import cma

    sim = GPUSimulator(rd, device='cuda')

    if warm_start is None:
        warm_start = {k: v[0] for k, v in PARAM_SPEC.items()}

    from sim_model import params_to_vec, vec_to_params
    x0 = params_to_vec(warm_start)
    lo = np.array([PARAM_SPEC[k][1] for k in PARAM_NAMES])
    hi = np.array([PARAM_SPEC[k][2] for k in PARAM_NAMES])

    # Normalize to [0, 1] for CMA-ES
    x0_norm = (x0 - lo) / (hi - lo)

    best_score = -1e9
    best_params = None

    def objective(x_norm):
        nonlocal best_score, best_params
        x = lo + x_norm * (hi - lo)
        x = np.clip(x, lo, hi)
        params = vec_to_params(x)
        pred = sim.run(params, n_sims=n_sims, seed=42)
        score = compute_score(target, pred)
        if score > best_score:
            best_score = score
            best_params = params
        return -score  # CMA-ES minimizes

    opts = {
        'maxfevals': max_evals,
        'bounds': [np.zeros_like(lo), np.ones_like(lo)],
        'verbose': -9,
        'seed': 42,
    }
    es = cma.CMAEvolutionStrategy(x0_norm, sigma, opts)

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        gen += 1
        if verbose and gen % 10 == 0:
            print(f"  Gen {gen}: best={best_score:.2f}")

    if verbose:
        print(f"  CMA-ES done: {es.result.evaluations} evals, best={best_score:.2f}")

    return best_params, best_score

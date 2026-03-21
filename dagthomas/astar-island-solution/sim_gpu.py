"""GPU-accelerated Monte Carlo simulator using PyTorch.

Uses PyTorch tensors on CUDA for massive parallelism on RTX 5090.
CuPy doesn't support Blackwell (sm_120) yet, but PyTorch does.

With 34 GB VRAM, can run 100,000+ parallel simulations.

Usage:
    python sim_gpu.py                  # Benchmark GPU vs CPU
    python sim_gpu.py --sims 50000     # Run 50k simulations
"""
import math
import time

import numpy as np
import torch

from sim_data import RoundData, load_round
from sim_model import (
    PARAM_SPEC, PARAM_NAMES, params_to_vec, vec_to_params, default_params,
    compute_score, Simulator,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUSimulator:
    """PyTorch GPU-accelerated Monte Carlo simulator.

    Processes entire batch in one shot on GPU.
    Memory per sim: 40*40*1 byte = 1.6 KB. 100k sims = 160 MB.
    """

    def __init__(self, rd: RoundData, device=None):
        self.device = device or DEVICE
        self.H, self.W = rd.terrain.shape
        self.n_sett = len(rd.settlements)

        sf = rd.sett_features
        self.sett_dist_maps = torch.as_tensor(
            sf["dist_maps"], dtype=torch.float32, device=self.device)  # (n_sett, H, W)
        self.sett_coastal = torch.as_tensor(sf["is_coastal"], device=self.device)
        self.sett_cluster = torch.as_tensor(
            sf["cluster_r3"], dtype=torch.float32, device=self.device)
        self.sett_has_port = torch.as_tensor(sf["has_port"], device=self.device)

        # Food within radius 2 (computed on CPU, transferred)
        terrain = rd.terrain
        positions = sf["positions"]
        food_r2 = np.zeros(self.n_sett, dtype=np.float32)
        for i in range(self.n_sett):
            sx, sy = positions[i]
            food = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) > 2:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if terrain[ny, nx] in (4, 11):
                            food += 1
            food_r2[i] = food
        self.sett_food_r2 = torch.as_tensor(food_r2, device=self.device)

        cf = rd.cell_features
        self.cell_coastal = torch.as_tensor(cf["is_coastal"], device=self.device)
        self.cell_is_forest = torch.as_tensor(cf["is_forest"], device=self.device)
        self.expand_mask = torch.as_tensor(
            cf["is_buildable"] & ~cf["is_settlement"], device=self.device)

        # Initial class grid
        terrain_class = np.zeros((self.H, self.W), dtype=np.int8)
        for code, cls in [(10, 0), (11, 0), (0, 0), (1, 1), (2, 2),
                          (3, 3), (4, 4), (5, 5)]:
            terrain_class[terrain == code] = cls
        self.terrain_class = torch.as_tensor(
            terrain_class, dtype=torch.int8, device=self.device)

        self.sett_y = torch.as_tensor(positions[:, 1], dtype=torch.long, device=self.device)
        self.sett_x = torch.as_tensor(positions[:, 0], dtype=torch.long, device=self.device)

    def compute_survival_probs(self, params: dict) -> torch.Tensor:
        food_norm = self.sett_food_r2 / 12.0
        coastal = self.sett_coastal.float()
        cluster_norm = self.sett_cluster / 3.0
        cluster_optimal = params.get("cluster_optimal", 2.0)
        cluster_quad = params.get("cluster_quad", -0.2)

        logits = (params["base_survival"]
                  + params["food_coeff"] * food_norm
                  + params["coastal_mod"] * coastal
                  + params["cluster_pen"] * cluster_norm
                  + cluster_quad * ((self.sett_cluster - cluster_optimal) ** 2) / 9.0)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def run(self, params: dict, n_sims: int = 10000,
            seed: int | None = None) -> np.ndarray:
        """Run n_sims on GPU in a single batch. Returns (H, W, 6) numpy array."""
        if seed is not None:
            torch.manual_seed(seed)

        H, W = self.H, self.W
        n_sett = self.n_sett
        dev = self.device

        surv_probs = self.compute_survival_probs(params)
        exp_str = params["expansion_str"]
        scale = params["expansion_scale"]
        power = params["decay_power"]
        max_reach = params["max_reach"]
        port_factor = params["port_factor"]
        forest_resist = params["forest_resist"]
        ruin_rate = params["ruin_rate"]
        forest_clear = params["forest_clear"]
        forest_reclaim = params["forest_reclaim"]
        exp_death = params["exp_death"]

        # 1. Settlement survival: (N, n_sett)
        alive = torch.rand(n_sims, n_sett, device=dev) < surv_probs.unsqueeze(0)

        # Initialize grids: (N, H, W)
        grids = self.terrain_class.unsqueeze(0).expand(n_sims, -1, -1).clone()

        # Settlement outcomes
        ruin_rolls = torch.rand(n_sims, n_sett, device=dev)
        for si in range(n_sett):
            sy = self.sett_y[si]
            sx = self.sett_x[si]
            alive_si = alive[:, si]
            has_port = self.sett_has_port[si].item()
            grids[alive_si, sy, sx] = 2 if has_port else 1
            dead = ~alive_si
            grids[dead & (ruin_rolls[:, si] < ruin_rate), sy, sx] = 3
            grids[dead & (ruin_rolls[:, si] >= ruin_rate), sy, sx] = 0

        # 2. Nearest alive distance: (N, H, W)
        nearest_dist = torch.full((n_sims, H, W), 999.0, device=dev)
        for si in range(n_sett):
            d = self.sett_dist_maps[si]  # (H, W)
            mask = alive[:, si].view(-1, 1, 1)  # (N, 1, 1)
            d_expanded = d.unsqueeze(0)  # (1, H, W)
            nearest_dist = torch.where(
                mask & (d_expanded < nearest_dist),
                d_expanded, nearest_dist)

        # 3. Expansion: Gaussian-power decay + hard cutoff
        normalized_d = nearest_dist / scale
        exp_prob = exp_str * torch.exp(-torch.pow(normalized_d, power))
        exp_prob = torch.where(nearest_dist <= max_reach, exp_prob,
                               torch.zeros_like(exp_prob))
        # Forest resistance
        forest_mask = self.cell_is_forest.unsqueeze(0)  # (1, H, W)
        exp_prob = torch.where(forest_mask, exp_prob * (1.0 - forest_resist), exp_prob)
        # Only expansion candidates
        expand_mask = self.expand_mask.unsqueeze(0)  # (1, H, W)
        exp_prob = torch.where(expand_mask, exp_prob, torch.zeros_like(exp_prob))

        expanded = torch.rand(n_sims, H, W, device=dev) < exp_prob
        coastal = self.cell_coastal.unsqueeze(0)
        is_port_exp = expanded & coastal & (torch.rand(n_sims, H, W, device=dev) < port_factor)
        is_sett_exp = expanded & ~is_port_exp
        grids[is_sett_exp] = 1
        grids[is_port_exp] = 2

        # 3b. Second-wave expansion from newly expanded settlements
        # Re-compute nearest distance including new settlements
        active = (grids == 1) | (grids == 2)  # (N, H, W)
        # Use convolution-like approach: for each cell, check neighbors for active settlements
        # Pad and shift to check Manhattan distance 1-3
        step2_prob = torch.zeros(n_sims, H, W, device=dev)
        step2_str = exp_str * 0.4  # Weaker cascade
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                d = abs(dy) + abs(dx)
                if d == 0 or d > 3:
                    continue
                # Check if there's an active settlement at offset (dy,dx)
                sy_lo, sy_hi = max(0, dy), min(H, H + dy)
                sx_lo, sx_hi = max(0, dx), min(W, W + dx)
                ey_lo, ey_hi = max(0, -dy), min(H, H - dy)
                ex_lo, ex_hi = max(0, -dx), min(W, W - dx)
                shifted_active = torch.zeros_like(active)
                shifted_active[:, sy_lo:sy_hi, sx_lo:sx_hi] = \
                    active[:, ey_lo:ey_hi, ex_lo:ex_hi]
                # Only count new settlements (expanded in this step)
                is_new = shifted_active & expanded[:, ey_lo:ey_hi, ex_lo:ex_hi].new_zeros(
                    n_sims, H, W).bool()
                # Simpler: just use shifted expanded mask
                shifted_exp = torch.zeros_like(expanded)
                shifted_exp[:, sy_lo:sy_hi, sx_lo:sx_hi] = \
                    expanded[:, ey_lo:ey_hi, ex_lo:ex_hi]
                p = step2_str * math.exp(-(d / scale) ** power)
                step2_prob += shifted_exp.float() * p

        step2_prob = torch.clamp(step2_prob, 0.0, 0.5)
        step2_prob = torch.where(forest_mask, step2_prob * (1.0 - forest_resist), step2_prob)
        # Only expand into unoccupied expansion candidates
        occupied = (grids != 0) & (grids != 4)
        step2_prob[occupied | ~expand_mask.expand_as(step2_prob)] = 0.0

        expanded_2 = torch.rand(n_sims, H, W, device=dev) < step2_prob
        is_port_2 = expanded_2 & coastal & (torch.rand(n_sims, H, W, device=dev) < port_factor)
        is_sett_2 = expanded_2 & ~is_port_2
        grids[is_sett_2] = 1
        grids[is_port_2] = 2

        all_expanded = expanded | expanded_2

        # 4. Expansion death
        exp_dies = all_expanded & (torch.rand(n_sims, H, W, device=dev) < exp_death)
        exp_ruin = exp_dies & (torch.rand(n_sims, H, W, device=dev) < ruin_rate)
        grids[exp_ruin] = 3
        grids[exp_dies & ~exp_ruin] = 0

        # 5. Forest clearing
        clear_prob = forest_clear * torch.exp(-torch.pow(nearest_dist / 3.0, 2.0))
        forest_cells = (grids == 4)
        forest_cleared = forest_cells & (torch.rand(n_sims, H, W, device=dev) < clear_prob)
        grids[forest_cleared] = 0

        # 6. Forest reclamation
        empty_cells = (grids == 0)
        reclaim_base = torch.where(
            forest_mask,
            torch.tensor(forest_reclaim * 2.0, device=dev),
            torch.tensor(forest_reclaim * 0.5, device=dev))
        reclaim_prob = reclaim_base * torch.clamp(nearest_dist / 5.0, 0.0, 1.0)
        reclaimed = empty_cells & (torch.rand(n_sims, H, W, device=dev) < reclaim_prob)
        grids[reclaimed] = 4

        # Accumulate class counts
        counts = torch.zeros(H, W, 6, dtype=torch.float64, device=dev)
        for cls in range(6):
            counts[:, :, cls] = (grids == cls).sum(dim=0).double()

        probs = (counts / n_sims).cpu().numpy()
        return probs


def benchmark():
    """Compare CPU vs GPU speed at various simulation counts."""
    rd = load_round("round5", 0)
    params = default_params()

    # CPU baseline
    cpu_sim = Simulator(rd)
    print("CPU (NumPy):")
    for n in [2000, 5000, 10000]:
        t0 = time.perf_counter()
        pred_cpu = cpu_sim.run(params, n_sims=n, seed=42)
        t1 = time.perf_counter()
        score = compute_score(rd.ground_truth, pred_cpu)
        print(f"  {n:6d} sims: {t1 - t0:.3f}s  score={score:.2f}")

    if not torch.cuda.is_available():
        print("\nCUDA not available")
        return

    # GPU
    gpu_sim = GPUSimulator(rd)
    print(f"\nGPU (PyTorch, {torch.cuda.get_device_name(0)}):")

    # Warm up
    _ = gpu_sim.run(params, n_sims=100, seed=42)
    torch.cuda.synchronize()

    for n in [2000, 5000, 10000, 50000, 100000, 200000]:
        try:
            t0 = time.perf_counter()
            pred_gpu = gpu_sim.run(params, n_sims=n, seed=42)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            score = compute_score(rd.ground_truth, pred_gpu)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {n:6d} sims: {t1 - t0:.3f}s  score={score:.2f}  VRAM={mem:.1f}GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  {n:6d} sims: OOM")
            break

    # Verify similarity
    pred_cpu = cpu_sim.run(params, n_sims=10000, seed=42)
    pred_gpu = gpu_sim.run(params, n_sims=10000, seed=42)
    diff = np.abs(pred_cpu - pred_gpu).mean()
    print(f"\nCPU vs GPU mean abs diff: {diff:.6f} (expected ~0.01, different RNG)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=None)
    args = parser.parse_args()

    if args.sims:
        rd = load_round("round5", 0)
        gpu_sim = GPUSimulator(rd)
        params = default_params()
        t0 = time.perf_counter()
        pred = gpu_sim.run(params, n_sims=args.sims, seed=42)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        score = compute_score(rd.ground_truth, pred)
        print(f"{args.sims} sims: {t1 - t0:.3f}s  score={score:.2f}")
    else:
        benchmark()

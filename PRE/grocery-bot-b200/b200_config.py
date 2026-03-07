"""GPU detection and B200-scaled solver parameters."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional


def detect_gpu() -> str:
    """Return GPU profile: 'b200', '5090', or 'generic'."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 'generic'
        props = torch.cuda.get_device_properties(0)
        name = props.name.lower()
        vram_gb = props.total_memory / (1024 ** 3)

        if 'b200' in name or vram_gb > 150:
            return 'b200'
        elif '5090' in name or (60 > vram_gb > 20):
            return '5090'
        else:
            return 'generic'
    except Exception:
        return 'generic'


def get_vram_gb() -> float:
    """Return total VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


@dataclass
class B200Params:
    """Solver parameters scaled to GPU capability."""
    # Single-bot DP
    max_states: int = 500_000
    chunk_size: int = 200_000
    # Joint DP
    joint_squad_size: int = 2
    joint_states: int = 100_000
    # Pass-1 orderings
    pass1_orderings: int = 3
    # Refinement
    refine_iters: int = 10
    lns_rounds: int = 5
    # Expert-specific
    max_dp_bots: int = 7
    # Speed bonus
    speed_bonus: float = 100.0
    speed_decay: float = 0.5
    # Parallel orderings (how many orderings to fit in VRAM at once)
    parallel_orderings: int = 1
    # Explore states (for pass-1 screening)
    explore_states: int = 50_000


# Parameter tables per difficulty per GPU
_PARAMS = {
    'b200': {
        'easy': B200Params(
            max_states=50_000_000, chunk_size=2_000_000,
            joint_squad_size=1, joint_states=0,
            pass1_orderings=1, refine_iters=0, lns_rounds=0,
            max_dp_bots=1, parallel_orderings=1, explore_states=50_000_000,
        ),
        'medium': B200Params(
            max_states=2_000_000, chunk_size=1_000_000,
            joint_squad_size=3, joint_states=10_000_000,
            pass1_orderings=6, refine_iters=50, lns_rounds=20,
            max_dp_bots=3, parallel_orderings=4, explore_states=200_000,
        ),
        'hard': B200Params(
            max_states=500_000, chunk_size=500_000,
            joint_squad_size=3, joint_states=5_000_000,
            pass1_orderings=120, refine_iters=100, lns_rounds=40,
            max_dp_bots=5, parallel_orderings=4, explore_states=100_000,
        ),
        'expert': B200Params(
            max_states=200_000, chunk_size=200_000,
            joint_squad_size=4, joint_states=2_000_000,
            pass1_orderings=200, refine_iters=100, lns_rounds=40,
            max_dp_bots=10, parallel_orderings=4, explore_states=50_000,
        ),
        'nightmare': B200Params(
            # Nightmare uses NightmareTrainer (V3 + perturbation search),
            # not GPU DP. These params control the trainer behavior.
            max_states=0, chunk_size=0,  # unused (no GPU DP)
            joint_squad_size=0, joint_states=0,
            pass1_orderings=0, refine_iters=0, lns_rounds=0,
            max_dp_bots=0, parallel_orderings=1, explore_states=0,
        ),
    },
    '5090': {
        'easy': B200Params(
            max_states=200_000, chunk_size=200_000,
            joint_squad_size=1, joint_states=0,
            pass1_orderings=1, refine_iters=0, lns_rounds=0,
            max_dp_bots=1, explore_states=200_000,
        ),
        'medium': B200Params(
            max_states=100_000, chunk_size=100_000,
            joint_squad_size=2, joint_states=100_000,
            pass1_orderings=3, refine_iters=10, lns_rounds=5,
            max_dp_bots=3, explore_states=50_000,
        ),
        'hard': B200Params(
            max_states=50_000, chunk_size=50_000,
            joint_squad_size=2, joint_states=50_000,
            pass1_orderings=3, refine_iters=10, lns_rounds=5,
            max_dp_bots=5, explore_states=25_000,
        ),
        'expert': B200Params(
            max_states=50_000, chunk_size=50_000,
            joint_squad_size=2, joint_states=50_000,
            pass1_orderings=3, refine_iters=10, lns_rounds=5,
            max_dp_bots=7, explore_states=25_000,
        ),
        'nightmare': B200Params(
            # Nightmare uses NightmareTrainer, not GPU DP
            max_states=0, chunk_size=0,
            joint_squad_size=0, joint_states=0,
            pass1_orderings=0, refine_iters=0, lns_rounds=0,
            max_dp_bots=0, explore_states=0,
        ),
    },
}


def get_params(difficulty: str, gpu: str = 'auto') -> B200Params:
    """Return solver parameters scaled to GPU capability."""
    if gpu == 'auto':
        gpu = detect_gpu()
    profile = _PARAMS.get(gpu, _PARAMS.get('5090', {}))
    params = profile.get(difficulty)
    if params is None:
        # Fallback to 5090 defaults
        params = _PARAMS['5090'].get(difficulty, B200Params())
    return params


def print_gpu_info():
    """Print GPU detection results."""
    gpu = detect_gpu()
    vram = get_vram_gb()
    print(f"GPU: {gpu} ({vram:.1f} GB VRAM)", file=sys.stderr)
    return gpu


if __name__ == '__main__':
    gpu = print_gpu_info()
    for diff in ['easy', 'medium', 'hard', 'expert', 'nightmare']:
        p = get_params(diff, gpu)
        if diff == 'nightmare':
            print(f"  {diff}: NightmareTrainer (V3 + perturbation search)",
                  file=sys.stderr)
        else:
            print(f"  {diff}: max_states={p.max_states:,}, joint_squad={p.joint_squad_size}, "
                  f"joint_states={p.joint_states:,}, orderings={p.pass1_orderings}, "
                  f"refine={p.refine_iters}, lns={p.lns_rounds}, dp_bots={p.max_dp_bots}",
                  file=sys.stderr)

"""Re-submit R18 with GPU sim ensemble + sett_survival + tuned params."""
import json
import time
from pathlib import Path

import cma
import numpy as np

from calibration import CalibrationModel
from client import AstarIslandClient
from predict import validate_prediction
from predict_gemini import gemini_predict, _DEFAULTS
from sim_data import RoundData
from sim_inference import (
    WARM_STARTS, _compute_obs_features, _knn_warm_start, _load_transfer_data,
    detect_regime_from_obs, get_adaptive_alpha, PARAM_SPEC, PARAM_NAMES,
    params_to_vec, vec_to_params,
)
from sim_model_gpu import GPUSimulator
from utils import (
    apply_floor, build_growth_front_map, build_obs_overlay, build_sett_survival,
    terrain_to_class,
)
import predict
import predict_gemini

# Load calibration
cal = CalibrationModel.from_all_rounds()
predict._calibration = cal
print(f"Calibration: {cal.get_stats()['rounds_loaded']} rounds")

# Load best params
bp = json.load(open("best_params.json"))
for k, v in _DEFAULTS.items():
    if k not in bp:
        bp[k] = v
predict_gemini._load_params = lambda: dict(bp)

# Connect
client = AstarIslandClient()
round_id = "b0f9d1bf-4b71-4e6e-816c-19c718d29056"
detail = client.get_round_detail(round_id)
seeds_count = detail["seeds_count"]
initial_states = detail["initial_states"]
print(f"R18: {seeds_count} seeds")

# Load observations
obs_dir = Path("data/rounds") / round_id
obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
observations = [json.loads(f.read_text()) for f in obs_files]
print(f"Observations: {len(observations)}")

# Regime detection
state0 = initial_states[0]
terrain0 = np.array(state0["grid"], dtype=int)
regime = detect_regime_from_obs(observations, terrain0)
sim_alpha = get_adaptive_alpha(regime)
print(f"Regime: {regime}, alpha: {sim_alpha}")

# Build obs cells for fitting
H, W = 40, 40
obs_cells = []
for obs in observations:
    vp = obs["viewport"]
    grid = obs["grid"]
    for row in range(len(grid)):
        for col in range(len(grid[0]) if grid else 0):
            my, mx = vp["y"] + row, vp["x"] + col
            if 0 <= my < H and 0 <= mx < W:
                cls = terrain_to_class(grid[row][col])
                obs_cells.append((my, mx, cls))

obs_y = np.array([c[0] for c in obs_cells])
obs_x = np.array([c[1] for c in obs_cells])
obs_cls = np.array([c[2] for c in obs_cells])
print(f"Obs cells: {len(obs_cells)}")

# GPU sim fitting
rd0 = RoundData("R18", 0, terrain0, state0["settlements"])
sim_gpu = GPUSimulator(rd0, device="cuda")

lo = np.array([PARAM_SPEC[k][1] for k in PARAM_NAMES])
hi = np.array([PARAM_SPEC[k][2] for k in PARAM_NAMES])


def objective(vec):
    params = vec_to_params(vec)
    pred = sim_gpu.run(params, n_sims=2000, seed=42)
    pred_safe = np.maximum(pred, 1e-6)
    probs = pred_safe[obs_y, obs_x, obs_cls]
    return -float(np.mean(np.log(probs)))


# KNN warm start
td = _load_transfer_data()
obs_features = _compute_obs_features(observations, terrain0)
knn_params, neighbors = _knn_warm_start(obs_features, td, k=3)
print(f"KNN neighbors: {neighbors}")
print(f"Obs features: surv={obs_features['survival_rate']:.2f} sett={obs_features['sett_rate']:.3f}")

t0 = time.perf_counter()
starts = [
    ("knn", params_to_vec(knn_params), 0.2),
    (regime, params_to_vec(WARM_STARTS[regime]), 0.5),
]
for r in ["collapse", "moderate", "boom"]:
    if r != regime:
        starts.append((r, params_to_vec(WARM_STARTS[r]), 0.5))

best_vec = None
best_ll = float("inf")
for name, x0, sig in starts:
    try:
        opts = {"maxfevals": 200, "bounds": [lo, hi], "verbose": -9, "seed": 42}
        es = cma.CMAEvolutionStrategy(x0, sig, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(s) for s in solutions]
            es.tell(solutions, fitnesses)
        if es.result.fbest < best_ll:
            best_ll = es.result.fbest
            best_vec = es.result.xbest
            print(f"  {name}: ll={-best_ll:.4f} (best)")
    except Exception as e:
        print(f"  {name}: failed ({e})")

sim_params = vec_to_params(best_vec) if best_vec is not None else WARM_STARTS[regime]
t1 = time.perf_counter()
print(f"Sim fitting: {t1-t0:.1f}s")

# Generate sim predictions
print("\nGenerating sim predictions (10K sims per seed)...")
sim_predictions = {}
for seed_idx in range(seeds_count):
    state_si = initial_states[seed_idx]
    terrain_si = np.array(state_si["grid"], dtype=int)
    rd_si = RoundData("R18", seed_idx, terrain_si, state_si["settlements"])
    sim_si = GPUSimulator(rd_si, device="cuda")
    sim_predictions[seed_idx] = sim_si.run(sim_params, n_sims=10000, seed=42)

# Build per-seed evidence
growth_front_maps = {}
obs_overlays = {}
sett_survivals = {}
for seed_idx in range(seeds_count):
    terrain_si = np.array(initial_states[seed_idx]["grid"], dtype=int)
    seed_obs = [o for o in observations if o.get("seed_index") == seed_idx]
    if seed_obs:
        growth_front_maps[seed_idx] = build_growth_front_map(seed_obs, terrain_si)
    obs_overlays[seed_idx] = build_obs_overlay(observations, terrain_si, seed_idx)
    sett_survivals[seed_idx] = build_sett_survival(
        observations, initial_states[seed_idx]["settlements"], seed_idx
    )

est_vigor = obs_features["sett_rate"] if obs_features["sett_rate"] > 0 else None

# Submit
print("\nSubmitting improved predictions...")
from utils import GlobalMultipliers, FeatureKeyBuckets
gm = GlobalMultipliers()
fk = FeatureKeyBuckets()

for seed_idx in range(seeds_count):
    state = initial_states[seed_idx]
    prediction = gemini_predict(
        state, gm, fk,
        variance_regime=regime.upper() if regime == "boom" else None,
        est_vigor=est_vigor,
        sim_pred=sim_predictions.get(seed_idx),
        sim_alpha=sim_alpha,
        growth_front_map=growth_front_maps.get(seed_idx),
        obs_overlay=obs_overlays.get(seed_idx),
        sett_survival=sett_survivals.get(seed_idx),
    )

    errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
    if errors:
        prediction = apply_floor(prediction)

    resp = client.submit(round_id, seed_idx, prediction.tolist())
    sett_avg = prediction[:, :, 1].mean()
    print(f"  Seed {seed_idx}: {resp.get('status', '?')} (sett={sett_avg:.3f})")

print("\nR18 re-submitted with GPU sim + sett_survival + tuned params!")

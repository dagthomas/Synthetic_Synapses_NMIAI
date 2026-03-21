"""Quick recalibration test — simplified LOO."""
import json, math, time, sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from predict_gemini import gemini_predict
from utils import GlobalMultipliers, FeatureKeyBuckets, terrain_to_class
import predict

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path("data/calibration")
OBS_DIR = Path("data/rounds")
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
}
ROUND_NAMES = list(ROUND_IDS.keys())
BOOM = {"round6", "round7", "round11"}


def score(gt, pred):
    gs = np.maximum(gt, 1e-10)
    ent = -np.sum(gs * np.log(gs), axis=-1)
    dyn = ent > 0.01
    ps = np.maximum(pred, 1e-10)
    kl = np.sum(gs * np.log(gs / ps), axis=-1)
    wkl = float(np.sum(ent[dyn] * kl[dyn]) / ent[dyn].sum()) if dyn.any() else 0
    return max(0, min(100, 100 * math.exp(-3 * wkl)))


def get_dynamic(grid):
    t = np.array(grid, dtype=int)
    d = ~((t == 10) | (t == 5))
    d[0, :] = False; d[-1, :] = False; d[:, 0] = False; d[:, -1] = False
    return d


# Step 1: Generate LOO predictions
print("Generating LOO predictions...")
all_data = {}

for rn in ROUND_NAMES:
    train = [r for r in ROUND_NAMES + ["round1"] if r != rn]
    cal = CalibrationModel()
    for tr in train:
        cal.add_round(DATA_DIR / tr)
    predict._calibration_model = cal

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
        state = detail["initial_states"][si]
        gt = np.array(json.loads((DATA_DIR / rn / f"analysis_seed_{si}.json").read_text())["ground_truth"])
        pred = gemini_predict(state, gm, fk)
        dynamic = get_dynamic(state["grid"])
        seeds.append((pred, gt, dynamic, state))
    all_data[rn] = seeds
    print(f"  {rn}: done")

print("All predictions generated.\n")


# Step 2: LOO recalibration
def run_recalibration(method_fn):
    results = {}
    for test_rn in ROUND_NAMES:
        # Collect training pairs from other rounds
        train_p, train_g = [], []
        for tr in ROUND_NAMES:
            if tr == test_rn:
                continue
            for pred, gt, dyn, state in all_data[tr]:
                train_p.append(pred[dyn])
                train_g.append(gt[dyn])
        train_p = np.concatenate(train_p)
        train_g = np.concatenate(train_g)

        # Fit calibrator
        calibrator = method_fn(train_p, train_g)

        # Apply to test round
        raw_scores, cal_scores = [], []
        for pred, gt, dyn, state in all_data[test_rn]:
            raw_scores.append(score(gt, pred))
            pred_cal = calibrator(pred)
            grid = np.array(state["grid"])
            pred_cal[grid == 10] = [1, 0, 0, 0, 0, 0]
            pred_cal[grid == 5] = [0, 0, 0, 0, 0, 1]
            pred_cal[0, :] = [1, 0, 0, 0, 0, 0]; pred_cal[-1, :] = [1, 0, 0, 0, 0, 0]
            pred_cal[:, 0] = [1, 0, 0, 0, 0, 0]; pred_cal[:, -1] = [1, 0, 0, 0, 0, 0]
            cal_scores.append(score(gt, pred_cal))
        results[test_rn] = (np.mean(raw_scores), np.mean(cal_scores))
    return results


# --- Histogram Binning ---
def make_histogram(train_p, train_g, n_bins=20):
    edges = {}
    values = {}
    for c in range(NUM_CLASSES):
        p = train_p[:, c]
        g = train_g[:, c]
        pct = np.linspace(0, 100, n_bins + 1)
        e = np.unique(np.percentile(p, pct))
        if len(e) < 3:
            edges[c] = np.array([0, 1])
            values[c] = np.array([g.mean()])
            continue
        idx = np.digitize(p, e[1:-1])
        v = []
        for b in range(len(e) - 1):
            mask = idx == b
            v.append(g[mask].mean() if mask.sum() > 0 else e[min(b, len(e) - 2)])
        edges[c] = e
        values[c] = np.array(v)

    def transform(pred):
        r = pred.copy()
        for c in range(NUM_CLASSES):
            p = pred[:, :, c].flatten()
            bi = np.clip(np.digitize(p, edges[c][1:-1]), 0, len(values[c]) - 1)
            r[:, :, c] = values[c][bi].reshape(MAP_H, MAP_W)
        r = np.maximum(r, 1e-10)
        r /= r.sum(axis=-1, keepdims=True)
        return r

    return transform


# --- Dirichlet Diagonal ---
def make_dirichlet_diag(train_p, train_g):
    log_p = np.log(np.maximum(train_p, 1e-30))
    x0 = np.concatenate([np.ones(6), np.zeros(6)])

    def loss(x):
        w, b = x[:6], x[6:]
        logits = log_p * w + b
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return -np.sum(train_g * np.log(np.maximum(probs, 1e-30))) / len(train_g)

    res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 300})
    w_opt, b_opt = res.x[:6], res.x[6:]
    print(f"    Dirichlet diag W={np.round(w_opt,3)}, b={np.round(b_opt,3)}")

    def transform(pred):
        lp = np.log(np.maximum(pred, 1e-30))
        shape = lp.shape
        flat = lp.reshape(-1, 6)
        logits = flat * w_opt + b_opt
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.maximum(probs.reshape(shape), 1e-10)

    return transform


# --- Dirichlet Full Matrix ---
def make_dirichlet_full(train_p, train_g):
    log_p = np.log(np.maximum(train_p, 1e-30))
    x0 = np.concatenate([np.eye(6).flatten(), np.zeros(6)])

    def loss(x):
        W, b = x[:36].reshape(6, 6), x[36:]
        logits = log_p @ W.T + b
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return -np.sum(train_g * np.log(np.maximum(probs, 1e-30))) / len(train_g)

    res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 300})
    W_opt, b_opt = res.x[:36].reshape(6, 6), res.x[36:]
    print(f"    Dirichlet full W diag={np.round(np.diag(W_opt),3)}")

    def transform(pred):
        lp = np.log(np.maximum(pred, 1e-30))
        shape = lp.shape
        flat = lp.reshape(-1, 6)
        logits = flat @ W_opt.T + b_opt
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.maximum(probs.reshape(shape), 1e-10)

    return transform


# --- Isotonic per-class ---
def make_isotonic(train_p, train_g):
    from sklearn.isotonic import IsotonicRegression
    irs = {}
    for c in range(6):
        ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        ir.fit(train_p[:, c], train_g[:, c])
        irs[c] = ir

    def transform(pred):
        r = pred.copy()
        for c in range(6):
            r[:, :, c] = irs[c].transform(pred[:, :, c].flatten()).reshape(MAP_H, MAP_W)
        r = np.maximum(r, 1e-10)
        r /= r.sum(axis=-1, keepdims=True)
        return r

    return transform


# Run all methods
methods = [
    ("Histogram (20)", lambda p, g: make_histogram(p, g, 20)),
    ("Histogram (50)", lambda p, g: make_histogram(p, g, 50)),
    ("Dirichlet Diag", make_dirichlet_diag),
    ("Dirichlet Full", make_dirichlet_full),
]
try:
    from sklearn.isotonic import IsotonicRegression
    methods.append(("Isotonic", make_isotonic))
except ImportError:
    print("sklearn not available, skipping isotonic")

for name, method_fn in methods:
    print(f"\nTesting {name}...")
    t0 = time.time()
    results = run_recalibration(method_fn)
    elapsed = time.time() - t0

    print(f"  {name} ({elapsed:.0f}s):")
    print(f"  {'Round':<10} {'Raw':>7} {'Cal':>7} {'Delta':>7}")
    raw_all, cal_all = [], []
    for rn in ROUND_NAMES:
        raw, cal = results[rn]
        d = cal - raw
        tag = " *" if rn in BOOM else ""
        print(f"  {rn:<10} {raw:7.2f} {cal:7.2f} {d:+7.2f}{tag}")
        raw_all.append(raw)
        cal_all.append(cal)
    ra, ca = np.mean(raw_all), np.mean(cal_all)
    rb = np.mean([results[r][0] for r in BOOM])
    cb = np.mean([results[r][1] for r in BOOM])
    print(f"  {'AVG':<10} {ra:7.2f} {ca:7.2f} {ca-ra:+7.2f}")
    print(f"  {'BOOM':<10} {rb:7.2f} {cb:7.2f} {cb-rb:+7.2f}")

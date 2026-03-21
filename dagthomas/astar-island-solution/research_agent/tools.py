"""Custom tools for the Astar Island research agent.

Each tool is a plain Python function with typed parameters and docstrings.
Google ADK automatically wraps them as FunctionTool when passed to an Agent.
"""
import json
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root and lazy imports
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parent.parent

import sys
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from autoexperiment import BacktestHarness, compute_score
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KNOWLEDGE_PATH = _PROJECT_DIR / "KNOWLEDGE.md"
DISCOVERIES_PATH = _PROJECT_DIR / "DISCOVERIES.md"
IMPROVEMENTS_PATH = _PROJECT_DIR / "IMPROVEMENTS.md"
AUTOLOOP_PATH = _PROJECT_DIR / "AUTOLOOP.md"
DATA_DIR = _PROJECT_DIR / "data" / "calibration"
OBS_DIR = _PROJECT_DIR / "data" / "rounds"
LOG_PATH = _PROJECT_DIR / "data" / "adk_research_log.jsonl"

ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
}
ROUND_NAMES = ["round2", "round3", "round4", "round5", "round6", "round7"]


# ---------------------------------------------------------------------------
# Experiment log (append-only JSONL)
# ---------------------------------------------------------------------------
class _ExperimentLog:
    """Singleton experiment log shared across tool invocations."""

    def __init__(self):
        self.entries: list[dict] = []
        self.best_score: float = 0.0
        self.best_code: str = ""
        self._load()

    def _load(self):
        if LOG_PATH.exists():
            for line in LOG_PATH.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                        avg = entry.get("scores", {}).get("avg", 0)
                        if avg > self.best_score:
                            self.best_score = avg
                            self.best_code = entry.get("code", "")
                    except json.JSONDecodeError:
                        continue

    def append(self, entry: dict):
        self.entries.append(entry)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        avg = entry.get("scores", {}).get("avg", 0)
        if avg > self.best_score:
            self.best_score = avg
            self.best_code = entry.get("code", "")


_log = _ExperimentLog()


# ---------------------------------------------------------------------------
# Backtest harness (lazy singleton)
# ---------------------------------------------------------------------------
_harness: BacktestHarness | None = None


def _get_harness() -> BacktestHarness:
    global _harness
    if _harness is None:
        _harness = BacktestHarness(seeds_per_round=5)
    return _harness


# ---------------------------------------------------------------------------
# Code compilation helpers
# ---------------------------------------------------------------------------
def _compile_pred_fn(code: str):
    """Compile experimental code into a callable prediction function."""
    exec_globals = {
        "np": np,
        "numpy": np,
        "math": math,
        "CalibrationModel": CalibrationModel,
        "build_feature_keys": build_feature_keys,
        "MAP_H": MAP_H,
        "MAP_W": MAP_W,
        "NUM_CLASSES": NUM_CLASSES,
        "_build_coastal_mask": _build_coastal_mask,
        "_build_feature_key_index": _build_feature_key_index,
        "build_calibration_lookup": build_calibration_lookup,
        "build_fk_empirical_lookup": build_fk_empirical_lookup,
        "FeatureKeyBuckets": FeatureKeyBuckets,
        "GlobalMultipliers": GlobalMultipliers,
        "terrain_to_class": terrain_to_class,
        "predict": predict,
    }
    exec(code, exec_globals)
    fn = exec_globals.get("experimental_pred_fn")
    if fn is None:
        raise RuntimeError(
            "Code does not define 'experimental_pred_fn'. "
            f"Defined names: {[k for k in exec_globals if not k.startswith('_')]}"
        )
    return fn


def _validate_prediction(pred: np.ndarray, terrain: np.ndarray) -> list[str]:
    """Check a prediction for hard-rule violations."""
    errors = []
    if pred.shape != (MAP_H, MAP_W, NUM_CLASSES):
        errors.append(f"Shape mismatch: expected ({MAP_H},{MAP_W},{NUM_CLASSES}), got {pred.shape}")
        return errors
    if np.any(np.isnan(pred)):
        errors.append(f"NaN values: {int(np.isnan(pred).sum())} entries")
    if np.any(pred < 0):
        errors.append(f"Negative values: {int((pred < 0).sum())} entries")
    row_sums = pred.sum(axis=-1)
    bad = np.abs(row_sums - 1.0) > 0.02
    if bad.any():
        errors.append(f"{int(bad.sum())} cells don't sum to 1.0")
    dynamic = ~((terrain == 10) | (terrain == 5))
    if np.any(pred[dynamic, 5] > 0.001):
        errors.append("Mountain class nonzero on dynamic cells")
    return errors


def _auto_fix_prediction(pred: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Best-effort fix for common prediction problems."""
    pred = np.nan_to_num(pred.copy(), nan=0.0, posinf=1.0, neginf=0.0)
    pred = np.maximum(pred, 0.0)
    row_sums = pred.sum(axis=-1, keepdims=True)
    pred = np.where(row_sums > 1e-10, pred / row_sums, 1.0 / NUM_CLASSES)
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0
    coastal = _build_coastal_mask(terrain)
    pred[dynamic_mask & ~coastal, 2] = 0.0
    dp = pred[dynamic_mask]
    nz = dp > 0
    dp = np.where(nz, np.maximum(dp, 0.005), 0.0)
    dp /= np.maximum(dp.sum(axis=-1, keepdims=True), 1e-10)
    pred[dynamic_mask] = dp
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]
    return pred


# ===================================================================
# TOOL FUNCTIONS (exposed to the ADK agent)
# ===================================================================

def run_backtest(code: str, name: str = "", hypothesis: str = "") -> dict:
    """Compile and backtest an experimental prediction function.

    The code must define a function called experimental_pred_fn with signature:
        def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray
    where state has 'grid' (40x40 terrain codes) and 'settlements' (list of dicts),
    and the function returns a (40, 40, 6) probability tensor.

    Args:
        code: Complete Python source defining experimental_pred_fn.
        name: Short descriptive name for this experiment (e.g. "coastal distance decay").
              ALWAYS provide a name — it appears in the TUI experiment table.
        hypothesis: 1-2 sentences explaining what change was made and why it should help.

    Returns:
        dict with keys: status, scores (per-round and avg), warnings, error, elapsed.
        If the code fails to compile or crashes during backtest, status will be 'error'.
    """
    result = {
        "status": "error",
        "scores": {},
        "warnings": [],
        "error": "",
        "elapsed": 0.0,
        "improvement_vs_best": 0.0,
    }

    # --- Compile ---
    try:
        pred_fn = _compile_pred_fn(code)
    except Exception as e:
        result["error"] = f"Compilation failed: {str(e)[:500]}"
        _log.append({
            "id": len(_log.entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "hypothesis": hypothesis,
            "status": "compile_error",
            "error": result["error"],
            "code": code[:2000],
        })
        return result

    # --- Backtest ---
    import time
    harness = _get_harness()
    round_names = list(harness.round_data.keys())
    scores = {}
    warnings = []
    t0 = time.time()

    for test_round in round_names:
        train_rounds = [r for r in round_names + ["round1"] if r != test_round]
        try:
            global_mult, fk_buckets = harness.build_obs_context(test_round, train_rounds)
        except Exception as e:
            result["error"] = f"Context build failed for {test_round}: {e}"
            return result

        round_scores = []
        for si, seed_data in enumerate(harness.round_data[test_round]["seeds"]):
            try:
                pred = pred_fn(seed_data["state"], global_mult, fk_buckets)
            except Exception as e:
                tb = traceback.format_exc()
                result["error"] = (
                    f"Prediction crashed on {test_round} seed {si}: {str(e)[:300]}\n"
                    f"Traceback:\n{tb[-500:]}"
                )
                _log.append({
                    "id": len(_log.entries),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": name,
                    "hypothesis": hypothesis,
                    "status": "runtime_error",
                    "error": result["error"],
                    "code": code[:2000],
                })
                return result

            terrain = np.array(seed_data["state"]["grid"], dtype=int)
            errs = _validate_prediction(pred, terrain)
            if errs:
                warnings.extend([f"{test_round}/s{si}: {e}" for e in errs])
                pred = _auto_fix_prediction(pred, terrain)

            score = compute_score(seed_data["gt"], pred)
            round_scores.append(score)

        scores[test_round] = round(float(np.mean(round_scores)), 3)

    elapsed = time.time() - t0
    scores["avg"] = round(float(np.mean([scores[r] for r in round_names if r in scores])), 3)

    improvement = scores["avg"] - _log.best_score if _log.best_score > 0 else 0.0

    result.update({
        "status": "success",
        "scores": scores,
        "warnings": warnings[:10],
        "elapsed": round(elapsed, 2),
        "improvement_vs_best": round(improvement, 3),
    })

    # Log
    _log.append({
        "id": len(_log.entries),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "hypothesis": hypothesis,
        "status": "success",
        "scores": scores,
        "improvement": improvement,
        "warnings": warnings[:5],
        "elapsed": round(elapsed, 2),
        "code": code[:4000],
    })

    # Save code if it's a new best
    if improvement > 0.01:
        best_path = _PROJECT_DIR / "data" / f"adk_best_{len(_log.entries)}.py"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(
            f"# ADK Research Agent Best Experiment #{len(_log.entries)}\n"
            f"# Score: avg={scores['avg']:.3f}, improvement={improvement:+.3f}\n"
            f"# Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
            f"import numpy as np\nimport math\n"
            f"from calibration import CalibrationModel, build_feature_keys\n"
            f"from config import MAP_H, MAP_W, NUM_CLASSES\n"
            f"from fast_predict import (\n"
            f"    _build_coastal_mask,\n"
            f"    _build_feature_key_index,\n"
            f"    build_calibration_lookup,\n"
            f"    build_fk_empirical_lookup,\n"
            f")\n"
            f"from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class\n"
            f"import predict\n\n"
            f"{code}\n",
            encoding="utf-8",
        )

    return result


def read_knowledge() -> str:
    """Read the KNOWLEDGE.md file containing all findings from R1-R5 analysis.

    Returns:
        The full contents of KNOWLEDGE.md which documents simulation rules,
        settlement dynamics, scoring formula, error analysis, architecture details,
        and known improvement opportunities.
    """
    sections = []
    for path, label in [
        (KNOWLEDGE_PATH, "KNOWLEDGE.md"),
        (DISCOVERIES_PATH, "DISCOVERIES.md"),
        (IMPROVEMENTS_PATH, "IMPROVEMENTS.md"),
    ]:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if len(content) > 6000:
                content = content[:6000] + "\n... (truncated)"
            sections.append(f"=== {label} ===\n{content}")
    return "\n\n".join(sections) if sections else "No knowledge files found."


def read_experiment_log() -> str:
    """Read recent experiment results from the ADK research log.

    Returns:
        A text summary of the last 15 experiments including scores,
        improvement deltas, errors, and the overall best score achieved.
    """
    if not _log.entries:
        return "No experiments have been run yet. Best score: 0.0"

    lines = [
        f"Total experiments: {len(_log.entries)}",
        f"Best score: {_log.best_score:.3f}",
    ]

    # Top 5
    scored = [e for e in _log.entries if e.get("scores") and e["scores"].get("avg")]
    scored.sort(key=lambda e: e["scores"]["avg"], reverse=True)
    if scored:
        lines.append("\nTop 5 experiments:")
        for e in scored[:5]:
            s = e["scores"]
            lines.append(
                f"  [#{e.get('id', '?')}] avg={s['avg']:.2f} "
                f"(R2={s.get('round2', 0):.1f} R3={s.get('round3', 0):.1f} "
                f"R4={s.get('round4', 0):.1f} R5={s.get('round5', 0):.1f})"
            )

    # Recent 15
    recent = _log.entries[-15:]
    lines.append(f"\nLast {len(recent)} experiments:")
    for e in recent:
        status = e.get("status", "unknown")
        score_str = ""
        if e.get("scores") and e["scores"].get("avg"):
            avg = e["scores"]["avg"]
            imp = e.get("improvement", 0)
            score_str = f" avg={avg:.2f} ({imp:+.3f})"
        err_str = ""
        if e.get("error"):
            err_str = f" ERROR: {e['error'][:100]}"
        lines.append(f"  [#{e.get('id', '?')}] {status}{score_str}{err_str}")

    # Failed count
    failed = [e for e in _log.entries if e.get("error")]
    if failed:
        lines.append(f"\nFailed: {len(failed)}/{len(_log.entries)}")

    return "\n".join(lines)


def get_round_analysis(round_name: str) -> str:
    """Get detailed per-round analysis including ground truth statistics.

    Loads the ground truth for the specified round and returns class distributions,
    entropy statistics, and per-seed scores for detailed error analysis.

    Args:
        round_name: One of 'round2', 'round3', 'round4', 'round5'.

    Returns:
        Detailed text analysis of the round's ground truth and our prediction performance.
    """
    if round_name not in ROUND_NAMES:
        return f"Invalid round name '{round_name}'. Valid: {ROUND_NAMES}"

    round_dir = DATA_DIR / round_name
    if not round_dir.exists():
        return f"No calibration data found for {round_name} at {round_dir}"

    try:
        detail = json.loads((round_dir / "round_detail.json").read_text())
    except Exception as e:
        return f"Failed to load round detail: {e}"

    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    lines = [f"=== {round_name.upper()} Analysis ==="]
    lines.append(f"Seeds: {detail['seeds_count']}")
    lines.append(f"Map size: {detail['map_width']}x{detail['map_height']}")

    for si in range(min(5, detail["seeds_count"])):
        analysis_path = round_dir / f"analysis_seed_{si}.json"
        if not analysis_path.exists():
            continue

        analysis = json.loads(analysis_path.read_text())
        gt = np.array(analysis["ground_truth"])
        gt_safe = np.maximum(gt, 1e-10)
        entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
        argmax = np.argmax(gt, axis=-1)
        dynamic = entropy > 0.01

        lines.append(f"\n--- Seed {si} ---")
        lines.append(f"  Dynamic cells: {int(dynamic.sum())}/{gt.shape[0] * gt.shape[1]}")
        lines.append(f"  Mean entropy: {float(entropy.mean()):.4f}")
        lines.append(f"  Max entropy: {float(entropy.max()):.4f}")

        # Class distribution (argmax)
        lines.append("  Ground truth distribution (argmax):")
        for cls_idx, cls_name in enumerate(class_names):
            count = int((argmax == cls_idx).sum())
            pct = 100 * count / (gt.shape[0] * gt.shape[1])
            lines.append(f"    {cls_name}: {count} ({pct:.1f}%)")

        # Mean GT probability per class on dynamic cells
        if dynamic.any():
            lines.append("  Mean GT probability on dynamic cells:")
            for cls_idx, cls_name in enumerate(class_names):
                mean_p = float(gt[dynamic, cls_idx].mean())
                lines.append(f"    {cls_name}: {mean_p:.4f}")

        # Settlement analysis
        terrain = np.array(analysis.get("initial_grid", detail["initial_states"][si]["grid"]))
        initial_sett = np.sum((terrain == 1) | (terrain == 2))
        final_sett = int((argmax == 1).sum())
        final_port = int((argmax == 2).sum())
        final_ruin = int((argmax == 3).sum())
        lines.append(f"  Initial settlements+ports: {int(initial_sett)}")
        lines.append(f"  Final argmax — settlement: {final_sett}, port: {final_port}, ruin: {final_ruin}")

    # Per-class error contribution (if we have the baseline prediction data)
    harness = _get_harness()
    if round_name in harness.round_data:
        lines.append("\n--- Per-class weighted KL contribution ---")
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != round_name]
        try:
            global_mult, fk_buckets = harness.build_obs_context(round_name, train_rounds)
            from gemini_researcher import make_baseline_pred_fn
            baseline_fn = make_baseline_pred_fn()
            for si, seed_data in enumerate(harness.round_data[round_name]["seeds"][:2]):
                pred = baseline_fn(seed_data["state"], global_mult, fk_buckets)
                gt = seed_data["gt"]
                gt_safe = np.maximum(gt, 1e-10)
                pred_safe = np.maximum(pred, 1e-10)
                entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
                dynamic = entropy > 0.01
                if not dynamic.any():
                    continue
                # Per-class KL
                kl_per_class = gt * np.log(gt_safe / pred_safe)  # (H,W,6)
                total_entropy = entropy[dynamic].sum()
                lines.append(f"  Seed {si} per-class wKL contribution:")
                for cls_idx, cls_name in enumerate(class_names):
                    cls_kl = kl_per_class[:, :, cls_idx]
                    cls_wkl = float(np.sum(entropy[dynamic] * cls_kl[dynamic]) / total_entropy)
                    lines.append(f"    {cls_name}: {cls_wkl:.5f}")
        except Exception as e:
            lines.append(f"  (Could not compute baseline error: {e})")

    return "\n".join(lines)


def write_prediction_code(name: str, hypothesis: str, code: str) -> dict:
    """Save a proposed prediction function to disk for reference.

    Use this BEFORE run_backtest to record the hypothesis and name alongside
    the code. The code is also saved to a timestamped file.

    Args:
        name: Short descriptive name for this experiment (under 60 chars).
        hypothesis: 1-3 sentences explaining why this change should improve scores.
        code: Complete Python source defining experimental_pred_fn.

    Returns:
        dict with status and file path where the code was saved.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:40]
    filename = f"experiment_{timestamp}_{safe_name}.py"
    save_path = _PROJECT_DIR / "data" / "experiments" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Experiment: {name}\n"
        f"# Hypothesis: {hypothesis}\n"
        f"# Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"import numpy as np\nimport math\n"
        f"from calibration import CalibrationModel, build_feature_keys\n"
        f"from config import MAP_H, MAP_W, NUM_CLASSES\n"
        f"from fast_predict import (\n"
        f"    _build_coastal_mask,\n"
        f"    _build_feature_key_index,\n"
        f"    build_calibration_lookup,\n"
        f"    build_fk_empirical_lookup,\n"
        f")\n"
        f"from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class\n"
        f"import predict\n\n"
    )
    save_path.write_text(header + code, encoding="utf-8")

    return {
        "status": "saved",
        "path": str(save_path),
        "name": name,
        "hypothesis": hypothesis[:200],
    }


def read_source_code(file_name: str, start_line: int = 1, end_line: int = 100) -> str:
    """Read source code from the project. Use this to understand the current
    prediction pipeline before proposing changes.

    Args:
        file_name: Name of file to read (e.g., 'predict.py', 'calibration.py',
                   'utils.py', 'fast_predict.py', 'config.py', 'explore.py')
        start_line: First line to read (1-indexed)
        end_line: Last line to read (1-indexed)

    Returns:
        The requested source code with line numbers.
    """
    allowed_files = [
        "predict.py", "calibration.py", "utils.py", "fast_predict.py",
        "config.py", "explore.py", "submit.py", "estimator.py",
        "autoexperiment.py",
    ]
    if file_name not in allowed_files:
        return f"File '{file_name}' not found. Available: {', '.join(allowed_files)}"

    path = _PROJECT_DIR / file_name
    if not path.exists():
        return f"File not found: {path}"

    lines = path.read_text(encoding="utf-8").split("\n")
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    selected = lines[start:end]

    result = f"# {file_name} lines {start_line}-{end_line} ({len(lines)} total)\n"
    for i, line in enumerate(selected, start=start_line):
        result += f"{i:4d}  {line}\n"
    return result


def list_source_files() -> str:
    """List all Python source files in the project with their sizes and brief descriptions.

    Returns:
        A formatted list of available source files.
    """
    files_info = {
        "predict.py": "Main prediction engine — get_static_prior, predict_for_seed, smart floor",
        "calibration.py": "CalibrationModel — hierarchical prior from R1-R5 ground truth",
        "utils.py": "GlobalMultipliers, FeatureKeyBuckets, ObservationAccumulator, GlobalTransitionMatrix",
        "fast_predict.py": "Vectorized prediction — fast_predict, build_calibration_lookup",
        "config.py": "Constants — MAP_H/W, NUM_CLASSES, TERRAIN_TO_CLASS, PROB_FLOOR",
        "explore.py": "Exploration strategy — viewport selection, observation collection",
        "submit.py": "Submission orchestrator — explore -> predict -> submit",
        "estimator.py": "ParameterEstimator — regime detection from settlement stats",
    }
    result = "Source files:\n"
    for name, desc in files_info.items():
        path = _PROJECT_DIR / name
        size = path.stat().st_size if path.exists() else 0
        lines = len(path.read_text().split("\n")) if path.exists() else 0
        result += f"  {name:25s} {lines:4d} lines  {desc}\n"
    return result

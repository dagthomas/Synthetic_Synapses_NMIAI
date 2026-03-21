#!/usr/bin/env python3
"""Gemini-powered autonomous research agent for Astar Island prediction pipeline.

Uses Google Gemini API to propose STRUCTURAL algorithmic changes (not just
parameter tweaks), implements them as self-contained prediction functions,
backtests against R1-R5 ground truth, and iterates indefinitely.

Usage:
    # Set API key via env var or interactive prompt
    export GEMINI_API_KEY=your_key_here

    # Run the research loop
    python gemini_researcher.py

    # Dry-run: show what Gemini proposes without running backtests
    python gemini_researcher.py --dry-run

    # Limit iterations
    python gemini_researcher.py --max-iters 10

    # Resume from experiment log
    python gemini_researcher.py --resume
"""
import argparse
import json
import math
import os
import re
import textwrap
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Load .env file if present
_PROJECT_DIR = Path(__file__).parent
_env_path = _PROJECT_DIR / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# ============================================================
# Lazy imports for project modules
# ============================================================
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

# ============================================================
# Constants
# ============================================================
LOG_PATH = _PROJECT_DIR / "data" / "gemini_research_log.jsonl"
KNOWLEDGE_PATH = _PROJECT_DIR / "KNOWLEDGE.md"
AUTOLOOP_PATH = _PROJECT_DIR / "AUTOLOOP.md"
IMPROVEMENTS_PATH = _PROJECT_DIR / "IMPROVEMENTS.md"
DISCOVERIES_PATH = _PROJECT_DIR / "DISCOVERIES.md"

DATA_DIR = _PROJECT_DIR / "data" / "calibration"
OBS_DIR = _PROJECT_DIR / "data" / "rounds"

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
    "round12": "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
}
ROUND_NAMES = ["round2", "round3", "round4", "round5", "round6", "round7", "round9", "round10", "round11", "round12"]


# ============================================================
# EXPERIMENT LOG
# ============================================================

class ResearchLog:
    """Append-only JSONL log for Gemini research experiments."""

    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.entries: list[dict] = []
        self.best_score = 0.0
        self.best_experiment_id = -1
        self._load()

    def _load(self):
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                        if entry.get("scores") and entry["scores"].get("avg", 0) > self.best_score:
                            self.best_score = entry["scores"]["avg"]
                            self.best_experiment_id = entry.get("id", -1)
                    except json.JSONDecodeError:
                        continue

    def append(self, entry: dict):
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def count(self) -> int:
        return len(self.entries)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get last N experiments for context."""
        return self.entries[-n:]

    def get_best(self, n: int = 5) -> list[dict]:
        """Get top N experiments by score."""
        scored = [e for e in self.entries if e.get("scores") and e["scores"].get("avg")]
        scored.sort(key=lambda e: e["scores"]["avg"], reverse=True)
        return scored[:n]

    def get_summary_text(self) -> str:
        """Build a text summary for Gemini context."""
        if not self.entries:
            return "No experiments have been run yet."

        lines = [f"Total experiments: {len(self.entries)}"]
        lines.append(f"Best score: {self.best_score:.3f}")

        # Best experiments
        best = self.get_best(5)
        if best:
            lines.append("\nTop 5 experiments:")
            for e in best:
                s = e["scores"]
                lines.append(
                    f"  [{e.get('id', '?')}] {e.get('name', '?')[:60]}: "
                    f"avg={s['avg']:.2f} "
                    f"boom={s.get('boom_avg', 0):.1f} nonboom={s.get('nonboom_avg', 0):.1f}"
                )

        # Recent experiments
        recent = self.get_recent(8)
        if recent:
            lines.append("\nLast 8 experiments:")
            for e in recent:
                status = "SUCCESS" if e.get("scores") else "FAILED"
                score_str = ""
                if e.get("scores"):
                    score_str = f" avg={e['scores'].get('avg', 0):.2f}"
                    improvement = ""
                    if e.get("improvement"):
                        improvement = f" ({e['improvement']})"
                    score_str += improvement
                error_str = ""
                if e.get("error"):
                    error_str = f" ERROR: {e['error'][:80]}"
                lines.append(
                    f"  [{e.get('id', '?')}] {status}{score_str}{error_str}"
                )
                # Show a compact version of what was proposed
                if e.get("proposal_summary"):
                    lines.append(f"      Proposal: {e['proposal_summary'][:100]}")

        # Failed experiments
        failed = [e for e in self.entries if e.get("error")]
        if failed:
            lines.append(f"\nFailed experiments: {len(failed)}/{len(self.entries)}")
            for e in failed[-3:]:
                lines.append(f"  [{e.get('id', '?')}] {e.get('error', '')[:120]}")

        return "\n".join(lines)


# ============================================================
# KNOWLEDGE BASE LOADER
# ============================================================

def load_knowledge_base() -> str:
    """Load all knowledge files into a single context string."""
    sections = []

    for path, label in [
        (KNOWLEDGE_PATH, "KNOWLEDGE.md"),
        (DISCOVERIES_PATH, "DISCOVERIES.md"),
        (IMPROVEMENTS_PATH, "IMPROVEMENTS.md (recent history)"),
    ]:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # Truncate if very long
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            sections.append(f"=== {label} ===\n{content}")

    return "\n\n".join(sections)


# ============================================================
# BASELINE PREDICTION FUNCTION
# ============================================================

def make_baseline_pred_fn():
    """Create the current production baseline prediction function.

    Uses predict_gemini.py directly — the actual production code.
    """
    from predict_gemini import gemini_predict

    def pred_fn(state, global_mult, fk_buckets):
        return gemini_predict(state, global_mult, fk_buckets)

    return pred_fn


def _make_baseline_pred_fn_OLD():
    """Old baseline — kept for reference."""
    params = {
        "fk_prior_weight": 1.5,
        "fk_max_strength": 20.0,
        "fk_min_count": 5,
        "fk_strength_fn": "sqrt",
        "mult_power": 0.3,
        "mult_smooth": 5.0,
        "mult_sett_lo": 0.15,
        "mult_sett_hi": 2.0,
        "mult_forest_lo": 0.5,
        "mult_forest_hi": 1.8,
        "mult_empty_lo": 0.75,
        "mult_empty_hi": 1.25,
        "floor_nonzero": 0.008,
        "cal_fine_base": 1.0,
        "cal_fine_divisor": 120.0,
        "cal_fine_max": 4.0,
        "cal_coarse_base": 0.75,
        "cal_coarse_divisor": 200.0,
        "cal_coarse_max": 3.0,
        "cal_base_base": 0.5,
        "cal_base_divisor": 1000.0,
        "cal_base_max": 1.5,
        "cal_global_weight": 0.4,
    }

    def pred_fn(state, global_mult, fk_buckets):
        grid = state["grid"]
        settlements = state["settlements"]
        terrain = np.array(grid, dtype=int)
        h, w = terrain.shape

        cal = predict.get_calibration()
        fkeys = build_feature_keys(terrain, settlements)
        idx_grid, unique_keys = _build_feature_key_index(fkeys)
        cal_priors = build_calibration_lookup(cal, unique_keys, params)
        fk_min = params["fk_min_count"]
        fk_emp, fk_cnt = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

        pred = cal_priors[idx_grid]
        emp_grid = fk_emp[idx_grid]
        cnt_grid = fk_cnt[idx_grid]
        has_fk = cnt_grid >= fk_min

        pw = params["fk_prior_weight"]
        ms = params["fk_max_strength"]
        strengths = np.minimum(ms, np.sqrt(cnt_grid))

        blended = pred * pw + emp_grid * strengths[:, :, np.newaxis]
        blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
        pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

        # Multiplier
        if global_mult.observed.sum() > 0:
            smooth = params["mult_smooth"] * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            ratio = (global_mult.observed + smooth) / np.maximum(
                global_mult.expected + smooth, 1e-6
            )
            ratio = np.power(ratio, params["mult_power"])
            ratio[0] = np.clip(ratio[0], params["mult_empty_lo"], params["mult_empty_hi"])
            ratio[5] = np.clip(ratio[5], 0.85, 1.15)
            for c in (1, 2, 3):
                ratio[c] = np.clip(ratio[c], params["mult_sett_lo"], params["mult_sett_hi"])
            ratio[4] = np.clip(ratio[4], params["mult_forest_lo"], params["mult_forest_hi"])
            pred *= ratio[np.newaxis, np.newaxis, :]
            pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

        # Structural zeros
        static_mask = (terrain == 10) | (terrain == 5)
        dynamic_mask = ~static_mask
        pred[dynamic_mask, 5] = 0.0
        coastal = _build_coastal_mask(terrain)
        inland_dynamic = dynamic_mask & ~coastal
        pred[inland_dynamic, 2] = 0.0

        # Floor
        floor = params["floor_nonzero"]
        dp = pred[dynamic_mask]
        nz = dp > 0
        dp = np.where(nz, np.maximum(dp, floor), 0.0)
        dp /= np.maximum(dp.sum(axis=-1, keepdims=True), 1e-10)
        pred[dynamic_mask] = dp

        # Lock static
        pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
        pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

        return pred

    return pred_fn


# ============================================================
# GEMINI API INTERFACE
# ============================================================

def get_gemini_api_key() -> str:
    """Get API key from env var or interactive prompt."""
    import os
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment.")
        key = input("Enter your Gemini API key: ").strip()
    if not key:
        raise ValueError("No API key provided. Set GEMINI_API_KEY env var or enter interactively.")
    return key


def init_gemini(api_key: str):
    """Initialize the Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package not installed. Run: pip install google-genai"
        )
    client = genai.Client(api_key=api_key)
    return client


def build_system_prompt(knowledge_base: str, experiment_log_summary: str) -> str:
    """Build the system prompt for Gemini."""
    return textwrap.dedent(f"""\
    You are an expert AI researcher working on the Astar Island prediction challenge.

    ## Your Task
    You propose STRUCTURAL algorithmic improvements to a prediction pipeline that
    predicts a 40x40x6 probability tensor for a Norse civilisation simulator.

    The pipeline predicts 6 classes per cell: [empty, settlement, port, ruin, forest, mountain].
    Score is entropy-weighted KL divergence (0-100, higher is better).
    Current best production backtest score: ~87.7 avg across R2-R12 (10 rounds LOO).
    Boom rounds (R6, R7, R11): ~85 avg. Non-boom: ~89 avg. Boom is the biggest weakness.

    ## Architecture Overview
    The prediction function signature is:
        pred_fn(state, global_mult, fk_buckets) -> np.ndarray  # shape (40,40,6)

    Where:
    - state: dict with "grid" (40x40 terrain codes) and "settlements" (list of dicts)
    - global_mult: GlobalMultipliers object (observed vs expected class counts)
      - global_mult.observed: np.ndarray shape (6,) — observed class counts
      - global_mult.expected: np.ndarray shape (6,) — expected class counts from priors
    - fk_buckets: FeatureKeyBuckets object (empirical distributions by feature key)
      - fk_buckets.get_empirical(key) -> (distribution_or_None, count)

    Terrain codes: 10=ocean, 11=plains, 0=empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain
    Class indices: 0=empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain

    ## Available Imports and Utilities
    Your code can use these (they are pre-imported in the execution environment):
    ```python
    import numpy as np
    import math
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

    # CalibrationModel hierarchy:
    # cal = predict.get_calibration()
    # cal.fine_sums[feature_key] -> np.ndarray(6,)  (sum of GT probs for this exact key)
    # cal.fine_counts[feature_key] -> int  (how many cells had this key)
    # cal.coarse_sums[(terrain, dist_bucket, coastal, has_port)] -> np.ndarray(6,)
    # cal.coarse_counts[coarse_key] -> int
    # cal.base_sums[terrain_code] -> np.ndarray(6,)
    # cal.base_counts[terrain_code] -> int
    # cal.global_probs -> np.ndarray(6,)  (overall class distribution)

    # Feature key structure: (terrain_code, dist_bucket, coastal_bool, forest_neighbors, has_port_flag)
    # dist_bucket: 0=on settlement, 1=dist 1, 2=dist 2, 3=dist 3, 4=dist 4+
    # forest_neighbors: 0-3 (count of adjacent forest cells, capped at 3)
    # has_port_flag: -1=not a settlement, 0=settlement without port, 1=settlement with port

    # build_feature_keys(terrain_np, settlements) -> list[list[tuple]]
    # _build_feature_key_index(fkeys) -> (idx_grid, unique_keys)
    # _build_coastal_mask(terrain) -> np.ndarray bool
    # build_calibration_lookup(cal, unique_keys, params) -> np.ndarray shape (N, 6)
    #   params needed: cal_fine_base, cal_fine_divisor, cal_fine_max,
    #                  cal_coarse_base, cal_coarse_divisor, cal_coarse_max,
    #                  cal_base_base, cal_base_divisor, cal_base_max,
    #                  cal_global_weight
    # build_fk_empirical_lookup(fk_buckets, unique_keys, min_count) -> (empiricals, counts)
    ```

    ## Hard Rules (NEVER violate these)
    1. Mountain (class 5) is ALWAYS 0 on non-mountain cells
    2. Port (class 2) is ALWAYS 0 on non-coastal cells
    3. Ocean cells MUST be [1,0,0,0,0,0], Mountain cells MUST be [0,0,0,0,0,1]
    4. All rows must sum to 1.0 (valid probability distributions)
    5. Use floor >= 0.005 for nonzero classes on dynamic cells (to avoid infinite KL divergence)
    6. The function must be deterministic (no random elements)

    ## Current Pipeline Steps (baseline — predict_gemini.py)
    1. Build calibrated prior from CalibrationModel (hierarchical: fine -> coarse -> base -> global)
    2. Blend with feature-key empirical data (prior_w=1.5, emp_max=20.0, sqrt strength)
    3. Apply global multipliers (power=0.3, per-class clamping)
    4. Cell-level distance-aware dampening (settlements get full mult, expansion cells dampened by 0.4)
    5. Entropy-weighted global temperature scaling (T_high=1.15, softens uncertain cells)
    6. Selective spatial smoothing (15%, settlement+ruin only, NOT port)
    7. Proportional redistribution of structural zeros
    8. Floor nonzero classes at 0.008, renormalize
    9. Lock static cells (ocean, mountain, borders)

    ## What NOT to propose
    - Pure parameter tweaks (e.g., "change fk_prior_weight from 5.0 to 4.0") — the autoloop
      already searched 10,000+ parameter combinations
    - Changes that break the hard rules above
    - Changes that add randomness to predictions
    - Overly complex changes with many nested loops that will be slow

    ## What TO propose (structural changes)
    - New features in the feature key (e.g., settlement population encoding)
    - Different blending strategies (e.g., entropy-weighted, distance-adaptive)
    - Spatial correlation models (e.g., neighbor averaging, Gaussian smoothing)
    - Per-terrain-type calibration weight overrides
    - Coastal-specific prediction models
    - Settlement density-aware adjustments
    - Non-linear transformations of the prior
    - New post-processing steps (e.g., spatial consistency enforcement)
    - Observation-count-adaptive strategies
    - Per-class blending strategies (e.g., trust empirical more for forest, prior more for settlement)

    ## Knowledge Base
    {knowledge_base}

    ## Experiment History
    {experiment_log_summary}
    """)


def build_proposal_prompt(iteration: int, experiment_log_summary: str) -> str:
    """Build the prompt asking Gemini to propose an experiment."""
    return textwrap.dedent(f"""\
    ## Experiment Iteration {iteration}

    Updated experiment history:
    {experiment_log_summary}

    Based on the knowledge base, experiment history, and your understanding of the
    prediction pipeline, propose ONE specific structural algorithmic change.

    Requirements:
    1. Explain the HYPOTHESIS: why should this change improve scores?
    2. Provide the COMPLETE prediction function code as a Python function called
       `experimental_pred_fn(state, global_mult, fk_buckets)` that returns a (40,40,6) numpy array.
    3. The function must be SELF-CONTAINED — all logic inside the function body (it can
       call the imported utilities listed above).
    4. Give the change a short NAME (under 60 chars).

    Format your response EXACTLY like this:

    NAME: <short descriptive name>

    HYPOTHESIS: <1-3 sentences explaining why this should help>

    ```python
    def experimental_pred_fn(state, global_mult, fk_buckets):
        # Your complete implementation here
        # Must return np.ndarray of shape (40, 40, 6)
        ...
        return pred
    ```

    IMPORTANT: Write the complete function. Do not use placeholders or "..." for parts
    of the pipeline. Include all steps: calibration lookup, FK blending, multipliers,
    structural zeros, floor, and static cell locking. You can modify any step but must
    include all of them.
    """)


# ============================================================
# CODE EXTRACTION AND EXECUTION
# ============================================================

def extract_proposal(response_text: str) -> dict:
    """Extract name, hypothesis, and code from Gemini's response.

    Returns dict with keys: name, hypothesis, code, raw_response
    """
    result = {
        "name": "unknown",
        "hypothesis": "",
        "code": "",
        "raw_response": response_text,
    }

    # Extract NAME
    name_match = re.search(r"NAME:\s*(.+?)(?:\n|$)", response_text)
    if name_match:
        result["name"] = name_match.group(1).strip()[:60]

    # Extract HYPOTHESIS
    hyp_match = re.search(r"HYPOTHESIS:\s*(.+?)(?:\n\n|```)", response_text, re.DOTALL)
    if hyp_match:
        result["hypothesis"] = hyp_match.group(1).strip()[:500]

    # Extract code block
    code_blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if code_blocks:
        # Take the largest code block (most likely the full function)
        code = max(code_blocks, key=len)
        result["code"] = code.strip()
    else:
        # Try to find function definition without code fences
        func_match = re.search(
            r"(def experimental_pred_fn\(.*?\):.*?)(?:\n\n[A-Z]|\Z)",
            response_text,
            re.DOTALL,
        )
        if func_match:
            result["code"] = func_match.group(1).strip()

    return result


def compile_pred_fn(code: str):
    """Compile the extracted code into a callable prediction function.

    Returns the function object or raises an exception.
    """
    # Set up the execution namespace with all allowed imports
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

    try:
        exec(code, exec_globals)
    except Exception as e:
        raise RuntimeError(f"Code compilation failed: {e}\n\nCode:\n{code[:500]}")

    fn = exec_globals.get("experimental_pred_fn")
    if fn is None:
        raise RuntimeError(
            "Code does not define 'experimental_pred_fn'. "
            "Found names: " + str([k for k in exec_globals if not k.startswith("_") and k not in (
                "np", "numpy", "math", "CalibrationModel", "build_feature_keys",
                "MAP_H", "MAP_W", "NUM_CLASSES", "_build_coastal_mask",
                "_build_feature_key_index", "build_calibration_lookup",
                "build_fk_empirical_lookup", "FeatureKeyBuckets", "GlobalMultipliers",
                "terrain_to_class", "predict",
            )])
        )

    return fn


def validate_prediction(pred: np.ndarray, terrain: np.ndarray) -> list[str]:
    """Validate that a prediction array is well-formed."""
    errors = []

    if pred.shape != (MAP_H, MAP_W, NUM_CLASSES):
        errors.append(f"Shape mismatch: expected ({MAP_H},{MAP_W},{NUM_CLASSES}), got {pred.shape}")
        return errors

    if np.any(np.isnan(pred)):
        errors.append(f"NaN values found: {np.isnan(pred).sum()} cells")

    if np.any(np.isinf(pred)):
        errors.append(f"Inf values found: {np.isinf(pred).sum()} cells")

    if np.any(pred < 0):
        errors.append(f"Negative values: {(pred < 0).sum()} entries")

    row_sums = pred.sum(axis=-1)
    bad_sums = np.abs(row_sums - 1.0) > 0.02
    if bad_sums.any():
        worst = np.abs(row_sums - 1.0).max()
        errors.append(f"{bad_sums.sum()} cells don't sum to 1.0 (worst: {worst:.4f})")

    # Check hard rules
    dynamic = ~((terrain == 10) | (terrain == 5))
    if np.any(pred[dynamic, 5] > 0.001):
        count = (pred[dynamic, 5] > 0.001).sum()
        errors.append(f"Mountain class nonzero on {count} dynamic cells (hard rule violation)")

    ocean = terrain == 10
    if np.any(np.abs(pred[ocean, 0] - 1.0) > 0.01):
        errors.append("Ocean cells not set to [1,0,0,0,0,0]")

    mountain = terrain == 5
    if np.any(np.abs(pred[mountain, 5] - 1.0) > 0.01):
        errors.append("Mountain cells not set to [0,0,0,0,0,1]")

    return errors


# ============================================================
# BACKTEST HARNESS (wraps BacktestHarness with validation)
# ============================================================

class ResearchHarness:
    """Extended harness that validates predictions and catches errors."""

    def __init__(self, seeds_per_round: int = 5):
        self.inner = BacktestHarness(seeds_per_round=seeds_per_round)
        self.seeds_per_round = seeds_per_round

        # Also load round5 data if available
        r5_dir = DATA_DIR / "round5"
        if r5_dir.exists() and "round5" not in self.inner.round_data:
            try:
                detail = json.loads((r5_dir / "round_detail.json").read_text())
                seeds = []
                for si in range(min(seeds_per_round, 5)):
                    analysis = json.loads(
                        (r5_dir / f"analysis_seed_{si}.json").read_text()
                    )
                    gt = np.array(analysis["ground_truth"])
                    seeds.append({"state": detail["initial_states"][si], "gt": gt})

                rid = ROUND_IDS.get("round5", "")
                obs_dir = OBS_DIR / rid
                obs_files = sorted(obs_dir.glob("obs_s*_q*.json")) if obs_dir.exists() else []

                self.inner.round_data["round5"] = {
                    "detail": detail,
                    "seeds": seeds,
                    "obs_files": obs_files,
                }
                print(f"  Also loaded round5 ({len(seeds)} seeds, {len(obs_files)} obs files)")
            except Exception as e:
                print(f"  Warning: could not load round5: {e}")

    def get_round_names(self) -> list[str]:
        return list(self.inner.round_data.keys())

    def evaluate(self, pred_fn, timeout: float = 120.0) -> dict:
        """Evaluate with timeout and validation.

        Returns dict with per-round scores, avg, and any warnings.
        """
        round_names = self.get_round_names()
        results = {}
        warnings = []
        start = time.time()

        for test_round in round_names:
            if time.time() - start > timeout:
                warnings.append(f"Timeout after {timeout}s, skipping remaining rounds")
                break

            train_rounds = [r for r in round_names + ["round1"] if r != test_round]
            global_mult, fk_buckets = self.inner.build_obs_context(test_round, train_rounds)

            scores = []
            for si in range(len(self.inner.round_data[test_round]["seeds"])):
                seed_data = self.inner.round_data[test_round]["seeds"][si]
                try:
                    pred = pred_fn(seed_data["state"], global_mult, fk_buckets)
                except Exception as e:
                    raise RuntimeError(
                        f"Prediction failed on {test_round} seed {si}: {e}"
                    )

                # Validate
                terrain = np.array(seed_data["state"]["grid"], dtype=int)
                errs = validate_prediction(pred, terrain)
                if errs:
                    warnings.extend([f"{test_round}/s{si}: {e}" for e in errs])
                    # Try to fix common issues
                    pred = _auto_fix_prediction(pred, terrain)

                score = compute_score(seed_data["gt"], pred)
                scores.append(score)

            results[test_round] = float(np.mean(scores))

        if results:
            results["avg"] = float(np.mean([results[r] for r in results if r != "avg"]))

        return {"scores": results, "warnings": warnings, "elapsed": time.time() - start}


def _auto_fix_prediction(pred: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Attempt to fix common prediction issues."""
    pred = pred.copy()

    # Fix NaN/Inf
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

    # Fix negatives
    pred = np.maximum(pred, 0.0)

    # Renormalize rows that don't sum to 1
    row_sums = pred.sum(axis=-1, keepdims=True)
    zero_rows = row_sums.squeeze() < 1e-10
    pred = np.where(row_sums > 1e-10, pred / row_sums, 1.0 / NUM_CLASSES)

    # Enforce hard rules
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # Floor
    dp = pred[dynamic_mask]
    nz = dp > 0
    dp = np.where(nz, np.maximum(dp, 0.005), 0.0)
    dp_sum = dp.sum(axis=-1, keepdims=True)
    dp = np.where(dp_sum > 0, dp / dp_sum, 1.0 / NUM_CLASSES)
    pred[dynamic_mask] = dp

    # Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred


# ============================================================
# MAIN RESEARCH LOOP
# ============================================================

def run_research_loop(
    api_key: str,
    dry_run: bool = False,
    max_iters: int = 0,
    resume: bool = True,
    seeds_per_round: int = 5,
    model_name: str = "gemini-3-flash-preview",
):
    """Main research loop: propose -> implement -> backtest -> iterate."""
    client = init_gemini(api_key)
    _model_name = model_name

    log = ResearchLog()
    knowledge_base = load_knowledge_base()

    if not dry_run:
        print("Initializing backtest harness...")
        harness = ResearchHarness(seeds_per_round=seeds_per_round)

        # Run baseline if not done yet
        if log.best_score == 0:
            print("\nRunning baseline evaluation...")
            baseline_fn = make_baseline_pred_fn()
            baseline_result = harness.evaluate(baseline_fn)
            baseline_scores = baseline_result["scores"]
            log.best_score = baseline_scores.get("avg", 0)

            log.append({
                "id": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": "baseline (production pipeline)",
                "hypothesis": "Current production pipeline baseline",
                "scores": baseline_scores,
                "improvement": "baseline",
                "elapsed": baseline_result["elapsed"],
            })

            print(f"Baseline: avg={baseline_scores['avg']:.2f} "
                  f"({', '.join(f'{k}={v:.1f}' for k, v in baseline_scores.items() if k != 'avg')})")
    else:
        print("[DRY RUN] Skipping harness initialization")
        harness = None

    # Build system prompt
    system_prompt = build_system_prompt(knowledge_base, log.get_summary_text())

    iteration = log.count()
    improvements = 0
    consecutive_failures = 0

    print(f"\nStarting Gemini research loop (iteration {iteration})")
    print(f"Model: {model_name}")
    print(f"Best score: {log.best_score:.3f}")
    print(f"{'='*70}")

    try:
        while True:
            if max_iters > 0 and iteration >= max_iters + log.count() - len(log.entries):
                # Adjust: max_iters counts from start of this session
                if iteration - (log.count() - len(log.entries)) >= max_iters:
                    print(f"\nReached max iterations ({max_iters})")
                    break

            iteration_start = time.time()
            print(f"\n--- Experiment {iteration} ---")

            # Step 1: Ask Gemini for a proposal
            print("Asking Gemini for a proposal...")
            proposal_prompt = build_proposal_prompt(iteration, log.get_summary_text())

            try:
                from google.genai import types
                response = client.models.generate_content(
                    model=_model_name,
                    contents=[system_prompt, proposal_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.8 + min(0.7, consecutive_failures * 0.1),
                        max_output_tokens=8192,
                    ),
                )
                response_text = response.text
            except Exception as e:
                print(f"  Gemini API error: {e}")
                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": "api_error",
                    "error": f"Gemini API error: {str(e)[:200]}",
                })
                iteration += 1
                consecutive_failures += 1
                # Back off on API errors
                wait = min(60, 5 * consecutive_failures)
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            # Step 2: Extract proposal
            print("Extracting proposal...")
            proposal = extract_proposal(response_text)

            print(f"  Name: {proposal['name']}")
            print(f"  Hypothesis: {proposal['hypothesis'][:120]}...")

            if not proposal["code"]:
                print("  ERROR: No code extracted from response")
                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": proposal["name"],
                    "hypothesis": proposal["hypothesis"],
                    "error": "No code block found in Gemini response",
                    "proposal_summary": proposal["hypothesis"][:200],
                    "raw_response_preview": response_text[:500],
                })
                iteration += 1
                consecutive_failures += 1
                continue

            if dry_run:
                print(f"\n[DRY RUN] Would execute this code:\n")
                print(proposal["code"][:2000])
                if len(proposal["code"]) > 2000:
                    print(f"\n... ({len(proposal['code'])} chars total)")
                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": proposal["name"],
                    "hypothesis": proposal["hypothesis"],
                    "code_length": len(proposal["code"]),
                    "dry_run": True,
                    "proposal_summary": proposal["hypothesis"][:200],
                })
                iteration += 1
                continue

            # Step 3: Compile the prediction function
            print("Compiling prediction function...")
            try:
                pred_fn = compile_pred_fn(proposal["code"])
            except Exception as e:
                error_msg = str(e)[:300]
                print(f"  Compilation ERROR: {error_msg}")
                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": proposal["name"],
                    "hypothesis": proposal["hypothesis"],
                    "error": f"Compilation failed: {error_msg}",
                    "proposal_summary": proposal["hypothesis"][:200],
                })
                iteration += 1
                consecutive_failures += 1
                continue

            # Step 4: Backtest
            print("Running backtest...")
            try:
                result = harness.evaluate(pred_fn, timeout=180.0)
                scores = result["scores"]
                warnings = result["warnings"]
                elapsed = result["elapsed"]

                if warnings:
                    print(f"  Warnings: {len(warnings)}")
                    for w in warnings[:3]:
                        print(f"    - {w}")

                avg = scores.get("avg", 0)
                delta = avg - log.best_score

                # Determine improvement status
                if delta > 0.01:
                    improvement_str = f"+{delta:.3f} NEW BEST"
                    log.best_score = avg
                    improvements += 1
                elif delta > -0.5:
                    improvement_str = f"{delta:+.3f} (close)"
                else:
                    improvement_str = f"{delta:+.3f} (worse)"

                print(f"  Score: avg={avg:.2f} ({improvement_str})")
                print(f"  Per-round: {', '.join(f'{k}={v:.1f}' for k, v in scores.items() if k != 'avg')}")
                print(f"  Elapsed: {elapsed:.1f}s")

                if delta > 0.01:
                    print(f"\n  *** NEW BEST SCORE: {avg:.3f} ***")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": proposal["name"],
                    "hypothesis": proposal["hypothesis"],
                    "scores": scores,
                    "improvement": improvement_str,
                    "warnings": warnings[:5] if warnings else [],
                    "elapsed": round(elapsed, 1),
                    "proposal_summary": proposal["hypothesis"][:200],
                    "code_length": len(proposal["code"]),
                })

                # Save code of best experiments
                if delta > 0.01:
                    best_code_path = _PROJECT_DIR / "data" / f"gemini_best_{iteration}.py"
                    best_code_path.parent.mkdir(parents=True, exist_ok=True)
                    best_code_path.write_text(
                        f"# Gemini Research Experiment {iteration}\n"
                        f"# Name: {proposal['name']}\n"
                        f"# Hypothesis: {proposal['hypothesis']}\n"
                        f"# Score: avg={avg:.3f}\n"
                        f"# Delta: {delta:+.3f}\n\n"
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
                        f"{proposal['code']}\n",
                        encoding="utf-8",
                    )
                    print(f"  Saved best code to {best_code_path}")

            except Exception as e:
                error_msg = str(e)[:300]
                tb = traceback.format_exc()[-500:]
                print(f"  Backtest ERROR: {error_msg}")
                log.append({
                    "id": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": proposal["name"],
                    "hypothesis": proposal["hypothesis"],
                    "error": f"Backtest failed: {error_msg}",
                    "traceback": tb,
                    "proposal_summary": proposal["hypothesis"][:200],
                })
                consecutive_failures += 1

            iteration += 1

            # Periodic status
            if iteration % 5 == 0:
                total_time = time.time() - iteration_start
                print(f"\n{'='*70}")
                print(f"Status: {iteration} experiments, {improvements} improvements, "
                      f"best={log.best_score:.3f}")
                print(f"{'='*70}")

    except KeyboardInterrupt:
        print(f"\n\nStopped by user after {iteration} experiments")
        print(f"Improvements: {improvements}")
        print(f"Best score: {log.best_score:.3f}")
        print(f"Log: {LOG_PATH}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini-powered autonomous research agent for Astar Island"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show Gemini proposals without running backtests",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Maximum iterations (0 = unlimited)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing experiment log (default: True)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Seeds per round for backtesting (1=fast, 5=accurate)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print experiment log summary and exit",
    )
    args = parser.parse_args()

    if args.summary:
        log = ResearchLog()
        print(log.get_summary_text())
        return

    api_key = get_gemini_api_key()

    run_research_loop(
        api_key=api_key,
        dry_run=args.dry_run,
        max_iters=args.max_iters,
        resume=args.resume,
        seeds_per_round=args.seeds,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

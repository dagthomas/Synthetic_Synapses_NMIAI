#!/usr/bin/env python3
"""Multi-model autonomous research agent (Gemini-powered).

Uses Google Gemini models via google-genai SDK:
  - Gemini 3.1 Pro Preview: High-quality prediction code generation
  - Gemini 3 Flash Preview: Fast analysis, direction picking, code extraction

Orchestration:
  1. Flash analyzes experiment log + identifies promising direction (2-5s)
  2. Pro writes the prediction function code (10-30s)
  3. Backtest evaluates it (3s)
  4. Flash analyzes results and decides next direction (2-5s)
  5. Repeat

Usage:
    python multi_researcher.py                    # Run indefinitely
    python multi_researcher.py --max-iters 10     # Limited iterations
    python multi_researcher.py --summary          # Show results
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Load .env
_PROJECT_DIR = Path(__file__).parent
_env_path = _PROJECT_DIR / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(_PROJECT_DIR))

LOG_PATH = _PROJECT_DIR / "data" / "multi_research_log.jsonl"
KNOWLEDGE_PATH = _PROJECT_DIR / "KNOWLEDGE.md"


# ============================================================
# MODEL CLIENTS
# ============================================================

def call_gemini_pro(prompt: str, model: str = "gemini-3.1-pro-preview", timeout: int = 120) -> str:
    """Call Gemini Pro for high-quality code generation (replaces Claude Opus)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "ERROR: GOOGLE_API_KEY not set"
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini Pro call failed: {e}"


def call_flash(prompt: str, timeout: int = 30) -> str:
    """Call Gemini Flash for fast analysis (replaces Claude Haiku)."""
    return call_gemini(prompt, model="gemini-3-flash-preview", timeout=timeout)


def call_gemini(prompt: str, model: str = "gemini-3-flash-preview", timeout: int = 30) -> str:
    """Call Gemini via google-genai SDK."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "ERROR: GOOGLE_API_KEY not set"
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Gemini call failed: {e}"


# ============================================================
# BACKTEST HARNESS (singleton)
# ============================================================

_harness = None
_harness_data = None


def get_harness():
    global _harness, _harness_data
    if _harness is None:
        from autoexperiment import BacktestHarness
        _harness = BacktestHarness(seeds_per_round=5)

        # Also preload round data for the harness
        from calibration import CalibrationModel, build_feature_keys
        from utils import GlobalMultipliers, FeatureKeyBuckets, terrain_to_class
        from config import MAP_H, MAP_W
        import predict

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

        _harness_data = {}
        ALL_ROUNDS = ["round1", "round2", "round3", "round4", "round5", "round6", "round7", "round8", "round9", "round10", "round11", "round12"]
        for test_round, rid in ROUND_IDS.items():
            train_rounds = [r for r in ALL_ROUNDS if r != test_round]
            cal = CalibrationModel()
            for tr in train_rounds:
                cal.add_round(DATA_DIR / tr)

            detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
            obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

            gm = GlobalMultipliers()
            fk = FeatureKeyBuckets()
            seed_priors = []
            seed_fkeys = []
            for si in range(5):
                state = detail["initial_states"][si]
                predict._calibration_model = cal
                prior = predict.get_static_prior(state["grid"], state["settlements"])
                seed_priors.append(prior)
                fkeys = build_feature_keys(np.array(state["grid"], dtype=int), state["settlements"])
                seed_fkeys.append(fkeys)

            for op in obs_files:
                obs = json.loads(op.read_text())
                sid = obs["seed_index"]
                vp, grid = obs["viewport"], obs["grid"]
                for row in range(len(grid)):
                    for col in range(len(grid[0]) if grid else 0):
                        my, mx = vp["y"] + row, vp["x"] + col
                        if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                            oc = terrain_to_class(grid[row][col])
                            gm.add_observation(oc, seed_priors[sid][my, mx])
                            fk.add_observation(seed_fkeys[sid][my][mx], oc)

            gts = []
            for si in range(5):
                analysis = json.loads(
                    (DATA_DIR / test_round / f"analysis_seed_{si}.json").read_text()
                )
                gts.append(np.array(analysis["ground_truth"]))

            _harness_data[test_round] = {
                "cal": cal,
                "detail": detail,
                "gm": gm,
                "fk": fk,
                "gts": gts,
            }

    return _harness, _harness_data


def run_backtest(code: str) -> dict:
    """Compile and backtest a prediction function."""
    import math
    from calibration import CalibrationModel, build_feature_keys
    from config import MAP_H, MAP_W, NUM_CLASSES
    from fast_predict import (
        _build_coastal_mask, _build_feature_key_index,
        build_calibration_lookup, build_fk_empirical_lookup,
    )
    from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
    import predict

    # Compile — include ALL imports the code might need
    from scipy.ndimage import uniform_filter, distance_transform_cdt, gaussian_filter
    from config import TERRAIN_TO_CLASS, PROB_FLOOR

    namespace = {
        # Core
        "np": np, "numpy": np, "math": math,
        # Calibration
        "CalibrationModel": CalibrationModel, "build_feature_keys": build_feature_keys,
        # Config
        "MAP_H": MAP_H, "MAP_W": MAP_W, "NUM_CLASSES": NUM_CLASSES,
        "TERRAIN_TO_CLASS": TERRAIN_TO_CLASS, "PROB_FLOOR": PROB_FLOOR,
        # Fast predict
        "_build_coastal_mask": _build_coastal_mask,
        "_build_feature_key_index": _build_feature_key_index,
        "build_calibration_lookup": build_calibration_lookup,
        "build_fk_empirical_lookup": build_fk_empirical_lookup,
        # Utils
        "FeatureKeyBuckets": FeatureKeyBuckets,
        "GlobalMultipliers": GlobalMultipliers,
        "terrain_to_class": terrain_to_class,
        # Predict module
        "predict": predict,
        # Scipy
        "uniform_filter": uniform_filter,
        "distance_transform_cdt": distance_transform_cdt,
        "gaussian_filter": gaussian_filter,
        # Common Python builtins that LLMs use
        "_": None,  # throwaway variable
        "defaultdict": __import__("collections").defaultdict,
        # predict_gemini helpers
        "gemini_predict": __import__("predict_gemini").gemini_predict,
        "_load_params": __import__("predict_gemini")._load_params,
        "_DEFAULTS": __import__("predict_gemini")._DEFAULTS,
    }
    try:
        exec(code, namespace)
    except Exception as e:
        return {"error": f"Compilation failed: {e}", "scores": {"avg": 0}}

    pred_fn = namespace.get("experimental_pred_fn")
    if pred_fn is None:
        return {"error": "No experimental_pred_fn defined", "scores": {"avg": 0}}

    if pred_fn is None:
        return {"error": "No experimental_pred_fn found after compilation", "scores": {"avg": 0}}

    # Run backtest
    _, harness_data = get_harness()

    def compute_score(gt, pred):
        gt_safe = np.maximum(gt, 1e-10)
        entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
        dynamic = entropy > 0.01
        pred_safe = np.maximum(pred, 1e-10)
        kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
        if dynamic.any():
            wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
        else:
            wkl = 0.0
        return max(0, min(100, 100 * math.exp(-3 * wkl)))

    def safe_predict(pred_fn, state, gm, fk):
        """Wrapper that catches common LLM code errors and auto-fixes."""
        # Ensure grid is a plain list (not numpy) to prevent hashability issues
        # when LLM code passes grid rows as dict keys
        if isinstance(state.get("grid"), np.ndarray):
            state = dict(state)
            state["grid"] = state["grid"].tolist()
        try:
            pred = pred_fn(state, gm, fk)
        except TypeError as e:
            # Common: unhashable type when numpy array used as key
            if "unhashable" in str(e):
                # Force grid to plain list and retry
                state2 = dict(state)
                state2["grid"] = [[int(c) for c in row] for row in state["grid"]]
                pred = pred_fn(state2, gm, fk)
            else:
                raise

        # Fix: function returned None
        if pred is None:
            raise ValueError("Function returned None — missing return statement")

        # Fix: returned a list instead of array
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred, dtype=float)

        # Fix: wrong shape
        if pred.shape != (MAP_H, MAP_W, NUM_CLASSES):
            if pred.size == MAP_H * MAP_W * NUM_CLASSES:
                pred = pred.reshape(MAP_H, MAP_W, NUM_CLASSES)
            else:
                raise ValueError(f"Wrong shape: {pred.shape}")

        # Fix: NaN or Inf
        if np.isnan(pred).any() or np.isinf(pred).any():
            pred = np.nan_to_num(pred, nan=1.0/NUM_CLASSES, posinf=1.0, neginf=0.0)

        # Fix: negative values
        pred = np.maximum(pred, 0.0)

        # Fix: rows don't sum to 1
        row_sums = pred.sum(axis=-1, keepdims=True)
        bad = row_sums < 1e-6
        pred[bad.squeeze()] = 1.0 / NUM_CLASSES
        row_sums = pred.sum(axis=-1, keepdims=True)
        pred = pred / row_sums

        return pred

    scores = {}
    for test_round, data in harness_data.items():
        predict._calibration_model = data["cal"]
        seed_scores = []
        for si in range(5):
            state = data["detail"]["initial_states"][si]
            try:
                pred = safe_predict(pred_fn, state, data["gm"], data["fk"])
                if pred.shape != (40, 40, 6):
                    return {"error": f"Wrong shape: {pred.shape}", "scores": {"avg": 0}}
                if np.isnan(pred).any():
                    return {"error": "NaN in prediction", "scores": {"avg": 0}}
                seed_scores.append(compute_score(data["gts"][si], pred))
            except Exception as e:
                return {"error": f"Crashed on {test_round} seed {si}: {e}", "scores": {"avg": 0}}
        scores[test_round] = float(np.mean(seed_scores))

    scores["avg"] = float(np.mean([scores[r] for r in harness_data]))
    return {"scores": scores, "error": None}


# ============================================================
# EXPERIMENT LOG
# ============================================================

class MultiLog:
    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.entries = []
        self.best_score = 0.0
        self.best_code = ""
        self._load()

    def _load(self):
        if self.path.exists():
            for line in self.path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        e = json.loads(line)
                        self.entries.append(e)
                        if e.get("scores", {}).get("avg", 0) > self.best_score:
                            self.best_score = e["scores"]["avg"]
                            self.best_code = e.get("code", "")
                    except json.JSONDecodeError:
                        continue

    def append(self, entry: dict):
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent_summary(self, n: int = 10) -> str:
        recent = self.entries[-n:] if self.entries else []
        lines = [f"Total: {len(self.entries)} experiments, Best: {self.best_score:.2f}"]
        for e in recent:
            avg = e.get("scores", {}).get("avg", 0)
            err = e.get("error", "")
            model = e.get("model", "?")
            name = e.get("name", "?")[:40]
            status = f"avg={avg:.1f}" if not err else f"ERR: {err[:50]}"
            lines.append(f"  [{model[:5]}] {name:40s} {status}")
        return "\n".join(lines)

    def print_summary(self):
        if not self.entries:
            print("No experiments yet.")
            return
        ok = [e for e in self.entries if not e.get("error")]
        print(f"\nExperiments: {len(self.entries)} ({len(ok)} successful)")
        print(f"Best score: {self.best_score:.3f}")
        if ok:
            top = sorted(ok, key=lambda e: e["scores"]["avg"], reverse=True)[:5]
            print("Top 5:")
            for e in top:
                s = e["scores"]
                print(f"  [{e.get('model','?')[:5]}] {e.get('name','?')[:40]:40s} "
                      f"avg={s['avg']:.1f} R2={s.get('round2',0):.1f} R3={s.get('round3',0):.1f} "
                      f"R4={s.get('round4',0):.1f} R5={s.get('round5',0):.1f}")


# ============================================================
# PROMPTS
# ============================================================

def get_knowledge_summary() -> str:
    """Quick summary for context."""
    if KNOWLEDGE_PATH.exists():
        text = KNOWLEDGE_PATH.read_text()
        # Truncate to key sections
        return text[:4000]
    return "No KNOWLEDGE.md found"


def build_analysis_prompt(log: MultiLog) -> str:
    """Prompt for Flash-Lite to analyze results and suggest direction."""
    return f"""You are analyzing a probabilistic prediction pipeline for a Norse civilisation simulator.
This is NOT a neural network — it's a Bayesian prediction system using calibrated priors,
feature-key bucketed empirical data, and global multipliers.

The pipeline predicts a 40x40x6 probability tensor (6 classes: empty, settlement, port, ruin,
forest, mountain). Scoring: entropy-weighted KL divergence.

Current best score: {log.best_score:.2f} (theoretical max ~98)
Recent experiments:
{log.get_recent_summary(15)}

Key error sources (from R5 analysis):
- Settlement class: 4.5% of KL (dominant)
- Port underpredicted on coastal cells
- Empty/Forest ratio errors near settlements

Already tried and FAILED (DO NOT suggest):
- Coastal settlement penalty, proximity multipliers, per-class distance blend
- Settlement-adjacent empty suppression, log-odds blending, cross-class coupling
- Observation-count-weighted blending (too aggressive)

The baseline scores 91.8. Most modifications score LOWER. Suggest ONE SMALL, CONSERVATIVE
change — a single parameter tweak or a minor formula adjustment within the existing pipeline.
Examples of the right scale:
- Change the smoothing kernel size from 3 to 5
- Adjust the temperature radius formula
- Modify one clamp range (e.g., settlement clamp from [0.15, 2.5] to [0.10, 3.0])
- Change the FK blend normalization from (pw+ew) to max(pw,ew)
- Adjust calibration weight for one level (e.g., coarse_max from 2.0 to 3.0)

Reply in 1-2 sentences. ONE small change."""


def build_code_prompt(direction: str, log: MultiLog) -> str:
    """Build a code-printer prompt for Gemini Pro with working template."""
    best_code = log.best_code[:3500] if log.best_code else ""

    return f"""You are a code printer. Print ONLY a complete Python function.
No explanations. No markdown. Your entire response is one function.

MODIFICATION TO MAKE: {direction}

ORIGINAL FUNCTION TO MODIFY:
{best_code}

CRITICAL RULES:
- state['grid'] is list[list[int]], convert: grid = np.array(state['grid'])
- Feature keys are TUPLES (hashable). Never use numpy arrays as dict keys.
- MUST return a np.ndarray of shape (40, 40, 6). Never return None.
- mountain(class 5) = 0 on non-mountain cells. port(class 2) = 0 on non-coastal.
- floor >= 0.005 for nonzero classes. All rows sum to 1.0.
- Do NOT import anything. All available in scope: np, math, build_feature_keys,
  MAP_H, MAP_W, NUM_CLASSES, _build_coastal_mask, _build_feature_key_index,
  build_calibration_lookup, build_fk_empirical_lookup, GlobalMultipliers,
  FeatureKeyBuckets, terrain_to_class, predict, uniform_filter,
  distance_transform_cdt, gaussian_filter, TERRAIN_TO_CLASS.
- Do NOT create new global variables or reference undefined names.
- ALWAYS end with 'return probs' (the (40,40,6) array).

YOUR RESPONSE:"""


def extract_code_gemini(response: str) -> str:
    """Use Gemini Flash to extract clean Python from a response."""
    extract_prompt = f"""Extract the Python function from this text.
Return ONLY raw Python code starting with 'def experimental_pred_fn'.
No markdown backticks. No explanation before or after. Just the function.

{response[:4000]}"""

    try:
        code = call_gemini(extract_prompt, model="gemini-3-flash-preview")
        if code.startswith("ERROR"):
            # Retry on error
            code = call_gemini(extract_prompt, model="gemini-3-flash-preview")
            if code.startswith("ERROR"):
                return _extract_code_fallback(response)
        # Clean markdown if Flash added any
        if "```" in code:
            parts = code.split("```")
            for i in range(1, len(parts), 2):
                part = parts[i].strip()
                if part.startswith("python\n"):
                    part = part[7:]
                if "def experimental_pred_fn" in part:
                    return part.strip()
        if "def experimental_pred_fn" in code:
            idx = code.index("def experimental_pred_fn")
            code = code[idx:].strip()
        else:
            code = code.strip()

        # Verify code has a return statement
        if "return " not in code:
            # Retry extraction
            code2 = call_gemini(extract_prompt, model="gemini-3-flash-preview")
            if "return " in code2:
                code = code2.strip()
                if "```" in code:
                    parts = code.split("```")
                    for i in range(1, len(parts), 2):
                        part = parts[i].strip()
                        if part.startswith("python\n"):
                            part = part[7:]
                        if "def experimental_pred_fn" in part:
                            code = part.strip()
                            break
                if "def experimental_pred_fn" in code:
                    code = code[code.index("def experimental_pred_fn"):].strip()

        # Safety: ensure extracted code ends with return probs
        if "def experimental_pred_fn" in code and "return probs" not in code:
            code = code.rstrip() + "\n    return probs"
        return code
    except Exception:
        return _extract_code_fallback(response)


def _extract_code_fallback(response: str) -> str:
    """Fallback: extract code without Gemini."""
    response = response.encode("ascii", errors="ignore").decode("ascii")

    # Try to find code block
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
            if "def experimental_pred_fn" in code:
                return code
    if "```" in response:
        parts = response.split("```")
        for i in range(1, len(parts), 2):  # odd indices are code blocks
            code = parts[i].strip()
            if code.startswith("python\n"):
                code = code[7:]
            if "def experimental_pred_fn" in code:
                return code

    # Try to find function definition directly
    if "def experimental_pred_fn" in response:
        idx = response.index("def experimental_pred_fn")
        code = response[idx:]
        # Trim trailing non-code text
        lines = code.split("\n")
        clean_lines = []
        found_body = False
        for line in lines:
            clean_lines.append(line)
            # Track if we've seen function body (skip early returns in guards)
            if not found_body and line.strip() and not line.strip().startswith("def "):
                found_body = True
            # Stop at 'return probs' at function level (the canonical final return)
            if found_body and line.strip() == "return probs":
                break
            # Stop at any function-level return, but only after 20+ lines of body
            if (found_body and len(clean_lines) > 20
                    and line.strip().startswith("return ")
                    and not line.startswith("    " * 2)):
                break
        result = "\n".join(clean_lines)
        # Safety: ensure function ends with return probs
        if "return probs" not in result.split("\n")[-1]:
            result += "\n    return probs"
        return result

    return response.strip()


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-model research agent")
    parser.add_argument("--max-iters", type=int, default=0, help="Max iterations (0=unlimited)")
    parser.add_argument("--haiku-only", action="store_true", help="(deprecated, ignored)")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    args = parser.parse_args()

    log = MultiLog()

    if args.summary:
        log.print_summary()
        return

    # Pre-load harness
    print("Loading backtest harness...")
    get_harness()
    print("Harness ready.\n")

    # If no best code yet, use Gemini's best as starting point
    if not log.best_code:
        best_file = _PROJECT_DIR / "predict_gemini.py"
        if best_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("pg", str(best_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            import inspect
            log.best_code = inspect.getsource(mod.gemini_predict)
            # Quick baseline
            print("Running baseline backtest...")
            # Use the gemini predict as baseline
            code = f"""
import numpy as np
import math
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict
from scipy.ndimage import uniform_filter, distance_transform_cdt

{log.best_code.replace('gemini_predict', 'experimental_pred_fn')}
"""
            result = run_backtest(code)
            if result.get("scores", {}).get("avg", 0) > 0:
                log.best_score = result["scores"]["avg"]
                log.best_code = code
                s = result["scores"]
                print(f"Baseline: avg={s['avg']:.1f} R2={s.get('round2',0):.1f} "
                      f"R3={s.get('round3',0):.1f} R4={s.get('round4',0):.1f} R5={s.get('round5',0):.1f}")
            else:
                print(f"Baseline failed: {result.get('error', '?')}")

    iteration = 0
    last_error = ""  # Track last error to feed back to Flash

    print(f"\n{'='*70}")
    print(f"Multi-Model Research Agent (Gemini-powered)")
    print(f"  Gemini 3.1 Pro:   ON (code generation)")
    print(f"  Gemini 3 Flash:   ON (analysis / direction / extraction)")
    print(f"  Best score: {log.best_score:.2f}")
    print(f"  Max iterations: {args.max_iters or 'unlimited'}")
    print(f"{'='*70}\n")

    try:
        while True:
            if args.max_iters > 0 and iteration >= args.max_iters:
                print(f"\nReached max iterations ({args.max_iters}).")
                break

            iteration += 1
            t_start = time.time()
            print(f"--- Iteration {iteration} ---")

            # Step 1: Flash analyzes and suggests direction
            print("  [flash] Analyzing experiments...")
            analysis_prompt = build_analysis_prompt(log)
            if last_error:
                analysis_prompt += f"\n\nLAST EXPERIMENT CRASHED: {last_error[:200]}\nAvoid this pattern."
                last_error = ""
            t0 = time.time()
            direction = call_flash(analysis_prompt, timeout=30)
            t_analysis = time.time() - t0
            print(f"  [flash] ({t_analysis:.1f}s) Direction: {direction[:120]}")

            if direction.startswith("ERROR"):
                print(f"  Flash failed, retrying...")
                direction = call_flash(build_analysis_prompt(log), timeout=45)
                if direction.startswith("ERROR"):
                    direction = "Improve the empty/forest ratio prediction on high-entropy cells near settlements"

            # Step 2: Generate code with Gemini Pro (best quality)
            print("  [pro] Writing prediction code...")
            t0 = time.time()
            response = call_gemini_pro(build_code_prompt(direction, log),
                                       model="gemini-3.1-pro-preview", timeout=300)
            t_code = time.time() - t0
            code_model = "pro"

            if response.startswith("ERROR"):
                print(f"  Code generation failed: {response[:100]}")
                log.append({
                    "id": iteration, "name": direction[:60], "model": code_model,
                    "error": response[:200], "scores": {"avg": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # Stage 2: Gemini Flash-Lite extracts clean code from Claude's response
            print("  [flash] Extracting code...")
            t_extract = time.time()
            code = extract_code_gemini(response)
            t_extract = time.time() - t_extract
            print(f"  [{code_model}] ({t_code:.1f}s code + {t_extract:.1f}s extract) {len(code)} chars")

            # Step 3: Backtest
            print("  [backtest] Running...")
            t0 = time.time()
            result = run_backtest(code)
            t_bt = time.time() - t0

            scores = result.get("scores", {})
            avg = scores.get("avg", 0)
            error = result.get("error")

            entry = {
                "id": iteration,
                "name": direction[:60],
                "model": code_model,
                "scores": {k: round(v, 2) for k, v in scores.items()},
                "error": error,
                "code": code,  # Always save code for review
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timings": {
                    "analysis": round(t_analysis, 1),
                    "code": round(t_code, 1),
                    "backtest": round(t_bt, 1),
                    "total": round(time.time() - t_start, 1),
                },
            }
            log.append(entry)

            # Save promising code to separate files for review
            ideas_dir = _PROJECT_DIR / "data" / "multi_ideas"
            ideas_dir.mkdir(exist_ok=True)
            if avg > 0:
                # Save all scored experiments
                tag = "good" if avg >= log.best_score - 1.0 else "ok"
                fname = f"idea_{iteration:04d}_{tag}_{avg:.1f}.py"
                (ideas_dir / fname).write_text(f"# Score: avg={avg:.2f}\n# Direction: {direction[:100]}\n# Error: {error or 'None'}\n\n{code}")
            elif error and len(code) > 100:
                # Save failed but promising code
                fname = f"idea_{iteration:04d}_failed.py"
                (ideas_dir / fname).write_text(f"# FAILED: {error[:150]}\n# Direction: {direction[:100]}\n\n{code}")

            if error:
                print(f"  [backtest] ({t_bt:.1f}s) FAILED: {error[:100]}")
                last_error = error[:200]  # Feed back to Haiku
            elif avg > log.best_score:
                log.best_score = avg
                log.best_code = code
                s = scores
                print(f"  [backtest] ({t_bt:.1f}s) ***NEW BEST: {avg:.2f}*** "
                      f"R2={s.get('round2',0):.1f} R3={s.get('round3',0):.1f} "
                      f"R4={s.get('round4',0):.1f} R5={s.get('round5',0):.1f}")
                # Save best code
                save_path = _PROJECT_DIR / "data" / "experiments" / f"multi_best_{iteration}.py"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(code)
            else:
                print(f"  [backtest] ({t_bt:.1f}s) avg={avg:.1f} (best={log.best_score:.1f})")

            total = time.time() - t_start
            print(f"  Total: {total:.1f}s\n")

            time.sleep(2)  # Brief pause between iterations

    except KeyboardInterrupt:
        print(f"\n\nStopped after {iteration} iterations.")

    log.print_summary()


if __name__ == "__main__":
    main()

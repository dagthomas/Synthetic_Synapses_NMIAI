"""Batch eval runner for Tripletex agent.

Tests classifier accuracy and optionally runs full agent evals.

Usage:
    python run_all_evals.py                        # all tasks, Norwegian
    python run_all_evals.py --langs no en          # multi-language
    python run_all_evals.py --tasks create_employee create_invoice
    python run_all_evals.py --skip-broken          # skip sandbox_broken
    python run_all_evals.py --classifier-only      # only test classifier
    python run_all_evals.py --agent-url http://host:8000
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from sim.task_definitions import ALL_TASKS, LANGUAGES
from sim.generator import generate_task
from sim.scorer import calculate_score
from tool_router import classify_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_all_evals")


def test_classifier(task_name: str, prompt: str) -> dict:
    """Test if classify_task returns the correct task type for this prompt."""
    predicted = classify_task(prompt)
    correct = predicted == task_name
    return {
        "task": task_name,
        "predicted": predicted or "(fallback)",
        "correct": correct,
    }


def run_agent_eval(agent_url: str, task_name: str, lang: str, base_url: str, token: str) -> dict:
    """Run a full agent eval: generate -> call agent -> verify -> score."""
    import requests
    from tripletex_client import TripletexClient
    from sim.verifier import verify_task
    from simulator import pre_create_for_deletion

    task_def = ALL_TASKS[task_name]

    # Generate task
    generated = generate_task(task_def, language=lang)
    prompt = generated["prompt"]
    expected = generated["expected"]

    # Classifier test
    classifier_result = test_classifier(task_name, prompt)

    # Pre-create for deletion/reversal tasks
    client = TripletexClient(base_url, token)
    pre_created_id = 0
    if task_def.pre_create:
        pre_created_id = pre_create_for_deletion(client, task_def, expected)

    # Call agent
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {"base_url": base_url, "session_token": token},
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("AGENT_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    try:
        resp = requests.post(
            f"{agent_url}/solve-debug", params={"source": "eval"},
            json=payload, headers=headers, timeout=300,
        )
        elapsed = time.time() - t0
        body = resp.json() if resp.status_code == 200 else {"error": resp.text[:500]}
    except requests.exceptions.ConnectionError:
        return {
            "task": task_name, "lang": lang, "status": "ERROR",
            "error": f"Cannot connect to agent at {agent_url}",
            "classifier": classifier_result,
        }
    except Exception as e:
        return {
            "task": task_name, "lang": lang, "status": "ERROR",
            "error": str(e), "classifier": classifier_result,
        }

    time.sleep(1)  # API propagation

    # Verify
    verify_client = TripletexClient(base_url, token)
    verification = verify_task(verify_client, task_def, expected, pre_created_id)

    api_calls = body.get("api_calls", task_def.baseline_calls)
    api_errors = body.get("api_errors", 0)
    score = calculate_score(
        verification["total_points"], verification["max_points"],
        task_def.tier, api_calls, api_errors, task_def.baseline_calls,
    )

    status = "PASS" if score["correctness"] >= 1.0 else (
        "PARTIAL" if score["correctness"] > 0 else "FAIL"
    )

    return {
        "task": task_name,
        "lang": lang,
        "status": status,
        "correctness": score["correctness"],
        "final_score": score["final_score"],
        "max_possible": score["max_possible"],
        "api_calls": api_calls,
        "api_errors": api_errors,
        "elapsed": round(elapsed, 1),
        "classifier": classifier_result,
        "checks": verification.get("checks", []),
    }


def print_summary(results: list):
    """Print formatted summary table."""
    print()
    print("=" * 90)
    print(f"  {'Task':<30} {'Lang':<5} {'Status':<8} {'Score':<12} {'Classifier':<12} {'Time':<6}")
    print("-" * 90)

    classifier_misses = []
    for r in results:
        clf = r.get("classifier", {})
        clf_status = "OK" if clf.get("correct") else "MISS"
        if not clf.get("correct"):
            classifier_misses.append(r)

        score_str = ""
        if "final_score" in r:
            score_str = f"{r['final_score']:.1f}/{r['max_possible']:.1f}"
        elif r.get("status") == "SKIP":
            score_str = "-"

        print(f"  {r['task']:<30} {r.get('lang', '-'):<5} {r['status']:<8} "
              f"{score_str:<12} {clf_status:<12} {r.get('elapsed', '-')!s:<6}")

    print("=" * 90)

    # Summary stats
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    partial = sum(1 for r in results if r.get("status") == "PARTIAL")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "ERROR")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    clf_ok = sum(1 for r in results if r.get("classifier", {}).get("correct"))
    clf_total = sum(1 for r in results if "classifier" in r and r.get("status") != "SKIP")

    print(f"\n  Results: {passed} PASS, {partial} PARTIAL, {failed} FAIL, {errors} ERROR, {skipped} SKIP")
    if clf_total:
        print(f"  Classifier: {clf_ok}/{clf_total} correct ({clf_ok/clf_total:.0%})")

    if classifier_misses:
        print(f"\n  Classifier misses:")
        for r in classifier_misses:
            clf = r["classifier"]
            print(f"    {r['task']:<30} predicted={clf['predicted']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Batch eval runner")
    parser.add_argument("--tasks", nargs="+", help="Task types to test (default: all)")
    parser.add_argument("--langs", nargs="+", default=["no"], help="Languages (default: no)")
    parser.add_argument("--skip-broken", action="store_true", help="Skip sandbox_broken tasks")
    parser.add_argument("--classifier-only", action="store_true",
                        help="Only test classifier accuracy (no agent calls)")
    parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent URL")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Determine tasks
    task_names = args.tasks or list(ALL_TASKS.keys())
    invalid = [t for t in task_names if t not in ALL_TASKS]
    if invalid:
        print(f"Unknown tasks: {', '.join(invalid)}")
        print(f"Available: {', '.join(ALL_TASKS.keys())}")
        sys.exit(1)

    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    need_agent = not args.classifier_only

    if need_agent and (not base_url or not token):
        print("ERROR: Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env")
        sys.exit(1)

    results = []
    total = sum(1 for t in task_names for _ in args.langs)
    done = 0

    for task_name in task_names:
        task_def = ALL_TASKS[task_name]

        if args.skip_broken and task_def.sandbox_broken:
            results.append({"task": task_name, "lang": "-", "status": "SKIP",
                            "classifier": {"task": task_name, "predicted": "-", "correct": True}})
            done += len(args.langs)
            continue

        for lang in args.langs:
            done += 1
            print(f"[{done}/{total}] {task_name} ({lang})...", end=" ", flush=True)

            if args.classifier_only:
                # Generate prompt and test classifier only
                try:
                    generated = generate_task(task_def, language=lang)
                    clf = test_classifier(task_name, generated["prompt"])
                    result = {
                        "task": task_name, "lang": lang,
                        "status": "PASS" if clf["correct"] else "FAIL",
                        "classifier": clf,
                    }
                    if not clf["correct"]:
                        result["prompt"] = generated["prompt"][:300]
                    results.append(result)
                    print("OK" if clf["correct"] else f"MISS (predicted={clf['predicted']})")
                except Exception as e:
                    results.append({
                        "task": task_name, "lang": lang, "status": "ERROR",
                        "error": str(e),
                        "classifier": {"task": task_name, "predicted": "error", "correct": False},
                    })
                    print(f"ERROR: {e}")
            else:
                try:
                    r = run_agent_eval(args.agent_url, task_name, lang, base_url, token)
                    results.append(r)
                    clf = r.get("classifier", {})
                    clf_tag = " [clf:MISS]" if not clf.get("correct") else ""
                    print(f"{r['status']} ({r.get('final_score', 0):.1f}/{r.get('max_possible', 0):.1f}){clf_tag}")
                except Exception as e:
                    results.append({
                        "task": task_name, "lang": lang, "status": "ERROR",
                        "error": str(e),
                        "classifier": {"task": task_name, "predicted": "error", "correct": False},
                    })
                    print(f"ERROR: {e}")

    # Print summary
    print_summary(results)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

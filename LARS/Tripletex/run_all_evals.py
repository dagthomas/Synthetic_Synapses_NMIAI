"""Run all eval tasks sequentially and collect results."""

import json
import os
import subprocess
import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TASKS = [
    "create_employee", "create_customer", "create_product",
    "create_department", "create_supplier", "create_contact",
    "update_employee", "update_customer", "update_product",
    "update_supplier", "update_department", "update_contact",
    "create_invoice", "create_multi_line_invoice", "create_project",
    "create_travel_expense", "invoice_with_payment", "create_credit_note",
    "create_employee_with_employment", "create_supplier_invoice",
    "create_travel_expense_with_costs", "create_project_with_pm",
    "delete_travel_expense", "delete_customer", "create_ledger_voucher",
    "reverse_voucher", "delete_invoice", "create_opening_balance",
    "delete_supplier", "delete_product", "delete_department",
    "delete_contact", "delete_employee",
]

def run_task(task_name):
    start = time.time()
    result = subprocess.run(
        [sys.executable, "simulator.py", "--task", task_name, "--lang", "no", "--no-cleanup"],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - start
    output = result.stdout + result.stderr

    score_line = ""
    correctness_line = ""
    status = "UNKNOWN"
    field_results = []
    
    for line in output.split(chr(10)):
        if "Final score:" in line:
            score_line = line.strip()
        if "Correctness:" in line:
            correctness_line = line.strip()
        if "Skipping" in line and "sandbox_broken" in line:
            status = "SKIPPED"
        if "[  OK]" in line or "[FAIL]" in line:
            field_results.append(line.strip())

    if status != "SKIPPED":
        if score_line:
            try:
                parts = score_line.split("Final score:")[-1].strip().split("/")
                final = float(parts[0].strip())
                max_score = float(parts[1].strip())
                if final >= max_score * 0.99:
                    status = "PERFECT"
                elif final > 0:
                    status = "PARTIAL"
                else:
                    status = "FAIL"
            except Exception:
                status = "UNKNOWN"
        elif result.returncode != 0:
            status = "ERROR"

    return {
        "task": task_name,
        "status": status,
        "score_line": score_line,
        "correctness_line": correctness_line,
        "field_results": field_results,
        "elapsed": round(elapsed, 1),
        "returncode": result.returncode,
        "output": output,
    }


def main():
    results = []
    total = len(TASKS)

    print(f"Running {total} eval tasks...")
    print()
    print(f"{'#':>3} {'Task':<35} {'Status':<10} {'Score':<20} {'Time':>6}")
    print("-" * 80)

    for i, task in enumerate(TASKS, 1):
        r = run_task(task)
        results.append(r)
        score_display = r["score_line"].replace("Final score:", "").strip() if r["score_line"] else "N/A"
        print(f"{i:>3} {task:<35} {r['status']:<10} {score_display:<20} {r['elapsed']:>5.0f}s")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    perfect = [r for r in results if r["status"] == "PERFECT"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    failed = [r for r in results if r["status"] in ("FAIL", "ERROR", "UNKNOWN")]
    skipped = [r for r in results if r["status"] == "SKIPPED"]

    print(f"  Perfect:  {len(perfect)}/{total}")
    print(f"  Partial:  {len(partial)}/{total}")
    print(f"  Failed:   {len(failed)}/{total}")
    print(f"  Skipped:  {len(skipped)}/{total}")

    if failed:
        print()
        print("FAILED TASKS:")
        for r in failed:
            print(f"  - {r['task']}: {r['status']}")

    if partial:
        print()
        print("PARTIAL TASKS (less than perfect):")
        for r in partial:
            print(f"  - {r['task']}: {r['score_line']}")
            for fl in r["field_results"]:
                if "FAIL" in fl:
                    print(f"      {fl}")

    basedir = os.path.dirname(os.path.abspath(__file__))
    fail_log = os.path.join(basedir, "eval_failures.log")
    with open(fail_log, "w", encoding="utf-8") as f:
        for r in results:
            if r["status"] in ("FAIL", "ERROR", "UNKNOWN", "PARTIAL"):
                f.write(chr(10) + "=" * 80 + chr(10))
                f.write(f"TASK: {r['task']} -- STATUS: {r['status']}" + chr(10))
                f.write("=" * 80 + chr(10))
                f.write(r["output"])
                f.write(chr(10))

    summary_file = os.path.join(basedir, "eval_results.json")
    with open(summary_file, "w") as f:
        summary = [{k: v for k, v in r.items() if k != "output"} for r in results]
        json.dump(summary, f, indent=2)

    print()
    print(f"Results: {summary_file}")
    print(f"Failures: {fail_log}")


if __name__ == "__main__":
    main()

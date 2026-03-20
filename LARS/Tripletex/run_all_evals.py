"""Run all evals sequentially and produce a summary."""
import subprocess
import re
import sys

TASKS = [
    # Tier 1
    "create_employee", "create_customer", "create_product", "create_department",
    "create_supplier", "create_contact", "update_employee", "update_customer",
    "update_product", "update_supplier",
    # Tier 2 (skip create_supplier_invoice — broken)
    "create_invoice", "create_multi_line_invoice", "create_project",
    "create_travel_expense", "invoice_with_payment", "create_credit_note",
    "create_employee_with_employment", "create_travel_expense_with_costs",
    "create_project_with_pm",
    # Tier 3 (skip delete_employee — broken)
    "delete_travel_expense", "delete_customer", "create_ledger_voucher",
    "reverse_voucher", "delete_invoice", "create_opening_balance",
    "delete_supplier", "delete_product", "delete_department", "delete_contact",
]

AGENT_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9000"

results = []
for i, task in enumerate(TASKS):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(TASKS)}] Running: {task}")
    print(f"{'='*60}")
    try:
        out = subprocess.run(
            ["python", "simulator.py", "--task", task, "--agent-url", AGENT_URL],
            capture_output=True, text=True, timeout=300,
            cwd=r"C:\Users\larsh\source\repos\AINM\Tripletex",
        )
        output = out.stdout + out.stderr
        # Extract correctness and final score
        correctness_m = re.search(r"Correctness:\s+[\d/]+ = (\d+)%", output)
        score_m = re.search(r"Final score:\s+([\d.]+)\s*/\s*([\d.]+)", output)
        tier_m = re.search(r"Tier (\d)", output)
        correctness = int(correctness_m.group(1)) if correctness_m else -1
        final_score = float(score_m.group(1)) if score_m else 0.0
        max_score = float(score_m.group(2)) if score_m else 0.0
        tier = int(tier_m.group(1)) if tier_m else 0
        results.append({
            "task": task, "tier": tier,
            "correctness": correctness, "score": final_score, "max": max_score,
        })
        status = "PASS" if correctness == 100 else "FAIL"
        print(f"  => {status}: {correctness}% correctness, score {final_score}/{max_score}")
    except subprocess.TimeoutExpired:
        results.append({"task": task, "tier": 0, "correctness": -1, "score": 0, "max": 0})
        print(f"  => TIMEOUT")
    except Exception as e:
        results.append({"task": task, "tier": 0, "correctness": -1, "score": 0, "max": 0})
        print(f"  => ERROR: {e}")

# Summary
print(f"\n\n{'='*70}")
print(f"  COMPREHENSIVE EVAL SUMMARY")
print(f"{'='*70}")
passed = [r for r in results if r["correctness"] == 100]
failed = [r for r in results if 0 <= r["correctness"] < 100]
errors = [r for r in results if r["correctness"] < 0]
total_score = sum(r["score"] for r in results)
total_max = sum(r["max"] for r in results)

print(f"  Total tasks:   {len(results)}")
print(f"  Passed (100%): {len(passed)}/{len(results)}")
print(f"  Failed:        {len(failed)}")
print(f"  Errors/TO:     {len(errors)}")
print(f"  Total score:   {total_score:.2f} / {total_max:.2f}")
print(f"  Skipped:       2 (create_supplier_invoice, delete_employee — sandbox broken)")

for tier in [1, 2, 3]:
    tier_results = [r for r in results if r["tier"] == tier]
    if tier_results:
        tier_passed = sum(1 for r in tier_results if r["correctness"] == 100)
        print(f"\n  Tier {tier}: {tier_passed}/{len(tier_results)} passed")
        for r in tier_results:
            status = "OK" if r["correctness"] == 100 else "FAIL"
            print(f"    [{status:4}] {r['task']:<30} {r['correctness']:>3}%  {r['score']:.1f}/{r['max']:.1f}")

if failed:
    print(f"\n  FAILURES:")
    for r in failed:
        print(f"    - {r['task']}: {r['correctness']}%")

print(f"{'='*70}")

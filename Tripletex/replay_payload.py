"""
Replay a saved payload through /solve-debug and generate a report.

Usage:
    python replay_payload.py payloads/20260320_010505_402543f0.json
    python replay_payload.py payloads/  # replay ALL payloads in folder
    python replay_payload.py payloads/20260320_01*.json  # glob pattern
    python replay_payload.py --list payloads/  # list payloads without running
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import date, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8003/solve-debug")
BASE_URL = os.environ["TRIPLETEX_BASE_URL"]
TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]
API_KEY = os.environ.get("AGENT_API_KEY", "")


def load_payload(path: str) -> dict:
    """Load a payload JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def replay(payload: dict, timeout: int = 180) -> dict:
    """Send payload to /solve-debug with live credentials."""
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # Override credentials with live token
    payload["tripletex_credentials"] = {
        "base_url": BASE_URL,
        "session_token": TOKEN,
    }

    t0 = time.time()
    resp = requests.post(AGENT_URL, json=payload, headers=headers, timeout=timeout)
    elapsed = time.time() - t0
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    return {"status_code": resp.status_code, "elapsed": elapsed, "body": body}


def print_result(path: str, resp: dict):
    """Print a detailed result for one payload."""
    body = resp["body"]
    tool_calls = body.get("tool_calls", [])
    api_calls = body.get("api_calls", 0)
    api_errors = body.get("api_errors", 0)

    failed_tools = [tc for tc in tool_calls if tc.get("result", {}).get("ok") is False]
    failed_apis = [a for a in body.get("api_log", []) if not a.get("ok")]

    status = "FAIL" if failed_tools or resp["status_code"] != 200 else "OK"

    print(f"\n{'='*60}")
    print(f"[{status}] {os.path.basename(path)}")
    print(f"  Prompt: {body.get('_prompt', '?')[:100]}...")
    print(f"  {resp['elapsed']:.1f}s | {len(tool_calls)} tools | {api_calls} API | {api_errors} errors")

    for tc in tool_calls:
        result = tc.get("result", {}) or {}
        ok = result.get("ok", "?")
        icon = "OK" if ok is True else "FAIL" if ok is False else "?"
        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in tc.get("args", {}).items())
        print(f"    [{icon}] {tc['tool']}({args_str})")
        if ok is False and result.get("error"):
            print(f"           ERROR: {result['error'][:200]}")

    for api in failed_apis:
        print(f"    [API FAIL] {api['method']} {api['url']} -> {api['status']}: {api.get('error', '')[:150]}")

    return {
        "file": os.path.basename(path),
        "status": status,
        "elapsed": resp["elapsed"],
        "tool_calls": tool_calls,
        "api_calls": api_calls,
        "api_errors": api_errors,
        "api_log": body.get("api_log", []),
        "agent_response": body.get("agent_response", ""),
        "prompt": body.get("_prompt", ""),
    }


def generate_report(all_results: list, payload_data: list) -> str:
    """Generate markdown report for all replayed payloads."""
    today = date.today().isoformat()
    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    total = len(all_results)
    total_time = sum(r["elapsed"] for r in all_results)
    total_api = sum(r.get("api_calls", 0) for r in all_results)
    total_errors = sum(r.get("api_errors", 0) for r in all_results)

    lines = []
    lines.append(f"# Payload Replay Report — {today}")
    lines.append("")
    lines.append(f"**Pass rate: {ok_count}/{total}** | Total: {total_time:.0f}s | API calls: {total_api} | API errors: {total_errors}")
    lines.append("")

    # Summary table
    lines.append("| # | Payload | Status | Time | Tools | API Calls | API Errors |")
    lines.append("|---|---------|--------|------|-------|-----------|------------|")
    for i, (r, p) in enumerate(zip(all_results, payload_data), 1):
        prompt_short = p.get("prompt", "")[:50]
        lines.append(f"| {i} | {r['file']} | {'PASS' if r['status'] == 'OK' else 'FAIL'} | {r['elapsed']:.0f}s | {len(r['tool_calls'])} | {r.get('api_calls', 0)} | {r.get('api_errors', 0)} |")
    lines.append("")

    # Detail per payload
    for i, (r, p) in enumerate(zip(all_results, payload_data), 1):
        lines.append(f"## {i}. {r['file']} — {'PASS' if r['status'] == 'OK' else 'FAIL'}")
        lines.append("")
        lines.append(f"**Prompt:** {p.get('prompt', 'N/A')}")
        lines.append("")

        tool_calls = r.get("tool_calls", [])
        if tool_calls:
            lines.append("**Tool calls:**")
            for j, tc in enumerate(tool_calls, 1):
                result = tc.get("result", {}) or {}
                ok = result.get("ok", "?")
                icon = "PASS" if ok is True else "FAIL" if ok is False else "?"
                args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                lines.append(f"{j}. `{tc['tool']}` [{icon}]")
                lines.append(f"   - Args: `{args_str}`")
                if ok is False:
                    lines.append(f"   - **Error: {result.get('error', 'unknown')}**")
                if result.get("data"):
                    lines.append(f"   - Response: `{result['data'][:300]}`")
            lines.append("")

        failed_apis = [a for a in r.get("api_log", []) if not a.get("ok")]
        if failed_apis:
            lines.append("**Failed API calls:**")
            for a in failed_apis:
                lines.append(f"- `{a['method']} {a['url']}` -> **{a['status']}**: {a.get('error', '')}")
            lines.append("")

        if r.get("agent_response"):
            lines.append(f"**Agent response:** {r['agent_response'][:500]}")
            lines.append("")

    return "\n".join(lines)


def resolve_paths(args_paths: list) -> list:
    """Resolve payload paths from args (files, dirs, globs)."""
    paths = []
    for p in args_paths:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        elif "*" in p or "?" in p:
            paths.extend(sorted(glob.glob(p)))
        elif os.path.isfile(p):
            paths.append(p)
        else:
            print(f"Warning: {p} not found, skipping")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Replay saved payloads through /solve-debug")
    parser.add_argument("paths", nargs="+", help="Payload JSON file(s), directory, or glob pattern")
    parser.add_argument("--list", action="store_true", help="List payloads without running")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    parser.add_argument("--report-file", type=str, default="", help="Save markdown report to file")
    parser.add_argument("--last", type=int, default=0, help="Only replay the last N payloads")
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    if args.last:
        paths = paths[-args.last:]

    if not paths:
        print("No payload files found.")
        sys.exit(1)

    if args.list:
        print(f"Found {len(paths)} payloads:")
        for p in paths:
            payload = load_payload(p)
            prompt = payload.get("prompt", "")[:80]
            print(f"  {os.path.basename(p)}: {prompt}")
        return

    print(f"Replaying {len(paths)} payload(s) via {AGENT_URL}")
    print(f"Using credentials: {BASE_URL}")

    all_results = []
    payload_data = []
    for p in paths:
        payload = load_payload(p)
        payload_data.append(payload)
        print(f"\n>>> Replaying {os.path.basename(p)}")
        print(f"    Prompt: {payload.get('prompt', '')[:100]}...")

        resp = replay(payload, timeout=args.timeout)
        # Stash prompt in body for report
        resp["body"]["_prompt"] = payload.get("prompt", "")
        result = print_result(p, resp)
        all_results.append(result)

    # Summary
    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"REPLAY SUMMARY: {ok_count}/{total} passed")
    print(f"{'='*60}")
    for r in all_results:
        icon = " OK " if r["status"] == "OK" else "FAIL"
        print(f"  [{icon}] {r['file']} ({r['elapsed']:.0f}s, {len(r['tool_calls'])} tools, {r['api_errors']} errs)")

    # Generate report
    report = generate_report(all_results, payload_data)
    report_path = args.report_file or os.path.join(os.path.dirname(__file__), "replay_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    main()

"""Populate the Tripletex sandbox with realistic test data.

CLI wrapper around dashboard/sandbox.py — same logic used by the
dashboard Sandbox tab.

Usage:
    python populate_sandbox.py          # populate everything
    python populate_sandbox.py --clean  # delete all created entities first
    python populate_sandbox.py --check  # check sandbox health only
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tripletex_client import TripletexClient
from dashboard.sandbox import check_health, seed_entities, clean_entities

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("populate")


def get_client() -> TripletexClient:
    base_url = os.environ.get("TRIPLETEX_BASE_URL", "")
    token = os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    if not base_url or not token:
        print("Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN in .env")
        sys.exit(1)
    return TripletexClient(base_url, token)


def main():
    parser = argparse.ArgumentParser(description="Populate Tripletex Sandbox")
    parser.add_argument("--clean", action="store_true",
                        help="Delete all entities first, then seed")
    parser.add_argument("--check", action="store_true",
                        help="Check sandbox health only (no changes)")
    args = parser.parse_args()

    c = get_client()

    # Test connectivity
    test = c.get("/employee", params={"count": 1, "fields": "id"})
    if "error" in test:
        print(f"Cannot connect to sandbox: {test}")
        sys.exit(1)

    if args.check:
        print("\n=== Sandbox Health Check ===\n")
        health = check_health(c)
        print(json.dumps(health, indent=2, ensure_ascii=False))
        status = "READY" if health["ready"] else "NEEDS SETUP"
        print(f"\nStatus: {status}")
        sys.exit(0 if health["ready"] else 1)

    print("\n=== Populating Tripletex Sandbox ===\n")
    result = seed_entities(c, types=["all"], clean=args.clean)

    print("\n=== Summary ===")
    for entity_type, r in result.get("results", {}).items():
        created = r.get("created", 0)
        errors = r.get("errors", [])
        status = f"{created} created"
        if errors:
            status += f", {len(errors)} errors"
        print(f"  {entity_type:20s} {status}")

    bank = result.get("bank_account", {})
    if bank:
        if bank.get("ok"):
            print(f"  {'bank_account_1920':20s} {'already set' if bank.get('already_set') else 'configured'}")
        else:
            print(f"  {'bank_account_1920':20s} FAILED: {bank.get('error', '')}")

    print(f"\n  Total created: {result['total_created']}")
    print(f"  Total errors:  {result['total_errors']}")
    print(f"  API calls: {c._call_count}, errors: {c._error_count}")


if __name__ == "__main__":
    main()

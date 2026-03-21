"""Auto-submit script: waits for a new round, downloads previous round data, trains, submits.

Usage:
    python auto_round.py                    # Wait for next round, then submit
    python auto_round.py --check-interval 60  # Check every 60 seconds (default: 120)
"""
import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from client import AstarIslandClient
from calibration import CalibrationModel
from config import MAP_H, MAP_W, NUM_CLASSES

DATA_DIR = Path(__file__).parent / "data"


def download_round_analysis(client, round_id, round_number):
    """Download and save ground truth analysis for a completed round."""
    cal_dir = DATA_DIR / "calibration" / f"round{round_number}"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (cal_dir / "analysis_seed_0.json").exists():
        print(f"  Round {round_number} analysis already exists, skipping download")
        return True

    try:
        detail = client.get_round_detail(round_id)
        with open(cal_dir / "round_detail.json", "w") as f:
            json.dump(detail, f)

        for si in range(detail["seeds_count"]):
            analysis = client.get_analysis(round_id, si)
            with open(cal_dir / f"analysis_seed_{si}.json", "w") as f:
                json.dump(analysis, f)

            gt = np.array(analysis["ground_truth"])
            expected = gt.sum(axis=(0, 1))
            sett_pct = (expected[1] + expected[2]) / 1600
            print(f"  Seed {si}: score={analysis.get('score', 0):.2f}, sett+port={sett_pct:.1%}")

        print(f"  Round {round_number} analysis saved to {cal_dir}")
        return True
    except Exception as e:
        print(f"  Failed to download round {round_number} analysis: {e}")
        return False


def run_submission(client, round_id, detail):
    """Run the full adaptive pipeline."""
    from explore import run_adaptive_exploration
    from predict_gemini import gemini_predict
    from predict import validate_prediction
    from utils import apply_floor
    import predict

    seeds_count = detail["seeds_count"]

    # Run adaptive exploration
    exploration = run_adaptive_exploration(client, round_id, detail)

    accumulators = exploration.get("accumulators", [None] * seeds_count)
    initial_states = detail["initial_states"]
    global_mult = exploration.get("global_multipliers")
    fk_buckets = exploration.get("feature_key_buckets")
    multi_store = exploration.get("multi_sample_store")
    variance_regime = exploration.get("variance_regime")

    if variance_regime:
        print(f"\nVariance regime: {variance_regime}")
    if global_mult is not None:
        mult = global_mult.get_multipliers()
        print(f"Global multipliers: sett={mult[1]:.3f}, port={mult[2]:.3f}, "
              f"ruin={mult[3]:.3f}, forest={mult[4]:.3f}")

    # Submit predictions
    print("\n" + "=" * 60)
    print("Generating and submitting predictions")
    print("=" * 60)

    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        print(f"\nSeed {seed_idx}:")
        prediction = gemini_predict(
            state, global_mult, fk_buckets,
            multi_store=multi_store,
            variance_regime=variance_regime,
        )

        errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
        if errors:
            print(f"  Validation errors: {errors}")
            prediction = apply_floor(prediction)
            errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
            if errors:
                print(f"  Still invalid: {errors}")
                continue

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        print(f"  Submitted: {resp.get('status', 'unknown')}")

    print(f"\nAll predictions submitted for round {round_id}")


def main():
    parser = argparse.ArgumentParser(description="Auto-submit for next round")
    parser.add_argument("--check-interval", type=int, default=120,
                        help="Seconds between checks for new round (default: 120)")
    args = parser.parse_args()

    # Flush stdout for real-time monitoring
    sys.stdout.reconfigure(line_buffering=True)

    client = AstarIslandClient()

    # Step 1: Find latest completed round and download its analysis
    print("=" * 60)
    print("Step 1: Download analysis for recently completed rounds")
    print("=" * 60)

    rounds = client.get_rounds()
    for r in rounds:
        rn = r.get("round_number", 0)
        status = r.get("status", "")
        rid = r["id"]
        if status == "completed" and rn >= 8:
            print(f"\nRound {rn} ({status}):")
            download_round_analysis(client, rid, rn)

    # Step 2: Reload calibration with new data
    print("\n" + "=" * 60)
    print("Step 2: Calibration model status")
    print("=" * 60)
    cal = CalibrationModel.from_all_rounds()
    print(f"  {cal.get_stats()}")

    # Step 3: Wait for active round
    print("\n" + "=" * 60)
    print(f"Step 3: Waiting for active round (checking every {args.check_interval}s)")
    print("=" * 60)

    last_submitted = None
    while True:
        try:
            active = client.get_active_round()
            if active:
                round_id = active["id"]
                round_number = active["round_number"]

                if round_id == last_submitted:
                    pass  # Already submitted this round
                else:
                    print(f"\n{'=' * 60}")
                    print(f"ACTIVE ROUND DETECTED: Round {round_number} ({round_id})")
                    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'=' * 60}")

                    # Check budget
                    budget = client.get_budget()
                    remaining = budget["queries_max"] - budget["queries_used"]

                    if remaining > 0:
                        detail = client.get_round_detail(round_id)
                        print(f"Map: {detail['map_width']}x{detail['map_height']}, "
                              f"{detail['seeds_count']} seeds, {remaining} queries remaining\n")
                        run_submission(client, round_id, detail)
                        last_submitted = round_id
                        print(f"\nSubmission complete. Continuing to monitor...")
                    else:
                        print(f"  No queries remaining for round {round_number}")
                        last_submitted = round_id
            else:
                # No active round — check if any recently completed rounds need analysis
                for r in client.get_rounds()[:3]:
                    if r.get("status") == "completed":
                        rn = r.get("round_number", 0)
                        rid = r["id"]
                        cal_dir = DATA_DIR / "calibration" / f"round{rn}"
                        if not (cal_dir / "analysis_seed_0.json").exists():
                            print(f"\nNew completed round {rn} — downloading analysis...")
                            download_round_analysis(client, rid, rn)

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()

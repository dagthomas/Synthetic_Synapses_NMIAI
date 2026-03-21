"""Visualization and statistical analysis for Astar Island.

Usage:
    python analyze.py                    # Analyze most recent completed round
    python analyze.py --round-id UUID    # Analyze specific round
    python analyze.py --leaderboard      # Show leaderboard
    python analyze.py --my-rounds        # Show our round history
    python analyze.py --save-results     # Save results.json for all completed rounds
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from client import AstarIslandClient
from config import CLASS_NAMES, MAP_H, MAP_W, NUM_CLASSES
from utils import initial_grid_to_classes, terrain_to_class

DATA_DIR = Path(__file__).parent / "data"


def print_grid(grid: np.ndarray, title: str = ""):
    """Print a compact text visualization of a class grid."""
    symbols = {0: ".", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
    if title:
        print(f"\n{title}")
    for row in grid:
        print("".join(symbols.get(c, "?") for c in row))


def analyze_seed(client: AstarIslandClient, round_id: str, seed_index: int) -> dict:
    """Fetch and analyze ground truth for one seed of a completed round."""
    data = client.get_analysis(round_id, seed_index)

    gt = np.array(data["ground_truth"])  # (H, W, 6)
    h, w = gt.shape[:2]

    # Entropy per cell
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)

    # Argmax class
    argmax = np.argmax(gt, axis=-1)
    dynamic_mask = entropy > 0.1

    analysis = {
        "seed_index": seed_index,
        "score": data.get("score"),
        "height": h,
        "width": w,
        "entropy_mean": round(float(entropy.mean()), 4),
        "entropy_max": round(float(entropy.max()), 4),
        "dynamic_cells": int(dynamic_mask.sum()),
        "total_cells": h * w,
    }

    # Ground truth class distribution (argmax)
    gt_distribution = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = int((argmax == cls_idx).sum())
        gt_distribution[cls_name.lower()] = count
    analysis["gt_distribution"] = gt_distribution

    # Compare with our prediction if available
    if data.get("prediction"):
        pred = np.array(data["prediction"])
        pred_argmax = np.argmax(pred, axis=-1)
        correct = (pred_argmax == argmax).sum()
        analysis["argmax_accuracy"] = round(float(correct / (h * w)), 4)

        per_class_accuracy = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = argmax == cls_idx
            if mask.sum() > 0:
                cls_correct = ((pred_argmax == cls_idx) & mask).sum()
                per_class_accuracy[cls_name.lower()] = round(float(cls_correct / mask.sum()), 4)
        analysis["per_class_accuracy"] = per_class_accuracy

        # KL divergence (our metric)
        pred_safe = np.maximum(pred, 1e-10)
        kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
        analysis["mean_kl"] = round(float(kl.mean()), 6)
        if dynamic_mask.any():
            weighted_kl = float(np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / entropy[dynamic_mask].sum())
            analysis["weighted_kl"] = round(weighted_kl, 6)

    return analysis


def save_round_results(client: AstarIslandClient, round_info: dict, round_id: str):
    """Save a complete results.json for a round.

    Includes: round metadata, per-seed scores, our rank, model version info,
    and per-seed analysis details.
    """
    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    results_path = round_dir / "results.json"

    detail = client.get_round_detail(round_id)

    results = {
        "round_id": round_id,
        "round_number": round_info.get("round_number"),
        "status": round_info.get("status"),
        "event_date": round_info.get("event_date"),
        "map_size": f"{detail['map_width']}x{detail['map_height']}",
        "seeds_count": detail["seeds_count"],
        "round_score": round_info.get("round_score"),
        "seed_scores": round_info.get("seed_scores"),
        "rank": round_info.get("rank"),
        "total_teams": round_info.get("total_teams"),
        "queries_used": round_info.get("queries_used", 0),
        "queries_max": round_info.get("queries_max", 50),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seeds": [],
    }

    # Per-seed analysis
    for seed_idx in range(detail["seeds_count"]):
        try:
            time.sleep(0.3)
            seed_analysis = analyze_seed(client, round_id, seed_idx)
            results["seeds"].append(seed_analysis)
        except Exception as e:
            results["seeds"].append({
                "seed_index": seed_idx,
                "error": str(e),
            })

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved {results_path}")
    return results


def compare_initial_vs_final(initial_grid: list[list[int]], ground_truth: np.ndarray) -> dict:
    """Compare initial terrain to final ground truth distributions."""
    initial_classes = initial_grid_to_classes(initial_grid)
    gt_argmax = np.argmax(ground_truth, axis=-1)
    h, w = initial_classes.shape

    transitions = {}
    for y in range(h):
        for x in range(w):
            initial = int(initial_classes[y, x])
            final = int(gt_argmax[y, x])
            key = f"{CLASS_NAMES[initial]}->{CLASS_NAMES[final]}"
            transitions[key] = transitions.get(key, 0) + 1

    return transitions


def show_leaderboard(client: AstarIslandClient):
    """Display the leaderboard."""
    lb = client.get_leaderboard()
    print(f"\n{'Rank':<6}{'Team':<25}{'Score':<10}{'Rounds':<8}{'Streak':<10}")
    print("-" * 59)
    for entry in lb:
        print(f"{entry['rank']:<6}{entry['team_name']:<25}"
              f"{entry['weighted_score']:<10.1f}"
              f"{entry['rounds_participated']:<8}"
              f"{entry.get('hot_streak_score', 0):<10.1f}")


def show_my_rounds(client: AstarIslandClient):
    """Display our rounds with scores."""
    rounds = client.get_my_rounds()
    print(f"\n{'Round':<8}{'Status':<12}{'Score':<10}{'Rank':<8}{'Seeds':<8}{'Queries':<10}")
    print("-" * 56)
    for r in rounds:
        score = r.get("round_score")
        score_str = f"{score:.1f}" if score is not None else "-"
        rank = r.get("rank")
        rank_str = str(rank) if rank is not None else "-"
        print(f"{r['round_number']:<8}{r['status']:<12}{score_str:<10}"
              f"{rank_str:<8}{r['seeds_submitted']:<8}"
              f"{r['queries_used']}/{r['queries_max']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Astar Island results")
    parser.add_argument("--round-id", type=str, help="Specific round UUID")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed index")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--my-rounds", action="store_true", help="Show my rounds")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results.json for all scored rounds")
    args = parser.parse_args()

    client = AstarIslandClient()

    if args.leaderboard:
        show_leaderboard(client)
        return

    if args.my_rounds:
        show_my_rounds(client)
        return

    if args.save_results:
        rounds = client.get_my_rounds()
        scored = [r for r in rounds
                  if r["status"] in ("completed", "scoring") and r.get("round_score") is not None]
        if not scored:
            print("No scored rounds yet.")
            show_my_rounds(client)
            return
        for r in scored:
            rid = r["id"]
            existing = DATA_DIR / "rounds" / rid / "results.json"
            if existing.exists():
                print(f"Round #{r['round_number']}: already saved ({existing})")
                continue
            print(f"Round #{r['round_number']} (score: {r['round_score']:.1f}, rank: {r.get('rank')}):")
            save_round_results(client, r, rid)
        return

    # Default: analyze a specific round
    if args.round_id:
        round_id = args.round_id
    else:
        rounds = client.get_my_rounds()
        completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
        if not completed:
            print("No completed rounds to analyze.")
            show_my_rounds(client)
            return
        round_id = completed[-1]["id"]
        round_info = completed[-1]
        print(f"Analyzing round #{round_info['round_number']} "
              f"(score: {round_info.get('round_score', 'N/A')})")

    detail = client.get_round_detail(round_id)
    seeds_count = detail["seeds_count"]
    seed_range = [args.seed] if args.seed is not None else range(seeds_count)

    for seed_idx in seed_range:
        print(f"\n{'=' * 50}")
        print(f"Seed {seed_idx}")
        print(f"{'=' * 50}")

        try:
            time.sleep(0.3)
            analysis = analyze_seed(client, round_id, seed_idx)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        print(f"  Score: {analysis.get('score', 'N/A')}")
        print(f"  Dynamic cells: {analysis['dynamic_cells']}/{analysis['total_cells']}")
        print(f"  Mean entropy: {analysis['entropy_mean']:.3f}")

        if "argmax_accuracy" in analysis:
            print(f"  Argmax accuracy: {analysis['argmax_accuracy']:.1%}")
            for cls_name, acc in analysis.get("per_class_accuracy", {}).items():
                print(f"    {cls_name}: {acc:.1%}")

        if "weighted_kl" in analysis:
            print(f"  Weighted KL: {analysis['weighted_kl']:.4f}")

        print("  Ground truth distribution:")
        for cls_name, count in analysis.get("gt_distribution", {}).items():
            pct = 100 * count / analysis["total_cells"]
            print(f"    {cls_name}: {count} ({pct:.1f}%)")

        # Transitions
        if seed_idx < len(detail.get("initial_states", [])):
            try:
                gt_data = client.get_analysis(round_id, seed_idx)
                if gt_data.get("initial_grid") and gt_data.get("ground_truth"):
                    transitions = compare_initial_vs_final(
                        gt_data["initial_grid"], np.array(gt_data["ground_truth"])
                    )
                    changes = {k: v for k, v in transitions.items()
                               if k.split("->")[0] != k.split("->")[1]}
                    if changes:
                        print("\n  Top terrain changes:")
                        for key, count in sorted(changes.items(), key=lambda x: -x[1])[:10]:
                            print(f"    {key}: {count}")
            except Exception:
                pass

    # Auto-save results
    round_dir = DATA_DIR / "rounds" / round_id
    if not (round_dir / "results.json").exists():
        rounds = client.get_my_rounds()
        round_info = next((r for r in rounds if r["id"] == round_id), None)
        if round_info and round_info.get("round_score") is not None:
            print(f"\nSaving results.json...")
            save_round_results(client, round_info, round_id)


if __name__ == "__main__":
    main()

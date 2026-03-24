"""Simulate a full prior game with 10 teams playing 5 rounds.

Creates:
- 10 corporations with different skill levels
- 5 rounds with shifting params (barren → volatile progression)
- Realistic query patterns and prediction quality
"""
import requests
import numpy as np
import json
import time

BASE = "http://localhost:9742"

def login(name, password):
    r = requests.post(f"{BASE}/auth/login", json={"name": name, "password": password})
    r.raise_for_status()
    return r.json()["token"]

def register(name, password):
    r = requests.post(f"{BASE}/auth/register", json={"name": name, "password": password})
    if r.status_code == 409:  # already exists
        try:
            return login(name, password)
        except:
            pass
    if not r.ok:
        print(f"  Register error for {name}: {r.status_code} {r.text[:200]}")
        r.raise_for_status()
    return r.json()["token"]

def api(token, method, path, json_data=None):
    headers = {"Authorization": f"Bearer {token}"}
    if method == "GET":
        r = requests.get(f"{BASE}{path}", headers=headers)
    else:
        r = requests.post(f"{BASE}{path}", headers=headers, json=json_data)
    r.raise_for_status()
    return r.json()

# ── 10 Corporations ────────────────────────────────────────────────────

CORPORATIONS = [
    # (name, skill_level: 1-10, query_aggressiveness: 1-10)
    ("NovaCrystal Industries", 9, 9),
    ("Obsidian Dynamics", 8, 7),
    ("Zenith Mineral Corp", 7, 8),
    ("Voidrunner Mining", 7, 6),
    ("Helix Prospecting", 6, 7),
    ("Regolith Solutions", 5, 5),
    ("Crimson Extraction", 5, 6),
    ("Phantom Analytics", 4, 4),
    ("Dustwalker Inc", 3, 3),
    ("Nebula Scouts", 2, 8),  # lots of queries, bad predictions
]

# Planet regimes shift across rounds
PLANET_REGIMES = [
    ("moderate", "Kepler-442b"),
    ("boom", "Proxima Centauri d"),
    ("collapse", "TRAPPIST-1e"),
    ("moderate", "Gliese 667Cc"),
    ("boom", "HD 40307g"),
]

print("=" * 60)
print("Q* FRONTIER — FULL GAME SIMULATION")
print("=" * 60)

# ── Setup ──────────────────────────────────────────────────────────────

admin_token = login("admin", "admin")
print(f"\nAdmin authenticated")

# Register corporations
team_tokens = {}
for name, skill, aggr in CORPORATIONS:
    team_tokens[name] = register(name, "corp123")
    print(f"  Corporation '{name}' registered (skill={skill}, aggr={aggr})")

# ── Play 5 rounds ─────────────────────────────────────────────────────

round_ids = []
for round_idx, (regime, planet_name) in enumerate(PLANET_REGIMES):
    print(f"\n{'─' * 60}")
    print(f"PLANET {round_idx + 1}: {planet_name} (regime: {regime})")
    print(f"{'─' * 60}")

    # Create and activate round
    result = api(admin_token, "POST", "/admin/api/rounds", {"regime": regime})
    rid = result["id"]
    round_ids.append(rid)
    api(admin_token, "POST", f"/admin/api/rounds/{rid}/activate")
    print(f"  Created & activated: {rid[:8]}...")

    # Get round detail
    detail = api(admin_token, "GET", f"/astar-island/rounds/{rid}")
    w, h = detail["map_width"], detail["map_height"]

    # Each corporation plays
    for corp_name, skill, aggr in CORPORATIONS:
        token = team_tokens[corp_name]

        # Determine query count based on aggressiveness
        n_queries = min(int(aggr * 5), 45)  # max 45 of 50

        rng = np.random.RandomState(hash(corp_name + str(round_idx)) % 2**31)

        # Make viewport queries
        observations = []
        for q in range(n_queries):
            seed_idx = q % 5
            vx = rng.randint(0, max(1, w - 15))
            vy = rng.randint(0, max(1, h - 15))
            try:
                obs = api(token, "POST", "/astar-island/simulate", {
                    "round_id": rid,
                    "seed_index": seed_idx,
                    "viewport_x": int(vx),
                    "viewport_y": int(vy),
                    "viewport_w": 15,
                    "viewport_h": 15,
                })
                observations.append((seed_idx, obs))
            except:
                break

        # Build and submit predictions for all 5 seeds
        for seed_idx in range(5):
            prediction = np.full((h, w, 6), 0.0)

            # Aggregate observations for this seed
            seed_obs = [o for si, o in observations if si == seed_idx]
            all_obs = [o for _, o in observations]

            # Compute observed class frequencies
            class_counts = np.zeros(6)
            total = 0
            obs_to_use = seed_obs if seed_obs else all_obs
            for obs in obs_to_use:
                for row in obs["grid"]:
                    for cell in row:
                        cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                        class_counts[cls_map.get(cell, 0)] += 1
                        total += 1

            if total > 0:
                base_probs = class_counts / total
            else:
                base_probs = np.array([0.6, 0.1, 0.05, 0.05, 0.15, 0.05])

            # Get initial grid
            init_grid = detail["initial_states"][seed_idx]["grid"]

            # Noise inversely proportional to skill
            noise_scale = 0.20 / skill  # skill=10 → 0.02 noise, skill=1 → 0.20

            for y in range(h):
                for x in range(w):
                    cell = init_grid[y][x]

                    if cell == 10:  # Void
                        prediction[y, x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif cell == 5:  # Obsidian Ridge
                        prediction[y, x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    else:
                        noise = rng.normal(0, noise_scale, 6)
                        probs = base_probs.copy() + noise

                        # Skill-based terrain awareness
                        cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                        init_cls = cls_map.get(cell, 0)
                        probs[init_cls] += 0.1 * (skill / 10.0)

                        if cell in (1, 2):  # Crystal/Refinery
                            probs[1] += 0.08 * (skill / 10.0)
                            probs[3] += 0.04 * (skill / 10.0)

                        if cell == 4:  # Xenoflora
                            probs[4] += 0.05 * (skill / 10.0)

                        probs = np.maximum(probs, 0.01)
                        probs = probs / probs.sum()
                        prediction[y, x] = probs

            try:
                api(token, "POST", "/astar-island/submit", {
                    "round_id": rid,
                    "seed_index": seed_idx,
                    "prediction": prediction.tolist(),
                })
            except Exception as e:
                print(f"    Submit error for {corp_name} seed {seed_idx}: {e}")

        print(f"  {corp_name:30s} | {len(observations):2d} scans | 5 reports submitted")

    # Score the round
    result = api(admin_token, "POST", f"/admin/api/rounds/{rid}/score")
    print(f"\n  >> Scored {result['predictions_scored']} predictions")

    # Show round results
    detail = api(admin_token, "GET", f"/admin/api/rounds/{rid}")
    scores = sorted(detail["team_scores"], key=lambda x: -(x["average_score"] or 0))
    print(f"\n  {'Corporation':30s} | {'Avg':>6s} | Seeds")
    print(f"  {'─' * 30}-+-{'─' * 6}-+-{'─' * 30}")
    for i, ts in enumerate(scores):
        avg = ts["average_score"]
        seeds_str = " ".join(f"{s:5.1f}" if s else "  -  " for s in ts["seed_scores"])
        medal = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {ts['team_name']:30s} | {avg:5.1f} | {seeds_str} {medal}")

# ── Final Leaderboard ────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("FINAL LEADERBOARD")
print(f"{'=' * 60}")
lb = requests.get(f"{BASE}/astar-island/leaderboard").json()
for entry in lb:
    print(f"  #{entry['rank']:2d}  {entry['team_name']:30s}  {entry['weighted_score']:6.1f} pts  ({entry['rounds_participated']} planets)")

print(f"\n{'=' * 60}")
print(f"Game complete! {len(PLANET_REGIMES)} planets explored by {len(CORPORATIONS)} corporations.")
print(f"Admin panel: http://localhost:9742/admin")
print(f"{'=' * 60}")

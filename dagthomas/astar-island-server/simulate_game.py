"""Simulate 3 teams playing 3 rounds of Astar Island."""
import requests
import numpy as np
import time
import json

BASE = "http://localhost:9742"

def login(name, password):
    r = requests.post(f"{BASE}/auth/login", json={"name": name, "password": password})
    r.raise_for_status()
    return r.json()["token"]

def register(name, password):
    r = requests.post(f"{BASE}/auth/register", json={"name": name, "password": password})
    if r.status_code == 409:  # already exists
        return login(name, password)
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

# ── Setup ──────────────────────────────────────────────────────────────

admin_token = login("admin", "admin")

# Register/login teams
teams = [
    ("Viking Raiders", "pass123"),
    ("Norse Predictors", "pass123"),
    ("Fjord Analytics", "pass123"),
]
team_tokens = {}
for name, pw in teams:
    team_tokens[name] = register(name, pw)
    print(f"  Team '{name}' ready")

# ── Create 3 rounds with different regimes ─────────────────────────────

regimes = ["boom", "moderate", "collapse"]
round_ids = []

for regime in regimes:
    print(f"\nCreating {regime} round...")
    result = api(admin_token, "POST", "/admin/api/rounds", {"regime": regime})
    rid = result["id"]
    round_ids.append(rid)
    print(f"  Round #{result['round_number']} ({regime}): {rid[:8]}...")

    # Activate
    api(admin_token, "POST", f"/admin/api/rounds/{rid}/activate")
    print(f"  Activated")

    # Get round detail for map dimensions
    detail = api(admin_token, "GET", f"/astar-island/rounds/{rid}")
    w = detail["map_width"]
    h = detail["map_height"]

    # Each team plays with different strategies
    for team_name, _ in teams:
        token = team_tokens[team_name]
        print(f"\n  Team '{team_name}' playing...")

        # ── Observe: make viewport queries ──
        observations = []
        # Each team queries different number of times and different viewports
        if team_name == "Viking Raiders":
            # Aggressive: 15 queries, scan systematically
            n_queries = 15
        elif team_name == "Norse Predictors":
            # Moderate: 10 queries
            n_queries = 10
        else:
            # Conservative: 5 queries
            n_queries = 5

        rng = np.random.RandomState(hash(team_name + regime) % 2**31)

        for q in range(n_queries):
            seed_idx = q % 5  # spread across seeds
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
                observations.append(obs)
            except Exception as e:
                print(f"    Query {q} failed: {e}")
                break

        print(f"    Made {len(observations)} queries")

        # ── Build predictions from observations ──
        for seed_idx in range(5):
            # Build prediction based on observations + noise
            # Base: semi-informed prediction (not uniform, not perfect)
            prediction = np.full((h, w, 6), 0.0)

            # Count terrain classes from observations of this seed
            seed_obs = [o for o in observations if True]  # use all obs as rough guide

            if seed_obs:
                # Use observation data to build rough class frequencies
                total_cells = 0
                class_counts = np.zeros(6)
                for obs in seed_obs:
                    for row in obs["grid"]:
                        for cell in row:
                            cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                            cls = cls_map.get(cell, 0)
                            class_counts[cls] += 1
                            total_cells += 1

                if total_cells > 0:
                    base_probs = class_counts / total_cells
                else:
                    base_probs = np.array([0.6, 0.1, 0.05, 0.05, 0.15, 0.05])
            else:
                base_probs = np.array([0.6, 0.1, 0.05, 0.05, 0.15, 0.05])

            # Get initial grid for this seed
            initial_state = detail["initial_states"][seed_idx]
            init_grid = initial_state["grid"]

            for y in range(h):
                for x in range(w):
                    cell = init_grid[y][x]

                    if cell == 10:  # Ocean → always empty class
                        prediction[y, x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    elif cell == 5:  # Mountain → always mountain
                        prediction[y, x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    else:
                        # Dynamic cell: use base probs with team-specific noise
                        noise_scale = {"Viking Raiders": 0.05, "Norse Predictors": 0.08, "Fjord Analytics": 0.12}
                        noise = rng.normal(0, noise_scale[team_name], 6)
                        probs = base_probs + noise

                        # Bias toward initial terrain type
                        cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                        init_cls = cls_map.get(cell, 0)
                        probs[init_cls] += 0.15  # boost initial type

                        # For settlements, boost settlement/ruin
                        if cell in (1, 2):
                            probs[1] += 0.1  # settlement
                            probs[3] += 0.05  # ruin possible

                        # Floor and normalize
                        probs = np.maximum(probs, 0.01)
                        probs = probs / probs.sum()
                        prediction[y, x] = probs

            # Submit prediction
            pred_list = prediction.tolist()
            try:
                api(token, "POST", "/astar-island/submit", {
                    "round_id": rid,
                    "seed_index": seed_idx,
                    "prediction": pred_list,
                })
            except Exception as e:
                print(f"    Submit seed {seed_idx} failed: {e}")

        print(f"    Submitted 5 predictions")

    # Score the round
    print(f"\n  Scoring round...")
    result = api(admin_token, "POST", f"/admin/api/rounds/{rid}/score")
    print(f"  Scored {result['predictions_scored']} predictions")

# ── Print final leaderboard ──────────────────────────────────────────

print("\n" + "="*50)
print("LEADERBOARD")
print("="*50)
lb = requests.get(f"{BASE}/astar-island/leaderboard").json()
for entry in lb:
    print(f"  #{entry['rank']} {entry['team_name']}: {entry['weighted_score']:.1f} ({entry['rounds_participated']} rounds)")

# Print per-round scores
for i, rid in enumerate(round_ids):
    detail = api(admin_token, "GET", f"/admin/api/rounds/{rid}")
    print(f"\nRound #{detail['round_number']} ({regimes[i]}):")
    for ts in sorted(detail["team_scores"], key=lambda x: -(x["average_score"] or 0)):
        avg = ts["average_score"]
        seeds = ", ".join(f"{s:.1f}" if s else "-" for s in ts["seed_scores"])
        print(f"  {ts['team_name']:20s}  avg={avg:.1f}  seeds=[{seeds}]  queries={ts['queries_used']}")

print("\nDone! Check admin panel at http://localhost:9742/admin")

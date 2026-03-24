"""Fast population: 10 teams, 5 rounds, minimal queries, parallel-ish."""
import requests
import numpy as np
import json
import concurrent.futures

BASE = "http://localhost:9742"

def login(name, pw):
    return requests.post(f"{BASE}/auth/login", json={"name": name, "password": pw}).json()["token"]

def register(name, pw):
    r = requests.post(f"{BASE}/auth/register", json={"name": name, "password": pw})
    if r.status_code == 409:
        return login(name, pw)
    r.raise_for_status()
    return r.json()["token"]

def api(token, method, path, data=None):
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = (requests.get if method == "GET" else requests.post)(f"{BASE}{path}", headers=h, json=data)
    r.raise_for_status()
    return r.json()

# Setup
admin = login("admin", "admin")
CORPS = [
    ("NovaCrystal Industries", 9), ("Obsidian Dynamics", 8), ("Zenith Mineral Corp", 7),
    ("Voidrunner Mining", 7), ("Helix Prospecting", 6), ("Regolith Solutions", 5),
    ("Crimson Extraction", 5), ("Phantom Analytics", 4), ("Dustwalker Inc", 3), ("Nebula Scouts", 2),
]
tokens = {}
for name, _ in CORPS:
    tokens[name] = register(name, "corp123")
print(f"{len(tokens)} corporations ready")

# Check existing rounds
existing = requests.get(f"{BASE}/astar-island/rounds").json()
completed = [r for r in existing if r["status"] == "completed"]
print(f"{len(completed)} rounds already completed, {len(existing)} total")

# Create rounds until we have 5 completed
regimes = ["moderate", "boom", "collapse", "boom", "moderate"]
target = 5

for ri in range(len(completed), target):
    regime = regimes[ri % len(regimes)]
    print(f"\n--- Round {ri+1}/{target} ({regime}) ---")

    # Create & activate
    result = api(admin, "POST", "/admin/api/rounds", {"regime": regime})
    rid = result["id"]
    api(admin, "POST", f"/admin/api/rounds/{rid}/activate")
    detail = api(admin, "GET", f"/astar-island/rounds/{rid}")
    w, h = detail["map_width"], detail["map_height"]
    print(f"  Created: {rid[:8]}...")

    # Each team: 3 queries + 5 predictions (minimal for speed)
    for name, skill in CORPS:
        token = tokens[name]
        rng = np.random.RandomState(hash(name + str(ri)) % 2**31)

        # 3 quick scans
        for q in range(3):
            try:
                api(token, "POST", "/astar-island/simulate", {
                    "round_id": rid, "seed_index": q % 5,
                    "viewport_x": int(rng.randint(0, 25)),
                    "viewport_y": int(rng.randint(0, 25)),
                    "viewport_w": 15, "viewport_h": 15,
                })
            except:
                pass

        # Submit 5 predictions
        for si in range(5):
            pred = np.full((h, w, 6), 0.0)
            init_grid = detail["initial_states"][si]["grid"]
            noise = 0.20 / skill

            for y in range(h):
                for x in range(w):
                    cell = init_grid[y][x]
                    if cell == 10:
                        pred[y, x] = [1, 0, 0, 0, 0, 0]
                    elif cell == 5:
                        pred[y, x] = [0, 0, 0, 0, 0, 1]
                    else:
                        p = np.array([0.55, 0.12, 0.05, 0.06, 0.17, 0.05])
                        p += rng.normal(0, noise, 6)
                        cls = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5}.get(cell, 0)
                        p[cls] += 0.12 * (skill / 10.0)
                        p = np.maximum(p, 0.01)
                        pred[y, x] = p / p.sum()
            try:
                api(token, "POST", "/astar-island/submit", {
                    "round_id": rid, "seed_index": si, "prediction": pred.tolist()
                })
            except Exception as e:
                print(f"    Submit fail {name} s{si}: {e}")

        print(f"  {name:30s} done")

    # Score
    result = api(admin, "POST", f"/admin/api/rounds/{rid}/score")
    print(f"  Scored: {result['predictions_scored']} predictions")

# Final leaderboard
print(f"\n{'='*60}")
print("FINAL LEADERBOARD")
print(f"{'='*60}")
lb = requests.get(f"{BASE}/astar-island/leaderboard").json()
for e in lb:
    print(f"  #{e['rank']:2d}  {e['team_name']:30s}  {e['weighted_score']:6.1f} pts  ({e['rounds_participated']} planets)")

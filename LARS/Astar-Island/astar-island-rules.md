# 🏝️ Astar Island — Complete Rules & Reference

> **NM i AI 2026** | Task type: Observation + probabilistic prediction  
> Platform: [app.ainm.no](https://app.ainm.no) | API: `https://api.ainm.no/astar-island`

---

## 📌 What is Astar Island?

Astar Island is a machine learning challenge where you observe a **black-box Norse civilisation simulator** through a limited viewport and predict the final world state.

The simulator runs a procedurally generated Norse world for **50 years** — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

**Goal:** Observe the world, learn the hidden rules, and predict the probability distribution of terrain types across the entire map.

---

## ⚙️ How It Works

```
1. A round starts        → Admin creates a round with a fixed map, hidden parameters, and 5 seeds
2. Observe via viewport  → POST /simulate with coordinates (max 15×15 cells)
3. Learn the hidden rules → Analyse observations to understand the forces governing the world
4. Generate predictions  → Build probability distributions for the full map
5. Submit predictions    → For each of 5 seeds: submit a W×H×6 probability tensor
6. Scoring               → Your prediction is compared against ground truth via KL divergence
```

**You have 50 queries total per round, shared across all 5 seeds.**  
Each query reveals only a 15×15 window of a 40×40 map — be strategic!

---

## 🗺️ Map & Terrain

### Terrain Types

| Internal Code | Terrain    | Class Index   | Description                                      |
|---------------|------------|---------------|--------------------------------------------------|
| `10`          | Ocean      | `0` (Empty)   | Impassable water, borders the map               |
| `11`          | Plains     | `0` (Empty)   | Flat land, buildable                             |
| `0`           | Empty      | `0`           | Generic empty cell                               |
| `1`           | Settlement | `1`           | Active Norse settlement                          |
| `2`           | Port       | `2`           | Coastal settlement with harbour                  |
| `3`           | Ruin       | `3`           | Collapsed settlement                             |
| `4`           | Forest     | `4`           | Provides food to adjacent settlements            |
| `5`           | Mountain   | `5`           | Impassable terrain                               |

> **Note:** Ocean, Plains and Empty all map to class `0` in predictions.  
> Mountains are static (never change). Forests are mostly static.  
> The interesting cells are those that can become Settlement, Port or Ruin.

### Map Generation (from map seed)

- Ocean borders surround the entire map
- Fjords cut inland from random edges
- Mountain chains form via random walks
- Forest patches cover land with clustered groves
- Initial settlements placed on land cells, spaced apart

**The map seed is visible to you** — you can reconstruct the initial terrain layout locally.

---

## 🏰 Simulation Lifecycle

Each of the 50 years cycles through these phases in order:

### 1. Growth
- Settlements produce food based on adjacent terrain
- Population grows when conditions are right
- Ports develop along coastlines
- Longships are built for naval operations
- Prosperous settlements expand by founding new settlements on nearby land

### 2. Conflict
- Settlements raid each other (longships extend raiding range significantly)
- Desperate settlements (low food) raid more aggressively
- Successful raids loot resources and damage the defender
- Sometimes conquered settlements change allegiance to the raiding faction

### 3. Trade
- Ports within range of each other can trade (if not at war)
- Trade generates wealth and food for both parties
- Technology diffuses between trading partners

### 4. Winter
- All settlements lose food
- Settlements can collapse from starvation, sustained raids, or harsh winters
- Collapsed settlements become **Ruins** and disperse population to nearby friendly settlements

### 5. Environment
- The natural world slowly reclaims abandoned land
- Thriving settlements may reclaim and rebuild ruined sites → new outposts
- Coastal ruins can be restored as ports
- Without settlement intervention: ruins are overtaken by forest or fade into open plains

### Settlement Properties
Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (`owner_id`).

Initial states expose settlement positions and port status only. Internal stats (population, food, wealth, defense) are only visible through simulation queries.

---

## 🎯 Scoring

### Ground Truth
The organisers run the simulator **hundreds of times** with the true hidden parameters per seed. This produces a probability distribution for each cell.

**Example:** A cell might have ground truth `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]`  
→ 60% chance of Settlement, 25% Port, 15% Ruin after 50 years.

### KL Divergence
For each cell, KL divergence measures how different your prediction is from the ground truth:

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)
```
- `p` = ground truth, `q` = your prediction
- **Lower KL = better match**

### Entropy Weighting
Static cells (ocean stays ocean, mountain stays mountain) have near-zero entropy and are **excluded from scoring**.  
Only dynamic cells contribute to your score, weighted by their entropy:

```
entropy(cell) = -Σ pᵢ × log(pᵢ)
```

Cells with higher entropy (more uncertain outcomes) count more toward your score. This focuses scoring on the interesting parts of the map.

### Score Formula

```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ──────────────────────────────────────────────────────────
                                  Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

| Score | Meaning                                                 |
|-------|---------------------------------------------------------|
| 100   | Perfect prediction (your distribution matches ground truth exactly) |
| 0     | Terrible prediction (high KL divergence)               |

The exponential decay means small improvements yield diminishing score gains.

### Per-Round Score
```
round_score = (score_seed_0 + score_seed_1 + ... + score_seed_4) / 5
```
If you don't submit for a seed → that seed scores 0. **Always submit something for all 5 seeds!**

### Leaderboard Score
```
leaderboard_score = max(round_score × round_weight) across all rounds
```
Later rounds may have higher weights. Only your single best result counts.

A **hot streak score** (average of your last 3 rounds) is also tracked.

### Game End
Each round has a prediction window (typically **2 hours 45 minutes**). After it closes:
1. Round status changes to `scoring`
2. All predictions are scored against ground truth
3. Per-seed scores are averaged to compute round score
4. Leaderboard updates with weighted averages
5. Round status changes to `completed`

---

## ⚠️ Common Pitfalls

### 🚨 Never assign probability 0.0 to any class!

KL divergence includes the term `pᵢ × log(pᵢ / qᵢ)`. If the ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`, the divergence goes to **infinity** — destroying your entire score for that cell.

Even if you're confident a cell is Forest, the ground truth may assign a small probability to Settlement or Ruin across thousands of simulations.

**Solution — always enforce a minimum probability floor:**
```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

This small safety margin costs almost nothing in score but protects against catastrophic KL blowups.

---

## 🔌 API Reference

### Base URL
```
https://api.ainm.no/astar-island
```

### Authentication
All team endpoints require authentication via:
- **Cookie:** `access_token` JWT cookie (set automatically when you log in at app.ainm.no)
- **Bearer token:** `Authorization: Bearer <token>` header

### Endpoints Overview

| Method | Path                                    | Auth   | Description                              |
|--------|-----------------------------------------|--------|------------------------------------------|
| GET    | `/rounds`                               | Public | List all rounds                          |
| GET    | `/rounds/{round_id}`                    | Public | Round details + initial states           |
| GET    | `/budget`                               | Team   | Query budget for active round            |
| POST   | `/simulate`                             | Team   | Observe one simulation through viewport  |
| POST   | `/submit`                               | Team   | Submit prediction tensor                 |
| GET    | `/my-rounds`                            | Team   | Rounds with your scores, rank, budget    |
| GET    | `/my-predictions/{round_id}`            | Team   | Your predictions with argmax/confidence  |
| GET    | `/analysis/{round_id}/{seed_index}`     | Team   | Post-round ground truth comparison       |
| GET    | `/leaderboard`                          | Public | Astar Island leaderboard                 |

### Round Statuses

| Status      | Meaning                                         |
|-------------|-------------------------------------------------|
| `pending`   | Round created but not yet started               |
| `active`    | Queries and submissions open                    |
| `scoring`   | Submissions closed, scoring in progress         |
| `completed` | Scores finalized                                |

### Rate Limits

| Endpoint        | Limit                      |
|-----------------|----------------------------|
| POST /simulate  | 5 requests/second per team |
| POST /submit    | 2 requests/second per team |

Exceeding these limits returns `429 Too Many Requests`.

---

## 📡 Detailed API Documentation

### GET /rounds
List all rounds with status and timing.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "active",
    "map_width": 40,
    "map_height": 40,
    "prediction_window_minutes": 165,
    "started_at": "2026-03-19T10:00:00Z",
    "closes_at": "2026-03-19T10:45:00Z",
    "round_weight": 1,
    "created_at": "2026-03-19T09:00:00Z"
  }
]
```

### GET /rounds/{round_id}
Returns round details including initial map states for all seeds. Use this to reconstruct the starting terrain locally.

> Settlement data in initial states shows only position and port status. Internal stats (population, food, wealth, defense) are not exposed.

```json
{
  "id": "uuid",
  "round_number": 1,
  "status": "active",
  "map_width": 40,
  "map_height": 40,
  "seeds_count": 5,
  "initial_states": [
    {
      "grid": [[10, 10, 10, ...], ...],
      "settlements": [
        { "x": 5, "y": 12, "has_port": true, "alive": true }
      ]
    }
  ]
}
```

---

### POST /simulate — Core Observation Endpoint

Run one stochastic simulation and reveal a viewport window. **Costs 1 query (max 50 per round).**

**Request:**
```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

| Field        | Type       | Description                                      |
|--------------|------------|--------------------------------------------------|
| `round_id`   | string     | UUID of the active round                         |
| `seed_index` | int (0–4)  | Which of the 5 seeds to simulate                 |
| `viewport_x` | int (>=0)  | Left edge of viewport (default 0)                |
| `viewport_y` | int (>=0)  | Top edge of viewport (default 0)                 |
| `viewport_w` | int (5–15) | Viewport width (default 15)                      |
| `viewport_h` | int (5–15) | Viewport height (default 15)                     |

**Response:**
```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

> The grid contains only the viewport region (`viewport_h × viewport_w`), not the full map.  
> The settlements list includes only settlements within the viewport.  
> Each call uses a **different random sim_seed** → different stochastic outcome.

**Error Codes:**

| Status | Meaning                                                        |
|--------|----------------------------------------------------------------|
| 400    | Round not active, or invalid seed_index                        |
| 403    | Not on a team                                                  |
| 404    | Round not found                                                |
| 429    | Query budget exhausted (50/50) or rate limit exceeded          |

---

### POST /submit — Submit Prediction

Submit your prediction for one seed. **You must submit all 5 seeds for a complete score.**

**Request:**
```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],
      ...
    ],
    ...
  ]
}
```

**Prediction Format — `H×W×6` tensor:**
- `prediction[y][x][class]`
- Outer dimension: H rows (height)
- Middle dimension: W columns (width)
- Inner dimension: **6 probabilities** (one per class)
- Each cell's 6 probabilities **must sum to 1.0** (±0.01 tolerance)
- All probabilities must be **non-negative**

**Class Indices:**

| Index | Class                          |
|-------|--------------------------------|
| `0`   | Empty (Ocean, Plains, Empty)   |
| `1`   | Settlement                     |
| `2`   | Port                           |
| `3`   | Ruin                           |
| `4`   | Forest                         |
| `5`   | Mountain                       |

> Resubmitting for the same seed **overwrites** your previous prediction. Only the last submission counts.

**Validation Errors:**

| Error                              | Cause                             |
|------------------------------------|-----------------------------------|
| Expected H rows, got N             | Wrong number of rows              |
| Row Y: expected W cols, got N      | Wrong number of columns           |
| Cell (Y,X): expected 6 probs, got N| Wrong probability vector length   |
| Cell (Y,X): probs sum to S         | Probabilities don't sum to 1.0    |
| Cell (Y,X): negative probability   | Negative value in probability vector |

---

### GET /budget — Check Query Budget

```json
{
  "round_id": "uuid",
  "queries_used": 23,
  "queries_max": 50,
  "active": true
}
```

---

### GET /my-rounds — Your Rounds with Scores

Returns all rounds enriched with your team's scores, submission counts, rank, and query budget.

| Field              | Type            | Description                                                |
|--------------------|-----------------|------------------------------------------------------------|
| `round_score`      | float \| null   | Your team's average score across all seeds                 |
| `seed_scores`      | float[] \| null | Per-seed scores                                            |
| `seeds_submitted`  | int             | Number of seeds your team has submitted predictions for    |
| `rank`             | int \| null     | Your team's rank for this round                            |
| `total_teams`      | int \| null     | Total teams scored in this round                           |
| `queries_used`     | int             | Simulation queries used by your team                       |
| `queries_max`      | int             | Maximum queries allowed (default 50)                       |
| `initial_grid`     | int[][]         | Initial terrain grid for the first seed                    |

---

### GET /my-predictions/{round_id} — Your Predictions

Returns your submitted predictions with derived argmax and confidence grids.

| Field              | Type            | Description                                                |
|--------------------|-----------------|------------------------------------------------------------|
| `seed_index`       | int             | Which seed this prediction is for (0–4)                    |
| `argmax_grid`      | int[][]         | H×W grid of predicted class indices                        |
| `confidence_grid`  | float[][]       | H×W grid of confidence values (max probability per cell)   |
| `score`            | float \| null   | Score for this seed (null if not yet scored)               |
| `submitted_at`     | string \| null  | ISO 8601 timestamp of submission                           |

---

### GET /analysis/{round_id}/{seed_index} — Post-Round Analysis

Only available after a round is completed (or during scoring).

```json
{
  "prediction": [[[0.85, 0.05, ...], ...], ...],
  "ground_truth": [[[0.90, 0.03, ...], ...], ...],
  "score": 78.2,
  "width": 40,
  "height": 40,
  "initial_grid": [[10, 10, ...], ...]
}
```

| Field          | Type            | Description                                                   |
|----------------|-----------------|---------------------------------------------------------------|
| `prediction`   | float[][][]     | Your submitted H×W×6 probability tensor                       |
| `ground_truth` | float[][][]     | Actual H×W×6 probability distribution (from Monte Carlo sims) |
| `score`        | float \| null   | Your score for this seed                                      |
| `initial_grid` | int[][] \| null | Initial terrain grid for this seed                            |

**Error Codes:**

| Status | Meaning                                                              |
|--------|----------------------------------------------------------------------|
| 400    | Round not completed/scoring yet, or invalid seed_index               |
| 403    | Not on a team                                                        |
| 404    | Round not found, or no prediction submitted for this seed            |

---

### GET /leaderboard — Public Leaderboard

Each team's score is their best round score of all time (weighted by round weight).

| Field                 | Type   | Description                                            |
|-----------------------|--------|--------------------------------------------------------|
| `weighted_score`      | float  | Best `round_score × round_weight` across all rounds    |
| `rounds_participated` | int    | Total rounds this team has submitted predictions       |
| `hot_streak_score`    | float  | Average score of last 3 rounds                        |
| `is_verified`         | bool   | Whether all team members are Vipps-verified            |
| `rank`                | int    | Current leaderboard rank                               |

---

## 🚀 Quick Start (Python)

```bash
pip install requests numpy
```

```python
import requests
import numpy as np

TOKEN = "your-jwt-token"
BASE  = "https://api.ainm.no/astar-island"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# 1. Find the active round
rounds   = requests.get(f"{BASE}/rounds", headers=HEADERS).json()
round_id = next(r["id"] for r in rounds if r["status"] == "active")

# 2. Observe via viewport
obs = requests.post(f"{BASE}/simulate", headers=HEADERS, json={
    "round_id":   round_id,
    "seed_index": 0,
    "viewport_x": 0, "viewport_y": 0,
    "viewport_w": 15, "viewport_h": 15
}).json()

print(f"Queries used: {obs['queries_used']} / {obs['queries_max']}")

# 3. Build prediction (example: uniform)
H, W = 40, 40
prediction = np.ones((H, W, 6)) / 6  # uniform over 6 classes

# 4. Enforce minimum probability floor (CRITICAL!)
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)

# 5. Submit for all 5 seeds
for seed_idx in range(5):
    resp = requests.post(f"{BASE}/submit", headers=HEADERS, json={
        "round_id":   round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist()
    })
    print(f"Seed {seed_idx}: {resp.json()['status']}")
```

---

## 📋 Quick Reference

| Parameter           | Value                               |
|---------------------|-------------------------------------|
| Map size            | 40×40 cells (default)               |
| Seeds per round     | 5                                   |
| Queries per round   | 50 total (shared across all seeds)  |
| Max viewport        | 15×15 cells                         |
| Min viewport        | 5×5 cells                           |
| Prediction format   | H×W×6 tensor                        |
| Simulation duration | 50 years per simulation             |
| Prediction window   | ~2 hours 45 minutes                 |
| Rate limit simulate | 5 req/sec per team                  |
| Rate limit submit   | 2 req/sec per team                  |
| Score range         | 0–100                               |
| Leaderboard score   | Best round score of all time        |

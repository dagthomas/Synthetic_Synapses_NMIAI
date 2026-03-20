# Sandbox Data Curator — Design Spec

## Problem

The Tripletex eval dashboard needs pre-existing sandbox data to run evaluations (especially update/delete/search tasks). Currently:

1. **Entities missing**: `populate_sandbox.py` exists but isn't integrated with the dashboard
2. **Sandbox resets**: The sandbox may reset/expire, wiping all seeded data
3. **No visibility**: No way to see sandbox state from the dashboard
4. **No preflight**: Evals launch without checking if required data exists, leading to failures

## Solution

Dashboard-integrated sandbox management with three components:

1. **Sandbox tab** — visibility and manual controls
2. **Backend API** — health check, seed, clean endpoints
3. **Pre-flight check** — warns before evals if sandbox data is missing

## Architecture

### New file: `dashboard/sandbox.py`

Extracts seed/clean logic from `populate_sandbox.py` into reusable functions. Returns structured results.

```python
def check_health(client: TripletexClient) -> dict
# Returns: { connected, base_url, entities: {type: {count, ok}}, modules, bank_account_1920, ready }

def seed_entities(client: TripletexClient, types: list[str] = ["all"], clean: bool = False) -> dict
# Returns: { results: {type: {created: N, errors: []}}, total_created, total_errors }

def clean_entities(client: TripletexClient) -> dict
# Returns: { results: {type: {deleted: N, errors: []}}, total_deleted }
```

### Modified: `dashboard/app.py` — 3 new endpoints

- `GET /api/sandbox/health` — connectivity + entity inventory
- `POST /api/sandbox/seed` — body: `{types: ["all"], clean: false}`
- `POST /api/sandbox/clean` — deletes all entities

### Modified: `dashboard/static/index.html` — new Sandbox tab

**Tab position**: Between "Test Tools" and "Report"

**Layout**:
- **Status bar**: Green/red indicator, "Sandbox Ready" or "Needs Setup", base URL, refresh button
- **Entity inventory**: Card grid (8 entity types), each with name, count, status icon, per-type seed button
- **Actions**: "Seed All", "Clean & Reseed", "Clean Only" buttons, progress indicator, log output area

**Pre-flight modal** in `runSelected()`:
- Calls `GET /api/sandbox/health` before launching batch
- If `ready: false`, shows modal with [Seed & Run] [Run Anyway] [Cancel]
- "Seed & Run" calls seed endpoint, waits, then launches evals

### Modified: `populate_sandbox.py`

Refactored to import from `dashboard/sandbox.py`. Stays as CLI wrapper.

## Entity Types Tracked

| Type | Min for "ready" | Seed creates |
|------|-----------------|-------------|
| employee | 1 | 5 |
| customer | 1 | 6 |
| product | 1 | 6 |
| department | 1 | 3 |
| invoice | 1 | 3 |
| travelExpense | 1 | 3 |
| project | 1 | 2 |
| contact | 1 | 2 |

Additionally checks:
- Department module enabled
- Project economy module enabled
- Bank account set on ledger 1920

## Pre-flight Check Flow

```
User clicks "Run" in eval tab
  -> GET /api/sandbox/health
  -> If ready=true: proceed with eval batch
  -> If ready=false: show modal
       [Seed & Run]: POST /api/sandbox/seed -> wait -> run batch
       [Run Anyway]: proceed without seeding
       [Cancel]: abort
```

## No New Dependencies

Uses existing `TripletexClient`. No new packages needed.

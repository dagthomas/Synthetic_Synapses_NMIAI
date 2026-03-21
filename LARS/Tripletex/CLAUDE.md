# Tripletex AI Agent — NM i AI Competition

## What This Is

A FastAPI agent that receives Norwegian accounting task prompts and executes them against the Tripletex API using Google ADK (Agent Development Kit) with Gemini. Competition scores on correctness (did you create the right entities with right fields?) and efficiency (fewer API calls = higher bonus).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up .env (copy from .env.example)
cp .env.example .env
# Fill in: GOOGLE_API_KEY, TRIPLETEX_BASE_URL, TRIPLETEX_SESSION_TOKEN

# 3. Start the agent server
python main.py --port 8000

# 4. Test it
python test_e2e.py                          # 6 end-to-end tests
python simulator.py --task create_employee  # single simulated task
python simulator.py --batch 10             # batch of 10 random tasks
```

## Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key (for agent LLM) |
| `TRIPLETEX_BASE_URL` | Yes | Sandbox API URL (e.g. `https://kkpqfuj-amager.tripletex.dev/v2`) |
| `TRIPLETEX_SESSION_TOKEN` | Yes | Sandbox session token |
| `AGENT_API_KEY` | No | Bearer token to protect /solve endpoint |
| `GEMINI_MODEL` | No | Model name (default: `gemini-2.5-pro`) |

## Architecture

```
Request flow:
  POST /solve → main.py._run_agent()
    → tool_router.classify_task(prompt)     # classify into 1 of 30 task types
    → tool_router.select_tools(...)          # pick 2-10 tools (not all 137)
    → agent.create_agent(tools, task_type)   # focused system instruction
    → InMemoryRunner.run_async()             # ADK runs Gemini with tools
      → Gemini calls tools → TripletexClient → Tripletex REST API
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server. Endpoints: `/solve` (competition), `/solve-debug` (full details) |
| `agent.py` | LlmAgent definition. System instructions split into COMMON_PREAMBLE + per-task TASK_INSTRUCTIONS |
| `tool_router.py` | Deterministic task classifier (keyword-based) + tool selector. 30 task types mapped |
| `config.py` | GOOGLE_API_KEY, GEMINI_MODEL, MAX_AGENT_TURNS (10) |
| `tripletex_client.py` | HTTP client wrapper. Tracks call count, error count, API log. Has per-request ID cache |
| `tools/__init__.py` | `build_tools_dict()` returns {name: fn}, `build_all_tools()` returns list |
| `tools/*.py` | 28 tool modules, ~137 total tool functions |

### Tool Router (`tool_router.py`)

Classifies prompts into 30 known task types via keyword scoring (Norwegian + English + 5 other languages). Falls back to all tools if unclassified. Zero latency, no external dependencies.

- `classify_task(prompt) -> str | None` — returns task type or None
- `select_tools(task_type, all_tools_dict, has_files) -> list` — returns filtered tool list
- `TASK_TOOL_MAP` — static mapping of task type → required tool names

### TripletexClient Caching

The client caches frequently-accessed IDs per request to avoid redundant GETs:
- `default_department` — used by create_employee
- `default_division` — used by create_employment, create_project
- `company_id` — used by project PM entitlements

Use `client.get_cached(key)` / `client.set_cached(key, value)`.

## Testing

### 1. End-to-End Tests (`test_e2e.py`)

Sends real prompts through `/solve-debug`, captures tool calls and API logs, verifies via Tripletex API.

```bash
python test_e2e.py                        # run 6 tests, print summary
python test_e2e.py --report-file report.md  # save detailed markdown report
```

Tests: create_employee, create_customer, create_invoice, create_travel_expense, create_voucher, create_supplier.

Requires: agent running on port 8003 (or set AGENT_URL env var).

### 2. Simulator (`simulator.py`)

Full competition simulation: generate prompt (via Gemini) → call agent /solve → verify results → score.

```bash
python simulator.py --list                    # list all 21 task types
python simulator.py --task create_employee    # specific task
python simulator.py --lang no                 # specific language
python simulator.py --batch 5                 # 5 random tasks
python simulator.py --dry-run                 # generate only, don't call agent
python simulator.py --agent-url http://host:8000  # custom agent URL
python simulator.py --no-cleanup              # skip entity cleanup
```

Requires: agent running + TRIPLETEX_BASE_URL + TRIPLETEX_SESSION_TOKEN in .env.

### 3. Tool Tests (`test_all_tools.py`)

Tests every tool function directly against the live Tripletex sandbox API (no agent involved).

```bash
python test_all_tools.py
```

### 4. Unit Tests (`tests/`)

```bash
pytest tests/
```

### 5. Dashboard Eval Runner (`dashboard/runner.py`)

Async eval runner with concurrency limit (3). Creates DB records, calls agent, verifies, scores. Used by the dashboard app.

## Dashboard

```bash
# Start dashboard (separate from agent)
cd dashboard && python app.py
# or from root:
uvicorn dashboard.app:app --port 8001
```

Dashboard UI at http://localhost:8001. Shows eval run history, scores, tool call details. SQLite DB at `dashboard/dashboard.db`.

## Scoring Formula

```
correctness = total_points / max_points
base_score = correctness × tier_multiplier (tier 1=×1, tier 2=×2, tier 3=×3)
efficiency_bonus = min(baseline_calls / actual_calls, 1.0) × tier_multiplier
                   - (api_errors × 0.15)                    # only if correctness = 1.0
final_score = base_score + efficiency_bonus
max_possible = tier_multiplier × 2
```

Every 4xx error costs 0.15 from the efficiency bonus. Fewer API calls = higher bonus.

## Task Types (30 total)

### Tier 1 — Basic (×1 multiplier)
- `create_employee` (1 call) — includes auto-recovery on email collision
- `create_customer` (1 call)
- `create_product` (1 call)
- `create_department` (1 call)
- `create_supplier` (1 call)
- `create_contact` (2 calls) — customer + contact
- `update_employee` (2 calls) — create + update phone
- `update_customer` (2 calls)
- `update_product` (2 calls)

### Tier 2 — Multi-step (×2 multiplier)
- `create_invoice` (4 calls) — customer + product + order + invoice
- `create_multi_line_invoice` (5-6 calls) — customer + N products + order + invoice
- `create_project` (2-3 calls) — customer + [employee] + project (PM entitlements auto-handled)
- `create_travel_expense` (2 calls) — employee + travel expense
- `travel_expense_with_costs` (3-4 calls) — + cost/mileage/per diem
- `invoice_with_payment` (5 calls) — invoice workflow + register_payment
- `create_credit_note` (5 calls) — invoice workflow + credit note
- `create_employee_with_employment` (2-5 calls) — employee + employment [+ details/time/leave]
- `supplier_invoice` (2-3 calls) — supplier + incoming invoice

### Tier 3 — Complex (×3 multiplier)
- `delete_travel_expense` (2 calls) — search + delete
- `delete_customer` (2 calls) — search + delete
- `delete_supplier` (2 calls) — search + delete
- `delete_product` (2 calls) — search + delete
- `create_ledger_voucher` (1 call) — balanced postings required
- `reverse_voucher` (2 calls) — search + reverse
- `delete_invoice` (5 calls) — create invoice + credit note
- `create_opening_balance` (1 call)
- `bank_reconciliation` (2-4 calls) — extract file + bank accounts + vouchers
- `process_invoice_file` (4-5 calls) — extract file + create invoice
- `year_end` (2-4 calls)
- `salary_with_bonus` (3-4 calls)

## Key Design Decisions

1. **Deterministic routing over RAG**: 30 known task types means keyword matching beats vector search — zero latency, deterministic, no infrastructure.

2. **Tool filtering**: Gemini sees 2-10 tools instead of 137. Reduces wrong tool calls, saves ~5K tokens per request.

3. **Focused instructions**: Each task type gets a ~200 token instruction instead of the full ~2800 token instruction. 77% reduction.

4. **Auto-recovery on email collision**: `create_employee` automatically searches and returns existing employee if email already taken. Saves an agent turn.

5. **Pre-validation**: `create_voucher` validates postings balance before POST. `create_order` validates orderLines JSON before POST. Prevents 422 → retry loops.

6. **ID caching**: Default department/division/company IDs cached per request. Saves 1-3 GET calls on multi-step workflows.

7. **MAX_AGENT_TURNS = 10**: Forces focused execution. Most tasks need 1-6 turns.

## Common Issues

- **"Det finnes allerede en bruker med denne e-postadressen"**: Employee email already exists. The tool now auto-recovers by returning the existing employee.
- **Voucher postings don't balance**: Tool pre-validates and returns clear error. Amounts must sum to 0 (positive=debit, negative=credit).
- **Project manager entitlement errors**: `_ensure_employee_ready()` in projects.py handles dateOfBirth, userType, employment, and entitlements automatically.
- **Missing invoiceDueDate**: Required by Tripletex. If not in prompt, agent should set it = invoiceDate.
- **isCustomer=True**: Required for all customer creation. Baked into the tool.

## File Structure

```
Tripletex/
├── main.py                  # FastAPI server (/solve, /solve-debug)
├── agent.py                 # LlmAgent + system instructions
├── tool_router.py           # Task classifier + tool selector
├── config.py                # API keys, model, MAX_AGENT_TURNS
├── tripletex_client.py      # HTTP client with caching + logging
├── tools/                   # 28 tool modules (~137 functions)
│   ├── __init__.py          # build_tools_dict(), build_all_tools()
│   ├── employees.py         # create/update/search employee
│   ├── customers.py         # create/update/search/delete customer
│   ├── products.py          # create/update/search/delete product
│   ├── invoicing.py         # order, invoice, payment, credit note
│   ├── ledger.py            # voucher, accounts, opening balance
│   ├── projects.py          # project + PM entitlements
│   ├── travel.py            # travel expense CRUD
│   ├── travel_extras.py     # costs, mileage, per diem
│   ├── employment.py        # employment records
│   ├── employee_extras.py   # categories, next of kin, leave, hours
│   ├── contacts.py          # contact persons
│   ├── departments.py       # departments
│   ├── supplier.py          # suppliers
│   ├── supplier_invoice.py  # supplier invoice approval
│   ├── incoming_invoice.py  # incoming invoices
│   ├── bank.py              # bank accounts, reconciliation
│   ├── salary.py            # salary types, transactions
│   ├── year_end.py          # year-end, VAT returns
│   ├── common.py            # get_entity_by_id, delete_entity
│   ├── files.py             # extract_file_content (PDF/CSV/image)
│   └── ... (more)
├── sim/                     # Simulator framework
│   ├── task_definitions.py  # 21 task types with field checks
│   ├── generator.py         # Gemini-based prompt generation
│   ├── verifier.py          # 929-line verification logic
│   └── scorer.py            # Competition scoring formula
├── simulator.py             # CLI simulator runner
├── test_e2e.py              # E2E agent tests with diagnostics
├── test_all_tools.py        # Direct tool API tests
├── dashboard/               # Eval dashboard (FastAPI + SQLite)
│   ├── app.py               # Dashboard web UI
│   ├── db.py                # SQLite schema + queries
│   ├── runner.py            # Async eval runner
│   └── frontend/            # HTML/JS frontend
├── tests/                   # pytest unit tests
├── payloads/                # Saved competition request payloads
└── .env.example             # Environment variable template
```

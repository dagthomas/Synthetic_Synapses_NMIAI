# Architecture

End-to-end system architecture for the Tripletex AI accounting agent.

---

## Request Flow

```mermaid
sequenceDiagram
    participant P as Competition Platform
    participant M as main.py (FastAPI)
    participant R as tool_router.py
    participant A as agent.py (Gemini)
    participant T as tools/*.py
    participant API as Tripletex API

    P->>M: POST /solve {prompt, credentials}
    M->>R: classify_task(prompt)
    R-->>M: "create_invoice"
    M->>R: select_tools("create_invoice")
    R-->>M: [process_invoice, create_order, ...]
    M->>A: create_agent(tools, task_type)

    loop Max 10 turns
        A->>T: Call tool (e.g., create_customer)
        T->>API: POST /customer
        API-->>T: {id: 123, ...}
        T-->>A: Result
    end

    A-->>M: Final response
    M-->>P: {status: "completed"}
```

---

## Component Map

```mermaid
flowchart TB
    subgraph Entry["FastAPI Server (main.py)"]
        SOLVE[POST /solve]
        DEBUG[POST /solve-debug]
        EVENTS[GET /events — SSE]
    end

    subgraph Core["Agent Core"]
        ROUTER[tool_router.py — 30 types]
        AGENT[agent.py — Gemini 2.5 Pro]
        CLIENT[tripletex_client.py — HTTP + cache]
    end

    subgraph Tools["28 Tool Modules (137 functions)"]
        BASIC[employees, customers, products]
        WORKFLOW[invoicing, projects, travel]
        PROCESS[process_invoice, process_salary]
        LEDGER[ledger, voucher, bank]
    end

    subgraph Testing["Testing & Eval"]
        SIM[sim/ — Generator + Verifier + Scorer]
        DASH[dashboard/ — React + FastAPI + SQLite]
        FIXER[auto_fixer.py — LLM code repair]
    end

    SOLVE --> ROUTER
    ROUTER --> AGENT
    AGENT --> Tools
    Tools --> CLIENT
    CLIENT --> API[Tripletex REST API]

    SIM --> SOLVE
    DASH --> SIM
    FIXER --> DASH
```

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~600 | FastAPI server, /solve endpoint, SSE events |
| `agent.py` | ~1800 | Gemini agent, system instructions, per-task prompts |
| `tool_router.py` | ~2100 | Task classifier (keyword scoring) + tool selector |
| `tripletex_client.py` | ~250 | HTTP wrapper with auth, caching, call tracking |
| `config.py` | ~15 | Environment config |
| `static_runner.py` | ~3500 | Deterministic pipeline (LLM extract + hardcoded steps) |
| `auto_fixer.py` | ~1000 | LLM-driven self-repair |
| `simulator.py` | ~400 | CLI competition simulator |
| `tools/__init__.py` | ~100 | Tool aggregation |
| `tools/invoicing.py` | ~400 | Invoice workflow (order → invoice → payment) |
| `tools/ledger.py` | ~350 | Voucher with balanced postings |
| `tools/projects.py` | ~300 | Project + PM entitlements |
| `sim/verifier.py` | ~929 | Field-by-field verification |
| `dashboard/app.py` | ~400 | Eval dashboard API |

---

## Data Flow

All state is per-request — no shared state between submissions:

```mermaid
flowchart TB
    REQ[Incoming request] --> CACHE[Per-request ID cache]
    CACHE --> DEPT[default_department]
    CACHE --> DIV[default_division]
    CACHE --> COMP[company_id]
    CACHE --> VAT[vat_type_map]

    REQ --> LOG[Call tracking]
    LOG --> COUNT[_call_count]
    LOG --> ERRORS[_error_count]
    LOG --> APILOG[_call_log — full audit trail]
```

The `TripletexClient` pre-warms caches on init (fetching default department, division, company ID, VAT maps) to save 1-3 API calls per request.

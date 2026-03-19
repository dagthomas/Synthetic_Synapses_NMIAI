# Tripletex AI Accounting Agent — Design Spec

## Summary

Single ADK (Google Agent Development Kit) agent with Gemini 2.5 Pro that completes accounting tasks in Tripletex via a ReAct tool-calling loop. Deployed locally as a FastAPI server with cloudflared tunnel.

## Context

AINM 2026 competition task. An AI agent receives natural-language accounting prompts in 7 languages, calls the Tripletex API to execute them, and is scored on correctness and efficiency. 30 task types across 3 tiers, 56 variants each, 5-minute timeout per submission.

## Architecture

```
HTTP POST /solve
    |
FastAPI endpoint (main.py)
    |  - Receive prompt, files, credentials
    |  - Decode attachments (base64 -> disk)
    |  - Build ADK Agent with credentials injected into tools
    |
ADK Agent (Gemini 2.5 Pro, ReAct loop)
    |  - System instruction: accounting assistant + API reference
    |  - Reasons about task, selects tools, observes results, repeats
    |
    +-- Employee tools (create, update, search)
    +-- Customer tools (create, update, search)
    +-- Product tools (create, search)
    +-- Invoicing tools (order, invoice, payment, credit note)
    +-- Travel expense tools (create, delete, search)
    +-- Project tools (create)
    +-- Department tools (create)
    +-- Ledger tools (accounts, postings, vouchers — Tier 3)
    +-- File tools (extract PDF/image text)
    +-- Utility tools (get_entity_by_id)
    |
Return {"status": "completed"}
```

## Credential Injection

Per-request credentials are injected into tools via closure. Each `/solve` request creates a fresh `TripletexClient` instance, and tool functions are generated as closures that capture it:

```python
# main.py — per-request tool creation
async def solve(request: Request):
    body = await request.json()
    creds = body["tripletex_credentials"]
    client = TripletexClient(creds["base_url"], creds["session_token"])

    # Tools are closures capturing this client
    tools = build_tools(client)
    agent = Agent(name="tripletex_accountant", model="gemini-2.5-pro",
                  instruction=SYSTEM_INSTRUCTION, tools=tools)
    # ... run agent ...
```

The LLM never sees credentials. Each concurrent request gets its own client and tools.

## Efficiency Strategy

Score formula: `final_score = correctness * tier_multiplier + efficiency_bonus (if correctness = 1.0)`.
Efficiency bonus is competitive — benchmarked against other teams every 12 hours. Two factors:
- **Call efficiency**: fewer API calls vs. best known solution = higher bonus
- **Error cleanliness**: every 4xx error reduces the bonus

Design principles:
- **Plan before executing** — the system instruction tells the agent to fully parse the prompt before making any API call
- **No speculative GETs** — don't search for entities you're about to create. POST directly.
- **No verification GETs** — trust POST responses. The `get_entity_by_id` tool exists for edge cases (corrections, deletions) where you MUST look up existing data, not for verifying your own writes
- **One retry max** — on error, read the message and fix in one attempt. No retry loops.
- **System instruction encodes optimal call sequences** per task type (e.g. "create invoice = POST /customer + POST /order + POST /invoice = 3 calls")

## Module Activation

Some Tripletex features require enabling modules before use (e.g. department accounting, project modules). The agent handles this via:

- **Pre-activation in `/solve`**: Before the agent starts, the endpoint calls `POST /modules` (or equivalent) to enable all needed modules. This is a fixed cost of 1-2 API calls per submission.
- **Fallback in system instruction**: If a tool call fails with a module-not-enabled error, the agent is instructed to enable the module and retry once.
- A dedicated `enable_module(module_name)` tool is available for the agent.

## Time Budget

With a 5-minute (300s) timeout:
- Gemini 2.5 Pro call: ~3-8 seconds per ReAct iteration
- Tripletex API call: ~0.5-2 seconds per call
- Worst case per iteration: ~10 seconds (LLM + API)
- **Max turns: 25** — gives ~250s worst case, 50s buffer
- Simple Tier 1 tasks: 3-5 turns (create entity)
- Complex Tier 3 tasks: 10-20 turns (multi-step with lookups)
- At turn 20+, the system instruction encourages the agent to execute immediately rather than deliberate further

## Decisions

### Why Google ADK?
- Native Gemini integration with built-in ReAct loop
- Tool calling, retry logic, and agent lifecycle managed by framework
- Code-first Python toolkit — no YAML configs or UI builders
- Production-ready (v1.0.0 stable)

### Why Gemini 2.5 Pro?
- Strong multilingual understanding (7 languages required)
- Good at structured reasoning and API call planning
- Native vision for image attachments
- User already has API key

### Why entity-specific tools (not generic CRUD)?
- Mirrors real Tripletex task patterns for realistic simulation
- Each tool validates required fields before calling API — fewer 4xx errors
- Tools encode domain knowledge (e.g. invoice requires order first)
- ~20-25 tools total — manageable for Gemini to select from

### Why single agent (not multi-agent)?
- Simplest architecture, fastest to build
- Gemini 2.5 Pro is capable enough without routing layer
- Can refactor to multi-agent later if needed
- 5 min timeout gives ample room for ReAct iterations

## Tool Inventory

### Employees
| Tool | Signature | API |
|---|---|---|
| `create_employee` | `(firstName, lastName, email, isAdministrator?, ...)` | `POST /employee` |
| `update_employee` | `(employee_id, **fields)` | `PUT /employee/{id}` |
| `search_employees` | `(firstName?, lastName?, email?)` | `GET /employee` |

### Customers
| Tool | Signature | API |
|---|---|---|
| `create_customer` | `(name, email, isCustomer, isSupplier?, ...)` | `POST /customer` |
| `update_customer` | `(customer_id, **fields)` | `PUT /customer/{id}` |
| `search_customers` | `(name?, email?)` | `GET /customer` |

### Products
| Tool | Signature | API |
|---|---|---|
| `create_product` | `(name, priceExcludingVat, ...)` | `POST /product` |
| `search_products` | `(name?)` | `GET /product` |

### Invoicing
| Tool | Signature | API |
|---|---|---|
| `create_order` | `(customer_id, orderLines, deliveryDate, ...)` | `POST /order` |
| `create_invoice` | `(customer_id, order_ids, invoiceDate, invoiceDueDate)` | `POST /invoice` |
| `register_payment` | `(invoice_id, amount, date)` | `POST /payment` |
| `create_credit_note` | `(invoice_id)` | `POST /invoice/creditNote` |

### Travel Expenses
| Tool | Signature | API |
|---|---|---|
| `create_travel_expense` | `(employee_id, title, date, ...)` | `POST /travelExpense` |
| `delete_travel_expense` | `(travel_expense_id)` | `DELETE /travelExpense/{id}` |
| `search_travel_expenses` | `(employee_id?)` | `GET /travelExpense` |

### Projects & Departments
| Tool | Signature | API |
|---|---|---|
| `create_project` | `(name, customer_id, ...)` | `POST /project` |
| `create_department` | `(name, departmentNumber, ...)` | `POST /department` |

### Ledger (Tier 3)
| Tool | Signature | API |
|---|---|---|
| `get_ledger_accounts` | `(number?, name?)` | `GET /ledger/account` |
| `get_ledger_postings` | `(dateFrom, dateTo, ...)` | `GET /ledger/posting` |
| `create_voucher` | `(date, description, postings)` | `POST /ledger/voucher` |
| `delete_voucher` | `(voucher_id)` | `DELETE /ledger/voucher/{id}` |

### Corrections
| Tool | Signature | API |
|---|---|---|
| `delete_entity` | `(entity_type, entity_id)` | `DELETE /{entity_type}/{id}` |

### Utilities
| Tool | Signature | Notes |
|---|---|---|
| `extract_file_content` | `(filename)` | PDF via pdfplumber, fallback to Gemini vision for scanned PDFs |
| `get_entity_by_id` | `(entity_type, id)` | Generic GET — for lookups/corrections only, not verification |
| `enable_module` | `(module_name)` | Enable Tripletex modules (departments, projects, etc.) |

## System Instruction

The agent receives a system instruction containing:

1. **Role**: "You are an accounting assistant executing tasks in Tripletex"
2. **Behavior rules**:
   - Use EXACT names, emails, amounts from the prompt
   - Don't guess unspecified fields
   - On error: read the message and correct in ONE retry
   - Norwegian characters (ae, oe, aa) used as-is
   - Account starts EMPTY — create prerequisites first
3. **Compressed API reference**: endpoints, required fields, common pitfalls
4. **Task patterns**: e.g. "invoice requires: search/create customer -> create order -> create invoice"

## Error Handling

| Situation | Strategy |
|---|---|
| Tripletex 422 (validation) | Tool returns error message to agent. Agent reads and retries with corrected input. |
| Tripletex 404 | Tool returns "not found". Agent searches broader or creates entity. |
| Tripletex 401 | Fatal — bad credentials. Log and return immediately. |
| Gemini timeout/error | Retry once, then return completed (partial points better than nothing). |
| 5 min approaching | Max turns limit on ADK agent as safety net. |

## File Handling

```
files[] from request
    -> base64 decode -> save to /tmp/
    -> Agent told: "Attached file: faktura.pdf"
    -> Agent calls extract_file_content("faktura.pdf")
    -> Tool uses pdfplumber (PDF) or Gemini vision (image)
    -> Returns extracted text to agent
```

## File Structure

```
Tripletex/
+-- main.py                  # FastAPI app, /solve endpoint
+-- agent.py                 # ADK Agent setup, system instruction
+-- tools/
|   +-- __init__.py
|   +-- employees.py         # create/update/search employee
|   +-- customers.py         # create/update/search customer
|   +-- products.py          # create/search product
|   +-- invoicing.py         # order, invoice, payment, credit note
|   +-- travel.py            # travel expenses
|   +-- projects.py          # projects
|   +-- departments.py       # departments
|   +-- ledger.py            # accounts, postings, vouchers (Tier 3)
|   +-- files.py             # PDF/image extraction
|   +-- common.py            # get_entity_by_id, shared HTTP client
+-- tripletex_client.py      # Requests wrapper with auth, logging, error parsing
+-- config.py                # Env vars (GOOGLE_API_KEY, etc.)
+-- requirements.txt
```

## Dependencies

```
fastapi
uvicorn
google-adk
google-genai
pdfplumber
Pillow
requests
python-dotenv
```

## Configuration

```env
GOOGLE_API_KEY=...
AGENT_API_KEY=...    # Optional, protect /solve endpoint
```

## Concurrency

Up to 3 concurrent submissions for verified teams. Each `/solve` request is fully isolated:
- Per-request `TripletexClient` instance (no shared state)
- Per-request file directory: `/tmp/{request_uuid}/` for attachments
- Per-request ADK Agent instance
- FastAPI async + uvicorn handles concurrency natively

## Testing Strategy

- **Sandbox integration tests**: Test each tool against the persistent sandbox account. Run before competing.
- **Local test harness**: Script that sends sample prompts to `/solve` locally, simulating the competition. Include prompts in all 7 languages.
- **Logging**: All Tripletex API calls logged with request/response bodies for post-mortem analysis.
- **Replay**: Save failed submission prompts and replay them locally for debugging.

## Notes

- Tripletex `/product` does not support PUT — product updates are not possible via API
- `extract_file_content`: if pdfplumber returns empty text (scanned PDF), fall back to rendering pages as images and using Gemini vision
- FastAPI middleware validates `Authorization: Bearer` header against `AGENT_API_KEY` if configured
- `enable_module` tool available for activating Tripletex modules (departments, projects, etc.)

## Priority Order

1. `tripletex_client.py` + sandbox testing (explore the API)
2. `agent.py` with system instruction
3. Tier 1 tools: employees, customers, invoicing
4. File processing (PDF/images)
5. Tier 2 tools: payments, credit notes, projects
6. Tier 3 tools: ledger, vouchers, corrections
7. Efficiency optimization (minimize calls, zero errors)

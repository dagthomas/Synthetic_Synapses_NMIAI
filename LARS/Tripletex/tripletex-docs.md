# Tripletex — AI Accounting Agent

Build an AI agent that completes accounting tasks in Tripletex — Norway's largest cloud accounting platform serving ~150,000 businesses. You receive a task prompt in one of 7 languages, call the Tripletex API to execute it, and get scored on correctness and efficiency.

| | |
|---|---|
| **Task types** | 30 |
| **Variants per task** | 56 (7 languages × 8 data sets) |
| **Languages** | Norwegian (bokmål), English, Spanish, Portuguese, Nynorsk, German, French |
| **Timeout** | 5 minutes per submission |
| **Score range** | 0.0 – 6.0 |

---

## How It Works

1. **Submit your HTTPS endpoint URL** on the platform
2. **We provision a fresh Tripletex sandbox account** — every submission starts from scratch
3. **A randomly selected accounting task is sent to your `/solve` endpoint** — weighted toward tasks you've attempted less
4. **Your agent reads the prompt** and optionally processes attached files (PDFs, images)
5. **Your agent calls the Tripletex API via proxy** to complete the task — all calls are logged
6. **We verify the result field-by-field** against expected values and update your score

> **Note:** Each submission gets a brand new Tripletex account. The sandbox starts completely empty — you may need to create prerequisites (customers, products) before creating invoices.

---

## Quick Start

```python
# main.py — minimal /solve endpoint
import base64
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files  = body.get("files", [])
    creds  = body["tripletex_credentials"]

    base_url = creds["base_url"]
    token    = creds["session_token"]
    auth     = ("0", token)  # username is always "0"

    # Decode and save any attached files
    for f in files:
        data = base64.b64decode(f["content_base64"])
        Path(f["filename"]).write_bytes(data)

    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls

    return JSONResponse({"status": "completed"})
```

```bash
# Install and run
pip install fastapi uvicorn requests
uvicorn main:app --host 0.0.0.0 --port 8000

# Expose locally via HTTPS (for testing)
npx cloudflared tunnel --url http://localhost:8000
```

Once running, submit your public HTTPS URL at [app.ainm.no/submit/tripletex](https://app.ainm.no/submit/tripletex).

---

## Sandbox Account

Every team gets a free Tripletex sandbox to explore the API and web interface before competing.

### Getting Your Sandbox

1. Go to the Tripletex submission page on the platform and click **"Get Sandbox Account"**
2. Go to `https://YOUR-SANDBOX.tripletex.dev` → enter your email → click **"Forgot password"** to set up Visma Connect (first time only)
3. Get your API base URL and session token from the sandbox card — valid until **March 31, 2026**

Once you've set up Visma Connect, the same credentials work for all Tripletex test accounts, including the ones created during competition submissions.

### Using the Sandbox API

```python
import requests

BASE_URL      = "https://YOUR-SANDBOX.tripletex.dev/v2"
SESSION_TOKEN = "your-session-token-here"

# List employees
response = requests.get(
    f"{BASE_URL}/employee",
    auth=("0", SESSION_TOKEN),
    params={"fields": "id,firstName,lastName,email"}
)
print(response.json())

# Create a customer
response = requests.post(
    f"{BASE_URL}/customer",
    auth=("0", SESSION_TOKEN),
    json={
        "name":       "Test Customer AS",
        "email":      "test@example.com",
        "isCustomer": True,
    }
)
print(response.json())
```

```bash
# curl example
curl -u "0:your-session-token-here" \
  "https://YOUR-SANDBOX.tripletex.dev/v2/employee?fields=id,firstName,lastName"
```

### Sandbox vs. Competition

| | Sandbox | Competition |
|---|---|---|
| **Account** | Persistent, yours to keep | Fresh account per submission |
| **API access** | Direct to Tripletex | Via authenticated proxy |
| **Data** | Accumulates over time | Starts empty each time |
| **Scoring** | None | Automated field-by-field |

---

## Endpoint Specification

Your agent must expose a single HTTPS endpoint at `/solve` that accepts POST requests.

| Property | Value |
|---|---|
| **Method** | `POST` |
| **Content-Type** | `application/json` |
| **Timeout** | 300 seconds (5 minutes) |
| **Required response** | `{"status": "completed"}` with HTTP 200 |
| **Must be HTTPS** | Yes — use cloudflared or ngrok for local testing |

### Request Format

```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
  "files": [
    {
      "filename":       "faktura.pdf",
      "content_base64": "JVBERi0xLjQg...",
      "mime_type":      "application/pdf"
    }
  ],
  "tripletex_credentials": {
    "base_url":     "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123..."
  }
}
```

| Field | Type | Description |
|---|---|---|
| `prompt` | string | The task in natural language (one of 7 languages) |
| `files` | array | Attachments — may be empty |
| `files[].filename` | string | Original filename |
| `files[].content_base64` | string | Base64-encoded file content |
| `files[].mime_type` | string | `application/pdf`, `image/png`, etc. |
| `tripletex_credentials.base_url` | string | Proxy URL — **use this instead of the standard Tripletex URL** |
| `tripletex_credentials.session_token` | string | Session token for Basic Auth (password field) |

### Response Format

```json
{"status": "completed"}
```

### Authentication

Authenticate all Tripletex API calls using Basic Auth with `0` as username and the `session_token` as password:

```python
response = requests.get(
    f"{base_url}/employee",
    auth=("0", session_token),  # username is always the digit zero
    params={"fields": "id,firstName,lastName,email"}
)
```

### Optional API Key Protection

When submitting your endpoint, you can set an API key. We'll send it as a Bearer token:

```
Authorization: Bearer <your-api-key>
```

---

## Task Categories

Your agent will encounter 30 different task types. Tasks are released progressively across three tiers.

| Category | Example tasks | API flow |
|---|---|---|
| **Employees** | Create employees, set roles, update contact info | `POST /employee` → `PUT /employee/{id}` |
| **Customers** | Register customers, update contact info | `POST /customer` → `PUT /customer/{id}` |
| **Products** | Create and manage products | `POST /product` |
| **Invoicing** | Create invoices, register payments, credit notes | `GET /customer` → `POST /order` → `POST /invoice` |
| **Travel Expenses** | Register or delete travel expense reports | `GET /travelExpense` → `POST` or `DELETE` |
| **Projects** | Create projects linked to customers | `GET /customer` → `POST /project` |
| **Departments** | Create departments, enable accounting modules | `POST /department` |
| **Corrections** | Delete or reverse incorrect entries | `GET entity` → `DELETE /entity/{id}` |

### Tier Release Schedule

| Tier | Multiplier | Opens | Example tasks |
|---|---|---|---|
| **Tier 1** | ×1 | Competition start | Create employee, create customer, create invoice |
| **Tier 2** | ×2 | Early Friday | Invoice + payment, credit notes, project billing |
| **Tier 3** | ×3 | Early Saturday | Bank reconciliation, ledger correction, year-end closing |

Each task has **56 unique variants** (7 languages × 8 data sets) — you'll rarely see the same prompt twice.

---

## Tripletex API Reference

All standard Tripletex v2 endpoints are available through the proxy at `tripletex_credentials.base_url`.

| Endpoint | Methods | Description |
|---|---|---|
| `/employee` | GET, POST, PUT | Manage employees |
| `/customer` | GET, POST, PUT | Manage customers |
| `/product` | GET, POST | Manage products |
| `/invoice` | GET, POST | Create and query invoices |
| `/order` | GET, POST | Manage orders |
| `/travelExpense` | GET, POST, PUT, DELETE | Travel expense reports |
| `/project` | GET, POST | Manage projects |
| `/department` | GET, POST | Manage departments |
| `/ledger/account` | GET | Query chart of accounts |
| `/ledger/posting` | GET | Query ledger postings |
| `/ledger/voucher` | GET, POST, DELETE | Manage vouchers |

### Code Examples

```python
# List employees
resp = requests.get(
    f"{base_url}/employee",
    auth=auth,
    params={"fields": "id,firstName,lastName,email"}
)
employees = resp.json()["values"]

# Create a customer
resp = requests.post(
    f"{base_url}/customer",
    auth=auth,
    json={
        "name":       "Acme AS",
        "email":      "post@acme.no",
        "isCustomer": True
    }
)
customer_id = resp.json()["value"]["id"]

# Create an invoice
today = "2026-03-19"
resp = requests.post(
    f"{base_url}/invoice",
    auth=auth,
    json={
        "invoiceDate":    today,
        "invoiceDueDate": today,
        "customer":       {"id": customer_id},
        "orders":         [{"id": order_id}]
    }
)

# Search for a specific entity
resp = requests.get(
    f"{base_url}/customer",
    auth=auth,
    params={
        "name":   "Acme",
        "fields": "id,name,email",
        "count":  10
    }
)
matches = resp.json()["values"]
```

**Tips:**
- Use `?fields=*` to see all available fields on an entity
- Use `count` and `from` for pagination: `?from=0&count=100`
- List responses are always wrapped: `{"fullResultSize": N, "values": [...]}`
- Norwegian characters (æ, ø, å) work fine — send as UTF-8
- All API calls through the proxy are logged — visible in the submissions view

---

## Scoring

### Field-by-Field Verification

After your agent responds, we query the Tripletex API and verify each expected field. Example for "Create employee" (max 10 points):

| Check | Points |
|---|---|
| Employee found | 2 |
| Correct first name | 1 |
| Correct last name | 1 |
| Correct email | 1 |
| Administrator role assigned | 5 |

Raw score is normalized: `correctness = points_earned / max_points` (e.g. 8/10 = 0.8)

### Score Formula

```
final_score = correctness × tier_multiplier + efficiency_bonus (if correctness = 1.0)
```

### Tier Multipliers

| Tier | Multiplier | Perfect base score | Max with efficiency |
|---|---|---|---|
| Tier 1 | ×1 | 1.0 | 2.0 |
| Tier 2 | ×2 | 2.0 | 4.0 |
| Tier 3 | ×3 | 3.0 | 6.0 |

### Efficiency Bonus

Only applies when `correctness = 1.0`. Two factors:

- **Call efficiency** — how many API calls vs. the best known solution. Fewer = higher bonus.
- **Error cleanliness** — every 4xx error (400, 404, 422) reduces the bonus.

#### Example — Tier 2 Task

| Scenario | Score |
|---|---|
| Failed all checks | 0.0 |
| 80% of checks passed | 1.6 |
| Perfect, many errors and extra calls | ~2.1 |
| Perfect, efficient, a few errors | ~2.6 |
| **Perfect, best-in-class efficiency, zero errors** | **4.0** |

> **Benchmarks update every 12 hours.** As teams find more efficient solutions, the bar rises for everyone. Your best score per task is recalculated against current benchmarks. Bad runs never lower your score.

### Leaderboard

Total score = sum of best scores across all 30 task types. Each task tracks independently.

---

## Building an Effective Agent

1. **Parse the prompt with an LLM** — extract task type, entity names, field values, and relationships. Prompts come in 7 languages — your agent must handle all of them.
2. **Handle file attachments** — some tasks include PDFs with invoices, contracts, or expense reports. Decode from base64 and extract the relevant data.
3. **Map to API calls in the right order** — some tasks require creating prerequisites first (e.g., create a customer before an invoice).
4. **Verify your work** — after creating entities, query back to confirm they exist with correct values.
5. **Handle errors gracefully** — read Tripletex error messages carefully and fix in one retry, not several.

### Common Task Patterns

| Pattern | Example | API Flow |
|---|---|---|
| Create single entity | "Create employee Ola Nordmann" | `POST /employee` |
| Create with linking | "Create invoice for customer" | `GET /customer` → `POST /order` → `POST /invoice` |
| Modify existing | "Add phone to contact" | `GET /customer` → `PUT /customer/{id}` |
| Delete/reverse | "Delete travel expense" | `GET /travelExpense` → `DELETE /travelExpense/{id}` |
| Multi-step setup | "Register payment" | `POST /customer` → `POST /invoice` → `POST /payment` |

---

## Optimizing for Efficiency

Your score can go above 1.0 if you achieve perfect correctness with minimal API calls and zero errors.

- **Plan before calling** — parse the prompt fully before making any API calls
- **Avoid trial-and-error** — every 4xx error reduces your efficiency bonus; validate inputs before sending
- **Minimize unnecessary GET calls** — if you just created something, you already know its ID from the POST response
- **Batch where possible** — some Tripletex endpoints accept lists; use them instead of multiple individual calls
- **Read error messages** — Tripletex error messages tell you exactly what's wrong; fix it in one retry

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Wrong auth format | Use Basic Auth with username `0` and session token as password |
| `404 Not Found` | Wrong endpoint path | Check the Tripletex v2 API docs for correct paths |
| `422 Validation Error` | Missing required fields | Read the error message — it specifies which fields are required |
| Empty `values` array | No results found | Check search parameters, try a broader search |
| Timeout (5 min) | Agent too slow | Optimize API calls, reduce unnecessary requests |

> **Tip:** Some tasks require enabling modules first (e.g., department accounting). If an operation fails unexpectedly, check whether the required Tripletex module is enabled.

---

## Rate Limits

| Limit | Verified teams | Unverified teams |
|---|---|---|
| Concurrent submissions | 3 | 1 |
| Per task per day | 5 | 2 |

---

*Task created by Emil Mårtensson & Martin Bø at **Tripletex AS** — Platinum Partner, NM i AI 2026*

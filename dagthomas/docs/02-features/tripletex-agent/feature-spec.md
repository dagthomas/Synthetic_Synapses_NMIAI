# Tripletex AI Accounting Agent

HTTPS endpoint that receives accounting task prompts in 7 languages and executes them via the Tripletex API.

---

## Features

- HTTPS `/solve` endpoint accepting task prompts
- Multi-language prompt interpretation (nb, en, es, pt, nn, de, fr)
- 30 task types across 7 categories: employees, customers, products, invoicing, travel expenses, projects, departments
- 56 variants per task type (1,680 total scenarios)
- Tiered scoring with efficiency bonuses

---

## User Flows

### Task Execution

```mermaid
sequenceDiagram
    participant Judge as Competition Judge
    participant Agent as /solve Endpoint
    participant LLM as Language Model
    participant API as Tripletex API

    Judge->>Agent: POST /solve {prompt, session_token}
    Agent->>LLM: Interpret prompt (any of 7 languages)
    LLM-->>Agent: Structured task + parameters
    Agent->>API: Execute accounting operations
    API-->>Agent: Results
    Agent-->>Judge: Response
```

---

## Scoring

```
score_per_task = correctness * tier_multiplier + efficiency_bonus
max_per_task = 6.0
```

- **Correctness:** Binary (task completed correctly or not)
- **Tier multiplier:** Higher for harder task types
- **Efficiency bonus:** Fewer API calls = higher bonus

---

## Acceptance Criteria

- [ ] HTTPS endpoint deployed and accessible
- [ ] Basic Auth: username `0`, password = session token
- [ ] Handles all 30 task types
- [ ] Handles all 7 input languages
- [ ] Response within timeout limits
- [ ] Correct Tripletex API usage per task type

---

## Intended Architecture

```mermaid
flowchart TB
    JUDGE["Competition Judge<br/>POST /solve"] --> ENDPOINT["HTTPS Endpoint<br/>Basic Auth: 0 / session_token"]

    ENDPOINT --> PARSE["Parse prompt<br/>(7 languages)"]
    PARSE --> LLM["LLM Agent<br/>Interpret task"]
    LLM --> ROUTE{"Task Type?"}

    ROUTE -->|"Employee"| EMP["Create/update employee<br/>via Tripletex API"]
    ROUTE -->|"Invoice"| INV["Create invoice<br/>with line items"]
    ROUTE -->|"Product"| PROD["Register product<br/>with pricing"]
    ROUTE -->|"Travel"| TRAVEL["Submit travel<br/>expense report"]
    ROUTE -->|"Project"| PROJ["Create/manage<br/>project"]
    ROUTE -->|"Department"| DEPT["Manage<br/>departments"]
    ROUTE -->|"Customer"| CUST["Create/update<br/>customer record"]

    EMP --> TRIPLETEX["Tripletex REST API<br/>Basic Auth"]
    INV --> TRIPLETEX
    PROD --> TRIPLETEX
    TRAVEL --> TRIPLETEX
    PROJ --> TRIPLETEX
    DEPT --> TRIPLETEX
    CUST --> TRIPLETEX

    TRIPLETEX --> RESULT["API Response"]
    RESULT --> RESPONSE["Return to Judge"]
```

## Status

**Not started.** Requires:
1. HTTPS endpoint deployment (not a zip submission like NorgesGruppen)
2. LLM integration for prompt interpretation
3. Tripletex API client for task execution
4. Task type routing and parameter extraction

Documentation available at `docs/tripletex/` (overview, endpoint, scoring, examples, sandbox).

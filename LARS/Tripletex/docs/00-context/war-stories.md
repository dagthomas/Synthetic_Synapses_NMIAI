# War Stories — Tripletex Failures and Lessons

---

## The Invoicing Black Hole — 29 Tries, Zero Points

Tasks 10-12 and 16 (create_invoice, multi_line_invoice, create_project, credit_note) consumed 29 attempts across 4 task types with zero points earned. Despite being Tier 2 (x2 multiplier, max 4.0 each), they remain completely unscored.

**Root cause hypothesis**: The `orderLines` JSON structure or a missing required field (likely `invoiceDueDate`). The irony: Task 17 (create_employee_with_employment) scores 3.50/4.00 with similar complexity — proving the agent architecture works, just not for this specific API surface.

**Lesson**: When a task type consistently fails, the problem is in the tool implementation, not the agent. Debug the tool directly against the API before throwing more agent attempts at it.

---

## The Update Paradox — Creates Work, Updates Don't

Tasks 1-6 (creates) work well, scoring 1.2-2.0 each. Tasks 7-9 (updates) fail miserably: 0.29, 0, 0. Same entities, same API, same agent — but updating is fundamentally harder.

**Why**: Creates are idempotent-ish (email collision → return existing). Updates require:
1. Search for the entity (by what? name? ID?)
2. GET the current version number (optimistic locking)
3. Merge new fields with existing (partial update vs full replace)
4. PUT with correct version

Each step can fail, and errors compound. The prompt says "update Ola Nordmann's phone number" but doesn't give you the employee ID — you have to search and disambiguate.

**Lesson**: Update operations need their own testing pipeline, separate from creates. The search + disambiguate + version + merge chain has 4 failure modes, each needing specific handling.

---

## The VAT Trap — priceExcludingVatCurrency ONLY

Early submissions sent both `priceExcludingVatCurrency` and `priceIncludingVatCurrency` when creating products. Tripletex rejected with a cryptic 422.

**Why**: Tripletex auto-calculates the inclusive price from the VAT type. Sending both creates a conflict — which one is authoritative? The answer: always send `priceExcludingVatCurrency` only.

**Lesson**: Accounting APIs have domain-specific constraints that aren't obvious from the endpoint docs. VAT is calculated, not specified.

---

## The PM Entitlements Chain — 7 API Calls for 1 Project

Creating a project seems simple: `POST /project`. But the project manager needs:
1. A `dateOfBirth` (Tripletex uses it for identity)
2. `userType = "EXTENDED"` (not STANDARD)
3. An employment record
4. Entitlement 45 (project access)
5. Entitlement 10 (project management)
6. Entitlement 8 (project economics)

And entitlements have dependencies: 45 must be granted before 10, which must be granted before 8.

**Result**: What should be 1 API call becomes 7, and each can fail independently. The `_ensure_employee_ready()` function handles all of this with caching (`pm_ready_{id}`) to avoid redundant checks.

**Lesson**: Enterprise APIs have deep dependency chains. Map the full chain before writing the tool, and cache intermediate state aggressively.

---

## The 137-Tool Token Catastrophe

Early versions gave Gemini all 137 tool functions. This:
- Wasted ~5,000 tokens per request
- Confused the model (picked wrong tools ~30% of the time)
- Slowed response time significantly

**Fix**: The task router maps each of 30 task types to 2-10 specific tools. Token usage dropped 77%. Wrong tool calls dropped to near zero.

**Lesson**: LLMs are worse with more options, not better. Tool filtering is one of the highest-ROI optimizations possible.

---

## The Email Collision Recovery

Competition prompts sometimes create employees with emails that already exist in the sandbox. The Tripletex API returns a 422 error. Without recovery, this costs:
- 1 failed API call (-0.15 efficiency)
- 1 retry call
- Lost time

**Fix**: `create_employee` catches the collision, searches for the existing employee by email, and returns it as if it was just created. Zero errors, zero retries.

**Lesson**: In competition systems, common error paths should be handled in the tool, not by the agent. The agent shouldn't even know it happened.

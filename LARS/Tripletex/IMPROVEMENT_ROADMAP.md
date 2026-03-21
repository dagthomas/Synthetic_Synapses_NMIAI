# Tripletex Competition — Tactical Improvement Roadmap

**Current Status**: 31.06 / 124.00 points (25.0%)
**Remaining Opportunity**: 92.94 points (75.0%)
**Time to Competition**: Unknown (optimize for speed)

---

## Phase 1: Quick Wins (2-4 hours, +24 points)

### Priority 1A: Tier 3 Unscored Tasks (1 try each = max potential)

These tasks received minimal investment (1-4 tries) but offer maximum Tier 3 rewards (6.0 points each).

#### Task 22 & 23: delete_product (1 try each → 6.0 points each)
**Status**: 0/6.0 each | Tier 3 | 1 try invested | **+12 points potential**

**Hypothesis**: Delete operations follow same pattern as task 21 (delete_supplier: 1.71).

**Action Plan**:
1. Review successful task 21 implementation
2. Copy pattern to delete_product
3. Test against Tripletex sandbox
4. Expected: 1-2 iterations max

**Debug Checklist**:
- [ ] Search query syntax correct for products
- [ ] Product entity ID retrieval working
- [ ] DELETE endpoint vs PATCH endpoint
- [ ] Cleanup of related entities needed?

---

#### Task 25: reverse_voucher (3 tries → 6.0 points)
**Status**: 0/6.0 | Tier 3 | 3 tries invested | **+6.0 points potential**

**Hypothesis**: Voucher operations work (task 24: 2.25/6.0), so reverse should be simple variation.

**Action Plan**:
1. Review task 24 (create_ledger_voucher) implementation
2. Identify reverse operation in Tripletex API docs
3. Likely: search voucher by ID → call /reverse endpoint
4. Expected: 1-2 iterations

**Debug Checklist**:
- [ ] Voucher search working (should be from task 24)
- [ ] Reverse endpoint signature correct
- [ ] Reverse operation requires balanced postings?
- [ ] Is reversing different from deleting?

---

#### Task 30: year_end (4 tries → 6.0 points)
**Status**: 0/6.0 | Tier 3 | 4 tries invested | **+6.0 points potential**

**Hypothesis**: Year-end is likely a multi-step workflow. Review tasks that succeeded with ≥2 steps.

**Action Plan**:
1. Check Tripletex API docs for year_end operations
2. Likely workflow: close period → generate reports → lock accounts
3. Reference task 28 (bank_reconciliation: 1.5/6.0) for multi-step pattern
4. Expected: 2-3 iterations

**Debug Checklist**:
- [ ] Is year_end a single endpoint call or multi-step?
- [ ] Required parameters (fiscal year, company, etc.)?
- [ ] Does it require prior setup (VAT returns, etc.)?
- [ ] Can it be tested on current sandbox state?

---

### Phase 1 Summary
- **Time Investment**: 2-4 hours
- **Point Gain**: +24 points
- **New Total**: 55.06 / 124.00 (44.4%)
- **Effort/Point Ratio**: 0.1-0.2 hours per point

---

## Phase 2: Invoicing Crisis Recovery (4-6 hours, +16 points)

### The Invoicing Bottleneck

Four tasks are completely unscored despite heavy investment:

| Task | Tries | Max | Investment/Point |
|------|-------|-----|------------------|
| 10: create_invoice | 7 | 4.0 | 1.75 hrs/point |
| 11: create_multi_line_invoice | 8 | 4.0 | 2.0 hrs/point |
| 12: create_project | 6 | 4.0 | 1.5 hrs/point |
| 16: create_credit_note | 8 | 4.0 | 2.0 hrs/point |

**Total Sunk Cost**: 29 tries (~2.9 hours) with zero return.

### Root Cause Analysis

Compare with successful multi-step task: **Task 17 (create_employee_with_employment: 3.5/4.0)** — this WORKS despite similar complexity.

**Key Differences**:
- create_employee_with_employment: Employee → Employment (2 entities, clear relationship)
- create_invoice: Customer → Product → Order → Invoice (4 entities, complex relationships)

**Possible Failure Modes**:
1. **JSON Structure Error**: `orderLines` array format incorrect
   - Check exact schema vs API docs
   - Verify nested object structure

2. **Missing Required Fields**: Invoices have many optional fields
   - Tripletex may reject for missing `invoiceDueDate` or `invoiceDate`
   - Check logs for 400/422 errors

3. **Entity Dependency Issues**:
   - Product may not be findable by name (requires ID)
   - Customer reference may be stale
   - Order creation may be failing silently

4. **API Version Mismatch**:
   - /v2 API may have different endpoint signatures
   - Possible breaking changes from old implementations

### Recovery Action Plan

#### Step 1: Diagnostic (30 min)
```bash
# Review agent tool logs for tasks 10, 11, 16
# Look for patterns in API errors:
grep -A5 "create_invoice\|create_order\|POST.*invoice" agent_logs.txt

# Check for:
# - 400 Bad Request (schema error)
# - 422 Unprocessable (validation error)
# - 404 Not Found (entity not found)
# - Successful HTTP 200 but wrong entity created
```

#### Step 2: Unit Test Each Step (1 hour)
```bash
# Test each component independently:
python test_all_tools.py | grep -i "customer\|product\|order\|invoice"

# Verify:
# 1. create_customer works
# 2. create_product works
# 3. create_order works with real product/customer IDs
# 4. POST /invoice with real order ID works
```

#### Step 3: Rebuild from Working Patterns (1-2 hours)

Use task 17 as reference template:
```
Task 17 Pattern:
1. Create employee (tool: create_employee)
2. Check employment status
3. Create/update employment (tool: create_employment)
4. Return result

Invoicing Pattern:
1. Create/get customer (task 2 works: 2.0/2.0)
2. Create/get product (task 3 works: 2.0/2.0)
3. Create order with product + customer
4. Create invoice from order
5. Return result
```

**Key Question**: Does task 2 & 3 success mean create_customer and create_product return usable IDs? If yes, why do steps 3-4 fail?

#### Step 4: Focused Iteration (1-2 hours)
- Fix ONE task (e.g., task 10: create_invoice)
- Get to 1.0+ score
- Apply pattern to task 11, 12, 16
- Expected convergence: 2-3 iterations per task

### Phase 2 Success Criteria
- Task 10: ≥1.0/4.0
- Task 11: ≥1.0/4.0
- Task 12: ≥1.0/4.0
- Task 16: ≥1.0/4.0
- **Minimum Gain**: +4 points (conservative), likely +8-12

---

## Phase 3: Tier 1 Updates Fix (2-3 hours, +4 points)

### The Update Problem

Three "update" tasks have terrible performance:
- Task 7: update_employee (0.29/2.0, 9 tries) ← FAILING
- Task 8: update_customer (0/2.0, 7 tries) ← UNSCORED
- Task 9: update_product (0/2.0, 7 tries) ← UNSCORED

**Pattern**: Create operations succeed (tasks 1-6), updates fail (tasks 7-9).

### Hypothesis

Updates may require:
1. **Entity existence check**: Must retrieve entity first
2. **Specific field handling**: Only update specific fields, preserve others
3. **Version conflict resolution**: Tripletex may use optimistic locking
4. **Correct PATCH vs PUT**: Wrong HTTP method?

### Action Plan

#### Step 1: Debug task 7 (already has 0.29, so close to working)
```bash
# Search logs for update_employee attempts
# Look for patterns in what DOES work (0.29 suggests partial success)

# Test against sandbox:
# 1. Create an employee (use task 1 implementation)
# 2. Update phone number (what task 7 attempts)
# 3. Verify field updated in GET response
```

#### Step 2: Identify Working Update Pattern
- Review Tripletex API docs for /employee/{id} PATCH vs PUT
- Check if partial updates allowed or if full payload required
- Test with minimal payload (just phone) vs full payload

#### Step 3: Apply to Tasks 8, 9
- Same pattern for customer and product
- Should be faster iteration once root cause found

### Phase 3 Success Criteria
- Task 7: ≥1.0/2.0 (up from 0.29)
- Task 8: ≥1.0/2.0 (up from 0)
- Task 9: ≥1.0/2.0 (up from 0)
- **Minimum Gain**: +3 points

---

## Phase 4: Tier 3 Weak Tasks Improvement (3-4 hours, +5-10 points)

### Low-Hanging Fruit in Tier 3

Five tasks score between 0.55-1.71 but have max 6.0 potential:

| Task | Score | Max | Gap | Gap % | Potential |
|------|-------|-----|-----|-------|-----------|
| 29: process_invoice_file | 0.55 | 6.0 | 5.45 | 91% | HIGH |
| 20: delete_customer | 0.60 | 6.0 | 5.40 | 90% | HIGH |
| 21: delete_supplier | 1.71 | 6.0 | 4.29 | 72% | MED-HIGH |
| 27: create_opening_balance | 1.50 | 6.0 | 4.50 | 75% | MED-HIGH |
| 28: bank_reconciliation | 1.50 | 6.0 | 4.50 | 75% | MED-HIGH |

### Optimization Strategy

#### Task 29: process_invoice_file (0.55 → 3.0+)
**Issue**: File processing may fail on format/content
**Fix**:
- Review file extraction logic
- Test with multiple file formats
- Check error handling for edge cases

#### Task 20: delete_customer (0.60 → 3.0+)
**Issue**: Likely same as task 21 pattern but with different entity
**Fix**:
- Task 21 (delete_supplier) scores 1.71 with 1 try
- Copy pattern exactly to delete_customer
- Should get ≥1.71 immediately

#### Tasks 27, 28: Ledger/Bank Operations (1.50 → 3.0+)
**Issue**: Partial implementation but not complete
**Fix**:
- Review API response to see what's missing
- Add missing fields to get full credit
- Test against sandbox to verify

### Phase 4 Success Criteria
- Task 20: ≥1.71/6.0 (copy task 21)
- Task 29: ≥1.5/6.0 (improve from 0.55)
- Task 27: ≥2.25/6.0 (improve from 1.50)
- Task 28: ≥2.25/6.0 (improve from 1.50)
- **Minimum Gain**: +2 points, likely +5-10

---

## Phase 5: Travel Expense Variations (2-3 hours, +3.75 points)

### The Travel Expense Mess

Three related tasks show high variance:
- Task 13: create_travel_expense (1.13/4.0)
- Task 14: travel_expense_with_costs (0.25/4.0) ← CRITICAL FAILING
- Task 19: delete_travel_expense (2.05/4.0)

**Issue**: Task 14 should build on task 13, but score is 9x worse (0.25 vs 1.13).

### Root Cause
Likely: Cost/mileage/per diem fields are:
1. Wrong JSON format
2. Missing required validation
3. Breaking task 13 even though task 13 "passes"

### Action Plan

#### Step 1: Review Task 13 Success
- What fields make task 13 score 1.13?
- What's missing to get to 4.0?

#### Step 2: Debug Task 14 Failure
- Add just ONE cost line to task 13
- See if score drops (indicates cost handling breaks it)
- If yes, fix cost structure

#### Step 3: Improve Task 19
- Delete operations should be straightforward
- Likely just needs better search criteria

### Phase 5 Success Criteria
- Task 13: ≥1.5/4.0 (up from 1.13)
- Task 14: ≥1.5/4.0 (up from 0.25) ← PRIORITY
- Task 19: ≥2.5/4.0 (up from 2.05)
- **Minimum Gain**: +1.5 points, likely +3.75

---

## Cumulative Impact

| Phase | Hours | Gain | Total Score |
|-------|-------|------|-------------|
| Start | 0 | — | 31.06 |
| Phase 1 (Quick Wins) | 2-4 | +24 | 55.06 |
| Phase 2 (Invoicing) | 4-6 | +8-12 | 63-67 |
| Phase 3 (Updates) | 2-3 | +3-4 | 66-71 |
| Phase 4 (Tier 3) | 3-4 | +5-10 | 71-81 |
| Phase 5 (Travel) | 2-3 | +3-4 | 74-85 |
| **TOTAL** | **13-20** | **+43-54** | **74-85** |

**Expected Final Score**: 74-85 / 124.00 (60-68% completion)
**ROI**: 2.15-4.15 points per hour invested

---

## Critical Success Factors

1. **Diagnostic First**: Understand why each task fails before fixing
2. **Copy Working Patterns**: Task 17, 21, 2-4 work; use as templates
3. **Test in Sandbox**: Verify each fix against live API before marking complete
4. **Limit Retries**: Phase 1 should be 1-2 tries max; if not working, escalate
5. **Track Metrics**: Monitor efficiency (points/try) in each phase

---

## Testing Checklist

- [ ] Run `python simulator.py --batch 5` after each phase
- [ ] Check agent logs for new errors
- [ ] Verify localStorage metadata reflects new scores
- [ ] Backup scoring progress before major changes
- [ ] Test all phases in sandbox before production

---

## File Structure for Reference

```
C:\Users\larsh\source\repos\AINM\LARS\Tripletex\
├── agent.py              # Core agent logic
├── tool_router.py        # Task classifier
├── tools/
│   ├── common.py         # search_entity, delete_entity patterns
│   ├── invoicing.py      # create_order, create_invoice (BROKEN)
│   ├── travel.py         # create_travel_expense (PARTIALLY WORKS)
│   ├── products.py       # create_product (WORKS)
│   └── ...
├── simulator.py          # Test runner
├── test_e2e.py          # E2E debugging
└── dashboard/
    └── frontend/
        └── components/
            └── panels/
                └── score-overview-panel.tsx  # Scoring UI
```

---

**Next Steps**: Start Phase 1 immediately. Report back after quick wins phase with actual scores.

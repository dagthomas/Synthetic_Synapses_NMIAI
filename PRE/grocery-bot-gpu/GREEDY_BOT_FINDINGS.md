# Greedy Bot Findings & Iteration Log

## Problem Statement
Expert has 10 bots but GPU DP for all 10 takes ~130s cold + ~600s refine — doesn't fit in 288s token window. Solution: DP-plan 5 bots, let 5 use simpler CPU strategy.

## Iteration 1: Basic Greedy (nearest needed item)
- Each greedy bot: find nearest needed item type → navigate → pickup → deliver → repeat
- **BUG**: Passed `type_id` as `item_idx` to `ACT_PICKUP`. Game engine expects item INDEX, not type ID.
- **Result**: All greedy bots stuck — pickup silently fails, inventory stays empty.
- **Score**: Same as DP-only (greedy bots contribute nothing).

## Iteration 2: Fixed item_idx lookup + type claiming
- Added `find_adjacent_item(bx, by, type_id)` → returns correct `item_idx`
- Added type claiming (coordination so bots don't all chase same item)
- **BUG**: `dist_to_type == 0` means adjacent, used `<= 1` (1 = one step away, NOT adjacent)
- **Result**: Pickups now work (3 per bot). But...
- **CRITICAL PROBLEM**: Greedy bots pick up items that DP bots ALSO planned to pick up!
  - Creates dead inventory on greedy bots (items DP bots expected to deliver)
  - Disrupts DP bot plans (they arrive at shelf, item already picked up by greedy)
  - Score DROPPED from 121 → 54 (catastrophic interference)

## Iteration 3: Preview-only strategy
- Greedy bots ONLY fetch items for the PREVIEW order (next order, not current)
- Never touch active order items → no interference with DP bots
- When greedy bot inventory matches newly-active order → delivers naturally
- **Score: 138 cold, 168 after iterate loop** — 47 points above all-DP baseline (121)
- Greedy bots end game at x=10-20 with full inventories (far from dropoff)
- **Problem**: 168 score was partly inflated by 500K states bug (default, should be 50K)

## Iteration 4: Queue at dropoff
- Fill inventory with preview items → move to queue NEAR DROPOFF
- Only deliver when items match active order AND DP bots don't already carry them
- **Result**: Score DROPPED 168 → 139 because greedy bots congested dropoff area
- **Learning**: Bots near dropoff physically BLOCK DP bots from reaching it (collision resolution)

## Iteration 5: Preview + deliver when matching + wait when full
- Fetch preview items (diverse types, skip types already in inventory)
- Deliver when items match active order, full inventory → go to dropoff
- Per-round type claiming for coordination between greedy bots
- **Score: 150** (Expert iterate, 50K states, seed 42) — down from 168 peak

## Iteration 6: Frequency-based fetching + no active types
- **Key insight**: Instead of only targeting the single preview order, target the
  **most common item types across ALL visible orders**
- Ranking: preview match > high-frequency non-active > distance
- NEVER pick active order types (DP bots handle those)
- Full inventory + near dropoff → move AWAY (avoid congesting dropoff for DP bots)
- **Score: 121 cold, 139 iterate** — consistent with Iter 3 safe approach

## Iteration 6b: Uncovered-active strategy (FAILED)
- Tried: Let greedy bots pick active types that DP bots DON'T currently carry
- Logic: `dp_carrying` shows what DP bots hold NOW → pick what's uncovered
- **CRITICAL FLAW**: `dp_carrying` only shows what DP bots carry RIGHT NOW, not what
  they WILL pick up on their planned path. Greedy bots race to shelves and grab items
  before DP bots arrive, disrupting the entire DP plan.
- **Score: 41 cold, 90 iterate** — CATASTROPHIC regression from 139
- **Learning**: You CANNOT safely have greedy bots pick active types. Even "uncovered"
  types become covered later when DP bots execute their plans.

## Why Greedy Bots Can't Contribute More (Fundamental Limitation)
1. If greedy bots pick **active types** → interferes with DP plans → catastrophic
2. If greedy bots pick **non-active types** → they never deliver → score contribution ≈ 0
3. The ONLY contribution comes from **preview items that become active** (order cycle)
4. With 16 types on Expert, random match probability is low
5. The +18 points (121→139) from greedy bots is near the ceiling for this approach
6. **Dead inventory is permanent** — wrong pickups waste slots forever
7. **Greedy bots can't see DP bots' FUTURE plans** — only current inventory

## Iteration 7: Hybrid JIT + Preview (FAILED)
- Combined JIT adjacent pickup with preview fetching
- JIT: only pick up active types when already ADJACENT to shelf
- Preview: fetch non-active types as usual
- **Score: 128** — worse than pure safe (139)
- Even adjacent JIT pickups disrupt DP coordination (greedy delivery timing conflicts)
- **Learning**: ANY interaction with active order items hurts, even opportunistic

## Iteration 8: Pure JIT (Patrol + On-Call) (FAILED)
- Bots patrol near high-frequency shelves with EMPTY inventory
- Only pick up active types when uncovered by DP + already adjacent
- Also chase uncovered active types up to distance 8
- **Score: 88 cold, 138 iterate** — slightly worse than safe approach
- Patrol positions cause aisle congestion with DP bots
- **Learning**: Moving toward active shelves (even "uncovered") still causes collisions

## WINNING STRATEGY: Pure Safe (No Active Types)
- **Never touch active order types** (DP bots handle those)
- Fetch preview and high-frequency non-active types speculatively
- Deliver when items match active order (after order cycles)
- Stay away from dropoff when idle
- **Score: 121 cold, 139 iterate** — BEST and most consistent
- **Contribution: +18 points** over DP-only (121→139)

## Better Alternatives (Not Yet Tried)
1. **More DP bots, fewer greedy**: 7 DP + 3 greedy instead of 5+5
2. **All DP with smaller states**: 10 DP × 25K states instead of 5 × 50K states
3. **Shadow DP**: Run lightweight DP for greedy bots (constrained to avoid DP bot items)

## Pipeline Strategy Change: Quick GPU + More Replays
- User insight: "do quick GPU solves, trust bot algorithms more, replay more"
- Reduced GPU time per iteration: Expert cold 120s→45s, warm 80s→25s
- Hard cold 80s→45s, warm 60s→25s
- This allows 6-8 iterations vs 2-3 within 280s budget
- Each iteration discovers more orders → better subsequent solves
- Increased plateau tolerance: 4→8 (more iterations before giving up)

## Key Learnings
1. **ACT_PICKUP takes item_index, NOT type_id** — game_engine.py line 442
2. **dist_to_type[t,y,x] == 0 means adjacent** to a shelf of that type (can pickup)
3. **Greedy bots interfere with DP plans** — picking up active order items destroys coordination
4. **Dead inventory is permanent** — no drop action, so wrong pickups are permanent waste
5. **All 10 DP bots is too slow** for 288s but gives best quality (121) — not enough
6. **Preview-only greedy HELPS** — +47 points above DP baseline (168 vs 121)
7. **The game is about coordination** — uncoordinated bots hurt more than help
8. **Items stay on shelves (never deplete)** — multiple bots CAN pick same type from same shelf
9. **Auto-delivery**: bots AT dropoff auto-deliver to new active order when it appears
10. **Dropoff congestion kills score** — greedy bots near dropoff block DP bots, NEVER queue there
11. **Type frequency matters** — picking common types across all orders > targeting one preview order
12. **Short GPU + more replays > long GPU** — 6 quick iterations discover more orders than 2 long ones

## Score Progression (Expert, seed 42, 50K states)
| Iteration | Strategy | Cold Score | After Iterate | Notes |
|-----------|----------|-----------|---------------|-------|
| Baseline | All 10 DP | 121 | 159 (500K bug) | Too slow for 288s |
| Iter 2 | Active-order greedy | 54 | 100 | Catastrophic interference |
| Iter 3 | Preview-only greedy | 138 | 168 (500K bug) | First real improvement |
| Iter 4 | Queue at dropoff | ~120 | 139 | Dropoff congestion |
| Iter 5 | Preview + deliver + wait | ~130 | 150 | Better but not great |
| Iter 6 | Freq-based, no active | 121 | 139 | Safe, consistent |
| Iter 6b | Uncovered-active | 41 | 90 | FAILED — catastrophic |

## Score Progression (Hard, seed 42, 50K states)
| Test | Score | Budget | Notes |
|------|-------|--------|-------|
| Previous best | 200 | 574s | R→O→R→O loop, 3 orderings |
| Current (greedy safe) | 182 | 280s | No regression |
| Short budget | 174 | 180s | 2 loops, fast iterate |

## Architecture Notes
- `greedy_plan_bots()` in `gpu_sequential_solver.py`
- Co-simulates all bots: DP bots replay actions, greedy bots decide per-round
- Uses `PrecomputedTables` for O(1) navigation (step_to_type, step_to_dropoff)
- `type_adj_items` lookup maps (type_id, bot_position) → item_index for valid pickups
- `find_adjacent_item(bx, by, type_id)` → correct item_idx for ACT_PICKUP
- `type_freq` dict: frequency of each type across all orders for smart fetching
- `ranked_types`: types sorted by frequency (most common first)

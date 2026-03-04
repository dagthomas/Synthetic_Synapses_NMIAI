const std = @import("std");
const config = @import("config");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");
const precomputed = @import("precomputed.zig");

const Difficulty = config.Difficulty;
const DIFFICULTY = config.difficulty;

const Pos = types.Pos;
const ItemType = types.ItemType;
const GameState = types.GameState;
const NeedList = types.NeedList;
const DistMap = types.DistMap;
const MAX_ITEMS = types.MAX_ITEMS;
const INV_CAP = types.INV_CAP;
const UNREACHABLE = types.UNREACHABLE;

// ── Trip Planning (Mini-TSP) ──────────────────────────────────────────
pub const Candidate = struct {
    item_idx: u16,
    adj: Pos,
    is_active: bool,
    dist_from_bot: u16,
    dist_from_drop: u16,
};

pub const TripPlan = struct {
    items: [INV_CAP]u16,
    adjs: [INV_CAP]Pos,
    item_count: u8,
    total_cost: u32,
    active_count: u8,
    preview_count: u8,
    completes_order: bool,
};

pub fn planBestTrip(
    state: *const GameState,
    dm_bot: *const DistMap,
    dm_drop: *const DistMap,
    bot_active: *const NeedList,
    bot_preview: *const NeedList,
    claimed: *const [MAX_ITEMS]i8,
    bi: usize,
    slots_free: u8,
    allow_preview: bool,
    rounds_left: u32,
    bot_count: u8,
    active_order_idx: i32,
) ?TripPlan {
    const active_remaining = bot_active.count;

    var cands: [32]Candidate = undefined;
    var cand_count: u8 = 0;

    var need_active_filled: [16]u8 = undefined;
    var need_active_types: [16]ItemType = undefined;
    var need_active_len: u8 = 0;
    var need_preview_filled: [16]u8 = undefined;
    var need_preview_types: [16]ItemType = undefined;
    var need_preview_len: u8 = 0;

    for (0..bot_active.count) |i| {
        var found = false;
        for (0..need_active_len) |j| {
            if (need_active_types[j].eql(bot_active.types[i])) {
                need_active_filled[j] += 1;
                found = true;
                break;
            }
        }
        if (!found) {
            need_active_types[need_active_len] = bot_active.types[i];
            need_active_filled[need_active_len] = 1;
            need_active_len += 1;
        }
    }
    for (0..bot_preview.count) |i| {
        var found = false;
        for (0..need_preview_len) |j| {
            if (need_preview_types[j].eql(bot_preview.types[i])) {
                need_preview_filled[j] += 1;
                found = true;
                break;
            }
        }
        if (!found) {
            need_preview_types[need_preview_len] = bot_preview.types[i];
            need_preview_filled[need_preview_len] = 1;
            need_preview_len += 1;
        }
    }

    var active_needed: [16]u8 = undefined;
    @memcpy(active_needed[0..need_active_len], need_active_filled[0..need_active_len]);
    var preview_needed: [16]u8 = undefined;
    @memcpy(preview_needed[0..need_preview_len], need_preview_filled[0..need_preview_len]);

    const TempCand = struct { ii: u16, dist: u16, is_active: bool, adj: Pos, d_back: u16 };
    var all_cands: [64]TempCand = undefined;
    var all_count: u8 = 0;

    for (0..state.item_count) |ii| {
        if (all_count >= 64) break;
        if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
        const item = &state.items[ii];

        var is_active_item = false;
        for (0..need_active_len) |j| {
            if (need_active_types[j].eql(item.item_type) and active_needed[j] > 0) {
                is_active_item = true;
                break;
            }
        }
        var is_preview_item = false;
        if (!is_active_item and allow_preview) {
            for (0..need_preview_len) |j| {
                if (need_preview_types[j].eql(item.item_type) and preview_needed[j] > 0) {
                    is_preview_item = true;
                    break;
                }
            }
        }
        if (!is_active_item and !is_preview_item) continue;

        const adj = pathfinding.findBestAdj(state, item.pos, dm_bot) orelse continue;
        const ux: u16 = @intCast(adj.x);
        const uy: u16 = @intCast(adj.y);
        const d_to = dm_bot[uy][ux];
        if (d_to >= UNREACHABLE) continue;
        const d_back = dm_drop[uy][ux];

        all_cands[all_count] = .{ .ii = @intCast(ii), .dist = d_to, .is_active = is_active_item, .adj = adj, .d_back = d_back };
        all_count += 1;
    }

    // Sort candidates by per-difficulty strategy
    const use_roundtrip_sort = switch (DIFFICULTY) {
        .easy => true, // Single bot: round-trip cost
        .medium => false, // Pure bot distance (orchestrator handles round-trip)
        .hard => false,
        .expert => false, // EXPERIMENT: pure distance
        .auto => bot_count <= 1 or bot_count >= 8,
    };
    for (0..all_count) |i| {
        var min_j = i;
        for (i + 1..all_count) |j| {
            if (use_roundtrip_sort) {
                const cost_j = @as(u32, all_cands[j].dist) + @as(u32, all_cands[j].d_back);
                const cost_min = @as(u32, all_cands[min_j].dist) + @as(u32, all_cands[min_j].d_back);
                if (cost_j < cost_min) min_j = j;
            } else {
                if (all_cands[j].dist < all_cands[min_j].dist) min_j = j;
            }
        }
        if (min_j != i) {
            const tmp = all_cands[i];
            all_cands[i] = all_cands[min_j];
            all_cands[min_j] = tmp;
        }
    }

    // Pick closest items per type-slot
    var sel_active_used: [16]u8 = undefined;
    @memset(sel_active_used[0..need_active_len], 0);
    var sel_preview_used: [16]u8 = undefined;
    @memset(sel_preview_used[0..need_preview_len], 0);

    for (0..all_count) |i| {
        if (cand_count >= 16) break;
        const tc = &all_cands[i];
        const item = &state.items[tc.ii];

        // Multi-bot: allow +1 extra candidate per type for congestion alternatives
        // Single-bot: exact counts only to prevent duplicate type picking
        const extra: u8 = if (bot_count > 1) 1 else 0;

        if (tc.is_active) {
            var slot_idx: ?usize = null;
            for (0..need_active_len) |j| {
                if (need_active_types[j].eql(item.item_type) and sel_active_used[j] < active_needed[j] + extra) {
                    slot_idx = j;
                    break;
                }
            }
            if (slot_idx) |si| {
                sel_active_used[si] += 1;
            } else continue;
        } else {
            var slot_idx: ?usize = null;
            for (0..need_preview_len) |j| {
                if (need_preview_types[j].eql(item.item_type) and sel_preview_used[j] < preview_needed[j] + extra) {
                    slot_idx = j;
                    break;
                }
            }
            if (slot_idx) |si| {
                sel_preview_used[si] += 1;
            } else continue;
        }

        cands[cand_count] = .{
            .item_idx = tc.ii,
            .adj = tc.adj,
            .is_active = tc.is_active,
            .dist_from_bot = tc.dist,
            .dist_from_drop = tc.d_back,
        };
        cand_count += 1;
    }

    if (cand_count == 0) return null;

    const n: u8 = @min(cand_count, 16);

    // Distance maps from each candidate's adj position (pre-computed lookups)
    var cand_dm: [16]DistMap = undefined;
    for (0..n) |i| {
        cand_dm[i] = pathfinding.getPrecomputedDm(state, cands[i].adj).*;
    }

    // Pre-compute future utility per candidate (preview items only)
    var cand_utility: [16]u16 = .{0} ** 16;
    if (precomputed.isActive() and active_order_idx >= 0) {
        const fut_start: u16 = @intCast(@as(u32, @intCast(active_order_idx)) + 2);
        for (0..n) |i| {
            if (!cands[i].is_active) {
                cand_utility[i] = precomputed.typeFutureUtility(state.items[cands[i].item_idx].item_type, fut_start, 8);
            }
        }
    }

    var best: ?TripPlan = null;
    var best_score: u64 = 0;

    // Evaluate single-item trips
    for (0..n) |a| {
        const cost = @as(u32, cands[a].dist_from_bot) + @as(u32, cands[a].dist_from_drop);
        if (cost + 3 > rounds_left) continue;
        const ac: u8 = if (cands[a].is_active) 1 else 0;
        const pc: u8 = if (cands[a].is_active) 0 else 1;
        if (ac == 0 and active_remaining > 0) continue;
        const completes = ac >= active_remaining and active_remaining > 0;
        const score = tripScore(cost, ac, pc, 1, completes, rounds_left, bot_count, cand_utility[a]);
        if (best == null or score > best_score) {
            best = .{ .items = undefined, .adjs = undefined, .item_count = 1, .total_cost = cost, .active_count = ac, .preview_count = pc, .completes_order = completes };
            best.?.items[0] = cands[a].item_idx;
            best.?.adjs[0] = cands[a].adj;
            best_score = score;
        }
    }

    if (slots_free < 2) return best;

    // Evaluate 2-item trips
    for (0..n) |a| {
        for (a + 1..n) |b| {
            if (!cands[a].is_active and !cands[b].is_active and !allow_preview) continue;

            // Real type counting: prevents duplicate-type trips from getting inflated scores
            const pair = [2]u16{ cands[a].item_idx, cands[b].item_idx };
            const real2 = countRealTypes(state, &pair, bot_active, bot_preview, allow_preview);
            const ac = real2.ac;
            const pc = real2.pc;
            if (ac + pc < 2) continue; // Skip trips with useless duplicate items
            if (ac == 0 and active_remaining > 0) continue;
            const completes = ac >= active_remaining and active_remaining > 0;

            const ab_ux: u16 = @intCast(cands[b].adj.x);
            const ab_uy: u16 = @intCast(cands[b].adj.y);
            const ab_dist = @as(u32, cand_dm[a][ab_uy][ab_ux]);
            const ba_ux: u16 = @intCast(cands[a].adj.x);
            const ba_uy: u16 = @intCast(cands[a].adj.y);
            const ba_dist = @as(u32, cand_dm[b][ba_uy][ba_ux]);

            const cost_ab = @as(u32, cands[a].dist_from_bot) + ab_dist + @as(u32, cands[b].dist_from_drop);
            const cost_ba = @as(u32, cands[b].dist_from_bot) + ba_dist + @as(u32, cands[a].dist_from_drop);

            const pair_utility = cand_utility[a] + cand_utility[b];
            if (cost_ab + 4 <= rounds_left) {
                const score = tripScore(cost_ab, ac, pc, 2, completes, rounds_left, bot_count, pair_utility);
                if (best == null or score > best_score) {
                    const adj_b = pathfinding.findBestAdj(state, state.items[cands[b].item_idx].pos, &cand_dm[a]) orelse cands[b].adj;
                    best = .{ .items = undefined, .adjs = undefined, .item_count = 2, .total_cost = cost_ab, .active_count = ac, .preview_count = pc, .completes_order = completes };
                    best.?.items[0] = cands[a].item_idx;
                    best.?.items[1] = cands[b].item_idx;
                    best.?.adjs[0] = cands[a].adj;
                    best.?.adjs[1] = adj_b;
                    best_score = score;
                }
            }
            if (cost_ba + 4 <= rounds_left) {
                const score = tripScore(cost_ba, ac, pc, 2, completes, rounds_left, bot_count, pair_utility);
                if (best == null or score > best_score) {
                    const adj_a = pathfinding.findBestAdj(state, state.items[cands[a].item_idx].pos, &cand_dm[b]) orelse cands[a].adj;
                    best = .{ .items = undefined, .adjs = undefined, .item_count = 2, .total_cost = cost_ba, .active_count = ac, .preview_count = pc, .completes_order = completes };
                    best.?.items[0] = cands[b].item_idx;
                    best.?.items[1] = cands[a].item_idx;
                    best.?.adjs[0] = cands[b].adj;
                    best.?.adjs[1] = adj_a;
                    best_score = score;
                }
            }
        }
    }

    if (slots_free < 3) return best;

    // Evaluate 3-item trips
    const n3 = @min(n, 12);
    for (0..n3) |a| {
        for (a + 1..n3) |b| {
            for (b + 1..n3) |c| {
                // Real type counting: prevents duplicate-type trips
                const triple = [3]u16{ cands[a].item_idx, cands[b].item_idx, cands[c].item_idx };
                const real3 = countRealTypes(state, &triple, bot_active, bot_preview, allow_preview);
                const ac = real3.ac;
                const pc = real3.pc;
                if (ac + pc < 3) continue; // Skip trips with useless duplicate items
                if (ac == 0 and active_remaining > 0) continue;
                const completes = ac >= active_remaining and active_remaining > 0;

                const items3 = [3]u8{ @intCast(a), @intCast(b), @intCast(c) };
                const perms = [6][3]u8{
                    .{ 0, 1, 2 }, .{ 0, 2, 1 }, .{ 1, 0, 2 },
                    .{ 1, 2, 0 }, .{ 2, 0, 1 }, .{ 2, 1, 0 },
                };

                for (perms) |perm| {
                    const p0 = items3[perm[0]];
                    const p1 = items3[perm[1]];
                    const p2 = items3[perm[2]];
                    const d01_ux: u16 = @intCast(cands[p1].adj.x);
                    const d01_uy: u16 = @intCast(cands[p1].adj.y);
                    const d12_ux: u16 = @intCast(cands[p2].adj.x);
                    const d12_uy: u16 = @intCast(cands[p2].adj.y);
                    const cost = @as(u32, cands[p0].dist_from_bot) +
                        @as(u32, cand_dm[p0][d01_uy][d01_ux]) +
                        @as(u32, cand_dm[p1][d12_uy][d12_ux]) +
                        @as(u32, cands[p2].dist_from_drop);

                    if (cost + 5 > rounds_left) continue;
                    const triple_utility = cand_utility[a] + cand_utility[b] + cand_utility[c];
                    const score = tripScore(cost, ac, pc, 3, completes, rounds_left, bot_count, triple_utility);
                    if (best == null or score > best_score) {
                        const adj1 = pathfinding.findBestAdj(state, state.items[cands[p1].item_idx].pos, &cand_dm[p0]) orelse cands[p1].adj;
                        const adj2 = pathfinding.findBestAdj(state, state.items[cands[p2].item_idx].pos, &cand_dm[p1]) orelse cands[p2].adj;
                        best = .{ .items = undefined, .adjs = undefined, .item_count = 3, .total_cost = cost, .active_count = ac, .preview_count = pc, .completes_order = completes };
                        best.?.items[0] = cands[p0].item_idx;
                        best.?.items[1] = cands[p1].item_idx;
                        best.?.items[2] = cands[p2].item_idx;
                        best.?.adjs[0] = cands[p0].adj;
                        best.?.adjs[1] = adj1;
                        best.?.adjs[2] = adj2;
                        best_score = score;
                    }
                }
            }
        }
    }

    return best;
}

/// Count real active/preview items for a set of item indices, respecting type uniqueness.
/// Prevents trips that pick duplicate types beyond what the order needs.
fn countRealTypes(
    state: *const GameState,
    item_indices: []const u16,
    bot_active: *const NeedList,
    bot_preview: *const NeedList,
    allow_preview: bool,
) struct { ac: u8, pc: u8 } {
    var check_a = bot_active.*;
    var check_p = bot_preview.*;
    var real_ac: u8 = 0;
    var real_pc: u8 = 0;
    for (item_indices) |ii| {
        const item_type = state.items[ii].item_type;
        if (check_a.contains(item_type)) {
            check_a.remove(item_type);
            real_ac += 1;
        } else if (allow_preview and check_p.contains(item_type)) {
            check_p.remove(item_type);
            real_pc += 1;
        }
        // else: duplicate type beyond what's needed — doesn't count
    }
    return .{ .ac = real_ac, .pc = real_pc };
}

pub fn tripScore(cost: u32, ac: u8, pc: u8, count: u8, completes_order: bool, rounds_left: u32, bot_count: u8, future_utility: u16) u64 {
    if (cost == 0) return std.math.maxInt(u64);
    // Active items are worth much more than preview items
    // When completing an order, preview items auto-deliver at dropoff (worth same as active)
    const preview_val: u32 = if (completes_order) 18 else 3;
    var value: u32 = @as(u32, ac) * 20 + @as(u32, pc) * preview_val;
    // Massive bonus for completing the order (triggers +5 score bonus in-game)
    if (completes_order) value += 80;
    // Extra bonus for completion in endgame (every point matters)
    if (completes_order and rounds_left < 60) value += 20;
    // Small bonus per item count
    value += @as(u32, count) * 2;
    _ = bot_count;
    // Preview items in completing trips auto-deliver at dropoff, saving a full round-trip
    // Each preview item saved ~10 rounds of pick + deliver on the next order
    if (completes_order and pc > 0) value += @as(u32, pc) * 150;
    // Future utility bonus: prefer preview items needed in many future orders
    // Conservative: max +15 bonus (min(utility, 5) * 3)
    if (pc > 0 and future_utility > 0) {
        value += @min(future_utility, 5) * 3;
    }
    // Penalize trips that use >50% of remaining time when under 60 rounds
    if (rounds_left < 60 and cost * 2 > rounds_left) {
        value = value / 2;
    }
    // Score = value-per-cost (efficiency metric)
    return @as(u64, value) * 10000 / @as(u64, cost);
}

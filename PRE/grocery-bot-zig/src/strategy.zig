const std = @import("std");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");
const trip_mod = @import("trip.zig");

const Pos = types.Pos;
const Dir = types.Dir;
const Cell = types.Cell;
const Bot = types.Bot;
const ItemType = types.ItemType;
const MapItem = types.MapItem;
const GameState = types.GameState;
const NeedList = types.NeedList;
const DistMap = types.DistMap;
const MAX_BOTS = types.MAX_BOTS;
const MAX_ITEMS = types.MAX_ITEMS;
const MAX_H = types.MAX_H;
const INV_CAP = types.INV_CAP;
const UNREACHABLE = types.UNREACHABLE;

// ── Persistent State ───────────────────────────────────────────────────
const HIST_LEN = 24;
pub const PersistentBot = struct {
    trip_ids: [INV_CAP][32]u8,
    trip_id_lens: [INV_CAP]u8,
    trip_adjs: [INV_CAP]Pos,
    trip_count: u8,
    trip_pos: u8,
    has_trip: bool,
    delivering: bool,
    stall_count: u16,
    last_pos: Pos,
    pos_hist: [HIST_LEN]Pos,
    pos_hist_idx: u8,
    pos_hist_len: u8,
    osc_count: u16,
    last_active_order_idx: i32,
    last_tried_pickup: bool,
    last_pickup_pos: Pos,
    last_pickup_item_pos: Pos,
    last_inv_len: u8,
    rounds_on_order: u16,
    last_dir: ?Dir,
    escape_rounds: u8,
};

var pbots: [MAX_BOTS]PersistentBot = undefined;
var pbots_initialized: bool = false;

// Score tracking for diagnostic logging (Fix 6)
var last_score: i32 = 0;
var rounds_since_score_change: u32 = 0;

// Expected positions after action (for shift detection in main.zig)
pub var expected_next_pos: [MAX_BOTS]Pos = undefined;
pub var expected_count: u8 = 0;

// 1-round offset adaptation: track pending actions and offset state
var pending_dirs: [MAX_BOTS]?Dir = .{null} ** MAX_BOTS;
var pending_is_move: [MAX_BOTS]bool = .{false} ** MAX_BOTS;
pub var offset_detected: bool = false;
var offset_check_mismatches: u8 = 0;
var offset_check_rounds: u8 = 0;

pub fn initPbots() void {
    for (0..MAX_BOTS) |i| {
        pbots[i].has_trip = false;
        pbots[i].trip_count = 0;
        pbots[i].trip_pos = 0;
        pbots[i].delivering = false;
        pbots[i].stall_count = 0;
        pbots[i].last_pos = .{ .x = -1, .y = -1 };
        pbots[i].pos_hist_idx = 0;
        pbots[i].pos_hist_len = 0;
        pbots[i].osc_count = 0;
        pbots[i].last_active_order_idx = -1;
        pbots[i].last_tried_pickup = false;
        pbots[i].last_pickup_pos = .{ .x = -1, .y = -1 };
        pbots[i].last_pickup_item_pos = .{ .x = -1, .y = -1 };
        pbots[i].last_inv_len = 0;
        pbots[i].rounds_on_order = 0;
        pbots[i].last_dir = null;
        pbots[i].escape_rounds = 0;
    }
    blacklist_count = 0;
    last_score = 0;
    rounds_since_score_change = 0;
    offset_detected = false;
    offset_check_mismatches = 0;
    offset_check_rounds = 0;
    for (0..MAX_BOTS) |bi| {
        pending_dirs[bi] = null;
        pending_is_move[bi] = false;
    }
    pbots_initialized = true;
}

fn pushPosHist(pb: *PersistentBot, pos: Pos) void {
    pb.pos_hist[pb.pos_hist_idx] = pos;
    pb.pos_hist_idx = (pb.pos_hist_idx + 1) % HIST_LEN;
    if (pb.pos_hist_len < HIST_LEN) pb.pos_hist_len += 1;
}

fn isOscillating(pb: *const PersistentBot, pos: Pos) bool {
    var count: u8 = 0;
    for (0..pb.pos_hist_len) |i| {
        if (pb.pos_hist[i].eql(pos)) {
            count += 1;
            if (count >= 4) return true;
        }
    }
    return false;
}

// Count how many inventory items DON'T match active or preview needs
pub fn countDeadInventory(bot: *const Bot, active: *const NeedList, preview_nl: *const NeedList) u8 {
    var dead: u8 = 0;
    var work_a = active.*;
    var work_p = preview_nl.*;
    for (0..bot.inv_len) |ii| {
        if (work_a.contains(bot.inv[ii])) {
            work_a.remove(bot.inv[ii]);
        } else if (work_p.contains(bot.inv[ii])) {
            work_p.remove(bot.inv[ii]);
        } else {
            dead += 1;
        }
    }
    return dead;
}

// ── Pickup Blacklist ──────────────────────────────────────────────────
const MAX_BLACKLIST = 64;
var pickup_blacklist: [MAX_BLACKLIST]struct { adj: Pos, item: Pos } = undefined;
var blacklist_count: u16 = 0;

fn isBlacklisted(adj_pos: Pos, item_pos: Pos) bool {
    _ = adj_pos;
    _ = item_pos;
    // Blacklist is tracked but not used for filtering currently
    return false;
}

fn addBlacklist(adj_pos: Pos, item_pos: Pos) void {
    if (blacklist_count >= MAX_BLACKLIST) return;
    for (0..blacklist_count) |i| {
        if (pickup_blacklist[i].adj.eql(adj_pos) and pickup_blacklist[i].item.eql(item_pos)) return;
    }
    pickup_blacklist[blacklist_count] = .{ .adj = adj_pos, .item = item_pos };
    blacklist_count += 1;
    std.debug.print("Blacklisted pickup: adj ({d},{d}) -> item ({d},{d}), total={d}\n", .{ adj_pos.x, adj_pos.y, item_pos.x, item_pos.y, blacklist_count });
}

// ── Helpers ────────────────────────────────────────────────────────────
fn writeMove(writer: anytype, bot_id: u8, dir: Dir) !void {
    const s = switch (dir) {
        .up => "move_up", .down => "move_down", .left => "move_left", .right => "move_right",
    };
    try writer.print("{{\"bot\":{d},\"action\":\"{s}\"}}", .{ bot_id, s });
}

// Move away from dropoff to free it for other bots
fn fleeDropoff(state: *const GameState, pos: Pos, bot_id: u8, bot_positions: *const [MAX_BOTS]Pos) ?Dir {
    const offx = [4]i16{ 0, 0, -1, 1 };
    const offy = [4]i16{ -1, 1, 0, 0 };
    const dirs = [4]Dir{ .up, .down, .left, .right };
    for (dirs, 0..) |d, di| {
        const nx = pos.x + offx[di];
        const ny = pos.y + offy[di];
        if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
        const cell = state.grid[@intCast(ny)][@intCast(nx)];
        if (cell == .wall or cell == .shelf) continue;
        // Check bot collision
        var blocked = false;
        for (0..state.bot_count) |bi| {
            if (bi == bot_id) continue;
            if (bot_positions[bi].x == nx and bot_positions[bi].y == ny) { blocked = true; break; }
        }
        if (!blocked) return d;
    }
    return null;
}

// Escape move: pick a direction NOT in recent history and not the reverse of last_dir
fn escapeDir(state: *const GameState, pos: Pos, pb: *const PersistentBot, bot_id: u8, bot_positions: *const [MAX_BOTS]Pos) ?Dir {
    const offx = [4]i16{ 0, 0, -1, 1 };
    const offy = [4]i16{ -1, 1, 0, 0 };
    const dirs = [4]Dir{ .up, .down, .left, .right };
    const reverse_dir: ?Dir = if (pb.last_dir) |ld| switch (ld) {
        .up => .down,
        .down => .up,
        .left => .right,
        .right => .left,
    } else null;

    // Score each direction: prefer directions that lead to unvisited positions
    var best_dir: ?Dir = null;
    var best_score: i16 = -100;
    for (dirs, 0..) |d, di| {
        const nx = pos.x + offx[di];
        const ny = pos.y + offy[di];
        if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
        const cell = state.grid[@intCast(ny)][@intCast(nx)];
        if (cell == .wall or cell == .shelf) continue;
        // Check bot collision
        var blocked = false;
        for (0..state.bot_count) |bi| {
            if (bi == bot_id) continue;
            if (bot_positions[bi].x == nx and bot_positions[bi].y == ny) { blocked = true; break; }
        }
        if (blocked) continue;

        var score: i16 = 0;
        // Penalize reverse of last direction
        if (reverse_dir) |rd| {
            if (d == rd) score -= 10;
        }
        // Penalize positions in history
        const new_pos = Pos{ .x = nx, .y = ny };
        for (0..pb.pos_hist_len) |i| {
            if (pb.pos_hist[i].eql(new_pos)) score -= 3;
        }
        // Prefer moving toward center of map
        const cx = @as(i16, @intCast(state.width / 2));
        const cy = @as(i16, @intCast(state.height / 2));
        const cur_dist = @abs(pos.x - cx) + @abs(pos.y - cy);
        const new_dist = @abs(nx - cx) + @abs(ny - cy);
        if (new_dist < cur_dist) score += 2;
        if (score > best_score) { best_score = score; best_dir = d; }
    }
    return best_dir;
}

fn updateBotPos(pos: *Pos, dir: Dir) void {
    switch (dir) {
        .up => pos.y -= 1, .down => pos.y += 1, .left => pos.x -= 1, .right => pos.x += 1,
    }
}

fn findItemById(state: *const GameState, id: []const u8) ?u16 {
    for (0..state.item_count) |i| {
        if (std.mem.eql(u8, state.items[i].idStr(), id)) return @intCast(i);
    }
    return null;
}

pub fn buildNeeds(state: *const GameState) struct { active: NeedList, preview: NeedList } {
    var active = NeedList.init();
    var preview = NeedList.init();
    for (0..state.order_count) |oi| {
        const order = &state.orders[oi];
        if (order.complete) continue;
        var need_counts: [64]struct { t: ItemType, c: i8 } = undefined;
        var need_len: u8 = 0;
        for (0..order.required_len) |ri| {
            const rt = order.required[ri];
            var found = false;
            for (0..need_len) |ni| {
                if (need_counts[ni].t.eql(rt)) { need_counts[ni].c += 1; found = true; break; }
            }
            if (!found) { need_counts[need_len] = .{ .t = rt, .c = 1 }; need_len += 1; }
        }
        for (0..order.delivered_len) |di| {
            const dt = order.delivered[di];
            for (0..need_len) |ni| {
                if (need_counts[ni].t.eql(dt)) { need_counts[ni].c -= 1; break; }
            }
        }
        const target = if (order.is_active) &active else &preview;
        for (0..need_len) |ni| {
            var c = need_counts[ni].c;
            while (c > 0) : (c -= 1) target.add(need_counts[ni].t);
        }
    }
    return .{ .active = active, .preview = preview };
}

// Advance trip when an item is picked up
fn advanceTrip(pb: *PersistentBot, state: *const GameState, picked_item: *const MapItem) void {
    if (pb.trip_pos < pb.trip_count) {
        const trip_id = pb.trip_ids[pb.trip_pos][0..pb.trip_id_lens[pb.trip_pos]];
        if (std.mem.eql(u8, picked_item.idStr(), trip_id)) {
            pb.trip_pos += 1;
            if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
            return;
        }
    }
    for (pb.trip_pos..pb.trip_count) |ti| {
        const tid = pb.trip_ids[ti][0..pb.trip_id_lens[ti]];
        const tidx = findItemById(state, tid);
        if (tidx) |tidx_val| {
            if (state.items[tidx_val].item_type.eql(picked_item.item_type)) {
                var si = ti;
                while (si + 1 < pb.trip_count) : (si += 1) {
                    pb.trip_ids[si] = pb.trip_ids[si + 1];
                    pb.trip_id_lens[si] = pb.trip_id_lens[si + 1];
                    pb.trip_adjs[si] = pb.trip_adjs[si + 1];
                }
                pb.trip_count -= 1;
                if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
                return;
            }
        }
    }
}

// ── Decision Engine ────────────────────────────────────────────────────
pub fn decideActions(state: *GameState, out_buf: []u8) ![]const u8 {
    if (!pbots_initialized) initPbots();

    const needs = buildNeeds(state);
    const active_orig = needs.active;
    const preview_orig = needs.preview;
    // pick_remaining: active items that still need picking from shelves
    // (subtracts items already in ANY bot's inventory)
    var pick_remaining = needs.active;
    for (0..state.bot_count) |bri| {
        for (0..state.bots[bri].inv_len) |ii| {
            pick_remaining.remove(state.bots[bri].inv[ii]);
        }
    }
    var preview = needs.preview;
    const rounds_left = if (state.max_rounds > state.round) state.max_rounds - state.round else 0;

    // Count bots carrying ONLY preview items (dead inventory) to limit preview picking
    var bots_with_preview_only: u8 = 0;
    for (0..state.bot_count) |bri| {
        if (state.bots[bri].inv_len == 0) continue;
        var has_any_active = false;
        var check_active = needs.active;
        for (0..state.bots[bri].inv_len) |ii| {
            if (check_active.contains(state.bots[bri].inv[ii])) {
                has_any_active = true;
                check_active.remove(state.bots[bri].inv[ii]);
            }
        }
        if (!has_any_active) bots_with_preview_only += 1;
    }
    // Max bots allowed to carry preview items: half the fleet for 5+ bots
    const max_preview_carriers: u8 = if (state.bot_count <= 2) state.bot_count else 1;

    // ── Fix 6: Track score changes for diagnostic logging ──
    if (state.score != last_score) {
        rounds_since_score_change = 0;
        last_score = state.score;
    } else {
        rounds_since_score_change += 1;
    }

    // ── Detect 1-round action offset ──────────────────────────────────
    // Offset appears mid-game unpredictably. False positives are catastrophic (score 35 vs 96).
    // Skip first 30 rounds (bots crowded, collisions cause false mismatches).
    // Require 5+ moving bots with 4+ mismatches for 3 consecutive rounds.
    if (expected_count > 0 and state.round >= 30 and !offset_detected) {
        var moving_mismatches: u8 = 0;
        var moving_count: u8 = 0;
        const check_count = @min(expected_count, state.bot_count);
        for (0..check_count) |bi| {
            if (pending_is_move[bi]) {
                moving_count += 1;
                if (!expected_next_pos[bi].eql(state.bots[bi].pos)) {
                    moving_mismatches += 1;
                }
            }
        }
        if (moving_count >= 5 and moving_mismatches >= 4) {
            offset_check_mismatches += 1;
        } else {
            if (offset_check_mismatches > 0) offset_check_mismatches -|= 1;
        }
        offset_check_rounds += 1;
        if (offset_check_mismatches >= 3) {
            offset_detected = true;
            std.debug.print("R{d} OFFSET MODE ENABLED ({d}/{d} moving mismatches)\n", .{ state.round, moving_mismatches, moving_count });
        }
    }

    // Compute effective positions: where bots will be when our action takes effect
    var eff_pos: [MAX_BOTS]Pos = undefined;
    for (0..state.bot_count) |bi| {
        if (offset_detected and pending_is_move[bi]) {
            if (pending_dirs[bi]) |pd| {
                var fp = state.bots[bi].pos;
                switch (pd) {
                    .up => fp.y -= 1,
                    .down => fp.y += 1,
                    .left => fp.x -= 1,
                    .right => fp.x += 1,
                }
                // Validate: must be walkable
                if (fp.x >= 0 and fp.y >= 0 and fp.x < state.width and fp.y < state.height) {
                    const cell = state.grid[@intCast(fp.y)][@intCast(fp.x)];
                    if (cell == .floor or cell == .dropoff) {
                        eff_pos[bi] = fp;
                    } else {
                        eff_pos[bi] = state.bots[bi].pos;
                    }
                } else {
                    eff_pos[bi] = state.bots[bi].pos;
                }
            } else {
                eff_pos[bi] = state.bots[bi].pos;
            }
        } else {
            eff_pos[bi] = state.bots[bi].pos;
        }
    }

    // ── Distance from dropoff (computed BEFORE order_stuck for Fix 2 reachability) ──
    var dm_drop: DistMap = undefined;
    pathfinding.bfsDistMap(state, state.dropoff, &dm_drop);

    // ── Detect stuck order (Fix 2: add reachability check) ──
    var order_stuck = false;
    {
        var temp = active_orig;
        for (0..state.bot_count) |bi| {
            const b = &state.bots[bi];
            for (0..b.inv_len) |ii| temp.remove(b.inv[ii]);
        }
        for (0..temp.count) |ni| {
            var found_reachable = false;
            for (0..state.item_count) |ii| {
                if (state.items[ii].item_type.eql(temp.types[ni])) {
                    // Fix 2: Check that the item has a reachable adjacent floor cell
                    if (pathfinding.findBestAdj(state, state.items[ii].pos, &dm_drop) != null) {
                        found_reachable = true;
                        break;
                    }
                }
            }
            if (!found_reachable) { order_stuck = true; break; }
        }
    }

    // Bot positions for collision (use effective positions)
    var bot_positions: [MAX_BOTS]Pos = undefined;
    for (0..state.bot_count) |bi| bot_positions[bi] = eff_pos[bi];

    // Claimed items
    var claimed: [MAX_ITEMS]i8 = undefined;
    @memset(claimed[0..state.item_count], -1);

    // Track orchestrator assignments per bot (hoisted for use in per-bot loop)
    var orch_active_assigned: [MAX_BOTS]u8 = undefined;
    @memset(orch_active_assigned[0..state.bot_count], 0);

    const current_active_order_idx = state.active_order_idx;

    // Detect stalls, oscillations, and validate persistent trips
    for (0..state.bot_count) |bi| {
        const bot = &state.bots[bi];
        const pb = &pbots[bi];

        pushPosHist(pb, bot.pos);

        if (pb.last_pos.eql(bot.pos)) {
            pb.stall_count += 1;
        } else {
            pb.stall_count = 0;
            // Derive last direction from position change
            const dx = bot.pos.x - pb.last_pos.x;
            const dy = bot.pos.y - pb.last_pos.y;
            if (dx == 1) { pb.last_dir = .right; }
            else if (dx == -1) { pb.last_dir = .left; }
            else if (dy == 1) { pb.last_dir = .down; }
            else if (dy == -1) { pb.last_dir = .up; }
        }
        pb.last_pos = bot.pos;

        // Stall reset (same position for 8+ rounds)
        if (pb.stall_count > 8) {
            pb.has_trip = false;
            pb.delivering = false;
            pb.stall_count = 0;
            pb.osc_count += 1;
        }

        // Oscillation detection with escape mechanism
        if (isOscillating(pb, bot.pos)) {
            pb.osc_count += 1;
            if (pb.osc_count >= 6) {
                pb.has_trip = false;
                pb.delivering = false;
                pb.escape_rounds = 4; // Force escape movement for 4 rounds
                pb.osc_count = 0;
                std.debug.print("R{d} Bot{d} ESCAPE at ({d},{d})\n", .{ state.round, bot.id, bot.pos.x, bot.pos.y });
            }
        } else if (pb.osc_count > 0) {
            pb.osc_count -|= 1;
        }
        if (pb.escape_rounds > 0) pb.escape_rounds -= 1;

        // Detect failed pickups (diagnostic only, NOT used for offset detection)
        if (pb.last_tried_pickup) {
            if (bot.inv_len == pb.last_inv_len) {
                std.debug.print("R{d} Bot{d} pickup FAILED at ({d},{d}) -> item ({d},{d}) inv={d}\n", .{
                    state.round, bot.id, pb.last_pickup_pos.x, pb.last_pickup_pos.y,
                    pb.last_pickup_item_pos.x, pb.last_pickup_item_pos.y, bot.inv_len,
                });
                addBlacklist(pb.last_pickup_pos, pb.last_pickup_item_pos);
            }
            pb.last_tried_pickup = false;
        }

        // Detect order change: reset trip plans
        if (pb.last_active_order_idx != current_active_order_idx) {
            pb.has_trip = false;
            pb.delivering = false;
            pb.osc_count = 0;
            pb.rounds_on_order = 0; // Fix 5: reset counter on order change
        }
        pb.last_active_order_idx = current_active_order_idx;

        // Fix 5: Track rounds on current order
        pb.rounds_on_order +|= 1;

        // Force reset after being stuck on same order too long
        if (pb.rounds_on_order > 30 and pb.rounds_on_order % 15 == 0) {
            pb.has_trip = false;
            pb.delivering = false;
            pb.osc_count = 0;
            std.debug.print("R{d} Bot{d} stuck-order reset (rounds_on_order={d})\n", .{ state.round, bot.id, pb.rounds_on_order });
        }

        // Validate persistent trip: do all remaining items still exist?
        if (pb.has_trip) {
            var valid = true;
            for (pb.trip_pos..pb.trip_count) |ti| {
                const id = pb.trip_ids[ti][0..pb.trip_id_lens[ti]];
                const idx = findItemById(state, id);
                if (idx == null) {
                    valid = false;
                    break;
                } else {
                    claimed[idx.?] = @intCast(bi);
                }
            }
            if (!valid) pb.has_trip = false;
        }
    }

    // ── Pre-compute BFS distance maps for all bots (from effective positions) ──
    var all_dm_bot: [MAX_BOTS]DistMap = undefined;
    for (0..state.bot_count) |bi| {
        pathfinding.bfsDistMap(state, eff_pos[bi], &all_dm_bot[bi]);
    }

    // ── Dropoff priority: rank delivering bots by distance ──────────────
    // Computed BEFORE orchestrator so it can exclude priority-delivering bots from assignment.
    const MAX_DROPOFF_ACTIVE: u8 = if (state.bot_count >= 6) 3 else if (state.bot_count > 1) 2 else 1;
    var dropoff_priority: [MAX_BOTS]bool = undefined;
    @memset(dropoff_priority[0..state.bot_count], false);
    if (state.bot_count > 1) {
        const DP = struct { bi: u8, dist: u16 };
        var delivering_bots: [MAX_BOTS]DP = undefined;
        var del_count: u8 = 0;
        for (0..state.bot_count) |bi| {
            if (!pbots[bi].delivering) continue;
            var ba = active_orig;
            var ha = false;
            for (0..state.bots[bi].inv_len) |ii| {
                if (ba.contains(state.bots[bi].inv[ii])) { ha = true; ba.remove(state.bots[bi].inv[ii]); }
            }
            if (!ha) continue;
            const d = pathfinding.distFromMap(&dm_drop, eff_pos[bi]);
            delivering_bots[del_count] = .{ .bi = @intCast(bi), .dist = d };
            del_count += 1;
        }
        for (0..del_count) |i| {
            var min_j = i;
            for (i + 1..del_count) |j| {
                if (delivering_bots[j].dist < delivering_bots[min_j].dist) min_j = j;
            }
            if (min_j != i) {
                const tmp = delivering_bots[i];
                delivering_bots[i] = delivering_bots[min_j];
                delivering_bots[min_j] = tmp;
            }
        }
        for (0..@min(del_count, MAX_DROPOFF_ACTIVE)) |i| {
            dropoff_priority[delivering_bots[i].bi] = true;
        }
        for (0..state.bot_count) |bi| {
            const d = pathfinding.distFromMap(&dm_drop, eff_pos[bi]);
            if (d <= 1) dropoff_priority[bi] = true;
        }
    } else {
        dropoff_priority[0] = true;
    }

    // ── Orchestrator: Global item-to-bot assignment ─────────────────────
    // Assigns each needed active item to the closest available bot,
    // preventing multiple bots from redundantly chasing the same types.
    const TypeTrack = struct { t: ItemType, needed: u8, assigned: u8 };
    if (state.bot_count > 1) {
        var bot_assigned: [MAX_BOTS]u8 = undefined;
        @memset(bot_assigned[0..state.bot_count], 0);

        // Count existing claims per bot (from persistent trips)
        for (0..state.item_count) |ii| {
            if (claimed[ii] >= 0) {
                const ci: u8 = @intCast(claimed[ii]);
                if (ci < state.bot_count) bot_assigned[ci] += 1;
            }
        }

        // Build type tracking: how many of each type we still need
        var type_track: [16]TypeTrack = undefined;
        var type_track_len: u8 = 0;
        {
            var tmp = pick_remaining;
            while (tmp.count > 0) {
                const t = tmp.types[0];
                var c: u8 = 0;
                var j: u8 = 0;
                while (j < tmp.count) {
                    if (tmp.types[j].eql(t)) {
                        c += 1;
                        tmp.types[j] = tmp.types[tmp.count - 1];
                        tmp.count -= 1;
                    } else {
                        j += 1;
                    }
                }
                type_track[type_track_len] = .{ .t = t, .needed = c, .assigned = 0 };
                type_track_len += 1;
            }
        }

        // Greedily assign: for each type, find closest (bot, item) pair
        for (0..type_track_len) |ti| {
            while (type_track[ti].assigned < type_track[ti].needed) {
                var best_dist: u16 = UNREACHABLE;
                var best_item: u16 = 0;
                var best_bot: u8 = 0;
                var found = false;

                for (0..state.item_count) |ii| {
                    if (claimed[ii] >= 0) continue;
                    if (!state.items[ii].item_type.eql(type_track[ti].t)) continue;

                    for (0..state.bot_count) |bk| {
                        // Skip bots delivering WITH dropoff priority (actively heading to dropoff)
                        // Bots delivering WITHOUT priority can still pick up items
                        if (pbots[bk].delivering and dropoff_priority[bk]) continue;
                        const free_slots = if (INV_CAP > state.bots[bk].inv_len) INV_CAP - state.bots[bk].inv_len else 0;
                        if (bot_assigned[bk] >= free_slots) continue;

                        const adj = pathfinding.findBestAdj(state, state.items[ii].pos, &all_dm_bot[bk]) orelse continue;
                        const d = all_dm_bot[bk][@intCast(adj.y)][@intCast(adj.x)];
                        if (d < best_dist) {
                            best_dist = d;
                            best_item = @intCast(ii);
                            best_bot = @intCast(bk);
                            found = true;
                        }
                    }
                }

                if (!found) break;
                claimed[best_item] = @intCast(best_bot);
                bot_assigned[best_bot] += 1;
                type_track[ti].assigned += 1;
            }

            // Block excess unclaimed instances of fully-covered types
            // Uses MAX_BOTS as sentinel (no real bot has this id, so all bots skip it)
            if (type_track[ti].assigned >= type_track[ti].needed) {
                for (0..state.item_count) |ii| {
                    if (claimed[ii] >= 0) continue;
                    if (state.items[ii].item_type.eql(type_track[ti].t)) {
                        claimed[ii] = @intCast(MAX_BOTS);
                    }
                }
            }
        }

        // Record active assignments per bot
        @memcpy(orch_active_assigned[0..state.bot_count], bot_assigned[0..state.bot_count]);

        // ── Phase 2: Assign preview items to empty idle bots ──────────
        // Severely limited to prevent dead inventory buildup.
        // With many bots, at most 1 bot picks preview.
        var total_preview_assigned: u8 = 0;
        const max_orch_preview: u8 = if (state.bot_count <= 2) 4 else if (state.bot_count <= 4) 2 else @min(preview.count, state.bot_count / 2);
        if (preview.count > 0 and bots_with_preview_only < max_preview_carriers) {
            var prev_track: [16]TypeTrack = undefined;
            var prev_track_len: u8 = 0;
            {
                var tmp = preview;
                while (tmp.count > 0) {
                    const t = tmp.types[0];
                    var c: u8 = 0;
                    var j: u8 = 0;
                    while (j < tmp.count) {
                        if (tmp.types[j].eql(t)) {
                            c += 1;
                            tmp.types[j] = tmp.types[tmp.count - 1];
                            tmp.count -= 1;
                        } else {
                            j += 1;
                        }
                    }
                    prev_track[prev_track_len] = .{ .t = t, .needed = c, .assigned = 0 };
                    prev_track_len += 1;
                }
            }
            for (0..prev_track_len) |ti| {
                if (total_preview_assigned >= max_orch_preview) break;
                while (prev_track[ti].assigned < prev_track[ti].needed) {
                    if (total_preview_assigned >= max_orch_preview) break;
                    var best_dist: u16 = UNREACHABLE;
                    var best_item: u16 = 0;
                    var best_bot: u8 = 0;
                    var found = false;
                    for (0..state.item_count) |ii| {
                        if (claimed[ii] >= 0) continue;
                        if (!state.items[ii].item_type.eql(prev_track[ti].t)) continue;
                        for (0..state.bot_count) |bk| {
                            if (pbots[bk].delivering) continue;
                            if (orch_active_assigned[bk] > 0) continue;
                            // Allow preview bots to fill inventory up to INV_CAP
                            const prev_free = if (INV_CAP > state.bots[bk].inv_len) INV_CAP - state.bots[bk].inv_len else 0;
                            if (bot_assigned[bk] >= prev_free) continue;
                            const adj = pathfinding.findBestAdj(state, state.items[ii].pos, &all_dm_bot[bk]) orelse continue;
                            const d = all_dm_bot[bk][@intCast(adj.y)][@intCast(adj.x)];
                            if (d < best_dist) {
                                best_dist = d;
                                best_item = @intCast(ii);
                                best_bot = @intCast(bk);
                                found = true;
                            }
                        }
                    }
                    if (!found) break;
                    claimed[best_item] = @intCast(best_bot);
                    bot_assigned[best_bot] += 1;
                    prev_track[ti].assigned += 1;
                    total_preview_assigned += 1;
                }
            }
        }

        if (state.round % 25 == 0) {
            var total_active: u8 = 0;
            var total_preview: u8 = 0;
            for (0..state.bot_count) |bk| {
                total_active += orch_active_assigned[bk];
                total_preview += bot_assigned[bk] -| orch_active_assigned[bk];
            }
            std.debug.print("R{d} ORCH: {d} types, {d} active + {d} preview assigned\n", .{ state.round, type_track_len, total_active, total_preview });
        }
    }

    var stream = std.io.fixedBufferStream(out_buf);
    const writer = stream.writer();
    try writer.writeAll("{\"actions\":[");

    if (state.round % 25 == 0) {
        std.debug.print("R{d} Score:{d} items_on_map={d} stuck={any} osc={d}\n", .{ state.round, state.score, state.item_count, order_stuck, pbots[0].osc_count });
    }

    for (0..state.bot_count) |bi| {
        if (bi > 0) try writer.writeAll(",");
        const bot = &state.bots[bi];
        const pb = &pbots[bi];
        const bpos = eff_pos[bi]; // Effective position (accounts for offset)

        pb.last_tried_pickup = false;

        const dm_bot = &all_dm_bot[bi];

        // Build per-bot need: active minus this bot's inventory
        var bot_active = active_orig;
        var has_active = false;
        for (0..bot.inv_len) |ii| {
            if (bot_active.contains(bot.inv[ii])) {
                has_active = true;
                bot_active.remove(bot.inv[ii]);
            }
        }
        var bot_preview = preview_orig;
        for (0..bot.inv_len) |ii| bot_preview.remove(bot.inv[ii]);

        // Preview strategy: only allow limited bots to pick preview items.
        // With many bots, preview items clog inventory and prevent active picking.
        // drop_off only removes matching items, so dead inventory is permanent!
        const allow_preview_for_bot = (orch_active_assigned[bi] == 0) and bot.inv_len == 0 and
            (state.bot_count <= 2 or bots_with_preview_only < max_preview_carriers);

        // Re-validate trip against current needs
        if (pb.has_trip and pb.trip_pos < pb.trip_count) {
            var check_a = bot_active;
            var check_p = bot_preview;
            var trip_still_valid = true;
            for (pb.trip_pos..pb.trip_count) |ti| {
                const id = pb.trip_ids[ti][0..pb.trip_id_lens[ti]];
                const idx = findItemById(state, id);
                if (idx == null) { trip_still_valid = false; break; }
                const it = state.items[idx.?].item_type;
                if (check_a.contains(it)) {
                    check_a.remove(it);
                } else if (allow_preview_for_bot and check_p.contains(it)) {
                    check_p.remove(it);
                } else {
                    trip_still_valid = false;
                    break;
                }
            }
            if (!trip_still_valid) {
                pb.has_trip = false;
            }
        }

        // ─── 1. At dropoff → drop off ───────────────────────────────
        // Only drop off if bot has items matching the active order (drop_off ignores non-matching items)
        if (bpos.eql(state.dropoff) and bot.inv_len > 0 and has_active) {
            try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
            pb.has_trip = false;
            pb.delivering = false;
            pending_is_move[bi] = false;
            pending_dirs[bi] = null;
            continue;
        }

        // ─── 1b. At dropoff but shouldn't be → evacuate ──
        if (bpos.eql(state.dropoff) and !has_active) {
            const flee_dir = fleeDropoff(state, bpos, @intCast(bi), &bot_positions);
            if (flee_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                pending_dirs[bi] = d;
                pending_is_move[bi] = true;
                continue;
            }
            try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
            pending_is_move[bi] = false;
            pending_dirs[bi] = null;
            continue;
        }

        // ─── 1c. Escape oscillation ────
        if (pb.escape_rounds > 0) {
            var esc_picked = false;
            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                const mdist = @abs(bpos.x - item.pos.x) + @abs(bpos.y - item.pos.y);
                if (mdist != 1) continue;
                if (bot.inv_len >= INV_CAP) break;
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                const esc_is_active = pick_remaining.contains(item.item_type);
                const esc_is_preview = allow_preview_for_bot and bot_preview.contains(item.item_type);
                if (!esc_is_active and !esc_is_preview) continue;
                try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                if (esc_is_active) pick_remaining.remove(item.item_type) else preview.remove(item.item_type);
                claimed[ii] = @intCast(bi);
                esc_picked = true;
                pending_is_move[bi] = false;
                pending_dirs[bi] = null;
                break;
            }
            if (!esc_picked) {
                const esc = escapeDir(state, bpos, pb, @intCast(bi), &bot_positions);
                if (esc) |d| {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    pending_dirs[bi] = d;
                    pending_is_move[bi] = true;
                    continue;
                }
            } else {
                continue;
            }
        }

        // ─── 2. Adjacent needed item → pick up ──────────────────────
        // Use bpos (effective position) for adjacency check
        var picked = false;
        for (0..2) |pass| {
            if (picked) break;
            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                const mdist = @abs(bpos.x - item.pos.x) + @abs(bpos.y - item.pos.y);
                if (mdist != 1) continue;
                if (bot.inv_len >= INV_CAP) break;
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;

                if (pass == 0) {
                    if (!pick_remaining.contains(item.item_type)) continue;
                } else {
                    if (!allow_preview_for_bot) continue;
                    if (!bot_preview.contains(item.item_type)) continue;
                }

                try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                pb.last_tried_pickup = true;
                pb.last_pickup_pos = bpos;
                pb.last_pickup_item_pos = item.pos;
                pb.last_inv_len = bot.inv_len;
                pick_remaining.remove(item.item_type);
                preview.remove(item.item_type);
                claimed[ii] = @intCast(bi);
                pending_is_move[bi] = false;
                pending_dirs[bi] = null;

                if (pb.has_trip) {
                    advanceTrip(pb, state, item);
                }

                picked = true;
                break;
            }
        }
        if (picked) continue;

        // ─── 3. Should we deliver? ──────────────────────────────────
        var active_on_map: u8 = 0;
        {
            var temp = pick_remaining;
            for (0..state.item_count) |ii| {
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                if (temp.contains(state.items[ii].item_type)) {
                    active_on_map += 1;
                    temp.remove(state.items[ii].item_type);
                }
            }
        }

        const inv_full = bot.inv_len >= INV_CAP;
        const all_active_got = pick_remaining.count == 0;
        const no_active_avail = active_on_map == 0;
        const trip_done = pb.has_trip and pb.trip_pos >= pb.trip_count;
        const effective_slots = if (INV_CAP > bot.inv_len) INV_CAP - bot.inv_len else 0;

        const dist_to_drop = pathfinding.distFromMap(&dm_drop, bpos);
        const endgame = rounds_left <= dist_to_drop + 3;

        // With many bots, don't rush to deliver with 1 item if far from dropoff
        // Let closer bots handle delivery while far bots pick preview items
        const far_with_few = state.bot_count >= 5 and !inv_full and dist_to_drop > 8 and bot.inv_len < 2 and all_active_got and !endgame;

        const should_deliver = has_active and !far_with_few and (inv_full or all_active_got or no_active_avail or order_stuck or endgame or trip_done);

        if (should_deliver) {
            pb.delivering = true;
            pb.has_trip = false;
        }
        // Far bots with few items: stop delivering, go pick preview instead
        if (far_with_few and pb.delivering) {
            pb.delivering = false;
        }

        if (!has_active) {
            pb.delivering = false;
        }

        // Diagnostic logging when score stalls
        if (rounds_since_score_change >= 10 and state.round % 5 == 0) {
            std.debug.print("R{d} STALL Bot{d} pos=({d},{d}) eff=({d},{d}) inv={d} has_active={any} delivering={any} eff_slots={d} trip={any} stuck={any} order_rds={d}\n", .{
                state.round, bot.id, bot.pos.x, bot.pos.y, bpos.x, bpos.y,
                bot.inv_len, has_active, pb.delivering, effective_slots,
                pb.has_trip, order_stuck, pb.rounds_on_order,
            });
        }

        // ─── 4. Delivering → go to dropoff ──────────────────────────
        if (pb.delivering and has_active) {
            const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) {
                if (res.first_dir) |d| {
                    // Anti-oscillation: if near dropoff and BFS would move us further away, wait (max 3 rounds)
                    const cur_mdist = @abs(bpos.x - state.dropoff.x) + @abs(bpos.y - state.dropoff.y);
                    if (cur_mdist <= 2 and state.bot_count > 1 and pb.stall_count < 3) {
                        var next_pos = bpos;
                        switch (d) {
                            .up => next_pos.y -= 1,
                            .down => next_pos.y += 1,
                            .left => next_pos.x -= 1,
                            .right => next_pos.x += 1,
                        }
                        const next_mdist = @abs(next_pos.x - state.dropoff.x) + @abs(next_pos.y - state.dropoff.y);
                        if (next_mdist > cur_mdist) {
                            // Would move away from dropoff — wait instead of detouring (max 2 rounds)
                            try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                            pending_is_move[bi] = false;
                            pending_dirs[bi] = null;
                            continue;
                        }
                    }
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    pending_dirs[bi] = d;
                    pending_is_move[bi] = true;
                    continue;
                }
            }
        }

        // ─── 5. Follow existing trip plan ────────────────────────────
        if (pb.has_trip and pb.trip_pos < pb.trip_count) {
            const trip_id = pb.trip_ids[pb.trip_pos][0..pb.trip_id_lens[pb.trip_pos]];
            const idx = findItemById(state, trip_id);
            if (idx) |item_idx| {
                const item = &state.items[item_idx];
                const adj = pb.trip_adjs[pb.trip_pos];
                if (bpos.eql(adj)) {
                    if (bot.inv_len < INV_CAP) {
                        try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                        pb.last_tried_pickup = true;
                        pb.last_pickup_pos = bpos;
                        pb.last_pickup_item_pos = item.pos;
                        pb.last_inv_len = bot.inv_len;
                        pick_remaining.remove(item.item_type);
                        preview.remove(item.item_type);
                        claimed[item_idx] = @intCast(bi);
                        pb.trip_pos += 1;
                        if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
                        pending_is_move[bi] = false;
                        pending_dirs[bi] = null;
                        continue;
                    }
                } else {
                    const res = pathfinding.bfs(state, bpos, adj, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        claimed[item_idx] = @intCast(bi);
                        pending_dirs[bi] = d;
                        pending_is_move[bi] = true;
                        continue;
                    };
                }
            }
            pb.has_trip = false;
        }

        // ─── 6. Plan new trip ──────────────────────────────────────
        if (effective_slots > 0 and (!pb.delivering or !dropoff_priority[bi])) {
            const t = trip_mod.planBestTrip(state, dm_bot, &dm_drop, &pick_remaining, &bot_preview, &claimed, bi, @intCast(effective_slots), allow_preview_for_bot, rounds_left);
            if (t) |tp| {
                pb.has_trip = true;
                pb.trip_count = tp.item_count;
                pb.trip_pos = 0;
                pb.osc_count = 0;
                for (0..tp.item_count) |ti| {
                    const item = &state.items[tp.items[ti]];
                    const id = item.idStr();
                    @memcpy(pb.trip_ids[ti][0..id.len], id);
                    pb.trip_id_lens[ti] = @intCast(id.len);
                    pb.trip_adjs[ti] = tp.adjs[ti];
                    claimed[tp.items[ti]] = @intCast(bi);
                    preview.remove(item.item_type);
                }

                const first_adj = tp.adjs[0];
                const first_item = &state.items[tp.items[0]];
                if (bpos.eql(first_adj)) {
                    if (bot.inv_len < INV_CAP) {
                        try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, first_item.idStr() });
                        pb.last_tried_pickup = true;
                        pb.last_pickup_pos = bpos;
                        pb.last_pickup_item_pos = first_item.pos;
                        pb.last_inv_len = bot.inv_len;
                        pb.trip_pos += 1;
                        if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
                        pending_is_move[bi] = false;
                        pending_dirs[bi] = null;
                        continue;
                    }
                } else {
                    const res = pathfinding.bfs(state, bpos, first_adj, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        pending_dirs[bi] = d;
                        pending_is_move[bi] = true;
                        continue;
                    };
                }
            }
        }

        // ─── 7. Have active items → go deliver ─────────────────────
        if (bot.inv_len > 0 and has_active) {
            pb.delivering = true;
            const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                // Anti-oscillation near dropoff (max 3 rounds wait)
                const cur_md = @abs(bpos.x - state.dropoff.x) + @abs(bpos.y - state.dropoff.y);
                if (cur_md <= 2 and state.bot_count > 1 and pb.stall_count < 3) {
                    var np = bpos;
                    switch (d) { .up => np.y -= 1, .down => np.y += 1, .left => np.x -= 1, .right => np.x += 1 }
                    const next_md = @abs(np.x - state.dropoff.x) + @abs(np.y - state.dropoff.y);
                    if (next_md > cur_md) {
                        try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                        pending_is_move[bi] = false;
                        pending_dirs[bi] = null;
                        continue;
                    }
                }
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                pending_dirs[bi] = d;
                pending_is_move[bi] = true;
                continue;
            };
        }

        // ─── 8. Has active items but couldn't deliver yet → go to dropoff ──
        if (bot.inv_len > 0 and has_active and !bpos.eql(state.dropoff)) {
            const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                pending_dirs[bi] = d;
                pending_is_move[bi] = true;
                continue;
            };
        }

        // ─── 8b. Not delivering → pre-position ─────
        if (!has_active and bot.inv_len == 0) {
            // Pre-position near items that are still needed (active first, then preview).
            // This puts idle bots in useful positions for quick pickups.
            var prev_targets: [16]struct { adj: Pos, dist: u16 } = undefined;
            var prev_target_count: u8 = 0;
            // First: try active items still needed on map
            for (0..state.item_count) |ii| {
                if (prev_target_count >= 16) break;
                if (!pick_remaining.contains(state.items[ii].item_type)) continue;
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                if (d >= UNREACHABLE) continue;
                prev_targets[prev_target_count] = .{ .adj = adj, .dist = d };
                prev_target_count += 1;
            }
            // Then: preview items
            for (0..state.item_count) |ii| {
                if (prev_target_count >= 16) break;
                if (!preview_orig.contains(state.items[ii].item_type)) continue;
                const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                if (d >= UNREACHABLE) continue;
                prev_targets[prev_target_count] = .{ .adj = adj, .dist = d };
                prev_target_count += 1;
            }
            if (prev_target_count > 0) {
                // Sort by distance
                for (0..prev_target_count) |i| {
                    var min_j = i;
                    for (i + 1..prev_target_count) |j| {
                        if (prev_targets[j].dist < prev_targets[min_j].dist) min_j = j;
                    }
                    if (min_j != i) {
                        const tmp = prev_targets[i];
                        prev_targets[i] = prev_targets[min_j];
                        prev_targets[min_j] = tmp;
                    }
                }
                // Compute rank among idle bots (not raw bi) to spread targets properly
                var idle_rank: u8 = 0;
                for (0..bi) |prev_bi| {
                    if (!pbots[prev_bi].delivering and state.bots[prev_bi].inv_len == 0) {
                        idle_rank += 1;
                    }
                }
                const pick_idx = idle_rank % prev_target_count;
                const target = prev_targets[pick_idx].adj;
                if (!bpos.eql(target)) {
                    const res = pathfinding.bfs(state, bpos, target, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        pending_dirs[bi] = d;
                        pending_is_move[bi] = true;
                        continue;
                    };
                }
            }
            // Fallback: if no preview targets, stay put (don't waste rounds going to a corner)
        }
        // Dead inventory bots (inv>0, !has_active):
        // If inventory matches preview, position near dropoff (within 2) for auto-delivery.
        // If ON dropoff, flee (handled by 1b above). Otherwise stay nearby.
        if (!has_active and bot.inv_len > 0) {
            var has_preview_match = false;
            for (0..bot.inv_len) |ii| {
                if (preview_orig.contains(bot.inv[ii])) { has_preview_match = true; break; }
            }
            const dd_dist = pathfinding.distFromMap(&dm_drop, bpos);
            if (has_preview_match) {
                // Items match preview → position near dropoff (within 3, not ON it to avoid blocking)
                if (dd_dist > 3) {
                    const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        pending_dirs[bi] = d;
                        pending_is_move[bi] = true;
                        continue;
                    };
                }
            } else if (dd_dist <= 2) {
                // Truly dead items near dropoff → flee to avoid blocking
                const flee_dir = fleeDropoff(state, bpos, @intCast(bi), &bot_positions);
                if (flee_dir) |d| {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    pending_dirs[bi] = d;
                    pending_is_move[bi] = true;
                    continue;
                }
            }
            // Otherwise: stay put, don't waste energy
        }

        // ─── 9. Wait ────────────────────────────────────────────────
        try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
        pending_is_move[bi] = false;
        pending_dirs[bi] = null;
    }

    try writer.writeAll("]}");

    // Record expected positions for shift detection
    for (0..state.bot_count) |bi| {
        if (offset_detected) {
            // In offset mode, the server applies LAST round's action to produce next state.
            // eff_pos[bi] already = current_pos + last_round's_pending_action = next state's position.
            expected_next_pos[bi] = eff_pos[bi];
        } else {
            expected_next_pos[bi] = bot_positions[bi];
        }
    }
    expected_count = @intCast(state.bot_count);

    return stream.getWritten();
}

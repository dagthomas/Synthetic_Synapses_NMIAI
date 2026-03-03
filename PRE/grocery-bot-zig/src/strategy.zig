const std = @import("std");
const config = @import("config");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");
const trip_mod = @import("trip.zig");
const spacetime = @import("spacetime.zig");

// Compile-time difficulty: .auto means runtime detection, otherwise locked to specific difficulty
const Difficulty = config.Difficulty;
const DIFFICULTY = config.difficulty;

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
const MAX_W = types.MAX_W;
const INV_CAP = types.INV_CAP;
const UNREACHABLE = types.UNREACHABLE;

// ── MAPF Configuration ───────────────────────────────────────────────
// Enable Space-Time A* with reservation table for Hard/Expert only.
// Easy/Medium keep current behavior (compile-time eliminated).
const USE_MAPF = switch (DIFFICULTY) {
    .easy, .medium, .auto => false,
    .hard, .expert => true,
};

/// Navigate from start to target using MAPF (ST-A*) when available, else BFS.
fn navigateTo(state: *const GameState, start: Pos, target: Pos, bot_id: u8, bot_positions: *const [MAX_BOTS]Pos) pathfinding.BfsResult {
    if (USE_MAPF and state.bot_count >= 5) {
        if (spacetime.planAndReserve(state, start, target, bot_id)) |result| {
            // ST-A* found a path (first_dir may be null = wait)
            return .{ .dist = if (result.path_len > 0) result.path_len else 1, .first_dir = result.first_dir };
        }
        // ST-A* failed — fall back to regular BFS
    }
    return pathfinding.bfs(state, start, target, bot_id, bot_positions);
}

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
    pickup_fail_count: u8,
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
        pbots[i].pickup_fail_count = 0;
    }
    blacklist_count = 0;
    last_score = 0;
    rounds_since_score_change = 0;
    pathfinding.resetPrecompute();
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
    // Kept disabled: failed pickups are timing-related, not permanently bad positions
    // Enabling this caused regression (44 vs typical 50-75)
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
        // Prefer moving toward nearest horizontal corridor (where bots can pass)
        // Corridors: y=1 (top), y=mid_y (middle), y=h-2 (bottom)
        const mid_y = @as(i16, @intCast(state.height / 2));
        const corr_top: i16 = 1;
        const corr_bot: i16 = @as(i16, @intCast(state.height)) - 2;
        const cur_corr = @min(@min(@abs(pos.y - corr_top), @abs(pos.y - mid_y)), @abs(pos.y - corr_bot));
        const new_corr = @min(@min(@abs(ny - corr_top), @abs(ny - mid_y)), @abs(ny - corr_bot));
        if (new_corr < cur_corr) score += 5; // Strong preference for corridor
        if (new_corr > cur_corr) score -= 2; // Avoid going deeper into aisle
        // Avoid crowded areas: prefer directions with fewer nearby bots
        if (state.bot_count >= 5) {
            var nearby: i16 = 0;
            for (0..state.bot_count) |bk| {
                if (bk == bot_id) continue;
                if (@abs(bot_positions[bk].x - nx) + @abs(bot_positions[bk].y - ny) <= 3) nearby += 1;
            }
            score -= nearby * 2;
        }
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

    // Pre-compute all BFS distance maps once at game start (walls never change)
    if (state.round == 0) {
        pathfinding.precomputeAllDistances(state);
        if (USE_MAPF) spacetime.init(state);
    }

    // Clear reservations at start of each round (MAPF replans every round)
    if (USE_MAPF and state.bot_count >= 5) {
        spacetime.clearReservations();
    }

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

    // Detect auto-delivery completion: active order exists but ALL items already delivered
    // This happens when auto-delivery at dropoff fills the entire new order but no bot
    // issues drop_off to trigger the +5 completion bonus
    const order_auto_complete = blk: {
        var has_active_order = false;
        for (0..state.order_count) |oi| {
            if (state.orders[oi].is_active and !state.orders[oi].complete) {
                has_active_order = true;
                break;
            }
        }
        break :blk has_active_order and active_orig.count == 0;
    };

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
    // Max bots allowed to carry preview items: limit to reduce dead inventory risk
    // CRITICAL: increasing this causes catastrophic dead-inv in Expert (10 bots)
    const max_preview_carriers: u8 = if (state.bot_count <= 2) state.bot_count else if (state.bot_count >= 8) 4 else if (state.bot_count >= 5) 2 else 1;

    // ── Fix 6: Track score changes for diagnostic logging ──
    if (state.score != last_score) {
        rounds_since_score_change = 0;
        last_score = state.score;
    } else {
        rounds_since_score_change += 1;
    }

    // Periodically clear blacklist so stale entries don't permanently block valid positions
    if (state.round > 0 and state.round % 50 == 0 and blacklist_count > 0) {
        std.debug.print("R{d} Clearing {d} blacklist entries\n", .{ state.round, blacklist_count });
        blacklist_count = 0;
    }

    // ── Detect 1-round action offset ──────────────────────────────────
    // Offset appears mid-game unpredictably. False positives are catastrophic.
    // Skip first 15 rounds (bots crowded initially).
    // Require 3+ moving bots with 67%+ mismatch ratio for 2 consecutive rounds.
    if (expected_count > 0 and state.round >= 15 and !offset_detected) {
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
        // Require >=67% mismatch ratio AND at least 3 mismatches
        if (moving_count >= 3 and moving_mismatches >= 3 and moving_mismatches * 3 >= moving_count * 2) {
            offset_check_mismatches += 1;
        } else {
            if (offset_check_mismatches > 0) offset_check_mismatches -|= 1;
        }
        offset_check_rounds += 1;
        if (offset_check_mismatches >= 2) {
            offset_detected = true;
            blacklist_count = 0;
            std.debug.print("R{d} OFFSET MODE ENABLED ({d}/{d} moving mismatches), blacklist cleared\n", .{ state.round, moving_mismatches, moving_count });
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

    // ── Distance from dropoff (pre-computed at round 0, walls never change) ──
    var dm_drop: DistMap = pathfinding.getPrecomputedDm(state, state.dropoff).*;

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
    var orch_preview_assigned: [MAX_BOTS]u8 = undefined;
    @memset(orch_preview_assigned[0..state.bot_count], 0);

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

        // Fast head-on deadlock detection (2 rounds instead of 6)
        // When two bots are adjacent and both stalled, they're in a corridor deadlock.
        // Higher-ID bot yields (escapes to nearest corridor) — lower-ID bot has movement priority.
        if (pb.stall_count >= 2 and pb.stall_count < 6 and state.bot_count > 1 and pb.escape_rounds == 0) {
            for (0..state.bot_count) |bk| {
                if (bk == bi) continue;
                if (pbots[bk].stall_count < 2) continue;
                const mdist = @abs(bot.pos.x - state.bots[bk].pos.x) + @abs(bot.pos.y - state.bots[bk].pos.y);
                if (mdist == 1) {
                    // Adjacent stalled bots → head-on deadlock
                    if (bi > bk) {
                        // Higher ID yields: escape to nearest corridor
                        pb.escape_rounds = 3;
                        pb.has_trip = false;
                        pb.stall_count = 0;
                        std.debug.print("R{d} Bot{d} FAST-YIELD to Bot{d} at ({d},{d})\n", .{ state.round, bot.id, state.bots[bk].id, bot.pos.x, bot.pos.y });
                    } else {
                        // Lower ID keeps priority — just reset stall so it retries
                        pb.stall_count = 0;
                    }
                    break;
                }
            }
        }

        // Stall reset (same position for 6+ rounds — faster recovery than 8)
        if (pb.stall_count > 6) {
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

        // Detect failed pickups — after 2 consecutive fails, invalidate trip to try different path
        if (pb.last_tried_pickup) {
            if (bot.inv_len == pb.last_inv_len) {
                pb.pickup_fail_count += 1;
                std.debug.print("R{d} Bot{d} pickup FAILED at ({d},{d}) -> item ({d},{d}) inv={d} fails={d}\n", .{
                    state.round, bot.id, pb.last_pickup_pos.x, pb.last_pickup_pos.y,
                    pb.last_pickup_item_pos.x, pb.last_pickup_item_pos.y, bot.inv_len, pb.pickup_fail_count,
                });
                addBlacklist(pb.last_pickup_pos, pb.last_pickup_item_pos);
                // After 2 consecutive fails, reset trip so bot picks a different path/adj cell
                if (pb.pickup_fail_count >= 2) {
                    pb.has_trip = false;
                    pb.pickup_fail_count = 0;
                }
            } else {
                pb.pickup_fail_count = 0; // Successful pickup resets counter
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
        // Trigger escape to physically break deadlocks — just resetting trip plans
        // causes bot to immediately re-plan the same failed route
        // STAGGER: Only 1-2 bots escape per cycle to maintain coordination.
        // Each bot's escape window is offset by bot ID to prevent mass escape.
        const escape_threshold: u16 = 25 + @as(u16, @intCast(bi)) * 4;
        if (pb.rounds_on_order > escape_threshold and (pb.rounds_on_order -| escape_threshold) % 12 == 0) {
            pb.has_trip = false;
            pb.delivering = false;
            pb.osc_count = 0;
            pb.escape_rounds = 3;
            std.debug.print("R{d} Bot{d} stuck-order ESCAPE (rounds_on_order={d})\n", .{ state.round, bot.id, pb.rounds_on_order });
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

    // ── Distance maps for all bots (pre-computed lookups, O(1) per bot) ──
    var all_dm_bot: [MAX_BOTS]DistMap = undefined;
    for (0..state.bot_count) |bi| {
        all_dm_bot[bi] = pathfinding.getPrecomputedDm(state, eff_pos[bi]).*;
    }

    // ── Dropoff priority: rank delivering bots by distance ──────────────
    // Computed BEFORE orchestrator so it can exclude priority-delivering bots from assignment.
    const MAX_DROPOFF_ACTIVE: u8 = if (state.bot_count > 1) @as(u8, @intCast(state.bot_count)) / 2 else 1;
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
            // Only auto-grant priority to bots that are actually delivering
            // Non-delivering bots near dropoff should still get orchestrator assignments
            if (!pbots[bi].delivering) continue;
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

        // Max active pickers: limit how many bots get active item assignments
        // Concentrates items on fewer bots → fewer delivery trips → less dropoff congestion
        const max_pickers: u8 = switch (DIFFICULTY) {
            .easy => state.bot_count,
            .medium => state.bot_count,
            .hard => 4,
            .expert => 4,
            .auto => if (state.bot_count >= 8) 8 else if (state.bot_count >= 5) 3 else state.bot_count,
        };

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
                        if (pbots[bk].delivering and dropoff_priority[bk]) continue;
                        const free_slots = if (INV_CAP > state.bots[bk].inv_len) INV_CAP - state.bots[bk].inv_len else 0;
                        if (bot_assigned[bk] >= free_slots) continue;
                        // Don't assign to a new bot if we've hit the picker limit
                        if (bot_assigned[bk] == 0) {
                            var unique_pickers: u8 = 0;
                            for (0..state.bot_count) |bp| {
                                if (bot_assigned[bp] > 0) unique_pickers += 1;
                            }
                            if (unique_pickers >= max_pickers) continue;
                        }

                        const adj = pathfinding.findBestAdj(state, state.items[ii].pos, &all_dm_bot[bk]) orelse continue;
                        const raw_d = all_dm_bot[bk][@intCast(adj.y)][@intCast(adj.x)];
                        const d_back = dm_drop[@intCast(adj.y)][@intCast(adj.x)];
                        // Orchestrator cost: per-difficulty strategy
                        const use_roundtrip = switch (DIFFICULTY) {
                            .easy => false, // Single bot: pure pickup distance
                            .medium => true, // Round-trip cost reduces total cycle time
                            .hard => true,
                            .expert => false,
                            .auto => state.bot_count > 1 and state.bot_count < 8,
                        };
                        const trip_cost: u16 = if (use_roundtrip and d_back < UNREACHABLE) raw_d + d_back else raw_d;
                        // Concentration bonus: prefer bots that already have assignments
                        const conc_bonus: u16 = switch (DIFFICULTY) {
                            .medium => if (bot_assigned[bk] > 0) 8 else 0,
                            .expert => if (bot_assigned[bk] > 0) 3 else 0,
                            .auto => if (bot_assigned[bk] > 0 and state.bot_count == 3) 8 else 0,
                            else => 0,
                        };
                        const d = if (trip_cost > conc_bonus) trip_cost - conc_bonus else 0;
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
            // Prevents duplicate type picking which creates dead inventory when orders change
            // Only for <=3 bots — larger bot counts benefit from backup pickers
            if (type_track[ti].assigned >= type_track[ti].needed and state.bot_count <= 3) {
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

        // ── Phase 2: Assign preview items to idle bots ────────────────
        // When all active items are picked, aggressively assign preview — these become
        // the next order's active items. No risk of permanent dead inventory.
        const all_active_covered_orch = pick_remaining.count == 0;
        var total_preview_assigned: u8 = 0;
        const max_orch_preview: u8 = if (all_active_covered_orch) preview.count else if (state.bot_count <= 2) 4 else if (state.bot_count <= 4) 2 else if (state.bot_count >= 8) 4 else 2;
        if (preview.count > 0 and (all_active_covered_orch or bots_with_preview_only < max_preview_carriers)) {
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
                            // Skip bots carrying active items — they need to deliver, not pick preview
                            var bk_has_active = false;
                            {
                                var check_a = active_orig;
                                for (0..state.bots[bk].inv_len) |inv_i| {
                                    if (check_a.contains(state.bots[bk].inv[inv_i])) {
                                        bk_has_active = true;
                                        check_a.remove(state.bots[bk].inv[inv_i]);
                                    }
                                }
                            }
                            if (bk_has_active) continue;
                            const prev_free = if (INV_CAP > state.bots[bk].inv_len) INV_CAP - state.bots[bk].inv_len else 0;
                            if (prev_free == 0) continue;
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
                    orch_preview_assigned[best_bot] += 1;
                }
            }

            // Block excess preview instances to prevent duplicate type picking
            // Prevents multiple bots from picking same preview type → dead inventory
            // Only for <=3 bots — larger counts benefit from backup preview pickers
            if (state.bot_count <= 3) {
                for (0..prev_track_len) |ti| {
                    if (prev_track[ti].assigned >= prev_track[ti].needed) {
                        for (0..state.item_count) |ii| {
                            if (claimed[ii] >= 0) continue;
                            if (state.items[ii].item_type.eql(prev_track[ti].t)) {
                                claimed[ii] = @intCast(MAX_BOTS);
                            }
                        }
                    }
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

    // ── MAPF Priority Pre-pass: reserve paths for delivering bots first ──
    // Delivering bots get priority in the reservation table (closest to dropoff first).
    // This ensures picking bots route around them, not the other way around.
    if (USE_MAPF and state.bot_count >= 5) {
        const DelBot = struct { bi: u8, dist: u16 };
        var del_bots: [MAX_BOTS]DelBot = undefined;
        var del_count: u8 = 0;
        for (0..state.bot_count) |bi| {
            if (!pbots[bi].delivering) continue;
            // Check if bot actually has active items
            var ba = active_orig;
            var ha = false;
            for (0..state.bots[bi].inv_len) |ii| {
                if (ba.contains(state.bots[bi].inv[ii])) {
                    ha = true;
                    ba.remove(state.bots[bi].inv[ii]);
                }
            }
            if (!ha) continue;
            const d = pathfinding.distFromMap(&dm_drop, eff_pos[bi]);
            del_bots[del_count] = .{ .bi = @intCast(bi), .dist = d };
            del_count += 1;
        }
        // Sort by distance (closest first = highest priority)
        for (0..del_count) |i| {
            var min_j = i;
            for (i + 1..del_count) |j| {
                if (del_bots[j].dist < del_bots[min_j].dist) min_j = j;
            }
            if (min_j != i) {
                const tmp = del_bots[i];
                del_bots[i] = del_bots[min_j];
                del_bots[min_j] = tmp;
            }
        }
        // Reserve paths for delivering bots → picking bots will route around them
        for (del_bots[0..del_count]) |db| {
            _ = spacetime.planAndReserve(state, eff_pos[db.bi], state.dropoff, db.bi);
        }
        // Reserve stalled bots as stationary obstacles (they won't move soon)
        for (0..state.bot_count) |bi| {
            if (pbots[bi].stall_count >= 5) {
                if (eff_pos[bi].x >= 0 and eff_pos[bi].y >= 0) {
                    spacetime.reserveStationary(
                        @intCast(eff_pos[bi].x),
                        @intCast(eff_pos[bi].y),
                        0,
                        types.MAX_TIME_HORIZON,
                        @intCast(bi),
                    );
                }
            }
        }
    }

    for (0..state.bot_count) |bi| {
        if (bi > 0) try writer.writeAll(",");
        const bot = &state.bots[bi];
        const pb = &pbots[bi];
        const bpos = eff_pos[bi]; // Effective position (accounts for offset)

        pb.last_tried_pickup = false;

        // MAPF: reserve this bot's current position at t=0 so subsequent bots avoid colliding here
        if (USE_MAPF and state.bot_count >= 5) {
            if (bpos.x >= 0 and bpos.y >= 0) {
                spacetime.reserve(@intCast(bpos.x), @intCast(bpos.y), 0, @intCast(bi));
            }
        }

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

        // Preview strategy: allow preview picking only for bots with NO active items to deliver.
        // Preview items become the next order — safe to pick. But picking preview MUST NOT
        // delay active order delivery. Only truly idle bots (nothing active) should pick preview.
        const all_active_covered = pick_remaining.count == 0;
        // Single-bot: only preview when ALL active items are already picked (prevents dead-inv lockout)
        // Small multi-bot (2-4): preview freely when active covered (no congestion risk)
        // Large multi-bot (5+): cap preview carriers even when covered to avoid mass dead-inventory
        const max_preview_covered: u8 = if (state.bot_count <= 4) state.bot_count else @as(u8, @intCast(state.bot_count)) / 2;
        // Safe preview: allow beyond carrier limit when bot is near dropoff AND order is far from completion.
        // Near dropoff → fast delivery. Many items remaining → order won't cycle soon. Low dead-inv risk.
        const dist_to_drop_bot = pathfinding.distFromMap(&dm_drop, bpos);
        const safe_preview = state.bot_count >= 5 and pick_remaining.count >= 2 and dist_to_drop_bot <= 15 and orch_active_assigned[bi] == 0;
        const allow_preview_for_bot = !has_active and
            (bot.inv_len < INV_CAP) and
            (if (state.bot_count <= 1) all_active_covered
            else if (state.bot_count <= 4) (all_active_covered or (orch_active_assigned[bi] == 0 and bots_with_preview_only < max_preview_carriers))
            else (orch_active_assigned[bi] == 0 and (bots_with_preview_only < (if (all_active_covered) max_preview_covered else max_preview_carriers) or safe_preview)));

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

        // ─── 0. Auto-delivery completion: order fully satisfied but not completed ──
        // When auto-delivery fills ALL items of the new active order, no bot has has_active=true
        // because active needs are empty. But drop_off still triggers the completion check (+5).
        if (order_auto_complete and bpos.eql(state.dropoff) and bot.inv_len > 0) {
            try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
            pb.has_trip = false;
            pb.delivering = false;
            pending_is_move[bi] = false;
            pending_dirs[bi] = null;
            continue;
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

        // ─── 1b. At dropoff but shouldn't be → evacuate (multi-bot) or fall through (single) ──
        if (bpos.eql(state.dropoff) and !has_active) {
            if (state.bot_count > 1) {
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
            // Single-bot: fall through to trip planning (saves 1 round per delivery)
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
                // escapeDir returned null — emit wait instead of falling through
                try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                pending_is_move[bi] = false;
                pending_dirs[bi] = null;
                continue;
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
                    if (!bot_preview.contains(item.item_type)) continue;
                    // Allow preview pickup for:
                    // a) Idle bots explicitly allowed (no active items), OR
                    // b) Delivering bots (5+ bots) when active items nearly covered (preview auto-delivers at dropoff)
                    // Allow preview pickup during delivery when:
                    // a) 5+ bots and order nearly done (pick_remaining <= 2) [original], OR
                    // b) <5 bots when order guaranteed to complete (pick_remaining == 0)
                    const delivering_preview_ok = pb.delivering and bot.inv_len < INV_CAP and rounds_left > 30 and
                        ((state.bot_count >= 5 and pick_remaining.count <= 2) or
                         (state.bot_count < 5 and pick_remaining.count == 0));
                    if (!allow_preview_for_bot and !delivering_preview_ok) continue;
                }

                try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                pb.last_tried_pickup = true;
                pb.last_pickup_pos = bpos;
                pb.last_pickup_item_pos = item.pos;
                pb.last_inv_len = bot.inv_len;
                if (pass == 0) {
                    pick_remaining.remove(item.item_type);
                } else {
                    preview.remove(item.item_type);
                }
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
        const endgame = rounds_left <= dist_to_drop + @as(u16, bot.inv_len) + 1;

        // With many bots, don't rush to deliver with 1 item if far from dropoff
        // Let closer bots handle delivery while far bots pick preview items
        const far_with_few = state.bot_count >= 5 and !inv_full and dist_to_drop > 8 and bot.inv_len < 2 and all_active_got and !endgame;

        const should_deliver = has_active and !far_with_few and (inv_full or all_active_got or no_active_avail or order_stuck or endgame or trip_done);

        if (should_deliver) {
            pb.delivering = true;
            pb.has_trip = false;
        }
        // (Delay delivery for preview fill-up removed — caused dead inventory cascade)
        // Far bots with few items: stop delivering, go pick preview instead
        if (far_with_few and pb.delivering) {
            pb.delivering = false;
        }

        if (!has_active) {
            pb.delivering = false;
        }

        // Endgame yielding: bots that can't reach dropoff in time should get out of the way
        // so bots that CAN deliver aren't blocked in corridors
        if (state.bot_count > 1 and rounds_left <= 15 and dist_to_drop > rounds_left and !bpos.eql(state.dropoff)) {
            // This bot can't deliver in time — flee from dropoff to clear paths
            const flee_dir = fleeDropoff(state, bpos, @intCast(bi), &bot_positions);
            if (flee_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                pending_dirs[bi] = d;
                pending_is_move[bi] = true;
                continue;
            }
        }

        // Diagnostic logging when score stalls
        if (rounds_since_score_change >= 10 and state.round % 5 == 0) {
            std.debug.print("R{d} STALL Bot{d} pos=({d},{d}) eff=({d},{d}) inv={d} has_active={any} delivering={any} eff_slots={d} trip={any} stuck={any} order_rds={d}\n", .{
                state.round, bot.id, bot.pos.x, bot.pos.y, bpos.x, bpos.y,
                bot.inv_len, has_active, pb.delivering, effective_slots,
                pb.has_trip, order_stuck, pb.rounds_on_order,
            });
        }

        // ─── 4. Delivering → go to dropoff (with opportunistic detour) ──
        if (pb.delivering and has_active) {
            // Graduated delivery: when many closer bots are already delivering,
            // be more willing to detour and pick items on the way to dropoff.
            // This staggers arrivals and reduces dropoff congestion.
            const delivery_rank: u8 = blk: {
                if (state.bot_count < 5) break :blk 0;
                var closer: u8 = 0;
                for (0..state.bot_count) |bk| {
                    if (bk == bi) continue;
                    if (!pbots[bk].delivering) continue;
                    const bk_dist = pathfinding.distFromMap(&dm_drop, bot_positions[bk]);
                    if (bk_dist < dist_to_drop) closer += 1;
                }
                break :blk closer;
            };
            // Opportunistic detour: pick up active items on the way to dropoff
            // Single-bot: generous detour (max 5 extra). Multi-bot: 1 normally, 3 when queued.
            const base_detour: u32 = if (state.bot_count <= 1) 5
                else if (delivery_rank >= 2) 3
                else 1;
            if (effective_slots > 0 and !endgame and dist_to_drop > 2) {
                // Would picking the last remaining item complete the order?
                const completing_detour = pick_remaining.count == 1;
                const max_extra: u32 = if (completing_detour) @as(u32, if (state.bot_count <= 1) 8 else 4) else base_detour;
                var best_det_ii: ?u16 = null;
                var best_det_adj: Pos = undefined;
                var best_det_extra: u32 = max_extra;
                for (0..state.item_count) |ii| {
                    if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                    if (!pick_remaining.contains(state.items[ii].item_type)) continue;
                    const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                    const ux: u16 = @intCast(adj.x);
                    const uy: u16 = @intCast(adj.y);
                    const d_to = dm_bot[uy][ux];
                    if (d_to >= UNREACHABLE) continue;
                    const d_back = dm_drop[uy][ux];
                    if (d_back >= UNREACHABLE) continue;
                    const via = @as(u32, d_to) + @as(u32, d_back);
                    const extra = if (via > dist_to_drop) via - dist_to_drop else 0;
                    if (extra < best_det_extra) {
                        best_det_extra = extra;
                        best_det_ii = @intCast(ii);
                        best_det_adj = adj;
                    }
                }
                // Preview detour: when all active items are picked (order completes on delivery),
                // grab a preview item on the way to dropoff for free auto-delivery
                // Helps single-bot and small multi-bot (< 5); hurts 5+ bots (delivery congestion)
                if (best_det_ii == null and pick_remaining.count == 0 and state.bot_count <= 5) {
                    // Zero-cost detours for 5 bots (extra=0 only); 1-step for 2-4 bots; generous for single
                    const prev_max_extra: u32 = if (state.bot_count <= 1) 4 else if (state.bot_count >= 5) 1 else 2;
                    for (0..state.item_count) |ii| {
                        if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                        if (!bot_preview.contains(state.items[ii].item_type)) continue;
                        const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                        const ux: u16 = @intCast(adj.x);
                        const uy: u16 = @intCast(adj.y);
                        const d_to = dm_bot[uy][ux];
                        if (d_to >= UNREACHABLE) continue;
                        const d_bk = dm_drop[uy][ux];
                        if (d_bk >= UNREACHABLE) continue;
                        const via = @as(u32, d_to) + @as(u32, d_bk);
                        const extra = if (via > dist_to_drop) via - dist_to_drop else 0;
                        if (extra < prev_max_extra and (best_det_ii == null or extra < best_det_extra)) {
                            best_det_extra = extra;
                            best_det_ii = @intCast(ii);
                            best_det_adj = adj;
                        }
                    }
                }
                if (best_det_ii) |det_ii| {
                    if (!bpos.eql(best_det_adj)) {
                        const det_res = navigateTo(state, bpos, best_det_adj, @intCast(bi), &bot_positions);
                        if (det_res.dist < UNREACHABLE) if (det_res.first_dir) |d| {
                            try writeMove(writer, bot.id, d);
                            updateBotPos(&bot_positions[bi], d);
                            claimed[det_ii] = @intCast(bi);
                            pending_dirs[bi] = d;
                            pending_is_move[bi] = true;
                            continue;
                        };
                    }
                }
            }
            const res = navigateTo(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
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
                    const res = navigateTo(state, bpos, adj, @intCast(bi), &bot_positions);
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
        if (effective_slots > 0 and (!pb.delivering or !dropoff_priority[bi] or (all_active_covered and !has_active))) {
            // Endgame trip size reduction: smaller trips complete faster in late game
            const trip_slots: u8 = switch (DIFFICULTY) {
                .expert => if (rounds_left < 20) @min(effective_slots, 1)
                    else if (rounds_left < 40) @min(effective_slots, 2)
                    else effective_slots,
                .hard => if (rounds_left < 20) @min(effective_slots, 1)
                    else if (rounds_left < 40) @min(effective_slots, 2)
                    else effective_slots,
                .auto => if (state.bot_count >= 5 and rounds_left < 20) @min(effective_slots, 1)
                    else if (state.bot_count >= 5 and rounds_left < 40) @min(effective_slots, 2)
                    else effective_slots,
                else => effective_slots,
            };
            // Allow preview in completing trips (single-bot only): if remaining active items fit in our slots,
            // adding preview items lets them auto-deliver when order completes at dropoff
            const completing_possible = state.bot_count <= 1 and pick_remaining.count > 0 and pick_remaining.count <= trip_slots;
            const allow_preview_in_trip = allow_preview_for_bot or completing_possible;
            const t = trip_mod.planBestTrip(state, dm_bot, &dm_drop, &pick_remaining, &bot_preview, &claimed, bi, @intCast(trip_slots), allow_preview_in_trip, rounds_left, @intCast(state.bot_count));
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
                    const res = navigateTo(state, bpos, first_adj, @intCast(bi), &bot_positions);
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
            const res = navigateTo(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
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

        // ─── 8. Not delivering → pre-position ─────
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
            // For 8+ bots: prioritize orchestrator-assigned items (bots pre-position for next order)
            if (state.bot_count >= 8) {
                for (0..state.item_count) |ii| {
                    if (prev_target_count >= 16) break;
                    if (!preview_orig.contains(state.items[ii].item_type)) continue;
                    if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                    const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                    const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                    if (d >= UNREACHABLE) continue;
                    prev_targets[prev_target_count] = .{ .adj = adj, .dist = d };
                    prev_target_count += 1;
                }
            }
            // Fallback or default: any preview items
            if (prev_target_count == 0) {
                for (0..state.item_count) |ii| {
                    if (prev_target_count >= 16) break;
                    if (!preview_orig.contains(state.items[ii].item_type)) continue;
                    const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                    const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                    if (d >= UNREACHABLE) continue;
                    prev_targets[prev_target_count] = .{ .adj = adj, .dist = d };
                    prev_target_count += 1;
                }
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
                    const res = navigateTo(state, bpos, target, @intCast(bi), &bot_positions);
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
        // Priority 1: if free slots exist and active items need picking, go pick them!
        // Priority 2: if preview match, camp near dropoff for auto-delivery
        // Priority 3: CLEAR CORRIDORS — move to map edges so delivering bots can pass
        if (!has_active and bot.inv_len > 0) {
            // Dead-inv bots with free slots should actively seek active items
            if (effective_slots > 0 and pick_remaining.count > 0) {
                var best_active_adj: ?Pos = null;
                var best_active_dist: u16 = UNREACHABLE;
                for (0..state.item_count) |ii| {
                    if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                    if (!pick_remaining.contains(state.items[ii].item_type)) continue;
                    const adj = pathfinding.findBestAdj(state, state.items[ii].pos, dm_bot) orelse continue;
                    const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                    if (d < best_active_dist) {
                        best_active_dist = d;
                        best_active_adj = adj;
                    }
                }
                if (best_active_adj) |target| {
                    if (!bpos.eql(target)) {
                        const res = navigateTo(state, bpos, target, @intCast(bi), &bot_positions);
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

            var has_preview_match = false;
            for (0..bot.inv_len) |ii| {
                if (preview_orig.contains(bot.inv[ii])) { has_preview_match = true; break; }
            }
            const dd_dist = pathfinding.distFromMap(&dm_drop, bpos);
            // Count dead-inv bots already near dropoff (dist <= 3)
            var campers_near_drop: u8 = 0;
            for (0..bi) |prev_bi| {
                if (pbots[prev_bi].delivering) continue;
                if (state.bots[prev_bi].inv_len == 0) continue;
                const pd = pathfinding.distFromMap(&dm_drop, bot_positions[prev_bi]);
                if (pd <= 3) campers_near_drop += 1;
            }
            const max_campers: u8 = if (state.bot_count <= 3) state.bot_count else 2;
            const can_camp = campers_near_drop < max_campers;

            if (has_preview_match and can_camp) {
                // Rush to dropoff when order nearly complete (auto-delivery exploit)
                const rush_to_dropoff = pick_remaining.count <= 2;
                const target_dist: u16 = if (rush_to_dropoff) 0 else 3;
                if (dd_dist > target_dist) {
                    const res = navigateTo(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        if (!rush_to_dropoff or !bpos.eql(state.dropoff)) {
                            try writeMove(writer, bot.id, d);
                            updateBotPos(&bot_positions[bi], d);
                            pending_dirs[bi] = d;
                            pending_is_move[bi] = true;
                            continue;
                        }
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
            } else if (effective_slots == 0 and state.bot_count >= 5) {
                // Full dead-inv bots: CLEAR CORRIDORS by moving to spawn corner (w-2, h-2)
                // These bots are pure blockers — get them out of the way so delivering bots can pass
                const spawn = Pos{ .x = @as(i16, @intCast(state.width)) - 2, .y = @as(i16, @intCast(state.height)) - 2 };
                if (!bpos.eql(spawn)) {
                    const res = navigateTo(state, bpos, spawn, @intCast(bi), &bot_positions);
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

        // ─── 9. Wait ────────────────────────────────────────────────
        try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
        pending_is_move[bi] = false;
        pending_dirs[bi] = null;

        // MAPF: reserve stationary position for non-moving bots so subsequent bots route around
        if (USE_MAPF and state.bot_count >= 5) {
            if (bpos.x >= 0 and bpos.y >= 0) {
                spacetime.reserveStationary(@intCast(bpos.x), @intCast(bpos.y), 0, types.MAX_TIME_HORIZON, @intCast(bi));
            }
        }
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

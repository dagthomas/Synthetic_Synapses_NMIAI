const std = @import("std");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");

const Pos = types.Pos;
const Dir = types.Dir;
const Cell = types.Cell;
const Bot = types.Bot;
const ItemType = types.ItemType;
const GameState = types.GameState;
const NeedList = types.NeedList;
const DistMap = types.DistMap;
const MAX_BOTS = types.MAX_BOTS;
const MAX_ITEMS = types.MAX_ITEMS;
const MAX_H = types.MAX_H;
const MAX_W = types.MAX_W;
const INV_CAP = types.INV_CAP;
const UNREACHABLE = types.UNREACHABLE;

// ── Per-Bot State ──────────────────────────────────────────────────────
const Phase = enum { restock, ready };

const QBot = struct {
    phase: Phase,
    types: [3]ItemType, // assigned item types (1-3 via round-robin)
    type_count: u8,
    last_pos: Pos,
    stall_count: u16,
    last_dir: ?Dir,
    escape_rounds: u8,
};

var qbots: [MAX_BOTS]QBot = undefined;
var initialized: bool = false;

// Public interface matching strategy.zig
pub var expected_next_pos: [MAX_BOTS]Pos = undefined;
pub var expected_count: u8 = 0;
pub var offset_detected: bool = false;

pub fn initPbots() void {
    initialized = false;
    expected_count = 0;
    offset_detected = false;
    for (0..MAX_BOTS) |i| {
        qbots[i].phase = .restock;
        qbots[i].type_count = 0;
        qbots[i].last_pos = .{ .x = -1, .y = -1 };
        qbots[i].stall_count = 0;
        qbots[i].last_dir = null;
        qbots[i].escape_rounds = 0;
    }
    pathfinding.resetPrecompute();
}

// ── Type Assignment ────────────────────────────────────────────────────
// Collect unique types from map items, assign 2 per bot round-robin.
fn assignTypes(state: *const GameState) void {
    var unique: [16]ItemType = undefined;
    var ucount: u8 = 0;
    for (0..state.item_count) |i| {
        var found = false;
        for (0..ucount) |t| {
            if (unique[t].eql(state.items[i].item_type)) { found = true; break; }
        }
        if (!found and ucount < 16) {
            unique[ucount] = state.items[i].item_type;
            ucount += 1;
        }
    }

    // Pair up types and assign to bots
    // With N types and B bots: pair (0,1) → bot 0, (2,3) → bot 1, etc.
    // Extra bots get duplicate pairs for redundancy
    const num_pairs = (ucount + 1) / 2;
    for (0..state.bot_count) |bi| {
        const pair_idx = bi % num_pairs;
        const t0 = pair_idx * 2;
        const t1 = pair_idx * 2 + 1;
        qbots[bi].types[0] = unique[t0];
        qbots[bi].type_count = 1;
        if (t1 < ucount) {
            qbots[bi].types[1] = unique[t1];
            qbots[bi].type_count = 2;
        }
    }

    // Log assignments
    for (0..state.bot_count) |bi| {
        const qb = &qbots[bi];
        if (qb.type_count == 2) {
            std.debug.print("QBot{d}: {s}, {s}\n", .{ bi, qb.types[0].str(), qb.types[1].str() });
        } else {
            std.debug.print("QBot{d}: {s}\n", .{ bi, qb.types[0].str() });
        }
    }
    std.debug.print("Queue strategy: {d} types, {d} bots, {d} pairs\n", .{ ucount, state.bot_count, num_pairs });
    initialized = true;
}

fn isMyType(bi: usize, t: ItemType) bool {
    for (0..qbots[bi].type_count) |ti| {
        if (qbots[bi].types[ti].eql(t)) return true;
    }
    return false;
}

fn hasActiveItems(bot: *const Bot, active: *const NeedList) bool {
    var work = active.*;
    for (0..bot.inv_len) |ii| {
        if (work.contains(bot.inv[ii])) return true;
    }
    return false;
}

fn countActiveItems(bot: *const Bot, active: *const NeedList) u8 {
    var count: u8 = 0;
    var work = active.*;
    for (0..bot.inv_len) |ii| {
        if (work.contains(bot.inv[ii])) {
            count += 1;
            work.remove(bot.inv[ii]);
        }
    }
    return count;
}

// How many more of my assigned types do I need to fill inventory?
fn needsRestock(bi: usize, bot: *const Bot) bool {
    if (bot.inv_len >= INV_CAP) return false;
    // Check if there are assigned types we could still pick up
    var have: [2]u8 = .{ 0, 0 };
    for (0..bot.inv_len) |ii| {
        for (0..qbots[bi].type_count) |ti| {
            if (qbots[bi].types[ti].eql(bot.inv[ii])) { have[ti] += 1; break; }
        }
    }
    // Target: fill slots evenly across assigned types
    const qb = &qbots[bi];
    if (qb.type_count == 2) {
        // Target: 2 of first type, 1 of second (or 1+2)
        return have[0] + have[1] < INV_CAP;
    } else {
        return have[0] < INV_CAP;
    }
}

// ── Helpers ────────────────────────────────────────────────────────────
fn writeMove(writer: anytype, bot_id: u8, dir: Dir) !void {
    const s = switch (dir) { .up => "move_up", .down => "move_down", .left => "move_left", .right => "move_right" };
    try writer.print("{{\"bot\":{d},\"action\":\"{s}\"}}", .{ bot_id, s });
}

fn updateBotPos(pos: *Pos, dir: Dir) void {
    switch (dir) { .up => pos.y -= 1, .down => pos.y += 1, .left => pos.x -= 1, .right => pos.x += 1 }
}

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
        var blocked = false;
        for (0..state.bot_count) |bi| {
            if (bi == bot_id) continue;
            if (bot_positions[bi].x == nx and bot_positions[bi].y == ny) { blocked = true; break; }
        }
        if (!blocked) return d;
    }
    return null;
}

// ── Main Decision Engine ──────────────────────────────────────────────
pub fn decideActions(state: *GameState, out_buf: []u8) ![]const u8 {
    if (state.round == 0) {
        pathfinding.precomputeAllDistances(state);
        assignTypes(state);
    }

    var stream = std.io.fixedBufferStream(out_buf);
    var writer = stream.writer();
    try writer.writeAll("{\"actions\":[");

    // Build active/preview needs
    var active = NeedList.init();
    var preview = NeedList.init();
    for (0..state.order_count) |oi| {
        const order = &state.orders[oi];
        if (order.complete) continue;
        var need = NeedList.init();
        for (0..order.required_len) |ri| need.add(order.required[ri]);
        for (0..order.delivered_len) |di| need.remove(order.delivered[di]);
        if (order.is_active) { active = need; } else { preview = need; }
    }

    const dm_drop = pathfinding.getPrecomputedDm(state, state.dropoff);

    var bot_positions: [MAX_BOTS]Pos = undefined;
    for (0..state.bot_count) |bi| bot_positions[bi] = state.bots[bi].pos;

    // Determine delivery priority: sort bots with active items by distance to dropoff
    var deliver_bots: [MAX_BOTS]struct { bi: u8, dist: u16 } = undefined;
    var dcount: u8 = 0;
    for (0..state.bot_count) |bi| {
        if (countActiveItems(&state.bots[bi], &active) > 0) {
            deliver_bots[dcount] = .{ .bi = @intCast(bi), .dist = pathfinding.distFromMap(dm_drop, state.bots[bi].pos) };
            dcount += 1;
        }
    }
    // Sort by distance (closest first)
    for (0..dcount) |i| {
        var min_j = i;
        for (i + 1..dcount) |j| {
            if (deliver_bots[j].dist < deliver_bots[min_j].dist) min_j = j;
        }
        if (min_j != i) {
            const tmp = deliver_bots[i];
            deliver_bots[i] = deliver_bots[min_j];
            deliver_bots[min_j] = tmp;
        }
    }

    // Only top 2 bots actively deliver (limit dropoff congestion)
    var should_deliver: [MAX_BOTS]bool = .{false} ** MAX_BOTS;
    for (0..@min(dcount, 2)) |i| {
        should_deliver[deliver_bots[i].bi] = true;
    }

    // Stall detection & phase transitions
    for (0..state.bot_count) |bi| {
        const bot = &state.bots[bi];
        const qb = &qbots[bi];

        if (qb.last_pos.eql(bot.pos)) {
            qb.stall_count += 1;
        } else {
            qb.stall_count = 0;
            const dx = bot.pos.x - qb.last_pos.x;
            const dy = bot.pos.y - qb.last_pos.y;
            if (dx == 1) qb.last_dir = .right
            else if (dx == -1) qb.last_dir = .left
            else if (dy == 1) qb.last_dir = .down
            else if (dy == -1) qb.last_dir = .up;
        }
        qb.last_pos = bot.pos;

        // Escape after 8 rounds stalled
        if (qb.stall_count >= 8) {
            qb.stall_count = 0;
            qb.escape_rounds = 3;
        }
        if (qb.escape_rounds > 0) qb.escape_rounds -= 1;

        // Phase transitions
        if (qb.phase == .restock and bot.inv_len >= INV_CAP) {
            qb.phase = .ready;
        } else if (qb.phase == .ready and bot.inv_len == 0) {
            qb.phase = .restock;
        }
    }

    // Per-bot actions
    for (0..state.bot_count) |bi| {
        if (bi > 0) try writer.writeAll(",");

        const bot = &state.bots[bi];
        const qb = &qbots[bi];
        const bpos = bot.pos;
        const has_active = hasActiveItems(bot, &active);

        // 0. At dropoff with active items → drop_off
        if (bpos.eql(state.dropoff) and bot.inv_len > 0 and has_active) {
            try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
            expected_next_pos[bi] = bpos;
            continue;
        }

        // 0b. Auto-completion: active order fully delivered but needs drop_off trigger
        if (bpos.eql(state.dropoff) and bot.inv_len > 0 and active.count == 0) {
            // Check if there IS an active order (all items delivered via chain)
            var has_active_order = false;
            for (0..state.order_count) |oi| {
                if (state.orders[oi].is_active and !state.orders[oi].complete) {
                    has_active_order = true;
                    break;
                }
            }
            if (has_active_order) {
                try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
                expected_next_pos[bi] = bpos;
                continue;
            }
        }

        // 1. At dropoff without active items → flee
        if (bpos.eql(state.dropoff) and !has_active and state.bot_count > 1) {
            if (fleeDropoff(state, bpos, @intCast(bi), &bot_positions)) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                expected_next_pos[bi] = bot_positions[bi];
                continue;
            }
        }

        // 2. Escape stall
        if (qb.escape_rounds > 0) {
            // Try any walkable direction not reverse of last
            const offx = [4]i16{ 0, 0, -1, 1 };
            const offy = [4]i16{ -1, 1, 0, 0 };
            const dirs = [4]Dir{ .up, .down, .left, .right };
            const reverse: ?Dir = if (qb.last_dir) |ld| switch (ld) {
                .up => .down, .down => .up, .left => .right, .right => .left,
            } else null;
            var escaped = false;
            for (dirs, 0..) |d, di| {
                if (reverse != null and d == reverse.?) continue;
                const nx = bpos.x + offx[di];
                const ny = bpos.y + offy[di];
                if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
                const cell = state.grid[@intCast(ny)][@intCast(nx)];
                if (cell == .wall or cell == .shelf) continue;
                var blocked = false;
                for (0..state.bot_count) |bk| {
                    if (bk == bi) continue;
                    if (bot_positions[bk].x == nx and bot_positions[bk].y == ny) { blocked = true; break; }
                }
                if (!blocked) {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    expected_next_pos[bi] = bot_positions[bi];
                    escaped = true;
                    break;
                }
            }
            if (escaped) continue;
        }

        // 3. Adjacent to needed item → pick up (ONLY assigned types in restock)
        var picked = false;
        if (bot.inv_len < INV_CAP) {
            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                const mdist = @abs(bpos.x - item.pos.x) + @abs(bpos.y - item.pos.y);
                if (mdist != 1) continue;

                // Strict type discipline: only pick assigned types
                if (!isMyType(bi, item.item_type)) continue;

                // In restock: always pick assigned types to fill inventory
                // In ready: only pick if it matches active or preview
                if (qb.phase == .ready) {
                    if (!active.contains(item.item_type) and !preview.contains(item.item_type)) continue;
                }

                try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                expected_next_pos[bi] = bpos;
                picked = true;
                break;
            }
        }
        if (picked) continue;

        // 4. Phase-specific navigation
        const dm_bot = pathfinding.getPrecomputedDm(state, bpos);

        if (should_deliver[bi]) {
            // DELIVER: navigate to dropoff
            if (!bpos.eql(state.dropoff)) {
                const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
                if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    expected_next_pos[bi] = bot_positions[bi];
                    continue;
                };
            }
            try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
            expected_next_pos[bi] = bpos;
            continue;
        }

        switch (qb.phase) {
            .restock => {
                // Find closest shelf with a type I need
                var best_adj: ?Pos = null;
                var best_dist: u16 = UNREACHABLE;
                // Figure out what I still need
                var have: [2]u8 = .{ 0, 0 };
                for (0..bot.inv_len) |ii| {
                    for (0..qbots[bi].type_count) |ti| {
                        if (qbots[bi].types[ti].eql(bot.inv[ii])) { have[ti] += 1; break; }
                    }
                }
                // Target fill: with 2 types → 2+1; with 1 type → 3
                var want_type: [2]u8 = .{ 0, 0 };
                if (qbots[bi].type_count == 2) {
                    want_type[0] = if (have[0] < 2) 2 - have[0] else 0;
                    want_type[1] = if (have[1] < 1) 1 - have[1] else 0;
                    // If slots remain, add more of first type
                    const total = have[0] + have[1] + want_type[0] + want_type[1];
                    if (total < INV_CAP) want_type[0] += INV_CAP - total;
                } else {
                    want_type[0] = INV_CAP - have[0];
                }

                for (0..state.item_count) |ii| {
                    const item = &state.items[ii];
                    var needed = false;
                    for (0..qbots[bi].type_count) |ti| {
                        if (want_type[ti] > 0 and qbots[bi].types[ti].eql(item.item_type)) {
                            needed = true;
                            break;
                        }
                    }
                    if (!needed) continue;
                    const adj = pathfinding.findBestAdj(state, item.pos, dm_bot) orelse continue;
                    const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                    if (d < best_dist) {
                        best_dist = d;
                        best_adj = adj;
                    }
                }

                if (best_adj) |target| {
                    if (!bpos.eql(target)) {
                        const res = pathfinding.bfs(state, bpos, target, @intCast(bi), &bot_positions);
                        if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                            try writeMove(writer, bot.id, d);
                            updateBotPos(&bot_positions[bi], d);
                            expected_next_pos[bi] = bot_positions[bi];
                            continue;
                        };
                    }
                }

                // At target or no target found → wait
                try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                expected_next_pos[bi] = bpos;
            },
            .ready => {
                // Navigate toward dropoff area (stop at distance 2-4)
                const dist_to_drop = pathfinding.distFromMap(dm_drop, bpos);
                if (dist_to_drop > 4) {
                    const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        expected_next_pos[bi] = bot_positions[bi];
                        continue;
                    };
                }
                // Near dropoff → wait for delivery signal
                try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                expected_next_pos[bi] = bpos;
            },
        }
    }

    try writer.writeAll("]}");
    expected_count = @intCast(state.bot_count);
    return stream.getWritten();
}

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
    // Fix 5: Track rounds on current order for stuck recovery
    rounds_on_order: u16,
};

var pbots: [MAX_BOTS]PersistentBot = undefined;
var pbots_initialized: bool = false;

// Score tracking for diagnostic logging (Fix 6)
var last_score: i32 = 0;
var rounds_since_score_change: u32 = 0;

// Expected positions after action (for shift detection in main.zig)
pub var expected_next_pos: [MAX_BOTS]Pos = undefined;
pub var expected_count: u8 = 0;

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
    }
    blacklist_count = 0;
    last_score = 0;
    rounds_since_score_change = 0;
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
    var active = needs.active;
    var preview = needs.preview;
    const rounds_left = if (state.max_rounds > state.round) state.max_rounds - state.round else 0;

    // ── Fix 6: Track score changes for diagnostic logging ──
    if (state.score != last_score) {
        rounds_since_score_change = 0;
        last_score = state.score;
    } else {
        rounds_since_score_change += 1;
    }

    // ── Distance from dropoff (computed BEFORE order_stuck for Fix 2 reachability) ──
    var dm_drop: DistMap = undefined;
    pathfinding.bfsDistMap(state, state.dropoff, &dm_drop);

    // ── Detect stuck order (Fix 2: add reachability check) ──
    var order_stuck = false;
    {
        var temp = active;
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

    // Bot positions for collision
    var bot_positions: [MAX_BOTS]Pos = undefined;
    for (0..state.bot_count) |bi| bot_positions[bi] = state.bots[bi].pos;

    // Claimed items
    var claimed: [MAX_ITEMS]i8 = undefined;
    @memset(claimed[0..state.item_count], -1);

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
        }
        pb.last_pos = bot.pos;

        // Stall reset (same position for 8+ rounds)
        if (pb.stall_count > 8) {
            pb.has_trip = false;
            pb.delivering = false;
            pb.stall_count = 0;
            pb.osc_count += 1;
        }

        // Fix 3: Oscillation detection — removed dropoff exclusion
        if (isOscillating(pb, bot.pos)) {
            pb.osc_count += 1;
            if (pb.osc_count % 10 == 0) {
                pb.has_trip = false;
                pb.delivering = false;
                std.debug.print("R{d} Bot{d} osc reset at ({d},{d}) osc={d}\n", .{ state.round, bot.id, bot.pos.x, bot.pos.y, pb.osc_count });
            }
        } else if (pb.osc_count > 0) {
            pb.osc_count -|= 1;
        }

        // Detect failed pickups
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

        // Fix 5: Force reset after being stuck on same order too long
        if (pb.rounds_on_order > 50 and pb.rounds_on_order % 25 == 0) {
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

        pb.last_tried_pickup = false;

        var dm_bot: DistMap = undefined;
        pathfinding.bfsDistMap(state, bot.pos, &dm_bot);

        // Build per-bot need: active minus this bot's inventory
        var bot_active = active;
        var has_active = false;
        for (0..bot.inv_len) |ii| {
            if (bot_active.contains(bot.inv[ii])) {
                has_active = true;
                bot_active.remove(bot.inv[ii]);
            }
        }
        var bot_preview = preview;
        for (0..bot.inv_len) |ii| bot_preview.remove(bot.inv[ii]);

        // Re-validate trip against current needs (items may no longer be needed after delivery)
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
                } else if (check_p.contains(it)) {
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
        if (bot.pos.eql(state.dropoff) and bot.inv_len > 0 and has_active) {
            try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
            pb.has_trip = false;
            pb.delivering = false;
            continue;
        }

        // ─── 1b. At dropoff with dead inventory → evacuate to free dropoff ──
        if (bot.pos.eql(state.dropoff) and bot.inv_len > 0 and !has_active) {
            // Move away from dropoff so other bots can deliver
            const flee_dir = fleeDropoff(state, bot.pos, @intCast(bi), &bot_positions);
            if (flee_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                continue;
            }
            // All exits blocked — wait and hope another bot moves
            try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
            continue;
        }

        // ─── 2. Adjacent needed item → pick up ──────────────────────
        var picked = false;
        for (0..2) |pass| {
            if (picked) break;
            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                const mdist = @abs(bot.pos.x - item.pos.x) + @abs(bot.pos.y - item.pos.y);
                if (mdist != 1) continue;
                if (bot.inv_len >= INV_CAP) break;
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;

                if (pass == 0) {
                    if (!bot_active.contains(item.item_type)) continue;
                } else {
                    // Don't opportunistically pick up preview items when active items
                    // are still needed — let the trip planner handle the ordering so
                    // preview items don't steal inventory slots from active items.
                    if (bot_active.count > 0) continue;
                    if (bot_active.contains(item.item_type)) continue;
                    if (!bot_preview.contains(item.item_type)) continue;
                }

                try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                pb.last_tried_pickup = true;
                pb.last_pickup_pos = bot.pos;
                pb.last_pickup_item_pos = item.pos;
                pb.last_inv_len = bot.inv_len;
                active.remove(item.item_type);
                preview.remove(item.item_type);
                bot_active.remove(item.item_type);
                bot_preview.remove(item.item_type);
                claimed[ii] = @intCast(bi);

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
            var temp = bot_active;
            for (0..state.item_count) |ii| {
                if (claimed[ii] >= 0 and claimed[ii] != @as(i8, @intCast(bi))) continue;
                if (temp.contains(state.items[ii].item_type)) {
                    active_on_map += 1;
                    temp.remove(state.items[ii].item_type);
                }
            }
        }

        const inv_full = bot.inv_len >= INV_CAP;
        const all_active_got = bot_active.count == 0;
        const no_active_avail = active_on_map == 0;
        const trip_done = pb.has_trip and pb.trip_pos >= pb.trip_count;
        const effective_slots = if (INV_CAP > bot.inv_len) INV_CAP - bot.inv_len else 0;

        const dist_to_drop = pathfinding.distFromMap(&dm_drop, bot.pos);
        const endgame = rounds_left <= dist_to_drop + 3;

        const should_deliver = has_active and (inv_full or all_active_got or no_active_avail or order_stuck or endgame or trip_done);

        if (should_deliver) {
            pb.delivering = true;
            pb.has_trip = false;
        }

        // ── Fix 1: CRITICAL — Reset delivering when no active items ──
        // When the bot has no active items to deliver, it should NEVER be in delivering state.
        // This prevents the deadlock where has_active=false + delivering=true blocks all actions.
        if (!has_active) {
            pb.delivering = false;
        }

        // ── Fix 6: Diagnostic logging when score stalls ──
        if (rounds_since_score_change >= 10 and state.round % 5 == 0) {
            std.debug.print("R{d} STALL Bot{d} pos=({d},{d}) inv={d} has_active={any} delivering={any} eff_slots={d} trip={any} stuck={any} order_rds={d}\n", .{
                state.round, bot.id, bot.pos.x, bot.pos.y,
                bot.inv_len, has_active, pb.delivering, effective_slots,
                pb.has_trip, order_stuck, pb.rounds_on_order,
            });
        }

        // ─── 4. Delivering → go to dropoff ──────────────────────────
        if (pb.delivering and has_active) {
            const res = pathfinding.bfs(state, bot.pos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) {
                if (res.first_dir) |d| {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    continue;
                }
            }
            // BFS unreachable (blocked by bots) — fall through to other steps
            // Don't use greedy fallback, which causes futile oscillation near dropoff
        }

        // ─── 5. Follow existing trip plan ────────────────────────────
        if (pb.has_trip and pb.trip_pos < pb.trip_count) {
            const trip_id = pb.trip_ids[pb.trip_pos][0..pb.trip_id_lens[pb.trip_pos]];
            const idx = findItemById(state, trip_id);
            if (idx) |item_idx| {
                const item = &state.items[item_idx];
                const adj = pb.trip_adjs[pb.trip_pos];
                if (bot.pos.eql(adj)) {
                    if (bot.inv_len < INV_CAP) {
                        try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                        pb.last_tried_pickup = true;
                        pb.last_pickup_pos = bot.pos;
                        pb.last_pickup_item_pos = item.pos;
                        pb.last_inv_len = bot.inv_len;
                        active.remove(item.item_type);
                        preview.remove(item.item_type);
                        claimed[item_idx] = @intCast(bi);
                        pb.trip_pos += 1;
                        if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
                        continue;
                    }
                } else {
                    const res = pathfinding.bfs(state, bot.pos, adj, @intCast(bi), &bot_positions);
                    if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        active.remove(item.item_type);
                        preview.remove(item.item_type);
                        claimed[item_idx] = @intCast(bi);
                        continue;
                    }
                }
            }
            pb.has_trip = false;
        }

        // ─── 6. Plan new trip ──────────────────────────────────────
        if (effective_slots > 0 and !pb.delivering) {
            const t = trip_mod.planBestTrip(state, &dm_bot, &dm_drop, &bot_active, &bot_preview, &claimed, bi, @intCast(effective_slots), true, rounds_left);
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
                    active.remove(item.item_type);
                    preview.remove(item.item_type);
                }

                const first_adj = tp.adjs[0];
                const first_item = &state.items[tp.items[0]];
                if (bot.pos.eql(first_adj)) {
                    if (bot.inv_len < INV_CAP) {
                        try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, first_item.idStr() });
                        pb.last_tried_pickup = true;
                        pb.last_pickup_pos = bot.pos;
                        pb.last_pickup_item_pos = first_item.pos;
                        pb.last_inv_len = bot.inv_len;
                        pb.trip_pos += 1;
                        if (pb.trip_pos >= pb.trip_count) pb.has_trip = false;
                        continue;
                    }
                } else {
                    const res = pathfinding.bfs(state, bot.pos, first_adj, @intCast(bi), &bot_positions);
                    if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        continue;
                    }
                }
            }
        }

        // ─── 7. Have active items → go deliver ─────────────────────
        if (bot.inv_len > 0 and has_active) {
            pb.delivering = true;
            const res = pathfinding.bfs(state, bot.pos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                continue;
            };
        }

        // ─── 8. Has active items but couldn't deliver yet → go to dropoff ──
        // Only navigate to dropoff when has_active — dead inventory bots
        // must NOT go to dropoff (step 1b evacuates them, this would undo it)
        if (bot.inv_len > 0 and has_active and !bot.pos.eql(state.dropoff)) {
            const res = pathfinding.bfs(state, bot.pos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                continue;
            };
        }

        // ─── 8b. Dead inventory near dropoff → clear corridor ─────
        // Bots with dead inventory block the corridor for delivering bots.
        // Move away from dropoff until distance >= 8 or at local max.
        if (bot.inv_len > 0 and !has_active) {
            const drop_dist = pathfinding.distFromMap(&dm_drop, bot.pos);
            if (drop_dist >= 8) {
                try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
                continue;
            }
            const offx = [4]i16{ 0, 0, -1, 1 };
            const offy = [4]i16{ -1, 1, 0, 0 };
            const dirs = [4]Dir{ .up, .down, .left, .right };
            var best_dir: ?Dir = null;
            var best_dist: u16 = drop_dist;
            for (dirs, 0..) |d, di| {
                const nx = bot.pos.x + offx[di];
                const ny = bot.pos.y + offy[di];
                if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
                const cell = state.grid[@intCast(ny)][@intCast(nx)];
                if (cell == .wall or cell == .shelf) continue;
                var occupied = false;
                for (0..state.bot_count) |bbi| {
                    if (bbi == bi) continue;
                    if (bot_positions[bbi].x == nx and bot_positions[bbi].y == ny) { occupied = true; break; }
                }
                if (occupied) continue;
                const nd = pathfinding.distFromMap(&dm_drop, .{ .x = nx, .y = ny });
                if (nd > best_dist) {
                    best_dist = nd;
                    best_dir = d;
                }
            }
            if (best_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                continue;
            }
        }

        // ─── 9. Wait ────────────────────────────────────────────────
        try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
    }

    try writer.writeAll("]}");

    // Record expected positions for shift detection
    for (0..state.bot_count) |bi| {
        expected_next_pos[bi] = bot_positions[bi];
    }
    expected_count = @intCast(state.bot_count);

    return stream.getWritten();
}

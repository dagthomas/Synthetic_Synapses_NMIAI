const std = @import("std");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");

const Pos = types.Pos;
const Dir = types.Dir;
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

pub var expected_next_pos: [MAX_BOTS]Pos = undefined;
pub var expected_count: u8 = 0;
pub var offset_detected: bool = false;

pub fn initPbots() void {
    expected_count = 0;
    offset_detected = false;
    pathfinding.resetPrecompute();
}

fn writeMove(writer: anytype, bot_id: u8, dir: Dir) !void {
    const s = switch (dir) { .up => "move_up", .down => "move_down", .left => "move_left", .right => "move_right" };
    try writer.print("{{\"bot\":{d},\"action\":\"{s}\"}}", .{ bot_id, s });
}

fn updateBotPos(pos: *Pos, dir: Dir) void {
    switch (dir) { .up => pos.y -= 1, .down => pos.y += 1, .left => pos.x -= 1, .right => pos.x += 1 }
}

pub fn decideActions(state: *GameState, out_buf: []u8) ![]const u8 {
    if (state.round == 0) {
        pathfinding.precomputeAllDistances(state);
    }

    var stream = std.io.fixedBufferStream(out_buf);
    var writer = stream.writer();
    try writer.writeAll("{\"actions\":[");

    // Build needs for active and preview orders
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

    // Build "unclaimed active" = active needs minus items already being carried by all bots
    var unclaimed = active;
    for (0..state.bot_count) |bi| {
        const bot = &state.bots[bi];
        for (0..bot.inv_len) |ii| {
            if (unclaimed.contains(bot.inv[ii])) {
                unclaimed.remove(bot.inv[ii]);
            }
        }
    }

    var bot_positions: [MAX_BOTS]Pos = undefined;
    for (0..state.bot_count) |bi| bot_positions[bi] = state.bots[bi].pos;

    for (0..state.bot_count) |bi| {
        if (bi > 0) try writer.writeAll(",");
        const bot = &state.bots[bi];
        const bpos = bot.pos;

        // Does this bot have items matching the active order?
        var active_match: u8 = 0;
        {
            var work = active;
            for (0..bot.inv_len) |ii| {
                if (work.contains(bot.inv[ii])) { active_match += 1; work.remove(bot.inv[ii]); }
            }
        }
        const has_active = active_match > 0;

        // Does this bot have items matching the preview order?
        var has_preview = false;
        {
            var work = preview;
            for (0..bot.inv_len) |ii| {
                if (work.contains(bot.inv[ii])) { has_preview = true; work.remove(bot.inv[ii]); }
            }
        }

        // 1. At dropoff with active items → drop_off
        if (bpos.eql(state.dropoff) and bot.inv_len > 0 and has_active) {
            try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
            expected_next_pos[bi] = bpos;
            continue;
        }

        // 1b. At dropoff, order fully covered → drop_off for chain reaction
        if (bpos.eql(state.dropoff) and bot.inv_len > 0 and active.count == 0) {
            var has_active_order = false;
            for (0..state.order_count) |oi| {
                if (state.orders[oi].is_active and !state.orders[oi].complete) { has_active_order = true; break; }
            }
            if (has_active_order) {
                try writer.print("{{\"bot\":{d},\"action\":\"drop_off\"}}", .{bot.id});
                expected_next_pos[bi] = bpos;
                continue;
            }
        }

        // 2. Adjacent to needed item → pick up
        var picked = false;
        if (bot.inv_len < INV_CAP) {
            // Try active unclaimed items first
            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                if (@abs(bpos.x - item.pos.x) + @abs(bpos.y - item.pos.y) != 1) continue;
                if (unclaimed.contains(item.item_type)) {
                    try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                    expected_next_pos[bi] = bpos;
                    unclaimed.remove(item.item_type);
                    picked = true;
                    break;
                }
            }
            // Try preview items if bot already has active items (chain reaction prep)
            if (!picked and has_active) {
                for (0..state.item_count) |ii| {
                    const item = &state.items[ii];
                    if (@abs(bpos.x - item.pos.x) + @abs(bpos.y - item.pos.y) != 1) continue;
                    if (preview.contains(item.item_type)) {
                        try writer.print("{{\"bot\":{d},\"action\":\"pick_up\",\"item_id\":\"{s}\"}}", .{ bot.id, item.idStr() });
                        expected_next_pos[bi] = bpos;
                        picked = true;
                        break;
                    }
                }
            }
        }
        if (picked) continue;

        // 3. Has active items → go deliver
        if (has_active) {
            const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
            if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                try writeMove(writer, bot.id, d);
                updateBotPos(&bot_positions[bi], d);
                expected_next_pos[bi] = bot_positions[bi];
                continue;
            };
        }

        // 4. Inventory not full and active still needs items → go get them
        if (bot.inv_len < INV_CAP and unclaimed.count > 0) {
            const dm_bot = pathfinding.getPrecomputedDm(state, bpos);
            var best_adj: ?Pos = null;
            var best_dist: u16 = UNREACHABLE;

            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                if (!unclaimed.contains(item.item_type)) continue;
                const adj = pathfinding.findBestAdj(state, item.pos, dm_bot) orelse continue;
                const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                if (d < best_dist) { best_dist = d; best_adj = adj; }
            }

            if (best_adj) |tgt| {
                // Claim this type so next bot doesn't duplicate
                for (0..state.item_count) |ii| {
                    const item = &state.items[ii];
                    const adj = pathfinding.findBestAdj(state, item.pos, dm_bot) orelse continue;
                    if (adj.eql(tgt)) {
                        unclaimed.remove(item.item_type);
                        break;
                    }
                }

                if (!bpos.eql(tgt)) {
                    const res = pathfinding.bfs(state, bpos, tgt, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        expected_next_pos[bi] = bot_positions[bi];
                        continue;
                    };
                }
            }
        }

        // 5. Inventory not full → get preview items
        if (bot.inv_len < INV_CAP and preview.count > 0) {
            const dm_bot = pathfinding.getPrecomputedDm(state, bpos);
            var best_adj: ?Pos = null;
            var best_dist: u16 = UNREACHABLE;

            for (0..state.item_count) |ii| {
                const item = &state.items[ii];
                if (!preview.contains(item.item_type)) continue;
                const adj = pathfinding.findBestAdj(state, item.pos, dm_bot) orelse continue;
                const d = dm_bot[@intCast(adj.y)][@intCast(adj.x)];
                if (d < best_dist) { best_dist = d; best_adj = adj; }
            }

            if (best_adj) |tgt| {
                if (!bpos.eql(tgt)) {
                    const res = pathfinding.bfs(state, bpos, tgt, @intCast(bi), &bot_positions);
                    if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                        try writeMove(writer, bot.id, d);
                        updateBotPos(&bot_positions[bi], d);
                        expected_next_pos[bi] = bot_positions[bi];
                        continue;
                    };
                }
            }
        }

        // 6. Has preview items → camp near dropoff
        if (bot.inv_len > 0 and has_preview) {
            const dm_drop = pathfinding.getPrecomputedDm(state, state.dropoff);
            const dist_drop = pathfinding.distFromMap(dm_drop, bpos);
            if (dist_drop > 2) {
                const res = pathfinding.bfs(state, bpos, state.dropoff, @intCast(bi), &bot_positions);
                if (res.dist < UNREACHABLE) if (res.first_dir) |d| {
                    try writeMove(writer, bot.id, d);
                    updateBotPos(&bot_positions[bi], d);
                    expected_next_pos[bi] = bot_positions[bi];
                    continue;
                };
            }
        }

        // 7. Wait
        try writer.print("{{\"bot\":{d},\"action\":\"wait\"}}", .{bot.id});
        expected_next_pos[bi] = bpos;
    }

    try writer.writeAll("]}");
    expected_count = @intCast(state.bot_count);
    return stream.getWritten();
}

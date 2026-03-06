const std = @import("std");
const types = @import("types.zig");

const ItemType = types.ItemType;
const GameState = types.GameState;
const MAX_ITEMS = types.MAX_ITEMS;
const MAX_BOTS = types.MAX_BOTS;
const MAX_ORDERS = types.MAX_ORDERS;
const MAX_H = types.MAX_H;

var json_buf: [1024 * 1024]u8 = undefined;

pub fn parseGameState(data: []const u8, state: *GameState) !bool {
    var fba = std.heap.FixedBufferAllocator.init(&json_buf);
    const alloc = fba.allocator();
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, data, .{});
    _ = &parsed;
    const root = parsed.value.object;

    const msg_type = root.get("type").?.string;
    if (std.mem.eql(u8, msg_type, "game_over")) {
        const score = root.get("score").?.integer;
        const rounds = root.get("rounds_used").?.integer;
        std.debug.print("Game over! Score: {d}, Rounds: {d}\n", .{ score, rounds });
        return false;
    }
    if (!std.mem.eql(u8, msg_type, "game_state")) return false;

    state.round = @intCast(root.get("round").?.integer);
    state.max_rounds = @intCast(root.get("max_rounds").?.integer);
    state.score = @intCast(root.get("score").?.integer);
    state.active_order_idx = @intCast(root.get("active_order_index").?.integer);

    const grid = root.get("grid").?.object;
    state.width = @intCast(grid.get("width").?.integer);
    state.height = @intCast(grid.get("height").?.integer);

    for (0..state.height) |y| for (0..state.width) |x| {
        state.grid[y][x] = .floor;
    };

    if (state.round == 0) {
        for (0..MAX_H) |y| @memset(&state.known_shelves[y], false);
    }

    const walls = grid.get("walls").?.array.items;
    for (walls) |w| {
        const wx: u16 = @intCast(w.array.items[0].integer);
        const wy: u16 = @intCast(w.array.items[1].integer);
        state.grid[wy][wx] = .wall;
    }

    // Parse dropoff(s): try drop_off_zones (array of positions), fall back to single drop_off
    state.dropoff_count = 0;
    if (root.get("drop_off_zones")) |zones_val| {
        // Multi-dropoff: [[x1,y1], [x2,y2], ...]
        const zones = zones_val.array.items;
        for (zones) |z| {
            if (state.dropoff_count >= types.MAX_DROPOFFS) break;
            const zx: i16 = @intCast(z.array.items[0].integer);
            const zy: i16 = @intCast(z.array.items[1].integer);
            state.dropoffs[state.dropoff_count] = .{ .x = zx, .y = zy };
            state.dropoff_count += 1;
            state.grid[@intCast(zy)][@intCast(zx)] = .dropoff;
        }
        if (state.dropoff_count > 0) {
            state.dropoff = state.dropoffs[0]; // primary = first
        }
    }
    if (state.dropoff_count == 0) {
        // Single dropoff (backward compat)
        const drop = root.get("drop_off").?.array.items;
        state.dropoff = .{ .x = @intCast(drop[0].integer), .y = @intCast(drop[1].integer) };
        state.dropoffs[0] = state.dropoff;
        state.dropoff_count = 1;
        state.grid[@intCast(state.dropoff.y)][@intCast(state.dropoff.x)] = .dropoff;
    }

    const items = root.get("items").?.array.items;
    state.item_count = @intCast(@min(items.len, MAX_ITEMS));
    for (0..state.item_count) |i| {
        const item = items[i].object;
        const id = item.get("id").?.string;
        const itype = item.get("type").?.string;
        const pos = item.get("position").?.array.items;
        state.items[i] = .{
            .id_buf = undefined,
            .id_len = @intCast(@min(id.len, 32)),
            .item_type = ItemType.fromStr(itype),
            .pos = .{ .x = @intCast(pos[0].integer), .y = @intCast(pos[1].integer) },
        };
        @memcpy(state.items[i].id_buf[0..state.items[i].id_len], id[0..state.items[i].id_len]);
        const sx: u16 = @intCast(pos[0].integer);
        const sy: u16 = @intCast(pos[1].integer);
        state.known_shelves[sy][sx] = true;
    }

    for (0..state.height) |y| for (0..state.width) |x| {
        if (state.known_shelves[y][x] and state.grid[y][x] == .floor) state.grid[y][x] = .shelf;
    };

    const bots = root.get("bots").?.array.items;
    state.bot_count = @intCast(@min(bots.len, MAX_BOTS));
    for (0..state.bot_count) |i| {
        const b = bots[i].object;
        const pos = b.get("position").?.array.items;
        const inv = b.get("inventory").?.array.items;
        state.bots[i] = .{
            .id = @intCast(b.get("id").?.integer),
            .pos = .{ .x = @intCast(pos[0].integer), .y = @intCast(pos[1].integer) },
            .inv = undefined,
            .inv_len = @intCast(@min(inv.len, types.INV_CAP)),
        };
        for (0..state.bots[i].inv_len) |ii| state.bots[i].inv[ii] = ItemType.fromStr(inv[ii].string);
    }

    const orders = root.get("orders").?.array.items;
    state.order_count = @intCast(@min(orders.len, MAX_ORDERS));
    for (0..state.order_count) |i| {
        const o = orders[i].object;
        const req = o.get("items_required").?.array.items;
        const del = o.get("items_delivered").?.array.items;
        const status = o.get("status").?.string;
        state.orders[i] = .{
            .required = undefined,
            .required_len = @intCast(@min(req.len, 16)),
            .delivered = undefined,
            .delivered_len = @intCast(@min(del.len, 16)),
            .is_active = std.mem.eql(u8, status, "active"),
            .complete = o.get("complete").?.bool,
        };
        for (0..state.orders[i].required_len) |ri| state.orders[i].required[ri] = ItemType.fromStr(req[ri].string);
        for (0..state.orders[i].delivered_len) |di| state.orders[i].delivered[di] = ItemType.fromStr(del[di].string);
    }

    if (state.round == 0) {
        std.debug.print("R0: {d}x{d} grid, {d} bots, {d} items, {d} dropoffs, primary ({d},{d})\n", .{
            state.width, state.height, state.bot_count, state.item_count, state.dropoff_count, state.dropoff.x, state.dropoff.y,
        });
    }
    return true;
}

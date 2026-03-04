const std = @import("std");
const types = @import("types.zig");

const ItemType = types.ItemType;
const NeedList = types.NeedList;
const Order = types.Order;

// ── Precomputed Order Data ──────────────────────────────────────────
// Loaded from capture.json at startup. Provides full order knowledge
// for lookahead planning (dead inventory prevention, pre-positioning).

pub const MAX_FUTURE_ORDERS = 64;
const MAX_ORDER_ITEMS = 16;

const FutureOrder = struct {
    required: [MAX_ORDER_ITEMS]ItemType,
    required_len: u8,
};

var orders: [MAX_FUTURE_ORDERS]FutureOrder = undefined;
var order_count: u16 = 0;
var is_loaded: bool = false;
var is_verified: bool = false;

var file_buf: [512 * 1024]u8 = undefined;
var json_buf: [512 * 1024]u8 = undefined;

/// Load precomputed orders from a capture.json file.
/// Returns true on success.
pub fn load(path: []const u8) bool {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Precomputed: cannot open {s}: {any}\n", .{ path, err });
        return false;
    };
    defer file.close();

    const bytes_read = file.readAll(&file_buf) catch |err| {
        std.debug.print("Precomputed: read error: {any}\n", .{err});
        return false;
    };
    if (bytes_read == 0) {
        std.debug.print("Precomputed: empty file\n", .{});
        return false;
    }

    var fba = std.heap.FixedBufferAllocator.init(&json_buf);
    const alloc = fba.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, alloc, file_buf[0..bytes_read], .{}) catch |err| {
        std.debug.print("Precomputed: JSON parse error: {any}\n", .{err});
        return false;
    };
    _ = &parsed;

    const root = parsed.value.object;
    const orders_arr = (root.get("orders") orelse {
        std.debug.print("Precomputed: no 'orders' key in JSON\n", .{});
        return false;
    }).array.items;

    order_count = 0;
    for (orders_arr) |order_val| {
        if (order_count >= MAX_FUTURE_ORDERS) break;
        const obj = order_val.object;
        const req = (obj.get("items_required") orelse continue).array.items;
        var fo = FutureOrder{ .required = undefined, .required_len = 0 };
        for (req) |item_val| {
            if (fo.required_len >= MAX_ORDER_ITEMS) break;
            fo.required[fo.required_len] = ItemType.fromStr(item_val.string);
            fo.required_len += 1;
        }
        orders[order_count] = fo;
        order_count += 1;
    }

    is_loaded = true;
    is_verified = false;
    std.debug.print("Precomputed: loaded {d} orders from {s}\n", .{ order_count, path });
    return true;
}

/// Verify precomputed data matches the live game by comparing order at active_order_idx.
/// Call on round 0 after receiving the first game state.
/// If mismatch (stale data from different day), disables precomputed data.
pub fn verify(active_order_idx: i32, live_order: *const Order) void {
    if (!is_loaded) return;
    if (active_order_idx < 0) return;

    const idx: u16 = @intCast(@as(u32, @intCast(active_order_idx)));
    if (idx >= order_count) {
        std.debug.print("Precomputed: DISABLED — active_order_idx {d} >= loaded {d}\n", .{ idx, order_count });
        is_loaded = false;
        return;
    }

    const fo = &orders[idx];
    // Compare: same number of required items AND all types match
    if (fo.required_len != live_order.required_len) {
        std.debug.print("Precomputed: DISABLED — order {d} length mismatch ({d} vs {d})\n", .{ idx, fo.required_len, live_order.required_len });
        is_loaded = false;
        return;
    }

    // Check that each required type in precomputed matches live (order may differ, so count-match)
    var pc_counts: [MAX_ORDER_ITEMS]struct { t: ItemType, c: u8 } = undefined;
    var pc_len: u8 = 0;
    for (0..fo.required_len) |i| {
        var found = false;
        for (0..pc_len) |j| {
            if (pc_counts[j].t.eql(fo.required[i])) {
                pc_counts[j].c += 1;
                found = true;
                break;
            }
        }
        if (!found) {
            pc_counts[pc_len] = .{ .t = fo.required[i], .c = 1 };
            pc_len += 1;
        }
    }

    var live_counts: [MAX_ORDER_ITEMS]struct { t: ItemType, c: u8 } = undefined;
    var live_len: u8 = 0;
    for (0..live_order.required_len) |i| {
        var found = false;
        for (0..live_len) |j| {
            if (live_counts[j].t.eql(live_order.required[i])) {
                live_counts[j].c += 1;
                found = true;
                break;
            }
        }
        if (!found) {
            live_counts[live_len] = .{ .t = live_order.required[i], .c = 1 };
            live_len += 1;
        }
    }

    if (pc_len != live_len) {
        std.debug.print("Precomputed: DISABLED — order {d} type count mismatch\n", .{idx});
        is_loaded = false;
        return;
    }

    for (0..pc_len) |i| {
        var matched = false;
        for (0..live_len) |j| {
            if (pc_counts[i].t.eql(live_counts[j].t) and pc_counts[i].c == live_counts[j].c) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            std.debug.print("Precomputed: DISABLED — order {d} type mismatch for '{s}'\n", .{ idx, pc_counts[i].t.str() });
            is_loaded = false;
            return;
        }
    }

    is_verified = true;
    std.debug.print("Precomputed: VERIFIED (order {d} matches)\n", .{idx});
}

/// Check if data is loaded and verified.
pub fn isActive() bool {
    return is_loaded and is_verified;
}

/// Check if data is loaded (may not be verified yet).
pub fn isLoaded() bool {
    return is_loaded;
}

/// Get a future order by absolute index. Returns null if out of range or not loaded.
pub fn getOrder(idx: u16) ?*const FutureOrder {
    if (!is_loaded or idx >= order_count) return null;
    return &orders[idx];
}

/// Check if an item type is needed in ANY order from from_idx onward.
/// Returns true when no data loaded (safe fallback — never wrongly marks as dead).
pub fn typeNeededInFuture(item_type: ItemType, from_idx: u16) bool {
    if (!is_loaded or !is_verified) return true; // Safe: assume needed
    var idx = from_idx;
    while (idx < order_count) : (idx += 1) {
        const fo = &orders[idx];
        for (0..fo.required_len) |i| {
            if (fo.required[i].eql(item_type)) return true;
        }
    }
    return false;
}

/// Build a NeedList for a specific future order.
pub fn futureNeeds(idx: u16) NeedList {
    var nl = NeedList.init();
    if (!is_loaded or idx >= order_count) return nl;
    const fo = &orders[idx];
    for (0..fo.required_len) |i| {
        nl.add(fo.required[i]);
    }
    return nl;
}

/// Count how many of the next `lookahead` orders (starting at from_idx) need this type.
pub fn typeFutureUtility(item_type: ItemType, from_idx: u16, lookahead: u16) u16 {
    if (!is_loaded or !is_verified) return 0;
    var count: u16 = 0;
    const end = @min(from_idx + lookahead, order_count);
    var idx = from_idx;
    while (idx < end) : (idx += 1) {
        const fo = &orders[idx];
        for (0..fo.required_len) |i| {
            if (fo.required[i].eql(item_type)) {
                count += 1;
                break; // Count each order at most once
            }
        }
    }
    return count;
}

/// Get the total number of loaded orders.
pub fn getOrderCount() u16 {
    return order_count;
}

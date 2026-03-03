const std = @import("std");

// ── Constants ──────────────────────────────────────────────────────────
pub const MAX_BOTS = 10;
pub const MAX_ITEMS = 512;
pub const MAX_ORDERS = 2;
pub const MAX_W = 32;
pub const MAX_H = 20;
pub const MAX_CELLS = MAX_W * MAX_H;
pub const INV_CAP = 3;
pub const UNREACHABLE: u16 = 9999;

// ── Types ──────────────────────────────────────────────────────────────
pub const Pos = struct {
    x: i16,
    y: i16,
    pub fn eql(self: Pos, other: Pos) bool {
        return self.x == other.x and self.y == other.y;
    }
};
pub const Dir = enum { up, down, left, right };
pub const Cell = enum(u8) { floor, wall, shelf, dropoff };

pub const Bot = struct {
    id: u8,
    pos: Pos,
    inv: [INV_CAP]ItemType,
    inv_len: u8,
};

pub const ItemType = struct {
    buf: [32]u8,
    len: u8,
    pub fn eql(self: ItemType, other: ItemType) bool {
        return self.len == other.len and std.mem.eql(u8, self.buf[0..self.len], other.buf[0..other.len]);
    }
    pub fn fromStr(s: []const u8) ItemType {
        var it = ItemType{ .buf = undefined, .len = @intCast(@min(s.len, 32)) };
        @memcpy(it.buf[0..it.len], s[0..it.len]);
        return it;
    }
    pub fn str(self: *const ItemType) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const MapItem = struct {
    id_buf: [32]u8,
    id_len: u8,
    item_type: ItemType,
    pos: Pos,
    pub fn idStr(self: *const MapItem) []const u8 {
        return self.id_buf[0..self.id_len];
    }
};

pub const Order = struct {
    required: [16]ItemType,
    required_len: u8,
    delivered: [16]ItemType,
    delivered_len: u8,
    is_active: bool,
    complete: bool,
};

pub const GameState = struct {
    round: u32,
    max_rounds: u32,
    width: u16,
    height: u16,
    grid: [MAX_H][MAX_W]Cell,
    known_shelves: [MAX_H][MAX_W]bool,
    bots: [MAX_BOTS]Bot,
    bot_count: u8,
    items: [MAX_ITEMS]MapItem,
    item_count: u16,
    orders: [MAX_ORDERS]Order,
    order_count: u8,
    dropoff: Pos,
    score: i32,
    active_order_idx: i32,
};

// ── MAPF Constants ────────────────────────────────────────────────────
pub const MAX_TIME_HORIZON: u16 = 12; // WHCA* planning window
pub const WRONG_WAY_PENALTY: u16 = 2; // Soft cost for going against aisle flow

pub const AisleDir = enum { down, up }; // Preferred vertical direction in an aisle column

pub const STResult = struct {
    first_dir: ?Dir, // Direction for this round (null = wait)
    path_len: u16, // Total path length found
};

// ── Distance Map ───────────────────────────────────────────────────────
pub const DistMap = [MAX_H][MAX_W]u16;

// ── Need list ──────────────────────────────────────────────────────────
pub const NeedList = struct {
    types: [16]ItemType,
    count: u8,
    pub fn init() NeedList { return .{ .types = undefined, .count = 0 }; }
    pub fn contains(self: *const NeedList, t: ItemType) bool {
        for (0..self.count) |i| if (self.types[i].eql(t)) return true;
        return false;
    }
    pub fn remove(self: *NeedList, t: ItemType) void {
        for (0..self.count) |i| {
            if (self.types[i].eql(t)) {
                self.types[i] = self.types[self.count - 1];
                self.count -= 1;
                return;
            }
        }
    }
    pub fn add(self: *NeedList, t: ItemType) void {
        if (self.count < 16) { self.types[self.count] = t; self.count += 1; }
    }
};

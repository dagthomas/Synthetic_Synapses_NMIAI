const std = @import("std");
const types = @import("types.zig");
const strategy = @import("strategy.zig");
const pathfinding = @import("pathfinding.zig");

const Pos = types.Pos;
const Cell = types.Cell;
const Bot = types.Bot;
const ItemType = types.ItemType;
const MapItem = types.MapItem;
const Order = types.Order;
const GameState = types.GameState;
const MAX_H = types.MAX_H;
const MAX_W = types.MAX_W;
const MAX_BOTS = types.MAX_BOTS;
const MAX_ITEMS = types.MAX_ITEMS;
const MAX_ORDERS = types.MAX_ORDERS;
const INV_CAP = types.INV_CAP;

const MAX_ROUNDS = 300;
const MAX_ALL_ORDERS = 64;

const ALL_TYPES = [16][]const u8{
    "milk", "bread", "eggs", "butter", "cheese", "pasta", "rice", "juice",
    "yogurt", "cereal", "flour", "sugar", "coffee", "tea", "oil", "salt",
};

pub const DiffConfig = struct {
    w: u16,
    h: u16,
    bots: u8,
    aisles: u8,
    type_count: u8,
    order_min: u8,
    order_max: u8,
};

pub const CONFIGS = struct {
    pub const easy = DiffConfig{ .w = 12, .h = 10, .bots = 1, .aisles = 2, .type_count = 4, .order_min = 3, .order_max = 4 };
    pub const medium = DiffConfig{ .w = 16, .h = 12, .bots = 3, .aisles = 3, .type_count = 8, .order_min = 3, .order_max = 5 };
    pub const hard = DiffConfig{ .w = 22, .h = 14, .bots = 5, .aisles = 4, .type_count = 12, .order_min = 3, .order_max = 5 };
    pub const expert = DiffConfig{ .w = 28, .h = 18, .bots = 10, .aisles = 5, .type_count = 16, .order_min = 4, .order_max = 6 };
};

// ── Sim Game State ─────────────────────────────────────────────────────
pub const SimOrder = struct {
    required: [8]ItemType,
    required_len: u8,
    delivered: [8]ItemType,
    delivered_len: u8,
    complete: bool,
    is_active: bool, // true for active, false for preview or completed
};

pub const SimGame = struct {
    cfg: DiffConfig,
    width: u16,
    height: u16,
    grid: [MAX_H][MAX_W]Cell,
    // Items
    items: [MAX_ITEMS]MapItem,
    item_count: u16,
    // Bots
    bot_pos: [MAX_BOTS][2]i16, // [x, y]
    bot_inv: [MAX_BOTS][INV_CAP]ItemType,
    bot_inv_len: [MAX_BOTS]u8,
    bot_count: u8,
    // Orders
    all_orders: [MAX_ALL_ORDERS]SimOrder,
    order_count: u16,
    next_order_idx: u16,
    active_idx: u16,
    // Map info
    dropoff: [2]i16,
    spawn: [2]i16,
    item_types: [16]ItemType,
    type_count: u8,
    // Score
    score: i32,
    items_delivered: u32,
    orders_completed: u32,
    // PRNG - Mersenne Twister for Python compatibility
    mt: MersenneTwister,

    // ── Mersenne Twister (Python-compatible) ──
    const MT_N = 624;
    const MT_M = 397;

    const MersenneTwister = struct {
        mt: [MT_N]u32,
        index: u32,

        fn init(seed: u32) MersenneTwister {
            var self = MersenneTwister{ .mt = undefined, .index = MT_N };
            self.mt[0] = seed;
            for (1..MT_N) |i| {
                self.mt[i] = 1812433253 *% (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) +% @as(u32, @intCast(i));
            }
            return self;
        }

        fn twist(self: *MersenneTwister) void {
            for (0..MT_N) |i| {
                const y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % MT_N] & 0x7fffffff);
                self.mt[i] = self.mt[(i + MT_M) % MT_N] ^ (y >> 1);
                if (y % 2 != 0) {
                    self.mt[i] ^= 0x9908b0df;
                }
            }
            self.index = 0;
        }

        fn next_u32(self: *MersenneTwister) u32 {
            if (self.index >= MT_N) self.twist();
            var y = self.mt[self.index];
            self.index += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >> 18;
            return y;
        }

        /// Python-compatible random.randbelow(n) — generates uniform [0, n)
        fn randbelow(self: *MersenneTwister, n: u32) u32 {
            if (n <= 1) return 0;
            // Python uses rejection sampling with bit masking
            var k: u32 = 0;
            var nn = n - 1;
            while (nn > 0) : (nn >>= 1) k += 1;
            // k = number of bits needed
            const mask: u32 = (@as(u32, 1) << @intCast(k)) - 1;
            while (true) {
                const r = self.next_u32() >> @intCast(32 - k);
                const v = r & mask;
                if (v < n) return v;
            }
        }

        /// Python-compatible random.randint(a, b) — inclusive both ends
        fn randint(self: *MersenneTwister, a: u32, b: u32) u32 {
            return a + self.randbelow(b - a + 1);
        }

        /// Python-compatible random.choice(list of length n)
        fn choice(self: *MersenneTwister, n: u32) u32 {
            return self.randbelow(n);
        }
    };

    // ── Map Building ────────────────────────────────────────────────────
    pub fn init(cfg: DiffConfig, seed: u32) SimGame {
        var game = SimGame{
            .cfg = cfg,
            .width = cfg.w,
            .height = cfg.h,
            .grid = undefined,
            .items = undefined,
            .item_count = 0,
            .bot_pos = undefined,
            .bot_inv = undefined,
            .bot_inv_len = undefined,
            .bot_count = cfg.bots,
            .all_orders = undefined,
            .order_count = 0,
            .next_order_idx = 2,
            .active_idx = 0,
            .dropoff = undefined,
            .spawn = undefined,
            .item_types = undefined,
            .type_count = cfg.type_count,
            .score = 0,
            .items_delivered = 0,
            .orders_completed = 0,
            .mt = MersenneTwister.init(seed),
        };

        // Init grid to floor
        for (0..MAX_H) |y| {
            for (0..MAX_W) |x| {
                game.grid[y][x] = .floor;
            }
        }

        const w = cfg.w;
        const h = cfg.h;

        // Border walls
        for (0..w) |x| {
            game.grid[0][x] = .wall;
            game.grid[h - 1][x] = .wall;
        }
        for (0..h) |y| {
            game.grid[y][0] = .wall;
            game.grid[y][w - 1] = .wall;
        }

        // Aisle layout: shelf-walkway-shelf with walls, starting at x=3, spaced by 4
        const mid_y = h / 2;

        var aisle_x: u16 = 3;
        for (0..cfg.aisles) |_| {
            // Shelf rows: top section and bottom section
            for (2..mid_y) |y| {
                game.grid[y][aisle_x -| 1] = .wall; // left aisle wall
                if (aisle_x + 3 < MAX_W) game.grid[y][aisle_x + 3] = .wall; // right aisle wall
                game.grid[y][aisle_x] = .shelf; // left shelf
                if (aisle_x + 2 < MAX_W) game.grid[y][aisle_x + 2] = .shelf; // right shelf
            }
            for (mid_y + 1..h - 2) |y| {
                game.grid[y][aisle_x -| 1] = .wall;
                if (aisle_x + 3 < MAX_W) game.grid[y][aisle_x + 3] = .wall;
                game.grid[y][aisle_x] = .shelf;
                if (aisle_x + 2 < MAX_W) game.grid[y][aisle_x + 2] = .shelf;
            }
            aisle_x += 4;
        }

        // Dropoff and spawn
        game.dropoff = .{ 1, @intCast(h - 2) };
        game.spawn = .{ @intCast(w - 2), @intCast(h - 2) };
        game.grid[@intCast(game.dropoff[1])][@intCast(game.dropoff[0])] = .dropoff;

        // Item types
        for (0..cfg.type_count) |i| {
            game.item_types[i] = ItemType.fromStr(ALL_TYPES[i]);
        }

        // Place items on shelves (sorted by (x,y))
        // Collect shelf positions sorted
        var shelf_positions: [MAX_ITEMS][2]u16 = undefined;
        var shelf_count: u16 = 0;
        for (0..h) |y| {
            for (0..w) |x| {
                if (game.grid[y][x] == .shelf) {
                    if (shelf_count < MAX_ITEMS) {
                        shelf_positions[shelf_count] = .{ @intCast(x), @intCast(y) };
                        shelf_count += 1;
                    }
                }
            }
        }
        // Python sorts by (x, y) — shelves is a set converted to sorted list
        // We collected by (y, x), need to sort by (x, y)
        sortShelfPositions(shelf_positions[0..shelf_count]);

        game.item_count = shelf_count;
        for (0..shelf_count) |i| {
            const sx = shelf_positions[i][0];
            const sy = shelf_positions[i][1];
            const itype = game.item_types[i % cfg.type_count];
            game.items[i] = .{
                .id_buf = undefined,
                .id_len = 0,
                .item_type = itype,
                .pos = .{ .x = @intCast(sx), .y = @intCast(sy) },
            };
            // Write item_id: "item_N"
            const id_str = std.fmt.bufPrint(&game.items[i].id_buf, "item_{d}", .{i}) catch "item_?";
            game.items[i].id_len = @intCast(id_str.len);
        }

        // Init bots at spawn
        for (0..cfg.bots) |i| {
            game.bot_pos[i] = .{ game.spawn[0], game.spawn[1] };
            game.bot_inv_len[i] = 0;
        }

        // Generate initial orders
        game.all_orders[0] = game.generateOrder(true);
        game.all_orders[1] = game.generateOrder(false);
        game.order_count = 2;

        return game;
    }

    /// Initialize game state from live capture data instead of seed.
    /// grid_bytes: [height * width] u8, values match Cell enum (0=floor,1=wall,2=shelf,3=dropoff)
    /// item_x/item_y: [num_items] u8 — sorted by (x,y) to match Python indexing
    /// item_type_id: [num_items] u8 — Python type IDs (0..num_types-1)
    /// All bots start at spawn with empty inventories. Orders must be overridden via overrideOrders().
    pub fn initFromCapture(
        width: u16,
        height: u16,
        dropoff_x: u16,
        dropoff_y: u16,
        grid_bytes: [*]const u8,
        num_items: u16,
        item_x: [*]const u8,
        item_y: [*]const u8,
        item_type_id: [*]const u8,
        num_types: u8,
        num_bots: u8,
    ) SimGame {
        var game = SimGame{
            .cfg = DiffConfig{ .w = @intCast(width), .h = @intCast(height), .bots = num_bots,
                               .aisles = 0, .type_count = num_types, .order_min = 3, .order_max = 5 },
            .width = width,
            .height = height,
            .grid = undefined,
            .items = undefined,
            .item_count = 0,
            .bot_pos = undefined,
            .bot_inv = undefined,
            .bot_inv_len = undefined,
            .bot_count = num_bots,
            .all_orders = undefined,
            .order_count = 0,
            .next_order_idx = 2,
            .active_idx = 0,
            .dropoff = .{ @intCast(dropoff_x), @intCast(dropoff_y) },
            .spawn = .{ @intCast(width - 2), @intCast(height - 2) },
            .item_types = undefined,
            .type_count = num_types,
            .score = 0,
            .items_delivered = 0,
            .orders_completed = 0,
            .mt = MersenneTwister.init(0),
        };

        // Init grid from byte array (0=floor,1=wall,2=shelf,3=dropoff)
        for (0..MAX_H) |y| {
            for (0..MAX_W) |x| {
                game.grid[y][x] = .floor;
            }
        }
        for (0..height) |y| {
            for (0..width) |x| {
                const code = grid_bytes[y * width + x];
                game.grid[y][x] = switch (code) {
                    1 => .wall,
                    2 => .shelf,
                    3 => .dropoff,
                    else => .floor,
                };
            }
        }

        // Item types: use ALL_TYPES names for distinctness (indices match Python type IDs)
        for (0..num_types) |i| {
            game.item_types[i] = ItemType.fromStr(ALL_TYPES[i % ALL_TYPES.len]);
        }
        for (num_types..16) |i| {
            game.item_types[i] = ItemType.fromStr(ALL_TYPES[i % ALL_TYPES.len]);
        }

        // Items (already sorted by (x,y) from Python)
        game.item_count = num_items;
        for (0..num_items) |i| {
            const tid: usize = @as(usize, item_type_id[i]) % num_types;
            game.items[i] = .{
                .id_buf = undefined,
                .id_len = 0,
                .item_type = game.item_types[tid],
                .pos = .{ .x = @intCast(item_x[i]), .y = @intCast(item_y[i]) },
            };
            const id_str = std.fmt.bufPrint(&game.items[i].id_buf, "item_{d}", .{i}) catch "item_?";
            game.items[i].id_len = @intCast(id_str.len);
        }

        // Bots at spawn with empty inventories
        for (0..num_bots) |i| {
            game.bot_pos[i] = .{ game.spawn[0], game.spawn[1] };
            game.bot_inv_len[i] = 0;
        }

        return game;
    }

    fn sortShelfPositions(positions: [][2]u16) void {
        // Insertion sort by (x, y)
        for (1..positions.len) |i| {
            const key = positions[i];
            var j: usize = i;
            while (j > 0) {
                const prev = positions[j - 1];
                if (prev[0] > key[0] or (prev[0] == key[0] and prev[1] > key[1])) {
                    positions[j] = positions[j - 1];
                    j -= 1;
                } else break;
            }
            positions[j] = key;
        }
    }

    fn generateOrder(self: *SimGame, is_active: bool) SimOrder {
        var order = SimOrder{
            .required = undefined,
            .required_len = 0,
            .delivered = undefined,
            .delivered_len = 0,
            .complete = false,
            .is_active = is_active,
        };

        const n = self.mt.randint(self.cfg.order_min, self.cfg.order_max);

        // Build available counts
        var avail_types: [16]u16 = .{0} ** 16;
        for (0..self.item_count) |i| {
            for (0..self.type_count) |ti| {
                if (self.item_types[ti].eql(self.items[i].item_type)) {
                    avail_types[ti] += 1;
                    break;
                }
            }
        }

        var temp_counts: [16]u16 = avail_types;
        var avail_count: u32 = 0;
        for (0..self.type_count) |ti| {
            if (avail_types[ti] > 0) avail_count += 1;
        }

        for (0..n) |_| {
            // Build usable list (types with remaining count)
            var usable: [16]u8 = undefined;
            var usable_count: u32 = 0;
            for (0..self.type_count) |ti| {
                if (temp_counts[ti] > 0) {
                    usable[usable_count] = @intCast(ti);
                    usable_count += 1;
                }
            }
            if (usable_count == 0) {
                // Fallback: use any available type
                for (0..self.type_count) |ti| {
                    if (avail_types[ti] > 0) {
                        usable[usable_count] = @intCast(ti);
                        usable_count += 1;
                    }
                }
            }
            if (usable_count == 0) usable_count = 1; // Safety

            const choice_idx = self.mt.choice(usable_count);
            const type_idx = usable[choice_idx];
            if (order.required_len < 8) {
                order.required[order.required_len] = self.item_types[type_idx];
                order.required_len += 1;
            }
            if (temp_counts[type_idx] > 0) temp_counts[type_idx] -= 1;
        }

        return order;
    }

    // ── Build GameState for Strategy ──────────────────────────────────────
    pub fn buildGameState(self: *const SimGame, round: u32, gs: *GameState) void {
        gs.round = round;
        gs.max_rounds = MAX_ROUNDS;
        gs.width = self.width;
        gs.height = self.height;
        gs.score = self.score;
        gs.active_order_idx = @intCast(self.active_idx);

        // Grid
        for (0..self.height) |y| {
            for (0..self.width) |x| {
                gs.grid[y][x] = self.grid[y][x];
            }
        }

        // Known shelves (set shelf cells)
        if (round == 0) {
            for (0..MAX_H) |y| @memset(&gs.known_shelves[y], false);
        }
        for (0..self.item_count) |i| {
            const ix: u16 = @intCast(self.items[i].pos.x);
            const iy: u16 = @intCast(self.items[i].pos.y);
            gs.known_shelves[iy][ix] = true;
        }
        // Mark shelf cells in grid
        for (0..self.height) |y| {
            for (0..self.width) |x| {
                if (gs.known_shelves[y][x] and gs.grid[y][x] == .floor) gs.grid[y][x] = .shelf;
            }
        }

        // Items
        gs.item_count = self.item_count;
        for (0..self.item_count) |i| {
            gs.items[i] = self.items[i];
        }

        // Bots
        gs.bot_count = self.bot_count;
        for (0..self.bot_count) |i| {
            gs.bots[i] = .{
                .id = @intCast(i),
                .pos = .{ .x = self.bot_pos[i][0], .y = self.bot_pos[i][1] },
                .inv = undefined,
                .inv_len = self.bot_inv_len[i],
            };
            for (0..self.bot_inv_len[i]) |ii| {
                gs.bots[i].inv[ii] = self.bot_inv[i][ii];
            }
        }

        // Orders: visible = up to 2 non-complete orders
        gs.order_count = 0;
        for (0..self.order_count) |i| {
            if (self.all_orders[i].complete) continue;
            if (gs.order_count >= MAX_ORDERS) break;
            const so = &self.all_orders[i];
            gs.orders[gs.order_count] = .{
                .required = undefined,
                .required_len = so.required_len,
                .delivered = undefined,
                .delivered_len = so.delivered_len,
                .is_active = so.is_active,
                .complete = so.complete,
            };
            for (0..so.required_len) |ri| {
                gs.orders[gs.order_count].required[ri] = so.required[ri];
            }
            for (0..so.delivered_len) |di| {
                gs.orders[gs.order_count].delivered[di] = so.delivered[di];
            }
            gs.order_count += 1;
        }

        // Dropoff
        gs.dropoff = .{ .x = self.dropoff[0], .y = self.dropoff[1] };
    }

    // ── Process Actions (Array API) ──────────────────────────────────────
    /// Action record for direct array-based action processing (no JSON).
    pub const ActionRec = struct {
        bot: u8,
        /// 0=wait,1=up,2=down,3=left,4=right,5=pickup,6=dropoff
        action: u8,
        /// Index into self.items[], -1 = none
        item_idx: i16,
    };

    pub fn processActionsArray(self: *SimGame, actions: []const ActionRec) void {
        for (actions) |rec| {
            const bi: usize = rec.bot;
            if (bi >= self.bot_count) continue;
            const bx = self.bot_pos[bi][0];
            const by = self.bot_pos[bi][1];
            switch (rec.action) {
                1 => self.tryMove(bi, bx, by, 0, -1),   // up
                2 => self.tryMove(bi, bx, by, 0, 1),    // down
                3 => self.tryMove(bi, bx, by, -1, 0),   // left
                4 => self.tryMove(bi, bx, by, 1, 0),    // right
                5 => {
                    if (rec.item_idx >= 0) {
                        const idx: usize = @intCast(rec.item_idx);
                        if (idx < self.item_count) {
                            self.tryPickupByIdx(bi, bx, by, idx);
                        }
                    }
                },
                6 => self.tryDropoff(bi, bx, by),       // dropoff
                else => {},  // 0=wait or unknown
            }
        }
    }

    fn tryPickupByIdx(self: *SimGame, bi: usize, bx: i16, by: i16, item_idx: usize) void {
        if (self.bot_inv_len[bi] >= INV_CAP) return;
        const ix = self.items[item_idx].pos.x;
        const iy = self.items[item_idx].pos.y;
        const mdist = @as(i32, @abs(bx - ix)) + @as(i32, @abs(by - iy));
        if (mdist == 1) {
            const inv_idx = self.bot_inv_len[bi];
            self.bot_inv[bi][inv_idx] = self.items[item_idx].item_type;
            self.bot_inv_len[bi] += 1;
        }
    }

    // ── Process Actions (JSON) ──────────────────────────────────────────
    pub fn processActions(self: *SimGame, action_json: []const u8) void {
        // Parse action JSON: {"actions":[{"bot":0,"action":"move_up"},...]
        // Simple parsing without allocator
        var idx: usize = 0;
        while (idx < action_json.len) {
            // Find next "bot":
            const bot_start = std.mem.indexOfPos(u8, action_json, idx, "\"bot\":") orelse break;
            idx = bot_start + 6;
            // Parse bot id
            const bot_id = parseNum(action_json, &idx);
            if (bot_id >= self.bot_count) {
                idx += 1;
                continue;
            }

            // Find "action":"
            const act_start = std.mem.indexOfPos(u8, action_json, idx, "\"action\":\"") orelse break;
            idx = act_start + 10;
            // Read action string
            const act_end = std.mem.indexOfPos(u8, action_json, idx, "\"") orelse break;
            const action_str = action_json[idx..act_end];
            idx = act_end + 1;

            const bi: usize = bot_id;
            const bx = self.bot_pos[bi][0];
            const by = self.bot_pos[bi][1];

            if (std.mem.eql(u8, action_str, "move_up")) {
                self.tryMove(bi, bx, by, 0, -1);
            } else if (std.mem.eql(u8, action_str, "move_down")) {
                self.tryMove(bi, bx, by, 0, 1);
            } else if (std.mem.eql(u8, action_str, "move_left")) {
                self.tryMove(bi, bx, by, -1, 0);
            } else if (std.mem.eql(u8, action_str, "move_right")) {
                self.tryMove(bi, bx, by, 1, 0);
            } else if (std.mem.eql(u8, action_str, "pick_up")) {
                // Find item_id
                const iid_start = std.mem.indexOfPos(u8, action_json, idx, "\"item_id\":\"") orelse continue;
                idx = iid_start + 11;
                const iid_end = std.mem.indexOfPos(u8, action_json, idx, "\"") orelse continue;
                const item_id_str = action_json[idx..iid_end];
                idx = iid_end + 1;
                self.tryPickup(bi, bx, by, item_id_str);
            } else if (std.mem.eql(u8, action_str, "drop_off")) {
                self.tryDropoff(bi, bx, by);
            }
            // "wait" → do nothing
        }
    }

    fn parseNum(data: []const u8, idx: *usize) u8 {
        var n: u8 = 0;
        while (idx.* < data.len and data[idx.*] >= '0' and data[idx.*] <= '9') {
            n = n * 10 + (data[idx.*] - '0');
            idx.* += 1;
        }
        return n;
    }

    fn tryMove(self: *SimGame, bi: usize, bx: i16, by: i16, dx: i16, dy: i16) void {
        const nx = bx + dx;
        const ny = by + dy;
        if (nx < 0 or ny < 0 or nx >= self.width or ny >= self.height) return;
        const cell = self.grid[@intCast(ny)][@intCast(nx)];
        if (cell == .wall or cell == .shelf) return;

        // Bot collision (spawn exempt)
        const is_spawn = (nx == self.spawn[0] and ny == self.spawn[1]);
        if (!is_spawn) {
            for (0..self.bot_count) |oi| {
                if (oi == bi) continue;
                if (self.bot_pos[oi][0] == nx and self.bot_pos[oi][1] == ny) return; // blocked
            }
        }

        self.bot_pos[bi] = .{ nx, ny };
    }

    fn tryPickup(self: *SimGame, bi: usize, bx: i16, by: i16, item_id_str: []const u8) void {
        if (self.bot_inv_len[bi] >= INV_CAP) return;

        for (0..self.item_count) |i| {
            if (!std.mem.eql(u8, self.items[i].id_buf[0..self.items[i].id_len], item_id_str)) continue;
            const ix = self.items[i].pos.x;
            const iy = self.items[i].pos.y;
            const mdist = @as(i32, @abs(bx - ix)) + @as(i32, @abs(by - iy));
            if (mdist == 1) {
                const inv_idx = self.bot_inv_len[bi];
                self.bot_inv[bi][inv_idx] = self.items[i].item_type;
                self.bot_inv_len[bi] += 1;
            }
            break;
        }
    }

    fn tryDropoff(self: *SimGame, bi: usize, bx: i16, by: i16) void {
        if (bx != self.dropoff[0] or by != self.dropoff[1]) return;
        if (self.bot_inv_len[bi] == 0) return;

        // Find active order
        var active_oi: ?usize = null;
        for (0..self.order_count) |i| {
            if (!self.all_orders[i].complete and self.all_orders[i].is_active) {
                active_oi = i;
                break;
            }
        }
        const oi = active_oi orelse return;

        // Deliver matching items
        var remaining: [INV_CAP]ItemType = undefined;
        var remaining_len: u8 = 0;

        for (0..self.bot_inv_len[bi]) |ii| {
            const inv_item = self.bot_inv[bi][ii];
            if (self.isNeeded(oi, inv_item)) {
                // Deliver
                const o = &self.all_orders[oi];
                if (o.delivered_len < 8) {
                    o.delivered[o.delivered_len] = inv_item;
                    o.delivered_len += 1;
                }
                self.score += 1;
                self.items_delivered += 1;
            } else {
                if (remaining_len < INV_CAP) {
                    remaining[remaining_len] = inv_item;
                    remaining_len += 1;
                }
            }
        }
        self.bot_inv[bi] = remaining;
        self.bot_inv_len[bi] = remaining_len;

        // Check if order complete
        if (self.isOrderComplete(oi)) {
            self.all_orders[oi].complete = true;
            self.score += 5;
            self.orders_completed += 1;

            // Activate preview
            for (0..self.order_count) |i| {
                if (!self.all_orders[i].complete and !self.all_orders[i].is_active) {
                    self.all_orders[i].is_active = true;
                    break;
                }
            }

            // Generate new preview
            if (self.order_count < MAX_ALL_ORDERS) {
                self.all_orders[self.order_count] = self.generateOrder(false);
                self.order_count += 1;
                self.next_order_idx += 1;
            }

            // Auto-delivery for all bots at dropoff
            const new_active_oi = self.findActiveOrder();
            if (new_active_oi) |noi| {
                for (0..self.bot_count) |b2| {
                    if (self.bot_pos[b2][0] == self.dropoff[0] and self.bot_pos[b2][1] == self.dropoff[1]) {
                        self.autoDeliver(b2, noi);
                    }
                }
            }
        }
    }

    fn isNeeded(self: *const SimGame, oi: usize, item: ItemType) bool {
        const o = &self.all_orders[oi];
        // Count how many of this type are required vs delivered
        var required_count: u8 = 0;
        var delivered_count: u8 = 0;
        for (0..o.required_len) |i| {
            if (o.required[i].eql(item)) required_count += 1;
        }
        for (0..o.delivered_len) |i| {
            if (o.delivered[i].eql(item)) delivered_count += 1;
        }
        return delivered_count < required_count;
    }

    fn isOrderComplete(self: *const SimGame, oi: usize) bool {
        const o = &self.all_orders[oi];
        // For each required item, check it's delivered
        var needed = [_]bool{false} ** 8;
        for (0..o.required_len) |i| needed[i] = true;

        for (0..o.delivered_len) |di| {
            for (0..o.required_len) |ri| {
                if (needed[ri] and o.required[ri].eql(o.delivered[di])) {
                    needed[ri] = false;
                    break;
                }
            }
        }
        for (0..o.required_len) |i| {
            if (needed[i]) return false;
        }
        return true;
    }

    fn findActiveOrder(self: *const SimGame) ?usize {
        for (0..self.order_count) |i| {
            if (!self.all_orders[i].complete and self.all_orders[i].is_active) return i;
        }
        return null;
    }

    fn autoDeliver(self: *SimGame, bi: usize, oi: usize) void {
        var remaining: [INV_CAP]ItemType = undefined;
        var remaining_len: u8 = 0;

        for (0..self.bot_inv_len[bi]) |ii| {
            const inv_item = self.bot_inv[bi][ii];
            if (self.isNeeded(oi, inv_item)) {
                const o = &self.all_orders[oi];
                if (o.delivered_len < 8) {
                    o.delivered[o.delivered_len] = inv_item;
                    o.delivered_len += 1;
                }
                self.score += 1;
                self.items_delivered += 1;
            } else {
                if (remaining_len < INV_CAP) {
                    remaining[remaining_len] = inv_item;
                    remaining_len += 1;
                }
            }
        }
        self.bot_inv[bi] = remaining;
        self.bot_inv_len[bi] = remaining_len;
    }

    // ── Run Full Game ────────────────────────────────────────────────────
    pub fn run(self: *SimGame) i32 {
        var gs: GameState = undefined;
        gs.bot_count = 0;
        var action_buf: [8192]u8 = undefined;

        strategy.initPbots();
        strategy.expected_count = 0;
        pathfinding.resetPrecompute();

        for (0..MAX_ROUNDS) |rnd| {
            self.buildGameState(@intCast(rnd), &gs);

            const response = strategy.decideActions(&gs, &action_buf) catch {
                continue;
            };

            self.processActions(response);
        }

        return self.score;
    }
};

// ── Public API ────────────────────────────────────────────────────────
pub fn runSeed(cfg: DiffConfig, seed: u32) i32 {
    var game = SimGame.init(cfg, seed);
    return game.run();
}

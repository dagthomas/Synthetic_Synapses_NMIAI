/// FFI exports for Python ctypes integration.
/// Provides fast C-callable replacements for cpu_verify and pre_simulate_locked.
///
/// Action codes match Python game_engine.py:
///   0=wait, 1=up, 2=down, 3=left, 4=right, 5=pickup, 6=dropoff
///
/// Item indices are into sim.items[] (physical shelf items), -1 = none.
/// Orders are passed as flat type-index arrays (uint8, 0-15).
const std = @import("std");
const sim = @import("sim.zig");

const SimGame = sim.SimGame;
const ActionRec = SimGame.ActionRec;
const MAX_ROUNDS = 300;
const MAX_ALL_ORDERS: u16 = 64;
const MAX_BOTS: u8 = 16;

fn getDiffConfig(difficulty: u8) sim.DiffConfig {
    return switch (difficulty) {
        0 => sim.CONFIGS.easy,
        1 => sim.CONFIGS.medium,
        2 => sim.CONFIGS.hard,
        3 => sim.CONFIGS.expert,
        else => sim.CONFIGS.easy,
    };
}

/// Override the game's randomly-generated orders with the provided order list.
/// order_types: flat concatenation of type indices for each order (0-15)
/// order_lens: number of items in each order [num_orders]
fn overrideOrders(
    game: *SimGame,
    order_types: [*]const u8,
    order_lens: [*]const u8,
    num_orders: u16,
) void {
    if (num_orders == 0) return;
    game.order_count = 0;
    var type_offset: usize = 0;
    for (0..num_orders) |i| {
        if (i >= MAX_ALL_ORDERS) break;
        const len = order_lens[i];
        var order = sim.SimOrder{
            .required = undefined,
            .required_len = len,
            .delivered = undefined,
            .delivered_len = 0,
            .complete = false,
            .is_active = (i == 0),
        };
        for (0..len) |j| {
            if (j >= 8) break;
            const type_idx: usize = order_types[type_offset + j] % game.type_count;
            order.required[j] = game.item_types[type_idx];
        }
        for (len..8) |j| {
            order.required[j] = game.item_types[0]; // placeholder for unused slots
        }
        type_offset += len;
        game.all_orders[i] = order;
        game.order_count += 1;
    }
    game.active_idx = 0;
    game.next_order_idx = if (game.order_count >= 2) 2 else game.order_count;
}

/// Build the per-round action array from flat inputs.
fn fillActions(
    buf: []ActionRec,
    actions: [*]const i8,
    action_items: [*]const i16,
    num_bots: u8,
    round: u32,
) u8 {
    const base = round * num_bots;
    var n: u8 = 0;
    for (0..num_bots) |b| {
        const raw_act = actions[base + b];
        const act: u8 = if (raw_act >= 0) @intCast(raw_act) else 0;
        buf[n] = .{
            .bot = @intCast(b),
            .action = act,
            .item_idx = action_items[base + b],
        };
        n += 1;
    }
    return n;
}

/// Verify a full 300-round game with injected actions and order list.
/// Returns final score.
/// actions: [300 * num_bots] action codes (int8)
/// action_items: [300 * num_bots] item indices (int16, -1 = none)
/// order_types: flat concat of type indices per order
/// order_lens: [num_orders] items per order
/// If num_orders == 0, uses seed-generated orders.
export fn ffi_verify(
    difficulty: u8,
    seed: u32,
    actions: [*]const i8,
    action_items: [*]const i16,
    num_bots: u8,
    order_types: [*]const u8,
    order_lens: [*]const u8,
    num_orders: u16,
) i32 {
    const cfg = getDiffConfig(difficulty);
    var game = SimGame.init(cfg, seed);

    if (num_orders > 0) {
        overrideOrders(&game, order_types, order_lens, num_orders);
    }

    var action_buf: [MAX_BOTS]ActionRec = undefined;

    for (0..MAX_ROUNDS) |r| {
        const na = fillActions(&action_buf, actions, action_items, num_bots, @intCast(r));
        game.processActionsArray(action_buf[0..na]);
    }

    return game.score;
}

/// Simulate all bots and record locked bot positions each round.
/// all_actions: [num_total_bots * 300] action codes
/// all_action_items: [num_total_bots * 300] item indices
/// locked_bot_ids: [num_locked] bot IDs to record positions for
/// out_pos_x/y: [num_locked * 300] output (index = locked_idx * 300 + round)
export fn ffi_presim_locked(
    difficulty: u8,
    seed: u32,
    all_actions: [*]const i8,
    all_action_items: [*]const i16,
    num_total_bots: u8,
    locked_bot_ids: [*]const u8,
    num_locked: u8,
    order_types: [*]const u8,
    order_lens: [*]const u8,
    num_orders: u16,
    out_pos_x: [*]i16,
    out_pos_y: [*]i16,
) void {
    const cfg = getDiffConfig(difficulty);
    var game = SimGame.init(cfg, seed);

    if (num_orders > 0) {
        overrideOrders(&game, order_types, order_lens, num_orders);
    }

    var action_buf: [MAX_BOTS]ActionRec = undefined;

    for (0..MAX_ROUNDS) |r| {
        const na = fillActions(&action_buf, all_actions, all_action_items, num_total_bots, @intCast(r));
        game.processActionsArray(action_buf[0..na]);

        // Record locked bot positions after this round
        for (0..num_locked) |li| {
            const bid = locked_bot_ids[li];
            out_pos_x[li * MAX_ROUNDS + r] = game.bot_pos[bid][0];
            out_pos_y[li * MAX_ROUNDS + r] = game.bot_pos[bid][1];
        }
    }
}

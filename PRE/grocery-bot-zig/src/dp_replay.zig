const std = @import("std");
const types = @import("types.zig");
const Pos = types.Pos;
const GameState = types.GameState;
const MAX_BOTS = types.MAX_BOTS;

// ── DP Plan Replay ─────────────────────────────────────────────────────
// Loads a pre-computed GPU DP action plan (dp_plan.json) and replays it.
// When bot positions match expected → send cached DP action (fast path).
// When desynced → return null, caller falls back to reactive strategy.

const MAX_ROUNDS = 300;
const MAX_ACTION_LEN = 128; // Max bytes per bot action JSON fragment

const BotAction = struct {
    json: [MAX_ACTION_LEN]u8,
    json_len: u16,
};

const RoundPlan = struct {
    expected_pos: [MAX_BOTS]Pos,
    actions: [MAX_BOTS]BotAction,
    num_bots: u8,
};

var plan: [MAX_ROUNDS]RoundPlan = undefined;
var plan_rounds: u16 = 0;
var plan_bots: u8 = 0;
var is_loaded: bool = false;

// Pre-built full response strings for fast-path sending
var cached_responses: [MAX_ROUNDS][4096]u8 = undefined;
var cached_response_lens: [MAX_ROUNDS]u16 = undefined;

var file_buf: [128 * 1024]u8 = undefined;

/// Load DP plan from dp_plan.json.
/// Returns true on success.
pub fn load(path: []const u8) bool {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("DP replay: cannot open {s}: {any}\n", .{ path, err });
        return false;
    };
    defer file.close();

    const bytes_read = file.readAll(&file_buf) catch |err| {
        std.debug.print("DP replay: read error: {any}\n", .{err});
        return false;
    };
    if (bytes_read == 0) {
        std.debug.print("DP replay: empty file\n", .{});
        return false;
    }

    // Parse JSON manually — extract rounds array
    // Format: {"rounds":[{"positions":[[x,y],...],"actions":[{"action":"move_up","item_id":"item_5"},...]}, ...]}
    const data = file_buf[0..bytes_read];

    // Find "num_bots" field
    if (findInt(data, "\"num_bots\"")) |nb| {
        plan_bots = @intCast(nb);
    } else {
        std.debug.print("DP replay: no num_bots field\n", .{});
        return false;
    }

    // Parse rounds array
    const rounds_start = std.mem.indexOf(u8, data, "\"rounds\"") orelse {
        std.debug.print("DP replay: no rounds array\n", .{});
        return false;
    };
    // Find the opening '[' after "rounds":
    const arr_start = std.mem.indexOfScalarPos(u8, data, rounds_start + 8, '[') orelse return false;

    plan_rounds = 0;
    var pos = arr_start + 1;

    while (plan_rounds < MAX_ROUNDS and pos < data.len) {
        // Find next round object '{'
        const obj_start = std.mem.indexOfScalarPos(u8, data, pos, '{') orelse break;
        // Find matching '}'  — need to handle nested objects
        const obj_end = findMatchingBrace(data, obj_start) orelse break;
        const round_data = data[obj_start .. obj_end + 1];

        // Parse positions: "positions":[[x,y],[x,y],...]
        var rp = &plan[plan_rounds];
        rp.num_bots = plan_bots;

        if (std.mem.indexOf(u8, round_data, "\"positions\"")) |pos_key| {
            const pos_arr_start = std.mem.indexOfScalarPos(u8, round_data, pos_key + 11, '[') orelse {
                pos = obj_end + 1;
                continue;
            };
            // Parse each [x,y] pair
            var bi: u8 = 0;
            var pp = pos_arr_start + 1;
            while (bi < plan_bots and pp < round_data.len) {
                const inner_start = std.mem.indexOfScalarPos(u8, round_data, pp, '[') orelse break;
                const inner_end = std.mem.indexOfScalarPos(u8, round_data, inner_start + 1, ']') orelse break;
                // Parse "x, y" from between [ and ]
                const coords = round_data[inner_start + 1 .. inner_end];
                const comma = std.mem.indexOfScalar(u8, coords, ',') orelse break;
                const x = std.fmt.parseInt(i16, std.mem.trim(u8, coords[0..comma], " "), 10) catch break;
                const y = std.fmt.parseInt(i16, std.mem.trim(u8, coords[comma + 1 ..], " "), 10) catch break;
                rp.expected_pos[bi] = Pos{ .x = x, .y = y };
                bi += 1;
                pp = inner_end + 1;
            }
        }

        // Parse actions: "actions":[{"action":"move_up","item_id":"item_5"}, ...]
        if (std.mem.indexOf(u8, round_data, "\"actions\"")) |act_key| {
            const act_arr_start = std.mem.indexOfScalarPos(u8, round_data, act_key + 9, '[') orelse {
                pos = obj_end + 1;
                continue;
            };
            var bi: u8 = 0;
            var ap = act_arr_start + 1;
            while (bi < plan_bots and ap < round_data.len) {
                const act_obj_start = std.mem.indexOfScalarPos(u8, round_data, ap, '{') orelse break;
                const act_obj_end = findMatchingBrace(round_data, act_obj_start) orelse break;
                const act_data = round_data[act_obj_start .. act_obj_end + 1];

                // Extract action name
                const action_name = extractString(act_data, "\"action\"") orelse "wait";
                const item_id = extractString(act_data, "\"item_id\"");

                // Build JSON fragment: {"bot":N,"action":"xxx"} or {"bot":N,"action":"pick_up","item_id":"xxx"}
                var ba = &rp.actions[bi];
                var fbs = std.io.fixedBufferStream(&ba.json);
                const w = fbs.writer();
                if (item_id) |iid| {
                    w.print("{{\"bot\":{d},\"action\":\"{s}\",\"item_id\":\"{s}\"}}", .{ bi, action_name, iid }) catch {};
                } else {
                    w.print("{{\"bot\":{d},\"action\":\"{s}\"}}", .{ bi, action_name }) catch {};
                }
                ba.json_len = @intCast(fbs.pos);

                bi += 1;
                ap = act_obj_end + 1;
            }
        }

        // Build cached full response for this round
        var resp_stream = std.io.fixedBufferStream(&cached_responses[plan_rounds]);
        const rw = resp_stream.writer();
        rw.writeAll("{\"actions\":[") catch {};
        for (0..plan_bots) |bi| {
            if (bi > 0) rw.writeAll(",") catch {};
            const ba = &rp.actions[bi];
            rw.writeAll(ba.json[0..ba.json_len]) catch {};
        }
        rw.writeAll("]}") catch {};
        cached_response_lens[plan_rounds] = @intCast(resp_stream.pos);

        plan_rounds += 1;
        pos = obj_end + 1;
    }

    is_loaded = plan_rounds > 0;
    plan_cursor = 0;
    consecutive_miss = 0;
    std.debug.print("DP replay: loaded {d} rounds for {d} bots from {s}\n", .{ plan_rounds, plan_bots, path });
    return is_loaded;
}

/// Check if all bots match expected positions for this round.
/// On mismatch: sends a wait (via returning false) and scans nearby plan
/// rounds next time to re-sync. This handles dropped/late packets gracefully.
var plan_cursor: u32 = 0; // which plan round we're currently at
var consecutive_miss: u16 = 0;
const MAX_MISS_BEFORE_ABANDON: u16 = 5;

pub fn isRoundSynced(state: *const GameState, round: u32) bool {
    if (!is_loaded) return false;
    if (state.bot_count != plan_bots) return false;

    // On round 0, always reset cursor
    if (round == 0) {
        plan_cursor = 0;
        consecutive_miss = 0;
    }

    // Try exact cursor position first
    if (plan_cursor < plan_rounds and positionsMatch(state, plan_cursor)) {
        consecutive_miss = 0;
        return true;
    }

    // Search nearby plan rounds (cursor-1 to cursor+3) for re-sync
    // This handles: dropped packets (cursor behind), or server lag (cursor ahead)
    const search_range: i32 = 4;
    var delta: i32 = -1;
    while (delta <= search_range) : (delta += 1) {
        if (delta == 0) continue; // already tried
        const try_cursor: i32 = @as(i32, @intCast(plan_cursor)) + delta;
        if (try_cursor >= 0 and try_cursor < plan_rounds) {
            const tc: u32 = @intCast(try_cursor);
            if (positionsMatch(state, tc)) {
                if (delta != 0) {
                    std.debug.print("DP replay: re-synced at R{d}, cursor {d} -> {d} (delta={d})\n", .{ round, plan_cursor, tc, delta });
                }
                plan_cursor = tc;
                consecutive_miss = 0;
                return true;
            }
        }
    }

    // No match in search window
    consecutive_miss += 1;
    if (consecutive_miss <= MAX_MISS_BEFORE_ABANDON) {
        // Don't advance cursor — wait for positions to catch up
        return false;
    }

    // Too many consecutive misses — plan is truly desynced
    return false;
}

fn positionsMatch(state: *const GameState, plan_round: u32) bool {
    const rp = &plan[plan_round];
    for (0..state.bot_count) |bi| {
        if (!state.bots[bi].pos.eql(rp.expected_pos[bi])) return false;
    }
    return true;
}

/// Get the cached DP response for the current cursor position.
/// Advances cursor for next round. Returns null if desynced.
pub fn getResponse(round: u32) ?[]const u8 {
    _ = round;
    if (!is_loaded or plan_cursor >= plan_rounds) return null;
    const resp = cached_responses[plan_cursor][0..cached_response_lens[plan_cursor]];
    plan_cursor += 1; // advance for next round
    return resp;
}

/// Check if plan is loaded.
pub fn isActive() bool {
    return is_loaded;
}

/// Get total rounds in plan.
pub fn getRoundCount() u16 {
    return plan_rounds;
}

// ── JSON parsing helpers ────────────────────────────────────────────────

fn findInt(data: []const u8, key: []const u8) ?i64 {
    const key_pos = std.mem.indexOf(u8, data, key) orelse return null;
    var p = key_pos + key.len;
    // Skip to ':'
    while (p < data.len and data[p] != ':') p += 1;
    p += 1; // skip ':'
    // Skip whitespace
    while (p < data.len and (data[p] == ' ' or data[p] == '\t' or data[p] == '\n' or data[p] == '\r')) p += 1;
    // Read integer
    var end = p;
    if (end < data.len and data[end] == '-') end += 1;
    while (end < data.len and data[end] >= '0' and data[end] <= '9') end += 1;
    return std.fmt.parseInt(i64, data[p..end], 10) catch null;
}

fn findMatchingBrace(data: []const u8, start: usize) ?usize {
    var depth: i32 = 0;
    var i = start;
    var in_string = false;
    while (i < data.len) : (i += 1) {
        if (data[i] == '"' and (i == 0 or data[i - 1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (data[i] == '{') depth += 1;
            if (data[i] == '}') {
                depth -= 1;
                if (depth == 0) return i;
            }
        }
    }
    return null;
}

fn extractString(data: []const u8, key: []const u8) ?[]const u8 {
    const key_pos = std.mem.indexOf(u8, data, key) orelse return null;
    var p = key_pos + key.len;
    // Skip to ':'
    while (p < data.len and data[p] != ':') p += 1;
    p += 1;
    // Skip whitespace
    while (p < data.len and (data[p] == ' ' or data[p] == '\t')) p += 1;
    // Read string value
    if (p >= data.len or data[p] != '"') return null;
    p += 1; // skip opening "
    const str_start = p;
    while (p < data.len and data[p] != '"') p += 1;
    if (p >= data.len) return null;
    return data[str_start..p];
}

const std = @import("std");
const ws = @import("ws.zig");
const types = @import("types.zig");
const strategy = @import("strategy.zig");
const parser = @import("parser.zig");

const GameState = types.GameState;
const Pos = types.Pos;
const MAX_BOTS = types.MAX_BOTS;

// ── Main ───────────────────────────────────────────────────────────────
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: grocery-bot <wss://game-dev.ainm.no/ws?token=...>\n", .{});
        return;
    }

    const url = args[1];
    var host: []const u8 = undefined;
    var path: []const u8 = undefined;
    var use_tls = false;
    var port: u16 = 80;

    if (std.mem.startsWith(u8, url, "wss://")) {
        use_tls = true;
        port = 443;
        const rest = url[6..];
        if (std.mem.indexOfScalar(u8, rest, '/')) |slash| { host = rest[0..slash]; path = rest[slash..]; } else { host = rest; path = "/"; }
    } else if (std.mem.startsWith(u8, url, "ws://")) {
        const rest = url[5..];
        if (std.mem.indexOfScalar(u8, rest, '/')) |slash| { host = rest[0..slash]; path = rest[slash..]; } else { host = rest; path = "/"; }
    } else {
        std.debug.print("Error: URL must start with ws:// or wss://\n", .{});
        return;
    }
    if (std.mem.indexOfScalar(u8, host, ':')) |colon| {
        port = try std.fmt.parseInt(u16, host[colon + 1 ..], 10);
        host = host[0..colon];
    }

    std.debug.print("Connecting to {s}:{d} (TLS={any})\n", .{ host, port, use_tls });
    var client = ws.WsClient.connect(allocator, host, port, path, use_tls) catch |err| {
        std.debug.print("Connection failed: {any}\n", .{err});
        return;
    };
    defer client.deinit();

    var state: GameState = undefined;
    state.bot_count = 0;
    var action_buf: [8192]u8 = undefined;

    // Shift detection state
    var consecutive_mismatches: u8 = 0;
    var skip_this_send: bool = false;
    var last_skip_round: i32 = -100; // Cooldown: don't skip again within 20 rounds

    const log_file = std.fs.cwd().createFile("game_log.jsonl", .{}) catch |err| {
        std.debug.print("Warning: Could not create game_log.jsonl: {any}\n", .{err});
        return;
    };
    defer log_file.close();

    while (true) {
        const data = client.recvMessage() catch |err| {
            std.debug.print("Recv error: {any}\n", .{err});
            break;
        };

        log_file.writeAll(data) catch {};
        log_file.writeAll("\n") catch {};

        const is_running = parser.parseGameState(data, &state) catch |err| {
            std.debug.print("Parse error: {any} (len={d})\n", .{ err, data.len });
            continue;
        };
        if (!is_running) break;

        // Initialize persistent state on first round
        if (state.round == 0) {
            strategy.initPbots();
            strategy.expected_count = 0;
            consecutive_mismatches = 0;
            skip_this_send = false;
        }

        // ── Shift detection: compare actual positions with expected ──
        if (state.round > 0 and strategy.expected_count > 0) {
            const check_count: u8 = @intCast(@min(state.bot_count, strategy.expected_count));
            var mismatches: u8 = 0;
            for (0..check_count) |bi| {
                if (!state.bots[bi].pos.eql(strategy.expected_next_pos[bi])) {
                    mismatches += 1;
                }
            }
            if (mismatches > 0) {
                consecutive_mismatches += 1;
                if (consecutive_mismatches <= 5 or consecutive_mismatches % 20 == 0) {
                    std.debug.print("R{d} Position mismatch: {d}/{d} bots wrong, streak={d}\n", .{
                        state.round, mismatches, check_count, consecutive_mismatches,
                    });
                    for (0..check_count) |bi| {
                        if (!state.bots[bi].pos.eql(strategy.expected_next_pos[bi])) {
                            std.debug.print("  Bot{d}: expected ({d},{d}) actual ({d},{d})\n", .{
                                state.bots[bi].id,
                                strategy.expected_next_pos[bi].x, strategy.expected_next_pos[bi].y,
                                state.bots[bi].pos.x, state.bots[bi].pos.y,
                            });
                        }
                    }
                }
                const round_i: i32 = @intCast(state.round);
                if (consecutive_mismatches >= 2 and !skip_this_send and (round_i - last_skip_round) > 20) {
                    std.debug.print("R{d} SHIFT DETECTED ({d} consecutive mismatches) - skipping send to re-sync\n", .{
                        state.round, consecutive_mismatches,
                    });
                    skip_this_send = true;
                    last_skip_round = round_i;
                    consecutive_mismatches = 0;
                }
            } else {
                if (consecutive_mismatches > 0) {
                    // Had mismatches but now matched — might be coincidental alignment under shift
                    // Only reset if we had a clean match
                    consecutive_mismatches = 0;
                }
            }
        }

        if (state.round % 50 == 0) std.debug.print("R{d}/{d} Score:{d}\n", .{ state.round, state.max_rounds, state.score });

        // ── If shift detected, skip this send to re-sync ──
        if (skip_this_send) {
            skip_this_send = false;
            log_file.writeAll("{\"skip\":true}\n") catch {};
            std.debug.print("R{d} SKIP: not sending response to break desync\n", .{state.round});
            // Still call decideActions to keep internal state updated, but don't send
            _ = strategy.decideActions(&state, &action_buf) catch {};
            // Clear expected_count AFTER decideActions (which sets it) so next round
            // doesn't false-trigger against stale expected positions
            strategy.expected_count = 0;
            continue; // Don't send — server will timeout and use no action
        }

        const response = strategy.decideActions(&state, &action_buf) catch |err| {
            std.debug.print("Decision error: {any}\n", .{err});
            continue;
        };

        log_file.writeAll(response) catch {};
        log_file.writeAll("\n") catch {};

        client.sendMessage(response) catch |err| {
            std.debug.print("Send error: {any}\n", .{err});
            break;
        };
    }
}

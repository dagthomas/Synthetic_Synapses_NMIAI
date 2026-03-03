const std = @import("std");
const ws = @import("ws.zig");
const types = @import("types.zig");
const strategy = @import("strategy.zig");
const parser = @import("parser.zig");

const GameState = types.GameState;
const Pos = types.Pos;
const MAX_BOTS = types.MAX_BOTS;

// ── Replay Mode ───────────────────────────────────────────────────────
fn runReplay(log_path: []const u8) !void {
    std.debug.print("=== REPLAY MODE: {s} ===\n", .{log_path});

    const file = std.fs.cwd().openFile(log_path, .{}) catch |err| {
        std.debug.print("Cannot open {s}: {any}\n", .{ log_path, err });
        return;
    };
    defer file.close();

    // Read the whole file
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const file_data = file.readToEndAlloc(allocator, 100 * 1024 * 1024) catch |err| {
        std.debug.print("Cannot read file: {any}\n", .{err});
        return;
    };
    defer allocator.free(file_data);

    var state: GameState = undefined;
    state.bot_count = 0;
    var action_buf: [8192]u8 = undefined;

    var total_rounds: u32 = 0;
    var total_waits: u32 = 0;
    var total_moves: u32 = 0;
    var total_pickups: u32 = 0;
    var total_dropoffs: u32 = 0;
    var final_score: i32 = 0;
    var last_score: i32 = 0;
    var score_gain_rounds: u32 = 0;

    // Process line by line
    var line_iter = std.mem.splitScalar(u8, file_data, '\n');
    while (line_iter.next()) |line| {
        if (line.len == 0) continue;
        // Skip our action response lines
        if (std.mem.startsWith(u8, line, "{\"actions\":")) continue;

        const is_running = parser.parseGameState(line, &state) catch |err| {
            std.debug.print("Parse error: {any}\n", .{err});
            continue;
        };
        if (!is_running) break;

        final_score = state.score;
        if (state.score > last_score) {
            score_gain_rounds += 1;
            last_score = state.score;
        }
        total_rounds += 1;

        if (state.round == 0) {
            strategy.initPbots();
            strategy.expected_count = 0;
        }

        const response = strategy.decideActions(&state, &action_buf) catch |err| {
            std.debug.print("R{d} Decision error: {any}\n", .{ state.round, err });
            continue;
        };

        // Count actions in our response
        var resp_iter = std.mem.splitSequence(u8, response, "\"action\":\"");
        _ = resp_iter.next(); // skip prefix
        while (resp_iter.next()) |action_part| {
            if (std.mem.startsWith(u8, action_part, "wait")) {
                total_waits += 1;
            } else if (std.mem.startsWith(u8, action_part, "move_")) {
                total_moves += 1;
            } else if (std.mem.startsWith(u8, action_part, "pick_up")) {
                total_pickups += 1;
            } else if (std.mem.startsWith(u8, action_part, "drop_off")) {
                total_dropoffs += 1;
            }
        }
    }

    std.debug.print("\n=== REPLAY RESULTS ===\n", .{});
    std.debug.print("Log file score: {d}\n", .{final_score});
    std.debug.print("Rounds processed: {d}\n", .{total_rounds});
    std.debug.print("Rounds where score increased: {d}\n", .{score_gain_rounds});
    std.debug.print("Bot-actions: {d} moves, {d} pickups, {d} dropoffs, {d} waits\n", .{ total_moves, total_pickups, total_dropoffs, total_waits });
    if (total_rounds > 0) {
        const total_actions = total_moves + total_pickups + total_dropoffs + total_waits;
        const pct_useful = if (total_actions > 0) (total_moves + total_pickups + total_dropoffs) * 100 / total_actions else 0;
        std.debug.print("Useful action rate: {d}%\n", .{pct_useful});
    }
    std.debug.print("======================\n", .{});
}

// ── Live Mode ─────────────────────────────────────────────────────────
fn runLive(url: []const u8) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

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
    var last_responded_round: i32 = -1;
    var desync_counter: u8 = 0;

    // Create timestamped log file for each game
    var log_name_buf: [64]u8 = undefined;
    const epoch = std.time.timestamp();
    const log_name = std.fmt.bufPrint(&log_name_buf, "game_log_{d}.jsonl", .{epoch}) catch "game_log.jsonl";
    const log_file = std.fs.cwd().createFile(log_name, .{}) catch |err| {
        std.debug.print("Warning: Could not create {s}: {any}\n", .{ log_name, err });
        return;
    };
    defer log_file.close();
    std.debug.print("Logging to: {s}\n", .{log_name});

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
        }

        std.debug.print("R{d}/{d} Score:{d}\n", .{ state.round, state.max_rounds, state.score });

        // Detect round gap (timeout caused server to advance without our response)
        const current_round: i32 = @intCast(state.round);
        // Skip duplicate/echo messages for rounds already responded to
        if (current_round <= last_responded_round) continue;
        if (last_responded_round >= 0 and current_round > last_responded_round + 1) {
            std.debug.print("R{d} DESYNC: expected R{d}, skipping to re-sync\n", .{ state.round, last_responded_round + 1 });
            _ = strategy.decideActions(&state, &action_buf) catch {};
            last_responded_round = current_round;
            continue;
        }

        // Detect 1-round action offset via position mismatch
        // Skip this check if strategy.zig has already detected and is handling the offset
        if (strategy.expected_count > 0 and state.round > 1 and !strategy.offset_detected) {
            var mismatches: u8 = 0;
            const check_count = @min(strategy.expected_count, state.bot_count);
            for (0..check_count) |bi| {
                if (!strategy.expected_next_pos[bi].eql(state.bots[bi].pos)) {
                    mismatches += 1;
                }
            }
            // Aggressive detection: even 1 mismatch counts (was 60%+1 threshold)
            const threshold: u8 = 1;
            if (mismatches >= threshold) {
                desync_counter += 1;
                // Trigger after 2 consecutive rounds (was 3)
                if (desync_counter >= 2) {
                    std.debug.print("R{d} POSITION DESYNC: {d}/{d} mismatches for {d} rounds, re-syncing\n", .{ state.round, mismatches, check_count, desync_counter });
                    _ = strategy.decideActions(&state, &action_buf) catch {};
                    strategy.expected_count = 0;
                    desync_counter = 0;
                    last_responded_round = current_round;
                    continue;
                }
            } else {
                desync_counter = 0; // Reset immediately on clean round (was gradual decay)
            }
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
        last_responded_round = current_round;
    }

    std.debug.print("GAME_OVER Score:{d} Rounds:{d}\n", .{ state.score, state.round });
}

// ── Main ───────────────────────────────────────────────────────────────
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage:\n  grocery-bot <wss://...>           Live mode\n  grocery-bot --replay <logfile>    Replay mode\n", .{});
        return;
    }

    if (std.mem.eql(u8, args[1], "--replay")) {
        if (args.len < 3) {
            std.debug.print("Usage: grocery-bot --replay <logfile.jsonl>\n", .{});
            return;
        }
        try runReplay(args[2]);
    } else {
        try runLive(args[1]);
    }
}

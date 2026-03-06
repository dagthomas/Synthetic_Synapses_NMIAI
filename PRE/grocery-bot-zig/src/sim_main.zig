const std = @import("std");
const config = @import("config");
const sim = @import("sim.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse args: sim [difficulty] [seed_start] [seed_count]
    // Or:        sim [difficulty] [single_seed]
    var diff_str: []const u8 = "hard";
    var seed_start: u32 = 7001;
    var seed_count: u32 = 40;

    // Detect compile-time difficulty
    const compile_diff = config.difficulty;
    if (compile_diff != .auto) {
        diff_str = switch (compile_diff) {
            .easy => "easy",
            .medium => "medium",
            .hard => "hard",
            .expert => "expert",
            .nightmare => "nightmare",
            .auto => "hard",
        };
    }

    if (args.len >= 2) diff_str = args[1];
    if (args.len >= 3) seed_start = std.fmt.parseInt(u32, args[2], 10) catch 7001;
    if (args.len >= 4) {
        seed_count = std.fmt.parseInt(u32, args[3], 10) catch 40;
    } else if (args.len == 3) {
        // Single seed mode
        seed_count = 1;
    }

    const cfg = if (std.mem.eql(u8, diff_str, "easy"))
        sim.CONFIGS.easy
    else if (std.mem.eql(u8, diff_str, "medium"))
        sim.CONFIGS.medium
    else if (std.mem.eql(u8, diff_str, "hard"))
        sim.CONFIGS.hard
    else if (std.mem.eql(u8, diff_str, "expert"))
        sim.CONFIGS.expert
    else
        sim.CONFIGS.hard;

    std.debug.print("Sim sweep: {s}, seeds {d}-{d} ({d} seeds)\n", .{ diff_str, seed_start, seed_start + seed_count - 1, seed_count });

    var scores: [1000]i32 = undefined;
    var score_count: u32 = 0;
    var total: i64 = 0;
    var max_score: i32 = 0;
    var min_score: i32 = 999;

    const timer_start = std.time.nanoTimestamp();

    for (0..seed_count) |i| {
        const seed = seed_start + @as(u32, @intCast(i));
        const score = sim.runSeed(cfg, seed);

        if (score_count < 1000) {
            scores[score_count] = score;
            score_count += 1;
        }
        total += score;
        if (score > max_score) max_score = score;
        if (score < min_score) min_score = score;

        std.debug.print("  Seed {d}: {d}\n", .{ seed, score });
    }

    const elapsed_ns = std.time.nanoTimestamp() - timer_start;
    const elapsed_ms = @divTrunc(elapsed_ns, 1_000_000);

    std.debug.print("\n==================================================\n", .{});
    std.debug.print("Results: {s}, {d} seeds\n", .{ diff_str, seed_count });
    if (score_count > 0) {
        const mean = @divTrunc(total * 10, score_count);
        std.debug.print("  Mean:  {d}.{d}\n", .{ @divTrunc(mean, 10), @mod(mean, 10) });
        std.debug.print("  Max:   {d}\n", .{max_score});
        std.debug.print("  Min:   {d}\n", .{min_score});

        // Compute median
        sortScores(scores[0..score_count]);
        const median = scores[score_count / 2];
        const p25 = scores[score_count / 4];
        const p75 = scores[score_count * 3 / 4];
        std.debug.print("  Median:{d}\n", .{median});
        std.debug.print("  P25:   {d}\n", .{p25});
        std.debug.print("  P75:   {d}\n", .{p75});
    }
    std.debug.print("  Time:  {d}ms ({d}ms/game)\n", .{ elapsed_ms, if (seed_count > 0) @divTrunc(elapsed_ms, seed_count) else 0 });
    std.debug.print("==================================================\n", .{});
}

fn sortScores(s: []i32) void {
    for (1..s.len) |i| {
        const key = s[i];
        var j: usize = i;
        while (j > 0 and s[j - 1] > key) {
            s[j] = s[j - 1];
            j -= 1;
        }
        s[j] = key;
    }
}

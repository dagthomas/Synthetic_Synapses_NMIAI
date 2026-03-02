const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const Difficulty = enum { auto, easy, medium, hard, expert };
    const difficulty = b.option(Difficulty, "difficulty", "Target difficulty (auto = runtime detection)") orelse .auto;

    const name = switch (difficulty) {
        .auto => "grocery-bot",
        .easy => "grocery-bot-easy",
        .medium => "grocery-bot-medium",
        .hard => "grocery-bot-hard",
        .expert => "grocery-bot-expert",
    };

    const options = b.addOptions();
    options.addOption(Difficulty, "difficulty", difficulty);

    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("config", options.createModule());

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the bot");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}

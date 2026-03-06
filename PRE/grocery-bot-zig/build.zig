const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const Difficulty = enum { auto, easy, medium, hard, expert, nightmare };
    const difficulty = b.option(Difficulty, "difficulty", "Target difficulty (auto = runtime detection)") orelse .auto;

    const name = switch (difficulty) {
        .auto => "grocery-bot",
        .easy => "grocery-bot-easy",
        .medium => "grocery-bot-medium",
        .hard => "grocery-bot-hard",
        .expert => "grocery-bot-expert",
        .nightmare => "grocery-bot-nightmare",
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

    // Simulator executable (pure Zig, no WebSocket)
    const sim_name = switch (difficulty) {
        .auto => "grocery-sim",
        .easy => "grocery-sim-easy",
        .medium => "grocery-sim-medium",
        .hard => "grocery-sim-hard",
        .expert => "grocery-sim-expert",
        .nightmare => "grocery-sim-nightmare",
    };

    const sim_exe = b.addExecutable(.{
        .name = sim_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/sim_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    sim_exe.root_module.addImport("config", options.createModule());

    b.installArtifact(sim_exe);

    const sim_step = b.step("sim", "Run the simulator");
    const sim_cmd = b.addRunArtifact(sim_exe);
    sim_step.dependOn(&sim_cmd.step);
    sim_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        sim_cmd.addArgs(args);
    }

    // Shared library for Python FFI (always named grocery-sim, difficulty=runtime)
    const lib = b.addLibrary(.{
        .name = "grocery-sim",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ffi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.root_module.addImport("config", options.createModule());
    b.installArtifact(lib);
}

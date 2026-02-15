const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rnn_mod = b.addModule("facaded_rnn", .{
        .root_source_file = b.path("facaded_rnn.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "example",
        .root_source_file = b.path("example.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("facaded_rnn", rnn_mod);
    exe.addIncludePath(b.path("../c/include"));
    exe.addObjectFile(b.path("../cpp/target/release/libfacaded_rnn_cpp.a"));
    exe.linkSystemLibrary("dl");
    exe.linkSystemLibrary("pthread");
    exe.linkSystemLibrary("m");
    exe.linkLibC();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}

const std = @import("std");
const rnn = @import("facaded_rnn");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // ── Create model ──
    const hidden_sizes = [_]u32{16};
    var model = try rnn.RnnModel.init(.{
        .input_size = 2,
        .hidden_sizes = &hidden_sizes,
        .output_size = 2,
        .cell_type = "lstm",
        .activation = "tanh",
        .output_activation = "linear",
        .loss = "mse",
        .learning_rate = 0.01,
        .gradient_clip = 5.0,
        .bptt_steps = 0,
        .backend = "cpu",
    });
    defer model.deinit();

    try stdout.print("Created model:\n", .{});
    try stdout.print("  Input size:  {}\n", .{model.inputSize()});
    try stdout.print("  Output size: {}\n", .{model.outputSize()});
    try stdout.print("  Layers:      {}\n", .{model.layerCount()});
    try stdout.print("  Hidden[0]:   {}\n", .{model.hiddenSize(0)});
    try stdout.print("  GPU:         {}\n", .{model.isGpuAvailable()});

    // ── Training data ──
    const inputs = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const targets = [_]f64{ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    // ── Train 100 epochs ──
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const losses = try model.train(allocator, &inputs, &targets, 3, 2, 2, 100);
    defer allocator.free(losses);

    try stdout.print("\nTraining (100 epochs):\n", .{});
    try stdout.print("  Epoch   1 loss: {d:.6}\n", .{losses[0]});
    try stdout.print("  Epoch 100 loss: {d:.6}\n", .{losses[99]});

    // ── Predict ──
    const pred_input = [_]f64{ 0.5, 0.5 };
    const pred_output = try model.predict(allocator, &pred_input, 1, 2);
    defer allocator.free(pred_output);

    try stdout.print("\nPrediction for [0.5, 0.5]: [{d:.6}, {d:.6}]\n", .{ pred_output[0], pred_output[1] });

    // ── Introspection ──
    try stdout.print("\nIntrospection:\n", .{});
    try stdout.print("  Learning rate:    {d:.6}\n", .{model.learningRate()});
    try stdout.print("  Gradient clip:    {d:.2}\n", .{model.gradientClip()});
    try stdout.print("  Dropout rate:     {d:.6}\n", .{model.dropoutRate()});
    try stdout.print("  Sequence length:  {}\n", .{model.sequenceLength()});

    // ── Gradient diagnostics ──
    const vanish = model.detectVanishingGradients(1e-7);
    try stdout.print("\n  Vanishing (< 1e-7): count={}, min={d:.10}\n", .{ vanish.count, vanish.value });

    const explode = model.detectExplodingGradients(100.0);
    try stdout.print("  Exploding (> 100):  count={}, max={d:.10}\n", .{ explode.count, explode.value });

    // ── Save & reload ──
    try model.save("/tmp/zig_test_model.json");
    try stdout.print("\nModel saved to /tmp/zig_test_model.json\n", .{});

    var loaded = try rnn.RnnModel.load("/tmp/zig_test_model.json", "cpu");
    defer loaded.deinit();
    try stdout.print("Reloaded: {} layers, hidden[0]={}\n", .{ loaded.layerCount(), loaded.hiddenSize(0) });

    // ── Modify parameters ──
    model.setLearningRate(0.005);
    try stdout.print("\nLR after set: {d:.6}\n", .{model.learningRate()});

    model.setDropoutRate(0.1);
    try stdout.print("Dropout after set: {d:.6}\n", .{model.dropoutRate()});

    try stdout.print("\nDone.\n", .{});
}

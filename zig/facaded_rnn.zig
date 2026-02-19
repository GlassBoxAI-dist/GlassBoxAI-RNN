/**
 * @file
 * @ingroup RNN_Wrappers
 */
const std = @import("std");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("facaded_rnn.h");
});

pub const RnnError = error{
    CreateFailed,
    LoadFailed,
    SaveFailed,
    PredictFailed,
    ForwardFailed,
    OutOfMemory,
};

pub const GradientDiagnostic = struct {
    count: i32,
    value: f64,
};

pub const ModelOptions = struct {
    input_size: u32,
    hidden_sizes: []const u32,
    output_size: u32,
    cell_type: [*:0]const u8 = "lstm",
    activation: [*:0]const u8 = "tanh",
    output_activation: [*:0]const u8 = "linear",
    loss: [*:0]const u8 = "mse",
    learning_rate: f64 = 0.01,
    gradient_clip: f64 = 5.0,
    bptt_steps: u32 = 0,
    backend: [*:0]const u8 = "auto",
};

pub const RnnModel = struct {
    handle: *c.RnnHandle,

    pub fn init(opts: ModelOptions) RnnError!RnnModel {
        const handle = c.rnn_create(
            opts.input_size,
            opts.hidden_sizes.ptr,
            @intCast(opts.hidden_sizes.len),
            opts.output_size,
            opts.cell_type,
            opts.activation,
            opts.output_activation,
            opts.loss,
            opts.learning_rate,
            opts.gradient_clip,
            opts.bptt_steps,
            opts.backend,
        );
        if (handle) |h| {
            return .{ .handle = h };
        }
        return RnnError.CreateFailed;
    }

    pub fn load(filename: [*:0]const u8, backend: [*:0]const u8) RnnError!RnnModel {
        const handle = c.rnn_load(filename, backend);
        if (handle) |h| {
            return .{ .handle = h };
        }
        return RnnError.LoadFailed;
    }

    pub fn deinit(self: *RnnModel) void {
        c.rnn_destroy(self.handle);
        self.handle = undefined;
    }

    pub fn save(self: *const RnnModel, filename: [*:0]const u8) RnnError!void {
        if (c.rnn_save(self.handle, filename) != 0) {
            return RnnError.SaveFailed;
        }
    }

    pub fn lastError() ?[*:0]const u8 {
        return c.rnn_last_error();
    }

    pub fn clearError() void {
        c.rnn_clear_error();
    }

    // ── Training & Inference ──

    pub fn predict(self: *RnnModel, allocator: Allocator, input_data: []const f64, num_timesteps: u32, input_size: u32) RnnError![]f64 {
        const total = c.rnn_predict(self.handle, input_data.ptr, num_timesteps, input_size, null, 0);
        if (total <= 0) return RnnError.PredictFailed;

        const buf = allocator.alloc(f64, @intCast(total)) catch return RnnError.OutOfMemory;
        _ = c.rnn_predict(self.handle, input_data.ptr, num_timesteps, input_size, buf.ptr, @intCast(total));
        return buf;
    }

    pub fn trainSequence(self: *RnnModel, input_data: []const f64, target_data: []const f64, num_timesteps: u32, input_size: u32, output_size: u32) f64 {
        return c.rnn_train_sequence(self.handle, input_data.ptr, target_data.ptr, num_timesteps, input_size, output_size);
    }

    pub fn train(self: *RnnModel, allocator: Allocator, input_data: []const f64, target_data: []const f64, num_timesteps: u32, input_size: u32, output_size: u32, epochs: u32) RnnError![]f64 {
        const losses = allocator.alloc(f64, epochs) catch return RnnError.OutOfMemory;
        _ = c.rnn_train(self.handle, input_data.ptr, target_data.ptr, num_timesteps, input_size, output_size, epochs, losses.ptr, epochs);
        return losses;
    }

    pub fn forwardSequence(self: *RnnModel, allocator: Allocator, input_data: []const f64, num_timesteps: u32, input_size: u32) RnnError![]f64 {
        const total = c.rnn_forward_sequence(self.handle, input_data.ptr, num_timesteps, input_size, null, 0);
        if (total <= 0) return RnnError.ForwardFailed;

        const buf = allocator.alloc(f64, @intCast(total)) catch return RnnError.OutOfMemory;
        _ = c.rnn_forward_sequence(self.handle, input_data.ptr, num_timesteps, input_size, buf.ptr, @intCast(total));
        return buf;
    }

    pub fn backwardSequence(self: *RnnModel, target_data: []const f64, num_timesteps: u32, output_size: u32) f64 {
        return c.rnn_backward_sequence(self.handle, target_data.ptr, num_timesteps, output_size);
    }

    pub fn resetStates(self: *RnnModel) void {
        c.rnn_reset_states(self.handle);
    }

    // ── Properties ──

    pub fn inputSize(self: *const RnnModel) u32 {
        return c.rnn_get_input_size(self.handle);
    }

    pub fn outputSize(self: *const RnnModel) u32 {
        return c.rnn_get_output_size(self.handle);
    }

    pub fn layerCount(self: *const RnnModel) u32 {
        return c.rnn_get_layer_count(self.handle);
    }

    pub fn hiddenSize(self: *const RnnModel, layer: u32) u32 {
        return c.rnn_get_hidden_size(self.handle, layer);
    }

    pub fn sequenceLength(self: *const RnnModel) u32 {
        return c.rnn_get_sequence_length(self.handle);
    }

    pub fn learningRate(self: *const RnnModel) f64 {
        return c.rnn_get_learning_rate(self.handle);
    }

    pub fn setLearningRate(self: *RnnModel, lr: f64) void {
        c.rnn_set_learning_rate(self.handle, lr);
    }

    pub fn gradientClip(self: *const RnnModel) f64 {
        return c.rnn_get_gradient_clip(self.handle);
    }

    pub fn dropoutRate(self: *const RnnModel) f64 {
        return c.rnn_get_dropout_rate(self.handle);
    }

    pub fn setDropoutRate(self: *RnnModel, rate: f64) void {
        c.rnn_set_dropout_rate(self.handle, rate);
    }

    pub fn isGpuAvailable(self: *const RnnModel) bool {
        return c.rnn_is_gpu_available(self.handle) != 0;
    }

    // ── Introspection ──

    pub fn getHiddenValue(self: *const RnnModel, layer: u32, timestep: u32, neuron: u32) f64 {
        return c.rnn_get_hidden_value(self.handle, layer, timestep, neuron);
    }

    pub fn setHiddenValue(self: *RnnModel, layer: u32, neuron: u32, value: f64) void {
        c.rnn_set_hidden_value(self.handle, layer, neuron, value);
    }

    pub fn getOutputValue(self: *const RnnModel, timestep: u32, index: u32) f64 {
        return c.rnn_get_output_value(self.handle, timestep, index);
    }

    pub fn getCellState(self: *const RnnModel, layer: u32, neuron: u32) f64 {
        return c.rnn_get_cell_state(self.handle, layer, neuron);
    }

    pub fn getGateValue(self: *const RnnModel, layer: u32, timestep: u32, neuron: u32, gate: [*:0]const u8) f64 {
        return c.rnn_get_gate_value(self.handle, layer, timestep, neuron, gate);
    }

    pub fn getPreactivation(self: *const RnnModel, layer: u32, timestep: u32, neuron: u32) f64 {
        return c.rnn_get_preactivation(self.handle, layer, timestep, neuron);
    }

    pub fn getInputValue(self: *const RnnModel, timestep: u32, index: u32) f64 {
        return c.rnn_get_input_value(self.handle, timestep, index);
    }

    pub fn getSequenceOutputs(self: *const RnnModel, allocator: Allocator) RnnError![]f64 {
        const total = c.rnn_get_sequence_outputs(self.handle, null, 0);
        if (total <= 0) return RnnError.PredictFailed;

        const buf = allocator.alloc(f64, @intCast(total)) catch return RnnError.OutOfMemory;
        _ = c.rnn_get_sequence_outputs(self.handle, buf.ptr, @intCast(total));
        return buf;
    }

    pub fn getSequenceHiddenStates(self: *const RnnModel, allocator: Allocator, layer: u32) RnnError![]f64 {
        const total = c.rnn_get_sequence_hidden_states(self.handle, layer, null, 0);
        if (total <= 0) return RnnError.PredictFailed;

        const buf = allocator.alloc(f64, @intCast(total)) catch return RnnError.OutOfMemory;
        _ = c.rnn_get_sequence_hidden_states(self.handle, layer, buf.ptr, @intCast(total));
        return buf;
    }

    // ── Gradient Diagnostics ──

    pub fn detectVanishingGradients(self: *const RnnModel, threshold: f64) GradientDiagnostic {
        var count: i32 = 0;
        var min_val: f64 = 0;
        c.rnn_detect_vanishing_gradients(self.handle, threshold, &count, &min_val);
        return .{ .count = count, .value = min_val };
    }

    pub fn detectExplodingGradients(self: *const RnnModel, threshold: f64) GradientDiagnostic {
        var count: i32 = 0;
        var max_val: f64 = 0;
        c.rnn_detect_exploding_gradients(self.handle, threshold, &count, &max_val);
        return .{ .count = count, .value = max_val };
    }
};


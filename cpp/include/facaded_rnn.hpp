/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * C++ wrapper for facaded_rnn GPU-accelerated RNN library
 */

#ifndef FACADED_RNN_HPP
#define FACADED_RNN_HPP

#include "facaded_rnn.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <utility>

namespace facaded_rnn {

class RnnError : public std::runtime_error {
public:
    explicit RnnError(const std::string& msg) : std::runtime_error(msg) {}
};

struct GradientDiagnostic {
    int32_t count;
    double  value;
};

class RNNModel {
public:
    RNNModel(
        uint32_t input_size,
        const std::vector<uint32_t>& hidden_sizes,
        uint32_t output_size,
        const std::string& cell_type       = "lstm",
        const std::string& activation      = "tanh",
        const std::string& output_activation = "linear",
        const std::string& loss            = "mse",
        double learning_rate               = 0.01,
        double gradient_clip               = 5.0,
        uint32_t bptt_steps                = 0,
        const std::string& backend         = "auto"
    ) {
        handle_ = rnn_create(
            input_size,
            hidden_sizes.data(),
            static_cast<uint32_t>(hidden_sizes.size()),
            output_size,
            cell_type.c_str(),
            activation.c_str(),
            output_activation.c_str(),
            loss.c_str(),
            learning_rate,
            gradient_clip,
            bptt_steps,
            backend.c_str()
        );
        check_handle();
    }

    ~RNNModel() {
        if (handle_) {
            rnn_destroy(handle_);
            handle_ = nullptr;
        }
    }

    RNNModel(const RNNModel&) = delete;
    RNNModel& operator=(const RNNModel&) = delete;

    RNNModel(RNNModel&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    RNNModel& operator=(RNNModel&& other) noexcept {
        if (this != &other) {
            if (handle_) rnn_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    static RNNModel load(const std::string& filename, const std::string& backend = "auto") {
        return RNNModel(filename, backend, LoadTag{});
    }

    void save(const std::string& filename) const {
        if (rnn_save(handle_, filename.c_str()) != 0) {
            throw_last_error("Failed to save model");
        }
    }

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs) {
        auto flat = flatten(inputs);
        uint32_t timesteps = static_cast<uint32_t>(inputs.size());
        uint32_t in_size   = timesteps > 0 ? static_cast<uint32_t>(inputs[0].size()) : 0;

        int32_t total = rnn_predict(handle_, flat.data(), timesteps, in_size, nullptr, 0);
        if (total <= 0) return {};

        std::vector<double> buf(total);
        rnn_predict(handle_, flat.data(), timesteps, in_size, buf.data(), total);

        return unflatten(buf, timesteps, rnn_get_output_size(handle_));
    }

    double train_sequence(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets
    ) {
        auto flat_in  = flatten(inputs);
        auto flat_tgt = flatten(targets);
        uint32_t timesteps = static_cast<uint32_t>(inputs.size());
        uint32_t in_size   = timesteps > 0 ? static_cast<uint32_t>(inputs[0].size()) : 0;
        uint32_t out_size  = timesteps > 0 ? static_cast<uint32_t>(targets[0].size()) : 0;

        return rnn_train_sequence(handle_, flat_in.data(), flat_tgt.data(), timesteps, in_size, out_size);
    }

    std::vector<double> train(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        uint32_t epochs
    ) {
        auto flat_in  = flatten(inputs);
        auto flat_tgt = flatten(targets);
        uint32_t timesteps = static_cast<uint32_t>(inputs.size());
        uint32_t in_size   = timesteps > 0 ? static_cast<uint32_t>(inputs[0].size()) : 0;
        uint32_t out_size  = timesteps > 0 ? static_cast<uint32_t>(targets[0].size()) : 0;

        std::vector<double> losses(epochs);
        rnn_train(handle_, flat_in.data(), flat_tgt.data(), timesteps, in_size, out_size, epochs, losses.data(), epochs);
        return losses;
    }

    std::vector<std::vector<double>> forward_sequence(const std::vector<std::vector<double>>& inputs) {
        auto flat = flatten(inputs);
        uint32_t timesteps = static_cast<uint32_t>(inputs.size());
        uint32_t in_size   = timesteps > 0 ? static_cast<uint32_t>(inputs[0].size()) : 0;

        int32_t total = rnn_forward_sequence(handle_, flat.data(), timesteps, in_size, nullptr, 0);
        if (total <= 0) return {};

        std::vector<double> buf(total);
        rnn_forward_sequence(handle_, flat.data(), timesteps, in_size, buf.data(), total);

        return unflatten(buf, timesteps, rnn_get_output_size(handle_));
    }

    double backward_sequence(const std::vector<std::vector<double>>& targets) {
        auto flat = flatten(targets);
        uint32_t timesteps = static_cast<uint32_t>(targets.size());
        uint32_t out_size  = timesteps > 0 ? static_cast<uint32_t>(targets[0].size()) : 0;

        return rnn_backward_sequence(handle_, flat.data(), timesteps, out_size);
    }

    void reset_states() { rnn_reset_states(handle_); }

    uint32_t input_size()       const { return rnn_get_input_size(handle_); }
    uint32_t output_size()      const { return rnn_get_output_size(handle_); }
    uint32_t layer_count()      const { return rnn_get_layer_count(handle_); }
    uint32_t sequence_length()  const { return rnn_get_sequence_length(handle_); }
    bool     gpu_available()    const { return rnn_is_gpu_available(handle_) != 0; }

    uint32_t hidden_size(uint32_t layer) const {
        return rnn_get_hidden_size(handle_, layer);
    }

    double learning_rate() const { return rnn_get_learning_rate(handle_); }
    void   set_learning_rate(double lr) { rnn_set_learning_rate(handle_, lr); }

    double gradient_clip() const { return rnn_get_gradient_clip(handle_); }

    double dropout_rate() const { return rnn_get_dropout_rate(handle_); }
    void   set_dropout_rate(double rate) { rnn_set_dropout_rate(handle_, rate); }

    double get_hidden_value(uint32_t layer, uint32_t timestep, uint32_t neuron) const {
        return rnn_get_hidden_value(handle_, layer, timestep, neuron);
    }

    void set_hidden_value(uint32_t layer, uint32_t neuron, double value) {
        rnn_set_hidden_value(handle_, layer, neuron, value);
    }

    double get_output_value(uint32_t timestep, uint32_t index) const {
        return rnn_get_output_value(handle_, timestep, index);
    }

    double get_cell_state(uint32_t layer, uint32_t neuron) const {
        return rnn_get_cell_state(handle_, layer, neuron);
    }

    double get_gate_value(uint32_t layer, uint32_t timestep, uint32_t neuron, const std::string& gate) const {
        return rnn_get_gate_value(handle_, layer, timestep, neuron, gate.c_str());
    }

    double get_preactivation(uint32_t layer, uint32_t timestep, uint32_t neuron) const {
        return rnn_get_preactivation(handle_, layer, timestep, neuron);
    }

    double get_input_value(uint32_t timestep, uint32_t index) const {
        return rnn_get_input_value(handle_, timestep, index);
    }

    std::vector<std::vector<double>> get_sequence_outputs() const {
        int32_t total = rnn_get_sequence_outputs(handle_, nullptr, 0);
        if (total <= 0) return {};
        std::vector<double> buf(total);
        rnn_get_sequence_outputs(handle_, buf.data(), total);
        uint32_t seq_len = rnn_get_sequence_length(handle_);
        return unflatten(buf, seq_len, rnn_get_output_size(handle_));
    }

    std::vector<std::vector<double>> get_sequence_hidden_states(uint32_t layer) const {
        int32_t total = rnn_get_sequence_hidden_states(handle_, layer, nullptr, 0);
        if (total <= 0) return {};
        std::vector<double> buf(total);
        rnn_get_sequence_hidden_states(handle_, layer, buf.data(), total);
        uint32_t seq_len = rnn_get_sequence_length(handle_);
        return unflatten(buf, seq_len, rnn_get_hidden_size(handle_, layer));
    }

    GradientDiagnostic detect_vanishing_gradients(double threshold) const {
        GradientDiagnostic d{};
        rnn_detect_vanishing_gradients(handle_, threshold, &d.count, &d.value);
        return d;
    }

    GradientDiagnostic detect_exploding_gradients(double threshold) const {
        GradientDiagnostic d{};
        rnn_detect_exploding_gradients(handle_, threshold, &d.count, &d.value);
        return d;
    }

private:
    RnnHandle* handle_ = nullptr;

    struct LoadTag {};

    RNNModel(const std::string& filename, const std::string& backend, LoadTag) {
        handle_ = rnn_load(filename.c_str(), backend.c_str());
        check_handle();
    }

    void check_handle() const {
        if (!handle_) {
            throw_last_error("Failed to create/load RNN model");
        }
    }

    static void throw_last_error(const char* fallback) {
        const char* err = rnn_last_error();
        if (err) {
            std::string msg(err);
            rnn_clear_error();
            throw RnnError(msg);
        }
        throw RnnError(fallback);
    }

    static std::vector<double> flatten(const std::vector<std::vector<double>>& data) {
        std::vector<double> flat;
        for (auto& row : data) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }

    static std::vector<std::vector<double>> unflatten(
        const std::vector<double>& flat,
        uint32_t rows,
        uint32_t cols
    ) {
        std::vector<std::vector<double>> result;
        result.reserve(rows);
        for (uint32_t r = 0; r < rows; ++r) {
            size_t start = static_cast<size_t>(r) * cols;
            size_t end   = start + cols;
            if (end > flat.size()) end = flat.size();
            result.emplace_back(flat.begin() + start, flat.begin() + end);
        }
        return result;
    }
};

} // namespace facaded_rnn

#endif /* FACADED_RNN_HPP */

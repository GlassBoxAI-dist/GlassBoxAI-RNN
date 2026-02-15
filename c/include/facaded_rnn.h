/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * C API for facaded_rnn GPU-accelerated RNN library
 *
 * All functions use an opaque RnnHandle* pointer. Create with rnn_create()
 * or rnn_load(), and free with rnn_destroy(). On error, functions return -1
 * or NULL; call rnn_last_error() for details.
 *
 * Data layout: input/target/output arrays are row-major flat arrays.
 *   For N timesteps with K features each, pass a contiguous double[N*K].
 *   Row i occupies indices [i*K .. (i+1)*K).
 *
 * Build the Rust FFI library first:
 *   cd cpp && cargo build --release --no-default-features
 *
 * Then link against libfacaded_rnn_cpp.a (static) or .so/.dylib (shared).
 */

#ifndef FACADED_RNN_H
#define FACADED_RNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ─────────────────────────────────────────────────────── */

typedef struct RnnHandle RnnHandle;

/* ── Error handling ────────────────────────────────────────────────────── */

/* Returns the last error message, or NULL if no error. */
const char* rnn_last_error(void);

/* Clears the last error. */
void rnn_clear_error(void);

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

/*
 * Create a new RNN model.
 *
 * cell_type:         "simplernn", "lstm", or "gru"   (default: "lstm")
 * activation:        "sigmoid", "tanh", "relu", "linear" (default: "tanh")
 * output_activation: same choices                    (default: "linear")
 * loss:              "mse" or "crossentropy"         (default: "mse")
 * backend:           "auto", "cpu", "cuda", "opencl", "hybrid" (default: "auto")
 *
 * Pass NULL for any string parameter to use its default.
 * Returns NULL on error (check rnn_last_error()).
 */
RnnHandle* rnn_create(
    uint32_t        input_size,
    const uint32_t* hidden_sizes,
    uint32_t        num_hidden_layers,
    uint32_t        output_size,
    const char*     cell_type,
    const char*     activation,
    const char*     output_activation,
    const char*     loss,
    double          learning_rate,
    double          gradient_clip,
    uint32_t        bptt_steps,
    const char*     backend
);

/*
 * Load a model from a JSON file.
 * Returns NULL on error.
 */
RnnHandle* rnn_load(const char* filename, const char* backend);

/* Free all resources associated with a model. Safe to call with NULL. */
void rnn_destroy(RnnHandle* handle);

/* Save a model to a JSON file. Returns 0 on success, -1 on error. */
int32_t rnn_save(const RnnHandle* handle, const char* filename);

/* ── Training & Inference ──────────────────────────────────────────────── */

/*
 * Run prediction (forward pass with state reset).
 *
 * input_data:    flat row-major array [num_timesteps * input_size]
 * output_buf:    caller-allocated buffer, or NULL to query required size
 * output_buf_len: number of doubles in output_buf
 *
 * Returns total number of output doubles (timesteps * output_size),
 * or -1 on error.
 */
int32_t rnn_predict(
    RnnHandle*    handle,
    const double* input_data,
    uint32_t      num_timesteps,
    uint32_t      input_size,
    double*       output_buf,
    uint32_t      output_buf_len
);

/*
 * Train on one sequence. Returns the loss, or NaN on error.
 */
double rnn_train_sequence(
    RnnHandle*    handle,
    const double* input_data,
    const double* target_data,
    uint32_t      num_timesteps,
    uint32_t      input_size,
    uint32_t      output_size
);

/*
 * Train for multiple epochs. Writes per-epoch loss into loss_buf.
 * Returns number of epochs completed, or -1 on error.
 */
int32_t rnn_train(
    RnnHandle*    handle,
    const double* input_data,
    const double* target_data,
    uint32_t      num_timesteps,
    uint32_t      input_size,
    uint32_t      output_size,
    uint32_t      epochs,
    double*       loss_buf,
    uint32_t      loss_buf_len
);

/*
 * Forward pass without state reset (accumulates state).
 * Same buffer contract as rnn_predict().
 */
int32_t rnn_forward_sequence(
    RnnHandle*    handle,
    const double* input_data,
    uint32_t      num_timesteps,
    uint32_t      input_size,
    double*       output_buf,
    uint32_t      output_buf_len
);

/*
 * Backward pass (call after forward_sequence).
 * Returns the loss, or NaN on error.
 */
double rnn_backward_sequence(
    RnnHandle*    handle,
    const double* target_data,
    uint32_t      num_timesteps,
    uint32_t      output_size
);

/* Reset all hidden/cell states to zero. */
void rnn_reset_states(RnnHandle* handle);

/* ── Model properties ──────────────────────────────────────────────────── */

uint32_t rnn_get_input_size(const RnnHandle* handle);
uint32_t rnn_get_output_size(const RnnHandle* handle);
uint32_t rnn_get_layer_count(const RnnHandle* handle);
uint32_t rnn_get_hidden_size(const RnnHandle* handle, uint32_t layer);
uint32_t rnn_get_sequence_length(const RnnHandle* handle);

double  rnn_get_learning_rate(const RnnHandle* handle);
void    rnn_set_learning_rate(RnnHandle* handle, double lr);
double  rnn_get_gradient_clip(const RnnHandle* handle);
double  rnn_get_dropout_rate(const RnnHandle* handle);
void    rnn_set_dropout_rate(RnnHandle* handle, double rate);
int32_t rnn_is_gpu_available(const RnnHandle* handle);

/* ── Facade introspection ──────────────────────────────────────────────── */

double rnn_get_hidden_value(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron);
void   rnn_set_hidden_value(RnnHandle* handle, uint32_t layer, uint32_t neuron, double value);
double rnn_get_output_value(const RnnHandle* handle, uint32_t timestep, uint32_t index);
double rnn_get_cell_state(const RnnHandle* handle, uint32_t layer, uint32_t neuron);

/*
 * gate: "forget", "input", "output", "cellcandidate" (LSTM)
 *       "update", "reset", "hiddencandidate"         (GRU)
 */
double rnn_get_gate_value(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron, const char* gate);
double rnn_get_preactivation(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron);
double rnn_get_input_value(const RnnHandle* handle, uint32_t timestep, uint32_t index);

/*
 * Bulk retrieval. Pass NULL for output_buf to query required size.
 * Returns total doubles written (or needed), or -1 on error.
 */
int32_t rnn_get_sequence_outputs(const RnnHandle* handle, double* output_buf, uint32_t buf_len);
int32_t rnn_get_sequence_hidden_states(const RnnHandle* handle, uint32_t layer, double* output_buf, uint32_t buf_len);

/* ── Gradient diagnostics ──────────────────────────────────────────────── */

void rnn_detect_vanishing_gradients(const RnnHandle* handle, double threshold, int32_t* out_count, double* out_min);
void rnn_detect_exploding_gradients(const RnnHandle* handle, double threshold, int32_t* out_count, double* out_max);

#ifdef __cplusplus
}
#endif

#endif /* FACADED_RNN_H */

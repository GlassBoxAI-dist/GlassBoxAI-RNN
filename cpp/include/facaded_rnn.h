/**
 * @file
 * @ingroup RNN_Internal_Logic
 */
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * C API for facaded_rnn GPU-accelerated RNN library
 */

#ifndef FACADED_RNN_H
#define FACADED_RNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RnnHandle RnnHandle;

const char* rnn_last_error(void);
void        rnn_clear_error(void);

RnnHandle* rnn_create(
    uint32_t    input_size,
    const uint32_t* hidden_sizes,
    uint32_t    num_hidden_layers,
    uint32_t    output_size,
    const char* cell_type,
    const char* activation,
    const char* output_activation,
    const char* loss,
    double      learning_rate,
    double      gradient_clip,
    uint32_t    bptt_steps,
    const char* backend
);

RnnHandle* rnn_load(const char* filename, const char* backend);
void       rnn_destroy(RnnHandle* handle);
int32_t    rnn_save(const RnnHandle* handle, const char* filename);

int32_t rnn_predict(
    RnnHandle*  handle,
    const double* input_data,
    uint32_t    num_timesteps,
    uint32_t    input_size,
    double*     output_buf,
    uint32_t    output_buf_len
);

double rnn_train_sequence(
    RnnHandle*  handle,
    const double* input_data,
    const double* target_data,
    uint32_t    num_timesteps,
    uint32_t    input_size,
    uint32_t    output_size
);

int32_t rnn_train(
    RnnHandle*  handle,
    const double* input_data,
    const double* target_data,
    uint32_t    num_timesteps,
    uint32_t    input_size,
    uint32_t    output_size,
    uint32_t    epochs,
    double*     loss_buf,
    uint32_t    loss_buf_len
);

int32_t rnn_forward_sequence(
    RnnHandle*  handle,
    const double* input_data,
    uint32_t    num_timesteps,
    uint32_t    input_size,
    double*     output_buf,
    uint32_t    output_buf_len
);

double rnn_backward_sequence(
    RnnHandle*  handle,
    const double* target_data,
    uint32_t    num_timesteps,
    uint32_t    output_size
);

void rnn_reset_states(RnnHandle* handle);

uint32_t rnn_get_input_size(const RnnHandle* handle);
uint32_t rnn_get_output_size(const RnnHandle* handle);
uint32_t rnn_get_layer_count(const RnnHandle* handle);
uint32_t rnn_get_hidden_size(const RnnHandle* handle, uint32_t layer);
uint32_t rnn_get_sequence_length(const RnnHandle* handle);

double rnn_get_learning_rate(const RnnHandle* handle);
void   rnn_set_learning_rate(RnnHandle* handle, double lr);
double rnn_get_gradient_clip(const RnnHandle* handle);
double rnn_get_dropout_rate(const RnnHandle* handle);
void   rnn_set_dropout_rate(RnnHandle* handle, double rate);
int32_t rnn_is_gpu_available(const RnnHandle* handle);

double rnn_get_hidden_value(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron);
void   rnn_set_hidden_value(RnnHandle* handle, uint32_t layer, uint32_t neuron, double value);
double rnn_get_output_value(const RnnHandle* handle, uint32_t timestep, uint32_t index);
double rnn_get_cell_state(const RnnHandle* handle, uint32_t layer, uint32_t neuron);
double rnn_get_gate_value(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron, const char* gate);
double rnn_get_preactivation(const RnnHandle* handle, uint32_t layer, uint32_t timestep, uint32_t neuron);
double rnn_get_input_value(const RnnHandle* handle, uint32_t timestep, uint32_t index);

int32_t rnn_get_sequence_outputs(const RnnHandle* handle, double* output_buf, uint32_t buf_len);
int32_t rnn_get_sequence_hidden_states(const RnnHandle* handle, uint32_t layer, double* output_buf, uint32_t buf_len);

void rnn_detect_vanishing_gradients(const RnnHandle* handle, double threshold, int32_t* out_count, double* out_min);
void rnn_detect_exploding_gradients(const RnnHandle* handle, double threshold, int32_t* out_count, double* out_max);

#ifdef __cplusplus
}
#endif

#endif /* FACADED_RNN_H */

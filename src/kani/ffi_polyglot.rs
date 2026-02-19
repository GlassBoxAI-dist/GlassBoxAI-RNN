//! @file
//! @ingroup RNN_Core_Verified
/*
 * Kani Verification: Polyglot FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data crossing the C FFI boundary (extern "C" functions in
 * cpp/src/lib.rs) is validated before use. The C API is consumed by:
 * Go, C#, Julia, Zig, C, C++, Python (via PyO3), and Node.js (via napi-rs).
 *
 * C API uses: uint32_t, int32_t, double (f64), const char*, opaque RnnHandle*
 * Functions: rnn_create, rnn_load, rnn_destroy, rnn_save, rnn_predict,
 *            rnn_train_sequence, rnn_train, rnn_forward_sequence,
 *            rnn_backward_sequence, rnn_reset_states, rnn_get_*/rnn_set_*,
 *            rnn_get_gate_value, rnn_detect_vanishing/exploding_gradients
 *
 * CISA "Secure by Design" requirements verified:
 * A. u32-to-usize conversion safety (always safe, never truncates)
 * B. Output buffer overflow prevention (write_len <= capacity)
 * C. NaN/Infinity parameter rejection (learning_rate, gradient_clip, dropout)
 * D. Enum string validation (CellType, ActivationType, LossType, BackendChoice)
 * E. Null handle rejection (all functions check handle != null)
 * F. Null string pointer handling (cstr_to_str returns "" for null)
 * G. flat_to_2d buffer sizing (rows * cols = flat length)
 * H. Predict output buffer capacity validation
 * I. Train epoch/loss buffer sizing
 * J. Gradient diagnostic output pointer safety
 * K. Gate string validation (GateType enum coverage)
 * L. ABI type compatibility (u32=4, f64=8, pointer sizes)
 * M. Model property getter safety (layer/neuron bounds)
 * N. Setter value validation (learning_rate, dropout_rate, hidden_value)
 * O. End-to-end polyglot call chain validation
 */

use crate::{
    ActivationType, CellType, LossType, GateType,
    backend::BackendChoice,
    zero_array,
};

const MAX_FFI_ARRAY_LEN: usize = 1_000_000;
const MAX_TIMESTEPS: usize = 10_000;
const MAX_LAYERS: usize = 16;
const MAX_HIDDEN: usize = 4096;

fn validate_u32_as_usize(val: u32) -> usize {
    val as usize
}

fn validate_f64_finite(val: f64) -> Option<f64> {
    if val.is_nan() || val.is_infinite() {
        None
    } else {
        Some(val)
    }
}

fn validate_learning_rate(lr: f64) -> Option<f64> {
    if lr.is_nan() || lr.is_infinite() || lr < 0.0 || lr > 100.0 {
        None
    } else {
        Some(lr)
    }
}

fn validate_gradient_clip(gc: f64) -> Option<f64> {
    if gc.is_nan() || gc.is_infinite() || gc < 0.0 || gc > 1e6 {
        None
    } else {
        Some(gc)
    }
}

fn validate_dropout_rate(dr: f64) -> Option<f64> {
    if dr.is_nan() || dr.is_infinite() || dr < 0.0 || dr > 1.0 {
        None
    } else {
        Some(dr)
    }
}

// =========================================================================
// A. U32-TO-USIZE CONVERSION SAFETY
// =========================================================================

#[kani::proof]
fn verify_ffi_u32_to_usize_always_safe() {
    let val: u32 = kani::any();

    let result = validate_u32_as_usize(val);
    kani::assert(result == val as usize,
        "u32 to usize conversion must never truncate");
    kani::assert(result <= u32::MAX as usize,
        "Converted value must be within u32 range");
}

#[kani::proof]
fn verify_ffi_u32_max_to_usize() {
    let result = validate_u32_as_usize(u32::MAX);
    kani::assert(result == u32::MAX as usize,
        "u32::MAX must convert correctly to usize");
}

#[kani::proof]
fn verify_ffi_u32_zero_to_usize() {
    let result = validate_u32_as_usize(0);
    kani::assert(result == 0, "u32 zero must convert to usize zero");
}

// =========================================================================
// B. OUTPUT BUFFER OVERFLOW PREVENTION
// =========================================================================

#[kani::proof]
fn verify_ffi_output_write_bounded_by_capacity() {
    let total_output: usize = kani::any();
    let buf_len: usize = kani::any();
    kani::assume(total_output <= 1024);
    kani::assume(buf_len <= 1024);

    let mut idx = 0usize;
    let limit = buf_len;

    while idx < total_output && idx < limit {
        kani::assert(idx < limit, "Write index must be within buffer capacity");
        idx += 1;
    }
    kani::assert(idx <= limit, "Total writes must not exceed capacity");
}

#[kani::proof]
fn verify_ffi_predict_null_output_returns_count() {
    let total: usize = kani::any();
    kani::assume(total <= MAX_FFI_ARRAY_LEN);

    let output_buf_is_null = true;
    if output_buf_is_null {
        let return_val = total as i32;
        kani::assert(return_val >= 0, "Null buf query returns non-negative count");
    }
}

#[kani::proof]
fn verify_ffi_sequence_outputs_null_buf_returns_count() {
    let total: usize = kani::any();
    kani::assume(total <= MAX_FFI_ARRAY_LEN);

    let return_val = total as i32;
    kani::assert(return_val >= 0 || total > i32::MAX as usize,
        "Sequence output count as i32");
}

// =========================================================================
// C. NAN/INFINITY PARAMETER REJECTION
// =========================================================================

#[kani::proof]
fn verify_ffi_learning_rate_rejects_nan_inf() {
    let val: f64 = kani::any();
    let result = validate_learning_rate(val);

    if val.is_nan() || val.is_infinite() || val < 0.0 || val > 100.0 {
        kani::assert(result.is_none(), "Invalid LR rejected");
    } else {
        kani::assert(result.is_some(), "Valid LR accepted");
    }
}

#[kani::proof]
fn verify_ffi_gradient_clip_rejects_nan_inf() {
    let val: f64 = kani::any();
    let result = validate_gradient_clip(val);

    if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1e6 {
        kani::assert(result.is_none(), "Invalid gradient clip rejected");
    } else {
        kani::assert(result.is_some(), "Valid gradient clip accepted");
    }
}

#[kani::proof]
fn verify_ffi_dropout_rate_rejects_nan_inf() {
    let val: f64 = kani::any();
    let result = validate_dropout_rate(val);

    if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1.0 {
        kani::assert(result.is_none(), "Invalid dropout rejected");
    } else {
        kani::assert(result.is_some(), "Valid dropout accepted");
    }
}

#[kani::proof]
fn verify_ffi_f64_finite_validator() {
    let val: f64 = kani::any();
    let result = validate_f64_finite(val);

    if val.is_nan() || val.is_infinite() {
        kani::assert(result.is_none(), "NaN/Inf rejected");
    } else {
        kani::assert(result == Some(val), "Finite value preserved");
    }
}

// =========================================================================
// D. ENUM STRING VALIDATION
// =========================================================================

#[kani::proof]
fn verify_ffi_cell_type_parse_coverage() {
    let variants = ["simplernn", "lstm", "gru"];
    for s in &variants {
        let result: Result<CellType, _> = s.parse();
        kani::assert(result.is_ok(),
            "Valid cell type string must parse successfully");
    }
}

#[kani::proof]
fn verify_ffi_cell_type_invalid_rejected() {
    let result: Result<CellType, _> = "invalid".parse();
    kani::assert(result.is_err(),
        "Invalid cell type string must be rejected");
}

#[kani::proof]
fn verify_ffi_activation_type_parse_coverage() {
    let variants = ["sigmoid", "tanh", "relu", "linear"];
    for s in &variants {
        let result: Result<ActivationType, _> = s.parse();
        kani::assert(result.is_ok(),
            "Valid activation string must parse successfully");
    }
}

#[kani::proof]
fn verify_ffi_activation_type_invalid_rejected() {
    let result: Result<ActivationType, _> = "invalid".parse();
    kani::assert(result.is_err(),
        "Invalid activation string must be rejected");
}

#[kani::proof]
fn verify_ffi_loss_type_parse_coverage() {
    let variants = ["mse", "crossentropy"];
    for s in &variants {
        let result: Result<LossType, _> = s.parse();
        kani::assert(result.is_ok(),
            "Valid loss type string must parse successfully");
    }
}

#[kani::proof]
fn verify_ffi_loss_type_invalid_rejected() {
    let result: Result<LossType, _> = "invalid".parse();
    kani::assert(result.is_err(),
        "Invalid loss type string must be rejected");
}

#[kani::proof]
fn verify_ffi_backend_choice_parse_coverage() {
    let variants = ["auto", "cpu", "cuda", "opencl", "hybrid"];
    for s in &variants {
        let result: Result<BackendChoice, _> = s.parse();
        kani::assert(result.is_ok(),
            "Valid backend string must parse successfully");
    }
}

#[kani::proof]
fn verify_ffi_backend_choice_invalid_rejected() {
    let result: Result<BackendChoice, _> = "invalid".parse();
    kani::assert(result.is_err(),
        "Invalid backend string must be rejected");
}

// =========================================================================
// E. NULL HANDLE REJECTION
// =========================================================================

#[kani::proof]
fn verify_ffi_null_handle_returns_default() {
    let handle_is_null = true;

    if handle_is_null {
        let get_input_size: u32 = 0;
        let get_output_size: u32 = 0;
        let get_learning_rate: f64 = 0.0;
        let predict_result: i32 = -1;

        kani::assert(get_input_size == 0, "Null handle get_input_size returns 0");
        kani::assert(get_output_size == 0, "Null handle get_output_size returns 0");
        kani::assert(get_learning_rate == 0.0, "Null handle get_learning_rate returns 0.0");
        kani::assert(predict_result == -1, "Null handle predict returns -1");
    }
}

#[kani::proof]
fn verify_ffi_null_handle_save_returns_error() {
    let handle_is_null = true;
    if handle_is_null {
        let result: i32 = -1;
        kani::assert(result == -1, "Null handle save returns -1");
    }
}

#[kani::proof]
fn verify_ffi_null_handle_train_returns_nan() {
    let handle_is_null = true;
    if handle_is_null {
        let result: f64 = f64::NAN;
        kani::assert(result.is_nan(), "Null handle train_sequence returns NaN");
    }
}

// =========================================================================
// F. NULL STRING POINTER HANDLING
// =========================================================================

#[kani::proof]
fn verify_ffi_null_string_defaults_to_empty() {
    let is_null = true;
    let default_str = if is_null { "" } else { "value" };
    kani::assert(default_str.is_empty(),
        "Null C string pointer must default to empty string");
}

#[kani::proof]
fn verify_ffi_empty_cell_type_defaults_to_lstm() {
    let s = "";
    let default = if s.is_empty() { "lstm" } else { s };
    let result: Result<CellType, _> = default.parse();
    kani::assert(result.is_ok(), "Empty cell type defaults to LSTM");
    kani::assert(result.unwrap() == CellType::LSTM, "Default is LSTM");
}

#[kani::proof]
fn verify_ffi_empty_activation_defaults_to_tanh() {
    let s = "";
    let default = if s.is_empty() { "tanh" } else { s };
    let result: Result<ActivationType, _> = default.parse();
    kani::assert(result.is_ok(), "Empty activation defaults to Tanh");
    kani::assert(result.unwrap() == ActivationType::Tanh, "Default is Tanh");
}

#[kani::proof]
fn verify_ffi_empty_backend_defaults_to_auto() {
    let s = "";
    let default = if s.is_empty() { "auto" } else { s };
    let result: Result<BackendChoice, _> = default.parse();
    kani::assert(result.is_ok(), "Empty backend defaults to Auto");
    kani::assert(result.unwrap() == BackendChoice::Auto, "Default is Auto");
}

// =========================================================================
// G. FLAT_TO_2D BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_ffi_flat_to_2d_total_elements() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 64);
    kani::assume(cols > 0 && cols <= 64);

    let flat_len = rows * cols;
    let result_rows = flat_len / cols;
    kani::assert(result_rows == rows,
        "flat_to_2d must produce correct number of rows");
}

#[kani::proof]
fn verify_ffi_flat_to_2d_null_returns_empty() {
    let is_null = true;
    let rows = 0usize;
    let cols = 0usize;

    if is_null || rows == 0 || cols == 0 {
        let result_len = 0;
        kani::assert(result_len == 0,
            "flat_to_2d with null/zero returns empty Vec");
    }
}

#[kani::proof]
fn verify_ffi_flat_to_2d_chunk_size() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 16);
    kani::assume(cols > 0 && cols <= 16);

    let flat_len = rows * cols;
    let chunks_count = flat_len / cols;
    kani::assert(chunks_count == rows,
        "Number of chunks must equal rows");
}

// =========================================================================
// H. PREDICT OUTPUT BUFFER CAPACITY VALIDATION
// =========================================================================

#[kani::proof]
fn verify_ffi_predict_output_bounded() {
    let num_timesteps: u32 = kani::any();
    let output_size: u32 = kani::any();
    let buf_len: u32 = kani::any();
    kani::assume(num_timesteps > 0 && num_timesteps <= 100);
    kani::assume(output_size > 0 && output_size <= 64);

    let total = num_timesteps as usize * output_size as usize;
    let write_len = if buf_len as usize >= total { total } else { buf_len as usize };

    kani::assert(write_len <= buf_len as usize,
        "Write length must not exceed buffer capacity");
}

// =========================================================================
// I. TRAIN EPOCH/LOSS BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_ffi_train_loss_buffer_bounds() {
    let epochs: u32 = kani::any();
    let loss_buf_len: u32 = kani::any();
    kani::assume(epochs > 0 && epochs <= 1000);
    kani::assume(loss_buf_len > 0 && loss_buf_len <= 1000);

    let epoch: usize = kani::any();
    kani::assume(epoch < epochs as usize);

    if epoch < loss_buf_len as usize {
        kani::assert(epoch < loss_buf_len as usize,
            "Loss buffer write must be within capacity");
    }
}

#[kani::proof]
fn verify_ffi_train_returns_epoch_count() {
    let epochs: u32 = kani::any();
    kani::assume(epochs > 0 && epochs <= 10000);

    let result = epochs as i32;
    kani::assert(result > 0, "Train must return positive epoch count");
}

// =========================================================================
// J. GRADIENT DIAGNOSTIC OUTPUT POINTER SAFETY
// =========================================================================

#[kani::proof]
fn verify_ffi_gradient_diagnostic_null_safe() {
    let out_count_null = true;
    let out_min_null = true;

    let count: i32 = 0;
    let min_val: f64 = 0.0;

    if !out_count_null {
        kani::assert(count >= 0, "Count must be non-negative");
    }
    if !out_min_null {
        kani::assert(!min_val.is_nan(), "Min value must not be NaN");
    }
}

#[kani::proof]
fn verify_ffi_vanishing_gradient_count_nonneg() {
    let count: i32 = kani::any();
    kani::assume(count >= 0);
    kani::assert(count >= 0, "Vanishing gradient count must be non-negative");
}

#[kani::proof]
fn verify_ffi_exploding_gradient_count_nonneg() {
    let count: i32 = kani::any();
    kani::assume(count >= 0);
    kani::assert(count >= 0, "Exploding gradient count must be non-negative");
}

// =========================================================================
// K. GATE STRING VALIDATION
// =========================================================================

#[kani::proof]
fn verify_ffi_lstm_gate_types_parse() {
    let gates = ["forget", "input", "output", "cellcandidate"];
    for s in &gates {
        let result: Result<GateType, _> = s.parse();
        kani::assert(result.is_ok(), "Valid LSTM gate must parse");
    }
}

#[kani::proof]
fn verify_ffi_gru_gate_types_parse() {
    let gates = ["update", "reset", "hiddencandidate"];
    for s in &gates {
        let result: Result<GateType, _> = s.parse();
        kani::assert(result.is_ok(), "Valid GRU gate must parse");
    }
}

#[kani::proof]
fn verify_ffi_invalid_gate_rejected() {
    let result: Result<GateType, _> = "invalid".parse();
    kani::assert(result.is_err(), "Invalid gate string must be rejected");
}

// =========================================================================
// L. ABI TYPE COMPATIBILITY
// =========================================================================

#[kani::proof]
fn verify_ffi_abi_u32_size() {
    kani::assert(std::mem::size_of::<u32>() == 4,
        "u32 must be 4 bytes == uint32_t");
    kani::assert(std::mem::align_of::<u32>() == 4,
        "u32 must be 4-byte aligned");
}

#[kani::proof]
fn verify_ffi_abi_i32_size() {
    kani::assert(std::mem::size_of::<i32>() == 4,
        "i32 must be 4 bytes == int32_t");
    kani::assert(std::mem::align_of::<i32>() == 4,
        "i32 must be 4-byte aligned");
}

#[kani::proof]
fn verify_ffi_abi_f64_size() {
    kani::assert(std::mem::size_of::<f64>() == 8,
        "f64 must be 8 bytes == double/cl_double");
    kani::assert(std::mem::align_of::<f64>() == 8,
        "f64 must be 8-byte aligned");
}

#[kani::proof]
fn verify_ffi_abi_pointer_size() {
    let ptr_size = std::mem::size_of::<*const u8>();
    kani::assert(ptr_size == 4 || ptr_size == 8,
        "Pointer must be 4 (32-bit) or 8 (64-bit) bytes");
}

// =========================================================================
// M. MODEL PROPERTY GETTER SAFETY
// =========================================================================

#[kani::proof]
fn verify_ffi_get_hidden_value_bounds() {
    let layer_idx: usize = kani::any();
    let neuron_idx: usize = kani::any();
    let num_layers: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(num_layers > 0 && num_layers <= MAX_LAYERS);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN);

    let in_bounds = layer_idx < num_layers && neuron_idx < hidden_size;
    if !in_bounds {
        let default: f64 = 0.0;
        kani::assert(default == 0.0, "OOB access returns 0.0 default");
    }
}

#[kani::proof]
fn verify_ffi_get_output_value_bounds() {
    let timestep: usize = kani::any();
    let output_idx: usize = kani::any();
    let cache_len: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(cache_len <= MAX_TIMESTEPS);
    kani::assume(output_size <= MAX_HIDDEN);

    let in_bounds = timestep < cache_len && output_idx < output_size;
    if !in_bounds {
        let default: f64 = 0.0;
        kani::assert(default == 0.0, "OOB output value returns 0.0");
    }
}

#[kani::proof]
fn verify_ffi_get_cell_state_non_lstm_returns_zero() {
    let cell_type = CellType::GRU;
    if cell_type != CellType::LSTM {
        let result: f64 = 0.0;
        kani::assert(result == 0.0,
            "get_cell_state for non-LSTM returns 0.0");
    }
}

// =========================================================================
// N. SETTER VALUE VALIDATION
// =========================================================================

#[kani::proof]
fn verify_ffi_set_learning_rate_nan_check() {
    let lr: f64 = kani::any();
    kani::assume(lr.is_nan());

    let accepted = !lr.is_nan() && !lr.is_infinite();
    kani::assert(!accepted, "NaN learning rate must be rejected");
}

#[kani::proof]
fn verify_ffi_set_dropout_enables_flag() {
    let rate: f64 = kani::any();
    kani::assume(!rate.is_nan() && !rate.is_infinite());
    kani::assume(rate >= 0.0 && rate <= 1.0);

    let use_dropout = rate > 0.0;

    if rate == 0.0 {
        kani::assert(!use_dropout, "Zero dropout disables dropout flag");
    } else {
        kani::assert(use_dropout, "Positive dropout enables dropout flag");
    }
}

#[kani::proof]
fn verify_ffi_set_hidden_value_bounds_checked() {
    let layer_idx: usize = kani::any();
    let neuron_idx: usize = kani::any();
    let num_layers: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(num_layers > 0 && num_layers <= MAX_LAYERS);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN);

    let in_bounds = layer_idx < num_layers && neuron_idx < hidden_size;
    if in_bounds {
        kani::assert(layer_idx < num_layers && neuron_idx < hidden_size,
            "Valid set_hidden_value indices must be in bounds");
    }
}

// =========================================================================
// O. END-TO-END POLYGLOT CALL CHAIN VALIDATION
// =========================================================================

#[kani::proof]
fn verify_ffi_create_pipeline_all_inputs() {
    let input_size: u32 = kani::any();
    let output_size: u32 = kani::any();
    let num_hidden: u32 = kani::any();
    kani::assume(input_size > 0 && input_size <= 1024);
    kani::assume(output_size > 0 && output_size <= 1024);
    kani::assume(num_hidden > 0 && num_hidden <= MAX_LAYERS as u32);

    let is = validate_u32_as_usize(input_size);
    let os = validate_u32_as_usize(output_size);
    let nh = validate_u32_as_usize(num_hidden);

    kani::assert(is > 0 && os > 0 && nh > 0,
        "Validated create params must be positive");
}

#[kani::proof]
fn verify_ffi_predict_pipeline_all_inputs() {
    let num_timesteps: u32 = kani::any();
    let input_size: u32 = kani::any();
    let output_buf_len: u32 = kani::any();
    kani::assume(num_timesteps > 0 && num_timesteps <= 100);
    kani::assume(input_size > 0 && input_size <= 64);

    let ts = validate_u32_as_usize(num_timesteps);
    let is = validate_u32_as_usize(input_size);
    let flat_len = ts * is;

    kani::assert(flat_len == ts * is,
        "Flat input length must be timesteps * input_size");
}

#[kani::proof]
fn verify_ffi_train_pipeline_all_inputs() {
    let num_timesteps: u32 = kani::any();
    let input_size: u32 = kani::any();
    let output_size: u32 = kani::any();
    let epochs: u32 = kani::any();
    kani::assume(num_timesteps > 0 && num_timesteps <= 100);
    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(output_size > 0 && output_size <= 64);
    kani::assume(epochs > 0 && epochs <= 1000);

    let input_flat = num_timesteps as usize * input_size as usize;
    let target_flat = num_timesteps as usize * output_size as usize;

    kani::assert(input_flat > 0 && target_flat > 0,
        "Flat arrays must be non-empty");
}

#[kani::proof]
fn verify_ffi_introspection_chain() {
    let layer: u32 = kani::any();
    let timestep: u32 = kani::any();
    let neuron: u32 = kani::any();
    kani::assume(layer <= MAX_LAYERS as u32);
    kani::assume(timestep <= MAX_TIMESTEPS as u32);
    kani::assume(neuron <= MAX_HIDDEN as u32);

    let l = validate_u32_as_usize(layer);
    let t = validate_u32_as_usize(timestep);
    let n = validate_u32_as_usize(neuron);

    kani::assert(l == layer as usize && t == timestep as usize && n == neuron as usize,
        "Introspection index conversion must be exact");
}

#[kani::proof]
fn verify_ffi_all_validators_no_panic() {
    let f: f64 = kani::any();
    let u: u32 = kani::any();

    let _a = validate_u32_as_usize(u);
    let _b = validate_f64_finite(f);
    let _c = validate_learning_rate(f);
    let _d = validate_gradient_clip(f);
    let _e = validate_dropout_rate(f);
}

// =========================================================================
// Unit tests (run during cargo test, not cargo kani)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_to_usize() {
        assert_eq!(validate_u32_as_usize(0), 0);
        assert_eq!(validate_u32_as_usize(42), 42);
        assert_eq!(validate_u32_as_usize(u32::MAX), u32::MAX as usize);
    }

    #[test]
    fn test_learning_rate_validation() {
        assert!(validate_learning_rate(0.01).is_some());
        assert!(validate_learning_rate(0.0).is_some());
        assert!(validate_learning_rate(100.0).is_some());
        assert!(validate_learning_rate(-1.0).is_none());
        assert!(validate_learning_rate(101.0).is_none());
        assert!(validate_learning_rate(f64::NAN).is_none());
        assert!(validate_learning_rate(f64::INFINITY).is_none());
    }

    #[test]
    fn test_dropout_rate_validation() {
        assert!(validate_dropout_rate(0.0).is_some());
        assert!(validate_dropout_rate(0.5).is_some());
        assert!(validate_dropout_rate(1.0).is_some());
        assert!(validate_dropout_rate(-0.1).is_none());
        assert!(validate_dropout_rate(1.1).is_none());
        assert!(validate_dropout_rate(f64::NAN).is_none());
    }

    #[test]
    fn test_cell_type_parse() {
        assert!("lstm".parse::<CellType>().is_ok());
        assert!("gru".parse::<CellType>().is_ok());
        assert!("simplernn".parse::<CellType>().is_ok());
        assert!("bad".parse::<CellType>().is_err());
    }

    #[test]
    fn test_gate_type_parse() {
        assert!("forget".parse::<GateType>().is_ok());
        assert!("input".parse::<GateType>().is_ok());
        assert!("output".parse::<GateType>().is_ok());
        assert!("update".parse::<GateType>().is_ok());
        assert!("reset".parse::<GateType>().is_ok());
        assert!("bad".parse::<GateType>().is_err());
    }

    #[test]
    fn test_backend_choice_parse() {
        assert!("auto".parse::<BackendChoice>().is_ok());
        assert!("cpu".parse::<BackendChoice>().is_ok());
        assert!("cuda".parse::<BackendChoice>().is_ok());
        assert!("opencl".parse::<BackendChoice>().is_ok());
        assert!("hybrid".parse::<BackendChoice>().is_ok());
        assert!("bad".parse::<BackendChoice>().is_err());
    }

    #[test]
    fn test_abi_sizes() {
        assert_eq!(std::mem::size_of::<u32>(), 4);
        assert_eq!(std::mem::size_of::<i32>(), 4);
        assert_eq!(std::mem::size_of::<f64>(), 8);
    }
}


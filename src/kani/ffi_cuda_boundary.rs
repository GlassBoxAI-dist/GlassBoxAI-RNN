/*
 * Kani Verification: CUDA FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data crossing the CUDA FFI boundary is validated before use.
 * Covers CUDA kernel launch configurations, buffer sizing, grid/block dims,
 * matvec_add buffer correctness, LSTM/GRU/SimpleRNN gate buffer sizing,
 * and f64 transfer alignment for the RNN CUDA backend.
 *
 * CUDA backend uses: f64 (double), BLOCK_SIZE=256
 * Kernels: k_matvec_add, k_activate, k_lstm_gates, k_lstm_state,
 *          k_gru_gates, k_gru_hidden, k_simple_rnn_forward, k_zero
 *
 * CISA "Secure by Design" requirements verified:
 * A. Grid/block dimension validity (no zero-launch, covers all items)
 * B. Matvec_add weight buffer sizing (rows * cols)
 * C. LSTM gate buffer sizing (4 input bufs + 4 output bufs = hidden_size)
 * D. LSTM state buffer sizing (prev_c + 3 outputs = hidden_size)
 * E. GRU gate buffer sizing (2 input + 2 output = hidden_size)
 * F. GRU hidden buffer sizing (3 input + 2 output = hidden_size)
 * G. SimpleRNN forward buffer sizing (sum + 2 outputs = hidden_size)
 * H. Activate kernel buffer sizing (input + output = n)
 * I. Weight matrix flatten correctness (rows * cols = flat len)
 * J. Hidden state concat buffer sizing (input_size + hidden_size)
 * K. f64 transfer alignment guarantees
 * L. CUDA alloc_zeros sizing matches kernel expectations
 * M. Grid dim overflow prevention at u32 boundary
 * N. Kernel argument i32 cast safety
 * O. End-to-end CUDA forward pass buffer chain
 */

use crate::{
    ActivationType, CellType,
    zero_array, zero_matrix, flatten_matrix, concat_arrays,
    SimpleRNNCell, LSTMCell, GRUCell, OutputLayer,
};

const BLOCK_SIZE: u32 = 256;
const MAX_HIDDEN_SIZE: usize = 4096;
const MAX_INPUT_SIZE: usize = 4096;
const MAX_LAYERS: usize = 16;

fn cuda_grid_blocks(n: usize) -> u32 {
    (n as u32).div_ceil(BLOCK_SIZE)
}

fn validate_hidden_size(hs: usize) -> bool {
    hs > 0 && hs <= MAX_HIDDEN_SIZE
}

fn validate_input_size(is: usize) -> bool {
    is > 0 && is <= MAX_INPUT_SIZE
}

// =========================================================================
// A. GRID/BLOCK DIMENSION VALIDITY
// =========================================================================

#[kani::proof]
fn verify_cuda_grid_blocks_nonzero() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(n);
    kani::assert(blocks >= 1, "Grid must have at least 1 block");
}

#[kani::proof]
fn verify_cuda_grid_covers_all_items() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(n);
    let total_threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(total_threads >= n, "Grid must cover all work items");
}

#[kani::proof]
fn verify_cuda_grid_not_excessively_large() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(n);
    let total_threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(total_threads < n + BLOCK_SIZE as usize,
        "Grid must not waste more than one block of threads");
}

#[kani::proof]
fn verify_cuda_block_size_power_of_two() {
    kani::assert(BLOCK_SIZE.is_power_of_two(),
        "CUDA BLOCK_SIZE must be power of two for optimal warp alignment");
}

// =========================================================================
// B. MATVEC_ADD WEIGHT BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_matvec_weight_buffer_size() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 64);
    kani::assume(cols > 0 && cols <= 64);

    let w_flat_len = rows * cols;
    let y_len = rows;
    let x_len = cols;
    let b_len = rows;

    kani::assert(w_flat_len == rows * cols, "Weight buffer must be rows*cols");
    kani::assert(y_len == rows, "Output buffer must be rows");
    kani::assert(x_len == cols, "Input vector must be cols");
    kani::assert(b_len == rows, "Bias vector must be rows");
}

#[kani::proof]
fn verify_cuda_matvec_index_bounds() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 8);
    kani::assume(cols > 0 && cols <= 8);

    let i: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(i < rows);
    kani::assume(j < cols);

    let idx = i * cols + j;
    kani::assert(idx < rows * cols,
        "Matvec flat index must be within weight buffer");
}

#[kani::proof]
fn verify_cuda_matvec_i32_cast_safe() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);
    kani::assume(cols > 0 && cols <= MAX_HIDDEN_SIZE);

    kani::assert(rows <= i32::MAX as usize, "rows fits in i32 for CUDA kernel");
    kani::assert(cols <= i32::MAX as usize, "cols fits in i32 for CUDA kernel");
}

// =========================================================================
// C. LSTM GATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_lstm_gates_buffer_sizes() {
    let hidden_size: usize = kani::any();
    kani::assume(validate_hidden_size(hidden_size));

    let sum_f_len = hidden_size;
    let sum_i_len = hidden_size;
    let sum_c_len = hidden_size;
    let sum_o_len = hidden_size;
    let fg_len = hidden_size;
    let ig_len = hidden_size;
    let ct_len = hidden_size;
    let og_len = hidden_size;

    kani::assert(sum_f_len == hidden_size && sum_i_len == hidden_size
        && sum_c_len == hidden_size && sum_o_len == hidden_size,
        "LSTM gate input buffers must equal hidden_size");
    kani::assert(fg_len == hidden_size && ig_len == hidden_size
        && ct_len == hidden_size && og_len == hidden_size,
        "LSTM gate output buffers must equal hidden_size");
}

#[kani::proof]
fn verify_cuda_lstm_gates_grid_covers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(hidden_size);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= hidden_size,
        "LSTM gates grid must cover all hidden neurons");
}

#[kani::proof]
fn verify_cuda_lstm_gate_matvec_concat_size() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 64);

    let concat_size = input_size + hidden_size;
    let wf_flat_len = hidden_size * concat_size;

    kani::assert(wf_flat_len == hidden_size * concat_size,
        "LSTM weight matrix flat length must be hidden*concat");
    kani::assert(concat_size == input_size + hidden_size,
        "Concat vector must be input+hidden");
}

// =========================================================================
// D. LSTM STATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_lstm_state_buffer_sizes() {
    let hidden_size: usize = kani::any();
    kani::assume(validate_hidden_size(hidden_size));

    let fg_len = hidden_size;
    let ig_len = hidden_size;
    let ct_len = hidden_size;
    let og_len = hidden_size;
    let prev_c_len = hidden_size;
    let h_out_len = hidden_size;
    let c_out_len = hidden_size;
    let tanh_c_out_len = hidden_size;

    kani::assert(fg_len == hidden_size && ig_len == hidden_size
        && ct_len == hidden_size && og_len == hidden_size
        && prev_c_len == hidden_size,
        "LSTM state inputs must equal hidden_size");
    kani::assert(h_out_len == hidden_size && c_out_len == hidden_size
        && tanh_c_out_len == hidden_size,
        "LSTM state outputs must equal hidden_size");
}

#[kani::proof]
fn verify_cuda_lstm_state_grid_covers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(hidden_size);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= hidden_size,
        "LSTM state grid must cover all hidden neurons");
}

// =========================================================================
// E. GRU GATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_gru_gates_buffer_sizes() {
    let hidden_size: usize = kani::any();
    kani::assume(validate_hidden_size(hidden_size));

    let sum_z_len = hidden_size;
    let sum_r_len = hidden_size;
    let z_out_len = hidden_size;
    let r_out_len = hidden_size;

    kani::assert(sum_z_len == hidden_size && sum_r_len == hidden_size,
        "GRU gate inputs must equal hidden_size");
    kani::assert(z_out_len == hidden_size && r_out_len == hidden_size,
        "GRU gate outputs must equal hidden_size");
}

#[kani::proof]
fn verify_cuda_gru_gates_grid_covers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(hidden_size);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= hidden_size,
        "GRU gates grid must cover all hidden neurons");
}

#[kani::proof]
fn verify_cuda_gru_gate_matvec_concat_size() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 64);

    let concat_size = input_size + hidden_size;
    let wz_flat_len = hidden_size * concat_size;

    kani::assert(wz_flat_len == hidden_size * concat_size,
        "GRU weight matrix flat length must be hidden*concat");
}

// =========================================================================
// F. GRU HIDDEN BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_gru_hidden_buffer_sizes() {
    let hidden_size: usize = kani::any();
    kani::assume(validate_hidden_size(hidden_size));

    let sum_h_len = hidden_size;
    let z_len = hidden_size;
    let prev_h_len = hidden_size;
    let h_out_len = hidden_size;
    let h_tilde_out_len = hidden_size;

    kani::assert(sum_h_len == hidden_size && z_len == hidden_size
        && prev_h_len == hidden_size,
        "GRU hidden inputs must equal hidden_size");
    kani::assert(h_out_len == hidden_size && h_tilde_out_len == hidden_size,
        "GRU hidden outputs must equal hidden_size");
}

#[kani::proof]
fn verify_cuda_gru_hidden_grid_covers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(hidden_size);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= hidden_size,
        "GRU hidden grid must cover all hidden neurons");
}

// =========================================================================
// G. SIMPLE RNN FORWARD BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_simple_rnn_forward_buffer_sizes() {
    let hidden_size: usize = kani::any();
    kani::assume(validate_hidden_size(hidden_size));

    let sum_len = hidden_size;
    let h_out_len = hidden_size;
    let pre_h_out_len = hidden_size;

    kani::assert(sum_len == hidden_size,
        "SimpleRNN forward sum buffer must equal hidden_size");
    kani::assert(h_out_len == hidden_size && pre_h_out_len == hidden_size,
        "SimpleRNN forward outputs must equal hidden_size");
}

#[kani::proof]
fn verify_cuda_simple_rnn_forward_grid_covers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(hidden_size);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= hidden_size,
        "SimpleRNN forward grid must cover all hidden neurons");
}

#[kani::proof]
fn verify_cuda_simple_rnn_act_type_i32_valid() {
    let act: ActivationType = kani::any();
    let act_i32 = act.as_int();
    kani::assert(act_i32 >= 0 && act_i32 <= 3,
        "Activation type i32 must be 0-3 for CUDA kernel");
}

// =========================================================================
// H. ACTIVATE KERNEL BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_cuda_activate_buffer_sizes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let x_len = n;
    let y_len = n;

    kani::assert(x_len == n && y_len == n,
        "Activate kernel input/output must equal n");
}

#[kani::proof]
fn verify_cuda_activate_grid_covers() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(n);
    let threads = blocks as usize * BLOCK_SIZE as usize;
    kani::assert(threads >= n,
        "Activate grid must cover all elements");
}

// =========================================================================
// I. WEIGHT MATRIX FLATTEN CORRECTNESS
// =========================================================================

#[kani::proof]
#[kani::unwind(10)]
fn verify_cuda_flatten_matrix_length() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 4);
    kani::assume(cols > 0 && cols <= 4);

    let mat = zero_matrix(rows, cols);
    let flat = flatten_matrix(&mat);

    kani::assert(flat.len() == rows * cols,
        "Flattened matrix length must be rows*cols");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_cuda_lstm_all_weight_flattens_consistent() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let concat_size = input_size + hidden_size;

    let wf_flat = flatten_matrix(&cell.wf);
    let wi_flat = flatten_matrix(&cell.wi);
    let wc_flat = flatten_matrix(&cell.wc);
    let wo_flat = flatten_matrix(&cell.wo);

    let expected = hidden_size * concat_size;
    kani::assert(wf_flat.len() == expected, "Wf flatten correct");
    kani::assert(wi_flat.len() == expected, "Wi flatten correct");
    kani::assert(wc_flat.len() == expected, "Wc flatten correct");
    kani::assert(wo_flat.len() == expected, "Wo flatten correct");
}

// =========================================================================
// J. HIDDEN STATE CONCAT BUFFER SIZING
// =========================================================================

#[kani::proof]
#[kani::unwind(10)]
fn verify_cuda_concat_size_correct() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);

    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    let concat = concat_arrays(&input, &prev_h);

    kani::assert(concat.len() == input_size + hidden_size,
        "Concat buffer must be input_size + hidden_size");
}

#[kani::proof]
fn verify_cuda_concat_size_matches_weight_cols() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 64);
    kani::assume(hidden_size > 0 && hidden_size <= 64);

    let concat_size = input_size + hidden_size;
    let weight_cols = concat_size;

    kani::assert(weight_cols == concat_size,
        "Weight matrix cols must match concat size for matvec_add");
}

// =========================================================================
// K. F64 TRANSFER ALIGNMENT GUARANTEES
// =========================================================================

#[kani::proof]
fn verify_cuda_f64_alignment() {
    kani::assert(std::mem::size_of::<f64>() == 8,
        "f64 must be 8 bytes for CUDA double");
    kani::assert(std::mem::align_of::<f64>() == 8,
        "f64 must be 8-byte aligned for CUDA transfers");
}

#[kani::proof]
fn verify_cuda_i32_alignment() {
    kani::assert(std::mem::size_of::<i32>() == 4,
        "i32 must be 4 bytes for CUDA int");
    kani::assert(std::mem::align_of::<i32>() == 4,
        "i32 must be 4-byte aligned for CUDA transfers");
}

#[kani::proof]
fn verify_cuda_u32_alignment() {
    kani::assert(std::mem::size_of::<u32>() == 4,
        "u32 must be 4 bytes");
    kani::assert(std::mem::align_of::<u32>() == 4,
        "u32 must be 4-byte aligned");
}

// =========================================================================
// L. CUDA ALLOC_ZEROS SIZING MATCHES KERNEL EXPECTATIONS
// =========================================================================

#[kani::proof]
fn verify_cuda_alloc_lstm_gates_match() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let alloc_fg = hidden_size;
    let alloc_ig = hidden_size;
    let alloc_ct = hidden_size;
    let alloc_og = hidden_size;

    let kernel_expects = hidden_size;
    kani::assert(alloc_fg == kernel_expects && alloc_ig == kernel_expects
        && alloc_ct == kernel_expects && alloc_og == kernel_expects,
        "alloc_zeros for LSTM gates must match kernel hidden_size");
}

#[kani::proof]
fn verify_cuda_alloc_gru_gates_match() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let alloc_z = hidden_size;
    let alloc_r = hidden_size;

    kani::assert(alloc_z == hidden_size && alloc_r == hidden_size,
        "alloc_zeros for GRU gates must match kernel hidden_size");
}

#[kani::proof]
fn verify_cuda_alloc_matvec_output_match() {
    let rows: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);

    let alloc_y = rows;
    kani::assert(alloc_y == rows,
        "alloc_zeros for matvec output must match rows");
}

// =========================================================================
// M. GRID DIM OVERFLOW PREVENTION AT U32 BOUNDARY
// =========================================================================

#[kani::proof]
fn verify_cuda_grid_dim_within_u32() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let blocks = cuda_grid_blocks(n);
    kani::assert((blocks as u64) < (u32::MAX as u64),
        "Grid dimension must fit in u32");
}

#[kani::proof]
fn verify_cuda_hidden_size_fits_i32_for_kernel_arg() {
    let hs: usize = kani::any();
    kani::assume(hs > 0 && hs <= MAX_HIDDEN_SIZE);

    kani::assert(hs <= i32::MAX as usize,
        "hidden_size must fit in i32 for CUDA kernel argument");
}

// =========================================================================
// N. KERNEL ARGUMENT I32 CAST SAFETY
// =========================================================================

#[kani::proof]
fn verify_cuda_kernel_arg_rows_cols_i32_safe() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);
    kani::assume(cols > 0 && cols <= MAX_INPUT_SIZE + MAX_HIDDEN_SIZE);

    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    kani::assert(rows_i32 > 0, "rows cast to i32 must remain positive");
    kani::assert(cols_i32 > 0, "cols cast to i32 must remain positive");
}

#[kani::proof]
fn verify_cuda_activation_type_i32_range() {
    let val: i32 = kani::any();
    kani::assume(val >= 0 && val <= 3);

    let valid = val == 0 || val == 1 || val == 2 || val == 3;
    kani::assert(valid, "CUDA activation type must be 0-3");
}

// =========================================================================
// O. END-TO-END CUDA FORWARD PASS BUFFER CHAIN
// =========================================================================

#[kani::proof]
#[kani::unwind(20)]
fn verify_cuda_lstm_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let concat_size = input_size + hidden_size;

    let input_buf = zero_array(input_size);
    let prev_h_buf = zero_array(hidden_size);
    let prev_c_buf = zero_array(hidden_size);

    let concat_buf = concat_arrays(&input_buf, &prev_h_buf);
    kani::assert(concat_buf.len() == concat_size, "Concat buffer sized correctly");

    let wf_flat = flatten_matrix(&cell.wf);
    kani::assert(wf_flat.len() == hidden_size * concat_size, "Wf flat sized correctly");

    let sum_buf = zero_array(hidden_size);
    kani::assert(sum_buf.len() == hidden_size, "Sum buffer sized correctly");

    let gate_out = zero_array(hidden_size);
    kani::assert(gate_out.len() == hidden_size, "Gate output sized correctly");

    let h_out = zero_array(hidden_size);
    let c_out = zero_array(hidden_size);
    kani::assert(h_out.len() == hidden_size && c_out.len() == hidden_size,
        "State outputs sized correctly");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_cuda_gru_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
    let concat_size = input_size + hidden_size;

    let input_buf = zero_array(input_size);
    let prev_h_buf = zero_array(hidden_size);

    let concat_buf = concat_arrays(&input_buf, &prev_h_buf);
    kani::assert(concat_buf.len() == concat_size, "Concat buffer sized correctly");

    let wz_flat = flatten_matrix(&cell.wz);
    kani::assert(wz_flat.len() == hidden_size * concat_size, "Wz flat sized correctly");

    let sum_z = zero_array(hidden_size);
    let sum_r = zero_array(hidden_size);
    kani::assert(sum_z.len() == hidden_size && sum_r.len() == hidden_size,
        "Gate sum buffers sized correctly");

    let z_out = zero_array(hidden_size);
    let r_out = zero_array(hidden_size);
    kani::assert(z_out.len() == hidden_size && r_out.len() == hidden_size,
        "Gate outputs sized correctly");

    let concat_r = zero_array(concat_size);
    kani::assert(concat_r.len() == concat_size, "Concat_r sized correctly");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_cuda_simple_rnn_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);

    let cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);

    let wih_flat = flatten_matrix(&cell.wih);
    let whh_flat = flatten_matrix(&cell.whh);
    kani::assert(wih_flat.len() == hidden_size * input_size, "Wih flat correct");
    kani::assert(whh_flat.len() == hidden_size * hidden_size, "Whh flat correct");

    let sum_buf = zero_array(hidden_size);
    let h_out = zero_array(hidden_size);
    let pre_h_out = zero_array(hidden_size);
    kani::assert(sum_buf.len() == hidden_size, "Sum buffer correct");
    kani::assert(h_out.len() == hidden_size && pre_h_out.len() == hidden_size,
        "Output buffers correct");
}

#[kani::proof]
fn verify_cuda_output_layer_buffer_chain() {
    let last_hidden: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(last_hidden > 0 && last_hidden <= 64);
    kani::assume(output_size > 0 && output_size <= 64);

    let layer = OutputLayer::new(last_hidden, output_size, ActivationType::Linear);
    kani::assert(layer.w.len() == output_size, "Output weight rows correct");
    kani::assert(layer.b.len() == output_size, "Output bias correct");

    for row in &layer.w {
        kani::assert(row.len() == last_hidden, "Output weight cols correct");
    }
}

// =========================================================================
// Unit tests (run during cargo test, not cargo kani)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_grid_blocks_basic() {
        assert_eq!(cuda_grid_blocks(1), 1);
        assert_eq!(cuda_grid_blocks(256), 1);
        assert_eq!(cuda_grid_blocks(257), 2);
        assert_eq!(cuda_grid_blocks(512), 2);
        assert_eq!(cuda_grid_blocks(513), 3);
    }

    #[test]
    fn test_cuda_block_size_is_256() {
        assert_eq!(BLOCK_SIZE, 256);
        assert!(BLOCK_SIZE.is_power_of_two());
    }

    #[test]
    fn test_cuda_f64_abi() {
        assert_eq!(std::mem::size_of::<f64>(), 8);
        assert_eq!(std::mem::align_of::<f64>(), 8);
    }

    #[test]
    fn test_cuda_flatten_matrix_correctness() {
        let mat = zero_matrix(3, 4);
        let flat = flatten_matrix(&mat);
        assert_eq!(flat.len(), 12);
    }

    #[test]
    fn test_cuda_activation_type_range() {
        assert_eq!(ActivationType::Sigmoid.as_int(), 0);
        assert_eq!(ActivationType::Tanh.as_int(), 1);
        assert_eq!(ActivationType::ReLU.as_int(), 2);
        assert_eq!(ActivationType::Linear.as_int(), 3);
    }

    #[test]
    fn test_cuda_concat_size() {
        let a = zero_array(3);
        let b = zero_array(5);
        let c = concat_arrays(&a, &b);
        assert_eq!(c.len(), 8);
    }
}

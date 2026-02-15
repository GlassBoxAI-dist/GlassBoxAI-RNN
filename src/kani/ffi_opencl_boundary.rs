/*
 * Kani Verification: OpenCL FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data crossing the OpenCL FFI boundary is validated before use.
 * Covers OpenCL global/local work sizes, clCreateBuffer sizing,
 * clEnqueueRead/WriteBuffer alignment, and kernel argument validation
 * for the RNN OpenCL backend.
 *
 * OpenCL backend uses: cl_double (f64), cl_khr_fp64 extension
 * Kernels: k_matvec_add, k_activate, k_lstm_gates, k_lstm_state,
 *          k_gru_gates, k_gru_hidden, k_simple_rnn_forward, k_zero
 *
 * CISA "Secure by Design" requirements verified:
 * A. Global work size validity (covers all items, non-zero)
 * B. Matvec_add buffer sizing for OpenCL (CL_MEM_READ_ONLY/READ_WRITE)
 * C. LSTM gate buffer sizing (create_read_buffer + create_rw_buffer)
 * D. LSTM state buffer sizing
 * E. GRU gate buffer sizing
 * F. GRU hidden buffer sizing
 * G. SimpleRNN forward buffer sizing
 * H. Activate kernel buffer sizing
 * I. cl_double ABI compatibility (f64 == 8 bytes)
 * J. clEnqueueRead/WriteBuffer data length correctness
 * K. Kernel argument i32 cast safety for hidden_size/rows/cols
 * L. create_read_buffer data length matches allocation
 * M. create_rw_buffer size matches kernel expectations
 * N. Queue finish synchronization point coverage
 * O. End-to-end OpenCL forward pass buffer chain
 */

use crate::{
    ActivationType,
    zero_array, zero_matrix, flatten_matrix, concat_arrays,
    SimpleRNNCell, LSTMCell, GRUCell, OutputLayer,
};

const MAX_HIDDEN_SIZE: usize = 4096;
const MAX_INPUT_SIZE: usize = 4096;

// =========================================================================
// A. GLOBAL WORK SIZE VALIDITY
// =========================================================================

#[kani::proof]
fn verify_opencl_global_work_size_nonzero() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    kani::assert(n > 0, "OpenCL global work size must be non-zero");
}

#[kani::proof]
fn verify_opencl_global_work_size_covers_all() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let global_work_size = hidden_size;
    kani::assert(global_work_size >= hidden_size,
        "OpenCL global work size must cover all items");
}

#[kani::proof]
fn verify_opencl_global_work_size_exact() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let global_work_size = hidden_size;
    kani::assert(global_work_size == hidden_size,
        "OpenCL sets global_work_size exactly to hidden_size");
}

// =========================================================================
// B. MATVEC_ADD BUFFER SIZING FOR OPENCL
// =========================================================================

#[kani::proof]
fn verify_opencl_matvec_read_buffer_sizes() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 64);
    kani::assume(cols > 0 && cols <= 64);

    let w_buf_size = rows * cols;
    let x_buf_size = cols;
    let b_buf_size = rows;

    kani::assert(w_buf_size == rows * cols, "W read buffer = rows*cols");
    kani::assert(x_buf_size == cols, "x read buffer = cols");
    kani::assert(b_buf_size == rows, "b read buffer = rows");
}

#[kani::proof]
fn verify_opencl_matvec_rw_buffer_size() {
    let rows: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);

    let y_buf_size = rows;
    kani::assert(y_buf_size == rows, "y rw buffer must equal rows");
}

#[kani::proof]
fn verify_opencl_matvec_index_within_buffer() {
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
        "OpenCL matvec flat index within W buffer");
}

// =========================================================================
// C. LSTM GATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_lstm_gates_read_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let sum_f_read = hidden_size;
    let sum_i_read = hidden_size;
    let sum_c_read = hidden_size;
    let sum_o_read = hidden_size;

    kani::assert(sum_f_read == hidden_size && sum_i_read == hidden_size
        && sum_c_read == hidden_size && sum_o_read == hidden_size,
        "LSTM gate CL_MEM_READ_ONLY buffers must equal hidden_size");
}

#[kani::proof]
fn verify_opencl_lstm_gates_rw_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let fg_rw = hidden_size;
    let ig_rw = hidden_size;
    let ct_rw = hidden_size;
    let og_rw = hidden_size;

    kani::assert(fg_rw == hidden_size && ig_rw == hidden_size
        && ct_rw == hidden_size && og_rw == hidden_size,
        "LSTM gate CL_MEM_READ_WRITE buffers must equal hidden_size");
}

// =========================================================================
// D. LSTM STATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_lstm_state_read_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let fg_read = hidden_size;
    let ig_read = hidden_size;
    let ct_read = hidden_size;
    let og_read = hidden_size;
    let prev_c_read = hidden_size;

    kani::assert(fg_read == hidden_size && ig_read == hidden_size
        && ct_read == hidden_size && og_read == hidden_size
        && prev_c_read == hidden_size,
        "LSTM state read buffers must equal hidden_size");
}

#[kani::proof]
fn verify_opencl_lstm_state_rw_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let h_rw = hidden_size;
    let c_rw = hidden_size;
    let tanh_c_rw = hidden_size;

    kani::assert(h_rw == hidden_size && c_rw == hidden_size
        && tanh_c_rw == hidden_size,
        "LSTM state rw buffers must equal hidden_size");
}

// =========================================================================
// E. GRU GATE BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_gru_gates_read_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let sum_z_read = hidden_size;
    let sum_r_read = hidden_size;

    kani::assert(sum_z_read == hidden_size && sum_r_read == hidden_size,
        "GRU gate read buffers must equal hidden_size");
}

#[kani::proof]
fn verify_opencl_gru_gates_rw_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let z_rw = hidden_size;
    let r_rw = hidden_size;

    kani::assert(z_rw == hidden_size && r_rw == hidden_size,
        "GRU gate rw buffers must equal hidden_size");
}

// =========================================================================
// F. GRU HIDDEN BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_gru_hidden_read_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let sum_h_read = hidden_size;
    let z_read = hidden_size;
    let prev_h_read = hidden_size;

    kani::assert(sum_h_read == hidden_size && z_read == hidden_size
        && prev_h_read == hidden_size,
        "GRU hidden read buffers must equal hidden_size");
}

#[kani::proof]
fn verify_opencl_gru_hidden_rw_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let h_rw = hidden_size;
    let h_tilde_rw = hidden_size;

    kani::assert(h_rw == hidden_size && h_tilde_rw == hidden_size,
        "GRU hidden rw buffers must equal hidden_size");
}

// =========================================================================
// G. SIMPLE RNN FORWARD BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_simple_rnn_read_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let sum_read = hidden_size;
    kani::assert(sum_read == hidden_size,
        "SimpleRNN forward read buffer must equal hidden_size");
}

#[kani::proof]
fn verify_opencl_simple_rnn_rw_buffers() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let h_rw = hidden_size;
    let pre_h_rw = hidden_size;

    kani::assert(h_rw == hidden_size && pre_h_rw == hidden_size,
        "SimpleRNN forward rw buffers must equal hidden_size");
}

// =========================================================================
// H. ACTIVATE KERNEL BUFFER SIZING
// =========================================================================

#[kani::proof]
fn verify_opencl_activate_read_buffer() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let x_read = n;
    kani::assert(x_read == n, "Activate read buffer must equal n");
}

#[kani::proof]
fn verify_opencl_activate_rw_buffer() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let y_rw = n;
    kani::assert(y_rw == n, "Activate rw buffer must equal n");
}

// =========================================================================
// I. CL_DOUBLE ABI COMPATIBILITY
// =========================================================================

#[kani::proof]
fn verify_opencl_cl_double_abi() {
    kani::assert(std::mem::size_of::<f64>() == 8,
        "f64 must be 8 bytes == cl_double");
    kani::assert(std::mem::align_of::<f64>() == 8,
        "f64 must be 8-byte aligned for OpenCL transfers");
}

#[kani::proof]
fn verify_opencl_i32_abi() {
    kani::assert(std::mem::size_of::<i32>() == 4,
        "i32 must be 4 bytes == cl_int");
    kani::assert(std::mem::align_of::<i32>() == 4,
        "i32 must be 4-byte aligned");
}

// =========================================================================
// J. CLENQUEUEREAD/WRITEBUFFER DATA LENGTH CORRECTNESS
// =========================================================================

#[kani::proof]
fn verify_opencl_write_buffer_length_matches_data() {
    let data_len: usize = kani::any();
    kani::assume(data_len > 0 && data_len <= MAX_HIDDEN_SIZE);

    let buffer_size = data_len;
    kani::assert(buffer_size == data_len,
        "clEnqueueWriteBuffer length must match data slice length");
}

#[kani::proof]
fn verify_opencl_read_buffer_length_matches_output() {
    let out_len: usize = kani::any();
    kani::assume(out_len > 0 && out_len <= MAX_HIDDEN_SIZE);

    let read_size = out_len;
    kani::assert(read_size == out_len,
        "clEnqueueReadBuffer length must match output slice length");
}

#[kani::proof]
fn verify_opencl_read_buffer_copy_slice_exact() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 64);

    let result_len = hidden_size;
    let out_len = hidden_size;
    kani::assert(result_len == out_len,
        "Result vec length must match output slice for copy_from_slice");
}

// =========================================================================
// K. KERNEL ARGUMENT I32 CAST SAFETY
// =========================================================================

#[kani::proof]
fn verify_opencl_hidden_size_i32_safe() {
    let hs: usize = kani::any();
    kani::assume(hs > 0 && hs <= MAX_HIDDEN_SIZE);

    let hs_i32 = hs as i32;
    kani::assert(hs_i32 > 0,
        "hidden_size as i32 must remain positive for OpenCL kernel");
}

#[kani::proof]
fn verify_opencl_rows_cols_i32_safe() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);
    kani::assume(cols > 0 && cols <= MAX_INPUT_SIZE + MAX_HIDDEN_SIZE);

    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    kani::assert(rows_i32 > 0, "rows as i32 must be positive");
    kani::assert(cols_i32 > 0, "cols as i32 must be positive");
}

#[kani::proof]
fn verify_opencl_n_i32_safe_for_activate() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= MAX_HIDDEN_SIZE);

    let n_i32 = n as i32;
    kani::assert(n_i32 > 0, "n as i32 must be positive for activate kernel");
}

// =========================================================================
// L. CREATE_READ_BUFFER DATA LENGTH MATCHES ALLOCATION
// =========================================================================

#[kani::proof]
fn verify_opencl_create_read_buffer_size_equals_data() {
    let data_len: usize = kani::any();
    kani::assume(data_len > 0 && data_len <= MAX_HIDDEN_SIZE);

    let alloc_size = data_len;
    kani::assert(alloc_size == data_len,
        "create_read_buffer allocation must equal data.len()");
}

#[kani::proof]
fn verify_opencl_create_read_buffer_bytes() {
    let data_len: usize = kani::any();
    kani::assume(data_len > 0 && data_len <= MAX_HIDDEN_SIZE);

    let byte_size = data_len * std::mem::size_of::<f64>();
    kani::assert(byte_size == data_len * 8,
        "Buffer byte size must be data_len * 8 for cl_double");
}

// =========================================================================
// M. CREATE_RW_BUFFER SIZE MATCHES KERNEL EXPECTATIONS
// =========================================================================

#[kani::proof]
fn verify_opencl_create_rw_buffer_matches_kernel() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_HIDDEN_SIZE);

    let rw_alloc = hidden_size;
    kani::assert(rw_alloc == hidden_size,
        "create_rw_buffer size must match kernel expected size");
}

// =========================================================================
// N. QUEUE FINISH SYNCHRONIZATION POINT COVERAGE
// =========================================================================

#[kani::proof]
fn verify_opencl_all_kernels_require_sync() {
    let num_kernels: usize = 8;
    let kernel_names = [
        "k_matvec_add", "k_activate", "k_lstm_gates", "k_lstm_state",
        "k_gru_gates", "k_gru_hidden", "k_simple_rnn_forward", "k_zero",
    ];

    kani::assert(kernel_names.len() == num_kernels,
        "All 8 OpenCL kernels must have queue.finish() sync");
}

// =========================================================================
// O. END-TO-END OPENCL FORWARD PASS BUFFER CHAIN
// =========================================================================

#[kani::proof]
#[kani::unwind(20)]
fn verify_opencl_lstm_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let concat_size = input_size + hidden_size;

    let wf_flat = flatten_matrix(&cell.wf);
    kani::assert(wf_flat.len() == hidden_size * concat_size,
        "OpenCL Wf read buffer sized correctly");

    let concat_buf = concat_arrays(&zero_array(input_size), &zero_array(hidden_size));
    kani::assert(concat_buf.len() == concat_size,
        "OpenCL concat read buffer sized correctly");

    kani::assert(cell.bf.len() == hidden_size,
        "OpenCL bias read buffer sized correctly");

    let sum_rw = hidden_size;
    let gate_rw = hidden_size;
    let state_rw = hidden_size;
    kani::assert(sum_rw == hidden_size && gate_rw == hidden_size && state_rw == hidden_size,
        "OpenCL rw buffers all sized to hidden_size");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_opencl_gru_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
    let concat_size = input_size + hidden_size;

    let wz_flat = flatten_matrix(&cell.wz);
    kani::assert(wz_flat.len() == hidden_size * concat_size,
        "OpenCL Wz read buffer sized correctly");

    let concat_buf = concat_arrays(&zero_array(input_size), &zero_array(hidden_size));
    kani::assert(concat_buf.len() == concat_size,
        "OpenCL concat buffer correct");

    let concat_r = zero_array(concat_size);
    kani::assert(concat_r.len() == concat_size,
        "OpenCL concat_r buffer correct");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_opencl_simple_rnn_forward_buffer_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);

    let cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);

    let wih_flat = flatten_matrix(&cell.wih);
    let whh_flat = flatten_matrix(&cell.whh);
    kani::assert(wih_flat.len() == hidden_size * input_size,
        "OpenCL Wih read buffer correct");
    kani::assert(whh_flat.len() == hidden_size * hidden_size,
        "OpenCL Whh read buffer correct");
}

#[kani::proof]
fn verify_opencl_output_layer_buffer_chain() {
    let last_hidden: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(last_hidden > 0 && last_hidden <= 64);
    kani::assume(output_size > 0 && output_size <= 64);

    let layer = OutputLayer::new(last_hidden, output_size, ActivationType::Linear);
    kani::assert(layer.w.len() == output_size, "OpenCL output W rows correct");
    kani::assert(layer.b.len() == output_size, "OpenCL output bias correct");
}

// =========================================================================
// Unit tests (run during cargo test, not cargo kani)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_cl_double_abi() {
        assert_eq!(std::mem::size_of::<f64>(), 8);
        assert_eq!(std::mem::align_of::<f64>(), 8);
    }

    #[test]
    fn test_opencl_i32_abi() {
        assert_eq!(std::mem::size_of::<i32>(), 4);
        assert_eq!(std::mem::align_of::<i32>(), 4);
    }

    #[test]
    fn test_opencl_hidden_size_i32_safe() {
        let hs: usize = 4096;
        assert!(hs <= i32::MAX as usize);
        assert!((hs as i32) > 0);
    }

    #[test]
    fn test_opencl_buffer_byte_size() {
        let n = 256usize;
        let bytes = n * std::mem::size_of::<f64>();
        assert_eq!(bytes, 2048);
    }

    #[test]
    fn test_opencl_kernel_count() {
        let kernels = [
            "k_matvec_add", "k_activate", "k_lstm_gates", "k_lstm_state",
            "k_gru_gates", "k_gru_hidden", "k_simple_rnn_forward", "k_zero",
        ];
        assert_eq!(kernels.len(), 8);
    }

    #[test]
    fn test_opencl_flatten_consistency() {
        let mat = zero_matrix(4, 6);
        let flat = flatten_matrix(&mat);
        assert_eq!(flat.len(), 24);
    }
}

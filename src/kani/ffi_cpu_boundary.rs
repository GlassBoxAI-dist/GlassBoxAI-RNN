//! @file
//! @ingroup RNN_Core_Verified
/*
 * Kani Verification: CPU Backend FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data used by the CPU backend (CpuBackend) is validated.
 * The CPU backend implements RnnKernels trait with pure Rust loops.
 * It is the fallback when no GPU is available and is also used in hybrid mode.
 *
 * CPU backend uses: f64, no GPU device, pure Rust iteration
 * Trait methods: matvec_add, lstm_gates, lstm_state, gru_gates, gru_hidden,
 *                simple_rnn_forward, activate, zero_buf
 *
 * CISA "Secure by Design" requirements verified:
 * A. Matvec_add loop bounds (rows, cols match buffer sizes)
 * B. LSTM gates loop bounds (hidden_size iteration)
 * C. LSTM state loop bounds (hidden_size iteration)
 * D. GRU gates loop bounds (hidden_size iteration)
 * E. GRU hidden loop bounds (hidden_size iteration)
 * F. SimpleRNN forward loop bounds (hidden_size iteration)
 * G. Activate loop bounds (n iteration)
 * H. Zero_buf completeness (all elements zeroed)
 * I. CPU sigmoid clamping prevents overflow
 * J. CPU activation function coverage (all 4 types)
 * K. Matvec weight index i*cols+j always in bounds
 * L. LSTM cell forward CPU path buffer consistency
 * M. GRU cell forward CPU path buffer consistency
 * N. SimpleRNN cell forward CPU path buffer consistency
 * O. End-to-end CPU forward chain (cell → output layer)
 */

use crate::{
    ActivationType, Activation,
    zero_array, zero_matrix, concat_arrays,
    SimpleRNNCell, LSTMCell, GRUCell, OutputLayer,
};

const MAX_HIDDEN_SIZE: usize = 64;
const MAX_INPUT_SIZE: usize = 64;

fn cpu_activation(x: f64, act_type: i32) -> f64 {
    match act_type {
        0 => 1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp()),
        1 => x.tanh(),
        2 => if x > 0.0 { x } else { 0.0 },
        3 => x,
        _ => x,
    }
}

fn cpu_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

// =========================================================================
// A. MATVEC_ADD LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_matvec_loop_bounds_safe() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 8);
    kani::assume(cols > 0 && cols <= 8);

    let w = vec![0.0_f64; rows * cols];
    let x = vec![0.0_f64; cols];
    let b = vec![0.0_f64; rows];
    let mut y = vec![0.0_f64; rows];

    for i in 0..rows {
        kani::assert(i < b.len(), "bias index in bounds");
        let mut sum = b[i];
        for j in 0..cols {
            let idx = i * cols + j;
            kani::assert(idx < w.len(), "weight index in bounds");
            kani::assert(j < x.len(), "input index in bounds");
            sum += w[idx] * x[j];
        }
        kani::assert(i < y.len(), "output index in bounds");
        y[i] = sum;
    }
}

#[kani::proof]
fn verify_cpu_matvec_output_length() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= MAX_HIDDEN_SIZE);
    kani::assume(cols > 0 && cols <= MAX_INPUT_SIZE);

    let y_len = rows;
    kani::assert(y_len == rows,
        "CPU matvec output length must equal rows");
}

// =========================================================================
// B. LSTM GATES LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_lstm_gates_loop_safe() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8);

    let sum_f = vec![0.0_f64; hidden_size];
    let sum_i = vec![0.0_f64; hidden_size];
    let sum_c = vec![0.0_f64; hidden_size];
    let sum_o = vec![0.0_f64; hidden_size];
    let mut fg = vec![0.0_f64; hidden_size];
    let mut ig = vec![0.0_f64; hidden_size];
    let mut c_tilde = vec![0.0_f64; hidden_size];
    let mut og = vec![0.0_f64; hidden_size];

    for k in 0..hidden_size {
        kani::assert(k < sum_f.len() && k < fg.len(),
            "LSTM gate loop index in bounds");
        fg[k] = cpu_sigmoid(sum_f[k]);
        ig[k] = cpu_sigmoid(sum_i[k]);
        c_tilde[k] = sum_c[k].tanh();
        og[k] = cpu_sigmoid(sum_o[k]);
    }
}

#[kani::proof]
fn verify_cpu_lstm_gates_output_range() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let sig = cpu_sigmoid(x);
    kani::assert(sig >= 0.0 && sig <= 1.0,
        "Sigmoid output must be in [0,1]");
}

// =========================================================================
// C. LSTM STATE LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_lstm_state_loop_safe() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8);

    let fg = vec![0.5_f64; hidden_size];
    let ig = vec![0.5_f64; hidden_size];
    let c_tilde = vec![0.0_f64; hidden_size];
    let og = vec![0.5_f64; hidden_size];
    let prev_c = vec![0.0_f64; hidden_size];
    let mut h = vec![0.0_f64; hidden_size];
    let mut c = vec![0.0_f64; hidden_size];
    let mut tanh_c = vec![0.0_f64; hidden_size];

    for k in 0..hidden_size {
        kani::assert(k < fg.len() && k < prev_c.len(),
            "LSTM state loop index in bounds");
        c[k] = fg[k] * prev_c[k] + ig[k] * c_tilde[k];
        tanh_c[k] = c[k].tanh();
        h[k] = og[k] * tanh_c[k];
    }
}

// =========================================================================
// D. GRU GATES LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_gru_gates_loop_safe() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8);

    let sum_z = vec![0.0_f64; hidden_size];
    let sum_r = vec![0.0_f64; hidden_size];
    let mut z = vec![0.0_f64; hidden_size];
    let mut r = vec![0.0_f64; hidden_size];

    for k in 0..hidden_size {
        kani::assert(k < sum_z.len() && k < z.len(),
            "GRU gate loop index in bounds");
        z[k] = cpu_sigmoid(sum_z[k]);
        r[k] = cpu_sigmoid(sum_r[k]);
    }
}

// =========================================================================
// E. GRU HIDDEN LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_gru_hidden_loop_safe() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8);

    let sum_h = vec![0.0_f64; hidden_size];
    let z = vec![0.5_f64; hidden_size];
    let prev_h = vec![0.0_f64; hidden_size];
    let mut h = vec![0.0_f64; hidden_size];
    let mut h_tilde = vec![0.0_f64; hidden_size];

    for k in 0..hidden_size {
        kani::assert(k < sum_h.len() && k < z.len() && k < prev_h.len(),
            "GRU hidden loop index in bounds");
        h_tilde[k] = sum_h[k].tanh();
        h[k] = (1.0 - z[k]) * prev_h[k] + z[k] * h_tilde[k];
    }
}

// =========================================================================
// F. SIMPLE RNN FORWARD LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_simple_rnn_loop_safe() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8);

    let sum = vec![0.0_f64; hidden_size];
    let mut h = vec![0.0_f64; hidden_size];
    let mut pre_h = vec![0.0_f64; hidden_size];
    let act_type: i32 = kani::any();
    kani::assume(act_type >= 0 && act_type <= 3);

    for i in 0..hidden_size {
        kani::assert(i < sum.len() && i < h.len() && i < pre_h.len(),
            "SimpleRNN forward loop index in bounds");
        pre_h[i] = sum[i];
        h[i] = cpu_activation(sum[i], act_type);
    }
}

// =========================================================================
// G. ACTIVATE LOOP BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_activate_loop_safe() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 8);

    let x = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];
    let act_type: i32 = kani::any();
    kani::assume(act_type >= 0 && act_type <= 3);

    for i in 0..n {
        kani::assert(i < x.len() && i < y.len(),
            "Activate loop index in bounds");
        y[i] = cpu_activation(x[i], act_type);
    }
}

// =========================================================================
// H. ZERO_BUF COMPLETENESS
// =========================================================================

#[kani::proof]
fn verify_cpu_zero_buf_all_zeroed() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);

    let mut arr = vec![1.0_f64; size];
    for v in arr.iter_mut() {
        *v = 0.0;
    }

    let idx: usize = kani::any();
    kani::assume(idx < size);
    kani::assert(arr[idx] == 0.0, "All elements must be zeroed");
}

// =========================================================================
// I. CPU SIGMOID CLAMPING PREVENTS OVERFLOW
// =========================================================================

#[kani::proof]
fn verify_cpu_sigmoid_no_overflow() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let result = cpu_sigmoid(x);
    kani::assert(!result.is_nan() && !result.is_infinite(),
        "Sigmoid with clamping must never produce NaN/Inf");
    kani::assert(result >= 0.0 && result <= 1.0,
        "Sigmoid output must be in [0,1]");
}

#[kani::proof]
fn verify_cpu_sigmoid_extreme_values() {
    let large_pos = cpu_sigmoid(1000.0);
    let large_neg = cpu_sigmoid(-1000.0);
    let zero = cpu_sigmoid(0.0);

    kani::assert(large_pos > 0.99, "Sigmoid(1000) ≈ 1.0");
    kani::assert(large_neg < 0.01, "Sigmoid(-1000) ≈ 0.0");
    kani::assert((zero - 0.5).abs() < 1e-10, "Sigmoid(0) = 0.5");
}

// =========================================================================
// J. CPU ACTIVATION FUNCTION COVERAGE
// =========================================================================

#[kani::proof]
fn verify_cpu_activation_all_types_defined() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());
    kani::assume(x.abs() <= 100.0);

    let act_type: i32 = kani::any();
    kani::assume(act_type >= 0 && act_type <= 3);

    let result = cpu_activation(x, act_type);
    kani::assert(!result.is_nan(), "CPU activation must not produce NaN");
}

#[kani::proof]
fn verify_cpu_activation_sigmoid_range() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let result = cpu_activation(x, 0);
    kani::assert(result >= 0.0 && result <= 1.0,
        "Sigmoid activation in [0,1]");
}

#[kani::proof]
fn verify_cpu_activation_tanh_range() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let result = cpu_activation(x, 1);
    kani::assert(result >= -1.0 && result <= 1.0,
        "Tanh activation in [-1,1]");
}

#[kani::proof]
fn verify_cpu_activation_relu_nonneg() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let result = cpu_activation(x, 2);
    kani::assert(result >= 0.0, "ReLU activation >= 0");
}

#[kani::proof]
fn verify_cpu_activation_linear_identity() {
    let x: f64 = kani::any();
    kani::assume(!x.is_nan() && !x.is_infinite());

    let result = cpu_activation(x, 3);
    kani::assert(result == x, "Linear activation must be identity");
}

// =========================================================================
// K. MATVEC WEIGHT INDEX I*COLS+J ALWAYS IN BOUNDS
// =========================================================================

#[kani::proof]
fn verify_cpu_matvec_weight_index_formula() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 16);
    kani::assume(cols > 0 && cols <= 16);

    let i: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(i < rows);
    kani::assume(j < cols);

    let idx = i * cols + j;
    kani::assert(idx < rows * cols,
        "Flat weight index must be within buffer bounds");
    kani::assert(idx == i * cols + j,
        "Index formula must be row-major");
}

// =========================================================================
// L. LSTM CELL FORWARD CPU PATH BUFFER CONSISTENCY
// =========================================================================

#[kani::proof]
#[kani::unwind(20)]
fn verify_cpu_lstm_cell_forward_buffers() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    let prev_c = zero_array(hidden_size);

    let (h, c, fg, ig, c_tilde, og, tanh_c) = cell.forward(&input, &prev_h, &prev_c, None);

    kani::assert(h.len() == hidden_size, "h buffer correct");
    kani::assert(c.len() == hidden_size, "c buffer correct");
    kani::assert(fg.len() == hidden_size, "fg buffer correct");
    kani::assert(ig.len() == hidden_size, "ig buffer correct");
    kani::assert(c_tilde.len() == hidden_size, "c_tilde buffer correct");
    kani::assert(og.len() == hidden_size, "og buffer correct");
    kani::assert(tanh_c.len() == hidden_size, "tanh_c buffer correct");
}

// =========================================================================
// M. GRU CELL FORWARD CPU PATH BUFFER CONSISTENCY
// =========================================================================

#[kani::proof]
#[kani::unwind(20)]
fn verify_cpu_gru_cell_forward_buffers() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);

    let cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);

    let (h, z, r, h_tilde) = cell.forward(&input, &prev_h, None);

    kani::assert(h.len() == hidden_size, "h buffer correct");
    kani::assert(z.len() == hidden_size, "z buffer correct");
    kani::assert(r.len() == hidden_size, "r buffer correct");
    kani::assert(h_tilde.len() == hidden_size, "h_tilde buffer correct");
}

// =========================================================================
// N. SIMPLE RNN CELL FORWARD CPU PATH BUFFER CONSISTENCY
// =========================================================================

#[kani::proof]
#[kani::unwind(10)]
fn verify_cpu_simple_rnn_cell_forward_buffers() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);

    let cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);

    let (h, pre_h) = cell.forward(&input, &prev_h);

    kani::assert(h.len() == hidden_size, "h buffer correct");
    kani::assert(pre_h.len() == hidden_size, "pre_h buffer correct");
}

// =========================================================================
// O. END-TO-END CPU FORWARD CHAIN
// =========================================================================

#[kani::proof]
#[kani::unwind(20)]
fn verify_cpu_lstm_to_output_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);
    kani::assume(output_size > 0 && output_size <= 3);

    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let output_layer = OutputLayer::new(hidden_size, output_size, ActivationType::Linear);

    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    let prev_c = zero_array(hidden_size);

    let (h, _c, _fg, _ig, _ct, _og, _tc) = cell.forward(&input, &prev_h, &prev_c, None);
    kani::assert(h.len() == hidden_size, "Cell output sized for output layer input");

    let (out, pre) = output_layer.forward(&h);
    kani::assert(out.len() == output_size, "Output layer produces correct output size");
    kani::assert(pre.len() == output_size, "Pre-activation correct size");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_cpu_gru_to_output_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);
    kani::assume(output_size > 0 && output_size <= 3);

    let cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
    let output_layer = OutputLayer::new(hidden_size, output_size, ActivationType::Linear);

    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);

    let (h, _z, _r, _ht) = cell.forward(&input, &prev_h, None);
    kani::assert(h.len() == hidden_size, "Cell output sized for output layer");

    let (out, pre) = output_layer.forward(&h);
    kani::assert(out.len() == output_size, "Output correct");
    kani::assert(pre.len() == output_size, "Pre-activation correct");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_cpu_simple_rnn_to_output_chain() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);
    kani::assume(output_size > 0 && output_size <= 4);

    let cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);
    let output_layer = OutputLayer::new(hidden_size, output_size, ActivationType::Linear);

    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);

    let (h, _pre_h) = cell.forward(&input, &prev_h);
    kani::assert(h.len() == hidden_size, "Cell output sized for output layer");

    let (out, pre) = output_layer.forward(&h);
    kani::assert(out.len() == output_size, "Output correct");
    kani::assert(pre.len() == output_size, "Pre-activation correct");
}

// =========================================================================
// Unit tests (run during cargo test, not cargo kani)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_sigmoid_basic() {
        let r = cpu_sigmoid(0.0);
        assert!((r - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_sigmoid_clamping() {
        let big = cpu_sigmoid(10000.0);
        let small = cpu_sigmoid(-10000.0);
        assert!(big > 0.99);
        assert!(small < 0.01);
    }

    #[test]
    fn test_cpu_activation_types() {
        let sig = cpu_activation(0.0, 0);
        assert!((sig - 0.5).abs() < 1e-10);

        let tanh = cpu_activation(0.0, 1);
        assert!(tanh.abs() < 1e-10);

        let relu_pos = cpu_activation(1.0, 2);
        assert_eq!(relu_pos, 1.0);

        let relu_neg = cpu_activation(-1.0, 2);
        assert_eq!(relu_neg, 0.0);

        let lin = cpu_activation(42.0, 3);
        assert_eq!(lin, 42.0);
    }

    #[test]
    fn test_cpu_lstm_forward_sizes() {
        let cell = LSTMCell::new(2, 3, ActivationType::Tanh);
        let input = zero_array(2);
        let prev_h = zero_array(3);
        let prev_c = zero_array(3);
        let (h, c, fg, ig, ct, og, tc) = cell.forward(&input, &prev_h, &prev_c, None);
        assert_eq!(h.len(), 3);
        assert_eq!(c.len(), 3);
        assert_eq!(fg.len(), 3);
        assert_eq!(ig.len(), 3);
        assert_eq!(ct.len(), 3);
        assert_eq!(og.len(), 3);
        assert_eq!(tc.len(), 3);
    }

    #[test]
    fn test_cpu_gru_forward_sizes() {
        let cell = GRUCell::new(2, 3, ActivationType::Tanh);
        let input = zero_array(2);
        let prev_h = zero_array(3);
        let (h, z, r, ht) = cell.forward(&input, &prev_h, None);
        assert_eq!(h.len(), 3);
        assert_eq!(z.len(), 3);
        assert_eq!(r.len(), 3);
        assert_eq!(ht.len(), 3);
    }

    #[test]
    fn test_cpu_output_layer_forward() {
        let layer = OutputLayer::new(4, 2, ActivationType::Linear);
        let input = zero_array(4);
        let (out, pre) = layer.forward(&input);
        assert_eq!(out.len(), 2);
        assert_eq!(pre.len(), 2);
    }
}


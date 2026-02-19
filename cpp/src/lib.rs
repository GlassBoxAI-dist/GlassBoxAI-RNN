//! @file
//! @ingroup RNN_Internal_Logic
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use facaded_rnn::{
    ActivationType, CellType, GateType, LossType, RNNFacade,
    backend::BackendChoice,
};

#[repr(C)]
pub struct RnnHandle {
    _private: [u8; 0],
}

fn box_to_handle(model: Box<RNNFacade>) -> *mut RnnHandle {
    Box::into_raw(model) as *mut RnnHandle
}

unsafe fn handle_to_ref<'a>(handle: *const RnnHandle) -> Option<&'a RNNFacade> {
    (handle as *const RNNFacade).as_ref()
}

unsafe fn handle_to_mut<'a>(handle: *mut RnnHandle) -> Option<&'a mut RNNFacade> {
    (handle as *mut RNNFacade).as_mut()
}

unsafe fn cstr_to_str<'a>(s: *const c_char) -> &'a str {
    if s.is_null() {
        return "";
    }
    CStr::from_ptr(s).to_str().unwrap_or("")
}

static LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);

fn set_error(msg: String) {
    if let Ok(mut guard) = LAST_ERROR.lock() {
        *guard = CString::new(msg).ok();
    }
}

#[no_mangle]
pub extern "C" fn rnn_last_error() -> *const c_char {
    match LAST_ERROR.lock() {
        Ok(guard) => match &*guard {
            Some(e) => e.as_ptr(),
            None => ptr::null(),
        },
        Err(_) => ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn rnn_clear_error() {
    if let Ok(mut guard) = LAST_ERROR.lock() {
        *guard = None;
    }
}

#[no_mangle]
pub extern "C" fn rnn_create(
    input_size: u32,
    hidden_sizes: *const u32,
    num_hidden_layers: u32,
    output_size: u32,
    cell_type: *const c_char,
    activation: *const c_char,
    output_activation: *const c_char,
    loss: *const c_char,
    learning_rate: f64,
    gradient_clip: f64,
    bptt_steps: u32,
    backend: *const c_char,
) -> *mut RnnHandle {
    let ct_str = unsafe { cstr_to_str(cell_type) };
    let act_str = unsafe { cstr_to_str(activation) };
    let out_act_str = unsafe { cstr_to_str(output_activation) };
    let loss_str = unsafe { cstr_to_str(loss) };
    let backend_str = unsafe { cstr_to_str(backend) };

    let ct: CellType = match if ct_str.is_empty() { "lstm" } else { ct_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };
    let act: ActivationType = match if act_str.is_empty() { "tanh" } else { act_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };
    let out_act: ActivationType = match if out_act_str.is_empty() { "linear" } else { out_act_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };
    let lt: LossType = match if loss_str.is_empty() { "mse" } else { loss_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };
    let bc: BackendChoice = match if backend_str.is_empty() { "auto" } else { backend_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };

    let hidden: Vec<usize> = if hidden_sizes.is_null() || num_hidden_layers == 0 {
        vec![32]
    } else {
        unsafe { slice::from_raw_parts(hidden_sizes, num_hidden_layers as usize) }
            .iter().map(|&h| h as usize).collect()
    };

    let mut model = RNNFacade::new(
        input_size as usize,
        hidden,
        output_size as usize,
        ct, act, out_act, lt,
        learning_rate, gradient_clip,
        bptt_steps as usize,
    );

    match facaded_rnn::select_backend_arc(bc) {
        Ok(b) => model.set_backend(b),
        Err(e) => { set_error(e.to_string()); return ptr::null_mut(); }
    }

    box_to_handle(Box::new(model))
}

#[no_mangle]
pub extern "C" fn rnn_load(filename: *const c_char, backend: *const c_char) -> *mut RnnHandle {
    let fname = unsafe { cstr_to_str(filename) };
    let backend_str = unsafe { cstr_to_str(backend) };

    let mut model = match RNNFacade::load_model(fname) {
        Ok(m) => m,
        Err(e) => { set_error(e.to_string()); return ptr::null_mut(); }
    };

    let bc: BackendChoice = match if backend_str.is_empty() { "auto" } else { backend_str }.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return ptr::null_mut(); }
    };

    match facaded_rnn::select_backend_arc(bc) {
        Ok(b) => model.set_backend(b),
        Err(e) => { set_error(e.to_string()); return ptr::null_mut(); }
    }

    box_to_handle(Box::new(model))
}

#[no_mangle]
pub extern "C" fn rnn_destroy(handle: *mut RnnHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut RNNFacade)); }
    }
}

#[no_mangle]
pub extern "C" fn rnn_save(handle: *const RnnHandle, filename: *const c_char) -> i32 {
    let model = match unsafe { handle_to_ref(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return -1; }
    };
    let fname = unsafe { cstr_to_str(filename) };
    match model.save_model(fname) {
        Ok(()) => 0,
        Err(e) => { set_error(e.to_string()); -1 }
    }
}

#[no_mangle]
pub extern "C" fn rnn_predict(
    handle: *mut RnnHandle,
    input_data: *const f64,
    num_timesteps: u32,
    input_size: u32,
    output_buf: *mut f64,
    output_buf_len: u32,
) -> i32 {
    let model = match unsafe { handle_to_mut(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return -1; }
    };

    let inputs = flat_to_2d(input_data, num_timesteps as usize, input_size as usize);
    let predictions = model.predict(&inputs);

    let total = predictions.iter().map(|r| r.len()).sum::<usize>();
    if output_buf.is_null() {
        return total as i32;
    }

    let mut idx = 0;
    for row in &predictions {
        for &v in row {
            if idx >= output_buf_len as usize { break; }
            unsafe { *output_buf.add(idx) = v; }
            idx += 1;
        }
    }

    total as i32
}

#[no_mangle]
pub extern "C" fn rnn_train_sequence(
    handle: *mut RnnHandle,
    input_data: *const f64,
    target_data: *const f64,
    num_timesteps: u32,
    input_size: u32,
    output_size: u32,
) -> f64 {
    let model = match unsafe { handle_to_mut(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return f64::NAN; }
    };

    let inputs = flat_to_2d(input_data, num_timesteps as usize, input_size as usize);
    let targets = flat_to_2d(target_data, num_timesteps as usize, output_size as usize);

    model.train_sequence(&inputs, &targets)
}

#[no_mangle]
pub extern "C" fn rnn_train(
    handle: *mut RnnHandle,
    input_data: *const f64,
    target_data: *const f64,
    num_timesteps: u32,
    input_size: u32,
    output_size: u32,
    epochs: u32,
    loss_buf: *mut f64,
    loss_buf_len: u32,
) -> i32 {
    let model = match unsafe { handle_to_mut(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return -1; }
    };

    let inputs = flat_to_2d(input_data, num_timesteps as usize, input_size as usize);
    let targets = flat_to_2d(target_data, num_timesteps as usize, output_size as usize);

    for epoch in 0..epochs as usize {
        let loss = model.train_sequence(&inputs, &targets);
        if !loss_buf.is_null() && epoch < loss_buf_len as usize {
            unsafe { *loss_buf.add(epoch) = loss; }
        }
    }

    epochs as i32
}

#[no_mangle]
pub extern "C" fn rnn_forward_sequence(
    handle: *mut RnnHandle,
    input_data: *const f64,
    num_timesteps: u32,
    input_size: u32,
    output_buf: *mut f64,
    output_buf_len: u32,
) -> i32 {
    let model = match unsafe { handle_to_mut(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return -1; }
    };

    let inputs = flat_to_2d(input_data, num_timesteps as usize, input_size as usize);
    let outputs = model.forward_sequence(&inputs);

    let total = outputs.iter().map(|r| r.len()).sum::<usize>();
    if output_buf.is_null() {
        return total as i32;
    }

    let mut idx = 0;
    for row in &outputs {
        for &v in row {
            if idx >= output_buf_len as usize { break; }
            unsafe { *output_buf.add(idx) = v; }
            idx += 1;
        }
    }

    total as i32
}

#[no_mangle]
pub extern "C" fn rnn_backward_sequence(
    handle: *mut RnnHandle,
    target_data: *const f64,
    num_timesteps: u32,
    output_size: u32,
) -> f64 {
    let model = match unsafe { handle_to_mut(handle) } {
        Some(m) => m,
        None => { set_error("Null handle".into()); return f64::NAN; }
    };

    let targets = flat_to_2d(target_data, num_timesteps as usize, output_size as usize);
    model.backward_sequence(&targets)
}

#[no_mangle]
pub extern "C" fn rnn_reset_states(handle: *mut RnnHandle) {
    if let Some(model) = unsafe { handle_to_mut(handle) } {
        model.reset_all_states();
    }
}

#[no_mangle]
pub extern "C" fn rnn_get_input_size(handle: *const RnnHandle) -> u32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| m.input_size as u32)
}

#[no_mangle]
pub extern "C" fn rnn_get_output_size(handle: *const RnnHandle) -> u32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| m.output_size as u32)
}

#[no_mangle]
pub extern "C" fn rnn_get_layer_count(handle: *const RnnHandle) -> u32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| m.get_layer_count() as u32)
}

#[no_mangle]
pub extern "C" fn rnn_get_hidden_size(handle: *const RnnHandle, layer: u32) -> u32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| m.get_hidden_size(layer as usize) as u32)
}

#[no_mangle]
pub extern "C" fn rnn_get_sequence_length(handle: *const RnnHandle) -> u32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| m.get_sequence_length() as u32)
}

#[no_mangle]
pub extern "C" fn rnn_get_learning_rate(handle: *const RnnHandle) -> f64 {
    unsafe { handle_to_ref(handle) }.map_or(0.0, |m| m.learning_rate)
}

#[no_mangle]
pub extern "C" fn rnn_set_learning_rate(handle: *mut RnnHandle, lr: f64) {
    if let Some(model) = unsafe { handle_to_mut(handle) } {
        model.learning_rate = lr;
    }
}

#[no_mangle]
pub extern "C" fn rnn_get_gradient_clip(handle: *const RnnHandle) -> f64 {
    unsafe { handle_to_ref(handle) }.map_or(0.0, |m| m.gradient_clip)
}

#[no_mangle]
pub extern "C" fn rnn_get_dropout_rate(handle: *const RnnHandle) -> f64 {
    unsafe { handle_to_ref(handle) }.map_or(0.0, |m| m.dropout_rate)
}

#[no_mangle]
pub extern "C" fn rnn_set_dropout_rate(handle: *mut RnnHandle, rate: f64) {
    if let Some(model) = unsafe { handle_to_mut(handle) } {
        model.dropout_rate = rate;
        model.use_dropout = rate > 0.0;
    }
}

#[no_mangle]
pub extern "C" fn rnn_is_gpu_available(handle: *const RnnHandle) -> i32 {
    unsafe { handle_to_ref(handle) }.map_or(0, |m| if m.is_gpu_available() { 1 } else { 0 })
}

#[no_mangle]
pub extern "C" fn rnn_get_hidden_value(handle: *const RnnHandle, layer: u32, timestep: u32, neuron: u32) -> f64 {
    unsafe { handle_to_ref(handle) }
        .map_or(0.0, |m| m.get_hidden_value(layer as usize, timestep as usize, neuron as usize))
}

#[no_mangle]
pub extern "C" fn rnn_set_hidden_value(handle: *mut RnnHandle, layer: u32, neuron: u32, value: f64) {
    if let Some(model) = unsafe { handle_to_mut(handle) } {
        model.set_hidden_value(layer as usize, neuron as usize, value);
    }
}

#[no_mangle]
pub extern "C" fn rnn_get_output_value(handle: *const RnnHandle, timestep: u32, index: u32) -> f64 {
    unsafe { handle_to_ref(handle) }
        .map_or(0.0, |m| m.get_output_value(timestep as usize, index as usize))
}

#[no_mangle]
pub extern "C" fn rnn_get_cell_state(handle: *const RnnHandle, layer: u32, neuron: u32) -> f64 {
    unsafe { handle_to_ref(handle) }
        .map_or(0.0, |m| m.get_cell_state(layer as usize, neuron as usize))
}

#[no_mangle]
pub extern "C" fn rnn_get_gate_value(
    handle: *const RnnHandle,
    layer: u32, timestep: u32, neuron: u32,
    gate: *const c_char,
) -> f64 {
    let model = match unsafe { handle_to_ref(handle) } {
        Some(m) => m,
        None => return 0.0,
    };
    let gate_str = unsafe { cstr_to_str(gate) };
    let gt: GateType = match gate_str.parse() {
        Ok(v) => v,
        Err(e) => { set_error(e); return 0.0; }
    };
    model.get_gate_value(layer as usize, timestep as usize, neuron as usize, gt)
}

#[no_mangle]
pub extern "C" fn rnn_get_preactivation(handle: *const RnnHandle, layer: u32, timestep: u32, neuron: u32) -> f64 {
    unsafe { handle_to_ref(handle) }
        .map_or(0.0, |m| m.get_preactivation(layer as usize, timestep as usize, neuron as usize))
}

#[no_mangle]
pub extern "C" fn rnn_get_input_value(handle: *const RnnHandle, timestep: u32, index: u32) -> f64 {
    unsafe { handle_to_ref(handle) }
        .map_or(0.0, |m| m.get_input_value(timestep as usize, index as usize))
}

#[no_mangle]
pub extern "C" fn rnn_get_sequence_outputs(
    handle: *const RnnHandle,
    output_buf: *mut f64,
    buf_len: u32,
) -> i32 {
    let model = match unsafe { handle_to_ref(handle) } {
        Some(m) => m,
        None => return -1,
    };
    let outputs = model.get_sequence_outputs();
    let total: usize = outputs.iter().map(|r| r.len()).sum();

    if output_buf.is_null() {
        return total as i32;
    }

    let mut idx = 0;
    for row in &outputs {
        for &v in row {
            if idx >= buf_len as usize { break; }
            unsafe { *output_buf.add(idx) = v; }
            idx += 1;
        }
    }
    total as i32
}

#[no_mangle]
pub extern "C" fn rnn_get_sequence_hidden_states(
    handle: *const RnnHandle,
    layer: u32,
    output_buf: *mut f64,
    buf_len: u32,
) -> i32 {
    let model = match unsafe { handle_to_ref(handle) } {
        Some(m) => m,
        None => return -1,
    };
    let states = model.get_sequence_hidden_states(layer as usize);
    let total: usize = states.iter().map(|r| r.len()).sum();

    if output_buf.is_null() {
        return total as i32;
    }

    let mut idx = 0;
    for row in &states {
        for &v in row {
            if idx >= buf_len as usize { break; }
            unsafe { *output_buf.add(idx) = v; }
            idx += 1;
        }
    }
    total as i32
}

#[no_mangle]
pub extern "C" fn rnn_detect_vanishing_gradients(
    handle: *const RnnHandle,
    threshold: f64,
    out_count: *mut i32,
    out_min: *mut f64,
) {
    if let Some(model) = unsafe { handle_to_ref(handle) } {
        let (count, min_val) = model.detect_vanishing_gradients(threshold);
        if !out_count.is_null() { unsafe { *out_count = count; } }
        if !out_min.is_null() { unsafe { *out_min = min_val; } }
    }
}

#[no_mangle]
pub extern "C" fn rnn_detect_exploding_gradients(
    handle: *const RnnHandle,
    threshold: f64,
    out_count: *mut i32,
    out_max: *mut f64,
) {
    if let Some(model) = unsafe { handle_to_ref(handle) } {
        let (count, max_val) = model.detect_exploding_gradients(threshold);
        if !out_count.is_null() { unsafe { *out_count = count; } }
        if !out_max.is_null() { unsafe { *out_max = max_val; } }
    }
}

fn flat_to_2d(data: *const f64, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    if data.is_null() || rows == 0 || cols == 0 {
        return Vec::new();
    }
    let flat = unsafe { slice::from_raw_parts(data, rows * cols) };
    flat.chunks(cols).map(|c| c.to_vec()).collect()
}


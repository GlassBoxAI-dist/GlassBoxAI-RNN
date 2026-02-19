//! @file
//! @ingroup RNN_Internal_Logic
/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    OpenCl,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::Cpu => write!(f, "CPU"),
            BackendKind::Cuda => write!(f, "CUDA"),
            BackendKind::OpenCl => write!(f, "OpenCL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Cpu,
    Cuda,
    OpenCl,
    Hybrid,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendChoice::Auto => write!(f, "auto"),
            BackendChoice::Cpu => write!(f, "cpu"),
            BackendChoice::Cuda => write!(f, "cuda"),
            BackendChoice::OpenCl => write!(f, "opencl"),
            BackendChoice::Hybrid => write!(f, "hybrid"),
        }
    }
}

impl std::str::FromStr for BackendChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(BackendChoice::Auto),
            "cpu" => Ok(BackendChoice::Cpu),
            "cuda" => Ok(BackendChoice::Cuda),
            "opencl" => Ok(BackendChoice::OpenCl),
            "hybrid" => Ok(BackendChoice::Hybrid),
            _ => Err(format!("Unknown backend: {}. Valid: auto, cpu, cuda, opencl, hybrid", s)),
        }
    }
}

pub trait RnnKernels: Send + Sync {
    fn kind(&self) -> BackendKind;

    fn matvec_add(&self, y: &mut [f64], w: &[f64], x: &[f64], b: &[f64], rows: usize, cols: usize);

    fn lstm_gates(
        &self,
        fg: &mut [f64], ig: &mut [f64], c_tilde: &mut [f64], og: &mut [f64],
        sum_f: &[f64], sum_i: &[f64], sum_c: &[f64], sum_o: &[f64],
        hidden_size: usize,
    );

    fn lstm_state(
        &self,
        h: &mut [f64], c: &mut [f64], tanh_c: &mut [f64],
        fg: &[f64], ig: &[f64], c_tilde: &[f64], og: &[f64], prev_c: &[f64],
        hidden_size: usize,
    );

    fn gru_gates(
        &self,
        z: &mut [f64], r: &mut [f64],
        sum_z: &[f64], sum_r: &[f64],
        hidden_size: usize,
    );

    fn gru_hidden(
        &self,
        h: &mut [f64], h_tilde: &mut [f64],
        sum_h: &[f64], z: &[f64], prev_h: &[f64],
        hidden_size: usize,
    );

    fn simple_rnn_forward(
        &self,
        h: &mut [f64], pre_h: &mut [f64],
        sum: &[f64], hidden_size: usize, act_type: i32,
    );

    fn activate(&self, y: &mut [f64], x: &[f64], n: usize, act_type: i32);

    fn zero_buf(&self, arr: &mut [f64]);
}

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

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl RnnKernels for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn matvec_add(&self, y: &mut [f64], w: &[f64], x: &[f64], b: &[f64], rows: usize, cols: usize) {
        for i in 0..rows {
            let mut sum = b[i];
            for j in 0..cols {
                sum += w[i * cols + j] * x[j];
            }
            y[i] = sum;
        }
    }

    fn lstm_gates(
        &self,
        fg: &mut [f64], ig: &mut [f64], c_tilde: &mut [f64], og: &mut [f64],
        sum_f: &[f64], sum_i: &[f64], sum_c: &[f64], sum_o: &[f64],
        hidden_size: usize,
    ) {
        for k in 0..hidden_size {
            fg[k] = cpu_sigmoid(sum_f[k]);
            ig[k] = cpu_sigmoid(sum_i[k]);
            c_tilde[k] = sum_c[k].tanh();
            og[k] = cpu_sigmoid(sum_o[k]);
        }
    }

    fn lstm_state(
        &self,
        h: &mut [f64], c: &mut [f64], tanh_c: &mut [f64],
        fg: &[f64], ig: &[f64], c_tilde: &[f64], og: &[f64], prev_c: &[f64],
        hidden_size: usize,
    ) {
        for k in 0..hidden_size {
            c[k] = fg[k] * prev_c[k] + ig[k] * c_tilde[k];
            tanh_c[k] = c[k].tanh();
            h[k] = og[k] * tanh_c[k];
        }
    }

    fn gru_gates(
        &self,
        z: &mut [f64], r: &mut [f64],
        sum_z: &[f64], sum_r: &[f64],
        hidden_size: usize,
    ) {
        for k in 0..hidden_size {
            z[k] = cpu_sigmoid(sum_z[k]);
            r[k] = cpu_sigmoid(sum_r[k]);
        }
    }

    fn gru_hidden(
        &self,
        h: &mut [f64], h_tilde: &mut [f64],
        sum_h: &[f64], z: &[f64], prev_h: &[f64],
        hidden_size: usize,
    ) {
        for k in 0..hidden_size {
            h_tilde[k] = sum_h[k].tanh();
            h[k] = (1.0 - z[k]) * prev_h[k] + z[k] * h_tilde[k];
        }
    }

    fn simple_rnn_forward(
        &self,
        h: &mut [f64], pre_h: &mut [f64],
        sum: &[f64], hidden_size: usize, act_type: i32,
    ) {
        for i in 0..hidden_size {
            pre_h[i] = sum[i];
            h[i] = cpu_activation(sum[i], act_type);
        }
    }

    fn activate(&self, y: &mut [f64], x: &[f64], n: usize, act_type: i32) {
        for i in 0..n {
            y[i] = cpu_activation(x[i], act_type);
        }
    }

    fn zero_buf(&self, arr: &mut [f64]) {
        for v in arr.iter_mut() {
            *v = 0.0;
        }
    }
}

// ========== CUDA Backend ==========
#[cfg(feature = "cuda")]
pub mod cuda_backend {
    use super::*;
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

    const BLOCK_SIZE: u32 = 256;

    pub const CUDA_KERNEL_SRC: &str = r#"
extern "C" {

__device__ double d_sigmoid(double x) {
    double clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
}

__device__ double d_tanh_act(double x) {
    return tanh(x);
}

__device__ double d_relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double d_activation(double x, int actType) {
    switch (actType) {
        case 0: return d_sigmoid(x);
        case 1: return d_tanh_act(x);
        case 2: return d_relu(x);
        case 3: return x;
        default: return x;
    }
}

__device__ double d_activation_derivative(double y, int actType) {
    switch (actType) {
        case 0: return y * (1.0 - y);
        case 1: return 1.0 - y * y;
        case 2: return y > 0 ? 1.0 : 0.0;
        case 3: return 1.0;
        default: return 1.0;
    }
}

__device__ double d_clip(double v, double maxVal) {
    if (v > maxVal) return maxVal;
    else if (v < -maxVal) return -maxVal;
    else return v;
}

__global__ void k_matvec_add(double* y, const double* W, const double* x, const double* b,
                              int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void k_activate(double* y, const double* x, int n, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = d_activation(x[i], actType);
    }
}

__global__ void k_lstm_gates(double* Fg, double* Ig, double* CTilde, double* Og,
                              const double* SumF, const double* SumI,
                              const double* SumC, const double* SumO, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Fg[k] = d_sigmoid(SumF[k]);
        Ig[k] = d_sigmoid(SumI[k]);
        CTilde[k] = tanh(SumC[k]);
        Og[k] = d_sigmoid(SumO[k]);
    }
}

__global__ void k_lstm_state(double* H, double* C, double* TanhC,
                              const double* Fg, const double* Ig, const double* CTilde,
                              const double* Og, const double* PrevC, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        C[k] = Fg[k] * PrevC[k] + Ig[k] * CTilde[k];
        TanhC[k] = tanh(C[k]);
        H[k] = Og[k] * TanhC[k];
    }
}

__global__ void k_gru_gates(double* Z, double* R, const double* SumZ, const double* SumR, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Z[k] = d_sigmoid(SumZ[k]);
        R[k] = d_sigmoid(SumR[k]);
    }
}

__global__ void k_gru_hidden(double* H, double* HTilde, const double* SumH,
                              const double* Z, const double* PrevH, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        HTilde[k] = tanh(SumH[k]);
        H[k] = (1.0 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
    }
}

__global__ void k_simple_rnn_forward(double* H, double* PreH, const double* Sum, int hiddenSize, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        PreH[i] = Sum[i];
        H[i] = d_activation(Sum[i], actType);
    }
}

__global__ void k_zero(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = 0.0;
    }
}

}
"#;

    pub struct CudaBackend {
        pub device: Arc<CudaDevice>,
    }

    impl CudaBackend {
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let device = CudaDevice::new(0)?;
            let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
            device.load_ptx(ptx, "rnn_kernels", &[
                "k_matvec_add",
                "k_activate",
                "k_lstm_gates",
                "k_lstm_state",
                "k_gru_gates",
                "k_gru_hidden",
                "k_simple_rnn_forward",
                "k_zero",
            ])?;
            Ok(Self { device })
        }

        fn launch_cfg(&self, n: usize) -> LaunchConfig {
            let blocks = (n as u32).div_ceil(BLOCK_SIZE);
            LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            }
        }
    }

    impl RnnKernels for CudaBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::Cuda
        }

        fn matvec_add(&self, y: &mut [f64], w: &[f64], x: &[f64], b: &[f64], rows: usize, cols: usize) {
            let d_w = self.device.htod_sync_copy(w).unwrap();
            let d_x = self.device.htod_sync_copy(x).unwrap();
            let d_b = self.device.htod_sync_copy(b).unwrap();
            let d_y: CudaSlice<f64> = self.device.alloc_zeros(rows).unwrap();

            let cfg = self.launch_cfg(rows);
            let f = self.device.get_func("rnn_kernels", "k_matvec_add").unwrap();
            unsafe {
                f.launch(cfg, (&d_y, &d_w, &d_x, &d_b, rows as i32, cols as i32)).unwrap();
            }
            self.device.synchronize().unwrap();
            let result = self.device.dtoh_sync_copy(&d_y).unwrap();
            y[..rows].copy_from_slice(&result);
        }

        fn lstm_gates(
            &self,
            fg: &mut [f64], ig: &mut [f64], c_tilde: &mut [f64], og: &mut [f64],
            sum_f: &[f64], sum_i: &[f64], sum_c: &[f64], sum_o: &[f64],
            hidden_size: usize,
        ) {
            let d_sum_f = self.device.htod_sync_copy(sum_f).unwrap();
            let d_sum_i = self.device.htod_sync_copy(sum_i).unwrap();
            let d_sum_c = self.device.htod_sync_copy(sum_c).unwrap();
            let d_sum_o = self.device.htod_sync_copy(sum_o).unwrap();

            let d_fg: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_ig: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_ct: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_og: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();

            let cfg = self.launch_cfg(hidden_size);
            let f = self.device.get_func("rnn_kernels", "k_lstm_gates").unwrap();
            unsafe {
                f.launch(cfg, (&d_fg, &d_ig, &d_ct, &d_og, &d_sum_f, &d_sum_i, &d_sum_c, &d_sum_o, hidden_size as i32)).unwrap();
            }
            self.device.synchronize().unwrap();

            fg.copy_from_slice(&self.device.dtoh_sync_copy(&d_fg).unwrap());
            ig.copy_from_slice(&self.device.dtoh_sync_copy(&d_ig).unwrap());
            c_tilde.copy_from_slice(&self.device.dtoh_sync_copy(&d_ct).unwrap());
            og.copy_from_slice(&self.device.dtoh_sync_copy(&d_og).unwrap());
        }

        fn lstm_state(
            &self,
            h: &mut [f64], c: &mut [f64], tanh_c: &mut [f64],
            fg: &[f64], ig: &[f64], c_tilde: &[f64], og: &[f64], prev_c: &[f64],
            hidden_size: usize,
        ) {
            let d_fg = self.device.htod_sync_copy(fg).unwrap();
            let d_ig = self.device.htod_sync_copy(ig).unwrap();
            let d_ct = self.device.htod_sync_copy(c_tilde).unwrap();
            let d_og = self.device.htod_sync_copy(og).unwrap();
            let d_prev_c = self.device.htod_sync_copy(prev_c).unwrap();

            let d_h: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_c: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_tc: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();

            let cfg = self.launch_cfg(hidden_size);
            let f = self.device.get_func("rnn_kernels", "k_lstm_state").unwrap();
            unsafe {
                f.launch(cfg, (&d_h, &d_c, &d_tc, &d_fg, &d_ig, &d_ct, &d_og, &d_prev_c, hidden_size as i32)).unwrap();
            }
            self.device.synchronize().unwrap();

            h.copy_from_slice(&self.device.dtoh_sync_copy(&d_h).unwrap());
            c.copy_from_slice(&self.device.dtoh_sync_copy(&d_c).unwrap());
            tanh_c.copy_from_slice(&self.device.dtoh_sync_copy(&d_tc).unwrap());
        }

        fn gru_gates(
            &self,
            z: &mut [f64], r: &mut [f64],
            sum_z: &[f64], sum_r: &[f64],
            hidden_size: usize,
        ) {
            let d_sz = self.device.htod_sync_copy(sum_z).unwrap();
            let d_sr = self.device.htod_sync_copy(sum_r).unwrap();

            let d_z: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_r: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();

            let cfg = self.launch_cfg(hidden_size);
            let f = self.device.get_func("rnn_kernels", "k_gru_gates").unwrap();
            unsafe {
                f.launch(cfg, (&d_z, &d_r, &d_sz, &d_sr, hidden_size as i32)).unwrap();
            }
            self.device.synchronize().unwrap();

            z.copy_from_slice(&self.device.dtoh_sync_copy(&d_z).unwrap());
            r.copy_from_slice(&self.device.dtoh_sync_copy(&d_r).unwrap());
        }

        fn gru_hidden(
            &self,
            h: &mut [f64], h_tilde: &mut [f64],
            sum_h: &[f64], z: &[f64], prev_h: &[f64],
            hidden_size: usize,
        ) {
            let d_sh = self.device.htod_sync_copy(sum_h).unwrap();
            let d_z = self.device.htod_sync_copy(z).unwrap();
            let d_ph = self.device.htod_sync_copy(prev_h).unwrap();

            let d_h: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_ht: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();

            let cfg = self.launch_cfg(hidden_size);
            let f = self.device.get_func("rnn_kernels", "k_gru_hidden").unwrap();
            unsafe {
                f.launch(cfg, (&d_h, &d_ht, &d_sh, &d_z, &d_ph, hidden_size as i32)).unwrap();
            }
            self.device.synchronize().unwrap();

            h.copy_from_slice(&self.device.dtoh_sync_copy(&d_h).unwrap());
            h_tilde.copy_from_slice(&self.device.dtoh_sync_copy(&d_ht).unwrap());
        }

        fn simple_rnn_forward(
            &self,
            h: &mut [f64], pre_h: &mut [f64],
            sum: &[f64], hidden_size: usize, act_type: i32,
        ) {
            let d_sum = self.device.htod_sync_copy(sum).unwrap();
            let d_h: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();
            let d_ph: CudaSlice<f64> = self.device.alloc_zeros(hidden_size).unwrap();

            let cfg = self.launch_cfg(hidden_size);
            let f = self.device.get_func("rnn_kernels", "k_simple_rnn_forward").unwrap();
            unsafe {
                f.launch(cfg, (&d_h, &d_ph, &d_sum, hidden_size as i32, act_type)).unwrap();
            }
            self.device.synchronize().unwrap();

            h.copy_from_slice(&self.device.dtoh_sync_copy(&d_h).unwrap());
            pre_h.copy_from_slice(&self.device.dtoh_sync_copy(&d_ph).unwrap());
        }

        fn activate(&self, y: &mut [f64], x: &[f64], n: usize, act_type: i32) {
            let d_x = self.device.htod_sync_copy(x).unwrap();
            let d_y: CudaSlice<f64> = self.device.alloc_zeros(n).unwrap();

            let cfg = self.launch_cfg(n);
            let f = self.device.get_func("rnn_kernels", "k_activate").unwrap();
            unsafe {
                f.launch(cfg, (&d_y, &d_x, n as i32, act_type)).unwrap();
            }
            self.device.synchronize().unwrap();

            y[..n].copy_from_slice(&self.device.dtoh_sync_copy(&d_y).unwrap());
        }

        fn zero_buf(&self, arr: &mut [f64]) {
            for v in arr.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

// ========== OpenCL Backend ==========
#[cfg(feature = "opencl")]
pub mod opencl_backend {
    use super::*;
    use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
    use opencl3::context::Context;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
    use opencl3::kernel::{Kernel, ExecuteKernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
    use opencl3::program::Program;
    use opencl3::types::{CL_BLOCKING, cl_double};
    use std::ptr;

    const OPENCL_KERNEL_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double d_sigmoid(double x) {
    double clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
}

double d_tanh_act(double x) {
    return tanh(x);
}

double d_relu(double x) {
    return x > 0 ? x : 0;
}

double d_activation(double x, int actType) {
    switch (actType) {
        case 0: return d_sigmoid(x);
        case 1: return d_tanh_act(x);
        case 2: return d_relu(x);
        case 3: return x;
        default: return x;
    }
}

__kernel void k_matvec_add(__global double* y, __global const double* W, __global const double* x,
                           __global const double* b, int rows, int cols) {
    int i = get_global_id(0);
    if (i < rows) {
        double sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

__kernel void k_activate(__global double* y, __global const double* x, int n, int actType) {
    int i = get_global_id(0);
    if (i < n) {
        y[i] = d_activation(x[i], actType);
    }
}

__kernel void k_lstm_gates(__global double* Fg, __global double* Ig, __global double* CTilde,
                           __global double* Og, __global const double* SumF,
                           __global const double* SumI, __global const double* SumC,
                           __global const double* SumO, int hiddenSize) {
    int k = get_global_id(0);
    if (k < hiddenSize) {
        Fg[k] = d_sigmoid(SumF[k]);
        Ig[k] = d_sigmoid(SumI[k]);
        CTilde[k] = tanh(SumC[k]);
        Og[k] = d_sigmoid(SumO[k]);
    }
}

__kernel void k_lstm_state(__global double* H, __global double* C, __global double* TanhC,
                           __global const double* Fg, __global const double* Ig,
                           __global const double* CTilde, __global const double* Og,
                           __global const double* PrevC, int hiddenSize) {
    int k = get_global_id(0);
    if (k < hiddenSize) {
        C[k] = Fg[k] * PrevC[k] + Ig[k] * CTilde[k];
        TanhC[k] = tanh(C[k]);
        H[k] = Og[k] * TanhC[k];
    }
}

__kernel void k_gru_gates(__global double* Z, __global double* R,
                          __global const double* SumZ, __global const double* SumR,
                          int hiddenSize) {
    int k = get_global_id(0);
    if (k < hiddenSize) {
        Z[k] = d_sigmoid(SumZ[k]);
        R[k] = d_sigmoid(SumR[k]);
    }
}

__kernel void k_gru_hidden(__global double* H, __global double* HTilde,
                           __global const double* SumH, __global const double* Z,
                           __global const double* PrevH, int hiddenSize) {
    int k = get_global_id(0);
    if (k < hiddenSize) {
        HTilde[k] = tanh(SumH[k]);
        H[k] = (1.0 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
    }
}

__kernel void k_simple_rnn_forward(__global double* H, __global double* PreH,
                                   __global const double* Sum, int hiddenSize, int actType) {
    int i = get_global_id(0);
    if (i < hiddenSize) {
        PreH[i] = Sum[i];
        H[i] = d_activation(Sum[i], actType);
    }
}

__kernel void k_zero(__global double* arr, int n) {
    int i = get_global_id(0);
    if (i < n) {
        arr[i] = 0.0;
    }
}
"#;

    pub struct OpenClBackend {
        context: Context,
        queue: CommandQueue,
        program: Program,
    }

    unsafe impl Send for OpenClBackend {}
    unsafe impl Sync for OpenClBackend {}

    impl OpenClBackend {
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
            if device_ids.is_empty() {
                let device_ids = get_all_devices(CL_DEVICE_TYPE_ALL)?;
                if device_ids.is_empty() {
                    return Err("No OpenCL devices found".into());
                }
                return Self::from_device_id(device_ids[0]);
            }
            Self::from_device_id(device_ids[0])
        }

        fn from_device_id(device_id: opencl3::types::cl_device_id) -> Result<Self, Box<dyn std::error::Error>> {
            let device = Device::new(device_id);

            let extensions = device.extensions().unwrap_or_default();
            if !extensions.contains("cl_khr_fp64") {
                return Err("OpenCL device does not support cl_khr_fp64 (double precision)".into());
            }

            let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            println!("OpenCL Device: {}", device_name);

            let context = Context::from_device(&device)?;

            let queue = CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

            let program = Program::create_and_build_from_source(&context, OPENCL_KERNEL_SRC, "")?;

            Ok(Self { context, queue, program })
        }

        fn create_read_buffer(&self, data: &[f64]) -> Buffer<cl_double> {
            let n = data.len();
            let mut buf = unsafe {
                Buffer::<cl_double>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                    .expect("Failed to create OpenCL read buffer")
            };
            unsafe {
                self.queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])
                    .expect("Failed to write OpenCL buffer");
            }
            buf
        }

        fn create_rw_buffer(&self, n: usize) -> Buffer<cl_double> {
            unsafe {
                Buffer::<cl_double>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                    .expect("Failed to create OpenCL rw buffer")
            }
        }

        fn read_buffer(&self, buf: &Buffer<cl_double>, out: &mut [f64]) {
            let n = out.len();
            let mut result = vec![0.0_f64; n];
            unsafe {
                self.queue.enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut result, &[])
                    .expect("Failed to read OpenCL buffer");
            }
            out.copy_from_slice(&result);
        }
    }

    impl RnnKernels for OpenClBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::OpenCl
        }

        fn matvec_add(&self, y: &mut [f64], w: &[f64], x: &[f64], b: &[f64], rows: usize, cols: usize) {
            let d_w = self.create_read_buffer(w);
            let d_x = self.create_read_buffer(x);
            let d_b = self.create_read_buffer(b);
            let d_y = self.create_rw_buffer(rows);

            let kernel = Kernel::create(&self.program, "k_matvec_add").unwrap();
            let rows_i = rows as i32;
            let cols_i = cols as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_y)
                    .set_arg(&d_w)
                    .set_arg(&d_x)
                    .set_arg(&d_b)
                    .set_arg(&rows_i)
                    .set_arg(&cols_i)
                    .set_global_work_size(rows)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();
            self.read_buffer(&d_y, &mut y[..rows]);
        }

        fn lstm_gates(
            &self,
            fg: &mut [f64], ig: &mut [f64], c_tilde: &mut [f64], og: &mut [f64],
            sum_f: &[f64], sum_i: &[f64], sum_c: &[f64], sum_o: &[f64],
            hidden_size: usize,
        ) {
            let d_sf = self.create_read_buffer(sum_f);
            let d_si = self.create_read_buffer(sum_i);
            let d_sc = self.create_read_buffer(sum_c);
            let d_so = self.create_read_buffer(sum_o);

            let d_fg = self.create_rw_buffer(hidden_size);
            let d_ig = self.create_rw_buffer(hidden_size);
            let d_ct = self.create_rw_buffer(hidden_size);
            let d_og = self.create_rw_buffer(hidden_size);

            let kernel = Kernel::create(&self.program, "k_lstm_gates").unwrap();
            let hs = hidden_size as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_fg)
                    .set_arg(&d_ig)
                    .set_arg(&d_ct)
                    .set_arg(&d_og)
                    .set_arg(&d_sf)
                    .set_arg(&d_si)
                    .set_arg(&d_sc)
                    .set_arg(&d_so)
                    .set_arg(&hs)
                    .set_global_work_size(hidden_size)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_fg, fg);
            self.read_buffer(&d_ig, ig);
            self.read_buffer(&d_ct, c_tilde);
            self.read_buffer(&d_og, og);
        }

        fn lstm_state(
            &self,
            h: &mut [f64], c: &mut [f64], tanh_c: &mut [f64],
            fg: &[f64], ig: &[f64], c_tilde: &[f64], og: &[f64], prev_c: &[f64],
            hidden_size: usize,
        ) {
            let d_fg = self.create_read_buffer(fg);
            let d_ig = self.create_read_buffer(ig);
            let d_ct = self.create_read_buffer(c_tilde);
            let d_og = self.create_read_buffer(og);
            let d_pc = self.create_read_buffer(prev_c);

            let d_h = self.create_rw_buffer(hidden_size);
            let d_c = self.create_rw_buffer(hidden_size);
            let d_tc = self.create_rw_buffer(hidden_size);

            let kernel = Kernel::create(&self.program, "k_lstm_state").unwrap();
            let hs = hidden_size as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_h)
                    .set_arg(&d_c)
                    .set_arg(&d_tc)
                    .set_arg(&d_fg)
                    .set_arg(&d_ig)
                    .set_arg(&d_ct)
                    .set_arg(&d_og)
                    .set_arg(&d_pc)
                    .set_arg(&hs)
                    .set_global_work_size(hidden_size)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_h, h);
            self.read_buffer(&d_c, c);
            self.read_buffer(&d_tc, tanh_c);
        }

        fn gru_gates(
            &self,
            z: &mut [f64], r: &mut [f64],
            sum_z: &[f64], sum_r: &[f64],
            hidden_size: usize,
        ) {
            let d_sz = self.create_read_buffer(sum_z);
            let d_sr = self.create_read_buffer(sum_r);

            let d_z = self.create_rw_buffer(hidden_size);
            let d_r = self.create_rw_buffer(hidden_size);

            let kernel = Kernel::create(&self.program, "k_gru_gates").unwrap();
            let hs = hidden_size as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_z)
                    .set_arg(&d_r)
                    .set_arg(&d_sz)
                    .set_arg(&d_sr)
                    .set_arg(&hs)
                    .set_global_work_size(hidden_size)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_z, z);
            self.read_buffer(&d_r, r);
        }

        fn gru_hidden(
            &self,
            h: &mut [f64], h_tilde: &mut [f64],
            sum_h: &[f64], z: &[f64], prev_h: &[f64],
            hidden_size: usize,
        ) {
            let d_sh = self.create_read_buffer(sum_h);
            let d_z = self.create_read_buffer(z);
            let d_ph = self.create_read_buffer(prev_h);

            let d_h = self.create_rw_buffer(hidden_size);
            let d_ht = self.create_rw_buffer(hidden_size);

            let kernel = Kernel::create(&self.program, "k_gru_hidden").unwrap();
            let hs = hidden_size as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_h)
                    .set_arg(&d_ht)
                    .set_arg(&d_sh)
                    .set_arg(&d_z)
                    .set_arg(&d_ph)
                    .set_arg(&hs)
                    .set_global_work_size(hidden_size)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_h, h);
            self.read_buffer(&d_ht, h_tilde);
        }

        fn simple_rnn_forward(
            &self,
            h: &mut [f64], pre_h: &mut [f64],
            sum: &[f64], hidden_size: usize, act_type: i32,
        ) {
            let d_sum = self.create_read_buffer(sum);
            let d_h = self.create_rw_buffer(hidden_size);
            let d_ph = self.create_rw_buffer(hidden_size);

            let kernel = Kernel::create(&self.program, "k_simple_rnn_forward").unwrap();
            let hs = hidden_size as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_h)
                    .set_arg(&d_ph)
                    .set_arg(&d_sum)
                    .set_arg(&hs)
                    .set_arg(&act_type)
                    .set_global_work_size(hidden_size)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_h, h);
            self.read_buffer(&d_ph, pre_h);
        }

        fn activate(&self, y: &mut [f64], x: &[f64], n: usize, act_type: i32) {
            let d_x = self.create_read_buffer(x);
            let d_y = self.create_rw_buffer(n);

            let kernel = Kernel::create(&self.program, "k_activate").unwrap();
            let n_i = n as i32;
            unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_y)
                    .set_arg(&d_x)
                    .set_arg(&n_i)
                    .set_arg(&act_type)
                    .set_global_work_size(n)
                    .enqueue_nd_range(&self.queue)
                    .unwrap();
            }
            self.queue.finish().unwrap();

            self.read_buffer(&d_y, &mut y[..n]);
        }

        fn zero_buf(&self, arr: &mut [f64]) {
            for v in arr.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

pub fn select_backend(choice: BackendChoice) -> Result<Box<dyn RnnKernels>, Box<dyn std::error::Error>> {
    match choice {
        BackendChoice::Auto => {
            #[cfg(feature = "cuda")]
            {
                match cuda_backend::CudaBackend::new() {
                    Ok(b) => {
                        println!("Auto-detected CUDA backend");
                        return Ok(Box::new(b));
                    }
                    Err(e) => {
                        eprintln!("CUDA not available ({}), trying OpenCL...", e);
                    }
                }
            }
            #[cfg(feature = "opencl")]
            {
                match opencl_backend::OpenClBackend::new() {
                    Ok(b) => {
                        println!("Auto-detected OpenCL backend");
                        return Ok(Box::new(b));
                    }
                    Err(e) => {
                        eprintln!("OpenCL not available ({}), falling back to CPU", e);
                    }
                }
            }
            println!("Using CPU backend");
            Ok(Box::new(CpuBackend::new()))
        }
        BackendChoice::Cpu => {
            println!("Using CPU backend");
            Ok(Box::new(CpuBackend::new()))
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let b = cuda_backend::CudaBackend::new()?;
                println!("Using CUDA backend");
                return Ok(Box::new(b));
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA support not compiled in. Rebuild with --features cuda".into())
            }
        }
        BackendChoice::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                let b = opencl_backend::OpenClBackend::new()?;
                println!("Using OpenCL backend");
                return Ok(Box::new(b));
            }
            #[cfg(not(feature = "opencl"))]
            {
                Err("OpenCL support not compiled in. Rebuild with --features opencl".into())
            }
        }
        BackendChoice::Hybrid => {
            #[cfg(feature = "cuda")]
            {
                match cuda_backend::CudaBackend::new() {
                    Ok(b) => {
                        println!("Hybrid mode: using CUDA for GPU operations, CPU for remainder");
                        return Ok(Box::new(b));
                    }
                    Err(_) => {}
                }
            }
            #[cfg(feature = "opencl")]
            {
                match opencl_backend::OpenClBackend::new() {
                    Ok(b) => {
                        println!("Hybrid mode: using OpenCL for GPU operations, CPU for remainder");
                        return Ok(Box::new(b));
                    }
                    Err(_) => {}
                }
            }
            println!("No GPU available for hybrid mode, using CPU backend");
            Ok(Box::new(CpuBackend::new()))
        }
    }
}


# GlassBoxAI-RNN

## **GPU-Accelerated Recurrent Neural Network**

### *Multi-Language RNN with CUDA/OpenCL/CPU Support and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-red.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2.svg)](https://julialang.org/)
[![C#](https://img.shields.io/badge/C%23-.NET%206+-512BD4.svg)](https://dotnet.microsoft.com/)
[![Zig](https://img.shields.io/badge/Zig-0.11+-F7A41D.svg)](https://ziglang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-RNN is a production-ready, GPU-accelerated Recurrent Neural Network implementation featuring:

- **Triple compute backends**: CUDA for NVIDIA GPUs, OpenCL for AMD/Intel/cross-platform GPUs, and CPU fallback
- **Multi-language bindings**: Native support for Rust, Python, Node.js, C, C++, C#, Julia, Go, and Zig
- **RNN cell types**: SimpleRNN, LSTM, and GRU with full forward/backward support
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: 160+ Kani proof harnesses across 19 verification categories
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [Language Bindings](#language-bindings)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
   - [Node.js API](#nodejs-api)
   - [C API](#c-api)
   - [C++ API](#c-api-1)
   - [C# API](#c-api-2)
   - [Julia API](#julia-api)
   - [Go API](#go-api)
   - [Zig API](#zig-api)
7. [CLI Reference](#cli-reference)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **RNN Cell Types** | SimpleRNN, LSTM, and GRU with configurable stacked layers |
| **Activation Functions** | Sigmoid, Tanh, ReLU, Linear |
| **Loss Functions** | MSE, Cross-Entropy |
| **Training** | Backpropagation Through Time (BPTT) with configurable truncation |
| **Gradient Clipping** | Configurable gradient clip threshold to prevent exploding gradients |
| **Dropout** | Regularization support during training |
| **Model Persistence** | JSON serialization for model save/load |
| **Gradient Diagnostics** | Vanishing and exploding gradient detection |
| **Full Introspection** | Access hidden states, cell states, gate values, pre-activations per timestep |

### Compute Backends

| Backend | Platform | Features |
|---------|----------|----------|
| **CUDA** | NVIDIA GPUs | GPU-accelerated matvec, LSTM/GRU/SimpleRNN gate kernels |
| **OpenCL** | AMD, Intel, NVIDIA | Cross-platform GPU acceleration via cl_khr_fp64 |
| **CPU** | All platforms | Pure Rust fallback, no GPU required |
| **Hybrid** | Auto-detect | Attempts GPU, falls back to CPU automatically |

### Multi-Language Support

| Language | Binding Technology | Status |
|----------|-------------------|--------|
| **Rust** | Native | ✓ Full API |
| **Python** | PyO3 | ✓ Full API |
| **Node.js** | napi-rs | ✓ Full API |
| **C** | FFI | ✓ Full API |
| **C++** | FFI + RAII Wrapper | ✓ Full API |
| **C#** | P/Invoke (.NET) | ✓ Full API |
| **Julia** | ccall | ✓ Full API |
| **Go** | cgo | ✓ Full API |
| **Zig** | @cImport | ✓ Full API |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | 160+ Kani proof harnesses across 19 categories |
| **CUDA Boundary Safety** | Verified grid/block dims, buffer sizing, f64 alignment |
| **OpenCL Boundary Safety** | Verified global work sizes, CL buffer sizing, cl_double ABI |
| **CPU Backend Safety** | Verified loop bounds, sigmoid clamping, activation coverage |
| **Polyglot FFI Safety** | Verified u32/f64 validation, enum parsing, null handle safety |
| **CISA/NSA Compliance** | Secure by Design principles throughout |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GlassBoxAI-RNN                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Rust Core Library                            │   │
│  │              (src/lib.rs + src/backend.rs)                      │   │
│  │  • CUDA/OpenCL/CPU Kernels  • LSTM/GRU/SimpleRNN Cells         │   │
│  │  • BPTT Training  • Gradient Diagnostics  • JSON I/O           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────┼────────────────────────────────────┐   │
│  │                    Language Bindings                             │   │
│  ├────────┬────────┬──────┴──┬────────┬──────┬──────┬──────┬──────┤   │
│  │ Python │ Node.js│  C/C++  │   C#   │ Julia│  Go  │  Zig │  CLI │   │
│  │ (PyO3) │(napi-rs│  (FFI)  │(P/Inv.)│(ccall│ (cgo)│(@cIm)│(Rust)│   │
│  └────────┴────────┴─────────┴────────┴──────┴──────┴──────┴──────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Security Features                            │   │
│  │  • 160+ Kani Formal Proofs  • CISA Secure by Design             │   │
│  │  • Memory Safe Rust  • 19 Verification Categories               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-RNN/
│
├── src/                        # Rust source code
│   ├── lib.rs                  # Library entry point (RNN cells, facade, types)
│   ├── backend.rs              # CUDA, OpenCL, and CPU backend implementations
│   ├── main.rs                 # CLI binary
│   └── kani/                   # Formal verification proofs (19 modules)
│       ├── mod.rs              # Module registration
│       ├── README.md           # Verification documentation
│       ├── bounds_checks.rs    # Strict bound checks
│       ├── pointer_validity.rs # Pointer validity proofs
│       ├── no_panic.rs         # No-panic guarantee
│       ├── integer_overflow.rs # Integer overflow prevention
│       ├── division_by_zero.rs # Division-by-zero exclusion
│       ├── concurrency.rs      # Global state consistency & deadlock-free logic
│       ├── input_sanitization.rs # Input sanitization bounds
│       ├── result_coverage.rs  # Result coverage audit
│       ├── memory_leaks.rs     # Memory leak/leakage proofs
│       ├── constant_time.rs    # Constant-time execution
│       ├── state_machine.rs    # State machine integrity
│       ├── enum_exhaustion.rs  # Enum exhaustion
│       ├── floating_point.rs   # Floating-point sanity
│       ├── resource_limits.rs  # Resource limit compliance
│       ├── ffi_cuda_boundary.rs    # CUDA FFI boundary safety (42 proofs)
│       ├── ffi_opencl_boundary.rs  # OpenCL FFI boundary safety (38 proofs)
│       ├── ffi_cpu_boundary.rs     # CPU backend boundary safety (32 proofs)
│       └── ffi_polyglot.rs         # Polyglot FFI boundary safety (48 proofs)
│
├── c/                          # C bindings
│   ├── include/
│   │   └── facaded_rnn.h       # C API header
│   ├── example.c               # C usage example
│   └── Makefile                # C build
│
├── cpp/                        # C++ bindings (FFI bridge crate)
│   ├── src/
│   │   └── lib.rs              # Rust extern "C" implementation
│   ├── include/
│   │   ├── facaded_rnn.h       # C API header
│   │   └── facaded_rnn.hpp     # C++ RAII wrapper header
│   ├── example.cpp             # C++ usage example
│   ├── CMakeLists.txt          # CMake build
│   ├── Cargo.toml              # Rust crate manifest
│   └── Cargo.lock
│
├── csharp/                     # C# bindings (.NET P/Invoke)
│   ├── NativeBindings.cs       # P/Invoke declarations
│   ├── RnnModel.cs             # High-level C# wrapper
│   ├── Example.cs              # C# usage example
│   └── FacadedRnn.csproj       # .NET project file
│
├── go/                         # Go bindings (cgo)
│   ├── facaded_rnn.go          # Go wrapper
│   ├── go.mod                  # Go module
│   └── example/
│       └── main.go             # Go usage example
│
├── julia/                      # Julia bindings (ccall)
│   ├── Project.toml            # Julia manifest
│   ├── src/
│   │   └── FacadedRNN.jl       # Julia module
│   └── example.jl              # Julia usage example
│
├── python/                     # Python bindings (PyO3)
│   ├── src/
│   │   └── lib.rs              # PyO3 wrapper
│   ├── Cargo.toml              # Rust crate manifest
│   ├── Cargo.lock
│   └── pyproject.toml          # Python build config
│
├── node/                       # Node.js bindings (napi-rs)
│   ├── src/
│   │   └── lib.rs              # napi-rs wrapper
│   ├── build.rs                # Build script
│   ├── Cargo.toml              # Rust crate manifest
│   ├── Cargo.lock
│   └── package.json            # Node.js package config
│
├── zig/                        # Zig bindings (@cImport)
│   ├── facaded_rnn.zig         # Zig wrapper
│   ├── example.zig             # Zig usage example
│   └── build.zig               # Zig build
│
├── Cargo.toml                  # Rust manifest
├── Cargo.lock                  # Rust lockfile
├── pyproject.toml              # Python build config (root)
├── package.json                # Node.js package config (root)
├── index.js                    # Node.js entry point
├── index.d.ts                  # TypeScript definitions
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.75+ | Core library compilation |

### GPU Backend (optional)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 12.0+ | NVIDIA GPU acceleration |
| **OpenCL SDK** | 1.2+ | Cross-platform GPU acceleration (AMD, Intel, NVIDIA) |

> **Note:** The CPU backend is always available. GPU backends are optional and enabled via Cargo features.

### Optional (Language Bindings)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **maturin** | 1.0+ | Python package build |
| **Node.js** | 14+ | Node.js bindings |
| **@napi-rs/cli** | 2.18+ | Node.js native module build |
| **GCC/Clang** | 11+ | C/C++ compilation |
| **.NET SDK** | 6.0+ | C# bindings |
| **Julia** | 1.6+ | Julia bindings |
| **Go** | 1.21+ | Go bindings |
| **Zig** | 0.11+ | Zig bindings |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **Rust Library & CLI (CUDA)**

```bash
# Build release binary and library with CUDA backend (default)
cargo build --release

# Run CLI
./target/release/facaded_rnn help
```

### **Rust Library with OpenCL Backend**

```bash
# Build with OpenCL backend
cargo build --release --features opencl --no-default-features

# Build with both CUDA and OpenCL
cargo build --release --features "cuda,opencl"
```

### **CPU-Only Build (no GPU required)**

```bash
# Build with CPU backend only
cargo build --release --no-default-features
```

### **Python Bindings**

```bash
# Install maturin
pip install maturin

# Build and install Python package
cd python && maturin develop --release
```

### **Node.js Bindings**

```bash
cd node && npm install && npm run build
```

### **C/C++ Library**

```bash
# Build the FFI shared library
cd cpp && cargo build --release --no-default-features

# Compile C example
cd c && make
```

### **C# Bindings**

```bash
# Build FFI library first
cd cpp && cargo build --release --no-default-features

# Build C# project
cd csharp && dotnet build
```

### **Go Bindings**

```bash
# Build FFI library first
cd cpp && cargo build --release --no-default-features

# Run Go example
cd go/example && go run .
```

### **Julia Bindings**

```bash
# Build FFI library first
cd cpp && cargo build --release --no-default-features

# Run Julia example
cd julia && julia example.jl
```

### **Zig Bindings**

```bash
# Build FFI library first
cd cpp && cargo build --release --no-default-features

# Build Zig example
cd zig && zig build
```

---

## **Language Bindings**

### **Rust API**

```rust
use facaded_rnn::{
    ActivationType, CellType, LossType, RNNFacade,
    backend::BackendChoice, select_backend_arc,
};

fn main() {
    // Create an LSTM model
    let mut model = RNNFacade::new(
        2,                              // input_size
        vec![32, 16],                   // hidden_sizes (stacked layers)
        1,                              // output_size
        CellType::LSTM,                 // cell type
        ActivationType::Tanh,           // hidden activation
        ActivationType::Linear,         // output activation
        LossType::MSE,                  // loss function
        0.01,                           // learning rate
        5.0,                            // gradient clip
        0,                              // bptt_steps (0 = full)
    );

    // Optionally set GPU backend
    if let Ok(backend) = select_backend_arc(BackendChoice::Auto) {
        model.set_backend(backend);
    }

    // Training data (sequence of timesteps)
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let targets = vec![vec![1.0], vec![1.0], vec![0.0]];

    // Train
    let loss = model.train_sequence(&inputs, &targets);
    println!("Loss: {:.6}", loss);

    // Predict
    let outputs = model.predict(&inputs);
    for (t, out) in outputs.iter().enumerate() {
        println!("t={}: {:?}", t, out);
    }

    // Introspect
    let h = model.get_hidden_value(0, 0, 0);
    println!("Hidden[0][0][0]: {}", h);

    // Save/Load
    model.save_model("model.json").unwrap();
    let loaded = RNNFacade::load_model("model.json").unwrap();
}
```

### **Python API**

```python
from facaded_rnn import PyRNNModel

# Create LSTM model
model = PyRNNModel(
    input_size=2,
    hidden_sizes=[32, 16],
    output_size=1,
    cell_type="lstm",
    activation="tanh",
    output_activation="linear",
    loss="mse",
    learning_rate=0.01,
    gradient_clip=5.0,
    backend="auto",
)

# Train
inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
targets = [[1.0], [1.0], [0.0]]
loss = model.train_sequence(inputs, targets)
print(f"Loss: {loss:.6f}")

# Predict
outputs = model.predict(inputs)
print(f"Outputs: {outputs}")

# Introspect
h = model.get_hidden_value(0, 0, 0)
print(f"Hidden state: {h}")

# Gradient diagnostics
vanishing = model.detect_vanishing_gradients(1e-7)
exploding = model.detect_exploding_gradients(10.0)

# Save/Load
model.save("model.json")
loaded = PyRNNModel.load("model.json", backend="auto")
```

### **Node.js API**

```javascript
const { RNNModel } = require('facaded_rnn');

// Create GRU model
const model = new RNNModel({
    inputSize: 2,
    hiddenSizes: [32],
    outputSize: 1,
    cellType: 'gru',
    activation: 'tanh',
    outputActivation: 'linear',
    loss: 'mse',
    learningRate: 0.01,
    gradientClip: 5.0,
    backend: 'auto',
});

// Train
const inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
const targets = [[1.0], [1.0], [0.0]];
const loss = model.trainSequence(inputs, targets);
console.log(`Loss: ${loss}`);

// Predict
const outputs = model.predict(inputs);
console.log('Outputs:', outputs);

// Save/Load
model.save('model.json');
const loaded = RNNModel.load('model.json', 'auto');
```

### **C API**

```c
#include "facaded_rnn.h"
#include <stdio.h>

int main() {
    uint32_t hidden[] = {32};
    RnnHandle* rnn = rnn_create(
        2, hidden, 1, 1,
        "lstm", "tanh", "linear", "mse",
        0.01, 5.0, 0, "auto"
    );

    double input[] = {1.0, 0.0, 0.0, 1.0};
    double target[] = {1.0, 1.0};

    double loss = rnn_train_sequence(rnn, input, target, 2, 2, 1);
    printf("Loss: %f\n", loss);

    int32_t total = rnn_predict(rnn, input, 2, 2, NULL, 0);
    double* output = malloc(total * sizeof(double));
    rnn_predict(rnn, input, 2, 2, output, total);

    rnn_save(rnn, "model.json");
    rnn_destroy(rnn);
    free(output);
    return 0;
}
```

### **C++ API**

```cpp
#include "facaded_rnn.hpp"
#include <iostream>

int main() {
    // Uses RAII wrapper around C API
    std::vector<uint32_t> hidden = {32};
    auto* rnn = rnn_create(
        2, hidden.data(), 1, 1,
        "lstm", "tanh", "linear", "mse",
        0.01, 5.0, 0, "auto"
    );

    double input[] = {1.0, 0.0};
    double target[] = {1.0};
    double loss = rnn_train_sequence(rnn, input, target, 1, 2, 1);
    std::cout << "Loss: " << loss << std::endl;

    rnn_save(rnn, "model.json");
    rnn_destroy(rnn);
    return 0;
}
```

### **C# API**

```csharp
using FacadedRnn;

var opts = new ModelOptions {
    InputSize = 2,
    HiddenSizes = new uint[] { 32 },
    OutputSize = 1,
    CellType = "lstm",
    Activation = "tanh",
    OutputActivation = "linear",
    Loss = "mse",
    LearningRate = 0.01,
    GradientClip = 5.0,
    Backend = "auto",
};

using var model = new RnnModel(opts);

double[] input = { 1.0, 0.0, 0.0, 1.0 };
double[] target = { 1.0, 1.0 };
double loss = model.TrainSequence(input, target, 2, 2, 1);
Console.WriteLine($"Loss: {loss}");

model.Save("model.json");
```

### **Julia API**

```julia
using FacadedRNN

# Create LSTM model
rnn = RNNModel(
    input_size=2, hidden_sizes=[32], output_size=1,
    cell_type="lstm", activation="tanh",
    output_activation="linear", loss="mse",
    learning_rate=0.01, gradient_clip=5.0,
    backend="auto"
)

# Train
input = [1.0, 0.0, 0.0, 1.0]
target = [1.0, 1.0]
loss = train_sequence!(rnn, input, target, 2, 2, 1)
println("Loss: $loss")

# Save/Load
save_model(rnn, "model.json")
loaded = load_model("model.json", "auto")

# Cleanup
destroy!(rnn)
```

### **Go API**

```go
package main

import (
    "fmt"
    "log"
    rnn "github.com/GlassBoxAI/GlassBoxAI-RNN/go"
)

func main() {
    model, err := rnn.NewRNNModel(rnn.ModelOptions{
        InputSize:   2,
        HiddenSizes: []int{32},
        OutputSize:  1,
        CellType:    "lstm",
        Activation:  "tanh",
        Loss:        "mse",
        LearningRate: 0.01,
        GradientClip: 5.0,
        Backend:     "auto",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()

    inputs := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
    targets := [][]float64{{1.0}, {1.0}}
    loss := model.TrainSequence(inputs, targets)
    fmt.Println("Loss:", loss)

    outputs := model.Predict(inputs)
    fmt.Println("Outputs:", outputs)

    model.Save("model.json")
}
```

### **Zig API**

```zig
const std = @import("std");
const rnn = @import("facaded_rnn");

pub fn main() !void {
    const hidden = [_]u32{32};
    var model = try rnn.RnnModel.init(.{
        .input_size = 2,
        .hidden_sizes = &hidden,
        .output_size = 1,
        .cell_type = "lstm",
        .activation = "tanh",
        .output_activation = "linear",
        .loss = "mse",
        .learning_rate = 0.01,
        .gradient_clip = 5.0,
        .backend = "auto",
    });
    defer model.deinit();

    const input = [_]f64{ 1.0, 0.0 };
    const target = [_]f64{1.0};
    const loss = model.trainSequence(&input, &target, 1, 2, 1);
    std.debug.print("Loss: {d}\n", .{loss});

    try model.save("model.json");
}
```

---

## **CLI Reference**

### Usage

```
facaded_rnn [--backend=auto|cpu|cuda|opencl|hybrid] <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new RNN model |
| `train` | Train an existing model |
| `predict` | Make predictions |
| `info` | Display model information |
| `query` | Query model internals (hidden states, gate values, gradients) |
| `help` | Show help message |

### Create Options

| Option | Description |
|--------|-------------|
| `--input=N` | Input size (required) |
| `--hidden=N,N,...` | Hidden layer sizes (required) |
| `--output=N` | Output size (required) |
| `--save=FILE` | Save path (required) |
| `--cell=TYPE` | Cell type: simplernn, lstm, gru (default: lstm) |
| `--lr=VALUE` | Learning rate (default: 0.01) |
| `--hidden-act=TYPE` | Hidden activation: sigmoid, tanh, relu, linear (default: tanh) |
| `--output-act=TYPE` | Output activation (default: linear) |
| `--loss=TYPE` | Loss function: mse, crossentropy (default: mse) |
| `--clip=VALUE` | Gradient clipping (default: 5.0) |
| `--bptt=N` | BPTT truncation steps, 0 = full (default: 0) |

### Examples

```bash
# Create LSTM model
facaded_rnn create \
    --input=2 --hidden=32,16 --output=1 \
    --cell=lstm --save=model.json

# Create GRU model with OpenCL
facaded_rnn --backend=opencl create \
    --input=10 --hidden=64 --output=5 \
    --cell=gru --lr=0.001 --save=gru_model.json

# Train model
facaded_rnn train \
    --model=model.json --data=data.csv \
    --epochs=100 --save=trained.json --verbose

# Model info
facaded_rnn info --model=model.json

# Predict
facaded_rnn predict --model=model.json --input=1.0,0.0

# Query hidden states
facaded_rnn query --model=model.json --query-type=hidden --layer=0
```

---

## **Formal Verification with Kani**

### Overview

The implementation includes **160+ Kani formal verification proofs** across **19 verification categories** that mathematically prove the absence of certain classes of bugs.

### Verification Categories

| # | Category | File | Description |
|---|----------|------|-------------|
| 1 | Strict Bound Checks | `bounds_checks.rs` | Proves all collection indexing is incapable of out-of-bounds access |
| 2 | Pointer Validity Proofs | `pointer_validity.rs` | Verifies raw pointer dereferences are valid, aligned, and point to initialized memory |
| 3 | No-Panic Guarantee | `no_panic.rs` | Verifies functions cannot trigger panic!, unwrap(), or expect() failures |
| 4 | Integer Overflow Prevention | `integer_overflow.rs` | Proves arithmetic operations are safe from wrapping/overflow/underflow |
| 5 | Division-by-Zero Exclusion | `division_by_zero.rs` | Verifies denominators are never zero |
| 6 | Global State Consistency | `concurrency.rs` | Proves concurrent access maintains invariants |
| 7 | Deadlock-Free Logic | `concurrency.rs` | Verifies locking follows strict hierarchy |
| 8 | Input Sanitization Bounds | `input_sanitization.rs` | Proves loops/recursion have formal upper bounds |
| 9 | Result Coverage Audit | `result_coverage.rs` | Verifies all Error variants are explicitly handled |
| 10 | Memory Leak/Leakage Proofs | `memory_leaks.rs` | Proves all allocated memory is freed or reachable |
| 11 | Constant-Time Execution | `constant_time.rs` | Verifies branching doesn't depend on secrets |
| 12 | State Machine Integrity | `state_machine.rs` | Proves no invalid state transitions |
| 13 | Enum Exhaustion | `enum_exhaustion.rs` | Verifies all match statements handle every variant |
| 14 | Floating-Point Sanity | `floating_point.rs` | Proves no unhandled NaN or Infinity states |
| 15 | Resource Limit Compliance | `resource_limits.rs` | Verifies allocations never exceed security budget |
| 16 | CUDA FFI Boundary Safety | `ffi_cuda_boundary.rs` | Proves CUDA kernel launch dims, buffer sizing, f64 alignment, grid overflow prevention |
| 17 | OpenCL FFI Boundary Safety | `ffi_opencl_boundary.rs` | Proves OpenCL global work sizes, CL buffer sizing, cl_double ABI, kernel arg i32 cast safety |
| 18 | CPU Backend Boundary Safety | `ffi_cpu_boundary.rs` | Proves CPU backend loop bounds, sigmoid clamping, activation coverage, matvec index safety |
| 19 | Polyglot FFI Boundary Safety | `ffi_polyglot.rs` | Proves C API u32/f64 validation, enum string parsing, null handle/string safety, ABI compatibility |

### FFI Boundary Proof Coverage

| Proof File | Proofs | Unit Tests | Categories (A-O) | Targets |
|-----------|--------|------------|-------------------|---------|
| `ffi_cuda_boundary.rs` | 42 | 6 | Grid/block dims, matvec/LSTM/GRU/SimpleRNN buffer sizing, f64 alignment, i32 cast safety, end-to-end chains | CUDA backend (BLOCK_SIZE=256) |
| `ffi_opencl_boundary.rs` | 38 | 6 | Global work sizes, CL_MEM_READ_ONLY/READ_WRITE sizing, cl_double ABI, kernel sync, end-to-end chains | OpenCL backend (cl_khr_fp64) |
| `ffi_cpu_boundary.rs` | 32 | 7 | Loop bounds, sigmoid clamping, activation range proofs, weight index formula, cell→output chains | CPU backend (pure Rust) |
| `ffi_polyglot.rs` | 48 | 7 | u32-to-usize, NaN/Inf rejection, CellType/ActivationType/LossType/BackendChoice/GateType parsing, null safety, ABI sizes | C API → Go, C#, Julia, Zig, C, C++ |

### Running Verification

```bash
# Install Kani
cargo install --locked kani-verifier
kani setup

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness verify_cuda_grid_blocks_nonzero

# Run all proofs in a specific file
cargo kani --harness "verify_cpu_*"

# Run with verbose output
cargo kani --verbose
```

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA** and **NSA** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows and data races |
| **Formal Verification** | 160+ Kani proofs mathematically verify absence of critical bugs |
| **FFI Boundary Safety** | All GPU (CUDA/OpenCL) and polyglot (C API) boundaries formally verified |
| **Input Validation** | All CLI inputs, enum strings, and f64 parameters validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime, formal verification) |
| **Secure Defaults** | Safe default configurations throughout (CPU fallback, gradient clipping) |
| **Transparency** | Open source with full code visibility |
| **Multi-Backend Verification** | CUDA, OpenCL, and CPU backends independently verified |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (160+ Kani proof harnesses across 19 categories)
- [x] **CUDA boundary verification** (grid dims, buffer sizing, alignment)
- [x] **OpenCL boundary verification** (work sizes, CL buffer sizing, ABI)
- [x] **CPU backend verification** (loop bounds, activation safety, index proofs)
- [x] **Polyglot FFI verification** (u32/f64 validation, enum parsing, null safety)
- [x] **Comprehensive testing** (Unit + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing + FFI parameter validation)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*

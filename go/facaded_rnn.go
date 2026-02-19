/**
 * @file
 * @ingroup RNN_Wrappers
 */
// Package facaded_rnn provides Go bindings for the facaded_rnn
// GPU-accelerated RNN library (CUDA/OpenCL/CPU).
//
// Build the Rust FFI library first:
//
//	cd cpp && cargo build --release --no-default-features
package facaded_rnn

/*
#cgo LDFLAGS: -L${SRCDIR}/../cpp/target/release -lfacaded_rnn_cpp -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../c/include

#include "facaded_rnn.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

type RNNError struct {
	Msg string
}

func (e *RNNError) Error() string {
	return fmt.Sprintf("facaded_rnn: %s", e.Msg)
}

func lastError(fallback string) error {
	ptr := C.rnn_last_error()
	if ptr != nil {
		msg := C.GoString(ptr)
		C.rnn_clear_error()
		return &RNNError{Msg: msg}
	}
	return &RNNError{Msg: fallback}
}

// ModelOptions configures a new RNN model.
type ModelOptions struct {
	InputSize        int
	HiddenSizes      []int
	OutputSize       int
	CellType         string  // "simplernn", "lstm", "gru" (default "lstm")
	Activation       string  // "sigmoid", "tanh", "relu", "linear" (default "tanh")
	OutputActivation string  // same choices (default "linear")
	Loss             string  // "mse", "crossentropy" (default "mse")
	LearningRate     float64 // default 0.01
	GradientClip     float64 // default 5.0
	BpttSteps        int     // default 0 (full)
	Backend          string  // "auto", "cpu", "cuda", "opencl", "hybrid" (default "auto")
}

func applyDefaults(o *ModelOptions) {
	if o.CellType == "" {
		o.CellType = "lstm"
	}
	if o.Activation == "" {
		o.Activation = "tanh"
	}
	if o.OutputActivation == "" {
		o.OutputActivation = "linear"
	}
	if o.Loss == "" {
		o.Loss = "mse"
	}
	if o.LearningRate == 0 {
		o.LearningRate = 0.01
	}
	if o.GradientClip == 0 {
		o.GradientClip = 5.0
	}
	if o.Backend == "" {
		o.Backend = "auto"
	}
}

// GradientDiagnostic holds the result of gradient analysis.
type GradientDiagnostic struct {
	Count int
	Value float64
}

// RNNModel wraps the Rust RNN model.
type RNNModel struct {
	handle *C.RnnHandle
}

// NewRNNModel creates a new RNN model with the given options.
func NewRNNModel(opts ModelOptions) (*RNNModel, error) {
	applyDefaults(&opts)

	hs := make([]C.uint32_t, len(opts.HiddenSizes))
	for i, v := range opts.HiddenSizes {
		hs[i] = C.uint32_t(v)
	}

	cCell := C.CString(opts.CellType)
	cAct := C.CString(opts.Activation)
	cOutAct := C.CString(opts.OutputActivation)
	cLoss := C.CString(opts.Loss)
	cBackend := C.CString(opts.Backend)
	defer C.free(unsafe.Pointer(cCell))
	defer C.free(unsafe.Pointer(cAct))
	defer C.free(unsafe.Pointer(cOutAct))
	defer C.free(unsafe.Pointer(cLoss))
	defer C.free(unsafe.Pointer(cBackend))

	var hsPtr *C.uint32_t
	if len(hs) > 0 {
		hsPtr = &hs[0]
	}

	handle := C.rnn_create(
		C.uint32_t(opts.InputSize),
		hsPtr,
		C.uint32_t(len(hs)),
		C.uint32_t(opts.OutputSize),
		cCell, cAct, cOutAct, cLoss,
		C.double(opts.LearningRate),
		C.double(opts.GradientClip),
		C.uint32_t(opts.BpttSteps),
		cBackend,
	)
	if handle == nil {
		return nil, lastError("failed to create model")
	}

	m := &RNNModel{handle: handle}
	runtime.SetFinalizer(m, (*RNNModel).Close)
	return m, nil
}

// LoadModel loads a model from a JSON file.
func LoadModel(filename, backend string) (*RNNModel, error) {
	if backend == "" {
		backend = "auto"
	}
	cFile := C.CString(filename)
	cBack := C.CString(backend)
	defer C.free(unsafe.Pointer(cFile))
	defer C.free(unsafe.Pointer(cBack))

	handle := C.rnn_load(cFile, cBack)
	if handle == nil {
		return nil, lastError("failed to load model")
	}

	m := &RNNModel{handle: handle}
	runtime.SetFinalizer(m, (*RNNModel).Close)
	return m, nil
}

// Close frees the underlying Rust resources.
func (m *RNNModel) Close() {
	if m.handle != nil {
		C.rnn_destroy(m.handle)
		m.handle = nil
	}
}

// Save writes the model to a JSON file.
func (m *RNNModel) Save(filename string) error {
	cFile := C.CString(filename)
	defer C.free(unsafe.Pointer(cFile))
	if C.rnn_save(m.handle, cFile) != 0 {
		return lastError("failed to save model")
	}
	return nil
}

func flatten(data [][]float64) []C.double {
	n := 0
	for _, row := range data {
		n += len(row)
	}
	flat := make([]C.double, 0, n)
	for _, row := range data {
		for _, v := range row {
			flat = append(flat, C.double(v))
		}
	}
	return flat
}

func unflatten(flat []C.double, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		row := make([]float64, cols)
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if idx < len(flat) {
				row[c] = float64(flat[idx])
			}
		}
		result[r] = row
	}
	return result
}

func dataPtr(flat []C.double) *C.double {
	if len(flat) == 0 {
		return nil
	}
	return &flat[0]
}

func dims(data [][]float64) (timesteps, featureSize C.uint32_t) {
	timesteps = C.uint32_t(len(data))
	if len(data) > 0 {
		featureSize = C.uint32_t(len(data[0]))
	}
	return
}

// Predict runs a forward pass with state reset and returns outputs.
func (m *RNNModel) Predict(inputs [][]float64) [][]float64 {
	flatIn := flatten(inputs)
	ts, inSz := dims(inputs)
	inPtr := dataPtr(flatIn)

	total := C.rnn_predict(m.handle, inPtr, ts, inSz, nil, 0)
	if total <= 0 {
		return nil
	}

	buf := make([]C.double, int(total))
	C.rnn_predict(m.handle, inPtr, ts, inSz, &buf[0], C.uint32_t(total))

	outSz := int(C.rnn_get_output_size(m.handle))
	return unflatten(buf, int(ts), outSz)
}

// TrainSequence trains on one sequence. Returns the loss.
func (m *RNNModel) TrainSequence(inputs, targets [][]float64) float64 {
	flatIn := flatten(inputs)
	flatTgt := flatten(targets)
	ts, inSz := dims(inputs)
	_, outSz := dims(targets)

	loss := C.rnn_train_sequence(
		m.handle, dataPtr(flatIn), dataPtr(flatTgt),
		ts, inSz, outSz,
	)
	return float64(loss)
}

// Train runs training for multiple epochs. Returns per-epoch losses.
func (m *RNNModel) Train(inputs, targets [][]float64, epochs int) []float64 {
	flatIn := flatten(inputs)
	flatTgt := flatten(targets)
	ts, inSz := dims(inputs)
	_, outSz := dims(targets)

	losses := make([]C.double, epochs)
	C.rnn_train(
		m.handle, dataPtr(flatIn), dataPtr(flatTgt),
		ts, inSz, outSz,
		C.uint32_t(epochs), &losses[0], C.uint32_t(epochs),
	)

	result := make([]float64, epochs)
	for i, v := range losses {
		result[i] = float64(v)
	}
	return result
}

// ForwardSequence runs a forward pass without state reset.
func (m *RNNModel) ForwardSequence(inputs [][]float64) [][]float64 {
	flatIn := flatten(inputs)
	ts, inSz := dims(inputs)
	inPtr := dataPtr(flatIn)

	total := C.rnn_forward_sequence(m.handle, inPtr, ts, inSz, nil, 0)
	if total <= 0 {
		return nil
	}

	buf := make([]C.double, int(total))
	C.rnn_forward_sequence(m.handle, inPtr, ts, inSz, &buf[0], C.uint32_t(total))

	outSz := int(C.rnn_get_output_size(m.handle))
	return unflatten(buf, int(ts), outSz)
}

// BackwardSequence runs backprop (call after ForwardSequence). Returns loss.
func (m *RNNModel) BackwardSequence(targets [][]float64) float64 {
	flatTgt := flatten(targets)
	ts, outSz := dims(targets)
	return float64(C.rnn_backward_sequence(m.handle, dataPtr(flatTgt), ts, outSz))
}

// ResetStates resets all hidden/cell states to zero.
func (m *RNNModel) ResetStates() {
	C.rnn_reset_states(m.handle)
}

// InputSize returns the input dimension.
func (m *RNNModel) InputSize() int {
	return int(C.rnn_get_input_size(m.handle))
}

// OutputSize returns the output dimension.
func (m *RNNModel) OutputSize() int {
	return int(C.rnn_get_output_size(m.handle))
}

// LayerCount returns the number of hidden layers.
func (m *RNNModel) LayerCount() int {
	return int(C.rnn_get_layer_count(m.handle))
}

// HiddenSize returns the hidden size for a given layer.
func (m *RNNModel) HiddenSize(layer int) int {
	return int(C.rnn_get_hidden_size(m.handle, C.uint32_t(layer)))
}

// SequenceLength returns the length of the last processed sequence.
func (m *RNNModel) SequenceLength() int {
	return int(C.rnn_get_sequence_length(m.handle))
}

// LearningRate returns the current learning rate.
func (m *RNNModel) LearningRate() float64 {
	return float64(C.rnn_get_learning_rate(m.handle))
}

// SetLearningRate updates the learning rate.
func (m *RNNModel) SetLearningRate(lr float64) {
	C.rnn_set_learning_rate(m.handle, C.double(lr))
}

// GradientClip returns the gradient clipping threshold.
func (m *RNNModel) GradientClip() float64 {
	return float64(C.rnn_get_gradient_clip(m.handle))
}

// DropoutRate returns the current dropout rate.
func (m *RNNModel) DropoutRate() float64 {
	return float64(C.rnn_get_dropout_rate(m.handle))
}

// SetDropoutRate updates the dropout rate.
func (m *RNNModel) SetDropoutRate(rate float64) {
	C.rnn_set_dropout_rate(m.handle, C.double(rate))
}

// GPUAvailable returns true if a GPU backend is active.
func (m *RNNModel) GPUAvailable() bool {
	return C.rnn_is_gpu_available(m.handle) != 0
}

// GetHiddenValue returns a hidden state value.
func (m *RNNModel) GetHiddenValue(layer, timestep, neuron int) float64 {
	return float64(C.rnn_get_hidden_value(m.handle,
		C.uint32_t(layer), C.uint32_t(timestep), C.uint32_t(neuron)))
}

// SetHiddenValue sets a hidden state value.
func (m *RNNModel) SetHiddenValue(layer, neuron int, value float64) {
	C.rnn_set_hidden_value(m.handle, C.uint32_t(layer), C.uint32_t(neuron), C.double(value))
}

// GetOutputValue returns a cached output value.
func (m *RNNModel) GetOutputValue(timestep, index int) float64 {
	return float64(C.rnn_get_output_value(m.handle, C.uint32_t(timestep), C.uint32_t(index)))
}

// GetCellState returns an LSTM cell state value.
func (m *RNNModel) GetCellState(layer, neuron int) float64 {
	return float64(C.rnn_get_cell_state(m.handle, C.uint32_t(layer), C.uint32_t(neuron)))
}

// GetGateValue returns a gate value. Gate names: "forget", "input", "output",
// "cellcandidate" (LSTM); "update", "reset", "hiddencandidate" (GRU).
func (m *RNNModel) GetGateValue(layer, timestep, neuron int, gate string) float64 {
	cGate := C.CString(gate)
	defer C.free(unsafe.Pointer(cGate))
	return float64(C.rnn_get_gate_value(m.handle,
		C.uint32_t(layer), C.uint32_t(timestep), C.uint32_t(neuron), cGate))
}

// GetPreactivation returns a pre-activation value.
func (m *RNNModel) GetPreactivation(layer, timestep, neuron int) float64 {
	return float64(C.rnn_get_preactivation(m.handle,
		C.uint32_t(layer), C.uint32_t(timestep), C.uint32_t(neuron)))
}

// GetInputValue returns a cached input value.
func (m *RNNModel) GetInputValue(timestep, index int) float64 {
	return float64(C.rnn_get_input_value(m.handle, C.uint32_t(timestep), C.uint32_t(index)))
}

// GetSequenceOutputs returns all cached outputs as [][]float64.
func (m *RNNModel) GetSequenceOutputs() [][]float64 {
	total := C.rnn_get_sequence_outputs(m.handle, nil, 0)
	if total <= 0 {
		return nil
	}
	buf := make([]C.double, int(total))
	C.rnn_get_sequence_outputs(m.handle, &buf[0], C.uint32_t(total))
	seqLen := m.SequenceLength()
	outSz := m.OutputSize()
	return unflatten(buf, seqLen, outSz)
}

// GetSequenceHiddenStates returns all cached hidden states for a layer.
func (m *RNNModel) GetSequenceHiddenStates(layer int) [][]float64 {
	total := C.rnn_get_sequence_hidden_states(m.handle, C.uint32_t(layer), nil, 0)
	if total <= 0 {
		return nil
	}
	buf := make([]C.double, int(total))
	C.rnn_get_sequence_hidden_states(m.handle, C.uint32_t(layer), &buf[0], C.uint32_t(total))
	seqLen := m.SequenceLength()
	hs := m.HiddenSize(layer)
	return unflatten(buf, seqLen, hs)
}

// DetectVanishingGradients checks for gradients below threshold.
func (m *RNNModel) DetectVanishingGradients(threshold float64) GradientDiagnostic {
	var count C.int32_t
	var minVal C.double
	C.rnn_detect_vanishing_gradients(m.handle, C.double(threshold), &count, &minVal)
	return GradientDiagnostic{Count: int(count), Value: float64(minVal)}
}

// DetectExplodingGradients checks for gradients above threshold.
func (m *RNNModel) DetectExplodingGradients(threshold float64) GradientDiagnostic {
	var count C.int32_t
	var maxVal C.double
	C.rnn_detect_exploding_gradients(m.handle, C.double(threshold), &count, &maxVal)
	return GradientDiagnostic{Count: int(count), Value: float64(maxVal)}
}

// String returns a summary of the model.
func (m *RNNModel) String() string {
	hiddens := make([]int, m.LayerCount())
	for i := range hiddens {
		hiddens[i] = m.HiddenSize(i)
	}
	gpu := "CPU"
	if m.GPUAvailable() {
		gpu = "GPU"
	}
	return fmt.Sprintf("RNNModel(input=%d, hidden=%v, output=%d, backend=%s)",
		m.InputSize(), hiddens, m.OutputSize(), gpu)
}

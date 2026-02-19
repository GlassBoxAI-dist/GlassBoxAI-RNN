/**
 * @file
 * @ingroup RNN_Wrappers
 */
using System;
using System.Runtime.InteropServices;

namespace FacadedRnn
{
    public class RnnException : Exception
    {
        public RnnException(string message) : base(message) { }

        internal static RnnException FromLastError(string fallback)
        {
            var ptr = NativeBindings.rnn_last_error();
            if (ptr != IntPtr.Zero)
            {
                var msg = Marshal.PtrToStringUTF8(ptr) ?? fallback;
                NativeBindings.rnn_clear_error();
                return new RnnException(msg);
            }
            return new RnnException(fallback);
        }
    }

    public struct GradientDiagnostic
    {
        public int Count;
        public double Value;
    }

    public class ModelOptions
    {
        public uint InputSize { get; set; }
        public uint[] HiddenSizes { get; set; } = Array.Empty<uint>();
        public uint OutputSize { get; set; }
        public string CellType { get; set; } = "lstm";
        public string Activation { get; set; } = "tanh";
        public string OutputActivation { get; set; } = "linear";
        public string Loss { get; set; } = "mse";
        public double LearningRate { get; set; } = 0.01;
        public double GradientClip { get; set; } = 5.0;
        public uint BpttSteps { get; set; } = 0;
        public string Backend { get; set; } = "auto";
    }

    public sealed class RnnModel : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        private RnnModel(IntPtr handle)
        {
            _handle = handle;
        }

        public static RnnModel Create(ModelOptions opts)
        {
            var handle = NativeBindings.rnn_create(
                opts.InputSize,
                opts.HiddenSizes,
                (uint)opts.HiddenSizes.Length,
                opts.OutputSize,
                opts.CellType,
                opts.Activation,
                opts.OutputActivation,
                opts.Loss,
                opts.LearningRate,
                opts.GradientClip,
                opts.BpttSteps,
                opts.Backend
            );
            if (handle == IntPtr.Zero)
                throw RnnException.FromLastError("Failed to create model");
            return new RnnModel(handle);
        }

        public static RnnModel Load(string filename, string backend = "auto")
        {
            var handle = NativeBindings.rnn_load(filename, backend);
            if (handle == IntPtr.Zero)
                throw RnnException.FromLastError("Failed to load model");
            return new RnnModel(handle);
        }

        public void Save(string filename)
        {
            if (NativeBindings.rnn_save(_handle, filename) != 0)
                throw RnnException.FromLastError("Failed to save model");
        }

        // ── Training & Inference ──

        public double[] Predict(double[] inputData, uint numTimesteps, uint inputSize)
        {
            int total = NativeBindings.rnn_predict(_handle, inputData, numTimesteps, inputSize, null, 0);
            if (total <= 0)
                throw RnnException.FromLastError("Predict failed");

            var buf = new double[total];
            NativeBindings.rnn_predict(_handle, inputData, numTimesteps, inputSize, buf, (uint)total);
            return buf;
        }

        public double TrainSequence(double[] inputData, double[] targetData, uint numTimesteps, uint inputSize, uint outputSize)
        {
            return NativeBindings.rnn_train_sequence(_handle, inputData, targetData, numTimesteps, inputSize, outputSize);
        }

        public double[] Train(double[] inputData, double[] targetData, uint numTimesteps, uint inputSize, uint outputSize, uint epochs)
        {
            var losses = new double[epochs];
            NativeBindings.rnn_train(_handle, inputData, targetData, numTimesteps, inputSize, outputSize, epochs, losses, epochs);
            return losses;
        }

        public double[] ForwardSequence(double[] inputData, uint numTimesteps, uint inputSize)
        {
            int total = NativeBindings.rnn_forward_sequence(_handle, inputData, numTimesteps, inputSize, null, 0);
            if (total <= 0)
                throw RnnException.FromLastError("Forward sequence failed");

            var buf = new double[total];
            NativeBindings.rnn_forward_sequence(_handle, inputData, numTimesteps, inputSize, buf, (uint)total);
            return buf;
        }

        public double BackwardSequence(double[] targetData, uint numTimesteps, uint outputSize)
        {
            return NativeBindings.rnn_backward_sequence(_handle, targetData, numTimesteps, outputSize);
        }

        public void ResetStates()
        {
            NativeBindings.rnn_reset_states(_handle);
        }

        // ── Properties ──

        public uint InputSize => NativeBindings.rnn_get_input_size(_handle);
        public uint OutputSize => NativeBindings.rnn_get_output_size(_handle);
        public uint LayerCount => NativeBindings.rnn_get_layer_count(_handle);
        public uint SequenceLength => NativeBindings.rnn_get_sequence_length(_handle);
        public uint HiddenSize(uint layer) => NativeBindings.rnn_get_hidden_size(_handle, layer);
        public bool IsGpuAvailable => NativeBindings.rnn_is_gpu_available(_handle) != 0;

        public double LearningRate
        {
            get => NativeBindings.rnn_get_learning_rate(_handle);
            set => NativeBindings.rnn_set_learning_rate(_handle, value);
        }

        public double GradientClip => NativeBindings.rnn_get_gradient_clip(_handle);

        public double DropoutRate
        {
            get => NativeBindings.rnn_get_dropout_rate(_handle);
            set => NativeBindings.rnn_set_dropout_rate(_handle, value);
        }

        // ── Introspection ──

        public double GetHiddenValue(uint layer, uint timestep, uint neuron)
            => NativeBindings.rnn_get_hidden_value(_handle, layer, timestep, neuron);

        public void SetHiddenValue(uint layer, uint neuron, double value)
            => NativeBindings.rnn_set_hidden_value(_handle, layer, neuron, value);

        public double GetOutputValue(uint timestep, uint index)
            => NativeBindings.rnn_get_output_value(_handle, timestep, index);

        public double GetCellState(uint layer, uint neuron)
            => NativeBindings.rnn_get_cell_state(_handle, layer, neuron);

        public double GetGateValue(uint layer, uint timestep, uint neuron, string gate)
            => NativeBindings.rnn_get_gate_value(_handle, layer, timestep, neuron, gate);

        public double GetPreactivation(uint layer, uint timestep, uint neuron)
            => NativeBindings.rnn_get_preactivation(_handle, layer, timestep, neuron);

        public double GetInputValue(uint timestep, uint index)
            => NativeBindings.rnn_get_input_value(_handle, timestep, index);

        public double[] GetSequenceOutputs()
        {
            int total = NativeBindings.rnn_get_sequence_outputs(_handle, null, 0);
            if (total <= 0) return Array.Empty<double>();
            var buf = new double[total];
            NativeBindings.rnn_get_sequence_outputs(_handle, buf, (uint)total);
            return buf;
        }

        public double[] GetSequenceHiddenStates(uint layer)
        {
            int total = NativeBindings.rnn_get_sequence_hidden_states(_handle, layer, null, 0);
            if (total <= 0) return Array.Empty<double>();
            var buf = new double[total];
            NativeBindings.rnn_get_sequence_hidden_states(_handle, layer, buf, (uint)total);
            return buf;
        }

        // ── Gradient Diagnostics ──

        public GradientDiagnostic DetectVanishingGradients(double threshold)
        {
            NativeBindings.rnn_detect_vanishing_gradients(_handle, threshold, out int count, out double minVal);
            return new GradientDiagnostic { Count = count, Value = minVal };
        }

        public GradientDiagnostic DetectExplodingGradients(double threshold)
        {
            NativeBindings.rnn_detect_exploding_gradients(_handle, threshold, out int count, out double maxVal);
            return new GradientDiagnostic { Count = count, Value = maxVal };
        }

        // ── IDisposable ──

        public void Dispose()
        {
            if (!_disposed)
            {
                NativeBindings.rnn_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~RnnModel()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeBindings.rnn_destroy(_handle);
            }
        }

        public override string ToString()
        {
            var gpu = IsGpuAvailable ? "GPU" : "CPU";
            return $"RnnModel(input={InputSize}, layers={LayerCount}, output={OutputSize}, backend={gpu})";
        }
    }
}

/**
 * @file
 * @ingroup RNN_Wrappers
 */
using System;
using System.Runtime.InteropServices;

namespace FacadedRnn
{
    internal static class NativeBindings
    {
        private const string LibName = "facaded_rnn_cpp";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rnn_last_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_clear_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rnn_create(
            uint input_size,
            uint[] hidden_sizes,
            uint num_hidden_layers,
            uint output_size,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string cell_type,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string activation,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string output_activation,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string loss,
            double learning_rate,
            double gradient_clip,
            uint bptt_steps,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backend
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rnn_load(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filename,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backend
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_destroy(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_save(IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_predict(
            IntPtr handle, double[] input_data, uint num_timesteps,
            uint input_size, double[]? output_buf, uint output_buf_len
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_train_sequence(
            IntPtr handle, double[] input_data, double[] target_data,
            uint num_timesteps, uint input_size, uint output_size
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_train(
            IntPtr handle, double[] input_data, double[] target_data,
            uint num_timesteps, uint input_size, uint output_size,
            uint epochs, double[] loss_buf, uint loss_buf_len
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_forward_sequence(
            IntPtr handle, double[] input_data, uint num_timesteps,
            uint input_size, double[]? output_buf, uint output_buf_len
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_backward_sequence(
            IntPtr handle, double[] target_data, uint num_timesteps,
            uint output_size
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_reset_states(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint rnn_get_input_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint rnn_get_output_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint rnn_get_layer_count(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint rnn_get_hidden_size(IntPtr handle, uint layer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint rnn_get_sequence_length(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_learning_rate(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_set_learning_rate(IntPtr handle, double lr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_gradient_clip(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_dropout_rate(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_set_dropout_rate(IntPtr handle, double rate);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_is_gpu_available(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_hidden_value(IntPtr handle, uint layer, uint timestep, uint neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_set_hidden_value(IntPtr handle, uint layer, uint neuron, double value);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_output_value(IntPtr handle, uint timestep, uint index);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_cell_state(IntPtr handle, uint layer, uint neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_gate_value(
            IntPtr handle, uint layer, uint timestep, uint neuron,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string gate
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_preactivation(IntPtr handle, uint layer, uint timestep, uint neuron);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rnn_get_input_value(IntPtr handle, uint timestep, uint index);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_get_sequence_outputs(IntPtr handle, double[]? output_buf, uint buf_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rnn_get_sequence_hidden_states(IntPtr handle, uint layer, double[]? output_buf, uint buf_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_detect_vanishing_gradients(IntPtr handle, double threshold, out int count, out double min_val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rnn_detect_exploding_gradients(IntPtr handle, double threshold, out int count, out double max_val);
    }
}

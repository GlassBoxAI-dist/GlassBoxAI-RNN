## @file
## @ingroup RNN_Wrappers
module FacadedRNN

export RNNModel,
    predict, train_sequence!, forward_sequence!, backward_sequence!,
    train!, reset_states!, save_model, load_model,
    input_size, output_size, layer_count, hidden_size, sequence_length,
    learning_rate, set_learning_rate!, gradient_clip,
    dropout_rate, set_dropout_rate!, gpu_available,
    get_hidden_value, set_hidden_value!, get_output_value,
    get_cell_state, get_gate_value, get_preactivation, get_input_value,
    get_sequence_outputs, get_sequence_hidden_states,
    detect_vanishing_gradients, detect_exploding_gradients

const LIB_NAME = Ref{String}("")

function find_library()
    candidates = [
        joinpath(@__DIR__, "..", "..", "cpp", "target", "release", "libfacaded_rnn_cpp.so"),
        joinpath(@__DIR__, "..", "..", "cpp", "target", "release", "libfacaded_rnn_cpp.dylib"),
        joinpath(@__DIR__, "..", "..", "cpp", "target", "release", "facaded_rnn_cpp.dll"),
        joinpath(@__DIR__, "..", "..", "cpp", "target", "debug", "libfacaded_rnn_cpp.so"),
        joinpath(@__DIR__, "..", "..", "cpp", "target", "debug", "libfacaded_rnn_cpp.dylib"),
        joinpath(@__DIR__, "..", "..", "cpp", "target", "debug", "facaded_rnn_cpp.dll"),
    ]
    for c in candidates
        p = abspath(c)
        if isfile(p)
            return p
        end
    end
    error("""
        Could not find libfacaded_rnn_cpp shared library.
        Build with: cd cpp && cargo build --release --no-default-features
        Or set FacadedRNN.set_library_path!("/path/to/libfacaded_rnn_cpp.so")
    """)
end

function set_library_path!(path::String)
    LIB_NAME[] = path
end

function libpath()
    if isempty(LIB_NAME[])
        LIB_NAME[] = find_library()
    end
    LIB_NAME[]
end

struct RNNError <: Exception
    msg::String
end

Base.showerror(io::IO, e::RNNError) = print(io, "RNNError: ", e.msg)

function check_error()
    ptr = ccall((:rnn_last_error, libpath()), Ptr{UInt8}, ())
    if ptr != C_NULL
        msg = unsafe_string(ptr)
        ccall((:rnn_clear_error, libpath()), Cvoid, ())
        throw(RNNError(msg))
    end
end

function check_handle(handle::Ptr{Cvoid})
    if handle == C_NULL
        ptr = ccall((:rnn_last_error, libpath()), Ptr{UInt8}, ())
        if ptr != C_NULL
            msg = unsafe_string(ptr)
            ccall((:rnn_clear_error, libpath()), Cvoid, ())
            throw(RNNError(msg))
        end
        throw(RNNError("Failed to create/load RNN model"))
    end
    handle
end

mutable struct RNNModel
    handle::Ptr{Cvoid}

    function RNNModel(handle::Ptr{Cvoid})
        m = new(check_handle(handle))
        finalizer(m) do obj
            if obj.handle != C_NULL
                ccall((:rnn_destroy, libpath()), Cvoid, (Ptr{Cvoid},), obj.handle)
                obj.handle = C_NULL
            end
        end
        m
    end
end

function RNNModel(;
    input_size::Integer,
    hidden_sizes::Vector{<:Integer},
    output_size::Integer,
    cell_type::String = "lstm",
    activation::String = "tanh",
    output_activation::String = "linear",
    loss::String = "mse",
    learning_rate::Float64 = 0.01,
    gradient_clip::Float64 = 5.0,
    bptt_steps::Integer = 0,
    backend::String = "auto",
)
    hs = UInt32.(hidden_sizes)
    handle = ccall(
        (:rnn_create, libpath()), Ptr{Cvoid},
        (UInt32, Ptr{UInt32}, UInt32, UInt32,
         Cstring, Cstring, Cstring, Cstring,
         Float64, Float64, UInt32, Cstring),
        UInt32(input_size), hs, UInt32(length(hs)), UInt32(output_size),
        cell_type, activation, output_activation, loss,
        learning_rate, gradient_clip, UInt32(bptt_steps), backend,
    )
    RNNModel(handle)
end

function load_model(filename::String; backend::String = "auto")
    handle = ccall(
        (:rnn_load, libpath()), Ptr{Cvoid},
        (Cstring, Cstring),
        filename, backend,
    )
    RNNModel(handle)
end

function save_model(model::RNNModel, filename::String)
    ret = ccall(
        (:rnn_save, libpath()), Int32,
        (Ptr{Cvoid}, Cstring),
        model.handle, filename,
    )
    if ret != 0
        check_error()
        error("Failed to save model")
    end
    nothing
end

function _flatten(data::Vector{Vector{Float64}})
    vcat(data...)
end

function _unflatten(flat::Vector{Float64}, rows::Integer, cols::Integer)
    [flat[(r-1)*cols+1 : r*cols] for r in 1:rows]
end

function predict(model::RNNModel, inputs::Vector{Vector{Float64}})
    flat = _flatten(inputs)
    timesteps = UInt32(length(inputs))
    in_size = timesteps > 0 ? UInt32(length(inputs[1])) : UInt32(0)

    total = ccall(
        (:rnn_predict, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}, UInt32),
        model.handle, flat, timesteps, in_size, C_NULL, UInt32(0),
    )
    if total <= 0
        return Vector{Vector{Float64}}()
    end

    buf = Vector{Float64}(undef, total)
    ccall(
        (:rnn_predict, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}, UInt32),
        model.handle, flat, timesteps, in_size, buf, UInt32(total),
    )

    out_sz = output_size(model)
    _unflatten(buf, Int(timesteps), Int(out_sz))
end

function train_sequence!(model::RNNModel, inputs::Vector{Vector{Float64}}, targets::Vector{Vector{Float64}})
    flat_in = _flatten(inputs)
    flat_tgt = _flatten(targets)
    timesteps = UInt32(length(inputs))
    in_size = timesteps > 0 ? UInt32(length(inputs[1])) : UInt32(0)
    out_size = timesteps > 0 ? UInt32(length(targets[1])) : UInt32(0)

    ccall(
        (:rnn_train_sequence, libpath()), Float64,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, UInt32, UInt32, UInt32),
        model.handle, flat_in, flat_tgt, timesteps, in_size, out_size,
    )
end

function train!(
    model::RNNModel,
    inputs::Vector{Vector{Float64}},
    targets::Vector{Vector{Float64}};
    epochs::Integer = 100,
    verbose::Bool = false,
)
    flat_in = _flatten(inputs)
    flat_tgt = _flatten(targets)
    timesteps = UInt32(length(inputs))
    in_size = timesteps > 0 ? UInt32(length(inputs[1])) : UInt32(0)
    out_size = timesteps > 0 ? UInt32(length(targets[1])) : UInt32(0)

    losses = Vector{Float64}(undef, epochs)
    ccall(
        (:rnn_train, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, UInt32, UInt32, UInt32,
         UInt32, Ptr{Float64}, UInt32),
        model.handle, flat_in, flat_tgt, timesteps, in_size, out_size,
        UInt32(epochs), losses, UInt32(epochs),
    )

    if verbose
        for i in 1:epochs
            if !isnan(losses[i]) && !isinf(losses[i]) && (i % 10 == 0 || i == epochs)
                @printf("Epoch %4d/%d - Loss: %.6f\n", i, epochs, losses[i])
            end
        end
    end

    losses
end

function forward_sequence!(model::RNNModel, inputs::Vector{Vector{Float64}})
    flat = _flatten(inputs)
    timesteps = UInt32(length(inputs))
    in_size = timesteps > 0 ? UInt32(length(inputs[1])) : UInt32(0)

    total = ccall(
        (:rnn_forward_sequence, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}, UInt32),
        model.handle, flat, timesteps, in_size, C_NULL, UInt32(0),
    )
    if total <= 0
        return Vector{Vector{Float64}}()
    end

    buf = Vector{Float64}(undef, total)
    ccall(
        (:rnn_forward_sequence, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}, UInt32),
        model.handle, flat, timesteps, in_size, buf, UInt32(total),
    )

    out_sz = output_size(model)
    _unflatten(buf, Int(timesteps), Int(out_sz))
end

function backward_sequence!(model::RNNModel, targets::Vector{Vector{Float64}})
    flat = _flatten(targets)
    timesteps = UInt32(length(targets))
    out_size = timesteps > 0 ? UInt32(length(targets[1])) : UInt32(0)

    ccall(
        (:rnn_backward_sequence, libpath()), Float64,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32),
        model.handle, flat, timesteps, out_size,
    )
end

function reset_states!(model::RNNModel)
    ccall((:rnn_reset_states, libpath()), Cvoid, (Ptr{Cvoid},), model.handle)
    nothing
end

function input_size(model::RNNModel)
    Int(ccall((:rnn_get_input_size, libpath()), UInt32, (Ptr{Cvoid},), model.handle))
end

function output_size(model::RNNModel)
    Int(ccall((:rnn_get_output_size, libpath()), UInt32, (Ptr{Cvoid},), model.handle))
end

function layer_count(model::RNNModel)
    Int(ccall((:rnn_get_layer_count, libpath()), UInt32, (Ptr{Cvoid},), model.handle))
end

function hidden_size(model::RNNModel, layer::Integer = 0)
    Int(ccall((:rnn_get_hidden_size, libpath()), UInt32, (Ptr{Cvoid}, UInt32), model.handle, UInt32(layer)))
end

function sequence_length(model::RNNModel)
    Int(ccall((:rnn_get_sequence_length, libpath()), UInt32, (Ptr{Cvoid},), model.handle))
end

function learning_rate(model::RNNModel)
    ccall((:rnn_get_learning_rate, libpath()), Float64, (Ptr{Cvoid},), model.handle)
end

function set_learning_rate!(model::RNNModel, lr::Float64)
    ccall((:rnn_set_learning_rate, libpath()), Cvoid, (Ptr{Cvoid}, Float64), model.handle, lr)
    nothing
end

function gradient_clip(model::RNNModel)
    ccall((:rnn_get_gradient_clip, libpath()), Float64, (Ptr{Cvoid},), model.handle)
end

function dropout_rate(model::RNNModel)
    ccall((:rnn_get_dropout_rate, libpath()), Float64, (Ptr{Cvoid},), model.handle)
end

function set_dropout_rate!(model::RNNModel, rate::Float64)
    ccall((:rnn_set_dropout_rate, libpath()), Cvoid, (Ptr{Cvoid}, Float64), model.handle, rate)
    nothing
end

function gpu_available(model::RNNModel)
    ccall((:rnn_is_gpu_available, libpath()), Int32, (Ptr{Cvoid},), model.handle) != 0
end

function get_hidden_value(model::RNNModel, layer::Integer, timestep::Integer, neuron::Integer)
    ccall(
        (:rnn_get_hidden_value, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32, UInt32),
        model.handle, UInt32(layer), UInt32(timestep), UInt32(neuron),
    )
end

function set_hidden_value!(model::RNNModel, layer::Integer, neuron::Integer, value::Float64)
    ccall(
        (:rnn_set_hidden_value, libpath()), Cvoid,
        (Ptr{Cvoid}, UInt32, UInt32, Float64),
        model.handle, UInt32(layer), UInt32(neuron), value,
    )
    nothing
end

function get_output_value(model::RNNModel, timestep::Integer, index::Integer)
    ccall(
        (:rnn_get_output_value, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32),
        model.handle, UInt32(timestep), UInt32(index),
    )
end

function get_cell_state(model::RNNModel, layer::Integer, neuron::Integer)
    ccall(
        (:rnn_get_cell_state, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32),
        model.handle, UInt32(layer), UInt32(neuron),
    )
end

function get_gate_value(model::RNNModel, layer::Integer, timestep::Integer, neuron::Integer, gate::String)
    ccall(
        (:rnn_get_gate_value, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32, UInt32, Cstring),
        model.handle, UInt32(layer), UInt32(timestep), UInt32(neuron), gate,
    )
end

function get_preactivation(model::RNNModel, layer::Integer, timestep::Integer, neuron::Integer)
    ccall(
        (:rnn_get_preactivation, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32, UInt32),
        model.handle, UInt32(layer), UInt32(timestep), UInt32(neuron),
    )
end

function get_input_value(model::RNNModel, timestep::Integer, index::Integer)
    ccall(
        (:rnn_get_input_value, libpath()), Float64,
        (Ptr{Cvoid}, UInt32, UInt32),
        model.handle, UInt32(timestep), UInt32(index),
    )
end

function get_sequence_outputs(model::RNNModel)
    total = ccall(
        (:rnn_get_sequence_outputs, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32),
        model.handle, C_NULL, UInt32(0),
    )
    if total <= 0
        return Vector{Vector{Float64}}()
    end

    buf = Vector{Float64}(undef, total)
    ccall(
        (:rnn_get_sequence_outputs, libpath()), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, UInt32),
        model.handle, buf, UInt32(total),
    )

    seq_len = sequence_length(model)
    out_sz = output_size(model)
    _unflatten(buf, seq_len, out_sz)
end

function get_sequence_hidden_states(model::RNNModel, layer::Integer = 0)
    total = ccall(
        (:rnn_get_sequence_hidden_states, libpath()), Int32,
        (Ptr{Cvoid}, UInt32, Ptr{Float64}, UInt32),
        model.handle, UInt32(layer), C_NULL, UInt32(0),
    )
    if total <= 0
        return Vector{Vector{Float64}}()
    end

    buf = Vector{Float64}(undef, total)
    ccall(
        (:rnn_get_sequence_hidden_states, libpath()), Int32,
        (Ptr{Cvoid}, UInt32, Ptr{Float64}, UInt32),
        model.handle, UInt32(layer), buf, UInt32(total),
    )

    seq_len = sequence_length(model)
    hs = hidden_size(model, Int(layer))
    _unflatten(buf, seq_len, hs)
end

function detect_vanishing_gradients(model::RNNModel, threshold::Float64)
    count = Ref{Int32}(0)
    min_val = Ref{Float64}(0.0)
    ccall(
        (:rnn_detect_vanishing_gradients, libpath()), Cvoid,
        (Ptr{Cvoid}, Float64, Ptr{Int32}, Ptr{Float64}),
        model.handle, threshold, count, min_val,
    )
    (count = Int(count[]), min_gradient = min_val[])
end

function detect_exploding_gradients(model::RNNModel, threshold::Float64)
    count = Ref{Int32}(0)
    max_val = Ref{Float64}(0.0)
    ccall(
        (:rnn_detect_exploding_gradients, libpath()), Cvoid,
        (Ptr{Cvoid}, Float64, Ptr{Int32}, Ptr{Float64}),
        model.handle, threshold, count, max_val,
    )
    (count = Int(count[]), max_gradient = max_val[])
end

function Base.show(io::IO, model::RNNModel)
    print(io, "RNNModel(input=", input_size(model),
        ", hidden=", [hidden_size(model, i) for i in 0:layer_count(model)-1],
        ", output=", output_size(model),
        ", backend=", gpu_available(model) ? "GPU" : "CPU", ")")
end

using Printf

end # module

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using FacadedRNN

println("=== FacadedRNN Julia Example ===\n")

model = RNNModel(
    input_size = 2,
    hidden_sizes = [16],
    output_size = 2,
    cell_type = "lstm",
    backend = "cpu",
)

println("Model: ", model)
println("  Input size:  ", input_size(model))
println("  Output size: ", output_size(model))
println("  Layers:      ", layer_count(model))
println("  Hidden[0]:   ", hidden_size(model, 0))
println("  GPU:         ", gpu_available(model))

inputs  = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
targets = [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

losses = train!(model, inputs, targets; epochs=100)

println("\nTraining:")
println("  Epoch   1 loss: ", round(losses[1]; digits=6))
println("  Epoch 100 loss: ", round(losses[100]; digits=6))

preds = predict(model, [[0.5, 0.5]])
println("\nPrediction for [0.5, 0.5]: ", preds)

save_model(model, "/tmp/julia_test_model.json")
loaded = load_model("/tmp/julia_test_model.json"; backend="cpu")
println("\nReloaded model: ", loaded)

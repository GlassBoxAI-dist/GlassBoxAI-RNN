#include "facaded_rnn.hpp"
#include <iostream>
#include <iomanip>

int main() {
    try {
        facaded_rnn::RNNModel model(
            2,              // input_size
            {16},           // hidden_sizes
            2,              // output_size
            "lstm",         // cell_type
            "tanh",         // activation
            "linear",       // output_activation
            "mse",          // loss
            0.01,           // learning_rate
            5.0,            // gradient_clip
            0,              // bptt_steps
            "cpu"           // backend
        );

        std::cout << "Created model:" << std::endl;
        std::cout << "  Input size:  " << model.input_size() << std::endl;
        std::cout << "  Output size: " << model.output_size() << std::endl;
        std::cout << "  Layers:      " << model.layer_count() << std::endl;
        std::cout << "  Hidden[0]:   " << model.hidden_size(0) << std::endl;
        std::cout << "  GPU:         " << (model.gpu_available() ? "Yes" : "No") << std::endl;

        // Training data: simple sequence
        std::vector<std::vector<double>> inputs  = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
        std::vector<std::vector<double>> targets = {{0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}};

        // Train for 100 epochs
        auto losses = model.train(inputs, targets, 100);

        std::cout << "\nTraining:" << std::endl;
        std::cout << "  Epoch   1 loss: " << std::fixed << std::setprecision(6) << losses[0] << std::endl;
        std::cout << "  Epoch 100 loss: " << losses[99] << std::endl;

        // Predict
        auto predictions = model.predict({{0.5, 0.5}});
        std::cout << "\nPrediction for [0.5, 0.5]: ";
        for (auto& row : predictions) {
            std::cout << "[";
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << row[i];
            }
            std::cout << "]";
        }
        std::cout << std::endl;

        // Save and reload
        model.save("/tmp/test_model.json");
        auto loaded = facaded_rnn::RNNModel::load("/tmp/test_model.json", "cpu");
        std::cout << "\nReloaded model layers: " << loaded.layer_count() << std::endl;

    } catch (const facaded_rnn::RnnError& e) {
        std::cerr << "RNN Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

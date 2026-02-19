/**
 * @file
 * @ingroup RNN_Internal_Logic
 */
/*
 * C API example for facaded_rnn
 *
 * Build:
 *   cd cpp && cargo build --release --no-default-features && cd ..
 *   make -C c
 *
 * Or manually:
 *   gcc -std=c99 -O2 -Ic/include c/example.c \
 *       cpp/target/release/libfacaded_rnn_cpp.a \
 *       -ldl -lpthread -lm -o example_c
 */

#include "facaded_rnn.h"
#include <stdio.h>
#include <stdlib.h>

static void check(const RnnHandle* h, const char* context) {
    if (!h) {
        const char* err = rnn_last_error();
        fprintf(stderr, "%s: %s\n", context, err ? err : "unknown error");
        rnn_clear_error();
        exit(1);
    }
}

int main(void) {
    /* ── Create model ────────────────────────────────────────────── */
    uint32_t hidden[] = {16};
    RnnHandle* model = rnn_create(
        2,          /* input_size */
        hidden, 1,  /* hidden_sizes, num_layers */
        2,          /* output_size */
        "lstm",     /* cell_type */
        "tanh",     /* activation */
        "linear",   /* output_activation */
        "mse",      /* loss */
        0.01,       /* learning_rate */
        5.0,        /* gradient_clip */
        0,          /* bptt_steps */
        "cpu"       /* backend */
    );
    check(model, "rnn_create");

    printf("Created model:\n");
    printf("  Input size:  %u\n", rnn_get_input_size(model));
    printf("  Output size: %u\n", rnn_get_output_size(model));
    printf("  Layers:      %u\n", rnn_get_layer_count(model));
    printf("  Hidden[0]:   %u\n", rnn_get_hidden_size(model, 0));
    printf("  GPU:         %s\n", rnn_is_gpu_available(model) ? "Yes" : "No");

    /* ── Training data ───────────────────────────────────────────── */
    /*  3 timesteps, input_size=2, output_size=2, row-major          */
    double inputs[]  = {0.1, 0.2,   0.3, 0.4,   0.5, 0.6};
    double targets[] = {0.3, 0.4,   0.5, 0.6,   0.7, 0.8};

    /* ── Train 100 epochs ────────────────────────────────────────── */
    double losses[100];
    int32_t trained = rnn_train(
        model,
        inputs, targets,
        3,    /* num_timesteps */
        2,    /* input_size */
        2,    /* output_size */
        100,  /* epochs */
        losses, 100
    );

    printf("\nTraining (%d epochs):\n", trained);
    printf("  Epoch   1 loss: %.6f\n", losses[0]);
    printf("  Epoch 100 loss: %.6f\n", losses[99]);

    /* ── Predict ─────────────────────────────────────────────────── */
    double pred_input[] = {0.5, 0.5};
    double pred_output[2];
    int32_t n = rnn_predict(model, pred_input, 1, 2, pred_output, 2);

    printf("\nPrediction for [0.5, 0.5]: [%.6f, %.6f]  (%d values)\n",
           pred_output[0], pred_output[1], n);

    /* ── Introspection ───────────────────────────────────────────── */
    printf("\nIntrospection:\n");
    printf("  Learning rate:    %.6f\n", rnn_get_learning_rate(model));
    printf("  Gradient clip:    %.2f\n", rnn_get_gradient_clip(model));
    printf("  Dropout rate:     %.6f\n", rnn_get_dropout_rate(model));
    printf("  Sequence length:  %u\n", rnn_get_sequence_length(model));

    /* ── Gradient diagnostics ────────────────────────────────────── */
    int32_t vanish_count;
    double  vanish_min;
    rnn_detect_vanishing_gradients(model, 1e-7, &vanish_count, &vanish_min);
    printf("\n  Vanishing (< 1e-7): count=%d, min=%.10f\n", vanish_count, vanish_min);

    int32_t explode_count;
    double  explode_max;
    rnn_detect_exploding_gradients(model, 100.0, &explode_count, &explode_max);
    printf("  Exploding (> 100):  count=%d, max=%.10f\n", explode_count, explode_max);

    /* ── Save & reload ───────────────────────────────────────────── */
    if (rnn_save(model, "/tmp/c_test_model.json") == 0) {
        printf("\nModel saved to /tmp/c_test_model.json\n");

        RnnHandle* loaded = rnn_load("/tmp/c_test_model.json", "cpu");
        check(loaded, "rnn_load");
        printf("Reloaded: %u layers, hidden[0]=%u\n",
               rnn_get_layer_count(loaded),
               rnn_get_hidden_size(loaded, 0));
        rnn_destroy(loaded);
    }

    /* ── Modify parameters ───────────────────────────────────────── */
    rnn_set_learning_rate(model, 0.005);
    printf("\nLR after set: %.6f\n", rnn_get_learning_rate(model));

    rnn_set_dropout_rate(model, 0.1);
    printf("Dropout after set: %.6f\n", rnn_get_dropout_rate(model));

    /* ── Cleanup ─────────────────────────────────────────────────── */
    rnn_destroy(model);
    printf("\nDone.\n");
    return 0;
}

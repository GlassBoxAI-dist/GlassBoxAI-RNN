package main

import (
	"fmt"
	"log"

	rnn "github.com/glassboxai/facaded_rnn"
)

func main() {
	model, err := rnn.NewRNNModel(rnn.ModelOptions{
		InputSize:   2,
		HiddenSizes: []int{16},
		OutputSize:  2,
		CellType:    "lstm",
		Backend:     "cpu",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	fmt.Println("Created:", model)
	fmt.Printf("  Input size:  %d\n", model.InputSize())
	fmt.Printf("  Output size: %d\n", model.OutputSize())
	fmt.Printf("  Layers:      %d\n", model.LayerCount())
	fmt.Printf("  Hidden[0]:   %d\n", model.HiddenSize(0))
	fmt.Printf("  GPU:         %v\n", model.GPUAvailable())

	inputs := [][]float64{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}
	targets := [][]float64{{0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}}

	losses := model.Train(inputs, targets, 100)
	fmt.Printf("\nTraining:\n")
	fmt.Printf("  Epoch   1 loss: %.6f\n", losses[0])
	fmt.Printf("  Epoch 100 loss: %.6f\n", losses[99])

	preds := model.Predict([][]float64{{0.5, 0.5}})
	fmt.Printf("\nPrediction for [0.5, 0.5]: %v\n", preds)

	if err := model.Save("/tmp/go_test_model.json"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nModel saved to /tmp/go_test_model.json")

	loaded, err := rnn.LoadModel("/tmp/go_test_model.json", "cpu")
	if err != nil {
		log.Fatal(err)
	}
	defer loaded.Close()
	fmt.Printf("Reloaded: %s\n", loaded)

	diag := model.DetectVanishingGradients(1e-7)
	fmt.Printf("\nVanishing gradients (< 1e-7): count=%d, min=%.10f\n", diag.Count, diag.Value)

	fmt.Println("\nDone.")
}

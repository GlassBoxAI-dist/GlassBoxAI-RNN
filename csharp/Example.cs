/**
 * @file
 * @ingroup RNN_Wrappers
 */
using System;
using FacadedRnn;

class Program
{
    static void Main()
    {
        // ── Create model ──
        using var model = RnnModel.Create(new ModelOptions
        {
            InputSize = 2,
            HiddenSizes = new uint[] { 16 },
            OutputSize = 2,
            CellType = "lstm",
            Activation = "tanh",
            OutputActivation = "linear",
            Loss = "mse",
            LearningRate = 0.01,
            GradientClip = 5.0,
            BpttSteps = 0,
            Backend = "cpu",
        });

        Console.WriteLine("Created model:");
        Console.WriteLine($"  Input size:  {model.InputSize}");
        Console.WriteLine($"  Output size: {model.OutputSize}");
        Console.WriteLine($"  Layers:      {model.LayerCount}");
        Console.WriteLine($"  Hidden[0]:   {model.HiddenSize(0)}");
        Console.WriteLine($"  GPU:         {(model.IsGpuAvailable ? "Yes" : "No")}");

        // ── Training data ──
        double[] inputs = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
        double[] targets = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

        // ── Train 100 epochs ──
        double[] losses = model.Train(inputs, targets, 3, 2, 2, 100);

        Console.WriteLine($"\nTraining (100 epochs):");
        Console.WriteLine($"  Epoch   1 loss: {losses[0]:F6}");
        Console.WriteLine($"  Epoch 100 loss: {losses[99]:F6}");

        // ── Predict ──
        double[] predOutput = model.Predict(new double[] { 0.5, 0.5 }, 1, 2);
        Console.WriteLine($"\nPrediction for [0.5, 0.5]: [{predOutput[0]:F6}, {predOutput[1]:F6}]");

        // ── Introspection ──
        Console.WriteLine("\nIntrospection:");
        Console.WriteLine($"  Learning rate:    {model.LearningRate:F6}");
        Console.WriteLine($"  Gradient clip:    {model.GradientClip:F2}");
        Console.WriteLine($"  Dropout rate:     {model.DropoutRate:F6}");
        Console.WriteLine($"  Sequence length:  {model.SequenceLength}");

        // ── Gradient diagnostics ──
        var vanish = model.DetectVanishingGradients(1e-7);
        Console.WriteLine($"\n  Vanishing (< 1e-7): count={vanish.Count}, min={vanish.Value:F10}");

        var explode = model.DetectExplodingGradients(100.0);
        Console.WriteLine($"  Exploding (> 100):  count={explode.Count}, max={explode.Value:F10}");

        // ── Save & reload ──
        model.Save("/tmp/csharp_test_model.json");
        Console.WriteLine("\nModel saved to /tmp/csharp_test_model.json");

        using var loaded = RnnModel.Load("/tmp/csharp_test_model.json", "cpu");
        Console.WriteLine($"Reloaded: {loaded.LayerCount} layers, hidden[0]={loaded.HiddenSize(0)}");

        // ── Modify parameters ──
        model.LearningRate = 0.005;
        Console.WriteLine($"\nLR after set: {model.LearningRate:F6}");

        model.DropoutRate = 0.1;
        Console.WriteLine($"Dropout after set: {model.DropoutRate:F6}");

        Console.WriteLine("\nDone.");
    }
}

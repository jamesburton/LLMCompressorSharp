// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Per-tensor observer that selects (min, max) bounds minimising the MSE between the float
/// tensor and its fake-quantized representation. Useful when outliers would otherwise dilate
/// a min/max range and waste quantization resolution.
/// </summary>
/// <remarks>
/// Mirrors <c>torch.ao.quantization.observer.MovingAverageMSEObserver</c>'s grid search step.
/// The grid sweeps fractions of <c>(absMin, absMax)</c> from <c>1/gridPoints</c> to <c>1.0</c>.
/// </remarks>
public sealed class MSEObserver : Observer
{
    private readonly int _gridPoints;
    private readonly List<Tensor> _samples = new();

    /// <summary>Initializes a new instance of the <see cref="MSEObserver"/> class.</summary>
    /// <param name="gridPoints">Number of grid points (≥ 2). Higher = more accurate, slower.</param>
    public MSEObserver(int gridPoints = 100)
    {
        if (gridPoints < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(gridPoints),
                gridPoints,
                "gridPoints must be ≥ 2.");
        }

        _gridPoints = gridPoints;
    }

    /// <inheritdoc />
    public override QuantizationStrategy Strategy => QuantizationStrategy.PerTensor;

    /// <inheritdoc />
    public override void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        // Keep a CPU-side clone so the underlying tensor can be disposed by the caller.
        _samples.Add(x.cpu().clone().detach());
    }

    /// <inheritdoc />
    public override QuantizationParameters GetQuantParams(int numBits, bool symmetric)
    {
        if (_samples.Count == 0)
        {
            throw new InvalidOperationException(
                "MSEObserver has no calibration samples. Call Update before GetQuantParams.");
        }

        // Concatenate all samples into a single flat tensor for the grid search.
        using var flatList = stack(_samples.Select(s => s.flatten()).ToArray(), dim: 0);
        using var flat = flatList.flatten();

        var absMax = flat.abs().max().item<float>();
        if (absMax == 0f)
        {
            return new QuantizationParameters(tensor(1f), tensor(0L), QuantizationStrategy.PerTensor);
        }

        var bestScale = absMax / ((1 << (numBits - 1)) - 1);
        var bestZp = 0L;
        var bestMse = float.PositiveInfinity;

        for (var i = 1; i <= _gridPoints; i++)
        {
            var fraction = (float)i / _gridPoints;
            var candidateMax = absMax * fraction;
            var candidateMin = -candidateMax;
            var (scale, zeroPoint) = ComputeAffineParams(candidateMin, candidateMax, numBits, symmetric);
            if (scale == 0f)
            {
                continue;
            }

            using var quantized = ApplyFakeQuant(flat, scale, zeroPoint, numBits, symmetric);
            using var err = (flat - quantized).pow(2).mean();
            var mse = err.item<float>();

            if (mse < bestMse)
            {
                bestMse = mse;
                bestScale = scale;
                bestZp = zeroPoint;
            }
        }

        return new QuantizationParameters(
            Scale: tensor(bestScale),
            ZeroPoint: tensor(bestZp),
            Strategy: QuantizationStrategy.PerTensor);
    }

    /// <inheritdoc />
    public override void Reset()
    {
        foreach (var s in _samples)
        {
            s.Dispose();
        }

        _samples.Clear();
    }

    private static Tensor ApplyFakeQuant(Tensor x, float scale, long zeroPoint, int numBits, bool symmetric)
    {
        var qMin = symmetric ? -((1 << (numBits - 1)) - 1) : 0L;
        var qMax = symmetric ? (1 << (numBits - 1)) - 1 : (1 << numBits) - 1;

        // q = clamp(round(x / scale + zero_point), qMin, qMax)
        var q = (x / scale).round_() + zeroPoint;
        q.clamp_(qMin, qMax);

        // x_fq = (q - zero_point) * scale
        return (q - zeroPoint) * scale;
    }
}

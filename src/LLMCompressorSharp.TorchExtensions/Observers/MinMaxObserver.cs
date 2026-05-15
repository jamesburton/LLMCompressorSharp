// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Per-tensor observer that accumulates the static minimum and maximum across calibration samples.
/// </summary>
/// <remarks>
/// Mirrors <c>torch.ao.quantization.observer.MinMaxObserver</c> from PyTorch.
/// </remarks>
public sealed class MinMaxObserver : Observer
{
    private float _min = float.PositiveInfinity;
    private float _max = float.NegativeInfinity;
    private bool _hasSamples;

    /// <inheritdoc />
    public override QuantizationStrategy Strategy => QuantizationStrategy.PerTensor;

    /// <inheritdoc />
    public override void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        // Compute scalars on the same device as the input; pull to CPU for accumulation.
        using var batchMin = x.min();
        using var batchMax = x.max();
        var bMin = batchMin.cpu().item<float>();
        var bMax = batchMax.cpu().item<float>();

        if (bMin < _min)
        {
            _min = bMin;
        }

        if (bMax > _max)
        {
            _max = bMax;
        }

        _hasSamples = true;
    }

    /// <inheritdoc />
    public override QuantizationParameters GetQuantParams(int numBits, bool symmetric)
    {
        if (!_hasSamples)
        {
            throw new InvalidOperationException(
                "MinMaxObserver has no calibration samples. Call Update before GetQuantParams.");
        }

        var (scale, zeroPoint) = ComputeAffineParams(_min, _max, numBits, symmetric);

        return new QuantizationParameters(
            Scale: tensor(scale),
            ZeroPoint: tensor(zeroPoint),
            Strategy: QuantizationStrategy.PerTensor);
    }

    /// <inheritdoc />
    public override void Reset()
    {
        _min = float.PositiveInfinity;
        _max = float.NegativeInfinity;
        _hasSamples = false;
    }
}

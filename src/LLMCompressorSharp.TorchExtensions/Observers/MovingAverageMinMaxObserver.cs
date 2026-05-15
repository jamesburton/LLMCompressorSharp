// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Per-tensor observer that tracks an exponential-moving-average of the per-batch min/max.
/// </summary>
/// <remarks>
/// EMA recurrence: <c>ema = (1 - α) * ema + α * batch</c>.
/// The first batch initializes the running min/max directly. Mirrors
/// <c>torch.ao.quantization.observer.MovingAverageMinMaxObserver</c>.
/// </remarks>
public sealed class MovingAverageMinMaxObserver : Observer
{
    private readonly float _alpha;
    private float _min;
    private float _max;
    private bool _hasSamples;

    /// <summary>Initializes a new instance of the <see cref="MovingAverageMinMaxObserver"/> class.</summary>
    /// <param name="averagingConstant">EMA smoothing factor in (0, 1]. Default mirrors llm-compressor (0.01).</param>
    public MovingAverageMinMaxObserver(float averagingConstant = 0.01f)
    {
        if (averagingConstant is <= 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(averagingConstant),
                averagingConstant,
                "averagingConstant must be in (0, 1].");
        }

        _alpha = averagingConstant;
    }

    /// <inheritdoc />
    public override QuantizationStrategy Strategy => QuantizationStrategy.PerTensor;

    /// <inheritdoc />
    public override void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        using var batchMin = x.min();
        using var batchMax = x.max();
        var bMin = batchMin.cpu().item<float>();
        var bMax = batchMax.cpu().item<float>();

        if (!_hasSamples)
        {
            _min = bMin;
            _max = bMax;
            _hasSamples = true;
        }
        else
        {
            _min = ((1 - _alpha) * _min) + (_alpha * bMin);
            _max = ((1 - _alpha) * _max) + (_alpha * bMax);
        }
    }

    /// <inheritdoc />
    public override QuantizationParameters GetQuantParams(int numBits, bool symmetric)
    {
        if (!_hasSamples)
        {
            throw new InvalidOperationException(
                "MovingAverageMinMaxObserver has no calibration samples. Call Update before GetQuantParams.");
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
        _min = 0f;
        _max = 0f;
        _hasSamples = false;
    }
}

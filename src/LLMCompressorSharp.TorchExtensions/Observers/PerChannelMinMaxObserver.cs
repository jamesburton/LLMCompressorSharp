// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Per-channel observer that accumulates the static minimum and maximum across calibration samples,
/// keeping one running min/max per element along <see cref="ChannelAxis"/>.
/// </summary>
/// <remarks>
/// Mirrors <c>torch.ao.quantization.observer.PerChannelMinMaxObserver</c> from PyTorch.
/// </remarks>
public sealed class PerChannelMinMaxObserver : Observer
{
    private readonly int _channelAxis;
    private Tensor? _min;
    private Tensor? _max;

    /// <summary>
    /// Initializes a new instance of the <see cref="PerChannelMinMaxObserver"/> class
    /// that reduces all axes except <paramref name="channelAxis"/>.
    /// </summary>
    /// <param name="channelAxis">The tensor axis that indexes channels.</param>
    public PerChannelMinMaxObserver(int channelAxis = 0)
    {
        _channelAxis = channelAxis;
    }

    /// <summary>Gets the tensor axis indexing channels.</summary>
    public int ChannelAxis => _channelAxis;

    /// <inheritdoc />
    public override QuantizationStrategy Strategy => QuantizationStrategy.PerChannel;

    /// <inheritdoc />
    public override void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (_channelAxis < 0 || _channelAxis >= x.shape.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(x),
                $"channelAxis {_channelAxis} is out of range for tensor of rank {x.shape.Length}.");
        }

        // Reduce all axes except channelAxis.
        var reduceDims = new List<long>();
        for (var i = 0; i < x.shape.Length; i++)
        {
            if (i != _channelAxis)
            {
                reduceDims.Add(i);
            }
        }

        using var batchMin = x.amin(reduceDims.ToArray(), keepdim: false);
        using var batchMax = x.amax(reduceDims.ToArray(), keepdim: false);

        if (_min is null)
        {
            _min = batchMin.cpu().clone();
            _max = batchMax.cpu().clone();
        }
        else
        {
            using var newMin = torch.minimum(_min, batchMin.cpu());
            using var newMax = torch.maximum(_max!, batchMax.cpu());
            _min.Dispose();
            _max!.Dispose();
            _min = newMin.clone();
            _max = newMax.clone();
        }
    }

    /// <inheritdoc />
    public override QuantizationParameters GetQuantParams(int numBits, bool symmetric)
    {
        if (_min is null || _max is null)
        {
            throw new InvalidOperationException(
                "PerChannelMinMaxObserver has no calibration samples. Call Update before GetQuantParams.");
        }

        var minArr = _min.data<float>().ToArray();
        var maxArr = _max.data<float>().ToArray();
        var n = minArr.Length;

        var scales = new float[n];
        var zeroPoints = new long[n];
        for (var i = 0; i < n; i++)
        {
            (scales[i], zeroPoints[i]) = ComputeAffineParams(minArr[i], maxArr[i], numBits, symmetric);
        }

        return new QuantizationParameters(
            Scale: tensor(scales),
            ZeroPoint: tensor(zeroPoints),
            Strategy: QuantizationStrategy.PerChannel);
    }

    /// <inheritdoc />
    public override void Reset()
    {
        _min?.Dispose();
        _max?.Dispose();
        _min = null;
        _max = null;
    }
}

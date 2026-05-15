// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Observers;

/// <summary>
/// Tests for <see cref="MinMaxObserver"/> — per-tensor static min/max calibration.
/// </summary>
public class MinMaxObserverTests
{
    [Fact]
    public void Strategy_IsPerTensor()
    {
        var observer = new MinMaxObserver();
        observer.Strategy.Should().Be(QuantizationStrategy.PerTensor);
    }

    [Fact]
    public void Update_AccumulatesMinAndMaxAcrossSamples()
    {
        var observer = new MinMaxObserver();

        using var batch1 = tensor(new float[] { -2f, 0f, 3f });
        using var batch2 = tensor(new float[] { -5f, 1f, 2f });
        using var batch3 = tensor(new float[] { -1f, 7f, 0f });

        observer.Update(batch1);
        observer.Update(batch2);
        observer.Update(batch3);

        var p = observer.GetQuantParams(numBits: 8, symmetric: false);
        var scale = p.Scale.item<float>();
        var zeroPoint = p.ZeroPoint.item<long>();

        // Range is [-5, 7]; for 8-bit asymmetric: scale = 12/255 ≈ 0.0471, zero_point ≈ 106.
        scale.Should().BeApproximately(12f / 255f, 1e-5f);
        zeroPoint.Should().Be(106L);
    }

    [Fact]
    public void GetQuantParams_Symmetric8Bit_UsesAbsMaxRange()
    {
        var observer = new MinMaxObserver();
        using var x = tensor(new float[] { -3f, 0f, 5f });
        observer.Update(x);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);

        p.Strategy.Should().Be(QuantizationStrategy.PerTensor);

        // absMax = 5; qMax = 127; scale = 5/127.
        p.Scale.item<float>().Should().BeApproximately(5f / 127f, 1e-5f);
        p.ZeroPoint.item<long>().Should().Be(0L);
    }

    [Fact]
    public void Reset_ClearsAccumulatedStatistics()
    {
        var observer = new MinMaxObserver();
        using var first = tensor(new float[] { -10f, 10f });
        observer.Update(first);

        observer.Reset();

        using var second = tensor(new float[] { -1f, 1f });
        observer.Update(second);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);
        p.Scale.item<float>().Should().BeApproximately(1f / 127f, 1e-5f);
    }

    [Fact]
    public void GetQuantParams_WithoutAnyUpdate_Throws()
    {
        var observer = new MinMaxObserver();
        var act = () => observer.GetQuantParams(numBits: 8, symmetric: true);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*no calibration samples*");
    }

    [Fact]
    public void GetQuantParams_AllZeros_ProducesScaleOfOne()
    {
        var observer = new MinMaxObserver();
        using var zeros = tensor(new float[] { 0f, 0f, 0f });
        observer.Update(zeros);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);
        p.Scale.item<float>().Should().Be(1f);
        p.ZeroPoint.item<long>().Should().Be(0L);
    }
}

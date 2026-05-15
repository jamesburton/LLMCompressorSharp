// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Observers;

/// <summary>
/// Tests for <see cref="MSEObserver"/> — grid-search per-tensor calibration that picks the
/// range minimising MSE between float and fake-quantized representations.
/// </summary>
public class MSEObserverTests
{
    [Fact]
    public void Strategy_IsPerTensor()
    {
        var observer = new MSEObserver(gridPoints: 10);
        observer.Strategy.Should().Be(QuantizationStrategy.PerTensor);
    }

    [Fact]
    public void GetQuantParams_ForUniformData_ProducesNonZeroScale()
    {
        var observer = new MSEObserver(gridPoints: 80);
        using var x = tensor(new float[] { -3f, -2.5f, -1f, 0f, 1f, 2.5f, 3f });
        observer.Update(x);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);

        p.Strategy.Should().Be(QuantizationStrategy.PerTensor);
        p.Scale.item<float>().Should().BePositive();
        p.Scale.item<float>().Should().BeInRange(0.001f, 0.5f);
    }

    [Fact]
    public void GetQuantParams_PicksRangeThatMinimisesMse()
    {
        // Build a dataset with a single moderate outlier just outside the majority range.
        // MSE grid-search should clip the outlier and produce a tighter scale than abs-max.
        // With 99 values in [0, 1] and 1 outlier at 1.5:
        //   - MinMax scale = 1.5 / 7 ≈ 0.214 (full range, includes outlier)
        //   - MSE-optimal clips the outlier; candidateMax ≈ 1.0 gives lower MSE than 1.5
        //     because the quantization step improvement on 99 majority values outweighs
        //     the small clipping cost on 1 value ((1.5-1)^2 / 100 = 0.0025 vs ≈ 0.0038).
        var values = new float[100];
        for (var i = 0; i < 99; i++)
        {
            values[i] = i / 98f; // 0..1 evenly (99 values)
        }

        values[99] = 1.5f; // moderate outlier — just outside the majority range

        var mseObs = new MSEObserver(gridPoints: 100);
        var minMaxObs = new MinMaxObserver();
        using var x = tensor(values);
        mseObs.Update(x);
        minMaxObs.Update(x);

        var mseScale = mseObs.GetQuantParams(numBits: 4, symmetric: true).Scale.item<float>();
        var mmScale = minMaxObs.GetQuantParams(numBits: 4, symmetric: true).Scale.item<float>();

        // MSE-optimal range avoids the outlier and produces a tighter scale than abs-max.
        mseScale.Should().BeLessThan(mmScale);
    }

    [Fact]
    public void Constructor_GridPointsTooLow_Throws()
    {
        var act = () => new MSEObserver(gridPoints: 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GetQuantParams_WithoutAnyUpdate_Throws()
    {
        var observer = new MSEObserver(gridPoints: 10);
        var act = () => observer.GetQuantParams(numBits: 8, symmetric: true);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Reset_ClearsAccumulatedStatistics()
    {
        var observer = new MSEObserver(gridPoints: 10);
        using var first = tensor(new float[] { -10f, 10f });
        observer.Update(first);

        observer.Reset();

        var act = () => observer.GetQuantParams(numBits: 8, symmetric: true);
        act.Should().Throw<InvalidOperationException>();
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Observers;

/// <summary>
/// Tests for <see cref="MovingAverageMinMaxObserver"/> — EMA per-tensor min/max calibration.
/// </summary>
public class MovingAverageMinMaxObserverTests
{
    [Fact]
    public void Strategy_IsPerTensor()
    {
        var observer = new MovingAverageMinMaxObserver(averagingConstant: 0.01f);
        observer.Strategy.Should().Be(QuantizationStrategy.PerTensor);
    }

    [Fact]
    public void Update_FirstBatch_SetsMinAndMaxDirectly()
    {
        var observer = new MovingAverageMinMaxObserver(averagingConstant: 0.5f);
        using var x = tensor(new float[] { -3f, 7f });
        observer.Update(x);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);

        // absMax = 7, scale = 7/127
        p.Scale.item<float>().Should().BeApproximately(7f / 127f, 1e-5f);
    }

    [Fact]
    public void Update_SubsequentBatches_UseExponentialMovingAverage()
    {
        // EMA formula: ema = (1 - alpha) * ema + alpha * batch
        var observer = new MovingAverageMinMaxObserver(averagingConstant: 0.5f);
        using var batch1 = tensor(new float[] { -10f, 10f });
        using var batch2 = tensor(new float[] { -2f, 2f });

        observer.Update(batch1);
        observer.Update(batch2);

        // After batch1: min=-10, max=10
        // After batch2 with alpha=0.5: min = 0.5*-10 + 0.5*-2 = -6; max = 0.5*10 + 0.5*2 = 6
        var p = observer.GetQuantParams(numBits: 8, symmetric: true);
        p.Scale.item<float>().Should().BeApproximately(6f / 127f, 1e-5f);
    }

    [Fact]
    public void Constructor_AveragingConstantOutOfRange_Throws()
    {
        var act = () => new MovingAverageMinMaxObserver(averagingConstant: 1.5f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Reset_ClearsAccumulatedStatistics()
    {
        var observer = new MovingAverageMinMaxObserver(averagingConstant: 0.5f);
        using var first = tensor(new float[] { -100f, 100f });
        observer.Update(first);

        observer.Reset();

        using var second = tensor(new float[] { -1f, 1f });
        observer.Update(second);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);

        // First batch after reset sets directly to [-1, 1]
        p.Scale.item<float>().Should().BeApproximately(1f / 127f, 1e-5f);
    }
}

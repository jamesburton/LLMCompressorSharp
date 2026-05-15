// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Observers;

/// <summary>
/// Tests for <see cref="PerChannelMinMaxObserver"/> — per-channel static min/max calibration.
/// </summary>
public class PerChannelMinMaxObserverTests
{
    [Fact]
    public void Strategy_IsPerChannel()
    {
        var observer = new PerChannelMinMaxObserver(channelAxis: 0);
        observer.Strategy.Should().Be(QuantizationStrategy.PerChannel);
    }

    [Fact]
    public void GetQuantParams_PerOutputChannel_ProducesOneScalePerRow()
    {
        // Two output channels (axis 0), three input dims each.
        var observer = new PerChannelMinMaxObserver(channelAxis: 0);
        using var x = tensor(new float[,]
        {
            { -2f, 0f, 4f },
            { -10f, 5f, 10f },
        });

        observer.Update(x);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);

        p.Strategy.Should().Be(QuantizationStrategy.PerChannel);
        p.Scale.shape.Should().Equal(new long[] { 2 });

        var scales = p.Scale.cpu().data<float>().ToArray();

        // absMax=4 for channel 0.
        scales[0].Should().BeApproximately(4f / 127f, 1e-5f);

        // absMax=10 for channel 1.
        scales[1].Should().BeApproximately(10f / 127f, 1e-5f);
    }

    [Fact]
    public void Update_AccumulatesAcrossSamples()
    {
        var observer = new PerChannelMinMaxObserver(channelAxis: 0);
        using var batch1 = tensor(new float[,]
        {
            { -1f, 2f },
            { -3f, 5f },
        });
        using var batch2 = tensor(new float[,]
        {
            { -5f, 1f },
            { -2f, 8f },
        });

        observer.Update(batch1);
        observer.Update(batch2);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);
        var scales = p.Scale.cpu().data<float>().ToArray();

        // Channel 0: combined [-5, 2] → absMax=5 → 5/127.
        scales[0].Should().BeApproximately(5f / 127f, 1e-5f);

        // Channel 1: combined [-3, 8] → absMax=8 → 8/127.
        scales[1].Should().BeApproximately(8f / 127f, 1e-5f);
    }

    [Fact]
    public void GetQuantParams_WithoutAnyUpdate_Throws()
    {
        var observer = new PerChannelMinMaxObserver(channelAxis: 0);
        var act = () => observer.GetQuantParams(numBits: 8, symmetric: true);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Reset_ClearsAccumulatedStatistics()
    {
        var observer = new PerChannelMinMaxObserver(channelAxis: 0);
        using var first = tensor(new float[,]
        {
            { -10f, 10f },
            { -20f, 20f },
        });
        observer.Update(first);

        observer.Reset();

        using var second = tensor(new float[,]
        {
            { -1f, 1f },
            { -2f, 2f },
        });
        observer.Update(second);

        var p = observer.GetQuantParams(numBits: 8, symmetric: true);
        var scales = p.Scale.cpu().data<float>().ToArray();
        scales[0].Should().BeApproximately(1f / 127f, 1e-5f);
        scales[1].Should().BeApproximately(2f / 127f, 1e-5f);
    }

    [Fact]
    public void ChannelAxis_OutOfRange_Throws()
    {
        var observer = new PerChannelMinMaxObserver(channelAxis: 5);
        using var x = tensor(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f },
        });

        var act = () => observer.Update(x);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

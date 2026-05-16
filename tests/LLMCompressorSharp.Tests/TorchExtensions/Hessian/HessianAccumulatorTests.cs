// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="HessianAccumulator"/> — per-layer running H = Σ Xᵀ X.
/// </summary>
public class HessianAccumulatorTests
{
    [Fact]
    public void Initial_HessianIsZeroMatrix()
    {
        using var accum = new HessianAccumulator(inFeatures: 4);
        accum.SampleCount.Should().Be(0L);

        using var h = accum.GetHessian();
        h.shape.Should().Equal(new long[] { 4, 4 });
        h.cpu().data<float>().ToArray().Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void Update_OneBatch_AccumulatesOuterProduct()
    {
        using var accum = new HessianAccumulator(inFeatures: 3);
        using var x = tensor(new float[,] { { 1f, 2f, 3f } });
        accum.Update(x);

        accum.SampleCount.Should().Be(1L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        arr.Should().Equal(new float[]
        {
            1f, 2f, 3f,
            2f, 4f, 6f,
            3f, 6f, 9f,
        });
    }

    [Fact]
    public void Update_MultipleBatches_AccumulatesAdditively()
    {
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var b1 = tensor(new float[,] { { 1f, 0f } });
        using var b2 = tensor(new float[,] { { 0f, 1f } });
        accum.Update(b1);
        accum.Update(b2);

        accum.SampleCount.Should().Be(2L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        arr.Should().Equal(new float[] { 1f, 0f, 0f, 1f });
    }

    [Fact]
    public void Update_HighRankInput_FlattensCorrectly()
    {
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,,]
        {
            {
                { 1f, 0f },
                { 0f, 1f },
                { 1f, 1f },
            },
            {
                { 1f, 0f },
                { 0f, 1f },
                { 1f, 1f },
            },
        });
        accum.Update(x);

        accum.SampleCount.Should().Be(6L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        arr.Should().Equal(new float[] { 4f, 2f, 2f, 4f });
    }

    [Fact]
    public void Update_WrongInFeatures_Throws()
    {
        using var accum = new HessianAccumulator(inFeatures: 3);
        using var x = tensor(new float[,] { { 1f, 2f } });
        var act = () => accum.Update(x);
        act.Should().Throw<ArgumentException>().WithMessage("*last dim*");
    }

    [Fact]
    public void Update_InputInFp16_AccumulatesInFp32()
    {
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,] { { 1f, 2f } }).to(ScalarType.Float16);
        accum.Update(x);

        using var h = accum.GetHessian();
        h.dtype.Should().Be(ScalarType.Float32);
    }

    [Fact]
    public void Reset_ZerosTheRunningState()
    {
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,] { { 1f, 2f } });
        accum.Update(x);
        accum.Reset();

        accum.SampleCount.Should().Be(0L);
        using var h = accum.GetHessian();
        h.cpu().data<float>().ToArray().Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void Constructor_NonPositiveInFeatures_Throws()
    {
        var act = () => new HessianAccumulator(inFeatures: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

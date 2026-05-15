// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Quantization;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Quantization;

/// <summary>
/// Tests for <see cref="FakeQuantize"/> — differentiable quantization with straight-through-estimator gradient.
/// </summary>
public class FakeQuantizeFunctionTests
{
    [Fact]
    public void Apply_RoundsToNearestQuantizationGrid_Symmetric8Bit()
    {
        // scale = 1.0, zeroPoint = 0, symmetric 8-bit: grid is integer values in [-127, 127]
        using var x = tensor(new float[] { -127.4f, -127.6f, 0.4f, 0.6f, 127.4f, 127.6f });
        using var y = FakeQuantize.Apply(x, scale: 1.0f, zeroPoint: 0L, numBits: 8, symmetric: true);

        var arr = y.data<float>().ToArray();
        arr[0].Should().Be(-127f); // -127.4 rounds to -127
        arr[1].Should().Be(-127f); // -127.6 clamps to -127
        arr[2].Should().Be(0f);
        arr[3].Should().Be(1f);
        arr[4].Should().Be(127f);
        arr[5].Should().Be(127f); // 127.6 clamps to 127
    }

    [Fact]
    public void Apply_RoundsAndDequantizesWithScale_Symmetric4Bit()
    {
        // 4-bit symmetric: grid is {-7, -6, ..., 0, ..., 6, 7}; with scale=0.5, dequantized values are {-3.5, -3, ..., 3, 3.5}
        using var x = tensor(new float[] { -3.6f, 0f, 3.6f });
        using var y = FakeQuantize.Apply(x, scale: 0.5f, zeroPoint: 0L, numBits: 4, symmetric: true);

        var arr = y.data<float>().ToArray();
        arr[0].Should().Be(-3.5f); // clamps to -7 then * 0.5 = -3.5
        arr[1].Should().Be(0f);
        arr[2].Should().Be(3.5f);
    }

    [Fact]
    public void Apply_BackwardIsStraightThroughEstimator()
    {
        using var x = tensor(new float[] { -2f, -1f, 0f, 1f, 2f }, requires_grad: true);

        using var y = FakeQuantize.Apply(x, scale: 1.0f, zeroPoint: 0L, numBits: 4, symmetric: true);
        using var loss = y.sum();
        loss.backward();

        // STE: gradient flows back as 1.0 for every input element (no gating).
        var grad = x.grad;
        grad.Should().NotBeNull();
        grad!.data<float>().ToArray().Should().AllSatisfy(g => g.Should().Be(1f));
    }

    [Fact]
    public void Apply_AsymmetricQuant_AddsAndSubtractsZeroPointCorrectly()
    {
        // Asymmetric 4-bit (0..15): scale=1, zero_point=8
        // x=0 → round(0+8)=8 → (8-8)*1 = 0
        // x=-8 → round(-8+8)=0 → (0-8)*1 = -8
        // x=7 → round(7+8)=15 → (15-8)*1 = 7
        using var x = tensor(new float[] { -8f, 0f, 7f });
        using var y = FakeQuantize.Apply(x, scale: 1.0f, zeroPoint: 8L, numBits: 4, symmetric: false);

        var arr = y.data<float>().ToArray();
        arr[0].Should().Be(-8f);
        arr[1].Should().Be(0f);
        arr[2].Should().Be(7f);
    }

    [Fact]
    public void Apply_InvalidNumBits_Throws()
    {
        using var x = tensor(new float[] { 1f, 2f });
        var act = () => FakeQuantize.Apply(x, scale: 1.0f, zeroPoint: 0L, numBits: 1, symmetric: true);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

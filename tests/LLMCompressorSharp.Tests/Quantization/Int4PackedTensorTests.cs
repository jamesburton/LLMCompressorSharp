// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Quantization;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Quantization;

/// <summary>
/// Tests for <see cref="Int4PackedTensor"/> — FP32-backed INT4 simulation.
/// </summary>
public class Int4PackedTensorTests
{
    [Fact]
    public void FromFloat_QuantizesToInt4Grid_Symmetric()
    {
        // Symmetric 4-bit: grid is {-7, -6, ..., 0, ..., 6, 7}
        using var x = tensor(new float[] { -8.5f, -7f, 0f, 7f, 9f });
        var packed = Int4PackedTensor.FromFloat(x, scale: 1.0f, zeroPoint: 0L, symmetric: true);

        var dequant = packed.Dequantize();
        var arr = dequant.data<float>().ToArray();
        arr[0].Should().Be(-7f); // clamps to -7
        arr[1].Should().Be(-7f);
        arr[2].Should().Be(0f);
        arr[3].Should().Be(7f);
        arr[4].Should().Be(7f);  // clamps to 7
    }

    [Fact]
    public void PackedStorage_HalvesElementCount()
    {
        using var x = tensor(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        var packed = Int4PackedTensor.FromFloat(x, scale: 1.0f, zeroPoint: 0L, symmetric: true);

        // 8 float elements → 4 bytes when packed (two int4 per byte)
        packed.PackedBytes.Length.Should().Be(4);
    }

    [Fact]
    public void RoundTrip_PackUnpack_PreservesQuantizedValues()
    {
        using var original = tensor(new float[] { -7f, -3f, 0f, 4f, 7f, -2f });
        var packed = Int4PackedTensor.FromFloat(original, scale: 1.0f, zeroPoint: 0L, symmetric: true);

        var packedBytes = packed.PackedBytes;
        var roundtripped = Int4PackedTensor.FromPackedBytes(
            packedBytes,
            elementCount: 6,
            scale: 1.0f,
            zeroPoint: 0L,
            symmetric: true);

        var arr = roundtripped.Dequantize().data<float>().ToArray();
        arr.Should().Equal(new float[] { -7f, -3f, 0f, 4f, 7f, -2f });
    }

    [Fact]
    public void Asymmetric_PackUnpack()
    {
        // Asymmetric 4-bit (0..15) with zeroPoint=8 puts 0 at integer 8
        using var x = tensor(new float[] { -8f, 0f, 7f });
        var packed = Int4PackedTensor.FromFloat(x, scale: 1.0f, zeroPoint: 8L, symmetric: false);

        var arr = packed.Dequantize().data<float>().ToArray();
        arr.Should().Equal(new float[] { -8f, 0f, 7f });
    }
}

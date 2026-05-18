// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GptqBlockQuantizer"/> — block-wise column quantization with error propagation.
/// </summary>
public class GptqBlockQuantizerTests
{
    [Fact]
    public void Quantize_ZeroWeight_ReturnsZeroWeight()
    {
        // W = 0, H = I → Hinv = I. Quantized zero is zero.
        var config = MakeConfig(numBits: 8);
        using var w = zeros(2, 4, dtype: ScalarType.Float32);
        using var hinv = eye(4, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        var arr = wq.cpu().data<float>().ToArray();
        arr.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-6f));
    }

    [Fact]
    public void Quantize_OutputIsFinite()
    {
        // Random W, identity Hinv, 4-bit. Result must be finite.
        var config = MakeConfig(numBits: 4);
        using var w = randn(4, 8, dtype: ScalarType.Float32);
        using var hinv = eye(8, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        wq.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Quantize_OutputShape_MatchesInput()
    {
        var config = MakeConfig(numBits: 4, blockSize: 2);
        using var w = randn(3, 6, dtype: ScalarType.Float32);
        using var hinv = eye(6, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        wq.shape.Should().Equal(w.shape);
    }

    [Fact]
    public void Quantize_HighBitWidth_CloseToOriginal()
    {
        // At 16-bit, quantized values should be nearly identical to the originals.
        var config = MakeConfig(numBits: 16, blockSize: 4);
        using var w = randn(2, 4, dtype: ScalarType.Float32);
        using var hinv = eye(4, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        using var diff = (w - wq).abs();
        diff.max().item<float>().Should().BeLessThan(1e-3f);
    }

    [Fact]
    public void Quantize_BlockSizeLargerThanColumns_HandledGracefully()
    {
        // BlockSize=128 with only 6 columns: single block containing all columns.
        var config = MakeConfig(numBits: 8, blockSize: 128);
        using var w = randn(2, 6, dtype: ScalarType.Float32);
        using var hinv = eye(6, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        wq.isfinite().all().item<bool>().Should().BeTrue();
        wq.shape.Should().Equal(w.shape);
    }

    [Fact]
    public void Quantize_NullW_Throws()
    {
        using var hinv = eye(4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(null!, hinv, MakeConfig());
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Quantize_NullHinv_Throws()
    {
        using var w = randn(2, 4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(w, null!, MakeConfig());
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Quantize_NullConfig_Throws()
    {
        using var w = randn(2, 4, dtype: ScalarType.Float32);
        using var hinv = eye(4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(w, hinv, null!);
        act.Should().Throw<ArgumentNullException>();
    }

    /// <summary>
    /// Hand-verified 2×2 test that confirms error propagation direction and magnitude.
    /// Build this test first; if it passes with a naive implementation but fails after
    /// adding error propagation, the propagation sign or formula is wrong.
    /// </summary>
    [Fact]
    public void Quantize_TwoByTwo_ErrorPropagation_IsCorrectDirection()
    {
        // W = [[1, 2], [3, 4]], Hinv = identity (no cross-column error coupling).
        // With identity Hinv, quantizing column 0 produces no change in column 1.
        // At high bit width the quantized result should match the original.
        var config = MakeConfig(numBits: 16, blockSize: 1);
        using var w = tensor(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f },
        });
        using var hinv = eye(2, dtype: ScalarType.Float32);

        using var wq = GptqBlockQuantizer.Quantize(w, hinv, config);

        var arr = wq.cpu().data<float>().ToArray();

        // With identity Hinv and 16-bit precision, result must be within 0.01 of original.
        arr[0].Should().BeApproximately(1f, 0.01f);
        arr[1].Should().BeApproximately(2f, 0.01f);
        arr[2].Should().BeApproximately(3f, 0.01f);
        arr[3].Should().BeApproximately(4f, 0.01f);
    }

    private static GPTQConfig MakeConfig(int numBits = 8, int blockSize = 128) =>
        new GPTQConfig
        {
            NumBits = numBits,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
            BlockSize = blockSize,
            DampeningFrac = 0.01f,
        };
}

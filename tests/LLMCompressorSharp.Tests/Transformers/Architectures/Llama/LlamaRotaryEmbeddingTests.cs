// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaRotaryEmbedding"/> — RoPE rotation.
/// </summary>
public class LlamaRotaryEmbeddingTests
{
    [Fact]
    public void Apply_PreservesShapes()
    {
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 2, 4, 8);
        using var k = randn(1, 2, 4, 8);
        using var positions = arange(4).unsqueeze(0);

        var (qRot, kRot) = rope.Apply(q, k, positions);
        using (qRot)
        using (kRot)
        {
            qRot.shape.Should().Equal(new long[] { 1, 2, 4, 8 });
            kRot.shape.Should().Equal(new long[] { 1, 2, 4, 8 });
        }
    }

    [Fact]
    public void Apply_PreservesVectorMagnitude()
    {
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 1, 4, 8);
        using var k = q.clone();
        using var positions = arange(4).unsqueeze(0);

        var (qRot, _) = rope.Apply(q, k, positions);
        using (qRot)
        {
            using var origNorm = q.pow(2).sum(new long[] { -1L });
            using var rotNorm = qRot.pow(2).sum(new long[] { -1L });
            var origArr = origNorm.cpu().data<float>().ToArray();
            var rotArr = rotNorm.cpu().data<float>().ToArray();
            for (var i = 0; i < origArr.Length; i++)
            {
                rotArr[i].Should().BeApproximately(origArr[i], 1e-3f);
            }
        }
    }

    [Fact]
    public void Apply_PositionZero_IsIdentity()
    {
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 1, 1, 8);
        using var k = q.clone();
        using var positions = zeros(1, 1, dtype: ScalarType.Int64);

        var (qRot, _) = rope.Apply(q, k, positions);
        using (qRot)
        {
            using var diff = (qRot - q).abs().max();
            diff.cpu().item<float>().Should().BeLessThan(1e-5f);
        }
    }

    [Fact]
    public void Constructor_OddHeadDim_Throws()
    {
        var act = () => new LlamaRotaryEmbedding(headDim: 7, maxPositionEmbeddings: 32, theta: 10000f);
        act.Should().Throw<ArgumentException>().WithMessage("*even*");
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Shape and basic-math tests for <see cref="LlamaRMSNorm"/>.
/// </summary>
public class LlamaRMSNormTests
{
    [Fact]
    public void Forward_PreservesInputShape()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-5f);
        using var input = randn(2, 4, 8);
        using var output = norm.forward(input);

        output.shape.Should().Equal(new long[] { 2, 4, 8 });
    }

    [Fact]
    public void Forward_WithUnitWeight_ProducesUnitRmsRow()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-8f);
        using var input = randn(1, 1, 8);
        using var output = norm.forward(input);

        using var sq = output.pow(2);
        using var meanSq = sq.mean(new long[] { -1L }, keepdim: false);
        meanSq.cpu().item<float>().Should().BeApproximately(1f, 1e-3f);
    }

    [Fact]
    public void Forward_WithZeroInput_ProducesZeroOutput()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 4, eps: 1e-5f);
        using var input = zeros(1, 1, 4);
        using var output = norm.forward(input);

        var arr = output.cpu().data<float>().ToArray();
        arr.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-5f));
    }

    [Fact]
    public void Weight_IsLearnableAndExposed()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-5f);
        var parameters = norm.parameters().ToList();
        parameters.Should().HaveCount(1);
        parameters[0].shape.Should().Equal(new long[] { 8 });
    }

    [Fact]
    public void Constructor_NonPositiveHiddenSize_Throws()
    {
        var act = () => new LlamaRMSNorm(hiddenSize: 0, eps: 1e-5f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaMLP"/>.
/// </summary>
public class LlamaMLPTests
{
    [Fact]
    public void Forward_PreservesInputShape()
    {
        using var mlp = new LlamaMLP(hiddenSize: 8, intermediateSize: 32);
        using var x = randn(2, 4, 8);
        using var y = mlp.forward(x);
        y.shape.Should().Equal(new long[] { 2, 4, 8 });
    }

    [Fact]
    public void Parameters_ThreeLinearLayersNoBias()
    {
        using var mlp = new LlamaMLP(hiddenSize: 8, intermediateSize: 32);
        var named = mlp.named_parameters().ToList();
        named.Should().HaveCount(3);
        var names = named.Select(p => p.name).ToList();
        names.Should().Contain(n => n.Contains("gate_proj"));
        names.Should().Contain(n => n.Contains("up_proj"));
        names.Should().Contain(n => n.Contains("down_proj"));
        names.Should().NotContain(n => n.Contains("bias"));
    }

    [Fact]
    public void Constructor_NonPositiveSize_Throws()
    {
        var act = () => new LlamaMLP(hiddenSize: 0, intermediateSize: 32);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

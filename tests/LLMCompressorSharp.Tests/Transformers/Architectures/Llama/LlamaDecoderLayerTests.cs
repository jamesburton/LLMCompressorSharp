// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaDecoderLayer"/>.
/// </summary>
public class LlamaDecoderLayerTests
{
    [Fact]
    public void Forward_PreservesShape()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        using var x = randn(2, 4, config.HiddenSize);
        using var positions = arange(4).unsqueeze(0).expand(2, 4);

        using var y = layer.forward(x, attentionMask: null, positionIds: positions);
        y.shape.Should().Equal(new long[] { 2, 4, config.HiddenSize });
    }

    [Fact]
    public void Forward_ResidualConnection_OutputIsNotIdenticalToInput()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        using var x = randn(1, 2, config.HiddenSize);
        using var positions = arange(2).unsqueeze(0);

        using var y = layer.forward(x, attentionMask: null, positionIds: positions);
        using var diff = (y - x).abs().max();
        diff.cpu().item<float>().Should().BeGreaterThan(1e-5f);
    }

    [Fact]
    public void NamedSubmodules_ContainExpectedComponents()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        var names = layer.named_modules().Select(m => m.name).ToList();
        names.Should().Contain(n => n.Contains("input_layernorm"));
        names.Should().Contain(n => n.Contains("self_attn"));
        names.Should().Contain(n => n.Contains("post_attention_layernorm"));
        names.Should().Contain(n => n.Contains("mlp"));
    }

    private static LlamaConfig MakeConfig()
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 1,
            NumAttentionHeads = 4,
            NumKeyValueHeads = 2,
            VocabSize = 100,
            MaxPositionEmbeddings = 16,
            RopeTheta = 10000f,
            RmsNormEps = 1e-5f,
            HiddenAct = "silu",
            TieWordEmbeddings = false,
        };
    }
}

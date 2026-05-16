// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaAttention"/> with grouped-query attention.
/// </summary>
public class LlamaAttentionTests
{
    [Fact]
    public void Forward_PreservesShape()
    {
        var config = MakeConfig();
        using var attn = new LlamaAttention(config);
        using var x = randn(2, 4, config.HiddenSize);
        using var positions = arange(4).unsqueeze(0).expand(2, 4);

        using var y = attn.forward(x, attentionMask: null, positionIds: positions);
        y.shape.Should().Equal(new long[] { 2, 4, config.HiddenSize });
    }

    [Fact]
    public void Parameters_FourLinearProjectionsNoBias()
    {
        using var attn = new LlamaAttention(MakeConfig());
        var named = attn.named_parameters().ToList();
        named.Select(p => p.name).Should().Contain(n => n.Contains("q_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("k_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("v_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("o_proj"));
        named.Should().HaveCount(4);
    }

    [Fact]
    public void Forward_CausalMask_PreventsAttendingToFutureTokens()
    {
        // With identical first-token input, the output at position 0 must be the same whether we
        // process 1 token or 3 tokens (causal mask isolates position 0 from positions 1, 2).
        var config = MakeConfig();
        using var attn = new LlamaAttention(config);
        using var input = randn(1, 3, config.HiddenSize);
        using var positions = arange(3).unsqueeze(0);

        using var fullOut = attn.forward(input, attentionMask: null, positionIds: positions);

        using var firstOnly = input.narrow(1, 0, 1);
        using var firstPositions = positions.narrow(1, 0, 1);
        using var prefixOut = attn.forward(firstOnly, attentionMask: null, positionIds: firstPositions);

        using var fullFirstToken = fullOut.narrow(1, 0, 1);
        using var diff = (fullFirstToken - prefixOut).abs().max();
        diff.cpu().item<float>().Should().BeLessThan(1e-4f);
    }

    [Fact]
    public void Constructor_HeadDimDoesNotDivideHidden_Throws()
    {
        var config = MakeConfig();
        config.NumAttentionHeads = 3; // 8 / 3 doesn't divide evenly
        var act = () => new LlamaAttention(config);
        act.Should().Throw<ArgumentException>();
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

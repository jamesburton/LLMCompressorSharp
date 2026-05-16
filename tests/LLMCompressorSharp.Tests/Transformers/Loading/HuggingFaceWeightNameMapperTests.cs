// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="HuggingFaceWeightNameMapper"/>.
/// </summary>
public class HuggingFaceWeightNameMapperTests
{
    [Theory]
    [InlineData("model.embed_tokens.weight", "embed_tokens.weight")]
    [InlineData("model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight")]
    [InlineData("model.layers.15.mlp.gate_proj.weight", "layers.15.mlp.gate_proj.weight")]
    [InlineData("model.layers.0.input_layernorm.weight", "layers.0.input_layernorm.weight")]
    [InlineData("model.layers.0.post_attention_layernorm.weight", "layers.0.post_attention_layernorm.weight")]
    [InlineData("model.norm.weight", "norm.weight")]
    [InlineData("lm_head.weight", "lm_head.weight")]
    public void MapName_StripsModelPrefix(string hfName, string expected)
    {
        HuggingFaceWeightNameMapper.MapName(hfName).Should().Be(expected);
    }

    [Theory]
    [InlineData("embed_tokens.weight", "model.embed_tokens.weight")]
    [InlineData("layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight")]
    [InlineData("norm.weight", "model.norm.weight")]
    [InlineData("lm_head.weight", "lm_head.weight")]
    public void UnmapName_AddsModelPrefix_ExceptLmHead(string ourName, string expected)
    {
        HuggingFaceWeightNameMapper.UnmapName(ourName).Should().Be(expected);
    }

    [Fact]
    public void MapDictionary_RemapsAllKeys()
    {
        using var t1 = zeros(2, 3);
        using var t2 = zeros(4);
        var hfDict = new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = t1,
            ["lm_head.weight"] = t2,
        };

        var mapped = HuggingFaceWeightNameMapper.MapDictionary(hfDict);

        mapped.Should().ContainKey("embed_tokens.weight");
        mapped.Should().ContainKey("lm_head.weight");
        mapped.Should().NotContainKey("model.embed_tokens.weight");
    }

    [Fact]
    public void MapName_NullOrEmpty_Throws()
    {
        var act = () => HuggingFaceWeightNameMapper.MapName(null!);
        act.Should().Throw<ArgumentException>();
    }
}

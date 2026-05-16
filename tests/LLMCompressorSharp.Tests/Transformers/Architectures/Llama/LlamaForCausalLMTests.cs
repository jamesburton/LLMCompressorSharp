// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaForCausalLM"/>.
/// </summary>
public class LlamaForCausalLMTests
{
    [Fact]
    public void Forward_ProducesLogitsOfShape_BatchSeqVocab()
    {
        var config = MakeConfig();
        using var model = new LlamaForCausalLM(config);
        using var inputIds = randint(low: 0, high: config.VocabSize, new long[] { 2, 5 }, dtype: ScalarType.Int64);

        using var logits = model.forward(inputIds);
        logits.shape.Should().Equal(new long[] { 2, 5, config.VocabSize });
    }

    [Fact]
    public void NamedSubmodules_ContainExpectedTopLevelComponents()
    {
        var config = MakeConfig();
        using var model = new LlamaForCausalLM(config);
        var names = model.named_modules().Select(m => m.name).ToList();
        names.Should().Contain(n => n.Contains("embed_tokens"));
        names.Should().Contain(n => n.Contains("layers"));
        names.Should().Contain(n => n.Contains("norm"));
        names.Should().Contain(n => n.Contains("lm_head"));
    }

    [Fact]
    public void TieWordEmbeddings_True_SharesEmbeddingAndLmHeadWeights()
    {
        var config = MakeConfig(tied: true);
        using var model = new LlamaForCausalLM(config);

        var embed = model.named_parameters()
            .First(p => p.name.Contains("embed_tokens.weight"))
            .parameter;
        var lmHead = model.named_parameters()
            .First(p => p.name.Contains("lm_head.weight"))
            .parameter;

        embed.shape.Should().Equal(lmHead.shape);
        using var diff = (embed - lmHead).abs().max();
        diff.cpu().item<float>().Should().BeLessThan(1e-6f);
    }

    [Fact]
    public void TieWordEmbeddings_False_HasSeparateLmHead()
    {
        var config = MakeConfig(tied: false);
        using var model = new LlamaForCausalLM(config);

        var paramNames = model.named_parameters().Select(p => p.name).ToList();
        paramNames.Should().Contain(n => n.Contains("lm_head.weight"));
        paramNames.Should().Contain(n => n.Contains("embed_tokens.weight"));
    }

    private static LlamaConfig MakeConfig(bool tied = true)
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 2,
            NumAttentionHeads = 4,
            NumKeyValueHeads = 2,
            VocabSize = 100,
            MaxPositionEmbeddings = 16,
            RopeTheta = 10000f,
            RmsNormEps = 1e-5f,
            HiddenAct = "silu",
            TieWordEmbeddings = tied,
        };
    }
}

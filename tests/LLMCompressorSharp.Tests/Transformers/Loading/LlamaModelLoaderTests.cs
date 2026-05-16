// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="LlamaModelLoader"/>.
/// </summary>
public class LlamaModelLoaderTests
{
    [Fact]
    public void Load_PopulatesAllParameters_WithMatchingShapes()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        try
        {
            using var model = new LlamaForCausalLM(config);

            LlamaModelLoader.Load(model, hfWeights);

            var loadedEmbed = model.named_parameters()
                .First(p => p.name.Contains("embed_tokens.weight"))
                .parameter;
            using var diff = (loadedEmbed - hfWeights["model.embed_tokens.weight"]).abs().max();
            diff.cpu().item<float>().Should().BeLessThan(1e-6f);
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_StrictMode_MissingWeight_Throws()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        hfWeights.Remove("model.layers.0.self_attn.q_proj.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: true);
            act.Should().Throw<HuggingFaceLoadException>().WithMessage("*q_proj*");
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_NonStrictMode_MissingWeight_DoesNotThrow()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        hfWeights.Remove("model.layers.0.self_attn.q_proj.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: false);
            act.Should().NotThrow();
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_ShapeMismatch_Throws()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        hfWeights["model.embed_tokens.weight"].Dispose();
        hfWeights["model.embed_tokens.weight"] = randn(config.VocabSize + 1, config.HiddenSize);
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights);
            act.Should().Throw<HuggingFaceLoadException>().WithMessage("*shape*");
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_TiedEmbeddings_DoesNotRequireLmHeadInSource()
    {
        var config = MakeConfig();
        config.TieWordEmbeddings = true;
        var hfWeights = BuildHfWeights(config);
        hfWeights["lm_head.weight"].Dispose();
        hfWeights.Remove("lm_head.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: true);
            act.Should().NotThrow();
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    private static LlamaConfig MakeConfig()
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 1,
            NumAttentionHeads = 2,
            NumKeyValueHeads = 2,
            VocabSize = 50,
            MaxPositionEmbeddings = 16,
            HiddenAct = "silu",
            TieWordEmbeddings = false,
        };
    }

    private static Dictionary<string, Tensor> BuildHfWeights(LlamaConfig config)
    {
        var headProj = config.NumAttentionHeads * config.HeadDim;
        var kvProj = config.NumKeyValueHeads * config.HeadDim;
        return new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = randn(config.VocabSize, config.HiddenSize),
            ["model.layers.0.self_attn.q_proj.weight"] = randn(headProj, config.HiddenSize),
            ["model.layers.0.self_attn.k_proj.weight"] = randn(kvProj, config.HiddenSize),
            ["model.layers.0.self_attn.v_proj.weight"] = randn(kvProj, config.HiddenSize),
            ["model.layers.0.self_attn.o_proj.weight"] = randn(config.HiddenSize, headProj),
            ["model.layers.0.mlp.gate_proj.weight"] = randn(config.IntermediateSize, config.HiddenSize),
            ["model.layers.0.mlp.up_proj.weight"] = randn(config.IntermediateSize, config.HiddenSize),
            ["model.layers.0.mlp.down_proj.weight"] = randn(config.HiddenSize, config.IntermediateSize),
            ["model.layers.0.input_layernorm.weight"] = randn(config.HiddenSize),
            ["model.layers.0.post_attention_layernorm.weight"] = randn(config.HiddenSize),
            ["model.norm.weight"] = randn(config.HiddenSize),
            ["lm_head.weight"] = randn(config.VocabSize, config.HiddenSize),
        };
    }
}

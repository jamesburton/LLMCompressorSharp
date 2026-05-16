// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="LlamaConfigParser"/> — JSON to LlamaConfig conversion.
/// </summary>
public class LlamaConfigParserTests
{
    [Fact]
    public void Parse_SmolLM2Config_ExtractsAllRequiredFields()
    {
        var json = @"
{
  ""architectures"": [""LlamaForCausalLM""],
  ""hidden_size"": 576,
  ""intermediate_size"": 1536,
  ""num_hidden_layers"": 30,
  ""num_attention_heads"": 9,
  ""num_key_value_heads"": 3,
  ""vocab_size"": 49152,
  ""max_position_embeddings"": 8192,
  ""rope_theta"": 100000.0,
  ""rms_norm_eps"": 1e-5,
  ""hidden_act"": ""silu"",
  ""tie_word_embeddings"": true
}";

        var config = LlamaConfigParser.Parse(json);

        config.HiddenSize.Should().Be(576);
        config.IntermediateSize.Should().Be(1536);
        config.NumHiddenLayers.Should().Be(30);
        config.NumAttentionHeads.Should().Be(9);
        config.NumKeyValueHeads.Should().Be(3);
        config.VocabSize.Should().Be(49152);
        config.MaxPositionEmbeddings.Should().Be(8192);
        config.RopeTheta.Should().BeApproximately(100000f, 0.1f);
        config.RmsNormEps.Should().BeApproximately(1e-5f, 1e-7f);
        config.HiddenAct.Should().Be("silu");
        config.TieWordEmbeddings.Should().BeTrue();
    }

    [Fact]
    public void Parse_IgnoresUnknownFields()
    {
        var json = @"
{
  ""hidden_size"": 16,
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100,
  ""attention_dropout"": 0.1,
  ""some_future_field"": ""whatever""
}";

        var act = () => LlamaConfigParser.Parse(json);
        act.Should().NotThrow();
    }

    [Fact]
    public void Parse_MissingRequiredField_Throws()
    {
        var json = @"
{
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100
}";

        var act = () => LlamaConfigParser.Parse(json);
        act.Should().Throw<HuggingFaceLoadException>()
            .WithMessage("*hidden_size*");
    }

    [Fact]
    public void Parse_DefaultsApplied_WhenOptionalFieldsAbsent()
    {
        var json = @"
{
  ""hidden_size"": 16,
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100
}";

        var config = LlamaConfigParser.Parse(json);

        config.MaxPositionEmbeddings.Should().Be(2048);
        config.RopeTheta.Should().BeApproximately(10000f, 1f);
        config.RmsNormEps.Should().BeApproximately(1e-5f, 1e-7f);
        config.HiddenAct.Should().Be("silu");
        config.TieWordEmbeddings.Should().BeTrue();
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        var act = () => LlamaConfigParser.Parse("not json {");
        act.Should().Throw<HuggingFaceLoadException>();
    }

    [Fact]
    public void LoadFromFile_NonExistentPath_Throws()
    {
        var act = () => LlamaConfigParser.LoadFromFile("/nonexistent/path/config.json");
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void LoadFromFile_ReadsAndParses()
    {
        var tmp = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tmp, @"{
  ""hidden_size"": 8,
  ""intermediate_size"": 32,
  ""num_hidden_layers"": 1,
  ""num_attention_heads"": 2,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 50
}");
            var config = LlamaConfigParser.LoadFromFile(tmp);
            config.HiddenSize.Should().Be(8);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}

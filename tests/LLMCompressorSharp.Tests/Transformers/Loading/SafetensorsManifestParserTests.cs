// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="SafetensorsManifestParser"/>.
/// </summary>
public class SafetensorsManifestParserTests
{
    [Fact]
    public void Parse_ProducesWeightMapAndTotalSize()
    {
        var json = @"
{
  ""metadata"": {""total_size"": 4711234567},
  ""weight_map"": {
    ""model.embed_tokens.weight"": ""model-00001-of-00002.safetensors"",
    ""lm_head.weight"": ""model-00002-of-00002.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);

        manifest.TotalSize.Should().Be(4711234567L);
        manifest.WeightMap.Should().HaveCount(2);
        manifest.WeightMap["model.embed_tokens.weight"].Should().Be("model-00001-of-00002.safetensors");
        manifest.WeightMap["lm_head.weight"].Should().Be("model-00002-of-00002.safetensors");
    }

    [Fact]
    public void Parse_TotalSizeOptional_NullWhenAbsent()
    {
        var json = @"
{
  ""weight_map"": {
    ""x.weight"": ""shard.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);
        manifest.TotalSize.Should().BeNull();
        manifest.WeightMap.Should().HaveCount(1);
    }

    [Fact]
    public void Parse_DistinctShardsList()
    {
        var json = @"
{
  ""weight_map"": {
    ""a"": ""model-00001-of-00003.safetensors"",
    ""b"": ""model-00002-of-00003.safetensors"",
    ""c"": ""model-00001-of-00003.safetensors"",
    ""d"": ""model-00003-of-00003.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);
        manifest.DistinctShardFiles.Should().BeEquivalentTo(new[]
        {
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        });
    }

    [Fact]
    public void Parse_MissingWeightMap_Throws()
    {
        var json = @"{""metadata"": {""total_size"": 100}}";
        var act = () => SafetensorsManifestParser.Parse(json);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*weight_map*");
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        var act = () => SafetensorsManifestParser.Parse("not json");
        act.Should().Throw<HuggingFaceLoadException>();
    }
}

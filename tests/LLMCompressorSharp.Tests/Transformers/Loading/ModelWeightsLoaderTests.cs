// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using TorchSharp.PyBridge;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="ModelWeightsLoader"/> — single-file and sharded safetensors loading.
/// </summary>
public class ModelWeightsLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public ModelWeightsLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"llmc-loader-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }

    [Fact]
    public void Load_SingleFileSafetensors_ReturnsAllTensors()
    {
        using var t1 = tensor(new float[] { 1f, 2f, 3f });
        using var t2 = tensor(new float[] { 4f, 5f });
        var saveDict = new Dictionary<string, Tensor>
        {
            ["embed.weight"] = t1,
            ["lm_head.weight"] = t2,
        };
        var path = Path.Combine(_tempDir, "model.safetensors");
        Safetensors.SaveStateDict(path, saveDict);

        var loaded = ModelWeightsLoader.LoadFromSnapshot(_tempDir);

        loaded.Should().HaveCount(2);
        loaded.Should().ContainKey("embed.weight");
        loaded.Should().ContainKey("lm_head.weight");
        loaded["embed.weight"].cpu().data<float>().ToArray().Should().Equal(new float[] { 1f, 2f, 3f });

        foreach (var t in loaded.Values)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void Load_ShardedSafetensors_MergesAcrossShards()
    {
        using var t1 = tensor(new float[] { 1f, 2f });
        using var t2 = tensor(new float[] { 3f, 4f });
        Safetensors.SaveStateDict(
            Path.Combine(_tempDir, "model-00001-of-00002.safetensors"),
            new Dictionary<string, Tensor> { ["a"] = t1 });
        Safetensors.SaveStateDict(
            Path.Combine(_tempDir, "model-00002-of-00002.safetensors"),
            new Dictionary<string, Tensor> { ["b"] = t2 });

        var manifestJson = @"
{
  ""metadata"": {""total_size"": 32},
  ""weight_map"": {
    ""a"": ""model-00001-of-00002.safetensors"",
    ""b"": ""model-00002-of-00002.safetensors""
  }
}";
        File.WriteAllText(Path.Combine(_tempDir, "model.safetensors.index.json"), manifestJson);

        var loaded = ModelWeightsLoader.LoadFromSnapshot(_tempDir);

        loaded.Should().HaveCount(2);
        loaded["a"].cpu().data<float>().ToArray().Should().Equal(new float[] { 1f, 2f });
        loaded["b"].cpu().data<float>().ToArray().Should().Equal(new float[] { 3f, 4f });

        foreach (var t in loaded.Values)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void Load_NeitherSingleNorSharded_Throws()
    {
        var act = () => ModelWeightsLoader.LoadFromSnapshot(_tempDir);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*model.safetensors*");
    }

    [Fact]
    public void Load_NonExistentDirectory_Throws()
    {
        var act = () => ModelWeightsLoader.LoadFromSnapshot("/nonexistent/path/12345");
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }
}

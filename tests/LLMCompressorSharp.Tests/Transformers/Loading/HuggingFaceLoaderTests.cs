// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using TorchSharp.PyBridge;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for the <see cref="HuggingFaceLoader"/> orchestrator end-to-end.
/// </summary>
public class HuggingFaceLoaderTests : IDisposable
{
    private readonly string _cacheRoot;

    /// <summary>Initializes a new instance of the <see cref="HuggingFaceLoaderTests"/> class.</summary>
    public HuggingFaceLoaderTests()
    {
        _cacheRoot = Path.Combine(Path.GetTempPath(), $"llmc-hfloader-test-{Guid.NewGuid():N}", "hub");
        Directory.CreateDirectory(_cacheRoot);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        try
        {
            var parent = Directory.GetParent(_cacheRoot)?.FullName;
            if (!string.IsNullOrEmpty(parent) && Directory.Exists(parent))
            {
                Directory.Delete(parent, recursive: true);
            }
        }
        catch
        {
            // best-effort
        }
    }

    [Fact]
    public void Load_FromSynthesizedSnapshot_ProducesPopulatedModel()
    {
        var repoId = "FakeOrg/FakeRepo";
        var revision = "main";
        var snapshotDir = Path.Combine(_cacheRoot, HuggingFaceCache.GetRepoFolderName(repoId), "snapshots", revision);
        Directory.CreateDirectory(snapshotDir);

        var configJson = @"
{
  ""hidden_size"": 8,
  ""intermediate_size"": 32,
  ""num_hidden_layers"": 1,
  ""num_attention_heads"": 2,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 50,
  ""max_position_embeddings"": 16,
  ""rope_theta"": 10000.0,
  ""rms_norm_eps"": 1e-5,
  ""hidden_act"": ""silu"",
  ""tie_word_embeddings"": false
}";
        File.WriteAllText(Path.Combine(snapshotDir, "config.json"), configJson);

        var saveDict = new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = randn(50, 8),
            ["model.layers.0.self_attn.q_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.k_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.v_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.o_proj.weight"] = randn(8, 8),
            ["model.layers.0.mlp.gate_proj.weight"] = randn(32, 8),
            ["model.layers.0.mlp.up_proj.weight"] = randn(32, 8),
            ["model.layers.0.mlp.down_proj.weight"] = randn(8, 32),
            ["model.layers.0.input_layernorm.weight"] = randn(8),
            ["model.layers.0.post_attention_layernorm.weight"] = randn(8),
            ["model.norm.weight"] = randn(8),
            ["lm_head.weight"] = randn(50, 8),
        };
        try
        {
            Safetensors.SaveStateDict(Path.Combine(snapshotDir, "model.safetensors"), saveDict);

            var env = new FakeEnvironment { HfHubCache = _cacheRoot };

            var result = HuggingFaceLoader.Load(repoId, revision, env);

            using (result.Model)
            {
                result.Config.HiddenSize.Should().Be(8);
                result.SnapshotPath.Should().Be(snapshotDir);

                using var inputIds = randint(0, 50, new long[] { 1, 3 }, dtype: ScalarType.Int64);
                using var logits = result.Model.forward(inputIds);
                logits.shape.Should().Equal(new long[] { 1, 3, 50 });
            }
        }
        finally
        {
            foreach (var t in saveDict.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_MissingSnapshot_Throws()
    {
        var env = new FakeEnvironment { HfHubCache = _cacheRoot };
        var act = () => HuggingFaceLoader.Load("NoSuch/Repo", "main", env);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void Load_MissingConfigJson_Throws()
    {
        var repoId = "Org/Repo";
        var snapshotDir = Path.Combine(_cacheRoot, HuggingFaceCache.GetRepoFolderName(repoId), "snapshots", "main");
        Directory.CreateDirectory(snapshotDir);

        var env = new FakeEnvironment { HfHubCache = _cacheRoot };
        var act = () => HuggingFaceLoader.Load(repoId, "main", env);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*config.json*");
    }

    private sealed class FakeEnvironment : IEnvironment
    {
        public string? HfHubCache { get; init; }

        public string? HfHome { get; init; }

        public string? XdgCacheHome { get; init; }

        public string UserProfile { get; init; } = "/home/test";
    }
}

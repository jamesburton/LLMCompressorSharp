// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Top-level orchestrator for loading a LLaMA model from the standard HuggingFace cache.
/// </summary>
/// <remarks>
/// Follows the rules in <c>docs/llmcompressorsharp/cache-conventions.md</c>: resolves the cache
/// root via <see cref="HuggingFaceCache"/>, looks up the snapshot directory, parses config.json,
/// and loads safetensors (single-file or sharded). Does not download — assumes the model is
/// already present in the cache. Phase 6 adds a downloader.
/// </remarks>
public static class HuggingFaceLoader
{
    /// <summary>Loads a model from the HuggingFace cache.</summary>
    /// <param name="repoId">Repo id in <c>org/repo</c> form.</param>
    /// <param name="revision">Revision or branch; default <c>main</c>.</param>
    /// <param name="environment">Environment provider for cache resolution; defaults to <see cref="SystemEnvironment.Instance"/>.</param>
    /// <returns>The loaded model + config + snapshot path.</returns>
    /// <exception cref="HuggingFaceLoadException">On any cache miss, parse, or shape error.</exception>
    public static LoadedLlamaModel Load(
        string repoId,
        string? revision = "main",
        IEnvironment? environment = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        var rev = revision ?? "main";
        var env = environment ?? SystemEnvironment.Instance;

        var cacheRoot = HuggingFaceCache.ResolveCacheRoot(env);
        var snapshotDir = HuggingFaceCache.GetSnapshotPath(cacheRoot, repoId, rev);

        if (!Directory.Exists(snapshotDir))
        {
            throw new HuggingFaceLoadException(
                $"Snapshot for '{repoId}@{rev}' not found at '{snapshotDir}'. "
                + $"Run `huggingface-cli download {repoId} --revision {rev}` or set HF_HUB_CACHE.");
        }

        var configPath = Path.Combine(snapshotDir, "config.json");
        if (!File.Exists(configPath))
        {
            throw new HuggingFaceLoadException(
                $"config.json not found in snapshot '{snapshotDir}' for repo '{repoId}@{rev}'.");
        }

        var config = LlamaConfigParser.LoadFromFile(configPath);
        var model = new LlamaForCausalLM(config);

        IDictionary<string, TorchSharp.torch.Tensor>? weights = null;
        try
        {
            weights = ModelWeightsLoader.LoadFromSnapshot(snapshotDir);
            LlamaModelLoader.Load(model, weights);
        }
        catch
        {
            model.Dispose();
            throw;
        }
        finally
        {
            if (weights is not null)
            {
                foreach (var t in weights.Values)
                {
                    t.Dispose();
                }
            }
        }

        return new LoadedLlamaModel(model, config, snapshotDir);
    }
}

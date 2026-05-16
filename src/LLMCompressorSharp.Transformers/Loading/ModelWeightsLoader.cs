// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Loads model weights from a HuggingFace snapshot directory, handling both single-file and
/// sharded safetensors layouts.
/// </summary>
public static class ModelWeightsLoader
{
    private const string SingleFile = "model.safetensors";
    private const string ShardManifestFile = "model.safetensors.index.json";

    /// <summary>Loads all weights from a snapshot directory into a merged dictionary.</summary>
    /// <param name="snapshotDir">The HF cache snapshot directory.</param>
    /// <returns>A dictionary mapping weight name (HF convention) to tensor.</returns>
    /// <exception cref="HuggingFaceLoadException">If the directory is missing or no weight file is present.</exception>
    public static IDictionary<string, Tensor> LoadFromSnapshot(string snapshotDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDir);

        if (!Directory.Exists(snapshotDir))
        {
            throw new HuggingFaceLoadException($"Snapshot directory not found: '{snapshotDir}'.");
        }

        var singleFilePath = Path.Combine(snapshotDir, SingleFile);
        var manifestPath = Path.Combine(snapshotDir, ShardManifestFile);

        if (File.Exists(singleFilePath))
        {
            return LoadSingleFile(singleFilePath);
        }

        if (File.Exists(manifestPath))
        {
            return LoadSharded(snapshotDir, manifestPath);
        }

        throw new HuggingFaceLoadException(
            $"Neither '{SingleFile}' nor '{ShardManifestFile}' found in snapshot '{snapshotDir}'.");
    }

    private static IDictionary<string, Tensor> LoadSingleFile(string path)
    {
        try
        {
            return Safetensors.LoadStateDict(path);
        }
        catch (Exception ex) when (ex is not HuggingFaceLoadException)
        {
            throw new HuggingFaceLoadException($"Failed to load safetensors file '{path}'.", ex);
        }
    }

    private static IDictionary<string, Tensor> LoadSharded(string snapshotDir, string manifestPath)
    {
        string manifestJson;
        try
        {
            manifestJson = File.ReadAllText(manifestPath);
        }
        catch (IOException ex)
        {
            throw new HuggingFaceLoadException($"Failed to read manifest '{manifestPath}'.", ex);
        }

        var manifest = SafetensorsManifestParser.Parse(manifestJson);

        var merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var shardFile in manifest.DistinctShardFiles)
        {
            var shardPath = Path.Combine(snapshotDir, shardFile);
            if (!File.Exists(shardPath))
            {
                throw new HuggingFaceLoadException(
                    $"Shard file '{shardFile}' referenced by manifest is missing in '{snapshotDir}'.");
            }

            IDictionary<string, Tensor> shardWeights;
            try
            {
                shardWeights = Safetensors.LoadStateDict(shardPath);
            }
            catch (Exception ex) when (ex is not HuggingFaceLoadException)
            {
                throw new HuggingFaceLoadException($"Failed to load shard '{shardPath}'.", ex);
            }

            foreach (var (name, tensor) in shardWeights)
            {
                if (merged.ContainsKey(name))
                {
                    throw new HuggingFaceLoadException(
                        $"Weight '{name}' appears in more than one shard.");
                }

                merged[name] = tensor;
            }
        }

        return merged;
    }
}

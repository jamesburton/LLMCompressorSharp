// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parsed contents of a <c>model.safetensors.index.json</c> sharded-checkpoint manifest.
/// </summary>
/// <param name="WeightMap">Map from weight name to the shard filename containing it.</param>
/// <param name="TotalSize">Optional total size in bytes across all shards (HF metadata).</param>
public sealed record SafetensorsManifest(
    IReadOnlyDictionary<string, string> WeightMap,
    long? TotalSize)
{
    /// <summary>Gets the distinct set of shard files referenced by <see cref="WeightMap"/>.</summary>
    public IReadOnlyCollection<string> DistinctShardFiles =>
        new HashSet<string>(this.WeightMap.Values, StringComparer.Ordinal);
}

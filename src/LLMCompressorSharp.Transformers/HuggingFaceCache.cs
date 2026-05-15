// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Resolves paths inside the standard HuggingFace cache layout so weights are shared with
/// other HuggingFace-aware tooling (huggingface_hub, transformers, datasets).
/// </summary>
/// <remarks>
/// Resolution precedence (matches huggingface_hub):
/// <list type="number">
///   <item><c>HF_HUB_CACHE</c> — explicit override of the hub cache root.</item>
///   <item><c>HF_HOME</c> — root for all HF caches; hub cache lives at <c>$HF_HOME/hub</c>.</item>
///   <item><c>XDG_CACHE_HOME</c> — XDG base directory; <c>$XDG_CACHE_HOME/huggingface/hub</c>.</item>
///   <item>User profile fallback: <c>~/.cache/huggingface/hub</c>.</item>
/// </list>
/// </remarks>
public static class HuggingFaceCache
{
    /// <summary>
    /// Returns the path to the HuggingFace hub cache root for the supplied environment.
    /// </summary>
    /// <param name="environment">Environment provider; pass <see cref="SystemEnvironment.Instance"/> in production.</param>
    /// <returns>Absolute path to the hub cache root directory.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="environment"/> is <see langword="null"/>.</exception>
    public static string ResolveCacheRoot(IEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);

        if (!string.IsNullOrEmpty(environment.HfHubCache))
        {
            return environment.HfHubCache;
        }

        if (!string.IsNullOrEmpty(environment.HfHome))
        {
            return Path.Combine(environment.HfHome, "hub");
        }

        if (!string.IsNullOrEmpty(environment.XdgCacheHome))
        {
            return Path.Combine(environment.XdgCacheHome, "huggingface", "hub");
        }

        return Path.Combine(environment.UserProfile, ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Returns the folder name for a HuggingFace repo, matching the layout used by huggingface_hub.
    /// </summary>
    /// <param name="repoId">Repo identifier in <c>org/repo</c> form (e.g. <c>HuggingFaceTB/SmolLM2-135M</c>).</param>
    /// <returns>Folder name (e.g. <c>models--HuggingFaceTB--SmolLM2-135M</c>).</returns>
    /// <exception cref="ArgumentException">If <paramref name="repoId"/> is not in <c>org/repo</c> form.</exception>
    public static string GetRepoFolderName(string repoId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var parts = repoId.Split('/');
        if (parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
        {
            throw new ArgumentException(
                $"Repo id must be in 'org/repo' form. Got: '{repoId}'.",
                nameof(repoId));
        }

        return $"models--{parts[0]}--{parts[1]}";
    }

    /// <summary>
    /// Returns the absolute path to a given snapshot of a repo inside the cache.
    /// </summary>
    /// <param name="cacheRoot">Cache root from <see cref="ResolveCacheRoot"/>.</param>
    /// <param name="repoId">Repo identifier in <c>org/repo</c> form.</param>
    /// <param name="revision">Revision SHA or tag name.</param>
    /// <returns>Absolute snapshot directory path.</returns>
    public static string GetSnapshotPath(string cacheRoot, string repoId, string revision)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);

        return Path.Combine(cacheRoot, GetRepoFolderName(repoId), "snapshots", revision);
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Default <see cref="IEnvironment"/> backed by <see cref="System.Environment"/>.
/// </summary>
public sealed class SystemEnvironment : IEnvironment
{
    /// <summary>Singleton instance.</summary>
    public static readonly SystemEnvironment Instance = new();

    private SystemEnvironment()
    {
    }

    /// <inheritdoc />
    public string? HfHubCache => Environment.GetEnvironmentVariable("HF_HUB_CACHE");

    /// <inheritdoc />
    public string? HfHome => Environment.GetEnvironmentVariable("HF_HOME");

    /// <inheritdoc />
    public string? XdgCacheHome => Environment.GetEnvironmentVariable("XDG_CACHE_HOME");

    /// <inheritdoc />
    public string UserProfile => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Provides access to environment-level inputs that <see cref="HuggingFaceCache"/> needs:
/// HuggingFace environment variables and the current user's home directory. Abstracted to
/// keep cache-resolution tests hermetic.
/// </summary>
public interface IEnvironment
{
    /// <summary>Gets the value of the <c>HF_HUB_CACHE</c> environment variable, or <see langword="null"/>.</summary>
    string? HfHubCache { get; }

    /// <summary>Gets the value of the <c>HF_HOME</c> environment variable, or <see langword="null"/>.</summary>
    string? HfHome { get; }

    /// <summary>Gets the value of the <c>XDG_CACHE_HOME</c> environment variable, or <see langword="null"/>.</summary>
    string? XdgCacheHome { get; }

    /// <summary>Gets the current user's profile directory (e.g. <c>~</c> on Unix, <c>%USERPROFILE%</c> on Windows).</summary>
    string UserProfile { get; }
}

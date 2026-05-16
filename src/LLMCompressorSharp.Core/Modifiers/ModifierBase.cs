// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Abstract base class for modifiers that adds lifecycle ordering enforcement,
/// target/ignore filtering, and a helper to enumerate targeted weights.
/// </summary>
/// <remarks>
/// Subclasses override the <c>On...</c> hooks. <see cref="Initialize"/> through
/// <see cref="Finalize"/> are sealed: they enforce the protocol before calling
/// the hook.
/// </remarks>
public abstract class ModifierBase : IModifier
{
    private bool _initialized;
    private bool _started;
    private bool _ended;
    private bool _finalized;

    /// <summary>Initializes a new instance of the <see cref="ModifierBase"/> class.</summary>
    /// <param name="name">A short identifier (e.g. "GPTQ").</param>
    /// <param name="targets">Name patterns of weights to target; <see langword="null"/> = all.</param>
    /// <param name="ignore">Name patterns to exclude; <see langword="null"/> = none.</param>
    protected ModifierBase(string name, IReadOnlyList<string>? targets = null, IReadOnlyList<string>? ignore = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Targets = targets ?? Array.Empty<string>();
        Ignore = ignore ?? Array.Empty<string>();
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>Gets the configured target patterns.</summary>
    public IReadOnlyList<string> Targets { get; }

    /// <summary>Gets the configured ignore patterns.</summary>
    public IReadOnlyList<string> Ignore { get; }

    /// <inheritdoc />
    public void Initialize(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was initialized twice.");
        }

        _initialized = true;
        OnInitialize(state);
    }

    /// <inheritdoc />
    public void OnStart(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureInitialized();
        if (_started)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was started twice.");
        }

        _started = true;
        OnStartCore(state);
    }

    /// <inheritdoc />
    public void OnBatch(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureStarted();
        if (_ended)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' received a batch after OnEnd.");
        }

        OnBatchCore(state);
    }

    /// <inheritdoc />
    public void OnEnd(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureStarted();
        if (_ended)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was ended twice.");
        }

        _ended = true;
        OnEndCore(state);
    }

    /// <inheritdoc />
    public void Finalize(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was finalized before being initialized.");
        }

        if (_finalized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was finalized twice.");
        }

        _finalized = true;
        OnFinalizeCore(state);
    }

    /// <summary>Implements the modifier's <see cref="IModifier.Initialize"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected abstract void OnInitialize(CompressionState state);

    /// <summary>Implements the modifier's <see cref="IModifier.OnStart"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnStartCore(CompressionState state)
    {
    }

    /// <summary>Implements the modifier's <see cref="IModifier.OnBatch"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnBatchCore(CompressionState state)
    {
    }

    /// <summary>Implements the modifier's <see cref="IModifier.OnEnd"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected abstract void OnEndCore(CompressionState state);

    /// <summary>Implements the modifier's <see cref="IModifier.Finalize"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnFinalizeCore(CompressionState state)
    {
    }

    /// <summary>Enumerates the names from <paramref name="state"/> that match <see cref="Targets"/> minus <see cref="Ignore"/>.</summary>
    /// <param name="state">The session state.</param>
    /// <returns>The targeted names in dictionary-enumeration order.</returns>
    protected IEnumerable<string> GetTargetedNames(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        return TargetMatcher.Filter(state.NamedWeights.Keys, Targets, Ignore);
    }

    /// <summary>Enumerates targeted (name, weight) pairs from the state.</summary>
    /// <param name="state">The session state.</param>
    /// <returns>The targeted (name, tensor) pairs.</returns>
    protected IEnumerable<KeyValuePair<string, Tensor>> GetTargetedWeights(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        foreach (var name in GetTargetedNames(state))
        {
            yield return new KeyValuePair<string, Tensor>(name, state.NamedWeights[name]);
        }
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was used before Initialize.");
        }
    }

    private void EnsureStarted()
    {
        EnsureInitialized();
        if (!_started)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was used before OnStart.");
        }
    }
}

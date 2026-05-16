// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// A compression algorithm exposed as a lifecycle of hooks.
/// </summary>
/// <remarks>
/// Implementations should be stateless across sessions: any state accumulated during one run
/// must be reset in <see cref="Initialize"/> and disposed in <see cref="Finalize"/>.
///
/// <para>Lifecycle: <c>Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize</c>.</para>
/// </remarks>
public interface IModifier
{
    /// <summary>Gets a short identifier used in logs and recipe round-trips (e.g. "GPTQ", "SmoothQuant").</summary>
    string Name { get; }

    /// <summary>Called once before any batches. Allocate observers, hooks, and accumulators.</summary>
    /// <param name="state">The session state.</param>
    void Initialize(CompressionState state);

    /// <summary>Called once after Initialize and before the first batch.</summary>
    /// <param name="state">The session state.</param>
    void OnStart(CompressionState state);

    /// <summary>Called for each calibration batch.</summary>
    /// <param name="state">The session state. <see cref="CompressionState.CurrentBatch"/> is non-null.</param>
    void OnBatch(CompressionState state);

    /// <summary>Called once after the last batch. The modifier should produce its compressed result here.</summary>
    /// <param name="state">The session state.</param>
    void OnEnd(CompressionState state);

    /// <summary>Called once after OnEnd, regardless of success. Dispose any tensors and detach hooks.</summary>
    /// <param name="state">The session state.</param>
    void Finalize(CompressionState state);
}

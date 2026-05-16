// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Orchestrates a list of <see cref="IModifier"/> instances through the
/// <c>Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize</c> lifecycle.
/// </summary>
/// <remarks>
/// Modifiers run in declaration order at every lifecycle phase. If any modifier throws,
/// the session marks itself <see cref="SessionStatus.Failed"/> and still attempts to run
/// <see cref="IModifier.Finalize"/> on previously-initialized modifiers.
/// </remarks>
public sealed class CompressionSession
{
    private readonly IReadOnlyList<IModifier> _modifiers;
    private readonly List<IModifier> _initialized = new();

    /// <summary>Initializes a new instance of the <see cref="CompressionSession"/> class.</summary>
    /// <param name="modifiers">The modifiers to run, in execution order.</param>
    public CompressionSession(IReadOnlyList<IModifier> modifiers)
    {
        ArgumentNullException.ThrowIfNull(modifiers);
        _modifiers = modifiers;
    }

    /// <summary>Gets the current session status.</summary>
    public SessionStatus Status { get; private set; } = SessionStatus.NotStarted;

    /// <summary>Gets the exception that caused failure, if any.</summary>
    public Exception? Failure { get; private set; }

    /// <summary>
    /// Runs the full lifecycle: initialize all modifiers, start, run calibration batches,
    /// end, and finalize.
    /// </summary>
    /// <param name="state">The session state. <see cref="CompressionState.CurrentBatch"/> is set by the session per batch.</param>
    /// <param name="batches">The calibration batches to iterate. May be empty.</param>
    /// <returns>The final <see cref="Status"/>.</returns>
    public SessionStatus Run(CompressionState state, IEnumerable<Tensor> batches)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(batches);

        if (Status != SessionStatus.NotStarted)
        {
            throw new InvalidOperationException("CompressionSession.Run may only be called once.");
        }

        Status = SessionStatus.Running;

        try
        {
            foreach (var m in _modifiers)
            {
                m.Initialize(state);
                _initialized.Add(m);
            }

            foreach (var m in _modifiers)
            {
                m.OnStart(state);
            }

            var index = 0;
            foreach (var batch in batches)
            {
                state.CurrentBatch = batch;
                state.CurrentBatchIndex = index;
                foreach (var m in _modifiers)
                {
                    m.OnBatch(state);
                }

                index++;
            }

            state.CurrentBatch = null;

            foreach (var m in _modifiers)
            {
                m.OnEnd(state);
            }

            Status = SessionStatus.Completed;
        }
        catch (Exception ex)
        {
            Status = SessionStatus.Failed;
            Failure = ex;
        }
        finally
        {
            FinalizeAll(state);
        }

        return Status;
    }

    private void FinalizeAll(CompressionState state)
    {
        foreach (var m in _initialized)
        {
            try
            {
                m.Finalize(state);
            }
            catch (Exception finalizeEx) when (Status == SessionStatus.Failed)
            {
                // Swallow secondary finalize errors when we're already in Failed state — keep the original Failure.
                _ = finalizeEx;
            }
        }
    }
}

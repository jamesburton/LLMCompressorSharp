// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Lifecycle events that IModifier implementations can observe.
/// </summary>
public enum CompressionEvent
{
    /// <summary>Fired once at the start of a session, before any batches.</summary>
    Initialize,

    /// <summary>Fired once after Initialize, before the first batch.</summary>
    Start,

    /// <summary>Fired for each calibration batch.</summary>
    Batch,

    /// <summary>Fired once after the last batch.</summary>
    End,

    /// <summary>Fired once at the very end, after End, for cleanup.</summary>
    Finalize,
}

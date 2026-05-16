// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Lifecycle status of a CompressionSession.
/// </summary>
public enum SessionStatus
{
    /// <summary>The session has been constructed but <c>Run</c> has not yet been called.</summary>
    NotStarted,

    /// <summary>The session is currently running modifiers.</summary>
    Running,

    /// <summary>The session ran to completion and all modifiers finalized successfully.</summary>
    Completed,

    /// <summary>A modifier threw or the session was aborted before completion.</summary>
    Failed,
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Thrown when a modifier's lifecycle methods are invoked in an invalid order
/// (e.g. <c>OnBatch</c> before <c>Initialize</c>).
/// </summary>
public sealed class ModifierLifecycleException : InvalidOperationException
{
    /// <summary>Initializes a new instance of the <see cref="ModifierLifecycleException"/> class.</summary>
    /// <param name="message">Description of the protocol violation.</param>
    public ModifierLifecycleException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ModifierLifecycleException"/> class with an inner exception.</summary>
    /// <param name="message">Description of the protocol violation.</param>
    /// <param name="innerException">The underlying cause.</param>
    public ModifierLifecycleException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

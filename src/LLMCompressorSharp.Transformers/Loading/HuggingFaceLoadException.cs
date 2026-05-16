// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Thrown when <c>HuggingFaceLoader</c> or any of its helpers fails to locate, parse,
/// or apply a HuggingFace model resource.
/// </summary>
public sealed class HuggingFaceLoadException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="HuggingFaceLoadException"/> class.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    public HuggingFaceLoadException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="HuggingFaceLoadException"/> class with an inner cause.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    /// <param name="innerException">Underlying exception.</param>
    public HuggingFaceLoadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Tokenization;

/// <summary>
/// Thrown when a tokenizer cannot be loaded — missing files, malformed JSON, or unsupported tokenizer type.
/// </summary>
public sealed class TokenizerLoadException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="TokenizerLoadException"/> class.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    public TokenizerLoadException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="TokenizerLoadException"/> class with an inner cause.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    /// <param name="innerException">Underlying exception.</param>
    public TokenizerLoadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

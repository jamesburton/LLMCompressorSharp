// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>Thrown when <see cref="RecipeParser"/> cannot parse a recipe YAML document.</summary>
public sealed class RecipeParseException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="RecipeParseException"/> class.</summary>
    /// <param name="message">Diagnostic message.</param>
    public RecipeParseException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="RecipeParseException"/> class with an inner exception.</summary>
    /// <param name="message">Diagnostic message.</param>
    /// <param name="innerException">The underlying YAML error.</param>
    public RecipeParseException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

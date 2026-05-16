// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>A validation rule executed by <see cref="RecipeValidator"/>.</summary>
public interface IRecipeRule
{
    /// <summary>Inspects <paramref name="recipe"/> and appends violation messages to <paramref name="violations"/>.</summary>
    /// <param name="recipe">The recipe under test.</param>
    /// <param name="violations">Mutable list to which violation messages are appended.</param>
    void Check(Recipe recipe, IList<string> violations);
}

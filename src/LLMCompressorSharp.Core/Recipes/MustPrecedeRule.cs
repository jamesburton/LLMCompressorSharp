// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Rule: every modifier of type <see cref="SuccessorType"/> must be preceded by at least one
/// modifier of type <see cref="PredecessorType"/> within the same recipe.
/// </summary>
public sealed class MustPrecedeRule : IRecipeRule
{
    /// <summary>Initializes a new instance of the <see cref="MustPrecedeRule"/> class.</summary>
    /// <param name="predecessorType">The required earlier modifier's type discriminator.</param>
    /// <param name="successorType">The later modifier's type discriminator.</param>
    public MustPrecedeRule(string predecessorType, string successorType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(predecessorType);
        ArgumentException.ThrowIfNullOrWhiteSpace(successorType);
        PredecessorType = predecessorType;
        SuccessorType = successorType;
    }

    /// <summary>Gets the required predecessor type.</summary>
    public string PredecessorType { get; }

    /// <summary>Gets the constrained successor type.</summary>
    public string SuccessorType { get; }

    /// <inheritdoc />
    public void Check(Recipe recipe, IList<string> violations)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentNullException.ThrowIfNull(violations);

        var seenPredecessor = false;
        foreach (var (_, _, config) in recipe.EnumerateModifiers())
        {
            if (config.Type == PredecessorType)
            {
                seenPredecessor = true;
            }
            else if (config.Type == SuccessorType && !seenPredecessor)
            {
                violations.Add(
                    $"Modifier '{SuccessorType}' must be preceded by a '{PredecessorType}' modifier.");
            }
        }
    }
}

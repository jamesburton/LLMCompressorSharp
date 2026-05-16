// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Validates that a <see cref="Recipe"/> obeys cross-modifier ordering constraints.
/// </summary>
/// <remarks>
/// Rules are registered via <see cref="AddRule(IRecipeRule)"/>. Phase 1b ships a single
/// rule — <see cref="MustPrecedeRule"/> — which requires a modifier of one type to
/// precede a modifier of another type within the recipe. Phase 2 adds rules for AWQ
/// preceding QuantizationModifier and similar.
/// </remarks>
public sealed class RecipeValidator
{
    private readonly List<IRecipeRule> _rules = new();

    /// <summary>Adds a rule. Rules execute in registration order.</summary>
    /// <param name="rule">The rule to add.</param>
    public void AddRule(IRecipeRule rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        _rules.Add(rule);
    }

    /// <summary>Validates <paramref name="recipe"/> against all registered rules.</summary>
    /// <param name="recipe">The recipe to validate.</param>
    /// <returns>The set of violations; empty when the recipe is valid.</returns>
    public IReadOnlyList<string> Validate(Recipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        var violations = new List<string>();
        foreach (var rule in _rules)
        {
            rule.Check(recipe, violations);
        }

        return violations;
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Materializes a <see cref="Recipe"/> into the ordered list of <see cref="IModifier"/>
/// instances that a <see cref="Compression.CompressionSession"/> will run.
/// </summary>
public static class RecipeBuilder
{
    /// <summary>Builds modifiers from a recipe, using factories from <see cref="ModifierRegistry"/>.</summary>
    /// <param name="recipe">The parsed recipe.</param>
    /// <returns>Modifiers in declaration order (stage order, then modifier order within a stage).</returns>
    /// <exception cref="RecipeParseException">If a referenced modifier type is unregistered.</exception>
    public static IReadOnlyList<IModifier> Build(Recipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        var built = new List<IModifier>();
        foreach (var (_, _, config) in recipe.EnumerateModifiers())
        {
            var registration = ModifierRegistry.Resolve(config.Type)
                ?? throw new RecipeParseException(
                    $"Modifier type '{config.Type}' is not registered. Did you call ModifierRegistry.Register<...>?");
            built.Add(registration.Factory(config));
        }

        return built;
    }
}

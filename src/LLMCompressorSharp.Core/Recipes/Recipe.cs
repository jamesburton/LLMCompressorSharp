// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// A compression recipe: an ordered list of stages, each containing modifier configurations.
/// </summary>
public sealed class Recipe
{
    /// <summary>Gets or sets the stages of this recipe.</summary>
    public IList<Stage> Stages { get; set; } = new List<Stage>();

    /// <summary>Enumerates every modifier config across all stages in order.</summary>
    /// <returns>The (stage-index, stage-name, modifier-config) triples.</returns>
    public IEnumerable<(int StageIndex, string StageName, ModifierConfig Config)> EnumerateModifiers()
    {
        for (var i = 0; i < Stages.Count; i++)
        {
            var stage = Stages[i];
            foreach (var m in stage.Modifiers)
            {
                yield return (i, stage.Name, m);
            }
        }
    }
}

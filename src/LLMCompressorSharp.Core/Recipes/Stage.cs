// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// A named stage within a <see cref="Recipe"/>. Stages execute in declaration order and
/// each stage's modifiers run sequentially within it.
/// </summary>
public sealed class Stage
{
    /// <summary>Gets or sets the stage's human-readable name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Gets or sets the modifier configurations belonging to this stage.</summary>
    public IList<ModifierConfig> Modifiers { get; set; } = new List<ModifierConfig>();
}

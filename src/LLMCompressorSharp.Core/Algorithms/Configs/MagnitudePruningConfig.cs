// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>MagnitudePruningModifier</c> — zero out weights below a magnitude threshold.
/// </summary>
public sealed class MagnitudePruningConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "MagnitudePruning";

    /// <summary>Gets or sets the target sparsity ratio in [0, 1]. Default: 0.5.</summary>
    /// <remarks>0.0 = no pruning; 1.0 = all weights zeroed.</remarks>
    public float Sparsity { get; set; } = 0.5f;
}

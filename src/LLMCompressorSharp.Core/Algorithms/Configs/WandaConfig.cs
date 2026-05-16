// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>WandaModifier</c> — pruning by <c>|w| × ||x||₂</c> saliency.
/// </summary>
/// <remarks>
/// The activation key for each targeted weight is the weight name with the <c>.weight</c> suffix
/// removed (e.g. weight <c>model.layer.0.q_proj.weight</c> → activation <c>model.layer.0.q_proj</c>).
/// </remarks>
public sealed class WandaConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "WANDA";

    /// <summary>Gets or sets the target sparsity ratio in [0, 1]. Default: 0.5.</summary>
    public float Sparsity { get; set; } = 0.5f;
}

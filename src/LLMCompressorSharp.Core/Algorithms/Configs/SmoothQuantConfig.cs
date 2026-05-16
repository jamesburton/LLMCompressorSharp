// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>SmoothQuantModifier</c> — channel-wise activation/weight rebalancing.
/// </summary>
public sealed class SmoothQuantConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "SmoothQuant";

    /// <summary>Gets or sets alpha — the smoothing strength in (0, 1). Default: 0.5.</summary>
    /// <remarks>
    /// <c>alpha = 0.5</c> migrates difficulty equally between activations and weights.
    /// <c>alpha → 1</c> migrates more difficulty into weights; <c>alpha → 0</c> keeps it in activations.
    /// </remarks>
    public float SmoothingStrength { get; set; } = 0.5f;

    /// <summary>Gets or sets the smooth ↔ balance mappings to process.</summary>
    public IList<SmoothQuantMapping> Mappings { get; set; } = new List<SmoothQuantMapping>();
}

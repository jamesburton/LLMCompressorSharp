// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;
using LLMCompressorSharp.TorchExtensions.Observers;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>RtnModifier</c> — round-to-nearest weight quantization.
/// </summary>
public sealed class RtnConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "RTN";

    /// <summary>Gets or sets the target bit-width. Default: 8.</summary>
    public int NumBits { get; set; } = 8;

    /// <summary>Gets or sets a value indicating whether to use symmetric quantization. Default: true.</summary>
    public bool Symmetric { get; set; } = true;

    /// <summary>Gets or sets the quantization strategy. Default: <see cref="QuantizationStrategy.PerTensor"/>.</summary>
    public QuantizationStrategy Strategy { get; set; } = QuantizationStrategy.PerTensor;

    /// <summary>Gets or sets the channel axis for per-channel quantization. Default: 0.</summary>
    public int ChannelAxis { get; set; } = 0;
}

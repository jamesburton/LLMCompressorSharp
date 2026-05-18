// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;
using LLMCompressorSharp.TorchExtensions.Observers;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>GPTQModifier</c> — Hessian-based block-wise weight quantization.
/// </summary>
/// <remarks>
/// Corresponds to the Python <c>GPTQModifier</c> knobs in llm-compressor.
/// The default scheme is W4A16 (4-bit weights, 16-bit activations):
/// <c>NumBits=4, Symmetric=true, Strategy=PerChannel, BlockSize=128, DampeningFrac=0.01</c>.
/// </remarks>
public sealed class GPTQConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "GPTQ";

    /// <summary>Gets or sets the target bit-width. Default: 4.</summary>
    public int NumBits { get; set; } = 4;

    /// <summary>Gets or sets a value indicating whether symmetric quantization is used. Default: true.</summary>
    /// <remarks>Symmetric = zero-point is always 0; range is [-2^(b-1)+1, 2^(b-1)-1].</remarks>
    public bool Symmetric { get; set; } = true;

    /// <summary>Gets or sets the quantization granularity. Default: <see cref="QuantizationStrategy.PerChannel"/>.</summary>
    /// <remarks>
    /// Per-channel at axis 0 means one scale/zero-point per output channel of the Linear weight
    /// (<c>[outFeatures, inFeatures]</c>). This is the W4A16 standard.
    /// </remarks>
    public QuantizationStrategy Strategy { get; set; } = QuantizationStrategy.PerChannel;

    /// <summary>Gets or sets the channel axis for per-channel quantization. Default: 0 (output channels of Linear.weight).</summary>
    public int ChannelAxis { get; set; } = 0;

    /// <summary>Gets or sets the number of weight columns processed per block iteration. Default: 128.</summary>
    public int BlockSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the Hessian diagonal dampening fraction. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// Applied as <c>H += dampening_frac × mean(diag(H)) × I</c> before Cholesky.
    /// The GPTQ paper uses 1%; increase to 0.1 for small calibration sets or poorly
    /// conditioned layers. Has no effect when the Hessian is already well-conditioned.
    /// </remarks>
    public float DampeningFrac { get; set; } = 0.01f;

    /// <summary>
    /// Gets or sets a value indicating whether to reorder columns by descending Hessian diagonal
    /// before quantization (activation ordering). Default: false.
    /// </summary>
    /// <remarks>
    /// Not yet implemented. Setting this to <see langword="true"/> causes <c>GPTQModifier</c>
    /// to throw <see cref="NotSupportedException"/>.
    /// </remarks>
    public bool ActOrder { get; set; } = false;
}

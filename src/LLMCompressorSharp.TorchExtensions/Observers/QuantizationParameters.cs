// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Quantization parameters computed by an <see cref="Observer"/>.
/// </summary>
/// <remarks>
/// For per-tensor strategies, <see cref="Scale"/> and <see cref="ZeroPoint"/> are scalar tensors.
/// For per-channel strategies, they are 1-D tensors with one element per output channel.
/// <see cref="GlobalScale"/> is optional and used by group-quantization schemes that fuse multiple
/// observers under a shared outer scale.
/// </remarks>
/// <param name="Scale">The quantization scale.</param>
/// <param name="ZeroPoint">The quantization zero-point.</param>
/// <param name="Strategy">The strategy this parameter set was computed under.</param>
/// <param name="GlobalScale">Optional outer scale shared across fused observers.</param>
public sealed record QuantizationParameters(
    Tensor Scale,
    Tensor ZeroPoint,
    QuantizationStrategy Strategy,
    Tensor? GlobalScale = null);

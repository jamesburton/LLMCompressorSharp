// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// The granularity at which an <see cref="Observer"/> accumulates statistics
/// and an <see cref="LLMCompressorSharp.TorchExtensions.Quantization.FakeQuantizeFunction"/> applies quantization.
/// </summary>
public enum QuantizationStrategy
{
    /// <summary>A single scale and zero-point for the entire tensor.</summary>
    PerTensor,

    /// <summary>One scale and zero-point per output channel.</summary>
    PerChannel,

    /// <summary>One scale and zero-point per token (typically axis 1 of an activation tensor).</summary>
    PerToken,
}

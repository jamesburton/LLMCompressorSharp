// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Quantization;

/// <summary>
/// Convenience helpers for applying <see cref="FakeQuantizeFunction"/>.
/// </summary>
public static class FakeQuantize
{
    /// <summary>
    /// Applies symmetric or asymmetric fake-quantization to <paramref name="x"/>.
    /// </summary>
    /// <param name="x">Input tensor.</param>
    /// <param name="scale">Quantization scale.</param>
    /// <param name="zeroPoint">Quantization zero-point.</param>
    /// <param name="numBits">Target bit-width (2 ≤ numBits ≤ 16).</param>
    /// <param name="symmetric">When true, the integer grid is [-(2^(b-1)-1), 2^(b-1)-1].</param>
    /// <returns>The fake-quantized tensor (autograd-friendly).</returns>
    /// <exception cref="ArgumentOutOfRangeException">If <paramref name="numBits"/> is outside [2, 16].</exception>
    public static Tensor Apply(Tensor x, float scale, long zeroPoint, int numBits, bool symmetric)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (numBits is < 2 or > 16)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numBits),
                numBits,
                "numBits must be between 2 and 16 inclusive.");
        }

        var qMin = symmetric ? -((1L << (numBits - 1)) - 1) : 0L;
        var qMax = symmetric ? (1L << (numBits - 1)) - 1 : (1L << numBits) - 1;

        return FakeQuantizeFunction.apply(x, scale, zeroPoint, qMin, qMax);
    }
}

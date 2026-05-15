// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Observers;

/// <summary>
/// Calibration-statistics collector that computes quantization parameters
/// (scale, zero-point) from sample tensors.
/// </summary>
/// <remarks>
/// Mirrors <c>torch.ao.quantization.observer.ObserverBase</c> from PyTorch.
/// Tracked upstream as PR_TO_TORCHSHARP R-002.
/// </remarks>
// TODO(PR_TO_TORCHSHARP R-002): Promote this hierarchy upstream as torch.ao.quantization.observer.* equivalents.
public abstract class Observer
{
    /// <summary>Gets the granularity at which this observer accumulates statistics.</summary>
    public abstract QuantizationStrategy Strategy { get; }

    /// <summary>
    /// Incorporates a calibration sample into the observer's running statistics.
    /// </summary>
    /// <param name="x">A sample tensor; treated as read-only.</param>
    public abstract void Update(Tensor x);

    /// <summary>
    /// Computes the quantization scale and zero-point implied by the accumulated statistics.
    /// </summary>
    /// <param name="numBits">Target bit-width (e.g. 4 or 8).</param>
    /// <param name="symmetric">True for symmetric quantization (zero-point is 0); false for asymmetric.</param>
    /// <returns>A <see cref="QuantizationParameters"/> appropriate for <see cref="Strategy"/>.</returns>
    public abstract QuantizationParameters GetQuantParams(int numBits, bool symmetric);

    /// <summary>Clears the accumulated statistics, returning the observer to its initial state.</summary>
    public abstract void Reset();

    /// <summary>
    /// Computes a scale + zero-point pair from min/max bounds, applying the symmetric rule when requested.
    /// </summary>
    /// <param name="min">Minimum value across the calibration set.</param>
    /// <param name="max">Maximum value across the calibration set.</param>
    /// <param name="numBits">Target bit-width.</param>
    /// <param name="symmetric">When true, the range is symmetric around zero.</param>
    /// <returns>A tuple of (scale, zeroPoint).</returns>
    protected static (float Scale, long ZeroPoint) ComputeAffineParams(
        float min,
        float max,
        int numBits,
        bool symmetric)
    {
        if (numBits is < 2 or > 16)
        {
            throw new ArgumentOutOfRangeException(nameof(numBits), numBits, "numBits must be between 2 and 16 inclusive.");
        }

        if (symmetric)
        {
            var absMax = MathF.Max(MathF.Abs(min), MathF.Abs(max));
            var qMax = (1 << (numBits - 1)) - 1;
            var scale = absMax == 0f ? 1f : absMax / qMax;
            return (scale, 0L);
        }
        else
        {
            var qMin = 0;
            var qMax = (1 << numBits) - 1;
            var range = max - min;
            var scale = range == 0f ? 1f : range / (qMax - qMin);
            var zeroPoint = (long)MathF.Round(qMin - (min / scale));
            zeroPoint = Math.Clamp(zeroPoint, qMin, qMax);
            return (scale, zeroPoint);
        }
    }
}

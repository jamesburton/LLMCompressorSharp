// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using static TorchSharp.torch;
using static TorchSharp.torch.autograd;

namespace LLMCompressorSharp.TorchExtensions.Quantization;

/// <summary>
/// Custom autograd function implementing fake quantization with a straight-through-estimator backward.
/// </summary>
/// <remarks>
/// Forward: <c>x_fq = (round(x / scale + zero_point).clamp(qMin, qMax) - zero_point) * scale</c><br/>
/// Backward: <c>dx = dy</c> (STE: identity gradient).
///
/// <para>Tracked upstream as PR_TO_TORCHSHARP R-001 (mirrors <c>torch.fake_quantize_per_tensor_affine</c>).</para>
/// </remarks>
// TODO(PR_TO_TORCHSHARP R-001): Replace with torch.fake_quantize_per_tensor_affine when upstream lands.
public sealed class FakeQuantizeFunction : SingleTensorFunction<FakeQuantizeFunction>
{
    /// <inheritdoc />
    public override string Name => "FakeQuantize";

    /// <summary>Forward pass: round-clamp-dequantize.</summary>
    /// <param name="ctx">Autograd context for saving tensors and metadata across forward/backward.</param>
    /// <param name="vars">
    /// Positional arguments: <c>vars[0]</c> = input tensor; <c>vars[1]</c> = scale (float);
    /// <c>vars[2]</c> = zeroPoint (long); <c>vars[3]</c> = qMin (long); <c>vars[4]</c> = qMax (long).
    /// </param>
    /// <returns>The fake-quantized tensor.</returns>
    public override Tensor forward(AutogradContext ctx, object[] vars)
    {
        var x = (Tensor)vars[0];
        var scale = (float)vars[1];
        var zeroPoint = (long)vars[2];
        var qMin = (long)vars[3];
        var qMax = (long)vars[4];

        // No tensors need to be saved for the STE backward (identity gradient — no input dependence).
        ctx.save_for_backward(new List<Tensor>());

        using var divided = x.div(scale);
        using var rounded = divided.round();
        using var shifted = rounded.add(zeroPoint);
        using var clamped = shifted.clamp(qMin, qMax);
        using var unshifted = clamped.sub(zeroPoint);
        return unshifted.mul(scale);
    }

    /// <summary>Backward pass: straight-through estimator (identity gradient w.r.t. <c>x</c>).</summary>
    /// <param name="ctx">Autograd context (unused for STE).</param>
    /// <param name="grad_output">Upstream gradient flowing into the fake-quantize output.</param>
    /// <returns>
    /// A list with one entry per input variable: the upstream gradient for <c>x</c>,
    /// followed by <c>null</c> entries for the non-differentiable scalar parameters.
    /// </returns>
    public override List<Tensor> backward(AutogradContext ctx, Tensor grad_output)
    {
        // STE: pass the upstream gradient straight through to x (vars[0]).
        // Null entries correspond to non-differentiable scalar arguments: scale, zeroPoint, qMin, qMax.
        return
        [
            grad_output,
            null!,
            null!,
            null!,
            null!,
        ];
    }
}

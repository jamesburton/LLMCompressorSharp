// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Computes the inverse Hessian used by GPTQ and SparseGPT via damped Cholesky decomposition.
/// </summary>
/// <remarks>
/// The computation is:
/// <list type="number">
///   <item>Apply diagonal dampening: <c>H' = H + λ·mean(diag(H))·I</c>.</item>
///   <item>Cholesky: <c>L = chol(H')</c> (lower-triangular factor).</item>
///   <item>Invert: <c>L⁻¹ = solve_triangular(L, I, upper=false)</c>.</item>
///   <item>Return: <c>H⁻¹ = L⁻ᵀ · L⁻¹</c>.</item>
/// </list>
/// If Cholesky fails (non-positive-definite after damping), falls back to <c>linalg.pinv</c>.
///
/// <para>
/// <b>Shared with Phase 4c (SparseGPT).</b> SparseGPT uses the same Hinv but replaces
/// the fake-quantize inner step with a mask-selection step.
/// </para>
/// </remarks>
public static class HessianInverseSolver
{
    /// <summary>
    /// Computes <c>H⁻¹</c> with diagonal damping for numerical stability.
    /// </summary>
    /// <param name="hessian">
    /// The accumulated Hessian <c>[d, d]</c>. Must be a square FP32 tensor.
    /// The input tensor is not modified — a working copy is used.
    /// </param>
    /// <param name="dampingFrac">
    /// Fraction of the mean diagonal to add as damping. Typical value: 0.01.
    /// Pass 0 to skip damping (only safe for perfectly conditioned matrices).
    /// </param>
    /// <returns>
    /// The inverse Hessian <c>[d, d]</c> (FP32). The caller owns disposal.
    /// </returns>
    /// <exception cref="ArgumentNullException">If <paramref name="hessian"/> is null.</exception>
    /// <exception cref="ArgumentException">If <paramref name="hessian"/> is not a square 2-D tensor.</exception>
    public static Tensor Compute(Tensor hessian, float dampingFrac)
    {
        ArgumentNullException.ThrowIfNull(hessian);

        if (hessian.ndim != 2 || hessian.shape[0] != hessian.shape[1])
        {
            throw new ArgumentException(
                $"Hessian must be a square 2-D tensor; got shape [{string.Join(", ", hessian.shape)}].",
                nameof(hessian));
        }

        var d = hessian.shape[0];

        // Work on a FP32 clone — never modify the caller's tensor.
        // Detach first so any autograd state on the caller's tensor is not propagated.
        using var hWork = hessian.detach().to(ScalarType.Float32).clone();

        // Apply diagonal damping: H += dampingFrac * mean(diag(H)) * I.
        if (dampingFrac > 0f)
        {
            var diagMean = hWork.diagonal().mean().item<float>();
            var lambda = dampingFrac * diagMean;

            // Add lambda to each diagonal element in-place via the diagonal view.
            using var diagView = hWork.diagonal();
            diagView.add_(lambda);
        }

        // Cholesky decomposition, with pseudoinverse fallback.
        try
        {
            // lower is the lower-triangular Cholesky factor: lower · lowerᵀ = H'.
            using var lower = linalg.cholesky(hWork);

            // Invert lower via triangular solve: lower · lowerInv = I.
            using var eyeD = eye(d, dtype: ScalarType.Float32);
            using var lowerInv = linalg.solve_triangular(lower, eyeD, upper: false);

            // H⁻¹ = lowerInvᵀ · lowerInv (symmetric positive-definite).
            return lowerInv.t().mm(lowerInv);
        }
        catch (Exception ex) when (IsCholeskyFailure(ex))
        {
            // Cholesky failed despite damping — fall back to pseudoinverse.
            // This is slower but handles pathologically ill-conditioned Hessians
            // (e.g. very small calibration sets, or layers that saw almost no activations).
            return linalg.pinv(hWork);
        }
    }

    /// <summary>
    /// Heuristically identifies Cholesky decomposition failures from TorchSharp exception messages.
    /// </summary>
    /// <param name="ex">The exception thrown by <c>linalg.cholesky</c>.</param>
    /// <returns><see langword="true"/> if the exception looks like a Cholesky failure.</returns>
    private static bool IsCholeskyFailure(Exception ex)
    {
        // TorchSharp wraps LibTorch errors in ExternalException or general Exception.
        // The message typically contains "not positive definite" or "singular".
        var msg = ex.Message ?? string.Empty;
        return msg.Contains("positive definite", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("singular", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("cholesky", StringComparison.OrdinalIgnoreCase)
            || ex is System.Runtime.InteropServices.ExternalException;
    }
}

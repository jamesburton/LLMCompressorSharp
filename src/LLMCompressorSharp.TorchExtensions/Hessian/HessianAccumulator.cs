// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Accumulates the Hessian <c>H = Σ Xᵀ X</c> for a single layer's inputs across calibration batches.
/// </summary>
/// <remarks>
/// Used by GPTQ and SparseGPT. The Hessian is always FP32 for numerical stability, regardless of
/// the input dtype. Inputs of rank > 2 are flattened to <c>[N, inFeatures]</c> where N is the
/// product of all leading dims.
/// </remarks>
public sealed class HessianAccumulator : IDisposable
{
    private readonly int inFeatures;
    private readonly Tensor hessian;
    private long sampleCount;
    private bool disposed;

    /// <summary>Initializes a new instance of the <see cref="HessianAccumulator"/> class.</summary>
    /// <param name="inFeatures">The input feature dimension (last axis of activation tensors).</param>
    /// <exception cref="ArgumentOutOfRangeException">If <paramref name="inFeatures"/> is not positive.</exception>
    public HessianAccumulator(int inFeatures)
    {
        if (inFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inFeatures), inFeatures, "inFeatures must be positive.");
        }

        this.inFeatures = inFeatures;
        this.hessian = zeros(inFeatures, inFeatures, dtype: ScalarType.Float32);
    }

    /// <summary>Gets the number of activation rows accumulated so far (across all Update calls).</summary>
    public long SampleCount => this.sampleCount;

    /// <summary>Gets the input feature dimension.</summary>
    public int InFeatures => this.inFeatures;

    /// <summary>Accumulates <c>Xᵀ X</c> into the running Hessian.</summary>
    /// <param name="x">Activation tensor with last dim equal to <see cref="InFeatures"/>. Higher ranks are flattened.</param>
    /// <exception cref="ArgumentNullException">If <paramref name="x"/> is null.</exception>
    /// <exception cref="ArgumentException">If the last dim of <paramref name="x"/> does not equal <see cref="InFeatures"/>.</exception>
    public void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        this.ThrowIfDisposed();

        if (x.shape[^1] != this.inFeatures)
        {
            throw new ArgumentException(
                $"Input last dim ({x.shape[^1]}) must equal inFeatures ({this.inFeatures}).",
                nameof(x));
        }

        using var xFp32 = x.to(ScalarType.Float32);
        using var flat = xFp32.reshape(-1L, (long)this.inFeatures);
        var nRows = (int)flat.shape[0];

        using var contribution = flat.t().matmul(flat);
        this.hessian.add_(contribution);

        this.sampleCount += nRows;
    }

    /// <summary>Returns a clone of the current Hessian tensor. The caller owns disposal of the returned tensor.</summary>
    /// <returns>A new <see cref="Tensor"/> matching the running Hessian.</returns>
    public Tensor GetHessian()
    {
        this.ThrowIfDisposed();
        return this.hessian.detach().clone();
    }

    /// <summary>Zeros the running Hessian and sample count.</summary>
    public void Reset()
    {
        this.ThrowIfDisposed();
        this.hessian.zero_();
        this.sampleCount = 0L;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!this.disposed)
        {
            this.hessian.Dispose();
            this.disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(this.disposed, this);
    }
}

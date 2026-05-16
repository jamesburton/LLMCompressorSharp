// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Root Mean Square LayerNorm — the normalization used in LLaMA-family decoders.
/// </summary>
/// <remarks>
/// <c>y = x · rsqrt(mean(x², dim=-1) + eps) · weight</c>. No bias term.
/// </remarks>
public sealed class LlamaRMSNorm : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly float eps;

    /// <summary>Initializes a new instance of the <see cref="LlamaRMSNorm"/> class.</summary>
    /// <param name="hiddenSize">The last-dim size of the input tensor.</param>
    /// <param name="eps">Epsilon added to the mean-square term for numerical stability.</param>
    public LlamaRMSNorm(int hiddenSize, float eps)
        : base(nameof(LlamaRMSNorm))
    {
        if (hiddenSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        }

        this.eps = eps;
        this.weight = Parameter(ones(hiddenSize));
        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        var inputDtype = x.dtype;
        using var x32 = x.to(ScalarType.Float32);
        using var variance = x32.pow(2).mean(new long[] { -1L }, keepdim: true);
        using var stabilized = variance.add(eps);
        using var rstd = stabilized.rsqrt();
        using var normalized = x32.mul(rstd);

        // Avoid use-after-dispose: when the input dtype already matches Float32 (the common case
        // in tests), result.to(inputDtype) may return the same tensor instance; disposing it via
        // `using` would then invalidate the returned reference. Return directly if no cast needed.
        var result = normalized.mul(this.weight);
        if (result.dtype == inputDtype)
        {
            return result;
        }

        using (result)
        {
            return result.to(inputDtype);
        }
    }
}

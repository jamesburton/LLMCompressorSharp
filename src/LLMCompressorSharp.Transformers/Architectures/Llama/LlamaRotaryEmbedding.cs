// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Rotary Position Embedding (RoPE). Pre-computes the inverse-frequency table once at
/// construction; <see cref="Apply"/> rotates query and key tensors by their token positions.
/// </summary>
public sealed class LlamaRotaryEmbedding : Module
{
    private readonly Tensor invFreq;

    /// <summary>Initializes a new instance of the <see cref="LlamaRotaryEmbedding"/> class.</summary>
    /// <param name="headDim">Per-head dimension; must be even.</param>
    /// <param name="maxPositionEmbeddings">Maximum sequence length the table supports.</param>
    /// <param name="theta">RoPE base frequency (LLaMA default 10000; SmolLM2 100000).</param>
    public LlamaRotaryEmbedding(int headDim, int maxPositionEmbeddings, float theta)
        : base(nameof(LlamaRotaryEmbedding))
    {
        if (headDim <= 0 || (headDim % 2) != 0)
        {
            throw new ArgumentException($"headDim must be a positive even integer (got {headDim}).", nameof(headDim));
        }

        // inv_freq[i] = 1 / theta^(2i / headDim) for i in [0, headDim/2)
        using var indices = arange(0, headDim, 2, dtype: ScalarType.Float32);
        using var exponents = indices / headDim;
        using var powers = exponents.mul(MathF.Log(theta)).exp();
        this.invFreq = powers.reciprocal();
        this.register_buffer("inv_freq", this.invFreq);
    }

    /// <summary>
    /// Applies RoPE rotation to query and key tensors.
    /// </summary>
    /// <param name="q">Query tensor of shape <c>[batch, num_heads, seq, head_dim]</c>.</param>
    /// <param name="k">Key tensor of shape <c>[batch, num_kv_heads, seq, head_dim]</c>.</param>
    /// <param name="positionIds">Position indices, shape <c>[batch, seq]</c>.</param>
    /// <returns>Rotated <c>(q, k)</c>.</returns>
    public (Tensor Q, Tensor K) Apply(Tensor q, Tensor k, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(positionIds);

        // Compute [batch, seq, head_dim/2] frequencies, then full [batch, seq, head_dim] embedding.
        using var posFloat = positionIds.to(ScalarType.Float32);
        using var freqs = posFloat.unsqueeze(-1).matmul(this.invFreq.unsqueeze(0));
        using var emb = cat(new[] { freqs, freqs }, dim: -1);
        using var cos = emb.cos().unsqueeze(1).to(q.dtype);
        using var sin = emb.sin().unsqueeze(1).to(q.dtype);

        var qRot = ApplyRotation(q, cos, sin);
        var kRot = ApplyRotation(k, cos, sin);
        return (qRot, kRot);
    }

    private static Tensor ApplyRotation(Tensor x, Tensor cos, Tensor sin)
    {
        using var rotated = RotateHalf(x);
        using var cosPart = x.mul(cos);
        using var sinPart = rotated.mul(sin);
        return cosPart.add(sinPart);
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var headDim = (int)x.shape[^1];
        var half = headDim / 2;
        using var firstHalf = x.narrow(-1, 0, half);
        using var secondHalf = x.narrow(-1, half, half);
        using var negSecond = secondHalf.neg();
        return cat(new[] { negSecond, firstHalf }, dim: -1);
    }
}

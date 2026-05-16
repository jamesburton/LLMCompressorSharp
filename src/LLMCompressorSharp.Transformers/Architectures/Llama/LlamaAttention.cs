// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Grouped-query multi-head self-attention with RoPE and a causal mask.
/// </summary>
public sealed class LlamaAttention : Module
{
    private readonly LlamaConfig config;
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly LlamaRotaryEmbedding rotary_emb;
    private readonly int kvGroupSize;

    /// <summary>Initializes a new instance of the <see cref="LlamaAttention"/> class.</summary>
    /// <param name="config">Architecture config (validated).</param>
    public LlamaAttention(LlamaConfig config)
        : base(nameof(LlamaAttention))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        this.config = config;

        this.kvGroupSize = config.NumAttentionHeads / config.NumKeyValueHeads;

        this.q_proj = Linear(config.HiddenSize, config.NumAttentionHeads * config.HeadDim, hasBias: false);
        this.k_proj = Linear(config.HiddenSize, config.NumKeyValueHeads * config.HeadDim, hasBias: false);
        this.v_proj = Linear(config.HiddenSize, config.NumKeyValueHeads * config.HeadDim, hasBias: false);
        this.o_proj = Linear(config.NumAttentionHeads * config.HeadDim, config.HiddenSize, hasBias: false);

        this.rotary_emb = new LlamaRotaryEmbedding(
            headDim: config.HeadDim,
            maxPositionEmbeddings: config.MaxPositionEmbeddings,
            theta: config.RopeTheta);

        RegisterComponents();
    }

    /// <summary>Self-attention forward pass.</summary>
    /// <param name="x">Input hidden states <c>[batch, seq, hidden]</c>.</param>
    /// <param name="attentionMask">Additive mask <c>[batch, 1, seq, seq]</c> or null (causal mask is built internally).</param>
    /// <param name="positionIds">Position indices <c>[batch, seq]</c>.</param>
    /// <returns>Output hidden states <c>[batch, seq, hidden]</c>.</returns>
    public Tensor forward(Tensor x, Tensor? attentionMask, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(positionIds);

        var batch = x.shape[0];
        var seq = x.shape[1];
        var numHeads = (long)this.config.NumAttentionHeads;
        var numKvHeads = (long)this.config.NumKeyValueHeads;
        var headDim = (long)this.config.HeadDim;

        using var qFlat = this.q_proj.forward(x);
        using var kFlat = this.k_proj.forward(x);
        using var vFlat = this.v_proj.forward(x);

        // Reshape to [batch, heads, seq, head_dim] for attention computation.
        using var qHeads = qFlat.view(batch, seq, numHeads, headDim).transpose(1, 2);
        using var kHeads = kFlat.view(batch, seq, numKvHeads, headDim).transpose(1, 2);
        using var vHeads = vFlat.view(batch, seq, numKvHeads, headDim).transpose(1, 2);

        var (qRot, kRot) = this.rotary_emb.Apply(qHeads, kHeads, positionIds);

        using (qRot)
        using (kRot)
        {
            // Expand k/v from num_kv_heads to num_heads for grouped-query attention.
            using var kRepeated = RepeatKv(kRot, this.kvGroupSize);
            using var vRepeated = RepeatKv(vHeads, this.kvGroupSize);

            // Scaled dot-product: [batch, heads, seq_q, seq_k].
            using var attnScores = qRot.matmul(kRepeated.transpose(-2, -1));
            using var scale = tensor(1f / MathF.Sqrt(headDim)).to(attnScores.dtype);
            using var scaled = attnScores.mul(scale);

            // Build causal mask [seq, seq] and broadcast over batch/heads dims.
            using var causalMask = BuildCausalMask(seq, x.device, x.dtype);
            using var withCausal = scaled.add(causalMask);

            // Apply optional external mask; only the branch with attentionMask allocates a new tensor.
            Tensor? extraMasked = attentionMask is not null ? withCausal.add(attentionMask) : null;
            var maskedScores = extraMasked ?? withCausal;

            using var probs = softmax(maskedScores, dim: -1);
            extraMasked?.Dispose();
            using var attnOut = probs.matmul(vRepeated);

            // Merge heads and project back to hidden dim.
            using var transposed = attnOut.transpose(1, 2).contiguous();
            using var combined = transposed.view(batch, seq, numHeads * headDim);
            return this.o_proj.forward(combined);
        }
    }

    /// <summary>
    /// Repeats key/value tensors <paramref name="repeats"/> times along the head dimension to
    /// expand from <c>num_kv_heads</c> to <c>num_heads</c> for grouped-query attention.
    /// </summary>
    /// <param name="kv">Tensor of shape <c>[batch, num_kv_heads, seq, head_dim]</c>.</param>
    /// <param name="repeats">Number of times to repeat (= num_heads / num_kv_heads).</param>
    /// <returns>Tensor of shape <c>[batch, num_heads, seq, head_dim]</c>.</returns>
    private static Tensor RepeatKv(Tensor kv, int repeats)
    {
        var batch = kv.shape[0];
        var numKv = kv.shape[1];
        var seq = kv.shape[2];
        var hd = kv.shape[3];

        // expand produces a non-contiguous view; contiguous() materialises it before reshape.
        using var unsqueezed = kv.unsqueeze(2);
        using var expanded = unsqueezed.expand(new long[] { batch, numKv, repeats, seq, hd });
        using var contiguous = expanded.contiguous();
        return contiguous.reshape(batch, numKv * repeats, seq, hd);
    }

    /// <summary>
    /// Builds a causal (lower-triangular) attention mask of shape <c>[seq, seq]</c>.
    /// Upper-triangular entries are <c>-inf</c>; lower-triangular (including diagonal) are <c>0</c>.
    /// </summary>
    /// <param name="seq">Sequence length.</param>
    /// <param name="device">Target device.</param>
    /// <param name="dtype">Target dtype (matches query/key dtype).</param>
    /// <returns>Additive causal mask of shape <c>[seq, seq]</c>.</returns>
    private static Tensor BuildCausalMask(long seq, torch.Device device, ScalarType dtype)
    {
        // Build a boolean upper-triangular mask (true = future token = block) and use torch.where
        // to place -inf at those positions and 0 elsewhere.
        // Avoids 0 * (-inf) = NaN that arises from triu(ones).mul(-inf).
        using var zeros = torch.zeros(new long[] { seq, seq }, dtype: ScalarType.Float32, device: device);
        using var neginf = torch.full(new long[] { seq, seq }, float.NegativeInfinity, dtype: ScalarType.Float32, device: device);
        using var ones = torch.ones(new long[] { seq, seq }, dtype: ScalarType.Bool, device: device);
        using var upperMask = ones.triu(diagonal: 1);
        using var mask = torch.where(upperMask, neginf, zeros);
        return mask.to(dtype);
    }
}

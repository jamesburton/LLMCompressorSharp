// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// One LLaMA decoder block: pre-norm + self-attention + residual; pre-norm + MLP + residual.
/// </summary>
public sealed class LlamaDecoderLayer : Module
{
    private readonly LlamaRMSNorm input_layernorm;
    private readonly LlamaAttention self_attn;
    private readonly LlamaRMSNorm post_attention_layernorm;
    private readonly LlamaMLP mlp;

    /// <summary>Initializes a new instance of the <see cref="LlamaDecoderLayer"/> class.</summary>
    /// <param name="config">Architecture configuration.</param>
    public LlamaDecoderLayer(LlamaConfig config)
        : base(nameof(LlamaDecoderLayer))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();

        this.input_layernorm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.self_attn = new LlamaAttention(config);
        this.post_attention_layernorm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.mlp = new LlamaMLP(config.HiddenSize, config.IntermediateSize);
        RegisterComponents();
    }

    /// <summary>Forward pass.</summary>
    /// <param name="x">Input hidden states <c>[batch, seq, hidden]</c>.</param>
    /// <param name="attentionMask">Optional additive attention mask.</param>
    /// <param name="positionIds">Position indices <c>[batch, seq]</c>.</param>
    /// <returns>Output hidden states <c>[batch, seq, hidden]</c>.</returns>
    public Tensor forward(Tensor x, Tensor? attentionMask, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(positionIds);

        using var attnInput = this.input_layernorm.forward(x);
        using var attnOut = this.self_attn.forward(attnInput, attentionMask, positionIds);
        using var afterAttn = x.add(attnOut);

        using var mlpInput = this.post_attention_layernorm.forward(afterAttn);
        using var mlpOut = this.mlp.forward(mlpInput);
        return afterAttn.add(mlpOut);
    }
}

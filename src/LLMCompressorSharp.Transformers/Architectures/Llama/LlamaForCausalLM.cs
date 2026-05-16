// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// LLaMA causal language model: embed_tokens → N × LlamaDecoderLayer → final RMSNorm → lm_head.
/// </summary>
public sealed class LlamaForCausalLM : Module<Tensor, Tensor>
{
    private readonly LlamaConfig config;
    private readonly Embedding embed_tokens;
    private readonly ModuleList<LlamaDecoderLayer> layers;
    private readonly LlamaRMSNorm norm;
    private readonly Linear lm_head;

    /// <summary>Initializes a new instance of the <see cref="LlamaForCausalLM"/> class.</summary>
    /// <param name="config">Architecture configuration.</param>
    public LlamaForCausalLM(LlamaConfig config)
        : base(nameof(LlamaForCausalLM))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        this.config = config;

        this.embed_tokens = Embedding(config.VocabSize, config.HiddenSize);

        var decoderLayers = new List<LlamaDecoderLayer>(config.NumHiddenLayers);
        for (var i = 0; i < config.NumHiddenLayers; i++)
        {
            decoderLayers.Add(new LlamaDecoderLayer(config));
        }

        this.layers = ModuleList<LlamaDecoderLayer>(decoderLayers.ToArray());
        this.norm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.lm_head = Linear(config.HiddenSize, config.VocabSize, hasBias: false);

        if (config.TieWordEmbeddings)
        {
            this.lm_head.weight = this.embed_tokens.weight!;
        }

        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor inputIds)
    {
        ArgumentNullException.ThrowIfNull(inputIds);

        var batch = inputIds.shape[0];
        var seq = inputIds.shape[1];

        using var positionIds = arange(seq, device: inputIds.device, dtype: ScalarType.Int64)
            .unsqueeze(0)
            .expand(batch, seq);

        using var hiddenStates = this.embed_tokens.forward(inputIds);
        var current = hiddenStates;
        var ownsCurrent = false;

        foreach (var layer in this.layers)
        {
            var next = layer.forward(current, attentionMask: null, positionIds: positionIds);
            if (ownsCurrent)
            {
                current.Dispose();
            }

            current = next;
            ownsCurrent = true;
        }

        try
        {
            using var normalized = this.norm.forward(current);
            return this.lm_head.forward(normalized);
        }
        finally
        {
            if (ownsCurrent)
            {
                current.Dispose();
            }
        }
    }
}

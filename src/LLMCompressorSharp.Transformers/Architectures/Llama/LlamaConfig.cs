// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Hyperparameters for a LLaMA-family decoder-only transformer.
/// </summary>
/// <remarks>
/// Field names match the HuggingFace <c>config.json</c> keys with PascalCase substitution.
/// Phase 3b adds a JSON parser; Phase 3a tests construct instances directly.
/// </remarks>
public sealed class LlamaConfig
{
    /// <summary>Gets or sets the residual stream dimension. Required.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Gets or sets the MLP hidden (inner) dimension. Required.</summary>
    public int IntermediateSize { get; set; }

    /// <summary>Gets or sets the number of decoder layers. Required.</summary>
    public int NumHiddenLayers { get; set; }

    /// <summary>Gets or sets the number of attention query heads. Required.</summary>
    public int NumAttentionHeads { get; set; }

    /// <summary>Gets or sets the number of KV heads for grouped-query attention. Equals <see cref="NumAttentionHeads"/> for MHA.</summary>
    public int NumKeyValueHeads { get; set; }

    /// <summary>Gets or sets the vocabulary size. Required.</summary>
    public int VocabSize { get; set; }

    /// <summary>Gets or sets the maximum position embedding length. Default: 2048.</summary>
    public int MaxPositionEmbeddings { get; set; } = 2048;

    /// <summary>Gets or sets the RoPE theta base. Default: 10000.</summary>
    public float RopeTheta { get; set; } = 10000f;

    /// <summary>Gets or sets the RMSNorm epsilon. Default: 1e-5.</summary>
    public float RmsNormEps { get; set; } = 1e-5f;

    /// <summary>Gets or sets the hidden activation. Phase 3a only supports "silu".</summary>
    public string HiddenAct { get; set; } = "silu";

    /// <summary>Gets or sets a value indicating whether <c>lm_head.weight</c> shares storage with <c>embed_tokens.weight</c>.</summary>
    public bool TieWordEmbeddings { get; set; } = true;

    /// <summary>Gets the per-head dimension <c>HiddenSize / NumAttentionHeads</c>.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Validates that required fields are present and consistent.</summary>
    /// <exception cref="ArgumentException">If any required field is missing or inconsistent.</exception>
    public void Validate()
    {
        if (HiddenSize <= 0)
        {
            throw new ArgumentException($"HiddenSize must be positive (got {HiddenSize}).");
        }

        if (NumAttentionHeads <= 0 || HiddenSize % NumAttentionHeads != 0)
        {
            throw new ArgumentException(
                $"NumAttentionHeads={NumAttentionHeads} must divide HiddenSize={HiddenSize} evenly.");
        }

        if (NumKeyValueHeads <= 0 || NumAttentionHeads % NumKeyValueHeads != 0)
        {
            throw new ArgumentException(
                $"NumKeyValueHeads={NumKeyValueHeads} must divide NumAttentionHeads={NumAttentionHeads} evenly.");
        }

        if (NumHiddenLayers <= 0 || IntermediateSize <= 0 || VocabSize <= 0)
        {
            throw new ArgumentException("NumHiddenLayers, IntermediateSize, and VocabSize must be positive.");
        }

        if (!string.Equals(HiddenAct, "silu", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Phase 3a only supports HiddenAct='silu' (got '{HiddenAct}').");
        }
    }
}

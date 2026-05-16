// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Maps between HuggingFace's <c>model.*</c>-prefixed parameter names and our flattened
/// <c>LlamaForCausalLM</c> parameter hierarchy.
/// </summary>
public static class HuggingFaceWeightNameMapper
{
    private const string HfModelPrefix = "model.";
    private const string LmHeadName = "lm_head.weight";

    /// <summary>Maps an HF parameter name to our equivalent.</summary>
    /// <param name="hfName">HuggingFace name (e.g. <c>model.layers.0.self_attn.q_proj.weight</c>).</param>
    /// <returns>The corresponding name in our parameter hierarchy.</returns>
    public static string MapName(string hfName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(hfName);
        return hfName.StartsWith(HfModelPrefix, StringComparison.Ordinal)
            ? hfName[HfModelPrefix.Length..]
            : hfName;
    }

    /// <summary>Maps one of our parameter names back to HF's convention.</summary>
    /// <param name="ourName">Our parameter name (e.g. <c>layers.0.self_attn.q_proj.weight</c>).</param>
    /// <returns>The corresponding HF name.</returns>
    public static string UnmapName(string ourName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ourName);
        if (string.Equals(ourName, LmHeadName, StringComparison.Ordinal))
        {
            return ourName;
        }

        return HfModelPrefix + ourName;
    }

    /// <summary>Builds a new dictionary with all keys remapped via <see cref="MapName"/>.</summary>
    /// <param name="hfDict">Source dictionary keyed by HF names.</param>
    /// <returns>New dictionary keyed by our names; tensor values are shared (not copied).</returns>
    public static Dictionary<string, Tensor> MapDictionary(IDictionary<string, Tensor> hfDict)
    {
        ArgumentNullException.ThrowIfNull(hfDict);
        var result = new Dictionary<string, Tensor>(hfDict.Count, StringComparer.Ordinal);
        foreach (var (key, value) in hfDict)
        {
            result[MapName(key)] = value;
        }

        return result;
    }
}

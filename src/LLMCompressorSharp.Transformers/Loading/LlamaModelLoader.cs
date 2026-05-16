// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Applies a HuggingFace-named state dict onto a <see cref="LlamaForCausalLM"/>.
/// </summary>
public static class LlamaModelLoader
{
    private const string LmHeadKey = "lm_head.weight";
    private const string EmbedTokensKey = "model.embed_tokens.weight";

    /// <summary>
    /// Copies weights from <paramref name="hfWeights"/> into the model's parameters.
    /// </summary>
    /// <param name="model">The target model.</param>
    /// <param name="hfWeights">Source state dict keyed by HuggingFace names.</param>
    /// <param name="strict">When true, throws if any model parameter has no source weight.</param>
    /// <exception cref="HuggingFaceLoadException">On missing weights (strict) or shape mismatches.</exception>
    public static void Load(LlamaForCausalLM model, IDictionary<string, Tensor> hfWeights, bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(hfWeights);

        var missing = new List<string>();

        using (no_grad())
        {
            foreach (var (paramName, parameter) in model.named_parameters())
            {
                var hfName = HuggingFaceWeightNameMapper.UnmapName(paramName);

                if (!hfWeights.TryGetValue(hfName, out var source))
                {
                    if (paramName.EndsWith(LmHeadKey, StringComparison.Ordinal)
                        && hfWeights.ContainsKey(EmbedTokensKey))
                    {
                        continue;
                    }

                    missing.Add(hfName);
                    continue;
                }

                if (!ShapesEqual(parameter.shape, source.shape))
                {
                    throw new HuggingFaceLoadException(
                        $"Shape mismatch for '{hfName}': model expects [{string.Join(",", parameter.shape)}] "
                        + $"but source has [{string.Join(",", source.shape)}].");
                }

                parameter.copy_(source.to(parameter.device));
            }
        }

        if (strict && missing.Count > 0)
        {
            throw new HuggingFaceLoadException(
                $"Missing weights in source for: {string.Join(", ", missing)}.");
        }
    }

    private static bool ShapesEqual(long[] a, long[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning;

/// <summary>
/// Zeros out the lowest-magnitude weights to reach a target sparsity per weight tensor.
/// Data-free; runs entirely in <see cref="ModifierBase.OnEndCore"/>.
/// </summary>
public sealed class MagnitudePruningModifier : ModifierBase
{
    private readonly float _sparsity;

    /// <summary>Initializes a new instance of the <see cref="MagnitudePruningModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public MagnitudePruningModifier(MagnitudePruningConfig config)
        : base("MagnitudePruning", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Sparsity is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(config),
                config.Sparsity,
                "Sparsity must be in [0, 1].");
        }

        _sparsity = config.Sparsity;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        if (_sparsity <= 0f)
        {
            return;
        }

        var targeted = GetTargetedNames(state).ToList();
        foreach (var name in targeted)
        {
            // Do not dispose the original tensor — the caller still owns its lifetime.
            var weight = state.NamedWeights[name];
            state.NamedWeights[name] = PruneToSparsity(weight, _sparsity);
        }
    }

    private static Tensor PruneToSparsity(Tensor weight, float sparsity)
    {
        if (sparsity >= 1f)
        {
            return zeros_like(weight);
        }

        using var absWeight = weight.abs();
        using var flat = absWeight.flatten();
        var numElements = (int)flat.shape[0];
        var numToKeep = (int)Math.Max(0, Math.Round(numElements * (1.0 - sparsity)));
        if (numToKeep >= numElements)
        {
            return weight.clone();
        }

        if (numToKeep == 0)
        {
            return zeros_like(weight);
        }

        var (topkValues, topkIndices) = flat.topk(numToKeep, largest: true);
        float threshold;
        using (topkValues)
        using (topkIndices)
        using (var kthSmallestKept = topkValues[numToKeep - 1])
        {
            threshold = kthSmallestKept.item<float>();
        }

        using var mask = absWeight.ge(threshold);
        return weight.mul(mask.to(weight.dtype));
    }
}

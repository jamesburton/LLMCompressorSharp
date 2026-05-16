// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning;

/// <summary>
/// Pruning by <c>|w| × ||x||₂</c> saliency. Activation L2 norms are accumulated per input channel
/// across calibration batches; at <see cref="ModifierBase.OnEndCore"/> the lowest-saliency weights
/// are zeroed to reach the configured sparsity.
/// </summary>
public sealed class WandaModifier : ModifierBase
{
    private const string WeightSuffix = ".weight";

    private readonly float _sparsity;
    private readonly Dictionary<string, double[]> _runningSumSquares = new();

    /// <summary>Initializes a new instance of the <see cref="WandaModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public WandaModifier(WandaConfig config)
        : base("WANDA", config?.Targets, config?.Ignore)
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
        _runningSumSquares.Clear();
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        EnsureActivations(state);
        var activations = state.LayerActivations!;

        foreach (var name in GetTargetedNames(state))
        {
            var activationKey = StripWeightSuffix(name);
            if (!activations.TryGetValue(activationKey, out var x))
            {
                continue;
            }

            var rank = x.shape.Length;
            var reduceDims = new long[rank - 1];
            for (var i = 0; i < rank - 1; i++)
            {
                reduceDims[i] = i;
            }

            using var squared = x.pow(2);
            using var reduced = reduceDims.Length == 0
                ? squared.clone()
                : squared.sum(reduceDims, keepdim: false);
            var batchContrib = reduced.cpu().data<float>().ToArray();

            if (!_runningSumSquares.TryGetValue(name, out var accum))
            {
                accum = new double[batchContrib.Length];
                _runningSumSquares[name] = accum;
            }

            if (accum.Length != batchContrib.Length)
            {
                throw new InvalidOperationException(
                    $"Activation feature dimension changed for '{activationKey}': "
                    + $"expected {accum.Length}, got {batchContrib.Length}.");
            }

            for (var i = 0; i < accum.Length; i++)
            {
                accum[i] += batchContrib[i];
            }
        }
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        EnsureActivations(state);
        var activations = state.LayerActivations!;

        foreach (var name in GetTargetedNames(state))
        {
            var activationKey = StripWeightSuffix(name);
            if (!_runningSumSquares.TryGetValue(name, out var sumSq))
            {
                if (activations.TryGetValue(activationKey, out var x))
                {
                    using var sq = x.pow(2);
                    var rank = x.shape.Length;
                    Tensor reduced;
                    if (rank == 1)
                    {
                        reduced = sq.clone();
                    }
                    else
                    {
                        var dims = new long[rank - 1];
                        for (var i = 0; i < rank - 1; i++)
                        {
                            dims[i] = i;
                        }

                        reduced = sq.sum(dims, keepdim: false);
                    }

                    using (reduced)
                    {
                        var batchContrib = reduced.cpu().data<float>().ToArray();
                        sumSq = new double[batchContrib.Length];
                        for (var i = 0; i < batchContrib.Length; i++)
                        {
                            sumSq[i] = batchContrib[i];
                        }
                    }
                }
                else
                {
                    throw new InvalidOperationException(
                        $"WANDA: no activation found for layer '{activationKey}' "
                        + $"(weight '{name}'). Populate CompressionState.LayerActivations.");
                }
            }

            var norms = new float[sumSq.Length];
            for (var i = 0; i < sumSq.Length; i++)
            {
                norms[i] = (float)Math.Sqrt(sumSq[i]);
            }

            var weight = state.NamedWeights[name];
            state.NamedWeights[name] = ApplyWandaPruning(weight, norms, _sparsity);
        }
    }

    private static void EnsureActivations(CompressionState state)
    {
        if (state.LayerActivations is null)
        {
            throw new InvalidOperationException(
                "WANDA: CompressionState.LayerActivations is null. WANDA requires per-layer activations.");
        }
    }

    private static string StripWeightSuffix(string name)
    {
        return name.EndsWith(WeightSuffix, StringComparison.Ordinal)
            ? name[..^WeightSuffix.Length]
            : name;
    }

    private static Tensor ApplyWandaPruning(Tensor weight, float[] inputNorms, float sparsity)
    {
        if (sparsity <= 0f)
        {
            return weight.clone();
        }

        if (sparsity >= 1f)
        {
            return zeros_like(weight);
        }

        var inFeatures = (int)weight.shape[^1];
        if (inputNorms.Length != inFeatures)
        {
            throw new InvalidOperationException(
                $"WANDA: input norms length {inputNorms.Length} does not match weight last dim {inFeatures}.");
        }

        using var normTensor = tensor(inputNorms).to(weight.dtype);

        var reshapeDims = new long[weight.shape.Length];
        for (var i = 0; i < reshapeDims.Length - 1; i++)
        {
            reshapeDims[i] = 1L;
        }

        reshapeDims[^1] = inFeatures;

        using var normBroadcast = normTensor.reshape(reshapeDims);
        using var absWeight = weight.abs();
        using var saliency = absWeight.mul(normBroadcast);
        using var flat = saliency.flatten();

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
        using (var kth = topkValues[numToKeep - 1])
        {
            threshold = kth.item<float>();
        }

        using var mask = saliency.ge(threshold);
        return weight.mul(mask.to(weight.dtype));
    }
}

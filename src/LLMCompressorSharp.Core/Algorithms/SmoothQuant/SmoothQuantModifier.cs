// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.SmoothQuant;

/// <summary>
/// Channel-wise activation/weight rebalancing: migrates per-channel quantization difficulty from
/// activations into weights. The mathematical identity <c>Y = (X / s) · (s · W)</c> is preserved.
/// </summary>
public sealed class SmoothQuantModifier : ModifierBase
{
    private readonly SmoothQuantConfig _config;
    private readonly Dictionary<string, float[]> _runningActMax = new();

    /// <summary>Initializes a new instance of the <see cref="SmoothQuantModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public SmoothQuantModifier(SmoothQuantConfig config)
        : base("SmoothQuant", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.SmoothingStrength is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(config),
                config.SmoothingStrength,
                "SmoothingStrength (alpha) must be in [0, 1].");
        }

        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        _runningActMax.Clear();
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        if (state.LayerActivations is null)
        {
            return;
        }

        foreach (var mapping in _config.Mappings)
        {
            if (!state.LayerActivations.TryGetValue(mapping.ActivationKey, out var x))
            {
                continue;
            }

            UpdateActivationMax(mapping.ActivationKey, x);
        }
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        foreach (var mapping in _config.Mappings)
        {
            ApplyMapping(state, mapping);
        }
    }

    private void UpdateActivationMax(string key, Tensor x)
    {
        var rank = x.shape.Length;
        Tensor reduced;
        if (rank == 1)
        {
            reduced = x.abs();
        }
        else
        {
            var dims = new long[rank - 1];
            for (var i = 0; i < rank - 1; i++)
            {
                dims[i] = i;
            }

            using var abs = x.abs();
            reduced = abs.amax(dims, keepdim: false);
        }

        using (reduced)
        {
            var batch = reduced.cpu().data<float>().ToArray();
            if (!_runningActMax.TryGetValue(key, out var existing))
            {
                _runningActMax[key] = batch;
            }
            else if (existing.Length != batch.Length)
            {
                throw new InvalidOperationException(
                    $"SmoothQuant: activation feature dim changed for '{key}': "
                    + $"expected {existing.Length}, got {batch.Length}.");
            }
            else
            {
                for (var i = 0; i < existing.Length; i++)
                {
                    if (batch[i] > existing[i])
                    {
                        existing[i] = batch[i];
                    }
                }
            }
        }
    }

    private void ApplyMapping(CompressionState state, SmoothQuantMapping mapping)
    {
        if (state.LayerActivations is null
            || !state.LayerActivations.TryGetValue(mapping.ActivationKey, out var x))
        {
            if (!_runningActMax.TryGetValue(mapping.ActivationKey, out _))
            {
                throw new InvalidOperationException(
                    $"SmoothQuant: no activation collected for '{mapping.ActivationKey}'. "
                    + "Populate CompressionState.LayerActivations.");
            }
        }
        else if (!_runningActMax.ContainsKey(mapping.ActivationKey))
        {
            UpdateActivationMax(mapping.ActivationKey, x);
        }

        var actMax = _runningActMax[mapping.ActivationKey];

        var smoothW = state.NamedWeights[mapping.SmoothWeight];
        if (smoothW.shape.Length != 1)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: smooth weight '{mapping.SmoothWeight}' must be 1-D (got rank {smoothW.shape.Length}).");
        }

        var balanceW = state.NamedWeights[mapping.BalanceWeight];
        if (balanceW.shape.Length != 2)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: balance weight '{mapping.BalanceWeight}' must be 2-D (got rank {balanceW.shape.Length}).");
        }

        var hidden = (int)smoothW.shape[0];
        if (actMax.Length != hidden)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: activation hidden dim {actMax.Length} does not match smooth weight {hidden}.");
        }

        if (balanceW.shape[1] != hidden)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: balance weight last dim {balanceW.shape[1]} does not match hidden {hidden}.");
        }

        using var absBalance = balanceW.abs();
        using var weightMax = absBalance.amax(new long[] { 0L }, keepdim: false);
        var weightMaxArr = weightMax.cpu().data<float>().ToArray();

        var alpha = _config.SmoothingStrength;
        var s = new float[hidden];
        for (var i = 0; i < hidden; i++)
        {
            var numerator = MathF.Pow(MathF.Max(actMax[i], 1e-5f), alpha);
            var denominator = MathF.Pow(MathF.Max(weightMaxArr[i], 1e-5f), 1f - alpha);
            s[i] = MathF.Max(numerator / denominator, 1e-5f);
        }

        using var scale = tensor(s).to(smoothW.dtype);
        using var newSmooth = smoothW.div(scale);
        state.NamedWeights[mapping.SmoothWeight] = newSmooth.detach().clone();

        using var newBalance = balanceW.mul(scale);
        state.NamedWeights[mapping.BalanceWeight] = newBalance.detach().clone();
    }
}

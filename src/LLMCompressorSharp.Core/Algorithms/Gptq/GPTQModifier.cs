// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Core.Algorithms.Gptq;

/// <summary>
/// GPTQ weight quantization modifier.
/// </summary>
/// <remarks>
/// Implements the GPTQ algorithm (Frantar et al. 2022) using Phase 4a's Hessian
/// accumulation infrastructure. For each targeted <c>Linear</c> layer:
/// <list type="number">
///   <item>Attaches a forward hook (via <see cref="ActivationHookManager"/>) during
///         <c>OnStart</c> to accumulate <c>H = Σ Xᵀ X</c> for each calibration batch.</item>
///   <item>During <c>OnEnd</c>: removes hooks, applies damped Cholesky inversion via
///         <see cref="HessianInverseSolver"/>, calls <see cref="GptqBlockQuantizer.Quantize"/>,
///         and writes the quantized weight back to <see cref="CompressionState.NamedWeights"/>.</item>
/// </list>
/// Requires <see cref="CompressionState.NamedModules"/> to be populated before <c>OnStart</c>.
/// </remarks>
public sealed class GPTQModifier : ModifierBase
{
    private readonly GPTQConfig _config;
    private ActivationHookManager? _hookManager;

    // Per-layer accumulator, keyed by weight name (e.g. "linear.weight").
    private Dictionary<string, HessianAccumulator>? _accumulators;

    /// <summary>Initializes a new instance of the <see cref="GPTQModifier"/> class.</summary>
    /// <param name="config">The GPTQ configuration.</param>
    /// <exception cref="ArgumentNullException">If <paramref name="config"/> is null.</exception>
    public GPTQModifier(GPTQConfig config)
        : base("GPTQ", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        // Validate ActOrder early — fail fast before any work is done.
        if (_config.ActOrder)
        {
            throw new NotSupportedException(
                "GPTQConfig.ActOrder=true is not yet implemented. "
                + "Column reordering by Hessian diagonal is deferred to a future release.");
        }
    }

    /// <inheritdoc />
    protected override void OnStartCore(CompressionState state)
    {
        if (state.NamedModules is null)
        {
            throw new InvalidOperationException(
                $"{nameof(GPTQModifier)} requires {nameof(CompressionState)}.{nameof(CompressionState.NamedModules)} "
                + "to be populated. Set it to the named Linear modules before calling OnStart.");
        }

        _hookManager = new ActivationHookManager();
        _accumulators = new Dictionary<string, HessianAccumulator>();

        // Register one hook+accumulator per targeted weight.
        // Weight key convention: "layerName.weight" maps to module key "layerName".
        foreach (var weightName in GetTargetedNames(state))
        {
            var layerName = weightName.EndsWith(".weight", StringComparison.Ordinal)
                ? weightName[..^".weight".Length]
                : weightName;

            if (!state.NamedModules.TryGetValue(layerName, out var module))
            {
                // Skip layers that are in NamedWeights but not in NamedModules.
                // This can happen for embedding layers or custom modules where hook-based
                // activation capture is not applicable.
                continue;
            }

            // Determine inFeatures from the weight tensor shape [outFeatures, inFeatures].
            var weightShape = state.NamedWeights[weightName].shape;
            var inFeatures = (int)weightShape[^1];

            var accumulator = new HessianAccumulator(inFeatures);
            _hookManager.RegisterFor(module, accumulator);
            _accumulators[weightName] = accumulator;
        }
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        // Hooks fire automatically during the caller's model.call() (forward) pass.
        // GPTQModifier itself does not drive forward passes — the CompressionSession's
        // caller does this externally. The hooks installed in OnStartCore capture the
        // activations and feed them to the per-layer HessianAccumulator.
        _ = state;
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        // Remove all hooks immediately — calibration is done.
        _hookManager?.Dispose();
        _hookManager = null;

        if (_accumulators is null)
        {
            return;
        }

        foreach (var (weightName, accumulator) in _accumulators)
        {
            using var accumulatorScope = NewDisposeScope();

            using var hessian = accumulator.GetHessian();
            using var hinv = HessianInverseSolver.Compute(hessian, _config.DampeningFrac);

            var originalWeight = state.NamedWeights[weightName];
            using var wq = GptqBlockQuantizer.Quantize(originalWeight, hinv, _config);

            // Detach the quantized weight from ALL dispose scopes before assigning to
            // state.NamedWeights — otherwise it would be freed when the accumulator scope ends.
            // The original tensor in state is not disposed here; the caller owns its lifetime,
            // we just swap the reference.
            var quantizedDetached = wq.detach().clone();
            quantizedDetached.DetachFromDisposeScope();
            state.NamedWeights[weightName] = quantizedDetached;
        }
    }

    /// <inheritdoc />
    protected override void OnFinalizeCore(CompressionState state)
    {
        // Belt-and-suspenders: ensure hooks are removed even if OnEnd was skipped.
        _hookManager?.Dispose();
        _hookManager = null;

        if (_accumulators is not null)
        {
            foreach (var accum in _accumulators.Values)
            {
                accum.Dispose();
            }

            _accumulators = null;
        }
    }
}

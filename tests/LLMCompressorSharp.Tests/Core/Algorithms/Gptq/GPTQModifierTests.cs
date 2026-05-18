// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GPTQModifier"/> lifecycle and output correctness.
/// </summary>
public class GPTQModifierTests
{
    [Fact]
    public void Modifier_Name_IsGPTQ()
    {
        var modifier = new GPTQModifier(DefaultConfig());
        modifier.Name.Should().Be("GPTQ");
    }

    [Fact]
    public void FullLifecycle_SyntheticLayer_OutputIsFinite()
    {
        // Synthetic: 4-in, 3-out Linear; 8-bit quantization; 5 calibration batches.
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig(numBits: 8));
            modifier.Initialize(state);
            modifier.OnStart(state);

            // Feed 5 calibration batches — hooks accumulate the Hessian automatically.
            for (int i = 0; i < 5; i++)
            {
                using var scope = NewDisposeScope();
                using var batch = randn(2, 4);  // [batch=2, inFeatures=4]

                // The session would normally call model.forward(batch). In this unit test,
                // we trigger the hook by calling layer.call(batch) directly (Phase 4a convention:
                // hooks fire on .call(), not .forward()).
                using var unused = layer.call(batch);

                modifier.OnBatch(state);
            }

            modifier.OnEnd(state);
            modifier.Finalize(state);

            // Quantized weight should be finite.
            state.NamedWeights["linear.weight"].isfinite().all().item<bool>().Should().BeTrue();
        }
    }

    [Fact]
    public void FullLifecycle_QuantizedWeight_CloseToOriginalAt16Bit()
    {
        // At 16-bit, quantized weights should be very close to the originals.
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var originalWeight = state.NamedWeights["linear.weight"].detach().clone();

            var modifier = new GPTQModifier(DefaultConfig(numBits: 16));
            modifier.Initialize(state);
            modifier.OnStart(state);

            for (int i = 0; i < 3; i++)
            {
                using var scope = NewDisposeScope();
                using var batch = randn(2, 4);
                using var unused = layer.call(batch);
                modifier.OnBatch(state);
            }

            modifier.OnEnd(state);
            modifier.Finalize(state);

            using var diff = (state.NamedWeights["linear.weight"] - originalWeight).abs();
            diff.max().item<float>().Should().BeLessThan(1e-3f);

            originalWeight.Dispose();
        }
    }

    [Fact]
    public void FullLifecycle_NoBatches_DoesNotThrow()
    {
        // Zero calibration batches: Hessian is all-zeros; modifier should handle gracefully
        // (damping regularises the zero matrix).
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig());
            modifier.Initialize(state);
            modifier.OnStart(state);

            // No OnBatch calls.
            modifier.OnEnd(state);
            modifier.Finalize(state);

            // Output weight should exist and be finite.
            state.NamedWeights["linear.weight"].isfinite().all().item<bool>().Should().BeTrue();
        }
    }

    [Fact]
    public void FullLifecycle_NullNamedModules_ThrowsInvalidOperation()
    {
        // GPTQModifier requires NamedModules to be populated.
        var namedWeights = new Dictionary<string, Tensor>
        {
            ["linear.weight"] = randn(3, 4),
        };
        var state = new CompressionState(namedWeights);

        // state.NamedModules is null (not set).
        var modifier = new GPTQModifier(DefaultConfig());
        modifier.Initialize(state);
        var act = () => modifier.OnStart(state);
        act.Should().Throw<InvalidOperationException>().WithMessage("*NamedModules*");
    }

    [Fact]
    public void ActOrder_True_ThrowsNotSupported()
    {
        var config = DefaultConfig();
        config.ActOrder = true;
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(config);
            var act = () => modifier.Initialize(state);
            act.Should().Throw<NotSupportedException>().WithMessage("*ActOrder*");
        }
    }

    [Fact]
    public void Finalize_DisposesHooks_HooksNoLongerFire()
    {
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig());
            modifier.Initialize(state);
            modifier.OnStart(state);

            // Run one batch to establish baseline.
            using var batch1 = randn(2, 4);
            using var unused1 = layer.call(batch1);
            modifier.OnBatch(state);

            modifier.OnEnd(state);
            modifier.Finalize(state);

            // After Finalize the hooks must be gone. A forward call should NOT update
            // the accumulator. We verify this indirectly by checking the weight is frozen.
            var weightAfterFinalize = state.NamedWeights["linear.weight"].detach().clone();
            using var batch2 = randn(2, 4);
            using var unused2 = layer.call(batch2);

            // Weight should not have changed (no re-quantization triggered).
            using var diff = (state.NamedWeights["linear.weight"] - weightAfterFinalize).abs();
            diff.max().item<float>().Should().BeLessThan(1e-8f);
            weightAfterFinalize.Dispose();
        }
    }

    private static GPTQConfig DefaultConfig(int numBits = 8) => new GPTQConfig
    {
        NumBits = numBits,
        Symmetric = true,
        Strategy = QuantizationStrategy.PerTensor,
        BlockSize = 4,
        DampeningFrac = 0.01f,
        Targets = null, // match all
        Ignore = null,
    };

    /// <summary>
    /// Builds a minimal <see cref="CompressionState"/> with one Linear layer plus its module.
    /// </summary>
    private static (CompressionState State, TorchSharp.Modules.Linear Layer) BuildSingleLayerState(
        int inFeatures = 4,
        int outFeatures = 3,
        string weightKey = "linear.weight")
    {
        var layer = Linear(inFeatures, outFeatures, hasBias: false);
        var namedWeights = new Dictionary<string, Tensor>
        {
            [weightKey] = layer.weight!.detach().clone(),
        };
        var layerKey = weightKey.Replace(".weight", string.Empty);
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
            {
                [layerKey] = layer,
            },
        };
        return (state, layer);
    }
}

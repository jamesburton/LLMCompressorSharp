// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="WandaModifier"/> — pruning by |w| × ||x||₂ saliency.
/// </summary>
public class WandaModifierTests
{
    [Fact]
    public void Name_IsWanda()
    {
        var modifier = new WandaModifier(new WandaConfig());
        modifier.Name.Should().Be("WANDA");
    }

    [Fact]
    public void OnEnd_SaliencyEqualsWeightTimesActivationNorm_PrunesLowestScores()
    {
        using var weight = ones(2, 3);
        using var activation = tensor(new float[,]
        {
            { 1f, 2f, 3f },
            { 1f, 2f, 3f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["layer"] = activation,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state, batchCount: 1);

        var pruned = state.NamedWeights["layer.weight"].cpu();

        // Column 2 (largest activation norm) must survive in both rows
        pruned[0, 2].item<float>().Should().Be(1f);
        pruned[1, 2].item<float>().Should().Be(1f);

        // Column 0 (smallest activation norm) must be pruned in both rows
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[1, 0].item<float>().Should().Be(0f);
    }

    [Fact]
    public void OnEnd_LargeWeightWithSmallActivation_StillPrunedWhenSaliencyLow()
    {
        using var weight = tensor(new float[,]
        {
            { 100f, 1f },
        });
        using var activation = tensor(new float[,]
        {
            { 0.01f, 100f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["proj.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = activation,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state, batchCount: 1);

        var pruned = state.NamedWeights["proj.weight"].cpu();
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[0, 1].item<float>().Should().Be(1f);
    }

    [Fact]
    public void OnEnd_NoActivationForTargetedLayer_Throws()
    {
        using var weight = ones(2, 3);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>(),
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*activation*layer*");
    }

    [Fact]
    public void OnEnd_LayerActivationsNullOnState_Throws()
    {
        using var weight = ones(2, 3);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void OnEnd_AccumulatesActivationNormsAcrossBatches()
    {
        using var weight = tensor(new float[,]
        {
            { 1f, 1f },
        });
        using var act1 = tensor(new float[,]
        {
            { 1f, 5f },
        });
        using var act2 = tensor(new float[,]
        {
            { 1f, 5f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["p.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["p"] = act1,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        modifier.Initialize(state);
        modifier.OnStart(state);

        state.CurrentBatch = act1;
        modifier.OnBatch(state);

        state.LayerActivations!["p"] = act2;
        state.CurrentBatch = act2;
        modifier.OnBatch(state);

        modifier.OnEnd(state);
        modifier.Finalize(state);

        var pruned = state.NamedWeights["p.weight"].cpu();
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[0, 1].item<float>().Should().Be(1f);
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new WandaModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_SparsityOutOfRange_Throws(float sparsity)
    {
        var act = () => new WandaModifier(new WandaConfig { Sparsity = sparsity });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    private static void RunLifecycle(WandaModifier modifier, CompressionState state, int batchCount)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        for (var i = 0; i < batchCount; i++)
        {
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}

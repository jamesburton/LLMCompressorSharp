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
/// Tests for <see cref="MagnitudePruningModifier"/>.
/// </summary>
public class MagnitudePruningModifierTests
{
    [Fact]
    public void OnEnd_FiftyPercentSparsity_ZerosLowestMagnitudeHalf()
    {
        using var weight = tensor(new float[] { 1f, -2f, 3f, -4f, 5f, -6f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state);

        var pruned = state.NamedWeights["w"].cpu().data<float>().ToArray();
        pruned[0].Should().Be(0f);
        pruned[1].Should().Be(0f);
        pruned[2].Should().Be(0f);
        pruned[3].Should().Be(-4f);
        pruned[4].Should().Be(5f);
        pruned[5].Should().Be(-6f);
    }

    [Fact]
    public void OnEnd_ZeroSparsity_LeavesWeightsUnchanged()
    {
        var original = new float[] { 1f, -2f, 3f, -4f };
        using var weight = tensor(original);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.0f });
        RunLifecycle(modifier, state);

        state.NamedWeights["w"].cpu().data<float>().ToArray().Should().Equal(original);
    }

    [Fact]
    public void OnEnd_FullSparsity_ZerosAllWeights()
    {
        using var weight = tensor(new float[] { 1f, -2f, 3f, -4f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 1.0f });
        RunLifecycle(modifier, state);

        state.NamedWeights["w"].cpu().data<float>().ToArray()
            .Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void OnEnd_AppliesPerWeightTensorIndependently()
    {
        using var w1 = tensor(new float[] { 1f, 100f });
        using var w2 = tensor(new float[] { 0.1f, 0.2f });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w1"] = w1,
            ["w2"] = w2,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state);

        var p1 = state.NamedWeights["w1"].cpu().data<float>().ToArray();
        var p2 = state.NamedWeights["w2"].cpu().data<float>().ToArray();
        p1[0].Should().Be(0f);
        p1[1].Should().Be(100f);
        p2[0].Should().Be(0f);
        p2[1].Should().BeApproximately(0.2f, 1e-6f);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_SparsityOutOfRange_Throws(float sparsity)
    {
        var act = () => new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = sparsity });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    private static void RunLifecycle(MagnitudePruningModifier modifier, CompressionState state)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}

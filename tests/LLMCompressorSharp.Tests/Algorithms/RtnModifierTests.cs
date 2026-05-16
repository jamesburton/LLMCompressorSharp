// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Rtn;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="RtnModifier"/> — round-to-nearest weight quantization.
/// </summary>
public class RtnModifierTests
{
    // NOTE: Input values are kept within the int8 symmetric grid [-127, 127] so that scale = 1.0
    // and assertions can use exact equality. Using a value outside this range (e.g. 200f) would
    // produce scale ≈ 1.5748, causing all grid-aligned assertions to fail.
    [Fact]
    public void OnEnd_SymmetricInt8_QuantizesTargetedWeightToGrid()
    {
        using var weight = tensor(new float[] { -127f, -64.4f, 0f, 64.6f, 127f, 127f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["layer.weight"].cpu().data<float>().ToArray();
        quantized[0].Should().Be(-127f);
        quantized[1].Should().Be(-64f);
        quantized[2].Should().Be(0f);
        quantized[3].Should().Be(65f);
        quantized[4].Should().Be(127f);

        // Value equal to absMax is clamped at the grid boundary.
        quantized[5].Should().Be(127f);
    }

    [Fact]
    public void OnEnd_PerChannelStrategy_QuantizesEachRowSeparately()
    {
        using var weight = tensor(new float[,]
        {
            { -2f, 0f, 2f },
            { -10f, 5f, 10f },
        });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerChannel,
            ChannelAxis = 0,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["layer.weight"].cpu();
        quantized[0, 0].item<float>().Should().BeApproximately(-2f, 1e-4f);
        quantized[0, 2].item<float>().Should().BeApproximately(2f, 1e-4f);
        quantized[1, 0].item<float>().Should().BeApproximately(-10f, 1e-4f);
        quantized[1, 2].item<float>().Should().BeApproximately(10f, 1e-4f);
    }

    [Fact]
    public void OnEnd_TargetsFiltering_OnlyAffectsMatchingWeights()
    {
        using var matching = tensor(new float[] { -10f, 0f, 10f });
        using var nonMatching = tensor(new float[] { -10f, 0f, 10f });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.q_proj.weight"] = matching,
            ["model.lm_head.weight"] = nonMatching,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 4,
            Symmetric = true,
            Targets = new[] { "model.layer.*" },
            Ignore = new[] { "*.lm_head.*" },
        });

        RunLifecycle(modifier, state);

        var quantizedMatch = state.NamedWeights["model.layer.0.q_proj.weight"].cpu().data<float>().ToArray();
        quantizedMatch[0].Should().BeApproximately(-10f, 2f);
        quantizedMatch[1].Should().BeApproximately(0f, 2f);

        var untouched = state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray();
        untouched.Should().Equal(new float[] { -10f, 0f, 10f });
    }

    [Fact]
    public void OnEnd_AsymmetricInt8_UsesZeroPoint()
    {
        using var weight = tensor(new float[] { -1f, 0f, 3f, 7f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = false,
            Strategy = QuantizationStrategy.PerTensor,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["w"].cpu().data<float>().ToArray();
        quantized[0].Should().BeApproximately(-1f, 0.1f);
        quantized[1].Should().BeApproximately(0f, 0.1f);
        quantized[3].Should().BeApproximately(7f, 0.1f);
    }

    [Fact]
    public void Name_MatchesConfigType()
    {
        var modifier = new RtnModifier(new RtnConfig());
        modifier.Name.Should().Be("RTN");
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new RtnModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    private static void RunLifecycle(RtnModifier modifier, CompressionState state)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}

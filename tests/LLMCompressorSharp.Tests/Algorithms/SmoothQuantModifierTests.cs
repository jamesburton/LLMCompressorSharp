// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="SmoothQuantModifier"/>.
/// </summary>
public class SmoothQuantModifierTests
{
    [Fact]
    public void Name_IsSmoothQuant()
    {
        var modifier = new SmoothQuantModifier(new SmoothQuantConfig());
        modifier.Name.Should().Be("SmoothQuant");
    }

    [Fact]
    public void OnEnd_AppliesScaleSuchThatOutputIsPreserved()
    {
        using var smoothW = tensor(new float[] { 2f, 4f });
        using var balanceW = tensor(new float[,]
        {
            { 1f, 1f },
            { 1f, 1f },
        });
        using var activation = tensor(new float[,]
        {
            { 1f, 4f },
            { -0.5f, -2f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["norm.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = activation,
            },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            SmoothingStrength = 0.5f,
            Mappings =
            {
                new SmoothQuantMapping("norm.weight", "proj.weight", "proj"),
            },
        });

        RunLifecycle(modifier, state, batchCount: 1);

        var gNew = state.NamedWeights["norm.weight"].cpu().data<float>().ToArray();
        var wNew = state.NamedWeights["proj.weight"].cpu();

        gNew[0].Should().BeApproximately(2f, 1e-4f);
        gNew[1].Should().BeApproximately(2f, 1e-4f);

        wNew[0, 0].item<float>().Should().BeApproximately(1f, 1e-4f);
        wNew[0, 1].item<float>().Should().BeApproximately(2f, 1e-4f);
        wNew[1, 0].item<float>().Should().BeApproximately(1f, 1e-4f);
        wNew[1, 1].item<float>().Should().BeApproximately(2f, 1e-4f);
    }

    [Fact]
    public void OnEnd_PreservesMathematicalIdentity()
    {
        using var g = tensor(new float[] { 1f, 2f, 3f });
        using var W = tensor(new float[,]
        {
            { 0.5f, 1f, -1f },
            { 2f, -0.5f, 0.25f },
        });
        using var X = tensor(new float[,]
        {
            { 0.1f, 0.5f, -1f },
            { 0.2f, -0.3f, 0.7f },
        });

        using var Xg = X.mul(g);
        using var yRef = Xg.matmul(W.t());
        var yRefArr = yRef.cpu().data<float>().ToArray();

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = g,
            ["proj.weight"] = W,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = Xg,
            },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            SmoothingStrength = 0.5f,
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });
        RunLifecycle(modifier, state, batchCount: 1);

        using var gNew = state.NamedWeights["g.weight"];
        using var WNew = state.NamedWeights["proj.weight"];
        using var XgNew = X.mul(gNew);
        using var ySmoothed = XgNew.matmul(WNew.t());
        var ySmoothedArr = ySmoothed.cpu().data<float>().ToArray();

        for (var i = 0; i < yRefArr.Length; i++)
        {
            ySmoothedArr[i].Should().BeApproximately(yRefArr[i], 1e-4f);
        }
    }

    [Fact]
    public void OnEnd_MissingMappingActivation_Throws()
    {
        using var smoothW = tensor(new float[] { 1f, 1f });
        using var balanceW = tensor(new float[,]
        {
            { 1f, 1f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>(),
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });

        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>().WithMessage("*activation*proj*");
    }

    [Fact]
    public void OnEnd_SmoothWeightNot1D_Throws()
    {
        using var smoothW = tensor(new float[,]
        {
            { 1f, 1f },
        });
        using var balanceW = tensor(new float[,]
        {
            { 1f, 1f },
        });
        using var activation = ones(2, 2);

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor> { ["proj"] = activation },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });
        var act = () => RunLifecycle(modifier, state, batchCount: 1);
        act.Should().Throw<InvalidOperationException>().WithMessage("*1-D*");
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_AlphaOutOfRange_Throws(float alpha)
    {
        var act = () => new SmoothQuantModifier(new SmoothQuantConfig { SmoothingStrength = alpha });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new SmoothQuantModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    private static void RunLifecycle(SmoothQuantModifier modifier, CompressionState state, int batchCount)
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

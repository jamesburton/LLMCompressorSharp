// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Output;
using LLMCompressorSharp.Core.Recipes;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Integration;

/// <summary>
/// End-to-end: parse YAML → register algorithms → build modifiers → run session → save → reload.
/// </summary>
[Collection("ModifierRegistry")]
public class EndToEndRecipeTests : IDisposable
{
    private readonly string _tempPath;

    public EndToEndRecipeTests()
    {
        ModifierRegistry.Clear();
        AlgorithmsRegistration.RegisterAll();
        _tempPath = Path.Combine(
            Path.GetTempPath(),
            $"llmc-e2e-{Guid.NewGuid():N}.safetensors");
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
        if (File.Exists(_tempPath))
        {
            File.Delete(_tempPath);
        }
    }

    [Fact]
    public void Recipe_RtnW8A16_ThenMagnitudePruning_ProducesSparseQuantizedWeights()
    {
        var yaml = @"
stages:
  - name: quantize
    modifiers:
      - type: RTN
        num_bits: 8
        symmetric: true
        targets: [model.*]
        ignore: [model.lm_head.*]
  - name: prune
    modifiers:
      - type: MagnitudePruning
        sparsity: 0.5
        targets: [model.*]
        ignore: [model.lm_head.*]
";

        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(2);

        using var layerWeight = tensor(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        using var lmHeadWeight = tensor(new float[] { 100f, 200f, 300f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.weight"] = layerWeight,
            ["model.lm_head.weight"] = lmHeadWeight,
        });

        var session = new CompressionSession(modifiers);
        var status = session.Run(state, Enumerable.Empty<Tensor>());

        status.Should().Be(SessionStatus.Completed);

        state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray()
            .Should().Equal(new float[] { 100f, 200f, 300f });

        var processed = state.NamedWeights["model.layer.0.weight"].cpu().data<float>().ToArray();
        processed.Count(v => v == 0f).Should().Be(4);
        processed.Count(v => v != 0f).Should().Be(4);

        // Round-trip through safetensors.
        SafetensorsWriter.Save(state, _tempPath);
        var reloaded = SafetensorsWriter.Load(_tempPath);
        try
        {
            reloaded.Should().ContainKey("model.layer.0.weight");
            reloaded["model.layer.0.weight"].cpu().data<float>().ToArray().Should().Equal(processed);
            reloaded["model.lm_head.weight"].cpu().data<float>().ToArray()
                .Should().Equal(new float[] { 100f, 200f, 300f });
        }
        finally
        {
            foreach (var t in reloaded.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Recipe_WandaPruning_UsesActivationsToGuideSparsity()
    {
        var yaml = @"
stages:
  - name: prune
    modifiers:
      - type: WANDA
        sparsity: 0.5
        targets: [model.*]
        ignore: [model.lm_head.*]
";

        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(1);

        using var weight = ones(2, 4);
        using var activation = tensor(new float[,]
        {
            { 0.1f, 0.1f, 0.1f, 10f },
            { 0.1f, 0.1f, 0.1f, 10f },
        });
        using var lmHead = ones(2, 4);

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.weight"] = weight,
            ["model.lm_head.weight"] = lmHead,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["model.layer.0"] = activation,
            },
        };

        var session = new CompressionSession(modifiers);
        var status = session.Run(state, new[] { activation });
        status.Should().Be(SessionStatus.Completed);

        var processed = state.NamedWeights["model.layer.0.weight"].cpu();
        processed[0, 3].item<float>().Should().Be(1f);
        processed[1, 3].item<float>().Should().Be(1f);

        state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray()
            .Should().AllSatisfy(v => v.Should().Be(1f));
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.Core.Recipes;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeBuilder"/> — recipe → modifier list materialization,
/// plus the end-to-end Recipe → CompressionSession integration.
/// </summary>
/// <remarks>
/// Placed in the "ModifierRegistry" collection to prevent parallel execution against other test classes
/// that mutate the shared static <see cref="ModifierRegistry"/>.
/// </remarks>
[Collection("ModifierRegistry")]
public class RecipeBuilderTests : IDisposable
{
    public RecipeBuilderTests()
    {
        ModifierRegistry.Clear();
        ModifierRegistry.Register<EchoConfig>("Echo", c => new EchoModifier(c));
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
    }

    [Fact]
    public void Build_ProducesModifiersInDeclarationOrder()
    {
        var yaml = @"
stages:
  - name: a
    modifiers:
      - type: Echo
        tag: first
  - name: b
    modifiers:
      - type: Echo
        tag: second
      - type: Echo
        tag: third
";
        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(3);
        modifiers.Cast<EchoModifier>().Select(e => e.Tag).Should().Equal("first", "second", "third");
    }

    [Fact]
    public void Build_UnregisteredType_Throws()
    {
        ModifierRegistry.Unregister("Echo");

        var recipe = new Recipe
        {
            Stages =
            {
                new Stage
                {
                    Name = "stage",
                    Modifiers = { new EchoConfig { Tag = "x" } },
                },
            },
        };

        var act = () => RecipeBuilder.Build(recipe);
        act.Should().Throw<RecipeParseException>().WithMessage("*Echo*not registered*");
    }

    [Fact]
    public void EndToEnd_RecipeRunsAllModifiersThroughSession()
    {
        var yaml = @"
stages:
  - name: only
    modifiers:
      - type: Echo
        tag: alpha
      - type: Echo
        tag: beta
";
        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);

        var session = new CompressionSession(modifiers);
        var state = new CompressionState(new Dictionary<string, Tensor>());

        var status = session.Run(state, new[] { ones(1), ones(1) });

        status.Should().Be(SessionStatus.Completed);
        modifiers.Cast<EchoModifier>()
            .Select(m => m.Log)
            .Should().AllSatisfy(l => l.Should().Equal("Init", "Start", "Batch", "Batch", "End", "Finalize"));
    }

    private sealed class EchoConfig : ModifierConfig
    {
        public override string Type => "Echo";

        public string Tag { get; set; } = string.Empty;
    }

    private sealed class EchoModifier : ModifierBase
    {
        public EchoModifier(EchoConfig config)
            : base("Echo", config.Targets, config.Ignore)
        {
            Tag = config.Tag;
        }

        public string Tag { get; }

        public List<string> Log { get; } = new();

        protected override void OnInitialize(CompressionState state) => Log.Add("Init");

        protected override void OnStartCore(CompressionState state) => Log.Add("Start");

        protected override void OnBatchCore(CompressionState state) => Log.Add("Batch");

        protected override void OnEndCore(CompressionState state) => Log.Add("End");

        protected override void OnFinalizeCore(CompressionState state) => Log.Add("Finalize");
    }
}

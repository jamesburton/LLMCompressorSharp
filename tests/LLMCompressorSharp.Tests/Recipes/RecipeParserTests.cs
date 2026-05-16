// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeParser"/> YAML deserialization with polymorphic modifier dispatch.
/// </summary>
/// <remarks>
/// Placed in the "ModifierRegistry" collection to prevent parallel execution against other test classes
/// that mutate the shared static <see cref="LLMCompressorSharp.Core.Recipes.ModifierRegistry"/>.
/// </remarks>
[Collection("ModifierRegistry")]
public class RecipeParserTests : IDisposable
{
    /// <summary>Initializes a new instance of the <see cref="RecipeParserTests"/> class.</summary>
    public RecipeParserTests()
    {
        ModifierRegistry.Clear();
        ModifierRegistry.Register<TestModifierConfig>("TestModifier", c => new TestModifier(c));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        ModifierRegistry.Clear();
    }

    [Fact]
    public void Parse_SingleStage_SingleModifier_ProducesRecipe()
    {
        var yaml = @"
stages:
  - name: quant_stage
    modifiers:
      - type: TestModifier
        scale: 0.5
";

        var recipe = RecipeParser.Parse(yaml);

        recipe.Stages.Should().HaveCount(1);
        recipe.Stages[0].Name.Should().Be("quant_stage");
        recipe.Stages[0].Modifiers.Should().HaveCount(1);
        var cfg = recipe.Stages[0].Modifiers[0].Should().BeOfType<TestModifierConfig>().Subject;
        cfg.Scale.Should().Be(0.5f);
        cfg.Type.Should().Be("TestModifier");
    }

    [Fact]
    public void Parse_ParsesTargetsAndIgnoreLists()
    {
        var yaml = @"
stages:
  - name: stage
    modifiers:
      - type: TestModifier
        targets: [layer.0, layer.1]
        ignore: [lm_head]
        scale: 1.0
";
        var recipe = RecipeParser.Parse(yaml);
        var cfg = (TestModifierConfig)recipe.Stages[0].Modifiers[0];
        cfg.Targets.Should().Equal("layer.0", "layer.1");
        cfg.Ignore.Should().Equal("lm_head");
    }

    [Fact]
    public void Parse_MultipleStages_ExecutionOrderPreserved()
    {
        var yaml = @"
stages:
  - name: first
    modifiers:
      - type: TestModifier
        scale: 0.1
  - name: second
    modifiers:
      - type: TestModifier
        scale: 0.2
";
        var recipe = RecipeParser.Parse(yaml);
        recipe.Stages.Should().HaveCount(2);
        recipe.Stages[0].Name.Should().Be("first");
        recipe.Stages[1].Name.Should().Be("second");
    }

    [Fact]
    public void Parse_UnregisteredModifierType_Throws()
    {
        var yaml = @"
stages:
  - name: s
    modifiers:
      - type: NotRegistered
        scale: 1.0
";
        var act = () => RecipeParser.Parse(yaml);
        act.Should().Throw<RecipeParseException>().WithMessage("*NotRegistered*not registered*");
    }

    [Fact]
    public void Parse_MissingTypeField_Throws()
    {
        var yaml = @"
stages:
  - name: s
    modifiers:
      - scale: 1.0
";
        var act = () => RecipeParser.Parse(yaml);
        act.Should().Throw<RecipeParseException>().WithMessage("*type*");
    }

    [Fact]
    public void Parse_EnumerateModifiers_YieldsStageIndexAndName()
    {
        var yaml = @"
stages:
  - name: a
    modifiers:
      - type: TestModifier
        scale: 0.1
      - type: TestModifier
        scale: 0.2
  - name: b
    modifiers:
      - type: TestModifier
        scale: 0.3
";
        var recipe = RecipeParser.Parse(yaml);
        var enumerated = recipe.EnumerateModifiers().ToArray();
        enumerated.Should().HaveCount(3);
        enumerated[0].StageIndex.Should().Be(0);
        enumerated[0].StageName.Should().Be("a");
        enumerated[2].StageIndex.Should().Be(1);
        enumerated[2].StageName.Should().Be("b");
    }

    private sealed class TestModifierConfig : ModifierConfig
    {
        public override string Type => "TestModifier";

        public float Scale { get; set; }
    }

    private sealed class TestModifier : ModifierBase
    {
        public TestModifier(TestModifierConfig config)
            : base("TestModifier", config.Targets, config.Ignore)
        {
            Scale = config.Scale;
        }

        public float Scale { get; }

        protected override void OnInitialize(CompressionState state)
        {
        }

        protected override void OnEndCore(CompressionState state)
        {
        }
    }
}

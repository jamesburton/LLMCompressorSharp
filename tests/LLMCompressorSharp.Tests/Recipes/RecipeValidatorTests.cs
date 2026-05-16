// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeValidator"/> and <see cref="MustPrecedeRule"/>.
/// </summary>
public class RecipeValidatorTests
{
    [Fact]
    public void Validate_EmptyRecipe_ReportsNoViolations()
    {
        var validator = new RecipeValidator();
        var recipe = new Recipe();

        var violations = validator.Validate(recipe);
        violations.Should().BeEmpty();
    }

    [Fact]
    public void MustPrecedeRule_PredecessorBeforeSuccessor_NoViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("AWQ", 1), ("Quantization", 1));

        validator.Validate(recipe).Should().BeEmpty();
    }

    [Fact]
    public void MustPrecedeRule_SuccessorWithoutPredecessor_ReportsViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 1));

        validator.Validate(recipe)
            .Should().ContainSingle()
            .Which.Should().Contain("'Quantization' must be preceded by a 'AWQ'");
    }

    [Fact]
    public void MustPrecedeRule_SuccessorBeforePredecessor_ReportsViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 1), ("AWQ", 1));

        validator.Validate(recipe).Should().ContainSingle();
    }

    [Fact]
    public void MustPrecedeRule_MultipleSuccessorsWithoutPredecessor_ReportsMultiple()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 2));

        validator.Validate(recipe).Should().HaveCount(2);
    }

    [Fact]
    public void Validate_MultipleRules_AllRun()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("A", "B"));
        validator.AddRule(new MustPrecedeRule("C", "D"));

        var recipe = BuildRecipe(("B", 1), ("D", 1));
        validator.Validate(recipe).Should().HaveCount(2);
    }

    private static Recipe BuildRecipe(params (string Type, int Count)[] entries)
    {
        var recipe = new Recipe();
        foreach (var (type, count) in entries)
        {
            var stage = new Stage { Name = type };
            for (var i = 0; i < count; i++)
            {
                stage.Modifiers.Add(new GenericConfig(type));
            }

            recipe.Stages.Add(stage);
        }

        return recipe;
    }

    private sealed class GenericConfig : ModifierConfig
    {
        public GenericConfig(string type)
        {
            TypeName = type;
        }

        public override string Type => TypeName;

        private string TypeName { get; }
    }
}

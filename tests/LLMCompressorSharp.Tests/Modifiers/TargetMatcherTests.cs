// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Modifiers;
using Xunit;

namespace LLMCompressorSharp.Tests.Modifiers;

/// <summary>
/// Tests for <see cref="TargetMatcher"/> — glob-style name pattern filtering.
/// </summary>
public class TargetMatcherTests
{
    [Fact]
    public void Matches_EmptyTargets_MatchesAnything()
    {
        TargetMatcher.Matches("anything", Array.Empty<string>(), Array.Empty<string>())
            .Should().BeTrue();
    }

    [Fact]
    public void Matches_ExactTarget_MatchesOnlyExactName()
    {
        var targets = new[] { "Linear" };
        TargetMatcher.Matches("Linear", targets, Array.Empty<string>()).Should().BeTrue();
        TargetMatcher.Matches("Linear2", targets, Array.Empty<string>()).Should().BeFalse();
    }

    [Theory]
    [InlineData("model.layers.0.q_proj", "model.layers.*.q_proj", true)]
    [InlineData("model.layers.15.q_proj", "model.layers.*.q_proj", true)]
    [InlineData("model.layers.0.k_proj", "model.layers.*.q_proj", false)]
    [InlineData("model.lm_head", "model.lm_head", true)]
    public void Matches_GlobStarMatchesAnySegment(string name, string pattern, bool expected)
    {
        TargetMatcher.Matches(name, new[] { pattern }, Array.Empty<string>()).Should().Be(expected);
    }

    [Fact]
    public void Matches_IgnoreWins_OverTargetMatch()
    {
        var targets = new[] { "*" };
        var ignore = new[] { "lm_head" };
        TargetMatcher.Matches("lm_head", targets, ignore).Should().BeFalse();
        TargetMatcher.Matches("other", targets, ignore).Should().BeTrue();
    }

    [Fact]
    public void Filter_ReturnsOnlyMatchingNames()
    {
        var names = new[] { "layer.0", "layer.1", "lm_head", "embeddings" };
        var matched = TargetMatcher.Filter(names, new[] { "layer.*" }, Array.Empty<string>()).ToArray();
        matched.Should().Equal("layer.0", "layer.1");
    }

    [Fact]
    public void Matches_NullName_Throws()
    {
        var act = () => TargetMatcher.Matches(null!, Array.Empty<string>(), Array.Empty<string>());
        act.Should().Throw<ArgumentNullException>();
    }
}

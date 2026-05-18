// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GPTQConfig"/> defaults and validation.
/// </summary>
public class GPTQConfigTests
{
    [Fact]
    public void Defaults_AreReasonable()
    {
        var config = new GPTQConfig();
        config.Type.Should().Be("GPTQ");
        config.NumBits.Should().Be(4);
        config.Symmetric.Should().BeTrue();
        config.Strategy.Should().Be(QuantizationStrategy.PerChannel);
        config.ChannelAxis.Should().Be(0);
        config.BlockSize.Should().Be(128);
        config.DampeningFrac.Should().BeApproximately(0.01f, 1e-6f);
        config.ActOrder.Should().BeFalse();
        config.Targets.Should().BeNull();
        config.Ignore.Should().BeNull();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(17)]
    public void NumBits_OutOfRange_IsInvalid(int bits)
    {
        // The modifier (not the config) enforces this at quantization time.
        // Verify config itself doesn't validate — it is just a data bag.
        var config = new GPTQConfig { NumBits = bits };
        config.NumBits.Should().Be(bits); // no throwing in the config
    }

    [Fact]
    public void ActOrder_True_IsAccepted_InConfig()
    {
        // Config accepts it; the modifier throws NotSupportedException if true.
        var config = new GPTQConfig { ActOrder = true };
        config.ActOrder.Should().BeTrue();
    }

    [Fact]
    public void BlockSize_IsConfigurable()
    {
        var config = new GPTQConfig { BlockSize = 64 };
        config.BlockSize.Should().Be(64);
    }
}

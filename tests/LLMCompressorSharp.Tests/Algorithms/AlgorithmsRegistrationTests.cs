// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Algorithms.Rtn;
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="AlgorithmsRegistration"/>.
/// </summary>
[Collection("ModifierRegistry")]
public class AlgorithmsRegistrationTests : IDisposable
{
    public AlgorithmsRegistrationTests()
    {
        ModifierRegistry.Clear();
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
    }

    [Fact]
    public void RegisterAll_RegistersAllBuiltInAlgorithms()
    {
        AlgorithmsRegistration.RegisterAll();

        ModifierRegistry.Resolve("RTN").Should().NotBeNull();
        ModifierRegistry.Resolve("MagnitudePruning").Should().NotBeNull();
        ModifierRegistry.Resolve("WANDA").Should().NotBeNull();
        ModifierRegistry.Resolve("SmoothQuant").Should().NotBeNull();
        ModifierRegistry.Resolve("GPTQ").Should().NotBeNull();
    }

    [Fact]
    public void RegisterRtn_RegistersOnlyRtn()
    {
        AlgorithmsRegistration.RegisterRtn();
        ModifierRegistry.Resolve("RTN").Should().NotBeNull();
        ModifierRegistry.Resolve("MagnitudePruning").Should().BeNull();
    }

    [Fact]
    public void Resolve_AfterRegistration_FactoryProducesCorrectModifier()
    {
        AlgorithmsRegistration.RegisterAll();

        var rtnReg = ModifierRegistry.Resolve("RTN");
        rtnReg.Should().NotBeNull();
        var rtnInstance = rtnReg!.Factory(new RtnConfig());
        rtnInstance.Should().BeOfType<RtnModifier>();

        var pruneReg = ModifierRegistry.Resolve("MagnitudePruning");
        var pruneInstance = pruneReg!.Factory(new MagnitudePruningConfig());
        pruneInstance.Should().BeOfType<MagnitudePruningModifier>();

        var wandaReg = ModifierRegistry.Resolve("WANDA");
        var wandaInstance = wandaReg!.Factory(new WandaConfig());
        wandaInstance.Should().BeOfType<WandaModifier>();

        var sqReg = ModifierRegistry.Resolve("SmoothQuant");
        var sqInstance = sqReg!.Factory(new SmoothQuantConfig());
        sqInstance.Should().BeOfType<SmoothQuantModifier>();

        var gptqReg = ModifierRegistry.Resolve("GPTQ");
        var gptqInstance = gptqReg!.Factory(new GPTQConfig());
        gptqInstance.Should().BeOfType<GPTQModifier>();
    }
}

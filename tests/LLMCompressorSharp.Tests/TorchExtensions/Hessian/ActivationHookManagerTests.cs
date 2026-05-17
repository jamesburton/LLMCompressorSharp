// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="ActivationHookManager"/> — multi-module forward-hook lifecycle.
/// </summary>
/// <remarks>
/// TorchSharp 0.107.0 forward hooks only fire when a module is invoked via <c>.call()</c>, not
/// <c>.forward()</c> directly. Tests use <c>.call()</c> to exercise the hook infrastructure.
/// </remarks>
public class ActivationHookManagerTests
{
    [Fact]
    public void RegisterFor_LinearLayer_HookFiresOnCall()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(linear, accumulator);

        using var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } });

        // Hooks fire via .call(), not .forward() in TorchSharp 0.107.0
        using var output = linear.call(x);

        accumulator.SampleCount.Should().Be(1L);
    }

    [Fact]
    public void RegisterFor_MultipleLayers_EachAccumulatorIndependent()
    {
        using var layer1 = Linear(3, 4, hasBias: false);
        using var layer2 = Linear(4, 2, hasBias: false);
        using var accum1 = new HessianAccumulator(inFeatures: 3);
        using var accum2 = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(layer1, accum1);
        manager.RegisterFor(layer2, accum2);

        using var x = tensor(new float[,] { { 1f, 2f, 3f } });
        using var y1 = layer1.call(x);
        using var y2 = layer2.call(y1);

        accum1.SampleCount.Should().Be(1L);
        accum2.SampleCount.Should().Be(1L);
    }

    [Fact]
    public void Dispose_RemovesAllHooks_NoFurtherAccumulation()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);

        var manager = new ActivationHookManager();
        manager.RegisterFor(linear, accumulator);

        using var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } });
        using var y1 = linear.call(x);
        accumulator.SampleCount.Should().Be(1L);

        manager.Dispose();

        using var y2 = linear.call(x);
        accumulator.SampleCount.Should().Be(1L);
    }

    [Fact]
    public void Clear_RemovesHooks_AllowsReregistration()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(linear, accumulator);
        using (var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } }))
        using (var out1 = linear.call(x))
        {
        }

        manager.Clear();

        manager.RegisterFor(linear, accumulator);
        using (var x = tensor(new float[,] { { 5f, 6f, 7f, 8f } }))
        using (var out2 = linear.call(x))
        {
        }

        accumulator.SampleCount.Should().Be(2L);
    }

    [Fact]
    public void RegisterFor_NullLayer_Throws()
    {
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();
        var act = () => manager.RegisterFor(null!, accumulator);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void RegisterFor_NullAccumulator_Throws()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var manager = new ActivationHookManager();
        var act = () => manager.RegisterFor(linear, null!);
        act.Should().Throw<ArgumentNullException>();
    }
}

// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Modifiers;

/// <summary>
/// Tests for <see cref="ModifierBase"/> — lifecycle enforcement and target filtering.
/// </summary>
public class ModifierBaseTests
{
    [Fact]
    public void Lifecycle_HappyPath_AllHooksFireInOrder()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();

        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnBatch(state);
        modifier.OnBatch(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);

        modifier.EventLog.Should().Equal("Initialize", "Start", "Batch", "Batch", "End", "Finalize");
    }

    [Fact]
    public void OnBatch_BeforeInitialize_ThrowsLifecycleException()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        var act = () => modifier.OnBatch(state);
        act.Should().Throw<ModifierLifecycleException>()
            .WithMessage("*before Initialize*");
    }

    [Fact]
    public void Initialize_Twice_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        var act = () => modifier.Initialize(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*initialized twice*");
    }

    [Fact]
    public void OnEnd_BeforeOnStart_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        var act = () => modifier.OnEnd(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*before OnStart*");
    }

    [Fact]
    public void OnBatch_AfterOnEnd_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        var act = () => modifier.OnBatch(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*after OnEnd*");
    }

    [Fact]
    public void GetTargetedNames_FiltersByTargetsAndIgnore()
    {
        var modifier = new RecordingModifier(targets: new[] { "layer.*" }, ignore: new[] { "layer.5" });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.0"] = zeros(1),
            ["layer.5"] = zeros(1),
            ["layer.7"] = zeros(1),
            ["other"] = zeros(1),
        });

        var matched = modifier.GetTargetedNamesPublic(state).ToArray();
        matched.Should().BeEquivalentTo("layer.0", "layer.7");
    }

    [Fact]
    public void Name_Required_Throws_OnEmpty()
    {
        var act = () => new RecordingModifier(name: " ");
        act.Should().Throw<ArgumentException>();
    }

    private static CompressionState CreateState()
    {
        return new CompressionState(new Dictionary<string, Tensor>());
    }

    private sealed class RecordingModifier : ModifierBase
    {
        public RecordingModifier(string name = "Recording", IReadOnlyList<string>? targets = null, IReadOnlyList<string>? ignore = null)
            : base(name, targets, ignore)
        {
        }

        public List<string> EventLog { get; } = new();

        public IEnumerable<string> GetTargetedNamesPublic(CompressionState state) => GetTargetedNames(state);

        protected override void OnInitialize(CompressionState state) => EventLog.Add("Initialize");

        protected override void OnStartCore(CompressionState state) => EventLog.Add("Start");

        protected override void OnBatchCore(CompressionState state) => EventLog.Add("Batch");

        protected override void OnEndCore(CompressionState state) => EventLog.Add("End");

        protected override void OnFinalizeCore(CompressionState state) => EventLog.Add("Finalize");
    }
}

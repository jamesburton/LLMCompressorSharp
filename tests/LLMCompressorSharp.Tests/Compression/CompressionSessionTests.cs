// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Compression;

/// <summary>
/// Tests for <see cref="CompressionSession"/> orchestrator.
/// </summary>
public class CompressionSessionTests
{
    [Fact]
    public void Run_WithNoBatches_FiresInitStartEndFinalizeOnAllModifiers()
    {
        var a = new RecordingModifier("A");
        var b = new RecordingModifier("B");
        var session = new CompressionSession(new IModifier[] { a, b });

        var status = session.Run(NewState(), Enumerable.Empty<Tensor>());

        status.Should().Be(SessionStatus.Completed);
        a.EventLog.Should().Equal("Initialize", "Start", "End", "Finalize");
        b.EventLog.Should().Equal("Initialize", "Start", "End", "Finalize");
    }

    [Fact]
    public void Run_WithBatches_FiresOnBatchPerCalibrationSample()
    {
        var m = new RecordingModifier("A");
        var session = new CompressionSession(new IModifier[] { m });
        var batches = new[] { ones(2, 2), ones(2, 2), ones(2, 2) };

        session.Run(NewState(), batches);

        m.EventLog.Should().Equal(
            "Initialize", "Start", "Batch", "Batch", "Batch", "End", "Finalize");
        m.BatchIndices.Should().Equal(0, 1, 2);
    }

    [Fact]
    public void Run_ModifiersRunInDeclarationOrderPerPhase()
    {
        var a = new RecordingModifier("A");
        var b = new RecordingModifier("B");
        var observed = new List<string>();
        a.OnPhase = phase => observed.Add($"A.{phase}");
        b.OnPhase = phase => observed.Add($"B.{phase}");

        var session = new CompressionSession(new IModifier[] { a, b });
        session.Run(NewState(), new[] { ones(1) });

        observed.Should().Equal(
            "A.Initialize",
            "B.Initialize",
            "A.Start",
            "B.Start",
            "A.Batch",
            "B.Batch",
            "A.End",
            "B.End",
            "A.Finalize",
            "B.Finalize");
    }

    [Fact]
    public void Run_WhenModifierThrowsDuringBatch_TransitionsToFailedAndStillFinalizes()
    {
        var failing = new RecordingModifier("Failing")
        {
            ThrowOnBatch = new InvalidOperationException("boom"),
        };
        var session = new CompressionSession(new IModifier[] { failing });

        var status = session.Run(NewState(), new[] { ones(1) });

        status.Should().Be(SessionStatus.Failed);
        session.Failure.Should().BeOfType<InvalidOperationException>().Which.Message.Should().Be("boom");
        failing.EventLog.Should().Contain("Finalize");
    }

    [Fact]
    public void Run_CallingTwice_Throws()
    {
        var session = new CompressionSession(Array.Empty<IModifier>());
        session.Run(NewState(), Enumerable.Empty<Tensor>());
        var act = () => session.Run(NewState(), Enumerable.Empty<Tensor>());
        act.Should().Throw<InvalidOperationException>().WithMessage("*only be called once*");
    }

    [Fact]
    public void Run_SetsCurrentBatchAndClearsItAfterEnd()
    {
        var sawBatch = false;
        Tensor? batchAfterEnd = null;
        var m = new RecordingModifier("A")
        {
            BatchObserver = s => { sawBatch = s.CurrentBatch is not null; },
            EndObserver = s => { batchAfterEnd = s.CurrentBatch; },
        };

        var session = new CompressionSession(new IModifier[] { m });
        session.Run(NewState(), new[] { ones(1) });

        sawBatch.Should().BeTrue();
        batchAfterEnd.Should().BeNull();
    }

    private static CompressionState NewState()
    {
        return new CompressionState(new Dictionary<string, Tensor>());
    }

    private sealed class RecordingModifier : ModifierBase
    {
        public RecordingModifier(string name)
            : base(name)
        {
        }

        public List<string> EventLog { get; } = new();

        public List<int> BatchIndices { get; } = new();

        public Action<string>? OnPhase { get; set; }

        public Action<CompressionState>? BatchObserver { get; set; }

        public Action<CompressionState>? EndObserver { get; set; }

        public Exception? ThrowOnBatch { get; set; }

        protected override void OnInitialize(CompressionState state)
        {
            EventLog.Add("Initialize");
            OnPhase?.Invoke("Initialize");
        }

        protected override void OnStartCore(CompressionState state)
        {
            EventLog.Add("Start");
            OnPhase?.Invoke("Start");
        }

        protected override void OnBatchCore(CompressionState state)
        {
            EventLog.Add("Batch");
            BatchIndices.Add(state.CurrentBatchIndex);
            BatchObserver?.Invoke(state);
            OnPhase?.Invoke("Batch");
            if (ThrowOnBatch is not null)
            {
                throw ThrowOnBatch;
            }
        }

        protected override void OnEndCore(CompressionState state)
        {
            EventLog.Add("End");
            EndObserver?.Invoke(state);
            OnPhase?.Invoke("End");
        }

        protected override void OnFinalizeCore(CompressionState state)
        {
            EventLog.Add("Finalize");
            OnPhase?.Invoke("Finalize");
        }
    }
}

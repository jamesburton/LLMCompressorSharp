// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Output;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Output;

/// <summary>
/// Tests for <see cref="SafetensorsWriter"/> — safetensors round-trip of compression state.
/// </summary>
public class SafetensorsWriterTests : IDisposable
{
    private readonly string _tempPath;

    public SafetensorsWriterTests()
    {
        _tempPath = Path.Combine(
            Path.GetTempPath(),
            $"llmc-test-{Guid.NewGuid():N}.safetensors");
    }

    public void Dispose()
    {
        if (File.Exists(_tempPath))
        {
            File.Delete(_tempPath);
        }
    }

    [Fact]
    public void Save_ProducesAFileOnDisk()
    {
        using var w = tensor(new float[] { 1f, 2f, 3f });
        var state = new CompressionState(new Dictionary<string, Tensor> { ["w"] = w });

        SafetensorsWriter.Save(state, _tempPath);

        File.Exists(_tempPath).Should().BeTrue();
        new FileInfo(_tempPath).Length.Should().BeGreaterThan(0L);
    }

    [Fact]
    public void Save_ThenLoad_PreservesTensorValuesAndShapes()
    {
        using var w1 = tensor(new float[] { -1f, 0f, 1f, 2f });
        using var w2 = tensor(new float[,] { { 1f, 2f }, { 3f, 4f } });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["a.weight"] = w1,
            ["b.weight"] = w2,
        });

        SafetensorsWriter.Save(state, _tempPath);

        var loaded = SafetensorsWriter.Load(_tempPath);

        loaded.Should().ContainKey("a.weight");
        loaded.Should().ContainKey("b.weight");
        loaded["a.weight"].cpu().data<float>().ToArray().Should().Equal(new float[] { -1f, 0f, 1f, 2f });
        loaded["a.weight"].shape.Should().Equal(new long[] { 4 });
        loaded["b.weight"].shape.Should().Equal(new long[] { 2, 2 });

        foreach (var t in loaded.Values)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void Save_NullState_Throws()
    {
        var act = () => SafetensorsWriter.Save(null!, _tempPath);
        act.Should().Throw<ArgumentNullException>();
    }
}

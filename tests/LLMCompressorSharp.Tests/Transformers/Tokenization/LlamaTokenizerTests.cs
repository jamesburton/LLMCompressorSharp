// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Tokenization;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Tokenization;

/// <summary>
/// Tests for <see cref="LlamaTokenizer"/>.
/// </summary>
public class LlamaTokenizerTests
{
    [Fact]
    public void Constructor_NullSnapshotDir_Throws()
    {
        var act = () => new LlamaTokenizer(null!);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_NonExistentDir_Throws()
    {
        var act = () => new LlamaTokenizer("/nonexistent/path/no-tokenizer-here");
        act.Should().Throw<TokenizerLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void Constructor_DirWithoutVocabFiles_Throws()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"llmc-tokenizer-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var act = () => new LlamaTokenizer(tempDir);
            act.Should().Throw<TokenizerLoadException>().WithMessage("*vocab.json*");
        }
        finally
        {
            Directory.Delete(tempDir);
        }
    }

    [Fact]
    public void Encode_SmolLM2_RoundTripsKnownText()
    {
        var snapshotDir = GetSnapshotDir();
        Assert.SkipUnless(
            snapshotDir is not null,
            "SmolLM2-135M tokenizer not cached. Run scripts/download-test-models.ps1 to enable this test.");

        using var tokenizer = new LlamaTokenizer(snapshotDir!);
        const string text = "Hello, world!";

        var ids = tokenizer.Encode(text);
        ids.Should().NotBeEmpty();

        var decoded = tokenizer.Decode(ids);
        decoded.Trim().Should().Contain("Hello");
        decoded.Should().Contain("world");
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmptyOrSpecialTokens()
    {
        var snapshotDir = GetSnapshotDir();
        Assert.SkipUnless(
            snapshotDir is not null,
            "SmolLM2-135M tokenizer not cached.");

        using var tokenizer = new LlamaTokenizer(snapshotDir!);
        var ids = tokenizer.Encode(string.Empty);
        ids.Should().NotBeNull();
    }

    // SmolLM2-135M is a GPT-2/BPE tokenizer: vocab.json + merges.txt required (no tokenizer.model).
    private const string SmolLm2RepoId = "HuggingFaceTB/SmolLM2-135M";
    private const string SmolLm2Revision = "main";

    private static string? GetSnapshotDir()
    {
        try
        {
            var cacheRoot = HuggingFaceCache.ResolveCacheRoot(SystemEnvironment.Instance);
            var dir = HuggingFaceCache.GetSnapshotPath(cacheRoot, SmolLm2RepoId, SmolLm2Revision);

            return Directory.Exists(dir)
                && File.Exists(Path.Combine(dir, "vocab.json"))
                && File.Exists(Path.Combine(dir, "merges.txt"))
                ? dir
                : null;
        }
        catch
        {
            return null;
        }
    }
}

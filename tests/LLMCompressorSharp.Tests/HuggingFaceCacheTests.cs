// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using Xunit;

namespace LLMCompressorSharp.Tests;

/// <summary>
/// Tests for <see cref="HuggingFaceCache"/> path resolution. The cache root must follow the
/// standard HF layout so weights are shared with the Python ecosystem (huggingface_hub, transformers, datasets).
/// </summary>
public class HuggingFaceCacheTests
{
    [Fact]
    public void ResolveCacheRoot_WhenHfHomeIsSet_UsesHfHomeSlashHub()
    {
        var env = new TestEnvironment
        {
            HfHome = "/custom/hf",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/custom/hf", "hub"));
    }

    [Fact]
    public void ResolveCacheRoot_WhenHfHubCacheIsSet_TakesPrecedenceOverHfHome()
    {
        var env = new TestEnvironment
        {
            HfHome = "/custom/hf",
            HfHubCache = "/explicit/cache",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be("/explicit/cache");
    }

    [Fact]
    public void ResolveCacheRoot_WhenXdgCacheHomeIsSet_UsesXdgSlashHuggingfaceSlashHub()
    {
        var env = new TestEnvironment
        {
            XdgCacheHome = "/xdg/cache",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/xdg/cache", "huggingface", "hub"));
    }

    [Fact]
    public void ResolveCacheRoot_WithNoEnvVars_FallsBackToUserProfileCache()
    {
        var env = new TestEnvironment
        {
            UserProfile = "/home/test-user",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/home/test-user", ".cache", "huggingface", "hub"));
    }

    [Theory]
    [InlineData("HuggingFaceTB/SmolLM2-135M", "models--HuggingFaceTB--SmolLM2-135M")]
    [InlineData("meta-llama/Llama-3.2-1B", "models--meta-llama--Llama-3.2-1B")]
    [InlineData("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")]
    public void GetRepoFolderName_MatchesHuggingFaceLayout(string repoId, string expectedFolder)
    {
        HuggingFaceCache.GetRepoFolderName(repoId).Should().Be(expectedFolder);
    }

    [Fact]
    public void GetSnapshotPath_ComposesRootRepoAndRevision()
    {
        var path = HuggingFaceCache.GetSnapshotPath(
            cacheRoot: "/cache/huggingface/hub",
            repoId: "HuggingFaceTB/SmolLM2-135M",
            revision: "abc123def");

        path.Should().Be(Path.Combine(
            "/cache/huggingface/hub",
            "models--HuggingFaceTB--SmolLM2-135M",
            "snapshots",
            "abc123def"));
    }

    [Fact]
    public void GetRepoFolderName_WithInvalidRepoId_Throws()
    {
        var act = () => HuggingFaceCache.GetRepoFolderName("invalid-no-slash");
        act.Should().Throw<ArgumentException>().WithMessage("*org/repo*");
    }

    private sealed class TestEnvironment : IEnvironment
    {
        public string? HfHubCache { get; init; }

        public string? HfHome { get; init; }

        public string? XdgCacheHome { get; init; }

        public string UserProfile { get; init; } = "/home/default";
    }
}

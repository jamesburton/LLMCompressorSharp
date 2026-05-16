// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Integration;

/// <summary>
/// End-to-end integration tests against the real SmolLM2-135M model. Skipped when the model
/// is not present in the local HuggingFace cache.
/// </summary>
public class SmolLM2EndToEndTests
{
    private const string RepoId = "HuggingFaceTB/SmolLM2-135M";
    private const string Revision = "main";

    [Fact]
    public void LoadSmolLM2_FromHfCache_ProducesUsableModel()
    {
        Assert.SkipUnless(
            IsCached(),
            "SmolLM2-135M not present in HF cache. Run scripts/download-test-models.ps1 first.");

        var loaded = HuggingFaceLoader.LoadWithTokenizer(RepoId, Revision);
        using (loaded.Model)
        {
            loaded.Config.HiddenSize.Should().Be(576);
            loaded.Config.NumHiddenLayers.Should().Be(30);
            loaded.Config.VocabSize.Should().Be(49152);
            loaded.Tokenizer.Should().NotBeNull();

            var ids = loaded.Tokenizer!.Encode("The capital of France is");
            ids.Should().NotBeEmpty();

            using var inputIds = tensor(ids.Select(i => (long)i).ToArray()).reshape(1, ids.Length);
            using var logits = loaded.Model.forward(inputIds);

            logits.shape.Should().Equal(new long[] { 1, ids.Length, loaded.Config.VocabSize });

            using var lastLogits = logits.select(1, ids.Length - 1).squeeze(0);
            using var argmax = lastLogits.argmax();
            var nextTokenId = argmax.cpu().item<long>();
            nextTokenId.Should().BeGreaterThanOrEqualTo(0L);
            nextTokenId.Should().BeLessThan(loaded.Config.VocabSize);
        }
    }

    [Fact]
    public void LoadSmolLM2_ForwardPassProducesFiniteLogits()
    {
        Assert.SkipUnless(
            IsCached(),
            "SmolLM2-135M not present in HF cache.");

        var loaded = HuggingFaceLoader.LoadWithTokenizer(RepoId, Revision);
        using (loaded.Model)
        {
            var ids = loaded.Tokenizer!.Encode("Hello");
            using var inputIds = tensor(ids.Select(i => (long)i).ToArray()).reshape(1, ids.Length);
            using var logits = loaded.Model.forward(inputIds);

            // All values must be finite (no NaN/Inf).
            // Use the (logits == logits) NaN check + isinf trick if isfinite isn't available.
            using var nanCheck = logits.eq(logits); /* false where NaN */
            using var infCheck = logits.abs().lt(float.PositiveInfinity); /* false where +/-inf */
            using var bothFinite = nanCheck.logical_and(infCheck);
            using var allFinite = bothFinite.all();
            allFinite.cpu().item<bool>().Should().BeTrue();
        }
    }

    private static bool IsCached()
    {
        try
        {
            var cacheRoot = HuggingFaceCache.ResolveCacheRoot(SystemEnvironment.Instance);
            var dir = HuggingFaceCache.GetSnapshotPath(cacheRoot, RepoId, Revision);
            return Directory.Exists(dir)
                && File.Exists(Path.Combine(dir, "config.json"))
                && File.Exists(Path.Combine(dir, "tokenizer.json"))
                && (File.Exists(Path.Combine(dir, "model.safetensors"))
                    || File.Exists(Path.Combine(dir, "model.safetensors.index.json")));
        }
        catch
        {
            return false;
        }
    }
}

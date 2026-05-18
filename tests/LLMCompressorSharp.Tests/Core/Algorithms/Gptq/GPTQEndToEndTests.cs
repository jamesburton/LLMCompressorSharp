// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// End-to-end GPTQ tests: synthetic and real-model smoke tests.
/// </summary>
public class GPTQEndToEndTests
{
    private const string SmolLM2RepoId = "HuggingFaceTB/SmolLM2-135M";
    private const string SmolLM2Revision = "main";

    /// <summary>
    /// Full compression session with GPTQ using a synthetic 8×16 Linear layer and
    /// 10 calibration batches. Verifies that:
    /// - output weight is finite
    /// - output weight is NOT identical to FP original at low bit-width (4-bit).
    /// </summary>
    [Fact]
    public void SyntheticLinear_W4A16_OutputIsFiniteAndQuantized()
    {
        // Create a synthetic linear layer: out=8, in=16.
        using var layer = Linear(16, 8, hasBias: false);
        var originalWeight = layer.weight!.detach().clone();

        var config = new GPTQConfig
        {
            NumBits = 4,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
            BlockSize = 4,
            DampeningFrac = 0.01f,
        };

        var namedWeights = new Dictionary<string, Tensor>
        {
            ["fc.weight"] = layer.weight!.detach().clone(),
        };
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
            {
                ["fc"] = layer,
            },
        };

        var modifier = new GPTQModifier(config);
        modifier.Initialize(state);
        modifier.OnStart(state);

        // Feed 10 calibration batches via layer.call() to trigger hooks.
        for (int i = 0; i < 10; i++)
        {
            using var scope = NewDisposeScope();
            using var batch = randn(4, 16);  // [batch=4, in=16]
            using var unused = layer.call(batch);
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        var quantizedWeight = state.NamedWeights["fc.weight"];

        // Output must be finite.
        quantizedWeight.isfinite().all().item<bool>().Should().BeTrue();

        // At 4-bit, some weights will differ from the original (quantization error).
        using var diff = (quantizedWeight - originalWeight).abs();
        diff.max().item<float>().Should().BeGreaterThan(1e-6f, "4-bit must introduce measurable quantization error");

        originalWeight.Dispose();
    }

    [Fact]
    public void SyntheticLinear_W16A16_OutputCloseToOriginal()
    {
        // At 16-bit, fake-quantization should be very close to identity.
        using var layer = Linear(8, 4, hasBias: false);
        var originalWeight = layer.weight!.detach().clone();

        var config = new GPTQConfig
        {
            NumBits = 16,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
            BlockSize = 4,
            DampeningFrac = 0.01f,
        };

        var namedWeights = new Dictionary<string, Tensor>
        {
            ["fc.weight"] = layer.weight!.detach().clone(),
        };
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
            {
                ["fc"] = layer,
            },
        };

        var modifier = new GPTQModifier(config);
        modifier.Initialize(state);
        modifier.OnStart(state);

        for (int i = 0; i < 5; i++)
        {
            using var scope = NewDisposeScope();
            using var batch = randn(2, 8);
            using var unused = layer.call(batch);
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        using var diff = (state.NamedWeights["fc.weight"] - originalWeight).abs();
        diff.max()
            .item<float>()
            .Should()
            .BeLessThan(0.01f, "16-bit quantization should have < 0.01 max absolute error");

        originalWeight.Dispose();
    }

    [Fact]
    public void ViaCompressionSession_SingleLayer_Completes()
    {
        // Verify integration with the CompressionSession orchestrator.
        using var layer = Linear(8, 4, hasBias: false);
        var config = new GPTQConfig { NumBits = 8, BlockSize = 4, DampeningFrac = 0.01f };

        var namedWeights = new Dictionary<string, Tensor>
        {
            ["fc.weight"] = layer.weight!.detach().clone(),
        };
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
            {
                ["fc"] = layer,
            },
        };

        var session = new CompressionSession(new[] { new GPTQModifier(config) });

        // Generate calibration batches; trigger layer.call() before passing each to session.OnBatch.
        // Note: the standard session doesn't call the model forward — GPTQ relies on the caller
        // driving the model. For this test we drive via layer.call() before each batch is observed.
        var batches = new List<Tensor>();
        for (int i = 0; i < 5; i++)
        {
            var b = randn(2, 8);

            // Trigger the hook so the accumulator sees b.
            using var unused = layer.call(b);
            batches.Add(b);
        }

        var status = session.Run(state, batches);

        foreach (var b in batches)
        {
            b.Dispose();
        }

        status.Should().Be(SessionStatus.Completed);
        state.NamedWeights["fc.weight"].isfinite().all().item<bool>().Should().BeTrue();
    }

    /// <summary>
    /// Smoke test: load one attention <c>q_proj</c> from SmolLM2-135M and GPTQ-quantize it.
    /// Skipped if the model is not in the HuggingFace cache.
    /// </summary>
    [Fact]
    public void SmolLM2_135M_QProjLayer_W4A16_OutputIsFinite()
    {
        Assert.SkipUnless(
            IsSmolLM2Cached(),
            "SmolLM2-135M not found in HF cache. Run scripts/download-test-models.ps1 first.");

        var loaded = HuggingFaceLoader.Load(SmolLM2RepoId, SmolLM2Revision);
        using (loaded.Model)
        {
            // Locate the q_proj of layer 0. The Llama hierarchy registers it under
            // "layers.0.self_attn.q_proj" in named_modules().
            const string qProjLayerName = "layers.0.self_attn.q_proj";
            const string qProjWeightName = "layers.0.self_attn.q_proj.weight";

            Module<Tensor, Tensor>? qProjModule = null;
            foreach (var (name, mod) in loaded.Model.named_modules())
            {
                if (name == qProjLayerName && mod is Module<Tensor, Tensor> linear)
                {
                    qProjModule = linear;
                    break;
                }
            }

            qProjModule.Should().NotBeNull("q_proj not found in named_modules — check the layer key");

            var qProjWeight = loaded.Model.state_dict()[qProjWeightName];

            var config = new GPTQConfig
            {
                NumBits = 4,
                Symmetric = true,
                Strategy = QuantizationStrategy.PerChannel,
                ChannelAxis = 0,
                BlockSize = 128,
                DampeningFrac = 0.01f,
            };

            var namedWeights = new Dictionary<string, Tensor>
            {
                [qProjWeightName] = qProjWeight.detach().clone(),
            };
            var state = new CompressionState(namedWeights)
            {
                NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
                {
                    [qProjLayerName] = qProjModule!,
                },
            };

            var modifier = new GPTQModifier(config);
            modifier.Initialize(state);
            modifier.OnStart(state);

            // Feed 4 short calibration batches — enough to build a non-degenerate Hessian.
            int hiddenSize = (int)qProjWeight.shape[^1];
            for (int i = 0; i < 4; i++)
            {
                using var scope = NewDisposeScope();
                using var batch = randn(1, 16, hiddenSize);  // [batch=1, seq=16, hidden]
                using var unused = qProjModule!.call(batch);
                modifier.OnBatch(state);
            }

            modifier.OnEnd(state);
            modifier.Finalize(state);

            // The quantized q_proj weight must be finite.
            state.NamedWeights[qProjWeightName].isfinite().all().item<bool>().Should().BeTrue();
        }
    }

    private static bool IsSmolLM2Cached()
    {
        try
        {
            var cacheRoot = HuggingFaceCache.ResolveCacheRoot(SystemEnvironment.Instance);
            var dir = HuggingFaceCache.GetSnapshotPath(cacheRoot, SmolLM2RepoId, SmolLM2Revision);
            return Directory.Exists(dir)
                && File.Exists(Path.Combine(dir, "config.json"))
                && (File.Exists(Path.Combine(dir, "model.safetensors"))
                    || File.Exists(Path.Combine(dir, "model.safetensors.index.json")));
        }
        catch
        {
            return false;
        }
    }
}

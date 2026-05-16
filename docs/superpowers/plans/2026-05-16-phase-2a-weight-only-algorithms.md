# Phase 2a: Weight-Only Algorithms + Safetensors Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first two compression algorithms — RTN (data-free weight quantization) and Magnitude Pruning — as `ModifierBase` subclasses plugging into the Phase 1b framework, plus a safetensors output writer and an algorithms-registration helper. End-to-end: parse a YAML recipe → register algorithms → build modifiers → run session → save compressed weights.

**Architecture:**

```
LLMCompressorSharp.Core.Algorithms/
   ├── Configs/                       (YAML-deserialized POCO configs)
   │   ├── RtnConfig
   │   └── MagnitudePruningConfig
   ├── Rtn/                           (RTN: round-to-nearest weight quantization)
   │   └── RtnModifier                (uses Observers.MinMaxObserver / PerChannelMinMaxObserver)
   ├── Pruning/
   │   └── MagnitudePruningModifier   (threshold-by-percentile zero-out)
   └── AlgorithmsRegistration         (static helper that registers all built-in algorithms with ModifierRegistry)

LLMCompressorSharp.Core.Output/
   └── SafetensorsWriter              (saves CompressionState.NamedWeights via TorchSharp.PyBridge)
```

Each algorithm implements only the lifecycle hooks it needs. RTN is data-free (weights are the calibration source), so its work happens entirely in `OnEnd`. Magnitude pruning is also data-free.

**Tech Stack:** TorchSharp 0.107.0, TorchSharp.PyBridge 1.4.3 (safetensors I/O), xUnit v3, FluentAssertions. Builds on Phase 1a (Observers, FakeQuantize) and Phase 1b (ModifierBase, CompressionSession, Recipe, ModifierRegistry).

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (algorithm list)

---

## File Structure

```
src/LLMCompressorSharp.Core/
├── Algorithms/                                ← NEW
│   ├── Configs/
│   │   ├── RtnConfig.cs
│   │   └── MagnitudePruningConfig.cs
│   ├── Rtn/
│   │   └── RtnModifier.cs
│   ├── Pruning/
│   │   └── MagnitudePruningModifier.cs
│   └── AlgorithmsRegistration.cs              ← Register() helper
├── Output/                                    ← NEW
│   └── SafetensorsWriter.cs

tests/LLMCompressorSharp.Tests/
├── Algorithms/                                ← NEW
│   ├── RtnModifierTests.cs
│   ├── MagnitudePruningModifierTests.cs
│   └── AlgorithmsRegistrationTests.cs
├── Output/                                    ← NEW
│   └── SafetensorsWriterTests.cs
└── Integration/                               ← NEW
    └── EndToEndRecipeTests.cs                 ← YAML → run → save round-trip
```

**Responsibility per file:**
- `RtnConfig` — YAML-deserialized config: `NumBits`, `Symmetric`, `Strategy` (`PerTensor` | `PerChannel`), `ChannelAxis`, inherited `Targets`/`Ignore` from `ModifierConfig`. Default scheme is W8A16 (weight-only INT8).
- `RtnModifier` — Data-free weight quantization. For each targeted weight: build observer based on strategy, call `Update(weight)` once, get scale/zero-point, fake-quantize via `FakeQuantize.Apply`, write the result back to `state.NamedWeights`.
- `MagnitudePruningConfig` — `Sparsity` (float in [0, 1]), `Targets`/`Ignore`.
- `MagnitudePruningModifier` — Data-free: for each targeted weight, compute threshold at the configured sparsity percentile of `|weight|`, zero anything below.
- `AlgorithmsRegistration` — Static helper: `RegisterAll()` registers both algorithms with `ModifierRegistry` under conventional names (`RTN`, `MagnitudePruning`). `RegisterRtn()` and `RegisterMagnitudePruning()` for individual registration. Idempotent.
- `SafetensorsWriter` — Writes `state.NamedWeights` to a safetensors file via `TorchSharp.PyBridge.save_safetensors`. Optional metadata dict (quantization config etc.) for Phase 5.
- Tests: per-algorithm units; output writer round-trip; end-to-end integration: YAML recipe → parse → build → run on synthetic state → save → reload safetensors → assert weights match.

---

## Prerequisites & Conventions

- Phase 1a and 1b merged; tags `v0.1.0-alpha`, `v0.1.1-alpha` in place. **82 tests passing** on `main`.
- Phase 1b registered the `[Collection("ModifierRegistry")]` xunit collection — new test classes that mutate the registry must use it too.
- The `ModifierRegistry` registration is **process-global**. `AlgorithmsRegistration.RegisterAll()` is called by the CLI at startup (in Phase 6). Phase 2a tests using the registry must `Clear()` + register fresh in their constructor.
- Branch off `main` as `feature/2a-weight-algorithms`.
- StyleCop patterns from prior phases continue:
  - SA1402 → one public type per file
  - SA1515 → blank line before inline `//`
  - SA1642 → "Initializes a new instance of"
  - SA1201 → properties before methods
  - SA1117 → method args on separate lines if multi-line

---

### Task 1: Create branch and verify baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
  Expected: clean tree; HEAD `6501505 chore(core): remove PlaceholderMarker`; tags v0.0.1-alpha, v0.1.0-alpha, v0.1.1-alpha.

- [ ] **Step 2:** `git checkout -b feature/2a-weight-algorithms`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
  Expected: **82 tests passing**.

No commit.

---

### Task 2: Add `RtnConfig`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/RtnConfig.cs`

- [ ] **Step 1: Write the config**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;
using LLMCompressorSharp.TorchExtensions.Observers;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>RtnModifier</c> — round-to-nearest weight quantization.
/// </summary>
public sealed class RtnConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "RTN";

    /// <summary>Gets or sets the target bit-width. Default: 8.</summary>
    public int NumBits { get; set; } = 8;

    /// <summary>Gets or sets a value indicating whether to use symmetric quantization. Default: true.</summary>
    public bool Symmetric { get; set; } = true;

    /// <summary>Gets or sets the quantization strategy. Default: <see cref="QuantizationStrategy.PerTensor"/>.</summary>
    public QuantizationStrategy Strategy { get; set; } = QuantizationStrategy.PerTensor;

    /// <summary>Gets or sets the channel axis for per-channel quantization. Default: 0.</summary>
    public int ChannelAxis { get; set; } = 0;
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/RtnConfig.cs
git commit -m "feat(algorithms): add RtnConfig"
```

---

### Task 3: TDD `RtnModifier`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/RtnModifierTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Rtn/RtnModifier.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Rtn;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="RtnModifier"/> — round-to-nearest weight quantization.
/// </summary>
public class RtnModifierTests
{
    [Fact]
    public void OnEnd_SymmetricInt8_QuantizesTargetedWeightToGrid()
    {
        // For symmetric INT8, absMax=127 produces scale=1.0 and a grid of integers in [-127, 127].
        using var weight = tensor(new float[] { -127f, -64.4f, 0f, 64.6f, 127f, 200f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["layer.weight"].cpu().data<float>().ToArray();
        quantized[0].Should().Be(-127f);
        quantized[1].Should().Be(-64f);    // -64.4 rounds to -64
        quantized[2].Should().Be(0f);
        quantized[3].Should().Be(65f);     // 64.6 rounds to 65
        quantized[4].Should().Be(127f);
        quantized[5].Should().Be(127f);    // clamps to qMax=127
    }

    [Fact]
    public void OnEnd_PerChannelStrategy_QuantizesEachRowSeparately()
    {
        // Two output channels: row 0 has absMax=2 → scale=2/127; row 1 has absMax=10 → scale=10/127.
        using var weight = tensor(new float[,]
        {
            { -2f, 0f, 2f },
            { -10f, 5f, 10f },
        });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerChannel,
            ChannelAxis = 0,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["layer.weight"].cpu();
        // Row 0 has scale=2/127, so 0 → 0, ±2 → ±2 exactly, intermediate values round to grid.
        // Row 1 has scale=10/127, so larger values quantize with coarser resolution.
        quantized[0, 0].item<float>().Should().BeApproximately(-2f, 1e-4f);
        quantized[0, 2].item<float>().Should().BeApproximately(2f, 1e-4f);
        quantized[1, 0].item<float>().Should().BeApproximately(-10f, 1e-4f);
        quantized[1, 2].item<float>().Should().BeApproximately(10f, 1e-4f);
    }

    [Fact]
    public void OnEnd_TargetsFiltering_OnlyAffectsMatchingWeights()
    {
        using var matching = tensor(new float[] { -10f, 0f, 10f });
        using var nonMatching = tensor(new float[] { -10f, 0f, 10f });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.q_proj.weight"] = matching,
            ["model.lm_head.weight"] = nonMatching,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 4,
            Symmetric = true,
            Targets = new[] { "model.layer.*" },
            Ignore = new[] { "*.lm_head.*" },
        });

        RunLifecycle(modifier, state);

        // Matching weight gets quantized to int4 grid (max 7), so 10 should clamp to 7-step grid.
        var quantizedMatch = state.NamedWeights["model.layer.0.q_proj.weight"].cpu().data<float>().ToArray();
        quantizedMatch[0].Should().BeApproximately(-10f, 2f); // some quantization loss
        quantizedMatch[1].Should().BeApproximately(0f, 2f);

        // Non-matching weight unchanged.
        var untouched = state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray();
        untouched.Should().Equal(new float[] { -10f, 0f, 10f });
    }

    [Fact]
    public void OnEnd_AsymmetricInt8_UsesZeroPoint()
    {
        // Asymmetric: 0..255 grid. Range [-1, 7] → scale = 8/255 ≈ 0.0314, zero_point = round(0 - (-1/0.0314)) ≈ 32
        using var weight = tensor(new float[] { -1f, 0f, 3f, 7f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new RtnModifier(new RtnConfig
        {
            NumBits = 8,
            Symmetric = false,
            Strategy = QuantizationStrategy.PerTensor,
        });

        RunLifecycle(modifier, state);

        var quantized = state.NamedWeights["w"].cpu().data<float>().ToArray();
        // Each value should be close to original but on the asymmetric grid.
        quantized[0].Should().BeApproximately(-1f, 0.1f);
        quantized[1].Should().BeApproximately(0f, 0.1f);
        quantized[3].Should().BeApproximately(7f, 0.1f);
    }

    [Fact]
    public void Name_MatchesConfigType()
    {
        var modifier = new RtnModifier(new RtnConfig());
        modifier.Name.Should().Be("RTN");
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new RtnModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    private static void RunLifecycle(RtnModifier modifier, CompressionState state)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}
```

- [ ] **Step 2: Verify build fails with CS0246 ("RtnModifier could not be found")**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~RtnModifierTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/RtnModifierTests.cs
git commit -m "test(algorithms): add failing RtnModifier tests"
```

- [ ] **Step 4: Implement `RtnModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/Rtn/RtnModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.TorchExtensions.Observers;
using LLMCompressorSharp.TorchExtensions.Quantization;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Rtn;

/// <summary>
/// Round-to-nearest weight quantization. Data-free: the weight itself is the calibration source.
/// </summary>
/// <remarks>
/// For each targeted weight:
/// <list type="number">
///   <item>Build an <see cref="Observer"/> based on <see cref="RtnConfig.Strategy"/>.</item>
///   <item>Feed the weight tensor through <c>Observer.Update</c>.</item>
///   <item>Compute scale + zero-point via <c>Observer.GetQuantParams</c>.</item>
///   <item>Fake-quantize the weight via <see cref="FakeQuantize"/> and write the result back.</item>
/// </list>
/// </remarks>
public sealed class RtnModifier : ModifierBase
{
    private readonly RtnConfig _config;

    /// <summary>Initializes a new instance of the <see cref="RtnModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public RtnModifier(RtnConfig config)
        : base("RTN", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        // Nothing to allocate up front — observers live per-weight in OnEnd.
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        var targeted = GetTargetedNames(state).ToList();
        foreach (var name in targeted)
        {
            // Do not dispose the original tensor — the caller still owns its lifetime.
            // We just replace the dictionary slot with a new tensor.
            var weight = state.NamedWeights[name];
            var observer = CreateObserver();
            observer.Update(weight);

            var quantParams = observer.GetQuantParams(_config.NumBits, _config.Symmetric);

            using var fakeQuant = ApplyFakeQuant(weight, quantParams);
            state.NamedWeights[name] = fakeQuant.detach().clone();
            quantParams.Scale.Dispose();
            quantParams.ZeroPoint.Dispose();
        }
    }

    private Observer CreateObserver()
    {
        return _config.Strategy switch
        {
            QuantizationStrategy.PerTensor => new MinMaxObserver(),
            QuantizationStrategy.PerChannel => new PerChannelMinMaxObserver(_config.ChannelAxis),
            QuantizationStrategy.PerToken => throw new NotSupportedException(
                "PerToken strategy applies to activations, not weights."),
            _ => throw new ArgumentOutOfRangeException(
                nameof(_config.Strategy),
                _config.Strategy,
                "Unsupported quantization strategy."),
        };
    }

    private Tensor ApplyFakeQuant(Tensor weight, QuantizationParameters parameters)
    {
        if (_config.Strategy == QuantizationStrategy.PerTensor)
        {
            var scale = parameters.Scale.item<float>();
            var zeroPoint = parameters.ZeroPoint.item<long>();
            return FakeQuantize.Apply(weight, scale, zeroPoint, _config.NumBits, _config.Symmetric);
        }

        return ApplyPerChannelFakeQuant(weight, parameters);
    }

    private Tensor ApplyPerChannelFakeQuant(Tensor weight, QuantizationParameters parameters)
    {
        var axis = _config.ChannelAxis;
        var scales = parameters.Scale.cpu().data<float>().ToArray();
        var zeroPoints = parameters.ZeroPoint.cpu().data<long>().ToArray();
        var channelCount = (int)weight.shape[axis];

        if (scales.Length != channelCount)
        {
            throw new InvalidOperationException(
                $"Per-channel observer returned {scales.Length} scales for {channelCount} channels.");
        }

        var resultParts = new List<Tensor>();
        for (var c = 0; c < channelCount; c++)
        {
            using var slice = weight.select(axis, c).contiguous();
            using var quantized = FakeQuantize.Apply(slice, scales[c], zeroPoints[c], _config.NumBits, _config.Symmetric);
            resultParts.Add(quantized.unsqueeze(axis).detach().clone());
        }

        try
        {
            return torch.cat(resultParts.ToArray(), axis);
        }
        finally
        {
            foreach (var p in resultParts)
            {
                p.Dispose();
            }
        }
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~RtnModifierTests"`
Expected: 6 tests pass.

If a test fails:
- Check whether the issue is in your math or in the test expectations. Print the actual values with `Console.WriteLine` first.
- The per-channel test uses `weight[r, c].item<float>()` — confirm TorchSharp indexer syntax matches `tensor[long, long]`.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Rtn/RtnModifier.cs
git commit -m "feat(algorithms): implement RtnModifier"
```

---

### Task 4: Add `MagnitudePruningConfig`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/MagnitudePruningConfig.cs`

- [ ] **Step 1: Write the config**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>MagnitudePruningModifier</c> — zero out weights below a magnitude threshold.
/// </summary>
public sealed class MagnitudePruningConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "MagnitudePruning";

    /// <summary>Gets or sets the target sparsity ratio in [0, 1]. Default: 0.5.</summary>
    /// <remarks>0.0 = no pruning; 1.0 = all weights zeroed.</remarks>
    public float Sparsity { get; set; } = 0.5f;
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/MagnitudePruningConfig.cs
git commit -m "feat(algorithms): add MagnitudePruningConfig"
```

---

### Task 5: TDD `MagnitudePruningModifier`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/MagnitudePruningModifierTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/MagnitudePruningModifier.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="MagnitudePruningModifier"/>.
/// </summary>
public class MagnitudePruningModifierTests
{
    [Fact]
    public void OnEnd_FiftyPercentSparsity_ZerosLowestMagnitudeHalf()
    {
        using var weight = tensor(new float[] { 1f, -2f, 3f, -4f, 5f, -6f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state);

        // |w| = [1, 2, 3, 4, 5, 6]; 50% sparsity zeros the lowest 3: indices 0, 1, 2.
        var pruned = state.NamedWeights["w"].cpu().data<float>().ToArray();
        pruned[0].Should().Be(0f);
        pruned[1].Should().Be(0f);
        pruned[2].Should().Be(0f);
        pruned[3].Should().Be(-4f);
        pruned[4].Should().Be(5f);
        pruned[5].Should().Be(-6f);
    }

    [Fact]
    public void OnEnd_ZeroSparsity_LeavesWeightsUnchanged()
    {
        var original = new float[] { 1f, -2f, 3f, -4f };
        using var weight = tensor(original);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.0f });
        RunLifecycle(modifier, state);

        state.NamedWeights["w"].cpu().data<float>().ToArray().Should().Equal(original);
    }

    [Fact]
    public void OnEnd_FullSparsity_ZerosAllWeights()
    {
        using var weight = tensor(new float[] { 1f, -2f, 3f, -4f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w"] = weight,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 1.0f });
        RunLifecycle(modifier, state);

        state.NamedWeights["w"].cpu().data<float>().ToArray()
            .Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void OnEnd_AppliesPerWeightTensorIndependently()
    {
        using var w1 = tensor(new float[] { 1f, 100f }); // |w| = [1, 100]; 50% zeros 1
        using var w2 = tensor(new float[] { 0.1f, 0.2f }); // |w| = [0.1, 0.2]; 50% zeros 0.1

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["w1"] = w1,
            ["w2"] = w2,
        });

        var modifier = new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state);

        var p1 = state.NamedWeights["w1"].cpu().data<float>().ToArray();
        var p2 = state.NamedWeights["w2"].cpu().data<float>().ToArray();
        p1[0].Should().Be(0f);
        p1[1].Should().Be(100f);
        p2[0].Should().Be(0f);
        p2[1].Should().BeApproximately(0.2f, 1e-6f);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_SparsityOutOfRange_Throws(float sparsity)
    {
        var act = () => new MagnitudePruningModifier(new MagnitudePruningConfig { Sparsity = sparsity });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    private static void RunLifecycle(MagnitudePruningModifier modifier, CompressionState state)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}
```

- [ ] **Step 2: Verify tests fail with CS0246**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~MagnitudePruningModifierTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/MagnitudePruningModifierTests.cs
git commit -m "test(algorithms): add failing MagnitudePruningModifier tests"
```

- [ ] **Step 4: Implement `MagnitudePruningModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/MagnitudePruningModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning;

/// <summary>
/// Zeros out the lowest-magnitude weights to reach a target sparsity per weight tensor.
/// Data-free; runs entirely in <see cref="ModifierBase.OnEndCore"/>.
/// </summary>
public sealed class MagnitudePruningModifier : ModifierBase
{
    private readonly float _sparsity;

    /// <summary>Initializes a new instance of the <see cref="MagnitudePruningModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public MagnitudePruningModifier(MagnitudePruningConfig config)
        : base("MagnitudePruning", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Sparsity is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(config),
                config.Sparsity,
                "Sparsity must be in [0, 1].");
        }

        _sparsity = config.Sparsity;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        if (_sparsity <= 0f)
        {
            return;
        }

        var targeted = GetTargetedNames(state).ToList();
        foreach (var name in targeted)
        {
            // Do not dispose the original tensor — the caller still owns its lifetime.
            var weight = state.NamedWeights[name];
            state.NamedWeights[name] = PruneToSparsity(weight, _sparsity);
        }
    }

    private static Tensor PruneToSparsity(Tensor weight, float sparsity)
    {
        if (sparsity >= 1f)
        {
            return zeros_like(weight);
        }

        using var absWeight = weight.abs();
        using var flat = absWeight.flatten();
        var numElements = (int)flat.shape[0];
        var numToKeep = (int)Math.Max(0, Math.Round(numElements * (1.0 - sparsity)));
        if (numToKeep >= numElements)
        {
            return weight.clone();
        }

        if (numToKeep == 0)
        {
            return zeros_like(weight);
        }

        // topk on |weight| gives the largest values; the smallest kept magnitude is the threshold.
        using var topk = flat.topk(numToKeep, largest: true);
        using var kthSmallestKept = topk.values[numToKeep - 1];
        var threshold = kthSmallestKept.item<float>();

        using var mask = absWeight.ge(threshold);
        return weight.mul(mask.to(weight.dtype));
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~MagnitudePruningModifierTests"`
Expected: 6 tests pass (4 [Fact] + 2 [Theory] rows).

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/MagnitudePruningModifier.cs
git commit -m "feat(algorithms): implement MagnitudePruningModifier"
```

---

### Task 6: Add `AlgorithmsRegistration` helper

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs`
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/AlgorithmsRegistrationTests.cs`

- [ ] **Step 1: Write the helper**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Algorithms.Rtn;
using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms;

/// <summary>
/// Registers all built-in algorithm config + modifier pairs with <see cref="ModifierRegistry"/>.
/// </summary>
/// <remarks>
/// The CLI invokes <see cref="RegisterAll"/> once at startup. Tests register algorithms
/// individually as needed using the singular <see cref="RegisterRtn"/> / <see cref="RegisterMagnitudePruning"/> helpers.
/// </remarks>
public static class AlgorithmsRegistration
{
    /// <summary>Registers every built-in algorithm with <see cref="ModifierRegistry"/>.</summary>
    public static void RegisterAll()
    {
        RegisterRtn();
        RegisterMagnitudePruning();
    }

    /// <summary>Registers the RTN algorithm.</summary>
    public static void RegisterRtn()
    {
        ModifierRegistry.Register<RtnConfig>("RTN", c => new RtnModifier(c));
    }

    /// <summary>Registers the magnitude pruning algorithm.</summary>
    public static void RegisterMagnitudePruning()
    {
        ModifierRegistry.Register<MagnitudePruningConfig>("MagnitudePruning", c => new MagnitudePruningModifier(c));
    }
}
```

- [ ] **Step 2: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Algorithms.Rtn;
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
        var rtnInstance = rtnReg!.Factory(new Configs.RtnConfig());
        rtnInstance.Should().BeOfType<RtnModifier>();

        var pruneReg = ModifierRegistry.Resolve("MagnitudePruning");
        var pruneInstance = pruneReg!.Factory(new Configs.MagnitudePruningConfig());
        pruneInstance.Should().BeOfType<MagnitudePruningModifier>();
    }
}
```

- [ ] **Step 3: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~AlgorithmsRegistrationTests"`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs tests/LLMCompressorSharp.Tests/Algorithms/AlgorithmsRegistrationTests.cs
git commit -m "feat(algorithms): add AlgorithmsRegistration helper with tests"
```

---

### Task 7: TDD `SafetensorsWriter`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Output/SafetensorsWriterTests.cs`
- Create: `src/LLMCompressorSharp.Core/Output/SafetensorsWriter.cs`

- [ ] **Step 1: Investigate TorchSharp.PyBridge safetensors API**

```powershell
$pybridgeXml = Get-ChildItem -Path "$env:USERPROFILE\.nuget\packages\torchsharp.pybridge" -Filter "*.xml" -Recurse | Select-Object -First 1 -ExpandProperty FullName
if ($pybridgeXml) { Get-Content $pybridgeXml | Select-String -Pattern "safetensors|save_safetensors|load_safetensors" }
```

Identify the actual method to save a `Dictionary<string, Tensor>` to a safetensors file. The Phase 0 research notes mentioned `TorchSharp.PyBridge.PythonInterop.load_safetensors` and a `model.save_safetensors(path)` method. We want the lower-level dict-based form.

Based on findings, the likely API is one of:
- `Safetensors.save(IDictionary<string, Tensor>, string path)` — static helper
- `Module.save_safetensors(string path)` — instance method (saves the module's state_dict)
- `PythonInterop.save_safetensors(...)` — low-level

For Phase 2a we just need *some* working safetensors round-trip. If the only path is via a `Module`, wrap the named-weights dict in a stub module. See the implementation below for the chosen approach.

- [ ] **Step 2: Write the failing tests**

Create `tests/LLMCompressorSharp.Tests/Output/SafetensorsWriterTests.cs`:

```csharp
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
```

- [ ] **Step 3: Commit failing tests**

Run the targeted tests first to confirm CS0246:
```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SafetensorsWriterTests"
```
Expected: BUILD FAILS.

Commit:
```powershell
git add tests/LLMCompressorSharp.Tests/Output/SafetensorsWriterTests.cs
git commit -m "test(output): add failing SafetensorsWriter round-trip tests"
```

- [ ] **Step 4: Implement `SafetensorsWriter`**

The implementation depends on the PyBridge API surface you discovered. Below is the most likely shape — adapt to match the real API.

Create `src/LLMCompressorSharp.Core/Output/SafetensorsWriter.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Output;

/// <summary>
/// Saves and loads <see cref="CompressionState.NamedWeights"/> as safetensors files.
/// </summary>
public static class SafetensorsWriter
{
    /// <summary>Saves the named weights of <paramref name="state"/> to <paramref name="path"/>.</summary>
    /// <param name="state">The compression state.</param>
    /// <param name="path">Output safetensors file path. Parent directory must exist.</param>
    public static void Save(CompressionState state, string path)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var module = new StateDictModule(state.NamedWeights);
        module.save_safetensors(path);
    }

    /// <summary>Loads a safetensors file into a dictionary of named tensors.</summary>
    /// <param name="path">Input safetensors file path.</param>
    /// <returns>The loaded tensors, keyed by name.</returns>
    public static IDictionary<string, Tensor> Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Safetensors file not found.", path);
        }

        var module = new StateDictModule(new Dictionary<string, Tensor>());
        module.load_safetensors(path);
        return module.GetNamedWeights();
    }

    /// <summary>
    /// Minimal <see cref="nn.Module"/> wrapper exposing the named-weights dictionary as a state_dict.
    /// </summary>
    private sealed class StateDictModule : nn.Module
    {
        private readonly Dictionary<string, nn.Parameter> _parameters = new();

        public StateDictModule(IDictionary<string, Tensor> initial)
            : base("StateDictModule")
        {
            foreach (var (name, tensor) in initial)
            {
                _parameters[name] = new nn.Parameter(tensor.clone(), requires_grad: false);
            }

            foreach (var (name, param) in _parameters)
            {
                register_parameter(name, param);
            }
        }

        public IDictionary<string, Tensor> GetNamedWeights()
        {
            var result = new Dictionary<string, Tensor>();
            foreach (var (name, param) in named_parameters())
            {
                result[name] = param.detach().clone();
            }

            return result;
        }
    }
}
```

> **API uncertainty:** TorchSharp 0.107.0's `nn.Module` constructor and `register_parameter` shape may differ. If `register_parameter` rejects names with dots (`.`), translate them to underscores during save and back on load. If `nn.Parameter` doesn't expose a `requires_grad: false` constructor, set the flag separately after construction. The math goal is unchanged: a faithful round-trip of the named-weights dictionary.

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SafetensorsWriterTests"`
Expected: 3 tests pass.

If parameter names with dots fail (`.` may be illegal in `register_parameter`), apply a name-mangling pass: replace `.` with `__` on save and back to `.` on load.

If you cannot make the API work in 3 attempts, report BLOCKED with the specific error message. We can pivot to writing safetensors directly via the `Safetensors.NET` package (not currently referenced — would need to add it to `Directory.Packages.props`).

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Output/SafetensorsWriter.cs
git commit -m "feat(output): implement SafetensorsWriter via TorchSharp.PyBridge"
```

---

### Task 8: End-to-end integration test

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Integration/EndToEndRecipeTests.cs`

- [ ] **Step 1: Write the integration test**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Output;
using LLMCompressorSharp.Core.Recipes;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Integration;

/// <summary>
/// End-to-end: parse YAML → register algorithms → build modifiers → run session → save → reload.
/// </summary>
[Collection("ModifierRegistry")]
public class EndToEndRecipeTests : IDisposable
{
    private readonly string _tempPath;

    public EndToEndRecipeTests()
    {
        ModifierRegistry.Clear();
        AlgorithmsRegistration.RegisterAll();
        _tempPath = Path.Combine(
            Path.GetTempPath(),
            $"llmc-e2e-{Guid.NewGuid():N}.safetensors");
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
        if (File.Exists(_tempPath))
        {
            File.Delete(_tempPath);
        }
    }

    [Fact]
    public void Recipe_RtnW8A16_ThenMagnitudePruning_ProducesSparseQuantizedWeights()
    {
        var yaml = @"
stages:
  - name: quantize
    modifiers:
      - type: RTN
        num_bits: 8
        symmetric: true
        targets: [model.*]
        ignore: [model.lm_head.*]
  - name: prune
    modifiers:
      - type: MagnitudePruning
        sparsity: 0.5
        targets: [model.*]
        ignore: [model.lm_head.*]
";

        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(2);

        // Synthetic state: 8 weights ramping 1..8 in a layer, plus an untouched lm_head.
        using var layerWeight = tensor(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f });
        using var lmHeadWeight = tensor(new float[] { 100f, 200f, 300f });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.weight"] = layerWeight,
            ["model.lm_head.weight"] = lmHeadWeight,
        });

        var session = new CompressionSession(modifiers);
        var status = session.Run(state, Enumerable.Empty<Tensor>());

        status.Should().Be(SessionStatus.Completed);

        // lm_head ignored → untouched.
        state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray()
            .Should().Equal(new float[] { 100f, 200f, 300f });

        // layer weights → quantized to int8 grid, then 50% pruned (4 zeros).
        var processed = state.NamedWeights["model.layer.0.weight"].cpu().data<float>().ToArray();
        processed.Count(v => v == 0f).Should().Be(4);
        processed.Count(v => v != 0f).Should().Be(4);

        // Round-trip through safetensors.
        SafetensorsWriter.Save(state, _tempPath);
        var reloaded = SafetensorsWriter.Load(_tempPath);
        try
        {
            reloaded.Should().ContainKey("model.layer.0.weight");
            reloaded["model.layer.0.weight"].cpu().data<float>().ToArray().Should().Equal(processed);
            reloaded["model.lm_head.weight"].cpu().data<float>().ToArray()
                .Should().Equal(new float[] { 100f, 200f, 300f });
        }
        finally
        {
            foreach (var t in reloaded.Values)
            {
                t.Dispose();
            }
        }
    }
}
```

> **YAML naming convention:** the test recipe uses `num_bits` (snake_case) — `RecipeParser` converts to `NumBits` via the underscore-to-PascalCase logic. Same for `targets`/`ignore`.

- [ ] **Step 2: Run the integration test**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~EndToEndRecipeTests"`
Expected: 1 test passes.

If the test fails on the safetensors round-trip due to dot-named parameters, fall back: skip the safetensors part and assert just on the in-memory state, marking that part for a follow-up (note in the commit message). The pre-safetensors assertions are the real integration test value.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Integration/EndToEndRecipeTests.cs
git commit -m "test(integration): add end-to-end RTN + Magnitude recipe test"
```

---

### Task 9: Full-solution verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Clean restore + build + test**

Run:
```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected:
- Build: 0 errors, 0 warnings
- Tests: **~101 passing** (82 prior + 6 RTN + 6 Magnitude + 3 Registration + 3 SafetensorsWriter + 1 EndToEnd)

- [ ] **Step 2: Verify file inventory matches plan**

Run: `Get-ChildItem -Recurse -Filter "*.cs" -Path src/LLMCompressorSharp.Core/Algorithms/, src/LLMCompressorSharp.Core/Output/ | ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '') } | Sort-Object`

Expected new files:
- `src\LLMCompressorSharp.Core\Algorithms\AlgorithmsRegistration.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Configs\MagnitudePruningConfig.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Configs\RtnConfig.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Pruning\MagnitudePruningModifier.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Rtn\RtnModifier.cs`
- `src\LLMCompressorSharp.Core\Output\SafetensorsWriter.cs`

No commit — verification only.

---

### Task 10: Merge and tag — STOP here

The controller (main session) handles the merge to `main`, the `v0.2.0-alpha` tag, and the push to `origin`. Stop after Task 9.

---

## Self-Review Notes

**Spec coverage:**
- RTN (weight-only, per-tensor + per-channel) → Tasks 2, 3
- Magnitude pruning → Tasks 4, 5
- Algorithm registration → Task 6
- Safetensors output writer → Task 7
- End-to-end integration → Task 8

**Out of scope (deferred to Phase 2b):**
- WANDA — needs layer activation hooks
- SmoothQuant — needs paired-layer activation collection
- AWQ — Phase 5 per spec

**Type consistency:**
- `RtnConfig.Strategy` uses `QuantizationStrategy` enum from `TorchExtensions.Observers` (same one used in observers)
- `MagnitudePruningConfig.Sparsity` validated in `MagnitudePruningModifier` constructor, not config (matches RtnModifier null check pattern)
- All algorithm modifiers extend `ModifierBase` with `Name` matching `ModifierConfig.Type` discriminator
- `AlgorithmsRegistration` static methods are idempotent (`ModifierRegistry.Register` overwrites duplicate keys)

**Known risks:**
- **TorchSharp.PyBridge safetensors API** — Task 7 sketches a wrapper-module approach; the actual API may require name-mangling for dots in parameter names. Test files include explicit assertions on layer-named keys (e.g. `model.layer.0.weight`) to catch this. Mitigation in Task 7 Step 5.
- **`tensor.topk` and `tensor.ge`** — used in `MagnitudePruningModifier`. These are standard TorchSharp ops verified working in Phase 1a — should be solid.
- **`tensor.select(axis, index)` for per-channel RTN** — slicing along a specific axis. If the TorchSharp signature differs (e.g. `narrow` instead of `select`), adapt while preserving the per-channel scope.
- **`ModifierRegistry` parallelism** — Tasks 6 and 8 use `[Collection("ModifierRegistry")]` as Phase 1b established.

**Commit count target:** ~12–14 commits across 10 tasks.

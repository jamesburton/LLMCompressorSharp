# Phase 2b: Activation-Dependent Algorithms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `WandaModifier` (pruning by `|w| × activation_norm` saliency) and `SmoothQuantModifier` (channel-wise activation/weight rebalancing). Extend `CompressionState` with a `LayerActivations` dictionary so tests can feed synthetic activations directly; Phase 3 will plug in hook-driven population without breaking the modifier API.

**Architecture:**

```
CompressionState (extended)
   ├── NamedWeights (existing)
   └── LayerActivations (NEW)  ← Dict<string, Tensor>?; populated by tests or Phase 3 hooks
                                  Convention: key matches the weight name without ".weight" suffix.

WandaModifier
   ├── On Initialize: clear running per-channel norms
   ├── On Batch: for each targeted weight, accumulate sum of |x|² across activation samples
   └── On End: compute norm = sqrt(accumulated); saliency = |W| · norm; prune lowest-saliency

SmoothQuantModifier
   ├── On Initialize: clear running per-channel activation max
   ├── On Batch: for each mapping, update running max from activation
   └── On End: compute scale s[i] = max(|X|)^α / max(|W|)^(1−α);
                 smooth_weight /= s ;  balance_weight[:, i] *= s[i]
                 (math identity Y = (X / s) · (s · W) preserved)
```

**Tech Stack:** TorchSharp 0.107.0, plus everything Phase 1a + 1b + 2a established.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (algorithm list — WANDA + SmoothQuant)

---

## File Structure

```
src/LLMCompressorSharp.Core/
├── Compression/
│   └── CompressionState.cs                    ← MODIFY: add LayerActivations property
├── Algorithms/
│   ├── Configs/
│   │   ├── WandaConfig.cs                     ← NEW
│   │   ├── SmoothQuantConfig.cs               ← NEW
│   │   └── SmoothQuantMapping.cs              ← NEW (record)
│   ├── Pruning/
│   │   └── WandaModifier.cs                   ← NEW
│   ├── SmoothQuant/
│   │   └── SmoothQuantModifier.cs             ← NEW
│   └── AlgorithmsRegistration.cs              ← MODIFY: also register WANDA + SmoothQuant

tests/LLMCompressorSharp.Tests/
├── Algorithms/
│   ├── WandaModifierTests.cs                  ← NEW
│   ├── SmoothQuantModifierTests.cs            ← NEW
│   └── AlgorithmsRegistrationTests.cs         ← MODIFY: assert WANDA + SmoothQuant registered too
├── Integration/
│   └── EndToEndRecipeTests.cs                 ← MODIFY: add WANDA + SmoothQuant integration test
```

**Responsibility per file:**
- `CompressionState.LayerActivations` — new optional `IDictionary<string, Tensor>?`. Modifiers read by stripping `.weight` suffix from the weight name; if not present, throws an actionable error.
- `WandaConfig` — `Sparsity` (float, [0,1]), inherited `Targets`/`Ignore`. Optional `ActivationKeySuffix` (default empty — strip `.weight` only).
- `WandaModifier` — accumulates per-input-channel L2 norm across calibration batches, then prunes lowest-saliency weights at OnEnd. Per-input-channel only (saliency along the last weight axis).
- `SmoothQuantMapping` — record `(string SmoothWeight, string BalanceWeight, string ActivationKey)`. The activation has shape `[..., hidden]`; smooth_weight is 1-D `[hidden]` (LayerNorm-gain style); balance_weight is 2-D `[out, hidden]`.
- `SmoothQuantConfig` — `SmoothingStrength` (alpha, 0..1, default 0.5), `Mappings` list. `Ignore` inherited but typically unused; targeting is mapping-driven.
- `SmoothQuantModifier` — for each mapping, collect per-channel activation max across batches; at OnEnd compute scale and apply. Identity Y = (X/s)·(s·W) preserved within FP32 tolerance.
- `AlgorithmsRegistration.RegisterWanda/RegisterSmoothQuant` + `RegisterAll` updated.

**Out of scope (deferred to Phase 3):**
- Hook-driven activation collection from a `Module`
- Paired-Linear smooth (the smooth_weight is 1-D in Phase 2b; Phase 3 generalises to 2-D Linears)
- GPTQ, SparseGPT (Phase 4)
- AWQ (Phase 5)

---

## Prerequisites & Conventions

- Phase 2a is merged. Tag `v0.2.0-alpha` on `main`. 101 tests passing.
- Branch off `main` as `feature/2b-activation-algorithms`.
- xunit collection `ModifierRegistry` still applies to tests that mutate the registry.
- StyleCop patterns from prior phases (SA1402, SA1515, SA1642, SA1201, SA1117, SA1500 multi-line arrays, etc.).

---

### Task 1: Branch + baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
Expected: clean tree, HEAD `eae5b85`, tags through `v0.2.0-alpha`.

- [ ] **Step 2:** `git checkout -b feature/2b-activation-algorithms`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
Expected: 101 passing.

No commit.

---

### Task 2: Extend `CompressionState` with `LayerActivations`

**Files:**
- Modify: `src/LLMCompressorSharp.Core/Compression/CompressionState.cs`

- [ ] **Step 1: Add the property**

Use the Edit tool to add a new property between `CurrentBatchIndex` and `RngSeed`. The new property:

```csharp
    /// <summary>
    /// Gets or sets per-layer activation tensors collected during calibration, or
    /// <see langword="null"/> if no activations are routed to the session.
    /// </summary>
    /// <remarks>
    /// Phase 2b modifiers (WANDA, SmoothQuant) read activations from this dictionary using
    /// the weight name stripped of its <c>.weight</c> suffix as the key. Phase 3 will populate
    /// this dictionary automatically via forward hooks on a TorchSharp <c>Module</c>; until
    /// then, tests populate it directly.
    /// </remarks>
    public IDictionary<string, Tensor>? LayerActivations { get; set; }
```

Place it directly after the `CurrentBatchIndex` property definition. Preserve all other code.

- [ ] **Step 2: Build the Core project**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Confirm the prior test suite still passes**

Run: `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
Expected: still 101 tests passing (no change — we only added an optional property).

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Compression/CompressionState.cs
git commit -m "feat(core): add LayerActivations to CompressionState"
```

---

### Task 3: Add `WandaConfig`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/WandaConfig.cs`

- [ ] **Step 1: Write the config**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>WandaModifier</c> — pruning by <c>|w| × ||x||₂</c> saliency.
/// </summary>
/// <remarks>
/// The activation key for each targeted weight is the weight name with the <c>.weight</c> suffix
/// removed (e.g. weight <c>model.layer.0.q_proj.weight</c> → activation <c>model.layer.0.q_proj</c>).
/// </remarks>
public sealed class WandaConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "WANDA";

    /// <summary>Gets or sets the target sparsity ratio in [0, 1]. Default: 0.5.</summary>
    public float Sparsity { get; set; } = 0.5f;
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/WandaConfig.cs
git commit -m "feat(algorithms): add WandaConfig"
```

---

### Task 4: TDD `WandaModifier`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/WandaModifierTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/WandaModifier.cs`

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
/// Tests for <see cref="WandaModifier"/> — pruning by |w| × ||x||₂ saliency.
/// </summary>
public class WandaModifierTests
{
    [Fact]
    public void Name_IsWanda()
    {
        var modifier = new WandaModifier(new WandaConfig());
        modifier.Name.Should().Be("WANDA");
    }

    [Fact]
    public void OnEnd_SaliencyEqualsWeightTimesActivationNorm_PrunesLowestScores()
    {
        // Weight shape [out=2, in=3]. Activation X shape [batch=2, in=3].
        // Activation: row 0 = [1, 2, 3]; row 1 = [1, 2, 3]; norm per input channel = sqrt(2)*|x|, but
        // since both batches are identical, norm[i] = sqrt(2*x_i²) = |x_i|*sqrt(2).
        // So norm = [sqrt(2), 2*sqrt(2), 3*sqrt(2)].
        // Weight rows: [1, 1, 1] and [1, 1, 1].
        // Saliency[r, c] = 1 * norm[c] for both rows = [√2, 2√2, 3√2] uniformly.
        // With 50% sparsity (3 of 6 zeroed), the lowest-saliency column (index 0, norm=√2) is pruned in both rows
        // plus one more from column 1 (tie-breaking — topk is deterministic but tie-breaking is unspecified;
        // assert weaker: column 2 (largest norm) MUST survive in both rows, column 0 must be pruned in both).
        using var weight = ones(2, 3);
        using var activation = tensor(new float[,]
        {
            { 1f, 2f, 3f },
            { 1f, 2f, 3f },
        });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["layer"] = activation,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state, batchCount: 1);

        var pruned = state.NamedWeights["layer.weight"].cpu();
        // Column 2 (largest activation norm) must survive in both rows
        pruned[0, 2].item<float>().Should().Be(1f);
        pruned[1, 2].item<float>().Should().Be(1f);
        // Column 0 (smallest activation norm) must be pruned in both rows
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[1, 0].item<float>().Should().Be(0f);
    }

    [Fact]
    public void OnEnd_LargeWeightWithSmallActivation_StillPrunedWhenSaliencyLow()
    {
        // Weight row [100, 1] — column 0 large weight; column 1 small weight.
        // Activation [0.01, 100] — column 0 tiny norm, column 1 huge norm.
        // Saliency: 100 * 0.01 = 1 (col 0); 1 * 100 = 100 (col 1).
        // 50% sparsity prunes the lower saliency → col 0 (large weight!) is zeroed; col 1 (small weight!) survives.
        using var weight = tensor(new float[,] { { 100f, 1f } });
        using var activation = tensor(new float[,] { { 0.01f, 100f } });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["proj.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = activation,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        RunLifecycle(modifier, state, batchCount: 1);

        var pruned = state.NamedWeights["proj.weight"].cpu();
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[0, 1].item<float>().Should().Be(1f);
    }

    [Fact]
    public void OnEnd_NoActivationForTargetedLayer_Throws()
    {
        using var weight = ones(2, 3);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>(),  // empty
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*activation*layer*");
    }

    [Fact]
    public void OnEnd_LayerActivationsNullOnState_Throws()
    {
        using var weight = ones(2, 3);
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.weight"] = weight,
        });
        // LayerActivations stays null

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void OnEnd_AccumulatesActivationNormsAcrossBatches()
    {
        // Two batches with same activation pattern; cumulative norm should equal single-batch * sqrt(2).
        // We can't test the norm directly, but we can test that the pruning outcome is consistent
        // (same columns survive whether we feed 1 or 2 identical batches).
        using var weight = tensor(new float[,] { { 1f, 1f } });
        using var act1 = tensor(new float[,] { { 1f, 5f } });
        using var act2 = tensor(new float[,] { { 1f, 5f } });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["p.weight"] = weight,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["p"] = act1,
            },
        };

        var modifier = new WandaModifier(new WandaConfig { Sparsity = 0.5f });
        modifier.Initialize(state);
        modifier.OnStart(state);

        // First batch
        state.CurrentBatch = act1;
        modifier.OnBatch(state);

        // Second batch: replace the activation
        state.LayerActivations!["p"] = act2;
        state.CurrentBatch = act2;
        modifier.OnBatch(state);

        modifier.OnEnd(state);
        modifier.Finalize(state);

        var pruned = state.NamedWeights["p.weight"].cpu();
        // Column 1 has larger norm contribution, so it survives.
        pruned[0, 0].item<float>().Should().Be(0f);
        pruned[0, 1].item<float>().Should().Be(1f);
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new WandaModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_SparsityOutOfRange_Throws(float sparsity)
    {
        var act = () => new WandaModifier(new WandaConfig { Sparsity = sparsity });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    private static void RunLifecycle(WandaModifier modifier, CompressionState state, int batchCount)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        for (var i = 0; i < batchCount; i++)
        {
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}
```

- [ ] **Step 2: Verify tests fail with CS0246**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~WandaModifierTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/WandaModifierTests.cs
git commit -m "test(algorithms): add failing WandaModifier tests"
```

- [ ] **Step 4: Implement `WandaModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/WandaModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning;

/// <summary>
/// Pruning by <c>|w| × ||x||₂</c> saliency. Activation L2 norms are accumulated per input channel
/// across calibration batches; at <see cref="ModifierBase.OnEndCore"/> the lowest-saliency weights
/// are zeroed to reach the configured sparsity.
/// </summary>
public sealed class WandaModifier : ModifierBase
{
    private const string WeightSuffix = ".weight";

    private readonly float _sparsity;
    private readonly Dictionary<string, double[]> _runningSumSquares = new();

    /// <summary>Initializes a new instance of the <see cref="WandaModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public WandaModifier(WandaConfig config)
        : base("WANDA", config?.Targets, config?.Ignore)
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
        _runningSumSquares.Clear();
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        EnsureActivations(state);
        var activations = state.LayerActivations!;

        foreach (var name in GetTargetedNames(state))
        {
            var activationKey = StripWeightSuffix(name);
            if (!activations.TryGetValue(activationKey, out var x))
            {
                continue;
            }

            // Sum of x² along all axes except the last (input feature dim).
            var rank = x.shape.Length;
            var reduceDims = new long[rank - 1];
            for (var i = 0; i < rank - 1; i++)
            {
                reduceDims[i] = i;
            }

            using var squared = x.pow(2);
            using var reduced = reduceDims.Length == 0
                ? squared.clone()
                : squared.sum(reduceDims, keepdim: false);
            var batchContrib = reduced.cpu().data<float>().ToArray();

            if (!_runningSumSquares.TryGetValue(name, out var accum))
            {
                accum = new double[batchContrib.Length];
                _runningSumSquares[name] = accum;
            }

            if (accum.Length != batchContrib.Length)
            {
                throw new InvalidOperationException(
                    $"Activation feature dimension changed for '{activationKey}': "
                    + $"expected {accum.Length}, got {batchContrib.Length}.");
            }

            for (var i = 0; i < accum.Length; i++)
            {
                accum[i] += batchContrib[i];
            }
        }
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        EnsureActivations(state);
        var activations = state.LayerActivations!;

        foreach (var name in GetTargetedNames(state))
        {
            var activationKey = StripWeightSuffix(name);
            if (!_runningSumSquares.TryGetValue(name, out var sumSq))
            {
                // No batch was seen yet; if activation is available, treat the current activation as a single batch.
                if (activations.TryGetValue(activationKey, out var x))
                {
                    using var sq = x.pow(2);
                    var rank = x.shape.Length;
                    Tensor reduced;
                    if (rank == 1)
                    {
                        reduced = sq.clone();
                    }
                    else
                    {
                        var dims = new long[rank - 1];
                        for (var i = 0; i < rank - 1; i++)
                        {
                            dims[i] = i;
                        }

                        reduced = sq.sum(dims, keepdim: false);
                    }

                    using (reduced)
                    {
                        var batchContrib = reduced.cpu().data<float>().ToArray();
                        sumSq = new double[batchContrib.Length];
                        for (var i = 0; i < batchContrib.Length; i++)
                        {
                            sumSq[i] = batchContrib[i];
                        }
                    }
                }
                else
                {
                    throw new InvalidOperationException(
                        $"WANDA: no activation found for layer '{activationKey}' "
                        + $"(weight '{name}'). Populate CompressionState.LayerActivations.");
                }
            }

            var norms = new float[sumSq.Length];
            for (var i = 0; i < sumSq.Length; i++)
            {
                norms[i] = (float)Math.Sqrt(sumSq[i]);
            }

            var weight = state.NamedWeights[name];
            state.NamedWeights[name] = ApplyWandaPruning(weight, norms, _sparsity);
        }
    }

    private void EnsureActivations(CompressionState state)
    {
        if (state.LayerActivations is null)
        {
            throw new InvalidOperationException(
                "WANDA: CompressionState.LayerActivations is null. WANDA requires per-layer activations.");
        }
    }

    private static string StripWeightSuffix(string name)
    {
        return name.EndsWith(WeightSuffix, StringComparison.Ordinal)
            ? name[..^WeightSuffix.Length]
            : name;
    }

    private static Tensor ApplyWandaPruning(Tensor weight, float[] inputNorms, float sparsity)
    {
        if (sparsity <= 0f)
        {
            return weight.clone();
        }

        if (sparsity >= 1f)
        {
            return zeros_like(weight);
        }

        var inFeatures = (int)weight.shape[^1];
        if (inputNorms.Length != inFeatures)
        {
            throw new InvalidOperationException(
                $"WANDA: input norms length {inputNorms.Length} does not match weight last dim {inFeatures}.");
        }

        using var normTensor = tensor(inputNorms).to(weight.dtype);

        // Broadcast norm across all axes except the last.
        var reshapeDims = new long[weight.shape.Length];
        for (var i = 0; i < reshapeDims.Length - 1; i++)
        {
            reshapeDims[i] = 1L;
        }

        reshapeDims[^1] = inFeatures;

        using var normBroadcast = normTensor.reshape(reshapeDims);
        using var absWeight = weight.abs();
        using var saliency = absWeight.mul(normBroadcast);
        using var flat = saliency.flatten();

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

        var (topkValues, topkIndices) = flat.topk(numToKeep, largest: true);
        float threshold;
        using (topkValues)
        using (topkIndices)
        using (var kth = topkValues[numToKeep - 1])
        {
            threshold = kth.item<float>();
        }

        using var mask = saliency.ge(threshold);
        return weight.mul(mask.to(weight.dtype));
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~WandaModifierTests"`
Expected: 8 tests pass (6 [Fact] + 2 [Theory] rows).

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/WandaModifier.cs
git commit -m "feat(algorithms): implement WandaModifier"
```

---

### Task 5: Add `SmoothQuantMapping` and `SmoothQuantConfig`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/SmoothQuantMapping.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/SmoothQuantConfig.cs`

- [ ] **Step 1: Create `SmoothQuantMapping.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// A SmoothQuant mapping: the smooth-layer weight (1-D, LayerNorm-gain style) whose output
/// channels are scaled down by <c>s</c>, the balance-layer weight (2-D, Linear-style) whose
/// input channels are scaled up by <c>s</c>, and the activation tensor key that feeds the
/// balance layer.
/// </summary>
/// <param name="SmoothWeight">Name of the smooth-layer weight in <c>state.NamedWeights</c>. Must be 1-D.</param>
/// <param name="BalanceWeight">Name of the balance-layer weight in <c>state.NamedWeights</c>. Must be 2-D <c>[out, hidden]</c>.</param>
/// <param name="ActivationKey">Key in <c>state.LayerActivations</c> for the activation feeding <see cref="BalanceWeight"/>.</param>
public sealed record SmoothQuantMapping(string SmoothWeight, string BalanceWeight, string ActivationKey);
```

- [ ] **Step 2: Create `SmoothQuantConfig.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>SmoothQuantModifier</c> — channel-wise activation/weight rebalancing.
/// </summary>
public sealed class SmoothQuantConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "SmoothQuant";

    /// <summary>Gets or sets alpha — the smoothing strength in (0, 1). Default: 0.5.</summary>
    /// <remarks>
    /// <c>alpha = 0.5</c> migrates difficulty equally between activations and weights.
    /// <c>alpha → 1</c> migrates more difficulty into weights; <c>alpha → 0</c> keeps it in activations.
    /// </remarks>
    public float SmoothingStrength { get; set; } = 0.5f;

    /// <summary>Gets or sets the smooth ↔ balance mappings to process.</summary>
    public IList<SmoothQuantMapping> Mappings { get; set; } = new List<SmoothQuantMapping>();
}
```

- [ ] **Step 3: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/SmoothQuantMapping.cs src/LLMCompressorSharp.Core/Algorithms/Configs/SmoothQuantConfig.cs
git commit -m "feat(algorithms): add SmoothQuant config types"
```

---

### Task 6: TDD `SmoothQuantModifier`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/SmoothQuantModifierTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/SmoothQuant/SmoothQuantModifier.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms;

/// <summary>
/// Tests for <see cref="SmoothQuantModifier"/>.
/// </summary>
public class SmoothQuantModifierTests
{
    [Fact]
    public void Name_IsSmoothQuant()
    {
        var modifier = new SmoothQuantModifier(new SmoothQuantConfig());
        modifier.Name.Should().Be("SmoothQuant");
    }

    [Fact]
    public void OnEnd_AppliesScaleSuchThatOutputIsPreserved()
    {
        // Setup:
        //   smooth weight (1-D): g = [2, 4]
        //   balance weight (2-D): W = [[1, 1], [1, 1]]  (shape [out=2, hidden=2])
        //   activation X (per channel max): [|x[:, 0]|max = 1, |x[:, 1]|max = 4]
        //   alpha = 0.5
        // Scale s[i] = max(|X|)^0.5 / max(|W|)^0.5 ; max(|W|[:, i]) per input column = [1, 1]
        // s = [1^0.5 / 1^0.5, 4^0.5 / 1^0.5] = [1, 2]
        // After smoothing:
        //   g_new = g / s = [2/1, 4/2] = [2, 2]
        //   W_new[:, i] = W[:, i] * s[i] → [[1, 2], [1, 2]]
        // Verify: (X / s) · (W * s) == X · W for any X.
        using var smoothW = tensor(new float[] { 2f, 4f });
        using var balanceW = tensor(new float[,] { { 1f, 1f }, { 1f, 1f } });
        using var activation = tensor(new float[,] { { 1f, 4f }, { -0.5f, -2f } });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["norm.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = activation,
            },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            SmoothingStrength = 0.5f,
            Mappings =
            {
                new SmoothQuantMapping("norm.weight", "proj.weight", "proj"),
            },
        });

        RunLifecycle(modifier, state, batchCount: 1);

        var gNew = state.NamedWeights["norm.weight"].cpu().data<float>().ToArray();
        var wNew = state.NamedWeights["proj.weight"].cpu();

        // Per the math above: g_new ≈ [2, 2]
        gNew[0].Should().BeApproximately(2f, 1e-4f);
        gNew[1].Should().BeApproximately(2f, 1e-4f);

        // W_new[:, 0] ≈ 1 (s[0] = 1), W_new[:, 1] ≈ 2 (s[1] = 2)
        wNew[0, 0].item<float>().Should().BeApproximately(1f, 1e-4f);
        wNew[0, 1].item<float>().Should().BeApproximately(2f, 1e-4f);
        wNew[1, 0].item<float>().Should().BeApproximately(1f, 1e-4f);
        wNew[1, 1].item<float>().Should().BeApproximately(2f, 1e-4f);
    }

    [Fact]
    public void OnEnd_PreservesMathematicalIdentity()
    {
        // For arbitrary smooth gain g, weight W, and activation X:
        //   y_original = (X * g) @ W^T
        //   y_smoothed = (X * g_new) @ W_new^T   should match within FP32 tolerance
        // where g_new = g / s, W_new[:, i] = W[:, i] * s[i], s = act_max^α / W_max_per_col^(1-α).
        using var g = tensor(new float[] { 1f, 2f, 3f });
        using var W = tensor(new float[,]
        {
            { 0.5f, 1f, -1f },
            { 2f, -0.5f, 0.25f },
        });
        using var X = tensor(new float[,]
        {
            { 0.1f, 0.5f, -1f },
            { 0.2f, -0.3f, 0.7f },
        });

        // Reference output: y = (X * g) @ W^T   (broadcasting g over batch dim)
        using var Xg = X.mul(g);
        using var yRef = Xg.matmul(W.t());
        var yRefArr = yRef.cpu().data<float>().ToArray();

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = g,
            ["proj.weight"] = W,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["proj"] = Xg,
            },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            SmoothingStrength = 0.5f,
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });
        RunLifecycle(modifier, state, batchCount: 1);

        using var gNew = state.NamedWeights["g.weight"];
        using var WNew = state.NamedWeights["proj.weight"];
        using var XgNew = X.mul(gNew);
        using var ySmoothed = XgNew.matmul(WNew.t());
        var ySmoothedArr = ySmoothed.cpu().data<float>().ToArray();

        for (var i = 0; i < yRefArr.Length; i++)
        {
            ySmoothedArr[i].Should().BeApproximately(yRefArr[i], 1e-4f);
        }
    }

    [Fact]
    public void OnEnd_MissingMappingActivation_Throws()
    {
        using var smoothW = tensor(new float[] { 1f, 1f });
        using var balanceW = tensor(new float[,] { { 1f, 1f } });

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>(),
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });

        var act = () => RunLifecycle(modifier, state, batchCount: 0);
        act.Should().Throw<InvalidOperationException>().WithMessage("*activation*proj*");
    }

    [Fact]
    public void OnEnd_SmoothWeightNot1D_Throws()
    {
        using var smoothW = tensor(new float[,] { { 1f, 1f } });  // 2-D — invalid
        using var balanceW = tensor(new float[,] { { 1f, 1f } });
        using var activation = ones(2, 2);

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["g.weight"] = smoothW,
            ["proj.weight"] = balanceW,
        })
        {
            LayerActivations = new Dictionary<string, Tensor> { ["proj"] = activation },
        };

        var modifier = new SmoothQuantModifier(new SmoothQuantConfig
        {
            Mappings = { new SmoothQuantMapping("g.weight", "proj.weight", "proj") },
        });
        var act = () => RunLifecycle(modifier, state, batchCount: 1);
        act.Should().Throw<InvalidOperationException>().WithMessage("*1-D*");
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Constructor_AlphaOutOfRange_Throws(float alpha)
    {
        var act = () => new SmoothQuantModifier(new SmoothQuantConfig { SmoothingStrength = alpha });
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Constructor_NullConfig_Throws()
    {
        var act = () => new SmoothQuantModifier(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    private static void RunLifecycle(SmoothQuantModifier modifier, CompressionState state, int batchCount)
    {
        modifier.Initialize(state);
        modifier.OnStart(state);
        for (var i = 0; i < batchCount; i++)
        {
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);
    }
}
```

- [ ] **Step 2: Verify tests fail with CS0246**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SmoothQuantModifierTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/SmoothQuantModifierTests.cs
git commit -m "test(algorithms): add failing SmoothQuantModifier tests"
```

- [ ] **Step 4: Implement `SmoothQuantModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/SmoothQuant/SmoothQuantModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.SmoothQuant;

/// <summary>
/// Channel-wise activation/weight rebalancing: migrates per-channel quantization difficulty from
/// activations into weights. The mathematical identity <c>Y = (X / s) · (s · W)</c> is preserved.
/// </summary>
public sealed class SmoothQuantModifier : ModifierBase
{
    private readonly SmoothQuantConfig _config;
    private readonly Dictionary<string, float[]> _runningActMax = new();

    /// <summary>Initializes a new instance of the <see cref="SmoothQuantModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public SmoothQuantModifier(SmoothQuantConfig config)
        : base("SmoothQuant", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.SmoothingStrength is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(config),
                config.SmoothingStrength,
                "SmoothingStrength (alpha) must be in [0, 1].");
        }

        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        _runningActMax.Clear();
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        if (state.LayerActivations is null)
        {
            return;
        }

        foreach (var mapping in _config.Mappings)
        {
            if (!state.LayerActivations.TryGetValue(mapping.ActivationKey, out var x))
            {
                continue;
            }

            UpdateActivationMax(mapping.ActivationKey, x);
        }
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        foreach (var mapping in _config.Mappings)
        {
            ApplyMapping(state, mapping);
        }
    }

    private void UpdateActivationMax(string key, Tensor x)
    {
        var rank = x.shape.Length;
        Tensor reduced;
        if (rank == 1)
        {
            reduced = x.abs();
        }
        else
        {
            var dims = new long[rank - 1];
            for (var i = 0; i < rank - 1; i++)
            {
                dims[i] = i;
            }

            using var abs = x.abs();
            reduced = abs.amax(dims, keepdim: false);
        }

        using (reduced)
        {
            var batch = reduced.cpu().data<float>().ToArray();
            if (!_runningActMax.TryGetValue(key, out var existing))
            {
                _runningActMax[key] = batch;
            }
            else if (existing.Length != batch.Length)
            {
                throw new InvalidOperationException(
                    $"SmoothQuant: activation feature dim changed for '{key}': "
                    + $"expected {existing.Length}, got {batch.Length}.");
            }
            else
            {
                for (var i = 0; i < existing.Length; i++)
                {
                    if (batch[i] > existing[i])
                    {
                        existing[i] = batch[i];
                    }
                }
            }
        }
    }

    private void ApplyMapping(CompressionState state, SmoothQuantMapping mapping)
    {
        if (state.LayerActivations is null
            || !state.LayerActivations.TryGetValue(mapping.ActivationKey, out var x))
        {
            // Use accumulated max if available; otherwise fall back to using current state.
            if (!_runningActMax.TryGetValue(mapping.ActivationKey, out _))
            {
                throw new InvalidOperationException(
                    $"SmoothQuant: no activation collected for '{mapping.ActivationKey}'. "
                    + "Populate CompressionState.LayerActivations.");
            }
        }
        else if (!_runningActMax.ContainsKey(mapping.ActivationKey))
        {
            UpdateActivationMax(mapping.ActivationKey, x);
        }

        var actMax = _runningActMax[mapping.ActivationKey];

        var smoothW = state.NamedWeights[mapping.SmoothWeight];
        if (smoothW.shape.Length != 1)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: smooth weight '{mapping.SmoothWeight}' must be 1-D (got rank {smoothW.shape.Length}).");
        }

        var balanceW = state.NamedWeights[mapping.BalanceWeight];
        if (balanceW.shape.Length != 2)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: balance weight '{mapping.BalanceWeight}' must be 2-D (got rank {balanceW.shape.Length}).");
        }

        var hidden = (int)smoothW.shape[0];
        if (actMax.Length != hidden)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: activation hidden dim {actMax.Length} does not match smooth weight {hidden}.");
        }

        if (balanceW.shape[1] != hidden)
        {
            throw new InvalidOperationException(
                $"SmoothQuant: balance weight last dim {balanceW.shape[1]} does not match hidden {hidden}.");
        }

        // Per-input-channel max of |balance weight|, reducing axis 0 (output dim).
        using var absBalance = balanceW.abs();
        using var weightMax = absBalance.amax(new long[] { 0L }, keepdim: false);
        var weightMaxArr = weightMax.cpu().data<float>().ToArray();

        var alpha = _config.SmoothingStrength;
        var s = new float[hidden];
        for (var i = 0; i < hidden; i++)
        {
            var numerator = MathF.Pow(MathF.Max(actMax[i], 1e-5f), alpha);
            var denominator = MathF.Pow(MathF.Max(weightMaxArr[i], 1e-5f), 1f - alpha);
            s[i] = MathF.Max(numerator / denominator, 1e-5f);
        }

        using var scale = tensor(s).to(smoothW.dtype);
        using var newSmooth = smoothW.div(scale);
        state.NamedWeights[mapping.SmoothWeight] = newSmooth.detach().clone();

        using var newBalance = balanceW.mul(scale);
        state.NamedWeights[mapping.BalanceWeight] = newBalance.detach().clone();
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SmoothQuantModifierTests"`
Expected: 8 tests pass (6 [Fact] + 2 [Theory] rows).

If `PreservesMathematicalIdentity` fails, the math has subtle indexing — recompute by hand for a 2×3 W, 2×3 X, 3-D g case to verify expected outputs. The key invariant is `(X·g) · W^T == (X · g/s) · (W*s)^T`, which holds when `s` is per-input-channel.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/SmoothQuant/SmoothQuantModifier.cs
git commit -m "feat(algorithms): implement SmoothQuantModifier"
```

---

### Task 7: Update `AlgorithmsRegistration` + tests

**Files:**
- Modify: `src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs`
- Modify: `tests/LLMCompressorSharp.Tests/Algorithms/AlgorithmsRegistrationTests.cs`

- [ ] **Step 1: Update `AlgorithmsRegistration.cs`**

Use the Edit tool to add two new methods and call them from `RegisterAll`:

Add new `using`s at the top:
```csharp
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
```

Update `RegisterAll`:
```csharp
    public static void RegisterAll()
    {
        RegisterRtn();
        RegisterMagnitudePruning();
        RegisterWanda();
        RegisterSmoothQuant();
    }
```

Add two new methods after `RegisterMagnitudePruning`:
```csharp
    /// <summary>Registers the WANDA algorithm.</summary>
    public static void RegisterWanda()
    {
        ModifierRegistry.Register<WandaConfig>("WANDA", c => new WandaModifier(c));
    }

    /// <summary>Registers the SmoothQuant algorithm.</summary>
    public static void RegisterSmoothQuant()
    {
        ModifierRegistry.Register<SmoothQuantConfig>("SmoothQuant", c => new SmoothQuantModifier(c));
    }
```

The `WandaModifier` and `SmoothQuantModifier` types must be reachable — add the corresponding `using` lines (`LLMCompressorSharp.Core.Algorithms.Pruning` is already there from Magnitude; `Algorithms.SmoothQuant` is new).

- [ ] **Step 2: Update `AlgorithmsRegistrationTests.cs`**

Use Edit to extend the three existing tests:

In `RegisterAll_RegistersAllBuiltInAlgorithms`, add assertions:
```csharp
        ModifierRegistry.Resolve("WANDA").Should().NotBeNull();
        ModifierRegistry.Resolve("SmoothQuant").Should().NotBeNull();
```

In `Resolve_AfterRegistration_FactoryProducesCorrectModifier`, add (after the existing Magnitude check):
```csharp
        var wandaReg = ModifierRegistry.Resolve("WANDA");
        var wandaInstance = wandaReg!.Factory(new WandaConfig());
        wandaInstance.Should().BeOfType<WandaModifier>();

        var sqReg = ModifierRegistry.Resolve("SmoothQuant");
        var sqInstance = sqReg!.Factory(new SmoothQuantConfig());
        sqInstance.Should().BeOfType<SmoothQuantModifier>();
```

Add the `using`s:
```csharp
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
```

- [ ] **Step 3: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~AlgorithmsRegistrationTests"`
Expected: 3 tests pass (extended assertions, same test count).

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs tests/LLMCompressorSharp.Tests/Algorithms/AlgorithmsRegistrationTests.cs
git commit -m "feat(algorithms): register WANDA and SmoothQuant in AlgorithmsRegistration"
```

---

### Task 8: Add WANDA integration test to `EndToEndRecipeTests`

**Files:**
- Modify: `tests/LLMCompressorSharp.Tests/Integration/EndToEndRecipeTests.cs`

- [ ] **Step 1: Add a new `[Fact]` method**

Use the Edit tool to add a method to `EndToEndRecipeTests` before its closing `}`:

```csharp
    [Fact]
    public void Recipe_WandaPruning_UsesActivationsToGuideSparsity()
    {
        var yaml = @"
stages:
  - name: prune
    modifiers:
      - type: WANDA
        sparsity: 0.5
        targets: [model.*]
        ignore: [model.lm_head.*]
";

        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(1);

        // 2-row weight: row 0 = [1, 1, 1, 1]; row 1 = [1, 1, 1, 1].
        // Activation per column: col 3 has dominant magnitude.
        using var weight = ones(2, 4);
        using var activation = tensor(new float[,]
        {
            { 0.1f, 0.1f, 0.1f, 10f },
            { 0.1f, 0.1f, 0.1f, 10f },
        });
        using var lmHead = ones(2, 4);

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["model.layer.0.weight"] = weight,
            ["model.lm_head.weight"] = lmHead,
        })
        {
            LayerActivations = new Dictionary<string, Tensor>
            {
                ["model.layer.0"] = activation,
            },
        };

        var session = new CompressionSession(modifiers);
        var status = session.Run(state, new[] { activation });
        status.Should().Be(SessionStatus.Completed);

        var processed = state.NamedWeights["model.layer.0.weight"].cpu();
        // Column 3 (dominant activation) must survive in both rows.
        processed[0, 3].item<float>().Should().Be(1f);
        processed[1, 3].item<float>().Should().Be(1f);

        // lm_head untouched.
        state.NamedWeights["model.lm_head.weight"].cpu().data<float>().ToArray()
            .Should().AllSatisfy(v => v.Should().Be(1f));
    }
```

- [ ] **Step 2: Run the test**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~EndToEndRecipeTests"`
Expected: 2 tests pass (the original + the new one).

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Integration/EndToEndRecipeTests.cs
git commit -m "test(integration): add WANDA end-to-end recipe test"
```

---

### Task 9: Full-solution verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Clean restore + build + test**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```
Expected:
- Build: 0 errors, 0 warnings
- Tests: **~118 passing** (101 prior + 8 Wanda + 8 SmoothQuant + 1 new EndToEnd test)

- [ ] **Step 2: Verify file inventory matches plan**

```powershell
Get-ChildItem -Recurse -Filter "*.cs" -Path src/LLMCompressorSharp.Core/Algorithms/ | ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '') } | Sort-Object
```
Expected new files:
- `src\LLMCompressorSharp.Core\Algorithms\Configs\SmoothQuantConfig.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Configs\SmoothQuantMapping.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Configs\WandaConfig.cs`
- `src\LLMCompressorSharp.Core\Algorithms\Pruning\WandaModifier.cs`
- `src\LLMCompressorSharp.Core\Algorithms\SmoothQuant\SmoothQuantModifier.cs`

No commit — verification only.

---

### Task 10: Merge and tag — STOP here

Controller handles the merge to `main` and the `v0.2.1-alpha` tag.

---

## Self-Review Notes

**Spec coverage:**
- WANDA (saliency = |w| × ‖x‖₂) → Tasks 3, 4
- SmoothQuant (channel scale balancing) → Tasks 5, 6
- LayerActivations routing infrastructure → Task 2
- Registration + integration → Tasks 7, 8

**Out of scope:** GPTQ (Phase 4), SparseGPT (Phase 4), AWQ (Phase 5), Module-derived hook-based activation collection (Phase 3).

**Type consistency:**
- `WandaConfig.Sparsity` validated in `WandaModifier` constructor (matches MagnitudePruning pattern)
- `SmoothQuantConfig.SmoothingStrength` validated in `SmoothQuantModifier` constructor
- `SmoothQuantMapping` record with three named string fields, used positionally in tests
- `CompressionState.LayerActivations` is `IDictionary<string, Tensor>?` — nullable; modifiers throw with actionable messages when null/missing

**Known risks:**
- **`tensor.amax(long[], keepdim)`** — verified working in PerChannelMinMaxObserver (Phase 1a). Should be solid.
- **`tensor.sum(long[], keepdim)`** — same family of reduce ops; should be fine.
- **`tensor[r, c].item<float>()` 2-D indexing** — used extensively in tests. Already verified in Phase 2a RtnModifier tests.
- **YAML deserialization of `IList<SmoothQuantMapping>`** — the RecipeParser converter handles strings and string lists but not nested objects. For Phase 2b we don't have a YAML SmoothQuant integration test (the recipe test in Task 8 only uses WANDA which has primitive-only fields). The SmoothQuant YAML→config path will be tested in Phase 6 when the CLI lands. **Documented as a known gap.**

**Commit count target:** ~10 commits across 9 tasks.

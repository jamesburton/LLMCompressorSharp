# Phase 4c: SparseGPT Algorithm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SparseGPTModifier` — a Hessian-based one-shot pruning algorithm that zeroes out low-saliency weights to reach a target sparsity level. Supports unstructured sparsity (any individual weight) and 2:4 N:M structured sparsity (required for NVIDIA Ampere sparse TensorCore acceleration). Builds directly on the Phase 4a Hessian infrastructure (`HessianAccumulator`, `ActivationHookManager`) and shares its Cholesky-inverse utility with Phase 4b (GPTQ).

**Architecture:**

```
LLMCompressorSharp.TorchExtensions/
└── Hessian/
    ├── HessianAccumulator.cs          ← Phase 4a (existing)
    ├── ForwardHookHandle.cs           ← Phase 4a (existing)
    ├── ActivationHookManager.cs       ← Phase 4a (existing)
    └── CholeskyHessianInverter.cs     ← NEW (shared with Phase 4b)

LLMCompressorSharp.Core/
└── Algorithms/
    ├── Configs/
    │   └── SparseGptConfig.cs         ← NEW
    ├── Pruning/
    │   ├── SparseGptModifier.cs       ← NEW
    │   └── MaskSelector/
    │       ├── IMaskSelector.cs       ← NEW
    │       ├── UnstructuredMaskSelector.cs  ← NEW
    │       └── TwoFourMaskSelector.cs ← NEW

tests/LLMCompressorSharp.Tests/
└── Algorithms/
    └── SparseGpt/                     ← NEW
        ├── CholeskyHessianInverterTests.cs
        ├── UnstructuredMaskSelectorTests.cs
        ├── TwoFourMaskSelectorTests.cs
        ├── SparseGptModifierTests.cs  ← synthetic end-to-end
        └── SparseGptSmokeTests.cs     ← real-model smoke (SkipUnless)
```

**Algorithm (SparseGPT per-layer steps):**

1. During calibration (`OnBatchCore`): Use `ActivationHookManager` to register forward hooks on targeted `nn.Linear` layers. Each hook feeds the layer's input activations into a `HessianAccumulator` (Phase 4a).
2. After calibration (`OnEndCore`): For each targeted layer, extract `H` and:
   - Apply diagonal dampening: `H += λ × mean(diag(H)) × I` where `λ = DampeningFrac` (default 0.01).
   - Cholesky-decompose `H`, then compute `H⁻¹` via back-substitution (`CholeskyHessianInverter`, shared with Phase 4b).
   - Iterate weight columns in blocks of `BlockSize` (default 128):
     - For each column `j` in block: compute per-row saliency `s_j = w_j² / (H⁻¹_jj)²`.
     - Select a binary mask: lowest-saliency entries to prune (zero), per the `MaskSelector` strategy.
     - Update remaining (unpruned) weights using the Hessian inverse to compensate for pruning error.
   - Apply the final mask to the layer weight: write back to `CompressionState.NamedWeights`.

> **Saliency formula clarification:** The formula in the task prompt (`w² / H⁻¹_jj`) is the unsquared variant.
> The SparseGPT paper (Frantar & Alistarh 2023, eq. 4), the upstream `docs/llm-compressor/algorithms/sparsegpt.md`,
> and `docs/llmcompressorsharp/algorithm-mapping.md` all use `w² / (H⁻¹_jj)²` — **denominator squared**.
> This plan uses the squared form. Implementer: if you see the unsquared form referenced elsewhere, treat this plan as authoritative.

**Mask selection is per-output-channel, not global:** For a weight matrix `W: [outFeatures, inFeatures]`, sparsity is applied independently per row (output channel) within each input-column block. For unstructured, the K lowest-saliency columns per row are zeroed. For 2:4, exactly 2 of each consecutive group of 4 input columns are zeroed per row. Verifying this per-row invariant is mandatory in tests.

**Tech Stack:** TorchSharp 0.107.0. `torch.linalg.cholesky`, `torch.linalg.solve_triangular`. FP32 throughout inner loop (weight copy in FP32; write back to original dtype). Phase 4a infrastructure in `LLMCompressorSharp.TorchExtensions`. New modifier code in `LLMCompressorSharp.Core`.

**Reference docs:** `docs/llm-compressor/algorithms/sparsegpt.md`, `docs/llmcompressorsharp/algorithm-mapping.md` §SparseGPT, `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §Phase 4.

---

## File Structure

```
src/LLMCompressorSharp.TorchExtensions/
└── Hessian/
    └── CholeskyHessianInverter.cs          ← NEW (shared with GPTQ / Phase 4b)

src/LLMCompressorSharp.Core/
└── Algorithms/
    ├── Configs/
    │   └── SparseGptConfig.cs              ← NEW
    └── Pruning/
        ├── SparseGptModifier.cs            ← NEW
        └── MaskSelector/
            ├── IMaskSelector.cs            ← NEW
            ├── UnstructuredMaskSelector.cs ← NEW
            └── TwoFourMaskSelector.cs      ← NEW

tests/LLMCompressorSharp.Tests/
└── Algorithms/
    └── SparseGpt/                          ← NEW
        ├── CholeskyHessianInverterTests.cs
        ├── UnstructuredMaskSelectorTests.cs
        ├── TwoFourMaskSelectorTests.cs
        ├── SparseGptModifierTests.cs
        └── SparseGptSmokeTests.cs
```

**Responsibility per file:**

- `CholeskyHessianInverter` — Takes a raw Hessian `H: [d, d]`, applies dampening, performs Cholesky decomposition, and returns `H⁻¹` via triangular solve. **Shared utility** — GPTQ (Phase 4b) uses the identical operation. Lives in `TorchExtensions/Hessian/` alongside the Phase 4a infra.

- `SparseGptConfig` — POCO + `ModifierConfig` subtype. Properties: `Sparsity`, `MaskStructure`, `BlockSize`, `DampeningFrac`, `PreserveSparsityMask` (Phase 5 stub — validated but not applied), `Targets`, `Ignore`.

- `IMaskSelector` — `Tensor Select(Tensor saliency, float sparsity)` — takes a saliency matrix `[outFeatures, blockSize]`, returns a boolean mask `[outFeatures, blockSize]` where `true` = keep, `false` = prune.

- `UnstructuredMaskSelector` — Per-row top-K selection. For each output row, keeps the `ceil((1 - sparsity) × blockSize)` highest-saliency columns.

- `TwoFourMaskSelector` — Per-row, per-group-of-4 selection. For each consecutive group of 4 input columns per row, keeps the top-2 by saliency. Ignores `sparsity` argument (fixed at 0.5); asserts `blockSize % 4 == 0`.

- `SparseGptModifier` — `ModifierBase` subclass. `OnInitialize`: set up `ActivationHookManager` and per-layer `HessianAccumulator` dict. `OnStartCore`: register hooks. `OnBatchCore`: model runs via session caller (hooks fire passively). `OnEndCore`: iterate layers — Cholesky invert + block-wise mask + error propagation + write-back. `OnFinalizeCore`: dispose hooks and accumulators.

**Out of scope:**
- OWL (`sparsity_profile: owl`) — per-layer sparsity from outlier statistics. Phase 5+ scope.
- Block-sparse patterns (groups of consecutive weights zeroed together) — future.
- Combined sparsity + quantization (`preserve_sparsity_mask: true` stacking) — Phase 5 recipe topic.
- `offload_hessians` (CPU offload of the `H` matrix) — performance tuning, not correctness. Future.
- Activation reordering (`act_order`) — GPTQ concept; not part of baseline SparseGPT.

---

## Prerequisites & Conventions

- Phase 4a is merged. Tag `v0.4.0-alpha`. 203 tests passing on `main`.
- Branch off `main` as `feature/4c-sparsegpt`.
- Target tag at end of this phase: `v0.6.0-alpha`.
- `.editorconfig` exempts snake_case in `src/LLMCompressorSharp.TorchExtensions/**.cs`. New modifier code under `src/LLMCompressorSharp.Core/Algorithms/Pruning/` uses standard PascalCase — no snake_case exemption.
- StyleCop rules in effect: SA1402, SA1500, SA1515, SA1642, SA1201, SA1312, SA1117, SA1118.
- xUnit.v3 1.0.0 — use `Assert.SkipUnless(condition, reason)` for conditional skips (NOT `Skip.IfNot`).
- FluentAssertions for assertions. No mocks for math.
- `using var scope = torch.NewDisposeScope()` is mandatory around every calibration-loop iteration.

**Phase 4a infrastructure confirmed API (do NOT re-derive):**

- TorchSharp 0.107.0 hooks fire via `.call()`, NOT `.forward()`.
- `register_forward_hook` is on `HookableModule<TPreHook, TPostHook>` (inherited by `Module<Tensor, Tensor>`).
- Post-hook delegate: `Func<Module<Tensor, Tensor>, Tensor, Tensor, Tensor>` — `(module, input, output)` returning output unchanged.
- Returns `HookRemover` with `.remove()`. Wrapped in `ForwardHookHandle`.
- `ActivationHookManager.RegisterFor(Module<Tensor, Tensor> layer, HessianAccumulator accumulator)` — single call installs the hook; hook passes the layer's `input` tensor to `accumulator.Update(input)`.
- `HessianAccumulator` accumulates `H = Σ XᵀX` in FP32. `GetHessian()` returns a clone; `SampleCount` gives row count.

**Shared with Phase 4b — coordination rules:**

`CholeskyHessianInverter` is the one utility shared between Phase 4b (GPTQ) and Phase 4c (SparseGPT). The two phases may land in either order:
- **If Phase 4b ships first:** `CholeskyHessianInverter` will already exist in `TorchExtensions/Hessian/`. Task 3 of this plan becomes "verify / reuse" — read the existing type, confirm it matches the interface spec below, add any missing constructor parameters, add any missing tests.
- **If Phase 4c ships first (this plan):** Task 3 introduces `CholeskyHessianInverter` here. When Phase 4b is subsequently implemented, its plan's Cholesky task becomes "reuse from 4c — add GPTQ-specific tests if any are missing".
- **If running in parallel:** Coordinate via the `feature/4b-gptq` branch. Do NOT duplicate the type; merge the branch that introduces it first, then rebase.

---

### Task 1: Branch + probe `linalg.cholesky` / `linalg.solve_triangular`

**Files:** (no file changes — research and branch setup only)

- [ ] **Step 1: Branch + baseline**

```powershell
git log --oneline -1
git checkout -b feature/4c-sparsegpt
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"
```

Expected: 203 passing.

- [ ] **Step 2: Probe `torch.linalg.cholesky` and `torch.linalg.solve_triangular`**

These are the two new TorchSharp API surfaces Phase 4c uses beyond what Phase 4a verified. Confirm their exact signatures in TorchSharp 0.107.0 before writing any code.

```csharp
// Probe sketch — adapt to a console app or a quick scratch test.
// Verify: (1) the method names, (2) the `upper` parameter on solve_triangular,
// (3) that the return type is a plain Tensor (not tuple), (4) that the operation
// can run on CPU in Release mode without a GPU.

using TorchSharp;
using static TorchSharp.torch;

// Build a 4×4 symmetric positive-definite matrix.
var A = tensor(new float[,]
{
    { 4f, 2f, 1f, 0f },
    { 2f, 5f, 2f, 1f },
    { 1f, 2f, 6f, 2f },
    { 0f, 1f, 2f, 7f },
});

// Cholesky — expected: lower-triangular L such that L @ L.T == A.
var L = linalg.cholesky(A);
Console.WriteLine($"L shape: {string.Join(",", L.shape)}");
Console.WriteLine($"L[0,0]: {L[0, 0].item<float>():F4}");  // should be ~2.0

// Solve L @ x = I  (lower triangular)
var eye = torch.eye(4);
var Linv = linalg.solve_triangular(L, eye, upper: false);
Console.WriteLine($"Linv shape: {string.Join(",", Linv.shape)}");

// H_inv = Linv.T @ Linv  (the standard GPTQ/SparseGPT Hessian inverse formula)
var Hinv = mm(Linv.t(), Linv);
Console.WriteLine($"Hinv[0,0]: {Hinv[0, 0].item<float>():F6}");  // should be small, positive

Console.WriteLine("OK — linalg.cholesky + linalg.solve_triangular confirmed.");
```

Verify:
- `torch.linalg.cholesky(tensor)` returns a `Tensor` (lower-triangular by default).
- `torch.linalg.solve_triangular(A, B, upper: bool)` returns a `Tensor`.
- If `solve_triangular` doesn't exist: alternative is `torch.triangular_solve(B, A, upper: false).solution` — document the fallback in `PR_TO_TORCHSHARP.md` as a P-003 entry.
- Confirm `tensor[range, range]` slicing syntax works for the block iteration pattern (already used in GPTQ sketch in `algorithm-mapping.md`, but worth sanity-checking).
- **Confirm `model.get_submodule(string path)` exists in TorchSharp 0.107.0.** The smoke test (Task 6) uses it to resolve `model.layers.0.self_attn.q_proj`. Probe sketch:

```csharp
// Does Module expose get_submodule(string)?
using var linear = torch.nn.Linear(4, 4);
using var seq = torch.nn.Sequential(("fc", linear));
var sub = seq.get_submodule("fc");          // expected: non-null Module
Console.WriteLine($"get_submodule: {sub?.GetType().Name ?? "NOT FOUND"}");
// If not found: use named_modules() enumeration instead and adapt Task 6's smoke test.
```

If `get_submodule` does not exist, replace line `var targetModule = model.get_submodule(targetLayer) as Module<Tensor, Tensor>;` in the Task 6 smoke test with a `named_modules()` walk:

```csharp
var targetModule = model.named_modules()
    .FirstOrDefault(m => m.name == targetLayer)
    .module as Module<Tensor, Tensor>;
```

No commit for this task.

---

### Task 2: `SparseGptConfig` + recipe registration

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/SparseGptConfig.cs`
- Modify: `src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs`

This task introduces the configuration type and wires it into the recipe system. No algorithm logic yet. Write the config, register it, and verify the recipe parser can round-trip a YAML snippet.

- [ ] **Step 1: Write the failing test**

Create `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptConfigTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Algorithms.SparseGpt;

/// <summary>Tests for <see cref="SparseGptConfig"/> defaults, validation, and YAML round-trip.</summary>
public class SparseGptConfigTests
{
    [Fact]
    public void Defaults_AreCorrect()
    {
        var config = new SparseGptConfig();
        config.Sparsity.Should().Be(0.5f);
        config.MaskStructure.Should().Be("0:0");
        config.BlockSize.Should().Be(128);
        config.DampeningFrac.Should().BeApproximately(0.01f, 1e-6f);
        config.PreserveSparsityMask.Should().BeFalse();
        config.Targets.Should().BeEmpty();
        config.Ignore.Should().BeEmpty();
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    public void Sparsity_OutOfRange_ThrowsOnModifierConstruction(float sparsity)
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var config = new SparseGptConfig { Sparsity = sparsity };
        var act = () => ModifierRegistry.Create(config);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData("0:0")]
    [InlineData("2:4")]
    public void MaskStructure_ValidValues_RoundTrip(string maskStructure)
    {
        var config = new SparseGptConfig { MaskStructure = maskStructure };
        config.MaskStructure.Should().Be(maskStructure);
    }

    [Fact]
    public void MaskStructure_Invalid_ThrowsOnModifierConstruction()
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var config = new SparseGptConfig { MaskStructure = "3:4" };
        var act = () => ModifierRegistry.Create(config);
        act.Should().Throw<ArgumentException>().WithMessage("*MaskStructure*");
    }

    [Fact]
    public void YamlRoundTrip_UnstructuredRecipe()
    {
        AlgorithmsRegistration.RegisterSparseGpt();

        const string yaml = """
            stages:
              - name: sparsegpt
                modifiers:
                  - type: SparseGPT
                    sparsity: 0.5
                    mask_structure: "0:0"
                    block_size: 128
                    dampening_frac: 0.01
                    targets: ["Linear"]
                    ignore: ["lm_head"]
            """;

        var recipe = RecipeParser.Parse(yaml);
        recipe.Stages.Should().HaveCount(1);
        var config = recipe.Stages[0].Modifiers[0].Should().BeOfType<SparseGptConfig>().Subject;
        config.Sparsity.Should().Be(0.5f);
        config.MaskStructure.Should().Be("0:0");
        config.BlockSize.Should().Be(128);
        config.Ignore.Should().ContainSingle().Which.Should().Be("lm_head");
    }

    [Fact]
    public void YamlRoundTrip_TwoFourRecipe()
    {
        AlgorithmsRegistration.RegisterSparseGpt();

        const string yaml = """
            stages:
              - name: sparsegpt
                modifiers:
                  - type: SparseGPT
                    sparsity: 0.5
                    mask_structure: "2:4"
            """;

        var recipe = RecipeParser.Parse(yaml);
        var config = recipe.Stages[0].Modifiers[0].Should().BeOfType<SparseGptConfig>().Subject;
        config.MaskStructure.Should().Be("2:4");
    }
}
```

- [ ] **Step 2: Confirm build fails (CS0246)**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~SparseGptConfigTests"
```

Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptConfigTests.cs
git commit -m "test(sparsegpt): add failing SparseGptConfig tests"
```

- [ ] **Step 4: Implement `SparseGptConfig`**

Create `src/LLMCompressorSharp.Core/Algorithms/Configs/SparseGptConfig.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;
using YamlDotNet.Serialization;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>SparseGptModifier</c> — Hessian-based one-shot pruning via the SparseGPT algorithm.
/// </summary>
/// <remarks>
/// SparseGPT zeroes out low-saliency weights using a second-order (Hessian) estimate of the reconstruction
/// error caused by pruning. Supports unstructured and 2:4 N:M structured sparsity.
/// </remarks>
public sealed class SparseGptConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "SparseGPT";

    /// <summary>Gets or sets the target sparsity ratio in [0, 1]. Default: 0.5 (50%).</summary>
    /// <remarks>0.0 = no pruning; 1.0 = all weights zeroed. For 2:4 structured sparsity, this is
    /// fixed at 0.5 regardless — the modifier will validate consistency.</remarks>
    public float Sparsity { get; set; } = 0.5f;

    /// <summary>
    /// Gets or sets the mask structure. <c>"0:0"</c> = unstructured (default);
    /// <c>"2:4"</c> = NVIDIA semi-structured 2:4 sparsity (2 zeros per 4 consecutive weights).
    /// </summary>
    [YamlMember(Alias = "mask_structure")]
    public string MaskStructure { get; set; } = "0:0";

    /// <summary>Gets or sets the number of input columns processed per block. Default: 128.</summary>
    /// <remarks>Must be a multiple of 4 when <see cref="MaskStructure"/> is <c>"2:4"</c>.</remarks>
    [YamlMember(Alias = "block_size")]
    public int BlockSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the Hessian diagonal dampening fraction. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// Dampening is applied as <c>H += DampeningFrac × mean(diag(H)) × I</c> before Cholesky
    /// decomposition. Increase if Cholesky fails with a "not positive definite" error on
    /// ill-conditioned layers. Typical range: 0.001–0.1.
    /// </remarks>
    [YamlMember(Alias = "dampening_frac")]
    public float DampeningFrac { get; set; } = 0.01f;

    /// <summary>
    /// Gets or sets whether to preserve an existing sparsity mask from a prior pruning pass.
    /// Default: <see langword="false"/>. Currently validated but not applied (Phase 5 stacked-recipe scope).
    /// </summary>
    [YamlMember(Alias = "preserve_sparsity_mask")]
    public bool PreserveSparsityMask { get; set; } = false;
}
```

- [ ] **Step 5: Register `SparseGPT` in `AlgorithmsRegistration`**

Open `src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs` and add the following method and a call from `RegisterAll`:

```csharp
/// <summary>Registers the SparseGPT algorithm.</summary>
public static void RegisterSparseGpt()
{
    ModifierRegistry.Register<SparseGptConfig>("SparseGPT", c => new SparseGptModifier(c));
}
```

Add `RegisterSparseGpt();` inside `RegisterAll()`. `SparseGptModifier` does not exist yet — it will be created in Task 5. If the project won't build without it, add a temporary `throw new NotImplementedException()` stub and replace it in Task 5.

- [ ] **Step 6: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~SparseGptConfigTests"
```

Expected: 7 tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/SparseGptConfig.cs
git add src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs
git commit -m "feat(sparsegpt): add SparseGptConfig and recipe registration"
```

---

### Task 3: `CholeskyHessianInverter` (shared with Phase 4b)

> **Shared with Phase 4b:** This utility performs the identical Cholesky-based Hessian inversion that GPTQ needs.
>
> - **EITHER** reuse Phase 4b's `CholeskyHessianInverter` if Phase 4b has already shipped at implementation time. Read the existing type, confirm it exposes `Tensor Invert(Tensor hessian, float dampeningFrac)`, add the Phase 4c-specific tests below.
> - **OR** introduce it here (as written below) if Phase 4c lands first. Phase 4b then reuses it.
>
> Do NOT duplicate. Coordinate on the `feature/4b-gptq` branch if running in parallel.

**Files:**
- Create (if not already present): `src/LLMCompressorSharp.TorchExtensions/Hessian/CholeskyHessianInverter.cs`
- Create: `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/CholeskyHessianInverterTests.cs`

- [ ] **Step 1: Write the failing tests**

Create `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/CholeskyHessianInverterTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>Tests for <see cref="CholeskyHessianInverter"/> — dampening + Cholesky + H⁻¹.</summary>
public class CholeskyHessianInverterTests
{
    [Fact]
    public void Invert_IdentityMatrix_ReturnsSelf()
    {
        // H = I; H_inv should also be I (dampening adds a fraction of mean(diag)=1).
        // After dampening: H' = I + 0.01 * 1.0 * I = 1.01 * I.
        // H'_inv = (1/1.01) * I.
        using var H = eye(4, dtype: ScalarType.Float32);
        using var Hinv = CholeskyHessianInverter.Invert(H, dampeningFrac: 0.01f);

        Hinv.shape.Should().Equal(new long[] { 4, 4 });
        // Diagonal should be 1/1.01 ≈ 0.9901.
        var diag = Hinv.diagonal().cpu().data<float>().ToArray();
        foreach (var v in diag)
        {
            v.Should().BeApproximately(1f / 1.01f, 1e-5f);
        }
    }

    [Fact]
    public void Invert_DiagonalMatrix_InvertsCorrectly()
    {
        // H = diag(2, 3, 4, 5). After dampening (mean(diag) = 3.5):
        //   H' = diag(2 + 0.01*3.5, 3 + 0.01*3.5, ...) = diag(2.035, 3.035, 4.035, 5.035).
        //   H'_inv = diag(1/2.035, 1/3.035, 1/4.035, 1/5.035).
        using var diagonalData = tensor(new float[] { 2f, 3f, 4f, 5f });
        using var H = torch.diag(diagonalData);
        using var Hinv = CholeskyHessianInverter.Invert(H, dampeningFrac: 0.01f);

        float meanDiag = 3.5f;
        float damp = 0.01f * meanDiag;
        var expectedDiag = new float[] {
            1f / (2f + damp),
            1f / (3f + damp),
            1f / (4f + damp),
            1f / (5f + damp),
        };
        var actualDiag = Hinv.diagonal().cpu().data<float>().ToArray();
        for (int i = 0; i < 4; i++)
        {
            actualDiag[i].Should().BeApproximately(expectedDiag[i], 1e-5f);
        }
    }

    [Fact]
    public void Invert_ZeroDampeningFrac_NoDampeningApplied()
    {
        // With dampening_frac = 0, H stays as-is for a well-conditioned matrix.
        using var diagonalData = tensor(new float[] { 4f, 9f });
        using var H = torch.diag(diagonalData);
        using var Hinv = CholeskyHessianInverter.Invert(H, dampeningFrac: 0f);

        var diag = Hinv.diagonal().cpu().data<float>().ToArray();
        diag[0].Should().BeApproximately(1f / 4f, 1e-5f);
        diag[1].Should().BeApproximately(1f / 9f, 1e-5f);
    }

    [Fact]
    public void Invert_ReturnsFp32()
    {
        using var H = eye(3, dtype: ScalarType.Float32);
        using var Hinv = CholeskyHessianInverter.Invert(H, dampeningFrac: 0.01f);
        Hinv.dtype.Should().Be(ScalarType.Float32);
    }

    [Fact]
    public void Invert_NullHessian_Throws()
    {
        var act = () => CholeskyHessianInverter.Invert(null!, dampeningFrac: 0.01f);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Invert_NonSquareHessian_Throws()
    {
        using var H = zeros(3, 4, dtype: ScalarType.Float32);
        var act = () => CholeskyHessianInverter.Invert(H, dampeningFrac: 0.01f);
        act.Should().Throw<ArgumentException>().WithMessage("*square*");
    }

    [Fact]
    public void Invert_NegativeDampeningFrac_Throws()
    {
        using var H = eye(3, dtype: ScalarType.Float32);
        var act = () => CholeskyHessianInverter.Invert(H, dampeningFrac: -0.01f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
```

- [ ] **Step 2: Confirm build fails, commit failing tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~CholeskyHessianInverterTests"
```

Expected: BUILD FAILS.

```powershell
git add tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/CholeskyHessianInverterTests.cs
git commit -m "test(torch-extensions): add failing CholeskyHessianInverter tests"
```

- [ ] **Step 3: Implement `CholeskyHessianInverter`**

Create `src/LLMCompressorSharp.TorchExtensions/Hessian/CholeskyHessianInverter.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Inverts a Hessian matrix using Cholesky decomposition with diagonal dampening.
/// </summary>
/// <remarks>
/// Shared by GPTQ (Phase 4b) and SparseGPT (Phase 4c). The inversion follows the standard
/// second-order weight compression recipe:
/// <list type="number">
/// <item><description>Apply dampening: <c>H += λ × mean(diag(H)) × I</c>, where <c>λ = dampeningFrac</c>.</description></item>
/// <item><description>Cholesky decompose: <c>L = chol(H)</c> (lower triangular).</description></item>
/// <item><description>Solve <c>L × L_inv = I</c> to get <c>L⁻¹</c>.</description></item>
/// <item><description>Return <c>H⁻¹ = L⁻¹ᵀ × L⁻¹</c>.</description></item>
/// </list>
/// The Hessian is expected in FP32 (as accumulated by <see cref="HessianAccumulator"/>).
/// </remarks>
public static class CholeskyHessianInverter
{
    /// <summary>
    /// Inverts the given Hessian matrix.
    /// </summary>
    /// <param name="hessian">The Hessian <c>H: [d, d]</c> accumulated via <c>H = Σ XᵀX</c>. Must be square and FP32.</param>
    /// <param name="dampeningFrac">
    /// Fraction of <c>mean(diag(H))</c> to add to the diagonal before decomposition.
    /// Typical range: 0.001–0.1. Default used by callers: 0.01. Pass 0.0 to skip dampening.
    /// </param>
    /// <returns>A new <c>Tensor</c> of shape <c>[d, d]</c> containing <c>H⁻¹</c>. The caller owns disposal.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="hessian"/> is null.</exception>
    /// <exception cref="ArgumentException">If <paramref name="hessian"/> is not a 2-D square tensor.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If <paramref name="dampeningFrac"/> is negative.</exception>
    /// <exception cref="InvalidOperationException">
    /// If Cholesky decomposition fails after dampening (matrix is not positive definite).
    /// Increase <paramref name="dampeningFrac"/> or check that the calibration dataset is not degenerate.
    /// </exception>
    public static Tensor Invert(Tensor hessian, float dampeningFrac)
    {
        ArgumentNullException.ThrowIfNull(hessian);
        if (hessian.shape.Length != 2 || hessian.shape[0] != hessian.shape[1])
        {
            throw new ArgumentException(
                $"Hessian must be a square 2-D tensor, got shape [{string.Join(", ", hessian.shape)}].",
                nameof(hessian));
        }

        if (dampeningFrac < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(dampeningFrac), dampeningFrac, "dampeningFrac must be >= 0.");
        }

        // Work on a clone to avoid mutating the caller's tensor.
        using var H = hessian.to(ScalarType.Float32).clone();

        // Apply diagonal dampening: H += λ × mean(diag(H)) × I.
        if (dampeningFrac > 0f)
        {
            using var diagH = H.diagonal();
            float meanDiag = diagH.mean().item<float>();
            float damp = dampeningFrac * meanDiag;
            H.fill_diagonal_(diagH + damp);
        }

        // Cholesky decomposition. Throws if H is not positive definite.
        // If this fails with a "not positive definite" error, increase dampeningFrac.
        Tensor L;
        try
        {
            L = linalg.cholesky(H); // lower triangular
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "Cholesky decomposition failed — the Hessian is not positive definite after dampening. "
                + $"Try increasing DampeningFrac (current: {dampeningFrac}). "
                + "Also ensure the calibration dataset has sufficient diversity (>= d samples for d input features).",
                ex);
        }

        // Solve L @ L_inv = I to get L⁻¹.
        // TorchSharp 0.107.0: linalg.solve_triangular(A, B, upper: bool).
        // If solve_triangular is unavailable, use: triangular_solve(eye, L, upper: false).solution
        // (see PR_TO_TORCHSHARP.md P-003 if that fallback is needed).
        using (L)
        {
            int d = (int)L.shape[0];
            using var eye = torch.eye(d, dtype: ScalarType.Float32, device: L.device);
            using var Linv = linalg.solve_triangular(L, eye, upper: false);

            // H⁻¹ = L⁻¹ᵀ × L⁻¹  (caller owns this tensor).
            return mm(Linv.t(), Linv);
        }
    }
}
```

> **If `linalg.solve_triangular` does not exist in TorchSharp 0.107.0:**
> Replace the call with `torch.triangular_solve(eye, L, upper: false).solution`.
> Add an entry to `PR_TO_TORCHSHARP.md` as P-003 (`linalg.solve_triangular` gap).
> Adapt the test to the fallback pattern if required.

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~CholeskyHessianInverterTests"
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/Hessian/CholeskyHessianInverter.cs
git commit -m "feat(torch-extensions): implement CholeskyHessianInverter (shared GPTQ/SparseGPT)"
```

---

### Task 4: TDD mask selection strategies

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/IMaskSelector.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/UnstructuredMaskSelector.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/TwoFourMaskSelector.cs`
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/UnstructuredMaskSelectorTests.cs`
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/TwoFourMaskSelectorTests.cs`

Mask selectors operate on a saliency matrix `[outFeatures, blockSize]` and return a boolean mask of the same shape (`true` = keep, `false` = prune). They are pure-tensor utilities — no Hessian setup required — which makes them easy to test in isolation.

> **Key invariant:** Selection is per-output-channel (per row), not global. Every row must independently satisfy the sparsity constraint. Tests MUST verify this.

- [ ] **Step 1: Write the failing tests**

Create `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/UnstructuredMaskSelectorTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms.SparseGpt;

/// <summary>Tests for <see cref="UnstructuredMaskSelector"/>.</summary>
public class UnstructuredMaskSelectorTests
{
    [Fact]
    public void Select_50PercentSparsity_HalfZeroedPerRow()
    {
        // 2 rows × 4 columns. Saliency distinct per row.
        //   Row 0: [0.1, 0.4, 0.3, 0.2] → keep top-2: indices 1,2 → prune 0,3.
        //   Row 1: [0.5, 0.1, 0.2, 0.4] → keep top-2: indices 0,3 → prune 1,2.
        var selector = new UnstructuredMaskSelector();
        using var saliency = tensor(new float[,]
        {
            { 0.1f, 0.4f, 0.3f, 0.2f },
            { 0.5f, 0.1f, 0.2f, 0.4f },
        });

        using var mask = selector.Select(saliency, sparsity: 0.5f);

        // Each row must have exactly 2 trues (kept) and 2 falses (pruned).
        var data = mask.cpu().data<bool>().ToArray();
        data[0..4].Where(v => v).Should().HaveCount(2); /* row 0 */
        data[4..8].Where(v => v).Should().HaveCount(2); /* row 1 */

        // Verify exact positions for row 0.
        mask[0, 1].item<bool>().Should().BeTrue();  // keep
        mask[0, 2].item<bool>().Should().BeTrue();  // keep
        mask[0, 0].item<bool>().Should().BeFalse(); // prune
        mask[0, 3].item<bool>().Should().BeFalse(); // prune
    }

    [Fact]
    public void Select_SparsityZero_AllKept()
    {
        var selector = new UnstructuredMaskSelector();
        using var saliency = tensor(new float[,] { { 1f, 2f, 3f, 4f } });
        using var mask = selector.Select(saliency, sparsity: 0f);

        mask.cpu().data<bool>().ToArray().Should().AllSatisfy(v => v.Should().BeTrue());
    }

    [Fact]
    public void Select_SparsityOne_AllPruned()
    {
        var selector = new UnstructuredMaskSelector();
        using var saliency = tensor(new float[,] { { 1f, 2f, 3f, 4f } });
        using var mask = selector.Select(saliency, sparsity: 1f);

        mask.cpu().data<bool>().ToArray().Should().AllSatisfy(v => v.Should().BeFalse());
    }

    [Fact]
    public void Select_MultipleRows_EachRowIndependent()
    {
        // Verify the per-row invariant: every row independently satisfies sparsity.
        var selector = new UnstructuredMaskSelector();
        using var saliency = rand(8, 128); // 8 output channels × 128 input columns

        using var mask = selector.Select(saliency, sparsity: 0.5f);

        // Each of the 8 rows must have exactly 64 kept entries.
        for (int row = 0; row < 8; row++)
        {
            using var rowMask = mask[row];
            int kept = (int)rowMask.to(ScalarType.Int32).sum().item<int>();
            kept.Should().Be(64, $"row {row} must keep exactly 64 of 128 entries at 50% sparsity");
        }
    }

    [Fact]
    public void Select_NullSaliency_Throws()
    {
        var selector = new UnstructuredMaskSelector();
        var act = () => selector.Select(null!, 0.5f);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Select_SparsityOutOfRange_Throws()
    {
        var selector = new UnstructuredMaskSelector();
        using var saliency = tensor(new float[,] { { 1f, 2f } });
        var act = () => selector.Select(saliency, sparsity: 1.5f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
```

Create `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/TwoFourMaskSelectorTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Algorithms.SparseGpt;

/// <summary>Tests for <see cref="TwoFourMaskSelector"/>.</summary>
public class TwoFourMaskSelectorTests
{
    [Fact]
    public void Select_EightColumns_TwoZerosPerGroupOfFourPerRow()
    {
        // 2 rows × 8 columns = 2 groups of 4 per row.
        //   Row 0 group 0: saliencies [0.1, 0.4, 0.3, 0.2] → keep indices 1,2.
        //   Row 0 group 1: saliencies [0.7, 0.2, 0.5, 0.9] → keep indices 0,3.
        //   Row 1 group 0: saliencies [0.5, 0.1, 0.8, 0.3] → keep indices 0,2.
        //   Row 1 group 1: saliencies [0.4, 0.6, 0.1, 0.9] → keep indices 1,3.
        var selector = new TwoFourMaskSelector();
        using var saliency = tensor(new float[,]
        {
            { 0.1f, 0.4f, 0.3f, 0.2f,   0.7f, 0.2f, 0.5f, 0.9f },
            { 0.5f, 0.1f, 0.8f, 0.3f,   0.4f, 0.6f, 0.1f, 0.9f },
        });

        using var mask = selector.Select(saliency, sparsity: 0.5f /* ignored for 2:4 */);

        // Row 0, group 0: keep columns 1 and 2.
        mask[0, 0].item<bool>().Should().BeFalse();
        mask[0, 1].item<bool>().Should().BeTrue();
        mask[0, 2].item<bool>().Should().BeTrue();
        mask[0, 3].item<bool>().Should().BeFalse();

        // Row 0, group 1: keep columns 4 (idx 0 within group) and 7 (idx 3 within group).
        mask[0, 4].item<bool>().Should().BeTrue();
        mask[0, 5].item<bool>().Should().BeFalse();
        mask[0, 6].item<bool>().Should().BeFalse();
        mask[0, 7].item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Select_MultipleRows_EachGroupIndependentPerRow()
    {
        // Verify: for each row, each group of 4 has exactly 2 trues.
        var selector = new TwoFourMaskSelector();
        using var saliency = rand(8, 128); // 8 rows × 128 cols = 32 groups per row

        using var mask = selector.Select(saliency, sparsity: 0.5f);

        for (int row = 0; row < 8; row++)
        {
            for (int g = 0; g < 128; g += 4)
            {
                int kept = 0;
                for (int col = g; col < g + 4; col++)
                {
                    if (mask[row, col].item<bool>())
                    {
                        kept++;
                    }
                }

                kept.Should().Be(2, $"row {row} group starting at col {g} must keep exactly 2 of 4 entries");
            }
        }
    }

    [Fact]
    public void Select_BlockSizeNotMultipleOf4_Throws()
    {
        var selector = new TwoFourMaskSelector();
        using var saliency = tensor(new float[,] { { 1f, 2f, 3f } }); // 3 columns — not multiple of 4
        var act = () => selector.Select(saliency, sparsity: 0.5f);
        act.Should().Throw<ArgumentException>().WithMessage("*multiple of 4*");
    }

    [Fact]
    public void Select_NullSaliency_Throws()
    {
        var selector = new TwoFourMaskSelector();
        var act = () => selector.Select(null!, 0.5f);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Select_MaskHasSameDimensionsAsSaliency()
    {
        var selector = new TwoFourMaskSelector();
        using var saliency = rand(4, 8);
        using var mask = selector.Select(saliency, sparsity: 0.5f);
        mask.shape.Should().Equal(saliency.shape);
    }
}
```

- [ ] **Step 2: Confirm build fails, commit failing tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~MaskSelectorTests"
```

Expected: BUILD FAILS.

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/UnstructuredMaskSelectorTests.cs
git add tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/TwoFourMaskSelectorTests.cs
git commit -m "test(sparsegpt): add failing mask selector tests"
```

- [ ] **Step 3: Implement `IMaskSelector`, `UnstructuredMaskSelector`, `TwoFourMaskSelector`**

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/IMaskSelector.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;

namespace LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;

/// <summary>
/// Selects which weights to retain (keep = <see langword="true"/>) based on a saliency matrix.
/// </summary>
/// <remarks>
/// Selection is applied per-output-channel (per row): each row of the saliency matrix is treated
/// independently. Implementations must guarantee that every row satisfies the sparsity constraint.
/// </remarks>
public interface IMaskSelector
{
    /// <summary>
    /// Computes a boolean keep-mask from a saliency matrix.
    /// </summary>
    /// <param name="saliency">
    /// The saliency scores of shape <c>[outFeatures, blockSize]</c>. Higher = more important.
    /// For SparseGPT: <c>saliency[i, j] = W[i, j]² / (H⁻¹[j, j])²</c>.
    /// </param>
    /// <param name="sparsity">The fraction of weights to prune in [0, 1].</param>
    /// <returns>
    /// A boolean tensor of the same shape as <paramref name="saliency"/>: <see langword="true"/> = keep,
    /// <see langword="false"/> = prune. The caller owns disposal.
    /// </returns>
    /// <exception cref="ArgumentNullException">If <paramref name="saliency"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If <paramref name="sparsity"/> is outside [0, 1].</exception>
    Tensor Select(Tensor saliency, float sparsity);
}
```

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/UnstructuredMaskSelector.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;

/// <summary>
/// Unstructured mask selector: zeroes any individual weight. Selects the top-K highest-saliency
/// entries to keep, independently per output channel (row).
/// </summary>
/// <remarks>
/// For a block of shape <c>[outFeatures, blockSize]</c> with <c>sparsity = 0.5</c>, each row
/// retains the top-<c>ceil(0.5 × blockSize)</c> weights by saliency and prunes the rest.
/// </remarks>
public sealed class UnstructuredMaskSelector : IMaskSelector
{
    /// <inheritdoc />
    public Tensor Select(Tensor saliency, float sparsity)
    {
        ArgumentNullException.ThrowIfNull(saliency);
        if (sparsity is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(sparsity), sparsity, "sparsity must be in [0, 1].");
        }

        int blockSize = (int)saliency.shape[1];
        int numToKeep = (int)Math.Ceiling(blockSize * (1f - sparsity));
        numToKeep = Math.Clamp(numToKeep, 0, blockSize);

        if (numToKeep == blockSize)
        {
            return ones_like(saliency, dtype: ScalarType.Bool);
        }

        if (numToKeep == 0)
        {
            return zeros_like(saliency, dtype: ScalarType.Bool);
        }

        // topk returns the k largest values per row; indices mark the kept columns.
        var (topkVals, topkIdx) = saliency.topk(numToKeep, dim: 1, largest: true, sorted: false);
        using (topkVals)
        {
            // Scatter trues into a zero mask at the topk positions.
            var mask = zeros_like(saliency, dtype: ScalarType.Bool);
            mask.scatter_(1, topkIdx, ones_like(topkIdx, dtype: ScalarType.Bool));
            return mask; // caller owns
        }
    }
}
```

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/TwoFourMaskSelector.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;

/// <summary>
/// 2:4 semi-structured mask selector: exactly 2 non-zero weights per every 4 consecutive input
/// columns, per output channel. Required for NVIDIA Ampere sparse TensorCore acceleration.
/// </summary>
/// <remarks>
/// The <c>sparsity</c> argument is accepted for API compatibility but is ignored — 2:4 is fixed at
/// 50% sparsity. The column dimension of <paramref name="saliency"/> must be a multiple of 4.
/// </remarks>
public sealed class TwoFourMaskSelector : IMaskSelector
{
    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="sparsity"/> is ignored for 2:4 (fixed at 0.5). Provide any value in [0,1].
    /// </remarks>
    public Tensor Select(Tensor saliency, float sparsity)
    {
        ArgumentNullException.ThrowIfNull(saliency);
        if (sparsity is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(sparsity), sparsity, "sparsity must be in [0, 1].");
        }

        long outFeatures = saliency.shape[0];
        long blockSize = saliency.shape[1];

        if (blockSize % 4 != 0)
        {
            throw new ArgumentException(
                $"TwoFourMaskSelector requires the column dimension (blockSize) to be a multiple of 4, "
                + $"but got {blockSize}.",
                nameof(saliency));
        }

        long numGroups = blockSize / 4;

        // Reshape to [outFeatures, numGroups, 4] to process groups independently.
        using var grouped = saliency.reshape(outFeatures, numGroups, 4);

        // topk(2) per group returns the 2 highest-saliency indices within each group of 4.
        var (topkVals, topkIdx) = grouped.topk(2, dim: 2, largest: true, sorted: false);
        using (topkVals)
        {
            // Scatter trues into a zero mask shaped [outFeatures, numGroups, 4].
            var groupMask = zeros(outFeatures, numGroups, 4, dtype: ScalarType.Bool, device: saliency.device);
            groupMask.scatter_(2, topkIdx, ones_like(topkIdx, dtype: ScalarType.Bool));

            // Reshape back to [outFeatures, blockSize].
            return groupMask.reshape(outFeatures, blockSize); // caller owns
        }
    }
}
```

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~UnstructuredMaskSelectorTests|FullyQualifiedName~TwoFourMaskSelectorTests"
```

Expected: all mask selector tests pass (11 tests).

> **Known scatter_ issue:** `mask.scatter_(dim, index, src)` with a boolean `src` may not exist in TorchSharp 0.107.0. Alternative: use `scatter_(dim, index, ones_like(topkIdx).to(ScalarType.Bool))` or perform `scatter_` with a float tensor and then cast to bool. Adapt as needed and document in `PR_TO_TORCHSHARP.md` if a workaround is required.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/IMaskSelector.cs
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/UnstructuredMaskSelector.cs
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/MaskSelector/TwoFourMaskSelector.cs
git commit -m "feat(sparsegpt): implement IMaskSelector, UnstructuredMaskSelector, TwoFourMaskSelector"
```

---

### Task 5: `SparseGptModifier` core

**Files:**
- Create: `src/LLMCompressorSharp.Core/Algorithms/Pruning/SparseGptModifier.cs`
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptModifierTests.cs`

This is the full end-to-end modifier: calibration via hooks, Hessian inversion, block-wise column iteration, mask + error propagation, weight write-back.

- [ ] **Step 1: Write the failing end-to-end synthetic test**

Create `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptModifierTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.Algorithms.SparseGpt;

/// <summary>Synthetic end-to-end tests for <see cref="SparseGptModifier"/>.</summary>
public class SparseGptModifierTests : IDisposable
{
    // A small deterministic Linear for all tests: in=8, out=4, no bias.
    private readonly Module<Tensor, Tensor> _linear;
    private readonly CompressionState _state;
    private bool _disposed;

    /// <summary>Initializes a new instance of the <see cref="SparseGptModifierTests"/> class.</summary>
    public SparseGptModifierTests()
    {
        manual_seed(42);
        _linear = Linear(8, 4, hasBias: false);
        // Populate CompressionState with the layer weight keyed as "linear.weight".
        var namedWeights = new Dictionary<string, Tensor>
        {
            ["linear.weight"] = ((Linear)_linear).weight!.detach().clone(),
        };
        _state = new CompressionState(namedWeights);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            _linear.Dispose();
            _disposed = true;
        }
    }

    [Fact]
    public void Unstructured_50Percent_ExactlyHalfOfWeightsAreZero()
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var config = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "0:0",
            BlockSize = 8,
            DampeningFrac = 0.01f,
            Targets = new[] { "linear.weight" },
        };

        var modifier = ModifierRegistry.Create(config);
        modifier.Initialize(_state);
        modifier.OnStart(_state);

        // Calibration: feed 16 samples through the Linear layer.
        // (SparseGptModifier hooks the layer and accumulates H = XᵀX passively.)
        using (var scope = NewDisposeScope())
        {
            for (int i = 0; i < 16; i++)
            {
                manual_seed(i);
                using var x = randn(2, 8);
                _state.CurrentBatch = x;
                _state.CurrentBatchIndex = i;
                modifier.OnBatch(_state);

                // Drive the forward pass so hooks fire.
                using var _ = _linear.call(x);
            }
        }

        modifier.OnEnd(_state);
        modifier.Finalize(_state);

        // Verify: exactly 50% of the 32 weights (4×8) are zero.
        var prunedWeight = _state.NamedWeights["linear.weight"];
        int numElements = (int)prunedWeight.numel();
        int numZero = (int)prunedWeight.eq(0f).to(ScalarType.Int32).sum().item<int>();
        numZero.Should().Be(numElements / 2, "50% sparsity must zero exactly half the weights");

        // Verify exact zeros — not merely ε-small.
        var data = prunedWeight.cpu().data<float>().ToArray();
        foreach (var w in data.Where(v => v == 0f))
        {
            w.Should().Be(0f); // tautological but explicit: must be bit-exact zero
        }
    }

    [Fact]
    public void Unstructured_50Percent_UnprunedOutputFinite()
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var config = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "0:0",
            BlockSize = 8,
            Targets = new[] { "linear.weight" },
        };
        var modifier = ModifierRegistry.Create(config);
        modifier.Initialize(_state);
        modifier.OnStart(_state);

        RunCalibration(modifier, _linear, _state, numBatches: 16);

        modifier.OnEnd(_state);
        modifier.Finalize(_state);

        // Load pruned weight back into the layer and run a forward pass.
        using (no_grad())
        {
            ((Linear)_linear).weight!.copy_(_state.NamedWeights["linear.weight"]);
        }

        manual_seed(99);
        using var testInput = randn(4, 8);
        using var output = _linear.call(testInput);
        var outputData = output.cpu().data<float>().ToArray();
        outputData.Should().OnlyContain(v => float.IsFinite(v), "all outputs must be finite after pruning");
    }

    [Fact]
    public void TwoFour_50Percent_EachGroupOfFourHasExactlyTwoZerosPerRow()
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var config = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "2:4",
            BlockSize = 8, // 2 groups of 4
            Targets = new[] { "linear.weight" },
        };
        var modifier = ModifierRegistry.Create(config);
        modifier.Initialize(_state);
        modifier.OnStart(_state);

        RunCalibration(modifier, _linear, _state, numBatches: 16);

        modifier.OnEnd(_state);
        modifier.Finalize(_state);

        var weight = _state.NamedWeights["linear.weight"].cpu();
        int outFeatures = (int)weight.shape[0];
        int inFeatures = (int)weight.shape[1];

        // For every row, for every group of 4 consecutive input columns:
        // exactly 2 must be zero.
        for (int row = 0; row < outFeatures; row++)
        {
            for (int g = 0; g < inFeatures; g += 4)
            {
                int zerosInGroup = 0;
                for (int col = g; col < g + 4; col++)
                {
                    if (weight[row, col].item<float>() == 0f)
                    {
                        zerosInGroup++;
                    }
                }

                zerosInGroup.Should().Be(2,
                    $"row {row} group [{g}..{g + 3}] must have exactly 2 zeros (2:4 pattern)");
            }
        }
    }

    [Fact]
    public void Lifecycle_FinalizeBeforeInitialize_Throws()
    {
        AlgorithmsRegistration.RegisterSparseGpt();
        var modifier = ModifierRegistry.Create(new SparseGptConfig());
        var act = () => modifier.Finalize(_state);
        act.Should().Throw<Exception>(); // ModifierLifecycleException
    }

    /// <summary>
    /// Verifies that SparseGPT's Hessian-guided error compensation produces a meaningfully
    /// lower reconstruction error than naïve magnitude pruning on correlated inputs.
    /// A correlated input produces an off-diagonal Hessian where second-order compensation
    /// matters; magnitude-only pruning ignores this and leaves higher reconstruction error.
    /// </summary>
    [Fact]
    public void Unstructured_50Percent_ReconstructionErrorLowerThanMagnitudePruning()
    {
        // Build correlated calibration data: X = base * scale, where rows are linearly dependent.
        // This creates a rank-deficient Hessian that makes Hessian-guided compensation non-trivial.
        manual_seed(7);
        using var baseVec = randn(1, 8);                         // [1, 8] — one basis direction
        var calibInputs = new List<Tensor>();
        for (int i = 0; i < 32; i++)
        {
            // Each sample is the base vector scaled by a random scalar: strong column correlation.
            using var scale = randn(4, 1);                       // [4, 1]
            calibInputs.Add(scale.mm(baseVec).detach().clone()); // [4, 8]
        }

        // Capture original weight (same for both algorithms).
        using var originalWeight = ((Linear)_linear).weight!.detach().clone(); // [4, 8]

        // ---- SparseGPT pruning ----
        AlgorithmsRegistration.RegisterSparseGpt();
        var sparsegptConfig = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "0:0",
            BlockSize = 8,
            DampeningFrac = 0.05f, // slightly higher to stabilise correlated Hessian
            Targets = new[] { "linear.weight" },
        };

        // Reset state to original weight for SparseGPT run.
        var sgState = new CompressionState(
            new Dictionary<string, Tensor> { ["linear.weight"] = originalWeight.clone() });

        var sgModifier = ModifierRegistry.Create(sparsegptConfig);
        sgModifier.Initialize(sgState);
        sgModifier.OnStart(sgState);
        for (int i = 0; i < calibInputs.Count; i++)
        {
            using var scope = NewDisposeScope();
            sgState.CurrentBatch = calibInputs[i];
            sgState.CurrentBatchIndex = i;
            sgModifier.OnBatch(sgState);
            using var _ = _linear.call(calibInputs[i]);
        }

        sgModifier.OnEnd(sgState);
        sgModifier.Finalize(sgState);
        using var sparsegptWeight = sgState.NamedWeights["linear.weight"].detach().clone();

        // ---- Magnitude pruning (baseline) ----
        // Keep top-50% by absolute value per row, zero the rest.
        using var magWeight = originalWeight.clone();
        int inFeat = (int)originalWeight.shape[1];
        int keepCount = inFeat / 2; // 4 of 8 per row
        for (int row = 0; row < (int)originalWeight.shape[0]; row++)
        {
            using var rowTensor = magWeight[row];
            using var absVals = rowTensor.abs();
            using var topkResult = absVals.topk(keepCount, dim: 0, largest: true, sorted: false);
            using var keepIdxSet = topkResult.indexes;
            using var magnitudeMask = zeros(inFeat, dtype: ScalarType.Bool);
            magnitudeMask.scatter_(0, keepIdxSet, tensor(true));
            rowTensor.masked_fill_(magnitudeMask.logical_not(), 0f);
        }

        // ---- Compare reconstruction errors on held-out inputs ----
        manual_seed(99);
        using var testInput = randn(8, 8); // [8, 8]
        using var originalOut = mm(testInput, originalWeight.t()); // [8, 4]
        using var sgOut = mm(testInput, sparsegptWeight.t());
        using var magOut = mm(testInput, magWeight.t());

        float sgError = (originalOut - sgOut).pow(2).mean().item<float>();
        float magError = (originalOut - magOut).pow(2).mean().item<float>();

        sgError.Should().BeLessThan(magError,
            $"SparseGPT (err={sgError:F6}) should have lower reconstruction error than magnitude pruning (err={magError:F6}) on correlated inputs");

        // Cleanup.
        foreach (var t in calibInputs)
        {
            t.Dispose();
        }
    }

    // Helper to drive forward passes so hooks accumulate the Hessian.
    private static void RunCalibration(
        IModifier modifier,
        Module<Tensor, Tensor> layer,
        CompressionState state,
        int numBatches)
    {
        for (int i = 0; i < numBatches; i++)
        {
            manual_seed(i);
            using var scope = NewDisposeScope();
            using var x = randn(2, 8);
            state.CurrentBatch = x;
            state.CurrentBatchIndex = i;
            modifier.OnBatch(state);
            using var _ = layer.call(x);
        }
    }
}
```

- [ ] **Step 2: Confirm build fails, commit failing tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~SparseGptModifierTests"
```

Expected: BUILD FAILS.

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptModifierTests.cs
git commit -m "test(sparsegpt): add failing SparseGptModifier end-to-end tests"
```

- [ ] **Step 3: Implement `SparseGptModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/Pruning/SparseGptModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Pruning.MaskSelector;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Core.Algorithms.Pruning;

/// <summary>
/// Hessian-based one-shot pruning via the SparseGPT algorithm (Frantar &amp; Alistarh 2023).
/// </summary>
/// <remarks>
/// <para>
/// SparseGPT minimises the per-layer reconstruction error <c>||Y - WX||²</c> subject to a sparsity
/// constraint on W. It uses the same Hessian infrastructure as GPTQ (Phase 4b) — accumulated during
/// calibration via forward hooks — but instead of quantising, it prunes: zeroing the
/// lowest-saliency weights and propagating the pruning error to the remaining weights using the
/// Hessian inverse.
/// </para>
/// <para>
/// Saliency for column j in block: <c>s_j = W_j² / (H⁻¹_jj)²</c> (denominator squared, per the
/// SparseGPT paper eq. 4). Selection is per-output-channel (per row of W), not global.
/// </para>
/// <para>
/// Lifecycle: <c>OnInitialize</c> sets up accumulators; <c>OnStartCore</c> registers forward hooks;
/// <c>OnBatchCore</c> is a no-op (hooks accumulate passively); <c>OnEndCore</c> runs the pruning;
/// <c>OnFinalizeCore</c> disposes hooks + accumulators.
/// </para>
/// </remarks>
public sealed class SparseGptModifier : ModifierBase
{
    private readonly float _sparsity;
    private readonly int _blockSize;
    private readonly float _dampeningFrac;
    private readonly IMaskSelector _maskSelector;

    // Per-layer state: layer module reference + Hessian accumulator.
    // Populated in OnStartCore, consumed in OnEndCore, disposed in OnFinalizeCore.
    private readonly Dictionary<string, (Module<Tensor, Tensor> Layer, HessianAccumulator Accumulator)> _layerState = new();
    private ActivationHookManager? _hookManager;

    /// <summary>Initializes a new instance of the <see cref="SparseGptModifier"/> class.</summary>
    /// <param name="config">The SparseGPT configuration.</param>
    /// <exception cref="ArgumentNullException">If <paramref name="config"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If <see cref="SparseGptConfig.Sparsity"/> is out of [0,1].</exception>
    /// <exception cref="ArgumentException">If <see cref="SparseGptConfig.MaskStructure"/> is not "0:0" or "2:4".</exception>
    public SparseGptModifier(SparseGptConfig config)
        : base("SparseGPT", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);

        if (config.Sparsity is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(config),
                config.Sparsity,
                "Sparsity must be in [0, 1].");
        }

        _maskSelector = config.MaskStructure switch
        {
            "0:0" => new UnstructuredMaskSelector(),
            "2:4" => new TwoFourMaskSelector(),
            _ => throw new ArgumentException(
                $"MaskStructure '{config.MaskStructure}' is not supported. Valid values: \"0:0\" (unstructured), \"2:4\" (semi-structured).",
                nameof(config)),
        };

        _sparsity = config.Sparsity;
        _blockSize = config.BlockSize;
        _dampeningFrac = config.DampeningFrac;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        _layerState.Clear();
    }

    /// <inheritdoc />
    protected override void OnStartCore(CompressionState state)
    {
        // Register a forward hook for each targeted Linear layer so activations are captured
        // during the calibration batches driven by the session caller.
        _hookManager = new ActivationHookManager();

        foreach (var weightName in GetTargetedNames(state))
        {
            // Resolve the layer module from the session state.
            // Convention: CompressionState.NamedWeights key is "<layerPath>.weight".
            // The corresponding module is retrieved via a helper on the state (Phase 3+).
            // For the current implementation: if the state exposes named modules, use them;
            // otherwise, skip (the modifier degrades gracefully without calibration data by
            // falling back to magnitude pruning — log a warning and call MagnitudePruningFallback).
            if (!TryGetLinearLayer(state, weightName, out var layer))
            {
                continue;
            }

            int inFeatures = (int)state.NamedWeights[weightName].shape[1];
            var accumulator = new HessianAccumulator(inFeatures);
            _hookManager.RegisterFor(layer!, accumulator);
            _layerState[weightName] = (layer!, accumulator);
        }
    }

    // OnBatchCore is intentionally left at its base no-op: the forward hooks accumulate
    // H passively when the session caller runs the model forward pass each batch.

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        foreach (var (weightName, (layer, accumulator)) in _layerState)
        {
            if (accumulator.SampleCount == 0)
            {
                // No calibration data — warn and skip. The weight is left unchanged.
                // TODO: structured logging; for now use Debug.WriteLine.
                System.Diagnostics.Debug.WriteLine(
                    $"[SparseGPT] Warning: no calibration samples for '{weightName}'. Skipping pruning.");
                continue;
            }

            using var scope = NewDisposeScope();

            // Retrieve the raw Hessian (FP32, shape [inFeatures, inFeatures]).
            using var H = accumulator.GetHessian();

            // Invert with dampening (shared with Phase 4b GPTQ).
            using var Hinv = CholeskyHessianInverter.Invert(H, _dampeningFrac);

            // Retrieve the weight in FP32 for the inner loop.
            var originalWeight = state.NamedWeights[weightName];
            using var W = originalWeight.to(ScalarType.Float32).clone();

            int outFeatures = (int)W.shape[0];
            int inFeatures = (int)W.shape[1];

            // Block-wise column iteration (left-to-right across input features).
            for (int blockStart = 0; blockStart < inFeatures; blockStart += _blockSize)
            {
                int blockEnd = Math.Min(blockStart + _blockSize, inFeatures);
                int actualBlockSize = blockEnd - blockStart;

                // Slice the weight block and its Hessian-inverse diagonal.
                using var wBlock = W[TensorIndex.Colon, blockStart..blockEnd].clone(); // [out, actualBlockSize]
                using var HinvBlock = Hinv[blockStart..blockEnd, blockStart..blockEnd];     // [actualBlockSize, actualBlockSize]
                using var HinvTail = blockEnd < inFeatures
                    ? Hinv[blockStart..blockEnd, blockEnd..]
                    : null;

                // Saliency: s[i,j] = W[i,j]² / (H⁻¹[j,j])²  (per paper eq. 4, denominator squared).
                // Computed from the SNAPSHOT of wBlock before any within-block error propagation.
                using var HinvDiag = HinvBlock.diagonal();                      // [actualBlockSize]
                using var HinvDiagSq = HinvDiag.pow(2).clamp_min(1e-10f);      // guard against div-by-zero
                using var saliency = wBlock.pow(2) / HinvDiagSq.unsqueeze(0);  // [out, actualBlockSize]

                // Select per-row mask ONCE from the snapshot saliency.
                using var keepMask = _maskSelector.Select(saliency, _sparsity);
                using var pruneMask = keepMask.logical_not();

                // Canonical SparseGPT inner loop (Frantar & Alistarh 2023, Algorithm 1):
                // W1 — mutable working copy updated column-by-column as errors propagate within the block.
                // Q1 — accumulates the final pruned column values (what gets written back to W).
                // Err1 — per-column errors used for cross-block tail propagation.
                using var W1 = wBlock.clone();    // [out, actualBlockSize] — mutable working copy
                using var Q1 = zeros_like(W1);    // [out, actualBlockSize] — accumulates pruned cols
                using var Err1 = zeros_like(W1);  // [out, actualBlockSize] — per-column errors

                for (int j = 0; j < actualBlockSize; j++)
                {
                    // Current value of column j (from the evolving working copy, not the snapshot).
                    using var w = W1[TensorIndex.Colon, j].clone(); // [out]

                    // Zero pruned entries.
                    using var q = w.clone();
                    q.masked_fill_(pruneMask[TensorIndex.Colon, j], 0f);
                    Q1[TensorIndex.Colon, j] = q;

                    float d = HinvBlock[j, j].item<float>();
                    if (MathF.Abs(d) < 1e-10f)
                    {
                        continue; // singular pivot — skip compensation for this column
                    }

                    // Per-row error for this column: (w_j - q_j) / H⁻¹[j,j].
                    using var err = (w - q) / d; // [out]
                    Err1[TensorIndex.Colon, j] = err;

                    // Propagate error forward within the block:
                    //   W1[:, j+1..end] -= err[:, None] @ H⁻¹[j, j+1..end][None, :]
                    if (j + 1 < actualBlockSize)
                    {
                        using var hRow = HinvBlock[j, (j + 1)..].unsqueeze(0); // [1, remaining]
                        using var update = err.unsqueeze(1).mm(hRow);           // [out, remaining]
                        W1[TensorIndex.Colon, (j + 1)..actualBlockSize].sub_(update);
                    }
                }

                // Write the pruned columns back into W (use Q1, NOT wBlockPruned — Q1 reflects
                // within-block error propagation; wBlockPruned discards it).
                W[TensorIndex.Colon, blockStart..blockEnd] = Q1;

                // Propagate accumulated block error to all subsequent blocks (columns beyond blockEnd).
                // Use Err1.mm(HinvTail) — NOT prunedError / HinvDiag — because Err1 is already
                // divided by H⁻¹[j,j] column-by-column above (the standard GPTQ/SparseGPT formula).
                if (HinvTail is not null)
                {
                    using var tailUpdate = Err1.mm(HinvTail); // [out, tail]
                    W[TensorIndex.Colon, blockEnd..].sub_(tailUpdate);
                }
            }

            // Write back to state in the original dtype.
            state.NamedWeights[weightName] = W.to(originalWeight.dtype);
        }
    }

    /// <inheritdoc />
    protected override void OnFinalizeCore(CompressionState state)
    {
        _hookManager?.Dispose();
        _hookManager = null;

        foreach (var (_, (_, accumulator)) in _layerState)
        {
            accumulator.Dispose();
        }

        _layerState.Clear();
    }

    // Resolves the Linear module for a given weight name from the session state.
    // Returns false if the module cannot be resolved (e.g. state doesn't expose modules yet).
    private static bool TryGetLinearLayer(
        CompressionState state,
        string weightName,
        out Module<Tensor, Tensor>? layer)
    {
        // Phase 3+ convention: CompressionState.NamedModules exposes the module graph.
        // Until Phase 3+ populates NamedModules, callers must inject the layer separately.
        // This method is a placeholder that always returns false in the absence of NamedModules.
        // Tests inject layer references by subclassing or by using the internal override point.
        // TODO: wire up CompressionState.NamedModules when the LLaMA pipeline populates it.
        layer = null;

        if (state is IModuleProvider provider)
        {
            var strippedName = weightName.EndsWith(".weight", StringComparison.Ordinal)
                ? weightName[..^".weight".Length]
                : weightName;
            return provider.TryGetModule(strippedName, out layer);
        }

        return false;
    }
}
```

> **Note on `TryGetLinearLayer` and the test setup:** The synthetic test (Task 5, Step 1) drives the calibration by calling `_linear.call(x)` after `modifier.OnBatch(state)`. The hook manager must have access to the `Linear` module to register on it. Two options at implementation time:
> 1. Extend `CompressionState` with a `NamedModules` dictionary (recommended). Add `IModuleProvider` interface or a direct property. The test then populates it alongside `NamedWeights`.
> 2. Accept the module reference via a separate `RegisterLayer(string name, Module<Tensor, Tensor> layer)` method on `SparseGptModifier`. Simpler, but less clean.
>
> The implementer chooses. If option 1, also add a minimal extension to `CompressionState` in this task and update the test to populate it. If option 2, adjust the test accordingly. Do not leave the modifier silently producing no-op output due to unresolved modules.

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~SparseGptModifierTests"
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Pruning/SparseGptModifier.cs
# Include any CompressionState changes if option 1 was chosen:
# git add src/LLMCompressorSharp.Core/Compression/CompressionState.cs
git commit -m "feat(sparsegpt): implement SparseGptModifier with unstructured and 2:4 pruning"
```

---

### Task 6: Real-model smoke test

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptSmokeTests.cs`

Verifies that SparseGPT can prune one `Linear` layer from a real loaded SmolLM2-135M model without producing NaN or Inf in the output, and that all pruned entries are bit-exact zeros. Uses `Assert.SkipUnless` to skip gracefully when the HF cache is absent.

- [ ] **Step 1: Write the smoke test**

Create `tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptSmokeTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.Algorithms.SparseGpt;

/// <summary>
/// Real-model smoke tests for <see cref="SparseGptModifier"/>. Skipped when the HuggingFace
/// cache does not contain SmolLM2-135M.
/// </summary>
[Trait("Category", "Integration")]
public class SparseGptSmokeTests
{
    private const string ModelId = "HuggingFaceTB/SmolLM2-135M";

    [Fact]
    public void SmolLM2_OneLayer_UnstructuredPruned_OutputFiniteAndExactZeros()
    {
        // Skip if model not in cache.
        bool modelCached = HuggingFaceLoader.IsModelCached(ModelId);
        Assert.SkipUnless(modelCached, $"Skipped: {ModelId} not in HuggingFace cache. Run download-test-models.ps1 to prime.");

        AlgorithmsRegistration.RegisterSparseGpt();

        // Load the model.
        using var loadedModel = HuggingFaceLoader.Load(ModelId, device: CPU);
        var model = loadedModel.Model;

        // Target the first q_proj Linear layer in decoder layer 0.
        const string targetLayer = "model.layers.0.self_attn.q_proj";
        const string targetWeight = targetLayer + ".weight";

        // Confirm the layer exists and is a Linear.
        var targetModule = model.get_submodule(targetLayer) as Module<Tensor, Tensor>;
        Assert.SkipUnless(
            targetModule is not null,
            $"Skipped: '{targetLayer}' is not a Module<Tensor, Tensor> in {ModelId}.");

        int inFeatures = (int)model.get_parameter(targetWeight)!.shape[1];
        int outFeatures = (int)model.get_parameter(targetWeight)!.shape[0];

        // Build state with just this one layer's weight.
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            [targetWeight] = model.get_parameter(targetWeight)!.detach().clone(),
        });

        // Also register the module so the hook can fire.
        // (If CompressionState.NamedModules exists, populate it here.)

        var config = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "0:0",
            BlockSize = 128,
            DampeningFrac = 0.01f,
            Targets = new[] { targetWeight },
        };

        var modifier = ModifierRegistry.Create(config);
        modifier.Initialize(state);
        modifier.OnStart(state);

        // Calibration: 8 random batches of shape [2, seq=64, inFeatures].
        // The hook fires when we call targetModule.call(x).
        manual_seed(0);
        for (int i = 0; i < 8; i++)
        {
            using var scope = NewDisposeScope();
            using var x = randn(2, 64, inFeatures);
            state.CurrentBatch = x;
            state.CurrentBatchIndex = i;
            modifier.OnBatch(state);
            using var _ = targetModule!.call(x);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        var prunedWeight = state.NamedWeights[targetWeight];
        int numElements = outFeatures * inFeatures;
        int numZero = (int)prunedWeight.eq(0f).to(ScalarType.Int32).sum().item<int>();

        // 50% sparsity ± 1 (rounding).
        numZero.Should().BeInRange(
            numElements / 2 - 1,
            numElements / 2 + 1,
            "50% sparsity should zero approximately half the weights");

        // Bit-exact zeros (not ε-small).
        var data = prunedWeight.cpu().data<float>().ToArray();
        data.Where(v => v != 0f && MathF.Abs(v) < 1e-7f).Should().BeEmpty(
            "pruned weights must be bit-exact zero, not denormals — check mask-apply path");

        // Forward pass with pruned weight produces finite output.
        using (no_grad())
        {
            ((Linear)targetModule!).weight!.copy_(prunedWeight);
        }

        manual_seed(42);
        using var testInput = randn(1, 16, inFeatures);
        using var output = targetModule!.call(testInput);
        output.cpu().data<float>().ToArray()
            .Should().OnlyContain(v => float.IsFinite(v), "pruned layer output must be finite");
    }

    [Fact]
    public void SmolLM2_OneLayer_TwoFourPruned_HardwarePatternSatisfied()
    {
        bool modelCached = HuggingFaceLoader.IsModelCached(ModelId);
        Assert.SkipUnless(modelCached, $"Skipped: {ModelId} not in HuggingFace cache.");

        AlgorithmsRegistration.RegisterSparseGpt();

        using var loadedModel = HuggingFaceLoader.Load(ModelId, device: CPU);
        var model = loadedModel.Model;

        const string targetWeight = "model.layers.0.self_attn.q_proj.weight";
        int inFeatures = (int)model.get_parameter(targetWeight)!.shape[1];
        int outFeatures = (int)model.get_parameter(targetWeight)!.shape[0];

        // inFeatures must be divisible by 4 for 2:4 sparsity.
        Assert.SkipUnless(
            inFeatures % 4 == 0,
            $"Skipped: inFeatures ({inFeatures}) is not divisible by 4 — cannot apply 2:4 sparsity.");

        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            [targetWeight] = model.get_parameter(targetWeight)!.detach().clone(),
        });

        var config = new SparseGptConfig
        {
            Sparsity = 0.5f,
            MaskStructure = "2:4",
            BlockSize = 128,
            Targets = new[] { targetWeight },
        };

        var targetModule = model.get_submodule("model.layers.0.self_attn.q_proj") as Module<Tensor, Tensor>;
        var modifier = ModifierRegistry.Create(config);
        modifier.Initialize(state);
        modifier.OnStart(state);

        manual_seed(0);
        for (int i = 0; i < 8; i++)
        {
            using var scope = NewDisposeScope();
            using var x = randn(2, 64, inFeatures);
            state.CurrentBatch = x;
            state.CurrentBatchIndex = i;
            modifier.OnBatch(state);
            using var _ = targetModule!.call(x);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        var weight = state.NamedWeights[targetWeight].cpu();

        // Verify 2:4 pattern: every group of 4 consecutive input columns per output row has exactly 2 zeros.
        for (int row = 0; row < outFeatures; row++)
        {
            for (int g = 0; g < inFeatures; g += 4)
            {
                int zerosInGroup = 0;
                for (int col = g; col < g + 4; col++)
                {
                    if (weight[row, col].item<float>() == 0f)
                    {
                        zerosInGroup++;
                    }
                }

                zerosInGroup.Should().Be(2, $"row {row} group [{g}..{g + 3}] must have 2 zeros");
            }
        }
    }
}
```

- [ ] **Step 2: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~SparseGptSmokeTests"
```

Expected: tests either pass (if SmolLM2-135M is in cache) or are reported as skipped with reason. No failure.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Algorithms/SparseGpt/SparseGptSmokeTests.cs
git commit -m "test(sparsegpt): add SmolLM2 real-model smoke tests"
```

---

### Task 7: Full verification

**Files:** (no file changes — verification and final check only)

- [ ] **Step 1: Clean full build + test run**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected: 0 errors, 0 warnings, **≥ 231 tests passing** (203 baseline + ~7 CholeskyHessianInverter + ~7 SparseGptConfig + ~11 mask selectors + ~5 SparseGptModifier + 0–2 smoke tests if cache is present = ~233).

- [ ] **Step 2: Verify exact-zero invariant in a dedicated check**

```powershell
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "FullyQualifiedName~SparseGptModifierTests&Category!=Gpu"
```

All 5 modifier tests must pass, including `Unstructured_50Percent_ExactlyHalfOfWeightsAreZero` (bit-exact zero assertion) and `Unstructured_50Percent_ReconstructionErrorLowerThanMagnitudePruning` (validates that Hessian-guided compensation actually improves on naïve magnitude pruning).

No commit for this task.

---

### Task 8: STOP — controller handles merge + tag

Tag will be `v0.6.0-alpha`.

---

## Self-Review Notes

**Spec coverage:**

| Requirement | Covered by |
|---|---|
| Hessian via `ActivationHookManager` | `SparseGptModifier.OnStartCore` — hooks per targeted layer |
| Dampening strategy (config knob, default 0.01) | `SparseGptConfig.DampeningFrac`; `CholeskyHessianInverter` |
| Cholesky inverse (shared with Phase 4b) | Task 3 `CholeskyHessianInverter` |
| Saliency formula `w² / (H⁻¹_jj)²` | `SparseGptModifier.OnEndCore` block iteration |
| Unstructured sparsity | `UnstructuredMaskSelector` (Task 4) |
| 2:4 N:M structured sparsity | `TwoFourMaskSelector` (Task 4) |
| Per-row selection invariant | `TwoFourMaskSelectorTests.Select_MultipleRows_EachGroupIndependentPerRow`, `UnstructuredMaskSelectorTests.Select_MultipleRows_EachRowIndependent` |
| `SparseGPTRecipe` / config knobs | `SparseGptConfig`: sparsity, mask_structure, block_size, dampening_frac, preserve_sparsity_mask, targets, ignore |
| Recipe YAML round-trip | `SparseGptConfigTests.YamlRoundTrip_*` |
| `IModifier` lifecycle integration | `SparseGptModifier : ModifierBase` |
| Registration | `AlgorithmsRegistration.RegisterSparseGpt()` |
| End-to-end synthetic test | `SparseGptModifierTests` (Task 5) |
| Exact-zero verification | `SparseGptModifierTests.Unstructured_50Percent_ExactlyHalfOfWeightsAreZero` |
| Reconstruction quality (vs magnitude baseline) | `SparseGptModifierTests.Unstructured_50Percent_ReconstructionErrorLowerThanMagnitudePruning` |
| Real-model smoke test | `SparseGptSmokeTests` (Task 6, `Assert.SkipUnless`) |

**Out of scope (document here so Phase 5 planners know where to pick up):**

- OWL (`sparsity_profile: owl`, `owl_m`, `owl_lmbda`) — per-layer sparsity derived from outlier statistics. Entirely separate calibration pass.
- Block-sparse patterns — future.
- Combined sparsity + quantization stacking (`preserve_sparsity_mask: true`) — Phase 5 recipe topic.
- `offload_hessians` (CPU offload of the Hessian matrix) — performance tuning, not correctness.
- Activation reordering (`act_order`) — GPTQ concept.

**Known risks (for model selection at execution):**

- **`SparseGptModifier.OnEndCore` inner loop numerical correctness** (high risk, mitigated): The plan now uses the canonical column-by-column pattern (`W1` mutable working copy, `Q1` accumulated pruned output, `Err1` per-column errors). The key invariants are: (a) `W1` is updated in-place per-column so each error propagates to subsequent columns within the block; (b) `Q1` (not `wBlockPruned`) is written back to `W`; (c) cross-block tail update uses `Err1.mm(HinvTail)` not `prunedError / HinvDiag`. The `ReconstructionErrorLowerThanMagnitudePruning` test in Task 5 will surface regression to a magnitude-pruning-equivalent result if the inner loop reverts to the single-snapshot pattern.
- **`CholeskyHessianInverter` Cholesky failure on small synthetic Hessians**: Synthetic calibration data with few samples produces ill-conditioned Hessians. The `blockSize=8` in the synthetic test with 16 batches of 2 rows each (32 samples for an 8-dimensional input) is borderline. Dampening `0.01f` should be sufficient; if tests flake with "not positive definite", increase to `0.05f` in the test config and document.
- **`scatter_` boolean source tensor** (Task 4): TorchSharp 0.107.0 may not support `scatter_` with a bool source tensor. The `UnstructuredMaskSelector` implementation uses `ones_like(topkIdx, dtype: ScalarType.Bool)` — if this fails, replace with float intermediary and cast.
- **Module resolution for hook registration** (Task 5 open question): The implementer must choose between extending `CompressionState` with `NamedModules` (preferred, matches the Phase 3+ architecture) or adding a `RegisterLayer` method to `SparseGptModifier`. The plan leaves this open. Whichever option chosen, the synthetic test must be updated to match.

**Commit count target:** ~9 commits across 8 tasks (including the `SparseGptConfig` stub build-fix commit if needed).

**Shared with Phase 4b — summary:**

`CholeskyHessianInverter` (Task 3) is the only file shared with Phase 4b. It lives in `src/LLMCompressorSharp.TorchExtensions/Hessian/` (same namespace as Phase 4a). If Phase 4b lands first, skip Task 3's implementation step and reuse. If Phase 4c lands first, Phase 4b's Cholesky task becomes "reuse from 4c".

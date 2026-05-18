# Phase 4b: GPTQ Algorithm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GPTQ weight quantization algorithm end-to-end. Builds directly on Phase 4a's `HessianAccumulator` + `ActivationHookManager` infrastructure. Uses Phase 1a's observer family for fake-quantization and Phase 1b's `IModifier` lifecycle for session integration. Produces a `GPTQModifier` that runs inside a standard `CompressionSession` and writes quantized (fake-quantized, still FP) weights back into `CompressionState.NamedWeights`.

**Architecture:**

```
LLMCompressorSharp.TorchExtensions/
└── Hessian/
    ├── HessianAccumulator.cs            ← EXISTING (Phase 4a)
    ├── ForwardHookHandle.cs             ← EXISTING (Phase 4a)
    ├── ActivationHookManager.cs         ← EXISTING (Phase 4a)
    └── HessianInverseSolver.cs          ← NEW  ← SHARED WITH PHASE 4c

LLMCompressorSharp.Core/
└── Algorithms/
    ├── Configs/
    │   └── GPTQConfig.cs                ← NEW
    └── Gptq/
        ├── GPTQModifier.cs              ← NEW
        └── GptqBlockQuantizer.cs        ← NEW (internal helper)

LLMCompressorSharp.Core/
└── Compression/
    └── CompressionState.cs              ← MODIFIED (add NamedModules)

tests/LLMCompressorSharp.Tests/
├── TorchExtensions/
│   └── Hessian/
│       └── HessianInverseSolverTests.cs ← NEW
└── Core/
    └── Algorithms/
        └── Gptq/
            ├── GPTQConfigTests.cs       ← NEW
            ├── GptqBlockQuantizerTests.cs ← NEW
            └── GPTQModifierTests.cs     ← NEW (lifecycle + end-to-end)
```

**How GPTQ uses Phase 4a's infrastructure:**

During `OnStart`, `GPTQModifier` creates one `HessianAccumulator` per targeted `Linear` module and registers a forward hook via `ActivationHookManager`. During `OnBatch`, the modifier calls the model's forward pass: hooks fire automatically and accumulate `H += XᵀX` for each layer. During `OnEnd`, hooks are removed, the accumulated Hessian is inverted per layer, and block-wise quantization runs with error propagation.

**Tech stack:** TorchSharp 0.107.0. Observer family from Phase 1a. `IModifier`/`ModifierBase` from Phase 1b. Block math in FP32. Fake-quantized output (FP32/FP16 with values snapped to a quantization grid) — not packed integers.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §GPTQ workflow, `docs/llm-compressor/algorithms/gptq.md`, `docs/llmcompressorsharp/algorithm-mapping.md` §GPTQ.

---

## Prerequisites & Conventions

- Phase 4a is merged. Tag `v0.4.0-alpha`. 203 tests passing on `main`.
- Branch off `main` as `feature/4b-gptq`.
- Project layout: new GPTQ code lives in `src/LLMCompressorSharp.Core/Algorithms/Gptq/` and `src/LLMCompressorSharp.Core/Algorithms/Configs/GPTQConfig.cs`, matching the RTN/SmoothQuant/WANDA convention.
- The `.editorconfig` snake_case exemption covers `src/LLMCompressorSharp.TorchExtensions/**` only. Code under `src/LLMCompressorSharp.Core/` uses standard PascalCase.
- StyleCop rules in effect: SA1402 (one type per file), SA1500 (curly braces), SA1515 (single-line comment spacing), SA1642 (constructor summary), SA1201 (element ordering), SA1312 (variable name casing), SA1117/SA1118 (parameter wrapping).
- Build: `dotnet build LLMCompressorSharp.slnx --configuration Release`
- Test: `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
- xunit.v3 1.0.0: use `Assert.SkipUnless(condition, reason)` for conditional skips — NOT `Skip.IfNot`.
- FluentAssertions for assertions. No mocks for math logic.

**TorchSharp 0.107.0 hook API (confirmed in Phase 4a):**
- `register_forward_hook` is on `HookableModule<TPreHook, TPostHook>`, inherited by `Module<Tensor, Tensor>`.
- Post-hook delegate: `Func<Module<Tensor, Tensor>, Tensor, Tensor, Tensor>` — `(module, input, output)` returning output.
- Hooks fire via `.call()`, not `.forward()`. Tests must invoke `.call()` to trigger hooks.
- Returns `HookRemover` with `.remove()`. Wrapped in `ForwardHookHandle`.
- `0 × -inf = NaN` trap: use `torch.where(boolMask, neginf, zeros)` not `triu(ones).mul(-inf)` — already encoded in Phase 4a's `LlamaAttention`.

**Damping convention:** GPTQ paper uses λ = 0.01 × mean(diag(H)). llm-compressor exposes this as `dampening_frac` (default 0.01). The plan uses the same name. Implementers may tune it down to `0.001` for well-conditioned Hessians; up to `0.1` if Cholesky fails on small calibration sets.

---

## File Structure

```
src/LLMCompressorSharp.TorchExtensions/Hessian/
├── HessianAccumulator.cs               ← existing
├── ForwardHookHandle.cs                ← existing
├── ActivationHookManager.cs            ← existing
└── HessianInverseSolver.cs             ← NEW

src/LLMCompressorSharp.Core/
├── Compression/
│   └── CompressionState.cs             ← MODIFIED (add NamedModules)
└── Algorithms/
    ├── Configs/
    │   └── GPTQConfig.cs               ← NEW
    ├── Gptq/
    │   ├── GptqBlockQuantizer.cs       ← NEW (internal static helper)
    │   └── GPTQModifier.cs             ← NEW
    └── AlgorithmsRegistration.cs       ← MODIFIED (add RegisterGptq)

tests/LLMCompressorSharp.Tests/
├── TorchExtensions/Hessian/
│   └── HessianInverseSolverTests.cs    ← NEW
└── Core/Algorithms/Gptq/
    ├── GPTQConfigTests.cs              ← NEW
    ├── GptqBlockQuantizerTests.cs      ← NEW
    └── GPTQModifierTests.cs            ← NEW
```

**Responsibility per file:**

- `HessianInverseSolver` — Static (or sealed) class. `Compute(Tensor H, float dampingFrac)` → `Tensor Hinv`. Applies damping in-place to a clone of `H`, runs Cholesky, inverts via triangular solve, falls back to `linalg.pinv` on failure. **Shared with Phase 4c** (SparseGPT reuses this to obtain Hinv; only the inner per-column step differs).

- `GPTQConfig` — `ModifierConfig` subclass. Knobs: `Scheme`, `NumBits`, `Symmetric`, `Strategy`, `ChannelAxis`, `BlockSize`, `DampeningFrac`, plus inherited `Targets` / `Ignore`.

- `GptqBlockQuantizer` — Internal static helper. `Quantize(Tensor W, Tensor Hinv, GPTQConfig config) → Tensor`. Implements the block-wise column loop with error propagation. No modifier lifecycle awareness — purely a function of (W, Hinv, config).

- `GPTQModifier` — `ModifierBase` subclass. Wires the lifecycle: `OnInitialize` creates accumulators; `OnStartCore` registers hooks via `ActivationHookManager`; `OnBatchCore` drives the forward pass; `OnEndCore` disposes hooks, inverts Hessians, calls `GptqBlockQuantizer.Quantize`, writes results; `OnFinalizeCore` disposes all tensors.

**Out of scope:**
- `actorder` (column reordering by descending `diag(H)`) — deferred. Flag in config as `bool ActOrder = false` with `NotSupportedException` if set to true.
- Per-group quantization beyond what the existing `Observer` family provides — the W4A16 default (per-channel at the output-channel axis) covers the common case.
- `offload_hessians` (CPU offload during calibration) — deferred to v0.5.x. `HessianAccumulator` can be moved `.cpu()` manually if needed.
- Packed-integer output (true INT4 storage) — Phase 5 / `Int4PackedTensor`.

---

### Task 1: Branch + Probe (Cholesky, solve_triangular, slice assignment)

**Files:** (no file changes — research only)

- [ ] **Step 1: Branch + baseline**

```powershell
git status --short
git log --oneline -1
git checkout -b feature/4b-gptq
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"
```

Expected: 203 tests passing.

- [ ] **Step 2: Probe TorchSharp linalg API**

The Cholesky + triangular-solve path is the most likely fragile point. Verify all four operations before writing implementation code.

```powershell
# Write a disposable probe project to verify the exact API surface.
# Run this interactively in a scratch file or via dotnet-script.

# Expected shapes:
#   H:     [4, 4] symmetric positive-definite
#   L:     [4, 4] lower-triangular factor
#   Linv:  [4, 4] inverse of L
#   Hinv:  [4, 4] = Linv.t().mm(Linv)
```

```csharp
// Probe file: verify in a scratch test method or dotnet-script
using TorchSharp;
using static TorchSharp.torch;

// --- Probe 1: linalg.cholesky ---
// H must be symmetric positive-definite.
// Create a random SPD matrix: A = M^T M + epsilon*I
using var m = randn(4, 4);
using var h = m.t().mm(m).add_(eye(4).mul(0.1f));
using var L = linalg.cholesky(h);        // does this compile + run? Upper/lower?
Console.WriteLine($"cholesky OK: L shape [{string.Join(",", L.shape)}]");

// --- Probe 2: linalg.solve_triangular ---
// Solve L X = I for X (lower triangular system, upper=false).
// Exact parameter name: 'upper' bool (keyword argument in C# binding).
using var eye4 = eye(4);
using var Linv = linalg.solve_triangular(L, eye4, upper: false);
Console.WriteLine($"solve_triangular OK: Linv shape [{string.Join(",", Linv.shape)}]");

// --- Probe 3: linalg.inv and linalg.pinv (fallback paths) ---
using var Hinv_direct = linalg.inv(h);
Console.WriteLine($"linalg.inv OK");
using var Hinv_pinv = linalg.pinv(h);
Console.WriteLine($"linalg.pinv OK");

// --- Probe 4: range-indexed slice assignment ---
// W: [outFeatures=3, inFeatures=6]
using var W = randn(3, 6);
using var Wq = W.clone();
// Assign one column:
using var col_q = W[TensorIndex.Colon, TensorIndex.Single(0)].clone();
Wq[TensorIndex.Colon, TensorIndex.Single(0)] = col_q;   // test single-column write-back
Console.WriteLine("single-col assign OK");
// Assign a range of columns:
using var block = W[TensorIndex.Colon, TensorIndex.Slice(0, 3)].clone();
Wq[TensorIndex.Colon, TensorIndex.Slice(0, 3)] = block;
Console.WriteLine("block-col range assign OK");
// In-place subtract from a range:
using var delta = randn(3, 3);
W.narrow(1, 3, 3).sub_(delta);  // or W[:, 3:] -= delta equivalent
Console.WriteLine("narrow().sub_() OK");
```

Key things to confirm and adapt if needed:

| Operation | What to verify | Adaptation if wrong |
|---|---|---|
| `linalg.cholesky(H)` | Namespace `torch.linalg.cholesky` vs `torch.linalg().cholesky()` | Try `torch.linalg.cholesky(h)` static call |
| `linalg.solve_triangular(L, B, upper: false)` | Exact keyword name `upper` vs `is_upper` | Also try positional `solve_triangular(L, B, false)` |
| `linalg.inv` | Namespace as above | Same adaptation |
| `linalg.pinv` | Available in 0.107.0? | Fallback: compute via `svd` + threshold |
| Slice assignment `W[..., Slice(i, j)] = tensor` | Does indexed assignment compile + run? | Fall back to `narrow` + `copy_` |
| `narrow(dim, start, length).sub_(...)` | In-place op on a narrow view | Standard TorchSharp — should work |

> **If `linalg.solve_triangular` is missing or its signature differs:** use `linalg.inv(L)` directly. This is slightly less numerically stable but equivalent for well-conditioned matrices. Document the substitution with a `// TODO(PR_TO_TORCHSHARP)` comment.

No commit for this task.

---

### Task 2: TDD `GPTQConfig` and `CompressionState.NamedModules`

**Files:**
- Modify: `src/LLMCompressorSharp.Core/Compression/CompressionState.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Configs/GPTQConfig.cs`
- Create: `tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQConfigTests.cs`

GPTQModifier must install forward hooks on `Module<Tensor, Tensor>` instances. The existing `CompressionState` only exposes `NamedWeights` (a `Dictionary<string, Tensor>`). Adding a `NamedModules` property (null by default) keeps existing tests unchanged and makes the module dictionary available to hook-based modifiers.

- [ ] **Step 1: Write the failing config + state tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.TorchExtensions.Observers;
using Xunit;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GPTQConfig"/> defaults and validation.
/// </summary>
public class GPTQConfigTests
{
    [Fact]
    public void Defaults_AreReasonable()
    {
        var config = new GPTQConfig();
        config.Type.Should().Be("GPTQ");
        config.NumBits.Should().Be(4);
        config.Symmetric.Should().BeTrue();
        config.Strategy.Should().Be(QuantizationStrategy.PerChannel);
        config.ChannelAxis.Should().Be(0);
        config.BlockSize.Should().Be(128);
        config.DampeningFrac.Should().BeApproximately(0.01f, 1e-6f);
        config.ActOrder.Should().BeFalse();
        config.Targets.Should().BeNull();
        config.Ignore.Should().BeNull();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(17)]
    public void NumBits_OutOfRange_IsInvalid(int bits)
    {
        // The modifier (not the config) enforces this at quantization time.
        // Verify config itself doesn't validate — it is just a data bag.
        var config = new GPTQConfig { NumBits = bits };
        config.NumBits.Should().Be(bits); // no throwing in the config
    }

    [Fact]
    public void ActOrder_True_IsAccepted_InConfig()
    {
        // Config accepts it; the modifier throws NotSupportedException if true.
        var config = new GPTQConfig { ActOrder = true };
        config.ActOrder.Should().BeTrue();
    }

    [Fact]
    public void BlockSize_IsConfigurable()
    {
        var config = new GPTQConfig { BlockSize = 64 };
        config.BlockSize.Should().Be(64);
    }
}
```

- [ ] **Step 2: Verify build fails (GPTQConfig not found)**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~GPTQConfigTests"
```

Expected: BUILD FAILS (CS0246).

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQConfigTests.cs
git commit -m "test(gptq): add failing GPTQConfig tests"
```

- [ ] **Step 4: Implement `GPTQConfig`**

Create `src/LLMCompressorSharp.Core/Algorithms/Configs/GPTQConfig.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Recipes;
using LLMCompressorSharp.TorchExtensions.Observers;

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// Configuration for <c>GPTQModifier</c> — Hessian-based block-wise weight quantization.
/// </summary>
/// <remarks>
/// Corresponds to the Python <c>GPTQModifier</c> knobs in llm-compressor.
/// The default scheme is W4A16 (4-bit weights, 16-bit activations):
/// <c>NumBits=4, Symmetric=true, Strategy=PerChannel, BlockSize=128, DampeningFrac=0.01</c>.
/// </remarks>
public sealed class GPTQConfig : ModifierConfig
{
    /// <inheritdoc />
    public override string Type => "GPTQ";

    /// <summary>Gets or sets the target bit-width. Default: 4.</summary>
    public int NumBits { get; set; } = 4;

    /// <summary>Gets or sets a value indicating whether symmetric quantization is used. Default: true.</summary>
    /// <remarks>Symmetric = zero-point is always 0; range is [-2^(b-1)+1, 2^(b-1)-1].</remarks>
    public bool Symmetric { get; set; } = true;

    /// <summary>Gets or sets the quantization granularity. Default: <see cref="QuantizationStrategy.PerChannel"/>.</summary>
    /// <remarks>
    /// Per-channel at axis 0 means one scale/zero-point per output channel of the Linear weight
    /// (<c>[outFeatures, inFeatures]</c>). This is the W4A16 standard.
    /// </remarks>
    public QuantizationStrategy Strategy { get; set; } = QuantizationStrategy.PerChannel;

    /// <summary>Gets or sets the channel axis for per-channel quantization. Default: 0 (output channels of Linear.weight).</summary>
    public int ChannelAxis { get; set; } = 0;

    /// <summary>Gets or sets the number of weight columns processed per block iteration. Default: 128.</summary>
    public int BlockSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the Hessian diagonal dampening fraction. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// Applied as <c>H += dampening_frac × mean(diag(H)) × I</c> before Cholesky.
    /// The GPTQ paper uses 1%; increase to 0.1 for small calibration sets or poorly
    /// conditioned layers. Has no effect when the Hessian is already well-conditioned.
    /// </remarks>
    public float DampeningFrac { get; set; } = 0.01f;

    /// <summary>
    /// Gets or sets a value indicating whether to reorder columns by descending Hessian diagonal
    /// before quantization (activation ordering). Default: false.
    /// </summary>
    /// <remarks>
    /// Not yet implemented. Setting this to <see langword="true"/> causes <c>GPTQModifier</c>
    /// to throw <see cref="NotSupportedException"/>.
    /// </remarks>
    public bool ActOrder { get; set; } = false;
}
```

- [ ] **Step 5: Extend `CompressionState` with `NamedModules`**

Add a single nullable property to `CompressionState` (no other changes):

```csharp
/// <summary>
/// Gets or sets the named module dictionary used by hook-based modifiers (e.g. GPTQ, SparseGPT).
/// Keys match those in <see cref="NamedWeights"/>, with the <c>.weight</c> suffix stripped.
/// When null, hook-based modifiers are not supported.
/// </summary>
/// <remarks>
/// Phase 4b populates this from the LLaMA module hierarchy via
/// <c>model.named_modules()</c> filtered to <c>Linear</c> layers.
/// </remarks>
public IDictionary<string, Module<Tensor, Tensor>>? NamedModules { get; set; }
```

- [ ] **Step 6: Run tests and commit**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~GPTQConfigTests"
```

Expected: 4 tests pass. All 203 prior tests still pass.

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Configs/GPTQConfig.cs
git add src/LLMCompressorSharp.Core/Compression/CompressionState.cs
git commit -m "feat(gptq): add GPTQConfig and NamedModules extension to CompressionState"
```

---

### Task 3: TDD `HessianInverseSolver`

> **Shared with Phase 4c (SparseGPT):** `HessianInverseSolver` lives in `TorchExtensions/Hessian/` and encapsulates damping + Cholesky inversion in one place. SparseGPT will import it directly — the only difference between GPTQ and SparseGPT is what happens column-by-column after Hinv is computed.

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/HessianInverseSolverTests.cs`
- Create: `src/LLMCompressorSharp.TorchExtensions/Hessian/HessianInverseSolver.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="HessianInverseSolver"/> — damped Cholesky inversion.
/// </summary>
public class HessianInverseSolverTests
{
    [Fact]
    public void Compute_IdentityMatrix_ReturnsIdentity()
    {
        // Hinv of identity (already perfectly conditioned) should be identity.
        using var H = eye(4, dtype: ScalarType.Float32);
        using var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.0f);

        // Hinv should be close to identity.
        using var expected = eye(4, dtype: ScalarType.Float32);
        using var diff = Hinv.sub(expected).abs();
        diff.max().item<float>().Should().BeLessThan(1e-5f);
    }

    [Fact]
    public void Compute_ScaledIdentity_ReturnsInverseScaledIdentity()
    {
        // H = 2*I → Hinv = 0.5*I
        using var H = eye(3, dtype: ScalarType.Float32).mul(2f);
        using var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.0f);

        using var expected = eye(3, dtype: ScalarType.Float32).mul(0.5f);
        using var diff = Hinv.sub(expected).abs();
        diff.max().item<float>().Should().BeLessThan(1e-5f);
    }

    [Fact]
    public void Compute_WithDamping_ProducesFiniteOutput()
    {
        // A rank-deficient matrix (not invertible without damping).
        // [[1, 1], [1, 1]] has rank 1. Damping regularises it.
        using var H = ones(2, 2, dtype: ScalarType.Float32);
        using var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.01f);

        // With damping, result should be finite.
        Hinv.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Compute_IllConditionedMatrix_FallsBackToPinv()
    {
        // A very ill-conditioned matrix that even with mild damping may fail Cholesky.
        // Use a near-zero diagonal to create a challenging case.
        using var H = eye(4, dtype: ScalarType.Float32).mul(1e-10f);
        using var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.0f);

        // Pseudoinverse fallback should produce finite output.
        Hinv.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Compute_NullHessian_Throws()
    {
        var act = () => HessianInverseSolver.Compute(null!, dampingFrac: 0.01f);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Compute_NonSquareHessian_Throws()
    {
        using var H = ones(3, 4, dtype: ScalarType.Float32);
        var act = () => HessianInverseSolver.Compute(H, dampingFrac: 0.01f);
        act.Should().Throw<ArgumentException>().WithMessage("*square*");
    }

    [Fact]
    public void Compute_OutputIsFloat32()
    {
        // Even if Hinv diverges in precision, it should remain float32.
        using var H = eye(3, dtype: ScalarType.Float32);
        using var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.01f);
        Hinv.dtype.Should().Be(ScalarType.Float32);
    }

    [Fact]
    public void Compute_OutputDoesNotAliasInput()
    {
        // Caller must be able to dispose H without affecting Hinv.
        using var H = eye(3, dtype: ScalarType.Float32);
        var Hinv = HessianInverseSolver.Compute(H, dampingFrac: 0.01f);
        H.Dispose();

        // Hinv should still be readable.
        Hinv.isfinite().all().item<bool>().Should().BeTrue();
        Hinv.Dispose();
    }
}
```

- [ ] **Step 2: Verify build fails, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/HessianInverseSolverTests.cs
git commit -m "test(torch-extensions): add failing HessianInverseSolver tests"
```

- [ ] **Step 3: Implement `HessianInverseSolver`**

> **Task 1 probe result required:** confirm whether the `upper` keyword name in `linalg.solve_triangular` and whether `linalg.pinv` is available before implementing. Adapt the code below accordingly.

Create `src/LLMCompressorSharp.TorchExtensions/Hessian/HessianInverseSolver.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Computes the inverse Hessian used by GPTQ and SparseGPT via damped Cholesky decomposition.
/// </summary>
/// <remarks>
/// The computation is:
/// <list type="number">
///   <item>Apply diagonal dampening: <c>H' = H + λ·mean(diag(H))·I</c>.</item>
///   <item>Cholesky: <c>L = chol(H')</c> (lower-triangular factor).</item>
///   <item>Invert: <c>L⁻¹ = solve_triangular(L, I, upper=false)</c>.</item>
///   <item>Return: <c>H⁻¹ = L⁻ᵀ · L⁻¹</c>.</item>
/// </list>
/// If Cholesky fails (non-positive-definite after damping), falls back to <c>linalg.pinv</c>.
/// </remarks>
/// <remarks>
/// <b>Shared with Phase 4c (SparseGPT).</b> SparseGPT uses the same Hinv but replaces
/// the fake-quantize inner step with a mask-selection step.
/// </remarks>
public static class HessianInverseSolver
{
    /// <summary>
    /// Computes <c>H⁻¹</c> with diagonal damping for numerical stability.
    /// </summary>
    /// <param name="H">
    /// The accumulated Hessian <c>[d, d]</c>. Must be a square FP32 tensor.
    /// The input tensor is not modified — a working copy is used.
    /// </param>
    /// <param name="dampingFrac">
    /// Fraction of the mean diagonal to add as damping. Typical value: 0.01.
    /// Pass 0 to skip damping (only safe for perfectly conditioned matrices).
    /// </param>
    /// <returns>
    /// The inverse Hessian <c>[d, d]</c> (FP32). The caller owns disposal.
    /// </returns>
    /// <exception cref="ArgumentNullException">If <paramref name="H"/> is null.</exception>
    /// <exception cref="ArgumentException">If <paramref name="H"/> is not a square 2-D tensor.</exception>
    public static Tensor Compute(Tensor H, float dampingFrac)
    {
        ArgumentNullException.ThrowIfNull(H);

        if (H.ndim != 2 || H.shape[0] != H.shape[1])
        {
            throw new ArgumentException(
                $"Hessian must be a square 2-D tensor; got shape [{string.Join(", ", H.shape)}].",
                nameof(H));
        }

        var d = H.shape[0];

        // Work on a FP32 clone — never modify the caller's tensor.
        using var hWork = H.to(ScalarType.Float32).detach().clone();

        // Apply diagonal damping: H += dampingFrac * mean(diag(H)) * I
        if (dampingFrac > 0f)
        {
            var diagMean = hWork.diagonal().mean().item<float>();
            var lambda = dampingFrac * diagMean;

            // Add lambda to each diagonal element in-place.
            using var diagView = hWork.diagonal();
            diagView.add_(lambda);
        }

        // Cholesky decomposition, with pseudoinverse fallback.
        try
        {
            using var L = linalg.cholesky(hWork);

            // Invert L via triangular solve: L * Linv = I → Linv = solve_triangular(L, I, upper=false).
            // NOTE (Task 1 probe): if `upper` keyword name differs, adapt here.
            using var eye_d = eye(d, dtype: ScalarType.Float32);
            using var Linv = linalg.solve_triangular(L, eye_d, upper: false);

            // H^-1 = Linv^T * Linv  (upper-triangular representation)
            return Linv.t().mm(Linv);
        }
        catch (Exception ex) when (IsCholeskyFailure(ex))
        {
            // Cholesky failed despite damping — fall back to pseudoinverse.
            // This is slower but handles pathologically ill-conditioned Hessians
            // (e.g. very small calibration sets, or layers that saw almost no activations).
            // TODO(PR_TO_TORCHSHARP): If linalg.pinv is missing, use SVD-based pseudoinverse.
            return linalg.pinv(hWork);
        }
    }

    /// <summary>
    /// Heuristically identifies Cholesky decomposition failures from TorchSharp exception messages.
    /// </summary>
    /// <param name="ex">The exception thrown by <c>linalg.cholesky</c>.</param>
    /// <returns><see langword="true"/> if the exception looks like a Cholesky failure.</returns>
    private static bool IsCholeskyFailure(Exception ex)
    {
        // TorchSharp wraps LibTorch errors in ExternalException or RuntimeError.
        // The message typically contains "not positive definite" or "singular".
        var msg = ex.Message;
        return msg.Contains("positive definite", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("singular", StringComparison.OrdinalIgnoreCase)
            || ex is System.Runtime.InteropServices.ExternalException;
    }
}
```

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~HessianInverseSolverTests"
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/Hessian/HessianInverseSolver.cs
git commit -m "feat(torch-extensions): implement HessianInverseSolver (shared with Phase 4c)"
```

---

### Task 4: TDD `GptqBlockQuantizer` (core algorithm)

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GptqBlockQuantizerTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Gptq/GptqBlockQuantizer.cs`

This is the numerically critical task. `GptqBlockQuantizer.Quantize` implements the GPTQ column loop: iterate columns in blocks, fake-quantize each column, compute the per-column error, propagate error to remaining columns using the inverse-Hessian row, then propagate accumulated block error to remaining blocks.

> **Cross-check warning:** The pseudocode in `docs/llmcompressorsharp/algorithm-mapping.md` is an approximation. Before implementing, cross-check the exact error-propagation formula against the upstream Python source at `src/llmcompressor/modifiers/gptq/gptq_quantize.py`. The canonical update per column `j` is:
>
> ```
> err_j  = (w_j - q_j) / Hinv[j, j]
> W[:, j+1:] -= err_j.unsqueeze(1) * Hinv[j, j+1:]
> ```
>
> A sign error or indexing off-by-one produces output that is finite but wrong. The hand-verified 3×3 test below is the sanity check.

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GptqBlockQuantizer"/> — block-wise column quantization with error propagation.
/// </summary>
public class GptqBlockQuantizerTests
{
    private static GPTQConfig MakeConfig(int numBits = 8, int blockSize = 128) =>
        new GPTQConfig
        {
            NumBits = numBits,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
            BlockSize = blockSize,
            DampeningFrac = 0.01f,
        };

    [Fact]
    public void Quantize_ZeroWeight_ReturnsZeroWeight()
    {
        // W = 0, H = I → Hinv = I. Quantized zero is zero.
        var config = MakeConfig(numBits: 8);
        using var W = zeros(2, 4, dtype: ScalarType.Float32);
        using var Hinv = eye(4, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        var arr = Wq.cpu().data<float>().ToArray();
        arr.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-6f));
    }

    [Fact]
    public void Quantize_OutputIsFinite()
    {
        // Random W, identity Hinv, 4-bit. Result must be finite.
        var config = MakeConfig(numBits: 4);
        using var W = randn(4, 8, dtype: ScalarType.Float32);
        using var Hinv = eye(8, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        Wq.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Quantize_OutputShape_MatchesInput()
    {
        var config = MakeConfig(numBits: 4, blockSize = 2);
        using var W = randn(3, 6, dtype: ScalarType.Float32);
        using var Hinv = eye(6, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        Wq.shape.Should().Equal(W.shape);
    }

    [Fact]
    public void Quantize_HighBitWidth_CloseToOriginal()
    {
        // At 16-bit, quantized values should be nearly identical to the originals.
        var config = MakeConfig(numBits: 16, blockSize = 4);
        using var W = randn(2, 4, dtype: ScalarType.Float32);
        using var Hinv = eye(4, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        using var diff = (W - Wq).abs();
        diff.max().item<float>().Should().BeLessThan(1e-3f);
    }

    [Fact]
    public void Quantize_BlockSizeLargerThanColumns_HandledGracefully()
    {
        // BlockSize=128 with only 6 columns: single block containing all columns.
        var config = MakeConfig(numBits: 8, blockSize = 128);
        using var W = randn(2, 6, dtype: ScalarType.Float32);
        using var Hinv = eye(6, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        Wq.isfinite().all().item<bool>().Should().BeTrue();
        Wq.shape.Should().Equal(W.shape);
    }

    [Fact]
    public void Quantize_NullW_Throws()
    {
        using var Hinv = eye(4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(null!, Hinv, MakeConfig());
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Quantize_NullHinv_Throws()
    {
        using var W = randn(2, 4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(W, null!, MakeConfig());
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Quantize_NullConfig_Throws()
    {
        using var W = randn(2, 4, dtype: ScalarType.Float32);
        using var Hinv = eye(4, dtype: ScalarType.Float32);
        var act = () => GptqBlockQuantizer.Quantize(W, Hinv, null!);
        act.Should().Throw<ArgumentNullException>();
    }

    /// <summary>
    /// Hand-verified 2×2 test that confirms error propagation direction and magnitude.
    /// Build this test first; if it passes with a naive implementation but fails after
    /// adding error propagation, the propagation sign or formula is wrong.
    /// </summary>
    [Fact]
    public void Quantize_TwoByTwo_ErrorPropagation_IsCorrectDirection()
    {
        // W = [[1, 2], [3, 4]], Hinv = identity (no cross-column error coupling).
        // With identity Hinv, quantizing column 0 produces no change in column 1.
        // At high bit width the quantized result should match the original.
        var config = MakeConfig(numBits: 16, blockSize = 1);
        using var W = tensor(new float[,] { { 1f, 2f }, { 3f, 4f } });
        using var Hinv = eye(2, dtype: ScalarType.Float32);

        using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, config);

        var arr = Wq.cpu().data<float>().ToArray();
        // With identity Hinv and 16-bit precision, result must be within 0.01 of original.
        arr[0].Should().BeApproximately(1f, 0.01f);
        arr[1].Should().BeApproximately(2f, 0.01f);
        arr[2].Should().BeApproximately(3f, 0.01f);
        arr[3].Should().BeApproximately(4f, 0.01f);
    }
}
```

- [ ] **Step 2: Verify build fails, commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GptqBlockQuantizerTests.cs
git commit -m "test(gptq): add failing GptqBlockQuantizer tests"
```

- [ ] **Step 3: Implement `GptqBlockQuantizer`**

Create `src/LLMCompressorSharp.Core/Algorithms/Gptq/GptqBlockQuantizer.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.TorchExtensions.Observers;
using LLMCompressorSharp.TorchExtensions.Quantization;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Gptq;

/// <summary>
/// Static helper that applies GPTQ block-wise column quantization with error propagation.
/// </summary>
/// <remarks>
/// This class is internal to the GPTQ algorithm; use <see cref="GPTQModifier"/> for full
/// integration with the compression session lifecycle.
///
/// <para>Algorithm (per layer):</para>
/// <list type="number">
///   <item>For each block of <c>block_size</c> columns:</item>
///   <item>For each column <c>j</c> in the block: fake-quantize <c>w_j</c>, compute
///         <c>err_j = (w_j − q_j) / Hinv[j, j]</c>, subtract
///         <c>err_j ⊗ Hinv[j, j+1..end]</c> from remaining columns in the block.</item>
///   <item>After the block: propagate the accumulated block error to all remaining columns
///         via <c>W[:, end:] -= (W[:, i..end] − Wq[:, i..end]) * Hinv[i..end, end:]</c>.</item>
/// </list>
///
/// <para>
/// <b>Implementation note:</b> the exact error formula above must be verified against
/// the upstream Python source (<c>src/llmcompressor/modifiers/gptq/gptq_quantize.py</c>)
/// before submitting. The sketch in <c>docs/llmcompressorsharp/algorithm-mapping.md</c>
/// is approximate. The end-to-end test in <c>GPTQModifierTests</c> will catch sign errors.
/// </para>
/// </remarks>
internal static class GptqBlockQuantizer
{
    /// <summary>
    /// Quantizes weight matrix <paramref name="W"/> using the GPTQ column loop.
    /// </summary>
    /// <param name="W">
    /// The weight matrix <c>[outFeatures, inFeatures]</c>. Not modified — a working copy is used.
    /// </param>
    /// <param name="Hinv">
    /// The inverse Hessian <c>[inFeatures, inFeatures]</c>, as returned by
    /// <see cref="LLMCompressorSharp.TorchExtensions.Hessian.HessianInverseSolver.Compute"/>.
    /// </param>
    /// <param name="config">The GPTQ configuration.</param>
    /// <returns>
    /// The fake-quantized weight matrix <c>[outFeatures, inFeatures]</c> (FP32).
    /// The caller owns disposal.
    /// </returns>
    /// <exception cref="ArgumentNullException">If any argument is null.</exception>
    public static Tensor Quantize(Tensor W, Tensor Hinv, GPTQConfig config)
    {
        ArgumentNullException.ThrowIfNull(W);
        ArgumentNullException.ThrowIfNull(Hinv);
        ArgumentNullException.ThrowIfNull(config);

        if (config.ActOrder)
        {
            throw new NotSupportedException(
                "GPTQConfig.ActOrder=true is not yet implemented. " +
                "Column reordering by Hessian diagonal is deferred to a future release.");
        }

        var outFeatures = (int)W.shape[0];
        var inFeatures = (int)W.shape[1];
        var blockSize = config.BlockSize;

        // Working copy in FP32 — modifies are in-place on this copy.
        using var wWork = W.to(ScalarType.Float32).detach().clone();
        var Wq = zeros_like(wWork);

        // Pre-compute per-tensor or per-channel quantization scale from the original weight.
        // This mirrors RtnModifier's observer pattern: we observe the full original weight
        // before the loop, so scale/zero-point are consistent across all blocks.
        var (scale, zeroPoint) = ComputeScaleZeroPoint(W, config);

        for (int blockStart = 0; blockStart < inFeatures; blockStart += blockSize)
        {
            using var blockScope = NewDisposeScope();
            int blockEnd = Math.Min(blockStart + blockSize, inFeatures);
            int actualBlockSize = blockEnd - blockStart;

            // Slices into the working weight and Hinv for this block.
            // NOTE (Task 1 probe): verify TensorIndex.Slice(start, end) vs i..end syntax.
            var wBlock = wWork.narrow(1, blockStart, actualBlockSize);  // [out, blockSize]
            var wqBlock = Wq.narrow(1, blockStart, actualBlockSize);    // [out, blockSize]
            var hinvBlock = Hinv.narrow(0, blockStart, actualBlockSize)
                               .narrow(1, blockStart, actualBlockSize); // [blockSize, blockSize]

            // Column-by-column quantization within the block.
            for (int j = 0; j < actualBlockSize; j++)
            {
                using var colScope = NewDisposeScope();
                var absJ = blockStart + j;

                // Extract weight column and quantize it.
                using var wCol = wWork.narrow(1, absJ, 1).squeeze(1);  // [out]
                using var wColQ = FakeQuantizeColumn(wCol, scale, zeroPoint, absJ, outFeatures, config);
                wqBlock.narrow(1, j, 1).squeeze(1).copy_(wColQ);

                // Compute per-output error: err = (w - q) / Hinv[j, j]
                var hinvJJ = Hinv[TensorIndex.Single(absJ), TensorIndex.Single(absJ)].item<float>();
                if (MathF.Abs(hinvJJ) < 1e-8f)
                {
                    // Degenerate column (zero Hessian diagonal) — skip error propagation.
                    continue;
                }

                using var err = (wCol - wColQ).div(hinvJJ);  // [out]

                // Propagate to remaining columns in this block.
                int remaining = actualBlockSize - j - 1;
                if (remaining > 0)
                {
                    // hinvRow: [remaining] = Hinv[absJ, absJ+1..blockEnd]
                    using var hinvRow = Hinv.narrow(1, absJ + 1, remaining)
                                           .narrow(0, absJ, 1).squeeze(0);  // [remaining]

                    // wBlock[:, j+1:] -= err.unsqueeze(1) * hinvRow.unsqueeze(0)
                    using var errOuter = err.unsqueeze(1).mm(hinvRow.unsqueeze(0));  // [out, remaining]
                    wBlock.narrow(1, j + 1, remaining).sub_(errOuter);
                }
            }

            // Propagate accumulated block error to remaining blocks.
            int tailSize = inFeatures - blockEnd;
            if (tailSize > 0)
            {
                // blockErr = wWork[:, blockStart..blockEnd] - Wq[:, blockStart..blockEnd]
                using var blockErr = wWork.narrow(1, blockStart, actualBlockSize)
                                         .sub(Wq.narrow(1, blockStart, actualBlockSize));  // [out, blockSize]

                // hinvTail: Hinv[blockStart..blockEnd, blockEnd..]  [blockSize, tail]
                using var hinvTail = Hinv.narrow(0, blockStart, actualBlockSize)
                                         .narrow(1, blockEnd, tailSize);

                // wWork[:, blockEnd:] -= blockErr @ hinvTail
                using var correction = blockErr.mm(hinvTail);  // [out, tail]
                wWork.narrow(1, blockEnd, tailSize).sub_(correction);
            }
        }

        // Free per-column scale/zero-point tensors.
        DisposeScaleZeroPoint(scale, zeroPoint, config);

        return Wq.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Computes scale and zero-point for the entire weight matrix, pre-loop.
    /// Per-tensor: returns (float, long) scalars wrapped in a 1-element array for uniformity.
    /// Per-channel: returns an array with one (float, long) per output channel.
    /// </summary>
    private static (float[] Scales, long[] ZeroPoints) ComputeScaleZeroPoint(
        Tensor W,
        GPTQConfig config)
    {
        if (config.Strategy == QuantizationStrategy.PerTensor)
        {
            var obs = new MinMaxObserver();
            obs.Update(W);
            var qp = obs.GetQuantParams(config.NumBits, config.Symmetric);
            var s = new float[] { qp.Scale.item<float>() };
            var z = new long[] { qp.ZeroPoint.item<long>() };
            qp.Scale.Dispose();
            qp.ZeroPoint.Dispose();
            return (s, z);
        }
        else if (config.Strategy == QuantizationStrategy.PerChannel)
        {
            var channelCount = (int)W.shape[config.ChannelAxis];
            var scales = new float[channelCount];
            var zeroPoints = new long[channelCount];
            for (int c = 0; c < channelCount; c++)
            {
                using var slice = W.select(config.ChannelAxis, c).contiguous();
                var obs = new MinMaxObserver();
                obs.Update(slice);
                var qp = obs.GetQuantParams(config.NumBits, config.Symmetric);
                scales[c] = qp.Scale.item<float>();
                zeroPoints[c] = qp.ZeroPoint.item<long>();
                qp.Scale.Dispose();
                qp.ZeroPoint.Dispose();
            }

            return (scales, zeroPoints);
        }
        else
        {
            throw new NotSupportedException(
                $"Quantization strategy '{config.Strategy}' is not supported by GptqBlockQuantizer.");
        }
    }

    private static void DisposeScaleZeroPoint(float[] scales, long[] zeroPoints, GPTQConfig config)
    {
        // Scale/zeroPoint here are plain arrays, no disposal needed. Method is a no-op placeholder
        // in case we switch to Tensor-based scale arrays in a future revision.
        _ = scales;
        _ = zeroPoints;
        _ = config;
    }

    /// <summary>
    /// Fake-quantizes a single weight column using the pre-computed scale and zero-point
    /// for the row (output-channel) or the whole tensor, depending on the strategy.
    /// </summary>
    private static Tensor FakeQuantizeColumn(
        Tensor wCol,
        float[] scales,
        long[] zeroPoints,
        int columnIndex,
        int outFeatures,
        GPTQConfig config)
    {
        _ = columnIndex; // not used for per-tensor; used indirectly via output-channel axis for per-channel

        if (config.Strategy == QuantizationStrategy.PerTensor)
        {
            return FakeQuantize.Apply(wCol, scales[0], zeroPoints[0], config.NumBits, config.Symmetric);
        }
        else
        {
            // Per-channel (axis=0 = output channel): each row of wCol uses its own scale.
            // wCol shape: [outFeatures]. Apply per-element with the per-output-channel params.
            var parts = new List<Tensor>(outFeatures);
            for (int row = 0; row < outFeatures; row++)
            {
                using var elem = wCol[TensorIndex.Single(row)].unsqueeze(0);
                parts.Add(FakeQuantize.Apply(elem, scales[row], zeroPoints[row], config.NumBits, config.Symmetric));
            }

            try
            {
                return cat(parts.ToArray(), dim: 0);
            }
            finally
            {
                foreach (var p in parts) { p.Dispose(); }
            }
        }
    }
}
```

> **Performance note (P3 from pitfalls.md):** The per-channel per-column inner loop above is `O(outFeatures × inFeatures)` individual element operations — slow for large models. This is acceptable for Phase 4b (correctness first). A vectorised alternative using `torch.vmap` or batch operations over the column can be introduced as a follow-up once the algorithm is verified correct.

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~GptqBlockQuantizerTests"
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Gptq/GptqBlockQuantizer.cs
git commit -m "feat(gptq): implement GptqBlockQuantizer — block-wise column quantization"
```

---

### Task 5: TDD `GPTQModifier` lifecycle + hook integration

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQModifierTests.cs`
- Create: `src/LLMCompressorSharp.Core/Algorithms/Gptq/GPTQModifier.cs`

`GPTQModifier` integrates all Phase 4a infrastructure and the Phase 4b block quantizer into the standard `IModifier` lifecycle. The key challenge: `GPTQModifier` receives `Module<Tensor, Tensor>` references from `CompressionState.NamedModules` (added in Task 2), installs hooks during `OnStartCore`, accumulates Hessians during `OnBatchCore`, then quantizes and writes results during `OnEndCore`.

- [ ] **Step 1: Write the failing lifecycle tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.Core.Algorithms.Gptq;

/// <summary>
/// Tests for <see cref="GPTQModifier"/> lifecycle and output correctness.
/// </summary>
public class GPTQModifierTests
{
    private static GPTQConfig DefaultConfig(int numBits = 8) => new GPTQConfig
    {
        NumBits = numBits,
        Symmetric = true,
        Strategy = QuantizationStrategy.PerTensor,
        BlockSize = 4,
        DampeningFrac = 0.01f,
        Targets = null, // match all
        Ignore = null,
    };

    /// <summary>
    /// Builds a minimal <see cref="CompressionState"/> with one Linear layer plus its module.
    /// </summary>
    private static (CompressionState State, Linear Layer) BuildSingleLayerState(
        int inFeatures = 4,
        int outFeatures = 3,
        string weightKey = "linear.weight")
    {
        var layer = Linear(inFeatures, outFeatures, hasBias: false);
        var namedWeights = new Dictionary<string, Tensor>
        {
            [weightKey] = layer.weight.detach().clone(),
        };
        var layerKey = weightKey.Replace(".weight", string.Empty);
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>>
            {
                [layerKey] = layer,
            },
        };
        return (state, layer);
    }

    [Fact]
    public void Modifier_Name_IsGPTQ()
    {
        var modifier = new GPTQModifier(DefaultConfig());
        modifier.Name.Should().Be("GPTQ");
    }

    [Fact]
    public void FullLifecycle_SyntheticLayer_OutputIsFinite()
    {
        // Synthetic: 4-in, 3-out Linear; 8-bit quantization; 5 calibration batches.
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig(numBits: 8));
            modifier.Initialize(state);
            modifier.OnStart(state);

            // Feed 5 calibration batches — hooks accumulate the Hessian automatically.
            for (int i = 0; i < 5; i++)
            {
                using var scope = NewDisposeScope();
                using var batch = randn(2, 4);  // [batch=2, inFeatures=4]

                // The session would normally call model.forward(batch). In this unit test,
                // we trigger the hook by calling layer.call(batch) directly (Phase 4a convention:
                // hooks fire on .call(), not .forward()).
                using var _ = layer.call(batch);

                modifier.OnBatch(state);
            }

            modifier.OnEnd(state);
            modifier.Finalize(state);

            // Quantized weight should be finite.
            state.NamedWeights["linear.weight"].isfinite().all().item<bool>().Should().BeTrue();
        }
    }

    [Fact]
    public void FullLifecycle_QuantizedWeight_CloseToOriginalAt16Bit()
    {
        // At 16-bit, quantized weights should be very close to the originals.
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var originalWeight = state.NamedWeights["linear.weight"].detach().clone();

            var modifier = new GPTQModifier(DefaultConfig(numBits: 16));
            modifier.Initialize(state);
            modifier.OnStart(state);

            for (int i = 0; i < 3; i++)
            {
                using var scope = NewDisposeScope();
                using var batch = randn(2, 4);
                using var _ = layer.call(batch);
                modifier.OnBatch(state);
            }

            modifier.OnEnd(state);
            modifier.Finalize(state);

            using var diff = (state.NamedWeights["linear.weight"] - originalWeight).abs();
            diff.max().item<float>().Should().BeLessThan(1e-3f);

            originalWeight.Dispose();
        }
    }

    [Fact]
    public void FullLifecycle_NoBatches_DoesNotThrow()
    {
        // Zero calibration batches: Hessian is all-zeros; modifier should handle gracefully
        // (damping regularises the zero matrix).
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig());
            modifier.Initialize(state);
            modifier.OnStart(state);
            // No OnBatch calls.
            modifier.OnEnd(state);
            modifier.Finalize(state);

            // Output weight should exist and be finite.
            state.NamedWeights["linear.weight"].isfinite().all().item<bool>().Should().BeTrue();
        }
    }

    [Fact]
    public void FullLifecycle_NullNamedModules_ThrowsInvalidOperation()
    {
        // GPTQModifier requires NamedModules to be populated.
        var namedWeights = new Dictionary<string, Tensor>
        {
            ["linear.weight"] = randn(3, 4),
        };
        var state = new CompressionState(namedWeights);
        // state.NamedModules is null (not set)

        var modifier = new GPTQModifier(DefaultConfig());
        modifier.Initialize(state);
        var act = () => modifier.OnStart(state);
        act.Should().Throw<InvalidOperationException>().WithMessage("*NamedModules*");
    }

    [Fact]
    public void ActOrder_True_ThrowsNotSupported()
    {
        var config = DefaultConfig();
        config.ActOrder = true;
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(config);
            modifier.Initialize(state);
            var act = () => modifier.OnStart(state);
            act.Should().Throw<NotSupportedException>().WithMessage("*ActOrder*");
        }
    }

    [Fact]
    public void Finalize_DisposesHooks_HooksNoLongerFire()
    {
        var (state, layer) = BuildSingleLayerState();
        using (layer)
        {
            var modifier = new GPTQModifier(DefaultConfig());
            modifier.Initialize(state);
            modifier.OnStart(state);

            // Run one batch to establish baseline.
            using var batch1 = randn(2, 4);
            using var _ = layer.call(batch1);
            modifier.OnBatch(state);

            modifier.OnEnd(state);
            modifier.Finalize(state);

            // After Finalize the hooks must be gone. A forward call should NOT update
            // the accumulator. We verify this indirectly by checking the weight is frozen.
            var weightAfterFinalize = state.NamedWeights["linear.weight"].detach().clone();
            using var batch2 = randn(2, 4);
            using var __ = layer.call(batch2);

            // Weight should not have changed (no re-quantization triggered).
            using var diff = (state.NamedWeights["linear.weight"] - weightAfterFinalize).abs();
            diff.max().item<float>().Should().BeLessThan(1e-8f);
            weightAfterFinalize.Dispose();
        }
    }
}
```

- [ ] **Step 2: Verify build fails, commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQModifierTests.cs
git commit -m "test(gptq): add failing GPTQModifier tests"
```

- [ ] **Step 3: Implement `GPTQModifier`**

Create `src/LLMCompressorSharp.Core/Algorithms/Gptq/GPTQModifier.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Core.Algorithms.Gptq;

/// <summary>
/// GPTQ weight quantization modifier.
/// </summary>
/// <remarks>
/// Implements the GPTQ algorithm (Frantar et al., 2022) using Phase 4a's Hessian
/// accumulation infrastructure. For each targeted <c>Linear</c> layer:
/// <list type="number">
///   <item>Attaches a forward hook (via <see cref="ActivationHookManager"/>) during
///         <c>OnStart</c> to accumulate <c>H = Σ Xᵀ X</c> for each calibration batch.</item>
///   <item>During <c>OnEnd</c>: removes hooks, applies damped Cholesky inversion via
///         <see cref="HessianInverseSolver"/>, calls <see cref="GptqBlockQuantizer.Quantize"/>,
///         and writes the quantized weight back to <see cref="CompressionState.NamedWeights"/>.</item>
/// </list>
/// Requires <see cref="CompressionState.NamedModules"/> to be populated before <c>OnStart</c>.
/// </remarks>
public sealed class GPTQModifier : ModifierBase
{
    private readonly GPTQConfig _config;
    private ActivationHookManager? _hookManager;

    // Per-layer accumulator, keyed by layer name (without .weight suffix).
    private Dictionary<string, HessianAccumulator>? _accumulators;

    /// <summary>Initializes a new instance of the <see cref="GPTQModifier"/> class.</summary>
    /// <param name="config">The GPTQ configuration.</param>
    public GPTQModifier(GPTQConfig config)
        : base("GPTQ", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
        // Validate ActOrder early.
        if (_config.ActOrder)
        {
            throw new NotSupportedException(
                "GPTQConfig.ActOrder=true is not yet implemented. " +
                "Column reordering by Hessian diagonal is deferred to a future release.");
        }
    }

    /// <inheritdoc />
    protected override void OnStartCore(CompressionState state)
    {
        if (state.NamedModules is null)
        {
            throw new InvalidOperationException(
                $"{nameof(GPTQModifier)} requires {nameof(CompressionState)}.{nameof(CompressionState.NamedModules)} " +
                "to be populated. Set it to the named Linear modules before calling OnStart.");
        }

        _hookManager = new ActivationHookManager();
        _accumulators = new Dictionary<string, HessianAccumulator>();

        // Register one hook+accumulator per targeted weight.
        // Weight key format: "layerName.weight" → layer key: "layerName"
        foreach (var weightName in GetTargetedNames(state))
        {
            var layerName = weightName.EndsWith(".weight", StringComparison.Ordinal)
                ? weightName[..^".weight".Length]
                : weightName;

            if (!state.NamedModules.TryGetValue(layerName, out var module))
            {
                // Log and skip layers that are in NamedWeights but not in NamedModules.
                // This can happen for embedding layers or custom modules.
                continue;
            }

            // Determine inFeatures from the weight tensor shape [outFeatures, inFeatures].
            var weightShape = state.NamedWeights[weightName].shape;
            var inFeatures = (int)weightShape[^1];

            var accumulator = new HessianAccumulator(inFeatures);
            _hookManager.RegisterFor(module, accumulator);
            _accumulators[weightName] = accumulator;
        }
    }

    /// <inheritdoc />
    protected override void OnBatchCore(CompressionState state)
    {
        // Hooks fire automatically during the caller's model.forward() call.
        // GPTQModifier itself does not need to drive forward passes —
        // the CompressionSession's caller does this externally, and the hooks
        // installed in OnStartCore capture the activations.
        // This is a no-op; the hook callbacks do the work.
        _ = state;
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        // Remove all hooks immediately — calibration is done.
        _hookManager?.Dispose();
        _hookManager = null;

        if (_accumulators is null)
        {
            return;
        }

        foreach (var (weightName, accumulator) in _accumulators)
        {
            using var accumulatorScope = NewDisposeScope();

            using var H = accumulator.GetHessian();
            using var Hinv = HessianInverseSolver.Compute(H, _config.DampeningFrac);

            var W = state.NamedWeights[weightName];
            using var Wq = GptqBlockQuantizer.Quantize(W, Hinv, _config);

            // Replace the weight. The original tensor in state is not disposed here —
            // the caller owns the original lifetime; we just swap the reference.
            state.NamedWeights[weightName] = Wq.detach().clone();
        }
    }

    /// <inheritdoc />
    protected override void OnFinalizeCore(CompressionState state)
    {
        // Belt-and-suspenders: ensure hooks are removed even if OnEnd was skipped.
        _hookManager?.Dispose();
        _hookManager = null;

        if (_accumulators is not null)
        {
            foreach (var accum in _accumulators.Values)
            {
                accum.Dispose();
            }

            _accumulators = null;
        }
    }
}
```

- [ ] **Step 4: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~GPTQModifierTests"
```

Expected: 6 tests pass.

- [ ] **Step 5: Register with `AlgorithmsRegistration`**

Modify `src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs`:

Add to `RegisterAll()`:
```csharp
RegisterGptq();
```

Add the new method:
```csharp
/// <summary>Registers the GPTQ algorithm.</summary>
public static void RegisterGptq()
{
    ModifierRegistry.Register<GPTQConfig>("GPTQ", c => new GPTQModifier(c));
}
```

Add the necessary using:
```csharp
using LLMCompressorSharp.Core.Algorithms.Gptq;
```

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Algorithms/Gptq/GPTQModifier.cs
git add src/LLMCompressorSharp.Core/Algorithms/AlgorithmsRegistration.cs
git commit -m "feat(gptq): implement GPTQModifier with ActivationHookManager + HessianInverseSolver integration"
```

---

### Task 6: End-to-End Tests (synthetic + SmolLM2-135M smoke test)

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQEndToEndTests.cs`

These tests don't test new code — they validate the full system from calibration through quantization in scenarios that mirror real use.

- [ ] **Step 1: Write the end-to-end tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Algorithms;
using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Gptq;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.TorchExtensions.Observers;
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
    // -----------------------------------------------------------------------
    // Synthetic tests — always run
    // -----------------------------------------------------------------------

    /// <summary>
    /// Full compression session with GPTQ using a synthetic 8×16 Linear layer and
    /// 10 calibration batches. Verifies that:
    /// - output weight is finite
    /// - output weight is numerically close to FP original at high bit-width (16-bit)
    /// - output weight is NOT identical to FP original at low bit-width (4-bit)
    /// </summary>
    [Fact]
    public void SyntheticLinear_W4A16_OutputIsFiniteAndQuantized()
    {
        // Create a synthetic linear layer: out=8, in=16.
        using var layer = Linear(16, 8, hasBias: false);
        var originalWeight = layer.weight.detach().clone();

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
            ["fc.weight"] = layer.weight.detach().clone(),
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
            using var _ = layer.call(batch);
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
        var originalWeight = layer.weight.detach().clone();

        var config = new GPTQConfig
        {
            NumBits = 16,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerTensor,
            BlockSize = 4,
            DampeningFrac = 0.01f,
        };

        var namedWeights = new Dictionary<string, Tensor> { ["fc.weight"] = layer.weight.detach().clone() };
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>> { ["fc"] = layer },
        };

        var modifier = new GPTQModifier(config);
        modifier.Initialize(state);
        modifier.OnStart(state);

        for (int i = 0; i < 5; i++)
        {
            using var scope = NewDisposeScope();
            using var batch = randn(2, 8);
            using var _ = layer.call(batch);
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        using var diff = (state.NamedWeights["fc.weight"] - originalWeight).abs();
        diff.max().item<float>().Should().BeLessThan(0.01f,
            "16-bit quantization should have < 0.01 max absolute error");

        originalWeight.Dispose();
    }

    [Fact]
    public void ViaCompressionSession_SingleLayer_Completes()
    {
        // Verify integration with the CompressionSession orchestrator.
        using var layer = Linear(8, 4, hasBias: false);
        var config = new GPTQConfig { NumBits = 8, BlockSize = 4, DampeningFrac = 0.01f };

        var namedWeights = new Dictionary<string, Tensor> { ["fc.weight"] = layer.weight.detach().clone() };
        var state = new CompressionState(namedWeights)
        {
            NamedModules = new Dictionary<string, Module<Tensor, Tensor>> { ["fc"] = layer },
        };

        var session = new CompressionSession(new[] { new GPTQModifier(config) });

        // Generate calibration batches externally; the session feeds them to OnBatch.
        // We also need hooks to fire, so we supply batches that trigger layer.call().
        // Note: the standard session doesn't call the model forward — GPTQ relies on
        // the caller driving the model. For this test we drive via OnBatch + a manual
        // layer.call() before each batch, which is the intended integration pattern.
        var batches = Enumerable.Range(0, 5)
            .Select(_ =>
            {
                using var scope = NewDisposeScope();
                var b = randn(2, 8);
                using var __ = layer.call(b);  // trigger hook before OnBatch
                return b.MoveToOuterDisposeScope();
            })
            .ToList();

        var status = session.Run(state, batches);

        foreach (var b in batches) { b.Dispose(); }

        status.Should().Be(SessionStatus.Completed);
        state.NamedWeights["fc.weight"].isfinite().all().item<bool>().Should().BeTrue();
    }

    // -----------------------------------------------------------------------
    // Real-model smoke test — skipped unless SmolLM2-135M is cached
    // -----------------------------------------------------------------------

    /// <summary>
    /// Smoke test: load one attention <c>q_proj</c> from SmolLM2-135M and GPTQ-quantize it.
    /// Skipped if the model is not in the HuggingFace cache.
    /// </summary>
    [Fact]
    public void SmolLM2_135M_QProjLayer_W4A16_OutputIsFinite()
    {
        // Locate the model snapshot in the HF cache.
        var hfCache = Environment.GetEnvironmentVariable("LLMC_TEST_CACHE")
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "huggingface", "hub");

        var snapshotDir = Path.Combine(
            hfCache,
            "models--HuggingFaceTB--SmolLM2-135M",
            "snapshots");

        var modelExists = Directory.Exists(snapshotDir)
            && Directory.GetDirectories(snapshotDir).Length > 0;

        Assert.SkipUnless(modelExists,
            "SmolLM2-135M not found in HF cache. " +
            "Run download-test-models.ps1 or set LLMC_TEST_CACHE to a directory containing the model.");

        // Load the model using Phase 3b's loader.
        var loader = new LLMCompressorSharp.Transformers.Loading.LlamaModelLoader();
        using var loaded = loader.Load("HuggingFaceTB/SmolLM2-135M", cacheRoot: hfCache);
        using var model = loaded.Model;

        // Grab the q_proj from layer 0. It's registered under "layers.0.self_attn.q_proj"
        // in the named_modules() hierarchy.
        Module<Tensor, Tensor>? qProjModule = null;
        Tensor? qProjWeight = null;
        string qProjWeightName = "layers.0.self_attn.q_proj.weight";
        string qProjLayerName = "layers.0.self_attn.q_proj";

        foreach (var (name, mod) in model.named_modules())
        {
            if (name == qProjLayerName && mod is Module<Tensor, Tensor> linear)
            {
                qProjModule = linear;
            }
        }

        qProjModule.Should().NotBeNull("q_proj not found in named_modules — check the layer key");

        qProjWeight = model.state_dict()[qProjWeightName];

        var config = new GPTQConfig
        {
            NumBits = 4,
            Symmetric = true,
            Strategy = QuantizationStrategy.PerChannel,
            ChannelAxis = 0,
            BlockSize = 128,
            DampeningFrac = 0.01f,
        };

        var namedWeights = new Dictionary<string, Tensor> { [qProjWeightName] = qProjWeight.detach().clone() };
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

        // Feed 4 short calibration batches — just enough to build a non-degenerate Hessian.
        for (int i = 0; i < 4; i++)
        {
            using var scope = NewDisposeScope();
            var hiddenSize = (int)qProjWeight.shape[^1];
            using var batch = randn(1, 16, hiddenSize);  // [batch=1, seq=16, hidden]
            using var _ = qProjModule!.call(batch);
            modifier.OnBatch(state);
        }

        modifier.OnEnd(state);
        modifier.Finalize(state);

        // The quantized q_proj weight must be finite.
        state.NamedWeights[qProjWeightName].isfinite().all().item<bool>().Should().BeTrue();
    }
}
```

- [ ] **Step 2: Run tests**

```powershell
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "FullyQualifiedName~GPTQEndToEndTests"
```

Expected:
- `SyntheticLinear_W4A16_OutputIsFiniteAndQuantized` — PASS
- `SyntheticLinear_W16A16_OutputCloseToOriginal` — PASS
- `ViaCompressionSession_SingleLayer_Completes` — PASS
- `SmolLM2_135M_QProjLayer_W4A16_OutputIsFinite` — PASS or SKIPPED (depending on cache)

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Core/Algorithms/Gptq/GPTQEndToEndTests.cs
git commit -m "test(gptq): add end-to-end synthetic + SmolLM2-135M smoke tests"
```

---

### Task 7: Full Verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Clean build + full test run**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected:
- 0 errors, 0 warnings
- ~240+ tests passing (203 prior + 4 GPTQConfig + 8 HessianInverseSolver + 9 GptqBlockQuantizer + 6 GPTQModifier + 3 GPTQEndToEnd synthetic = ~233; ±2 for config registration test)
- SmolLM2 smoke test PASS or SKIP (not FAIL)

- [ ] **Step 2: Verify GPTQ is registered in AlgorithmsRegistration**

```powershell
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "FullyQualifiedName~AlgorithmsRegistrationTests"
```

If no registration test exists, add a quick sanity check (can fold into a GPTQConfig test):

```csharp
[Fact]
public void RegisterGptq_CanRoundTripViaRegistry()
{
    AlgorithmsRegistration.RegisterGptq();
    // Verify the config type string resolves.
    // (This test relies on ModifierRegistry.Create being testable — adapt to its actual API.)
}
```

No commit for this task.

---

### Task 8: STOP — Controller handles merge + tag

Target tag: `v0.5.0-alpha`.

The implementing agent must stop here. Merge to `main` and tagging are performed by the controller.

---

## Self-Review Notes

**Spec coverage:**

| Requirement | Covered by |
|---|---|
| Hessian accumulation via Phase 4a hooks | `GPTQModifier.OnStartCore` + `ActivationHookManager` |
| Diagonal damping before Cholesky | `HessianInverseSolver.Compute` |
| Cholesky inversion + triangular solve | `HessianInverseSolver.Compute` |
| Pseudoinverse fallback on ill-conditioned H | `HessianInverseSolver.Compute` catch block |
| Block-wise column iteration (blockSize=128) | `GptqBlockQuantizer.Quantize` |
| Error propagation within block | `GptqBlockQuantizer.Quantize` column loop |
| Error propagation across blocks | `GptqBlockQuantizer.Quantize` tail correction |
| Integration with Phase 1a observer family | `GptqBlockQuantizer.ComputeScaleZeroPoint` + `FakeQuantizeColumn` |
| Integration with Phase 1b `IModifier` lifecycle | `GPTQModifier : ModifierBase` |
| `GPTQConfig` with all knobs | Task 2 |
| `CompressionState.NamedModules` extension | Task 2 |
| `AlgorithmsRegistration.RegisterGptq` | Task 5 |
| End-to-end synthetic test | `GPTQEndToEndTests` |
| SmolLM2-135M real-model smoke test | `GPTQEndToEndTests.SmolLM2_135M_QProjLayer_W4A16_OutputIsFinite` |
| `Assert.SkipUnless` for cache-dependent test | Smoke test |

**Out of scope (callouts already in plan):**

- `actorder` (column reordering by Hessian diagonal) — `GPTQConfig.ActOrder=false` default, `NotSupportedException` if enabled.
- Per-group quantization (group_size < inFeatures) — deferred. Observer family handles per-channel; sub-channel groups are a Phase 5 concern.
- `offload_hessians` (CPU offload after calibration) — deferred to v0.5.x. Manual `.cpu()` call is the workaround.
- Packed INT4 output — Phase 5 / `Int4PackedTensor`.

**Shared with Phase 4c (SparseGPT) — explicit callout:**

`HessianInverseSolver` in `TorchExtensions/Hessian/` is the shared utility. SparseGPT will call `HessianInverseSolver.Compute(H, dampingFrac)` and then apply a mask-selection step instead of fake-quantization. The block iteration code in `GptqBlockQuantizer` is GPTQ-specific and is NOT shared — SparseGPT's inner step diverges enough to justify its own analogous class.

**Known risks and mitigations:**

- **TorchSharp `linalg.solve_triangular` signature** — Task 1 probe. If `upper:` keyword name differs, adapt. Fallback: `linalg.inv(L)`.
- **Slice assignment syntax** — Task 1 probe. `W[TensorIndex.Colon, TensorIndex.Slice(i, j)] = tensor` may not compile or may not work in-place. Verified fallback: `narrow(1, i, len).copy_(tensor)`.
- **Error propagation math** — `GptqBlockQuantizerTests` includes a hand-verified 2×2 case. If tests pass but end-to-end accuracy is poor, re-examine the error formula against the upstream Python source.
- **Hook firing on `.call()` vs `.forward()`** — TorchSharp 0.107.0 hooks fire on `.call()`. The `GPTQModifierTests` explicitly uses `.call()` for this reason. Hooks will NOT fire if the integration test calls `.forward()` directly.
- **`DisposeScope` discipline** — calibration loops wrap each batch in `using var scope = NewDisposeScope()`. The `HessianAccumulator`'s internal tensor survives because it is owned by the accumulator, not by the scope.

**Model selection recommendation for Phase 4b execution:**

Use **opus**. The block-wise error-propagation loop (Task 4) is numerically unforgiving — a sign error or off-by-one in the slice produces output that is finite but incorrect, and the unit tests cannot catch everything. TorchSharp's indexing and in-place semantics at the `narrow`/`sub_`/`mm` boundary are also subtly different from PyTorch. Phase 4a was clean infrastructure; Phase 4b is where correctness demands the most careful reasoning. Opus is worth the cost here.

**Commit count target:** 9 commits across 8 tasks (plus Task 1 which has no commit).

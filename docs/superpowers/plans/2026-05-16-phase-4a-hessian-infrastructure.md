# Phase 4a: Hessian Accumulator + Forward Hook Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-layer Hessian accumulation infrastructure that GPTQ and SparseGPT need. Wraps TorchSharp's `register_forward_hook` so calibration batches feed into a running `H += Xᵀ X` matrix without per-modifier hook bookkeeping. Tested entirely against synthetic `nn.Linear` modules; no real models needed.

**Architecture:**

```
LLMCompressorSharp.TorchExtensions/
└── Hessian/                                ← NEW
    ├── HessianAccumulator.cs               ← per-layer running H = Σ Xᵀ X
    ├── ForwardHookHandle.cs                ← IDisposable wrapper around hook registration
    └── ActivationHookManager.cs            ← multi-module hook lifecycle
```

The Hessian for a Linear layer with weight `W: [out, in]` and input `X: [..., in]` is `H = Xᵀ X` of shape `[in, in]`. For each calibration batch, we flatten `X` to `[N, in]` and accumulate `H += Xᵀ X` (which is symmetric positive semi-definite — exactly what GPTQ's Cholesky needs).

**Tech Stack:** TorchSharp 0.107.0 forward hooks. Numerical work in FP32 (Hessian for FP16 weights still accumulated in FP32 for stability). Builds on Phase 1a's observer hierarchy and Phase 1b's `IModifier` lifecycle.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` (GPTQ workflow), `PR_TO_TORCHSHARP.md` (R-002 observer family).

---

## File Structure

```
src/LLMCompressorSharp.TorchExtensions/
└── Hessian/                                  ← NEW
    ├── HessianAccumulator.cs
    ├── ForwardHookHandle.cs
    └── ActivationHookManager.cs

tests/LLMCompressorSharp.Tests/
└── TorchExtensions/
    └── Hessian/                              ← NEW
        ├── HessianAccumulatorTests.cs
        ├── ForwardHookHandleTests.cs
        └── ActivationHookManagerTests.cs
```

**Responsibility per file:**

- `ForwardHookHandle` — `IDisposable` wrapper around whatever TorchSharp returns from `register_forward_hook(...)`. Calling `Dispose` removes the hook. Provides type safety + deterministic cleanup at calibration end.

- `HessianAccumulator` — Per-instance state: a running `Tensor H : [in, in]` initialised to zeros and a sample count. `Update(Tensor x)` flattens `x` to `[N, in]`, computes `Xᵀ X` in FP32, and adds to `H`. `GetHessian()` returns a clone; `Reset()` zeros. Threadsafe-by-convention only (no locks — calibration runs single-threaded).

- `ActivationHookManager` — Higher-level wrapper. `RegisterFor(Module layer, HessianAccumulator accumulator)` installs a forward hook on the given module that calls `accumulator.Update(x)` for every forward call. Returns a `ForwardHookHandle`. `Clear()` removes all registered hooks. Modeled after Python's context-manager idiom.

**Out of scope:**
- The GPTQ inner loop (Cholesky, block-wise quantization, error propagation) — Phase 4b.
- The SparseGPT mask selection — Phase 4c.
- KV-cache-aware activation collection — deferred indefinitely.

---

## Prerequisites & Conventions

- Phase 3c is merged. Tag `v0.3.2-alpha`. 186 tests passing on `main`.
- `.editorconfig` exempts snake_case in `src/LLMCompressorSharp.TorchExtensions/**.cs` — TorchSharp method names are fine.
- Branch off `main` as `feature/4a-hessian-infrastructure`.
- StyleCop conventions from prior phases: SA1402, SA1500, SA1515, SA1642, SA1201, SA1312, SA1117, SA1118.

---

### Task 1: Branch + investigate `register_forward_hook` API

**Files:** (no file changes — research only)

- [ ] **Step 1: Branch + baseline**

```powershell
git status --short
git log --oneline -1
git checkout -b feature/4a-hessian-infrastructure
dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"
```
Expected: 186 passing.

- [ ] **Step 2: Probe the hook API**

The plan assumes `Module.register_forward_hook` exists in TorchSharp 0.107.0. Verify before writing code:

```powershell
# Quick probe: write a tiny test program inline.
$probePath = Join-Path $env:TEMP "torchsharp-hook-probe.cs"
@'
using TorchSharp;
using static TorchSharp.torch;

var linear = nn.Linear(4, 2, hasBias: false);

// Try the snake_case method directly
var handle = linear.register_forward_hook((module, inputs, output) =>
{
    Console.WriteLine($"forward hook fired: input shape [{string.Join(",", inputs[0].shape)}]");
    return null;
});

using var x = ones(3, 4);
using var y = linear.forward(x);

Console.WriteLine($"output shape: [{string.Join(",", y.shape)}]");

handle.remove();  // or .Dispose() — verify which
Console.WriteLine("OK");
'@ | Out-File -FilePath $probePath -Encoding UTF8

# Look at TorchSharp.dll metadata to find the exact signature.
# (Reflection probes fail due to Protobuf transitive dep; just trust Phase 0 research notes
#  and the plan's assumed API. If the implementation in Task 3 builds, the signature is right.)
```

The plan trusts that `Module.register_forward_hook(Func<Module, Tensor[], Tensor, Tensor?>)` exists. If it doesn't, the implementer adapts; common variants:
- `register_forward_hook(HookDelegate hook)` where `HookDelegate` is a named delegate type
- The hook returns the (possibly modified) output as a `Tensor` not nullable; in that case return the original `output` to leave it untouched.

No commit for this task.

---

### Task 2: Add `ForwardHookHandle`

**Files:**
- Create: `src/LLMCompressorSharp.TorchExtensions/Hessian/ForwardHookHandle.cs`

- [ ] **Step 1: Write the wrapper**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Disposable wrapper around a TorchSharp forward-hook registration. Disposing removes the hook.
/// </summary>
/// <remarks>
/// TorchSharp 0.107.0's <c>register_forward_hook</c> returns a hook-id or a removable handle
/// (the exact type varies by version). This class normalises the surface so callers always have
/// an <see cref="IDisposable"/>. Implementation detail: the <c>removeAction</c> is captured at
/// registration time and invoked on first <see cref="Dispose"/>; subsequent calls are no-ops.
/// </remarks>
public sealed class ForwardHookHandle : IDisposable
{
    private Action? removeAction;

    /// <summary>Initializes a new instance of the <see cref="ForwardHookHandle"/> class.</summary>
    /// <param name="removeAction">Action that removes the underlying hook. Called once on <see cref="Dispose"/>.</param>
    public ForwardHookHandle(Action removeAction)
    {
        ArgumentNullException.ThrowIfNull(removeAction);
        this.removeAction = removeAction;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        var action = this.removeAction;
        this.removeAction = null;
        action?.Invoke();
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.TorchExtensions/LLMCompressorSharp.TorchExtensions.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/Hessian/ForwardHookHandle.cs
git commit -m "feat(torch-extensions): add ForwardHookHandle wrapper"
```

---

### Task 3: TDD `HessianAccumulator`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/HessianAccumulatorTests.cs`
- Create: `src/LLMCompressorSharp.TorchExtensions/Hessian/HessianAccumulator.cs`

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
/// Tests for <see cref="HessianAccumulator"/> — per-layer running H = Σ Xᵀ X.
/// </summary>
public class HessianAccumulatorTests
{
    [Fact]
    public void Initial_HessianIsZeroMatrix()
    {
        using var accum = new HessianAccumulator(inFeatures: 4);
        accum.SampleCount.Should().Be(0L);

        using var h = accum.GetHessian();
        h.shape.Should().Equal(new long[] { 4, 4 });
        h.cpu().data<float>().ToArray().Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void Update_OneBatch_AccumulatesOuterProduct()
    {
        // X = [[1, 2, 3]] (shape [1, 3]); H = Xᵀ X = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        using var accum = new HessianAccumulator(inFeatures: 3);
        using var x = tensor(new float[,] { { 1f, 2f, 3f } });
        accum.Update(x);

        accum.SampleCount.Should().Be(1L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        arr.Should().Equal(new float[]
        {
            1f, 2f, 3f,
            2f, 4f, 6f,
            3f, 6f, 9f,
        });
    }

    [Fact]
    public void Update_MultipleBatches_AccumulatesAdditively()
    {
        // Two batches; each contributes one outer product. Combined H is the sum.
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var b1 = tensor(new float[,] { { 1f, 0f } });
        using var b2 = tensor(new float[,] { { 0f, 1f } });
        accum.Update(b1);
        accum.Update(b2);

        accum.SampleCount.Should().Be(2L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        // Sum of [[1,0],[0,0]] + [[0,0],[0,1]] = [[1,0],[0,1]]
        arr.Should().Equal(new float[] { 1f, 0f, 0f, 1f });
    }

    [Fact]
    public void Update_HighRankInput_FlattensCorrectly()
    {
        // Input shape [batch=2, seq=3, in=2] flattens to [6, 2]; each row contributes outer product.
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,,]
        {
            { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } },
            { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } },
        });
        accum.Update(x);

        accum.SampleCount.Should().Be(6L);
        using var h = accum.GetHessian();
        var arr = h.cpu().data<float>().ToArray();
        // Six rows: [1,0]x2, [0,1]x2, [1,1]x2
        // Outer products: 2*[[1,0],[0,0]] + 2*[[0,0],[0,1]] + 2*[[1,1],[1,1]]
        //                = [[2,0],[0,0]] + [[0,0],[0,2]] + [[2,2],[2,2]]
        //                = [[4,2],[2,4]]
        arr.Should().Equal(new float[] { 4f, 2f, 2f, 4f });
    }

    [Fact]
    public void Update_WrongInFeatures_Throws()
    {
        using var accum = new HessianAccumulator(inFeatures: 3);
        using var x = tensor(new float[,] { { 1f, 2f } });
        var act = () => accum.Update(x);
        act.Should().Throw<ArgumentException>().WithMessage("*last dim*");
    }

    [Fact]
    public void Update_InputInFp16_AccumulatesInFp32()
    {
        // Float16 input → Hessian still float32 for stability.
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,] { { 1f, 2f } }).to(ScalarType.Float16);
        accum.Update(x);

        using var h = accum.GetHessian();
        h.dtype.Should().Be(ScalarType.Float32);
    }

    [Fact]
    public void Reset_ZerosTheRunningState()
    {
        using var accum = new HessianAccumulator(inFeatures: 2);
        using var x = tensor(new float[,] { { 1f, 2f } });
        accum.Update(x);
        accum.Reset();

        accum.SampleCount.Should().Be(0L);
        using var h = accum.GetHessian();
        h.cpu().data<float>().ToArray().Should().AllSatisfy(v => v.Should().Be(0f));
    }

    [Fact]
    public void Constructor_NonPositiveInFeatures_Throws()
    {
        var act = () => new HessianAccumulator(inFeatures: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
```

- [ ] **Step 2: Verify build fails (CS0246)**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~HessianAccumulatorTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/HessianAccumulatorTests.cs
git commit -m "test(torch-extensions): add failing HessianAccumulator tests"
```

- [ ] **Step 4: Implement `HessianAccumulator`**

Create `src/LLMCompressorSharp.TorchExtensions/Hessian/HessianAccumulator.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Accumulates the Hessian <c>H = Σ Xᵀ X</c> for a single layer's inputs across calibration batches.
/// </summary>
/// <remarks>
/// Used by GPTQ and SparseGPT. The Hessian is always FP32 for numerical stability, regardless of
/// the input dtype. Inputs of rank > 2 are flattened to <c>[N, inFeatures]</c> where N is the
/// product of all leading dims.
/// </remarks>
public sealed class HessianAccumulator : IDisposable
{
    private readonly int inFeatures;
    private readonly Tensor hessian;
    private long sampleCount;
    private bool disposed;

    /// <summary>Initializes a new instance of the <see cref="HessianAccumulator"/> class.</summary>
    /// <param name="inFeatures">The input feature dimension (last axis of activation tensors).</param>
    public HessianAccumulator(int inFeatures)
    {
        if (inFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inFeatures), inFeatures, "inFeatures must be positive.");
        }

        this.inFeatures = inFeatures;
        this.hessian = zeros(inFeatures, inFeatures, dtype: ScalarType.Float32);
    }

    /// <summary>Gets the number of activation rows accumulated so far (across all Update calls).</summary>
    public long SampleCount => this.sampleCount;

    /// <summary>Gets the input feature dimension.</summary>
    public int InFeatures => this.inFeatures;

    /// <summary>Accumulates <c>Xᵀ X</c> into the running Hessian.</summary>
    /// <param name="x">Activation tensor with last dim equal to <see cref="InFeatures"/>. Higher ranks are flattened.</param>
    /// <exception cref="ArgumentNullException">If <paramref name="x"/> is null.</exception>
    /// <exception cref="ArgumentException">If the last dim of <paramref name="x"/> does not equal <see cref="InFeatures"/>.</exception>
    public void Update(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        this.ThrowIfDisposed();

        if (x.shape[^1] != this.inFeatures)
        {
            throw new ArgumentException(
                $"Input last dim ({x.shape[^1]}) must equal inFeatures ({this.inFeatures}).",
                nameof(x));
        }

        // Flatten to [N, inFeatures] in FP32.
        using var xFp32 = x.to(ScalarType.Float32);
        using var flat = xFp32.reshape(-1L, (long)this.inFeatures);
        var nRows = (int)flat.shape[0];

        // Hessian += Xᵀ X
        using var contribution = flat.t().matmul(flat);
        this.hessian.add_(contribution);

        this.sampleCount += nRows;
    }

    /// <summary>Returns a clone of the current Hessian tensor. The caller owns disposal of the returned tensor.</summary>
    /// <returns>A new <see cref="Tensor"/> matching the running Hessian.</returns>
    public Tensor GetHessian()
    {
        this.ThrowIfDisposed();
        return this.hessian.detach().clone();
    }

    /// <summary>Zeros the running Hessian and sample count.</summary>
    public void Reset()
    {
        this.ThrowIfDisposed();
        this.hessian.zero_();
        this.sampleCount = 0L;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!this.disposed)
        {
            this.hessian.Dispose();
            this.disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(this.disposed, this);
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~HessianAccumulatorTests"`
Expected: 8 tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/Hessian/HessianAccumulator.cs
git commit -m "feat(torch-extensions): implement HessianAccumulator"
```

---

### Task 4: TDD `ActivationHookManager`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/ActivationHookManagerTests.cs`
- Create: `src/LLMCompressorSharp.TorchExtensions/Hessian/ActivationHookManager.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="ActivationHookManager"/> — multi-module forward-hook lifecycle.
/// </summary>
public class ActivationHookManagerTests
{
    [Fact]
    public void RegisterFor_LinearLayer_HookFiresOnForward()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(linear, accumulator);

        using var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } });
        using var _ = linear.forward(x);

        accumulator.SampleCount.Should().Be(1L);
    }

    [Fact]
    public void RegisterFor_MultipleLayers_EachAccumulatorIndependent()
    {
        using var layer1 = Linear(3, 4, hasBias: false);
        using var layer2 = Linear(4, 2, hasBias: false);
        using var accum1 = new HessianAccumulator(inFeatures: 3);
        using var accum2 = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(layer1, accum1);
        manager.RegisterFor(layer2, accum2);

        using var x = tensor(new float[,] { { 1f, 2f, 3f } });
        using var y1 = layer1.forward(x);
        using var y2 = layer2.forward(y1);

        accum1.SampleCount.Should().Be(1L);
        accum2.SampleCount.Should().Be(1L);
    }

    [Fact]
    public void Dispose_RemovesAllHooks_NoFurtherAccumulation()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);

        var manager = new ActivationHookManager();
        manager.RegisterFor(linear, accumulator);

        using var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } });
        using var y1 = linear.forward(x);
        accumulator.SampleCount.Should().Be(1L);

        manager.Dispose();

        using var y2 = linear.forward(x);
        accumulator.SampleCount.Should().Be(1L); // unchanged after dispose
    }

    [Fact]
    public void Clear_RemovesHooks_AllowsReregistration()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();

        manager.RegisterFor(linear, accumulator);
        using (var x = tensor(new float[,] { { 1f, 2f, 3f, 4f } }))
        using (var _ = linear.forward(x))
        {
            // SampleCount = 1
        }

        manager.Clear();

        manager.RegisterFor(linear, accumulator);
        using (var x = tensor(new float[,] { { 5f, 6f, 7f, 8f } }))
        using (var _ = linear.forward(x))
        {
            // SampleCount += 1 → 2
        }

        accumulator.SampleCount.Should().Be(2L);
    }

    [Fact]
    public void RegisterFor_NullLayer_Throws()
    {
        using var accumulator = new HessianAccumulator(inFeatures: 4);
        using var manager = new ActivationHookManager();
        var act = () => manager.RegisterFor(null!, accumulator);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void RegisterFor_NullAccumulator_Throws()
    {
        using var linear = Linear(4, 2, hasBias: false);
        using var manager = new ActivationHookManager();
        var act = () => manager.RegisterFor(linear, null!);
        act.Should().Throw<ArgumentNullException>();
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/ActivationHookManagerTests.cs
git commit -m "test(torch-extensions): add failing ActivationHookManager tests"
```

- [ ] **Step 3: Implement `ActivationHookManager`**

Create `src/LLMCompressorSharp.TorchExtensions/Hessian/ActivationHookManager.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Registers and tracks forward hooks that feed a <see cref="HessianAccumulator"/> for each layer.
/// Disposing removes every registered hook.
/// </summary>
/// <remarks>
/// Designed for the GPTQ / SparseGPT calibration loop: install hooks before iterating calibration
/// batches, run the model, dispose the manager. Each accumulator becomes the layer's Hessian.
/// </remarks>
public sealed class ActivationHookManager : IDisposable
{
    private readonly List<ForwardHookHandle> handles = new();
    private bool disposed;

    /// <summary>Registers a forward hook on <paramref name="layer"/> that feeds <paramref name="accumulator"/>.</summary>
    /// <param name="layer">The TorchSharp module to hook. Typically a <c>nn.Linear</c>.</param>
    /// <param name="accumulator">The accumulator that receives each forward call's input tensor.</param>
    /// <returns>A handle that, when disposed, removes this specific hook. The manager also tracks
    /// it for batch removal via <see cref="Clear"/> or <see cref="Dispose"/>.</returns>
    public ForwardHookHandle RegisterFor(Module layer, HessianAccumulator accumulator)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(accumulator);
        this.ThrowIfDisposed();

        // TorchSharp 0.107.0 hook signature: (Module, Tensor[], Tensor) -> Tensor? (or Tensor)
        // We capture the accumulator in the closure.
        var hookId = layer.register_forward_hook((m, inputs, output) =>
        {
            // inputs may be a Tensor or Tensor[] depending on TorchSharp version.
            // For nn.Linear the input is the first arg of forward, so inputs[0] is X.
            if (inputs is { Length: > 0 } && inputs[0] is Tensor x)
            {
                accumulator.Update(x);
            }

            return output;  // unchanged
        });

        var handle = new ForwardHookHandle(() =>
        {
            try
            {
                layer.remove_forward_hook(hookId);
            }
            catch
            {
                // Best-effort removal — if the API differs, the hook leaks until layer is disposed.
            }
        });

        this.handles.Add(handle);
        return handle;
    }

    /// <summary>Removes all hooks registered through this manager.</summary>
    public void Clear()
    {
        this.ThrowIfDisposed();
        foreach (var handle in this.handles)
        {
            handle.Dispose();
        }

        this.handles.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!this.disposed)
        {
            this.Clear();
            this.disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(this.disposed, this);
    }
}
```

> **API uncertainty:** The exact `register_forward_hook` and `remove_forward_hook` signatures in TorchSharp 0.107.0 are not documented in this plan; the implementer verifies them in Task 1. Three likely shapes:
>
> 1. `register_forward_hook(Func<Module, Tensor[], Tensor, Tensor?>)` returning `HookHandle` with a `Remove()` method.
> 2. `register_forward_hook(HookDelegate hook)` where `HookDelegate` is a named type with `(Module, IList<Tensor>, Tensor) -> Tensor?`.
> 3. `register_forward_hook(...)` returning `int` (a hook id), with separate `remove_forward_hook(int)`.
>
> Adapt the implementation to whichever shape exists. The behaviour must remain: hook fires on forward → calls `accumulator.Update(x)`, returns output unchanged.

- [ ] **Step 4: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~ActivationHookManagerTests"`
Expected: 6 tests pass.

**If `register_forward_hook` doesn't exist or has a different signature:**
1. Inspect the actual TorchSharp `Module` class via IntelliSense.
2. The Phase 0 research notes say hooks were added in TorchSharp 0.103+; 0.107.0 has them.
3. If after 3 attempts the signature can't be made to fit, report BLOCKED with the specific error. We can pivot to a manual-recursion approach (`foreach (var batch in batches) { foreach (var layer in layers) { x = layer.forward(x); accumulator.Update(x); ... } }`) that avoids hooks — uglier but reliable.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/Hessian/ActivationHookManager.cs
git commit -m "feat(torch-extensions): implement ActivationHookManager"
```

---

### Task 5: TDD `ForwardHookHandle` standalone tests

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/ForwardHookHandleTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using Xunit;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="ForwardHookHandle"/>.
/// </summary>
public class ForwardHookHandleTests
{
    [Fact]
    public void Dispose_InvokesRemoveAction()
    {
        var called = 0;
        var handle = new ForwardHookHandle(() => called++);
        handle.Dispose();
        called.Should().Be(1);
    }

    [Fact]
    public void Dispose_CalledTwice_OnlyInvokesActionOnce()
    {
        var called = 0;
        var handle = new ForwardHookHandle(() => called++);
        handle.Dispose();
        handle.Dispose();
        called.Should().Be(1);
    }

    [Fact]
    public void Constructor_NullAction_Throws()
    {
        var act = () => new ForwardHookHandle(null!);
        act.Should().Throw<ArgumentNullException>();
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~ForwardHookHandleTests"`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/TorchExtensions/Hessian/ForwardHookHandleTests.cs
git commit -m "test(torch-extensions): add ForwardHookHandle tests"
```

---

### Task 6: Full verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Clean run**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```
Expected: 0 errors, 0 warnings, **~203 tests passing** (186 + 8 HessianAccumulator + 6 ActivationHookManager + 3 ForwardHookHandle = 203).

No commit.

---

### Task 7: STOP — controller handles merge + tag

Tag will be `v0.4.0-alpha`.

---

## Self-Review Notes

**Spec coverage:**
- Per-layer Hessian accumulation infrastructure for GPTQ / SparseGPT → Task 3
- Forward hook lifecycle wrapper → Tasks 2, 4, 5

**Out of scope (Phase 4b, 4c):**
- GPTQ Cholesky + block-wise quantization
- SparseGPT mask selection
- Activation reordering (act_order)
- KV cache hooks

**Type consistency:**
- `HessianAccumulator` is `IDisposable` and owns its internal Hessian tensor
- `ActivationHookManager` is `IDisposable` and tracks all `ForwardHookHandle`s
- All three types live in `LLMCompressorSharp.TorchExtensions.Hessian` namespace

**Known risks:**
- **`register_forward_hook` exact signature** — Task 1 verifies. Three fallback strategies documented.
- **`reshape(-1L, x)` overload** — already verified working in Phase 1a observers.
- **`tensor.t().matmul(tensor)` outer product** — standard TorchSharp ops; verified.
- **FP16 input → FP32 Hessian** — `tensor.to(ScalarType.Float32)` works (Phase 3 model loader already uses it).

**Commit count target:** ~6 commits across 7 tasks.

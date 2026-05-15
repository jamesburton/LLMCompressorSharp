# .NET 10 — Overview

.NET 10 (November 2025) ships C# 14, stable `System.Numerics.Tensors`, first-party file-based app scripting, and significant JIT/GC improvements relevant to numeric workloads.

**Key docs:** https://learn.microsoft.com/en-us/dotnet/core/whats-new/dotnet-10/

---

## Summary for ML/Compression Workloads

| Area | What Changed | Impact |
|---|---|---|
| `System.Numerics.Tensors` | Stable API (no longer `[Experimental]`) | Reliable base for tensor math |
| `TensorSpan<T>` slicing | Zero-copy slices (new in .NET 10) | No-cost sub-tensor views |
| `IReadOnlyTensor` | Non-generic interface added | Polymorphic tensor handling |
| C# 14 extension operators | `Tensor<T>` supports `+`, `-`, `*` etc. | Natural tensor math syntax |
| JIT loop inversion | Graph-based, unlocks unrolling | Faster numeric inner loops |
| JIT stack allocation | Small fixed arrays stack-allocated | Fewer GC allocs for temp buffers |
| JIT inlining improvements | `try-finally`, cascading devirt | Better hot-path inlining |
| AVX10.2 intrinsics | `Avx10v2` class added (off by default) | Future SIMD headroom |
| Arm64 write barriers | Dynamic switching (parity with x64) | 8–20% GC pause reduction |
| File-based apps | First-party `.cs` → native exe | Replaces `.csx` / dotnet-script |
| Native AOT | Default for file-based apps | Zero JIT warm-up for CLI tools |

---

## Contents

- [language.md](./language.md) — C# 14 features (extensions, `field` keyword, span conversions, null-conditional assignment)
- [sdk-tooling.md](./sdk-tooling.md) — File-based apps, `dnx`, CLI improvements, NuGet pruning
- [performance.md](./performance.md) — JIT, stack allocation, loop inversion, Native AOT
- [numerics.md](./numerics.md) — `System.Numerics.Tensors`, `TensorPrimitives` full operation catalogue

---

## Key Gaps vs Python/PyTorch

| Gap | Notes |
|---|---|
| No BLAS/LAPACK in-box | Need TorchSharp, MathNet.Numerics, or custom SIMD for large matmul |
| No `bfloat16` type | `System.Half` (float16) works; BF16 requires TorchSharp or custom |
| No autograd | No BCL equivalent; TorchSharp provides this |
| No graph execution engine | `TensorPrimitives` is eager-only |
| No ML.NET updates noted for .NET 10 | ML.NET is a separate NuGet, targets .NET 6+ |

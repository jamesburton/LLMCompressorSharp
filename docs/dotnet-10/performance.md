# .NET 10 — Runtime Performance & Native AOT

---

## JIT Compiler Improvements

### Struct Argument Code Generation

Physical promotion now handles **shared-register packing** for struct members of smaller-than-register types (e.g., multiple `int` fields packed into a 64-bit register). This eliminates intermediate memory round-trips when passing small value-type structs to functions — directly relevant to passing small weight or tensor descriptor structs.

### Loop Inversion (Graph-Based)

Switched from lexical analysis to **graph-based loop recognition** using natural loop detection. More loops are now correctly inverted to `do-while` form, which unlocks:
- Loop cloning
- Loop unrolling
- Induction variable optimisations

**Impact for numeric code:** Inner loops over weight matrices, calibration sample buffers, and tensor operation chains benefit directly.

### Array Interface Devirtualisation

The JIT can now devirtualise and inline interface methods on arrays (e.g., `IEnumerable<T>` iteration via `foreach`). Previously, array interfaces had a special implementation that blocked devirtualisation. `foreach (var w in weights)` over a `float[]` now avoids virtual dispatch.

### Inlining Improvements

| Improvement | Description |
|---|---|
| Cascading devirtualisation | Methods that become devirtualisable after earlier inlining can now themselves be inlined |
| `try-finally` methods | Can now be inlined (previously blocked) |
| Return type propagation | If all return sites in a callee yield the same type, that type is used to devirtualise subsequent calls |
| Profile-driven size limits | Hot callers can inline larger callees than the default size limit |
| Callers with profile data | No longer permanently mark callees as `NoInlining` |

### Code Layout (3-opt TSP Heuristic)

Basic block reordering now models the problem as an **asymmetric Travelling Salesman Problem** and applies a 3-opt heuristic. This improves hot path density, reduces branch distances, and improves instruction cache utilisation — beneficial for long calibration/quantization loops.

---

## Stack Allocation

### Small Fixed-Sized Arrays

**Value type arrays** (no GC pointers, size known at compile time): stack-allocated when they don't escape the method.  
**Reference type arrays**: stack-allocated when lifetime is scoped to the creating method.

```csharp
// These may be stack-allocated in .NET 10:
float[] block = new float[128];       // temporary weight block
int[] indices = new int[32];          // column indices for quantization
```

### Escape Analysis for Struct Fields

Objects referenced only via local struct fields are no longer incorrectly marked as escaping — more intermediate computation structs stay on the stack.

### Delegate / Closure Allocation

`Func<T>` objects that don't outlive their scope are now **stack-allocated** (the closure class itself is not yet; planned for a future release). Calibration loops that capture local variables in lambdas benefit.

---

## AVX10.2 Intrinsics

`System.Runtime.Intrinsics.X86.Avx10v2` class added. Currently **disabled by default** (hardware not yet widely available), but the API is ready. When hardware ships, this unlocks additional SIMD width for numeric operations.

---

## Arm64 Write Barrier

Dynamic write-barrier switching (previously x64-only) is now available on **Arm64**. The new default Arm64 write barrier handles GC regions more precisely.

**Benchmark result:** 8–20%+ reduction in GC pause times on Arm64 — relevant for Apple Silicon Mac development environments.

---

## Native AOT

### What It Is

Ahead-of-time compilation to a **self-contained native binary** at publish time. No JIT at runtime; no .NET runtime installation required on the target machine.

```xml
<!-- Enable in project file -->
<PublishAot>true</PublishAot>
```

```bash
dotnet publish -r linux-x64 -c Release
```

### .NET 10 Improvements

| Improvement | Detail |
|---|---|
| File-based apps default to AOT | Single-file `.cs` apps are AOT by default; opt out with `#:property PublishAot=false` |
| NativeAOT pre-initialiser | Now supports `conv.*` and `neg` IL opcodes; more static initialisers are pre-computed |
| `IsAotCompatible` attribute | Libraries can declare AOT compatibility; `VerifyReferenceAotCompatibility=true` warns on incompatible deps |
| `OptimizationPreference` | `Size` or `Speed` MSBuild property to bias between binary size and throughput |

### Supported Platforms

Windows x64/Arm64/x86, Linux x64/Arm64/Arm, macOS x64/Arm64, iOS Arm64, tvOS Arm64, Android x64/Arm64 (experimental).

### AOT Limitations

| Limitation | Implication |
|---|---|
| No `Assembly.LoadFile`, `Reflection.Emit` | Dynamic plugin loading not possible |
| No C++/CLI or built-in COM (Windows) | Native interop requires P/Invoke |
| All generic instantiations pre-generated | Can significantly increase binary size for heavily-generic code |
| `System.Linq.Expressions`: interpreted only | No compiled delegate expressions at runtime |
| Not all runtime libraries annotated | Some third-party packages incompatible |
| `TorchSharp` compatibility | TorchSharp uses P/Invoke which is AOT-compatible, but `TorchSharp.PyBridge` likely is not |

### AOT for a Compression CLI Tool

A compression tool (`compress.cs`) compiled to AOT is feasible if:
- `TorchSharp-cpu` or `TorchSharp-cuda-windows` are used (P/Invoke is AOT-compatible)
- `TorchSharp.PyBridge` is not needed (or limited to non-AOT build)
- No `dynamic` or `Reflection.Emit` usage
- All generic types used are instantiated at compile time

For a CUDA-dependent compression tool, the main binary is AOT but the LibTorch native `.dll`/`.so` is loaded at runtime via `NativeLibrary.Load()` — this is fully AOT-compatible.

# TorchSharp — Known Limitations vs PyTorch

---

## Critical Gaps for a Compression Library

### 1. No `torch.quantization` Pipeline

PyTorch has `torch.quantization.prepare()`, `convert()`, `FakeQuantize`, `MinMaxObserver`, `PerChannelMinMaxObserver`, quantized layer variants (`QuantizedLinear`, `QuantizedConv2d`), and `fuse_modules()`. None of these exist in TorchSharp.

**What exists:** `quantize_per_tensor`, `quantize_per_channel`, `dequantize` primitives (added 0.107.0). All higher-level infrastructure must be built from scratch.

**Tracking:** Issue #1298 (opened April 2024): "TorchSharp doesn't support quantized models" — open as of May 2026.

### 2. No DataParallel / DistributedDataParallel

Multi-GPU training and inference require manual tensor placement. Compressing very large models (Llama 70B, 405B) that require multiple GPUs for calibration is not supported in a high-level way. Manual model sharding across GPUs is possible but requires significant plumbing.

### 3. No Model Graph Tracing

PyTorch FX is used by llm-compressor's sequential pipeline to automatically partition a model into subgraphs for layer-by-layer calibration. TorchSharp has no equivalent to `torch.fx.symbolic_trace()`.

**Impact:** The sequential pipeline cannot be directly ported. Must manually define model subgraph boundaries (e.g., explicit per-layer calibration loops), which requires knowing the architecture at compile time.

### 4. No `torch.compile`

The `torch.compile` JIT compilation pathway (TorchInductor) does not exist in TorchSharp. AutoRound uses `enable_torch_compile: true` in Python to speed up its inner loop — this optimisation path is unavailable in .NET.

---

## Other Gaps

### No TorchScript Creation from .NET

TorchScript modules can be **loaded** (from `.pt` files created in Python) but cannot be **created** from .NET modules. There is no `.trace()` or `.script()` equivalent.

### No ONNX Export

No built-in `torch.onnx.export()`. Export path: use Python, then use ONNX Runtime in .NET for inference.

### No `torch.hub` Model Loading

`torch.hub.load(repo, model)` is unavailable. Only `torch.hub.download_url_to_file` is implemented.

### No CUDA Memory Management APIs

`torch.cuda.empty_cache()`, `memory_allocated()`, `memory_reserved()`, `memory_stats()` are not exposed. VRAM pressure monitoring requires NVML (P/Invoke) or external tools.

### Limited Sparse Tensor Support

Only sparse COO format: `sparse_coo_tensor()` creation and `to_sparse()` conversion. No:
- CSR/CSC sparse format
- Sparse matrix multiply (`mm` with sparse inputs)
- Structured sparsity acceleration

### No `torch.nn.utils.weight_norm` / `spectral_norm`

Weight normalisation utilities are missing. The `NN/Utils` directory only contains `PackedSequence.cs` and `RNNUtils.cs`.

### `torch.export` Load Restriction

Can only load AOTInductor-compiled `.pt2` files. Models saved with `torch.export.save()` (the non-compiled form) will fail to load.

---

## Performance Overhead

### P/Invoke Object Allocation

Each tensor operation allocates a new C# `Tensor` wrapper object (confirmed in issue #1442). For small/fast GPU operations (`randn`, `add`, `slice`), this wrapper allocation overhead dominates over the actual GPU kernel time.

**Measured:** TorchSharp is slower than PyTorch on CUDA for: `randn`, `matmul`, `concat`, `slice`, `add`. 

**Mitigation:** Use in-place operations (`add_()`, `mul_()`, `relu_()` etc.) which return `this` and avoid new object allocation. For many compression algorithms (Hessian accumulation, block-wise weight updates), in-place patterns are natural.

### Memory Management Discipline

The .NET GC cannot observe GPU memory pressure. Missing `Dispose()` calls on GPU tensors cause silent VRAM leaks that eventually crash with `OutOfMemoryError` from CUDA. Rules:
- Always use `using` statements or `DisposeScope` for GPU tensors
- Call `GC.Collect()` + `GC.WaitForPendingFinalizers()` after releasing large allocations if not using `DisposeScope`
- `DisposeScope` adds runtime overhead; nested scopes increase GC pressure

### No macOS Intel Support

Dropped at v0.103.0. Last version with Intel Mac support: v0.102.8.

---

## API Ergonomics

### Snake_case in C#

TorchSharp intentionally uses Python-style snake_case method names (`tensor.view_as_real()`, `torch.amax()`, `module.named_parameters()`). This conflicts with C# naming conventions and StyleCop/Roslyn analyser rules. Set up SA/CA suppressions for TorchSharp call sites.

### No `torch.nn.functional` Namespace

Functional versions of layers are scattered: normalisation functional ops live in `NN/Normalization/Functional.cs`; most others are methods on the `torch` static class. There is no clean `torch.nn.functional` equivalent namespace.

### Thread Safety

Tensors are not thread-safe. Creating a `DisposeScope` on one thread and disposing tensors created on another is undefined behaviour. Parallel calibration loops require separate `DisposeScope` per thread.

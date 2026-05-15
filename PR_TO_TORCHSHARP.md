# PR_TO_TORCHSHARP — Upstream Contribution Tracker

Tracks every TorchSharp gap we hit or notice, split into **Required** (we worked
around it; we owe a PR) and **Proposed** (would benefit the community; we
haven't needed it ourselves yet).

Inline code references each entry with `// TODO(PR_TO_TORCHSHARP <id>)` so we
can find every site when we prepare the actual PR.

---

## Entry Template

```markdown
### [<id>] <Short title>

**Status:** Required | Proposed
**Category:** Quantization | Autograd | Memory | Tracing | Other
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `path/to/file.cs` (action)

**Why upstream cares:** <one paragraph>

**Our workaround:** `LLMCompressorSharp.TorchExtensions.<Type>` (file)

**PR readiness:** <estimate + blockers>
```

---

## Required

### [R-001] FakeQuantize autograd function

**Status:** Required — pending implementation in `LLMCompressorSharp.TorchExtensions` (Phase 1).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/NN/Quantization/FakeQuantize.cs` (new)
- `src/TorchSharp/Tensor/torch.cs` (add `torch.fake_quantize_per_tensor_affine` static)
- `test/TorchSharpTest/TestQuantization.cs` (new tests)

**Why upstream cares:** Mirrors `torch.fake_quantize_per_tensor_affine` from PyTorch. Bridges a key gap to a full `torch.quantization` story without requiring a higher-level API. Useful for any quantization-aware workflow, not only ours.

**Our workaround:** `LLMCompressorSharp.TorchExtensions.FakeQuantizeFunction` (custom autograd via `SingleTensorFunction<T>`). To be implemented in Phase 1.

**PR readiness:** Self-contained; STE backward is standard. Estimate: half-day cleanup once our workaround is stable.

---

### [R-002] MinMax observers as TorchSharp modules

**Status:** Required — pending implementation in `LLMCompressorSharp.TorchExtensions` (Phase 1).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/NN/Quantization/Observers/MinMaxObserver.cs` (new)
- `src/TorchSharp/NN/Quantization/Observers/PerChannelMinMaxObserver.cs` (new)
- `src/TorchSharp/NN/Quantization/Observers/MovingAverageMinMaxObserver.cs` (new)
- `test/TorchSharpTest/TestObservers.cs` (new tests)

**Why upstream cares:** Mirrors `torch.ao.quantization.observer.*`. Required to build any calibration workflow.

**Our workaround:** Observer hierarchy in `LLMCompressorSharp.TorchExtensions.Observers.*`. To be implemented in Phase 1.

**PR readiness:** Self-contained per observer. Estimate: 1 day cleanup per observer once stable.

---

### [R-003] True packed `QInt4` dtype

**Status:** Required — likely fork-required (blocks pure-extensions strategy).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- Native LibTorch bindings (Pinvoke layer)
- `src/TorchSharp/Tensor/torch.ScalarType.cs`
- `src/TorchSharp/Tensor/Tensor.factories.cs`

**Why upstream cares:** Real 4-bit packed storage is essential for production memory savings. Currently `QInt8`/`QUInt8` are the only packed quantized dtypes.

**Our workaround:** `LLMCompressorSharp.TorchExtensions.Int4PackedTensor` — FP32-backed simulation. Provides correctness but no memory benefit. To be implemented in Phase 1.

**PR readiness:** Blocked on native binding work. Likely escalates to fork-submodule path per the design spec (Section 3). To be re-evaluated end of Phase 4 when actual int4 storage matters for accuracy validation.

---

## Proposed

### [P-001] Expose CUDA memory stats

**Status:** Proposed
**Category:** Memory
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/Tensor/torch.cuda.cs` (add `memory_allocated`, `memory_reserved`, `memory_stats`)

**Why upstream cares:** Mirrors `torch.cuda.memory_allocated()` etc. Useful for any application needing to monitor VRAM during long-running calibration / training.

**Our workaround:** Optional P/Invoke to NVML in `LLMCompressorSharp.TorchExtensions.NvmlMemoryStats`. Works but is platform-specific (Linux/Windows with NVIDIA driver). To be implemented in Phase 1.

**PR readiness:** Small. Estimate: 1–2 days once we have a working internal version to validate against.

---

### [P-002] FX-style symbolic tracing

**Status:** Proposed — large scope, likely permanent gap.
**Category:** Tracing
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- New `src/TorchSharp/Fx/` namespace
- Substantial — would require ahead-of-time C# code analysis or runtime hook instrumentation

**Why upstream cares:** PyTorch's `torch.fx` enables many ecosystem tools (model rewriting, ONNX export, llm-compressor's sequential pipeline). A .NET equivalent would unlock similar tooling.

**Our workaround:** Manual per-architecture sequential calibration loops (e.g. `Architectures/Llama/SequentialCalibration.cs`). Avoids tracing entirely. To be implemented in Phase 4.

**PR readiness:** Out of scope for upstream contribution in the foreseeable future. Documented here as a known gap.

---

## How To Use This File

1. When you hit a TorchSharp limitation in code, add an entry under the right section.
2. Mark the inline workaround with `// TODO(PR_TO_TORCHSHARP <id>)` so we can find all call sites when the PR is ready.
3. Once a workaround is stable + tested, raise the corresponding PR upstream. Link the PR URL back here when opened.
4. When a PR merges, remove the inline TODOs and replace the workaround with the upstream API.

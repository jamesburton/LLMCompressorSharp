# LLMCompressorSharp — Critical Pitfalls

---

## P1 — No HuggingFace Transformers Equivalent (BLOCKER)

**Severity:** Critical for a general-purpose compressor; workaroundable for single-architecture tools.

**Problem:** Loading LLaMA 3, Qwen 3, DeepSeek, Gemma, or any modern LLM requires:
1. Model architecture code (Python's `modeling_llama.py` etc.) — thousands of lines per model family
2. `config.json` parsing for hyperparameters (hidden size, num heads, num layers, etc.)
3. Safetensors weight loading (handled by `TorchSharp.PyBridge`)
4. Tokenizer (handled by `Microsoft.ML.Tokenizers` for common models)

There is no .NET library that covers point 1 comprehensively.

**Mitigations (ranked by effort):**
1. **Scope to one architecture.** Implement `LlamaForCausalLM` manually in TorchSharp (~500 lines). Covers Llama 2/3, Mistral, Vicuna, etc.
2. **TorchScript bridge.** In Python, `torch.jit.trace(model, ...)` the model and save `.pt`. Load in TorchSharp via `torch.jit.load()`. Access named parameters for compression; run forward passes for calibration. **Upside:** No architecture reimplementation. **Downside:** Requires Python once at setup; traced model is frozen in training/eval mode; no architecture-specific patching for MoE.
3. **Python.NET subprocess.** Call Python from .NET for model loading; transfer tensor data over IPC. High complexity, Python dependency.
4. **Don't target arbitrary models.** Scope to GGUF-format models via llama.cpp weights → convert to TorchSharp tensors for compression → output GGUF. But llama.cpp uses packed formats that require unpacking first.

**Recommendation:** Scope to LLaMA-family + manual architecture implementation for v1.

---

## P2 — GPU Memory Management Discipline

**Severity:** High. Will cause production crashes without careful implementation.

**Problem:** TorchSharp tensors hold native CUDA memory. The .NET GC does not understand VRAM pressure. A calibration loop that creates thousands of temporary tensors without explicit disposal will silently exhaust VRAM and crash with a CUDA OOM error — often far from the offending code.

**Rules that must be enforced:**
```csharp
// ✅ CORRECT: every tensor in a loop is scoped
for (int i = 0; i < calibrationSamples; i++)
{
    using var scope = torch.NewDisposeScope();
    var input = GetBatch(i);       // disposed at end of scope
    var output = model.forward(input);  // disposed at end of scope
    AccumulateStats(output);       // copy scalars out before scope ends
}

// ❌ WRONG: tensors accumulate until GC
for (int i = 0; i < calibrationSamples; i++)
{
    var output = model.forward(GetBatch(i));  // VRAM leak
    stats.Add(output);
}
```

**For Hessian accumulation:** The Hessian tensor `H` must outlive individual `DisposeScope` blocks — use `MoveToOuterDisposeScope()` or keep explicit handles. Hessian offloading (`H = H.cpu()`) after each layer reduces peak VRAM at the cost of latency.

**Testing:** Write a test that runs 1000 calibration steps on a small model and asserts VRAM usage doesn't grow monotonically. Use `nvidia-smi` polling or NVML P/Invoke to measure.

---

## P3 — P/Invoke Performance Overhead in Inner Loops

**Severity:** Medium. Will affect wall-clock time for GPTQ/SparseGPT but not correctness.

**Problem:** Each TorchSharp tensor operation allocates a new C# `Tensor` wrapper object. For tight inner loops (the column-by-column GPTQ loop processes potentially 12,288+ columns for a large LLM), this overhead adds up.

**Measured overhead (issue #1442):** TorchSharp is slower than PyTorch on CUDA for small ops: `add`, `slice`, `randn`. For `matmul` on large matrices, overhead is negligible.

**Mitigations:**
1. **In-place operations everywhere.** Replace `W = W + err` with `W.add_(err)`. In the GPTQ loop, most weight updates are in-place.
2. **Batch column operations.** Process `blockSize` columns at once rather than one at a time where the algorithm allows.
3. **Profile before optimising.** The Cholesky solve and large matmul (most of GPTQ's flops) run at GPU speed. Only the indexing overhead per block iteration is affected.

---

## P4 — No FX Graph Tracing for Sequential Pipeline

**Severity:** Medium. Forces architecture-specific code.

**Problem:** The sequential pipeline in llm-compressor uses PyTorch FX to automatically partition any model into its decoder blocks for layer-by-layer calibration. Without FX in TorchSharp, this automation is lost.

**Impact:** For each model family you support, you must write explicit code to:
1. Move each transformer block to GPU
2. Run calibration batches through that block
3. Capture output activations for the next block
4. Move the block back to CPU (or offload to disk)

```csharp
// Manual sequential calibration for LLaMA
for (int layerIdx = 0; layerIdx < model.NumLayers; layerIdx++)
{
    var layer = model.Layers[layerIdx];
    layer.cuda();

    foreach (var (input, position_ids, attention_mask) in cachedActivations)
    {
        using var scope = torch.NewDisposeScope();
        var output = layer.forward(input, position_ids, attention_mask);
        nextActivations.Add(output.MoveToOuterDisposeScope());
    }

    layer.cpu();
    cachedActivations = nextActivations;
    nextActivations = new();
}
```

This is 30–50 lines per model family — manageable, but not automatic.

---

## P5 — No Multi-GPU Support

**Severity:** Medium for large models; irrelevant for 7B–13B models.

**Problem:** TorchSharp has no `DataParallel` or `DistributedDataParallel`. Compressing Llama 70B (requires ~140GB FP16 = 2× A100 80GB) or Llama 405B is not possible with sequential offloading if a single GPU is insufficient.

**Workaround:** Sequential pipeline offloading to CPU RAM handles models that don't fit on one GPU — as long as CPU RAM is sufficient (~300GB for 70B in FP16). Disk offloading also works but is slow.

**Practical limit:** Models up to ~70B can be quantized on a machine with sufficient CPU RAM (256GB+) using single-GPU sequential offloading. 405B requires multi-GPU — not achievable with TorchSharp alone.

---

## P6 — AutoRound Without `torch.compile`

**Severity:** Low-Medium (affects only AutoRound algorithm).

**Problem:** AutoRound's inner optimisation loop uses `torch.compile` in Python to significantly speed up the per-block tuning. TorchSharp has no `torch.compile`.

**Impact:** AutoRound will be 3–10× slower than the Python version per block. For a 7B model with 32 layers × 200 iterations each, this is substantial.

**Mitigations:**
- Use AOTInductor `.pt2` export for the AutoRound inner function (requires Python for compilation; inference in .NET) — complex and negates the "pure .NET" goal.
- Accept the performance hit. If AutoRound accuracy is needed, recommend users use the Python llm-compressor and import results.
- Implement a faster simple variant (AutoQuant-style) using GPTQ error propagation as the baseline.

---

## P7 — Numerical Precision of Cholesky Decomposition

**Severity:** Low in practice, but a debugging nightmare when it fails.

**Problem:** GPTQ and SparseGPT require stable Cholesky decomposition of the accumulated Hessian. With small calibration datasets, small models, or extreme weight distributions, `H` can become ill-conditioned, causing Cholesky to fail with a `not positive definite` error.

**Mitigations:**
- Apply dampening (`H += λ * diag(H)`) before Cholesky — the `dampening_frac` parameter (default `0.01`).
- Catch Cholesky failures and fall back to pseudoinverse (`torch.linalg.pinv`).
- Add a check: `if (H.diagonal().min().item<float>() < 0) throw new InvalidOperationException(...)`.
- Ensure `H` is accumulated in `float32` even if the model weights are `float16` — sum of squares in FP16 overflows for large layers.

---

## P8 — OutputFormat Compatibility

**Severity:** Low for compression; Medium for downstream deployment.

**Problem:** The `compressed-tensors` output format (used by vLLM) includes not just the quantized weights but also metadata in `config.json` that describes the quantization scheme, group sizes, and zero_points. Writing a fully spec-compliant `compressed-tensors` config from .NET requires reverse-engineering the schema from the Python library.

**Mitigation options:**
1. Target **GGUF format** (llama.cpp) as primary output — well-documented binary spec, several .NET GGUF readers already exist. Usable with LLamaSharp.
2. Target **ONNX with quantized layers** — exportable from .NET via TorchSharp → custom ONNX builder, loadable by ONNX Runtime.
3. Write safetensors + a minimal `quantization_config.json` compatible with a specific vLLM version — fragile but functional.
4. Use Python llm-compressor for the final save step (only the compression runs in .NET).

---

## Summary Risk Table

| Pitfall | Severity | Mitigation Effort |
|---|---|---|
| No HuggingFace Transformers | Critical | High (architecture reimplementation) |
| GPU memory leaks | High | Medium (code patterns + testing) |
| P/Invoke overhead | Medium | Low (in-place ops) |
| No FX graph tracing | Medium | Medium (manual per-architecture) |
| No multi-GPU | Medium | Workaround (CPU offload) |
| AutoRound without compile | Low-Medium | Low (accept limitation) |
| Cholesky stability | Low | Low (dampening) |
| Output format compatibility | Low-Medium | Medium (GGUF or custom) |

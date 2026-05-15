# LLMCompressorSharp — Feasibility Study

A .NET 10 / C# reimplementation of [llm-compressor](https://github.com/vllm-project/llm-compressor) using TorchSharp.

---

## Executive Summary

**Verdict: Feasible for core compression algorithms; the largest gap is model loading.**

The mathematical core of every algorithm (GPTQ, SparseGPT, SmoothQuant, AWQ, WANDA, RTN) is implementable using TorchSharp's BLAS/LAPACK operations, forward hooks, and quantization primitives. The infrastructure (modifier lifecycle, observer pattern, recipe YAML format, session management) maps cleanly to C# design patterns.

The blocking gap is the absence of a .NET equivalent to HuggingFace Transformers. Loading Llama 3, Qwen 3, or DeepSeek from HuggingFace requires:
1. Model architecture definitions (C# equivalents of thousands of lines of Python `modeling_*.py`)
2. Safetensors weight loading (available via `TorchSharp.PyBridge`)
3. Tokenizers (`Microsoft.ML.Tokenizers` covers common models)
4. Config parsing (`config.json` → model hyperparameters)

A minimal viable implementation scoped to **one model family** (e.g., GPT-2 or LLaMA-3 architecture) is achievable in 3–4 months. A general-purpose compressor matching llm-compressor's breadth would take 12–18+ months.

---

## Feasibility by Component

| Component | Feasibility | Notes |
|---|---|---|
| RTN (QuantizationModifier) | ✅ Easy | `quantize_per_tensor/channel` available in TorchSharp 0.107.0 |
| SmoothQuant | ✅ Moderate | Forward hooks added to TorchSharp in 2024; per-channel scale math is trivial |
| AWQ | ✅ Moderate | Forward hooks + grid search; fake-quant via custom autograd function |
| GPTQ | ✅ Moderate-Hard | All BLAS ops present (`matmul`, `linalg.cholesky`, `linalg.solve_triangular`); careful memory management needed |
| SparseGPT | ✅ Moderate-Hard | Same as GPTQ + masking logic |
| WANDA | ✅ Moderate | Simpler than SparseGPT; no Hessian inversion |
| Magnitude Pruning | ✅ Easy | Trivial with `abs()`, `topk()`, masking |
| AutoRound | ⚠️ Hard | No `torch.compile` equivalent; inner loop will be slow without it |
| SpinQuant / QuIP | ⚠️ Hard | Niche rotation math; implementable but significant effort |
| Model loading (general) | ❌ Major gap | No .NET HuggingFace Transformers equivalent |
| Sequential pipeline (FX-based) | ❌ Not portable | No `torch.fx` in TorchSharp; must manually define subgraph boundaries |
| DDP / multi-GPU calibration | ❌ Missing | TorchSharp has no DDP; single GPU only |
| vLLM deployment of output | ⚠️ Via Python | compressed-tensors format is open; could output from .NET, deploy via Python vLLM |

---

## Recommended Approach

### Scope for a Realistic v1

Target **LLaMA-family models** (Llama 2/3, Mistral — same architecture) with:
- Architecture: implement `LlamaForCausalLM` manually in TorchSharp (well-documented, widely studied)
- Weight loading: `TorchSharp.PyBridge.load_safetensors()` + config JSON parsing
- Tokenizer: `Microsoft.ML.Tokenizers` (supports LLaMA/SentencePiece)
- Compression: RTN, SmoothQuant, GPTQ, SparseGPT, AWQ, WANDA
- Output: safetensors + quantization config JSON (compatible with compressed-tensors spec)
- Validation: export to GGUF format via llama.cpp CLI tools, benchmark with LLamaSharp

### Alternative: Hybrid with Python for Model Loading

Use Python (via Python.NET or subprocess) only for model loading and tokenization, then pass tensors to a .NET compression pipeline. This avoids reimplementing transformer architectures but adds a Python dependency — reducing the value proposition.

### Alternative: TorchScript Bridge

Load models via TorchScript `.pt` files (trace the model in Python once), then implement compression algorithms purely in .NET operating on the loaded weights. No architecture reimplementation needed, but model loading is still Python-dependent for the initial trace.

### dotllm / Alternative .NET LLM Libraries

As of May 2026, no mature "dotllm" library exists in the .NET ecosystem that provides HuggingFace-compatible transformer architecture definitions. The closest options are:
- `LLamaSharp` — inference only (llama.cpp GGUF), not suitable for weight manipulation during compression
- `Microsoft.ML.TorchSharp` — BERT-family NLP tasks only, not for general LLM architectures
- A hand-rolled architecture implementation is the only viable path for full independence from Python

---

## Critical Success Factors

1. **Memory management discipline.** GPU VRAM leaks from un-disposed TorchSharp tensors are the #1 practical blocker. Every calibration loop must use `DisposeScope`. Budget 20% of implementation time for memory correctness.

2. **In-place operations for performance.** Hessian accumulation and block-wise weight updates perform many small tensor operations. Use `add_()`, `mm_()` patterns to avoid per-op `Tensor` object allocation overhead.

3. **Manual model subgraph boundaries.** Without FX tracing, the sequential pipeline must be re-implemented by explicitly calling forward on each transformer block in a loop — viable but requires architecture-specific code per model family.

4. **Recipe validation.** The modifier ordering constraints (AWQ before QuantizationModifier, etc.) can be enforced via C# fluent builder validation or a `Recipe` validation pass before execution.

5. **Calibration data.** Use `Microsoft.ML.Tokenizers` + a simple HTTP/file loader for calibration datasets. The dataset variety available in HuggingFace Datasets has no .NET equivalent — a minimal implementation can hard-code a small local dataset.

---

## Related Docs

- [algorithm-mapping.md](./algorithm-mapping.md) — Algorithm-by-algorithm C# mapping
- [pitfalls.md](./pitfalls.md) — Critical risks and how to mitigate them
- [roadmap.md](./roadmap.md) — Phased implementation plan

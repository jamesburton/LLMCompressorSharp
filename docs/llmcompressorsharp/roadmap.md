# LLMCompressorSharp — Implementation Roadmap

Phased plan for a .NET 10 C# reimplementation of llm-compressor using TorchSharp.

---

## Scope Assumptions

- Target model family: **LLaMA-architecture** (Llama 2/3, Mistral, Qwen, Phi share the same decoder-only structure)
- Target compression algorithms: RTN, SmoothQuant, GPTQ, SparseGPT, AWQ, WANDA
- Target output format: GGUF (primary, via llama.cpp CLI) + safetensors + quantization JSON (for vLLM via compressed-tensors)
- Runtime: .NET 10, TorchSharp 0.107.0+
- GPU: Single GPU (CUDA); CPU offload for models larger than VRAM

---

## Phase 1 — Core Infrastructure (4–6 weeks)

**Goal:** A working compression framework with no model-specific code.

### Tasks

1. **Project structure**
   - `LLMCompressorSharp` — core library (NuGet)
   - `LLMCompressorSharp.Cli` — command-line tool (`compress.cs` file-based app)
   - `LLMCompressorSharp.Tests` — xUnit tests

2. **Modifier base class + lifecycle**
   ```
   IModifier → ModifierBase → (GptqModifier, SmoothQuantModifier, ...)
   ```
   - `Initialize(CompressionState)`, `OnBatch(...)`, `OnEnd(...)`, `Finalize(...)`
   - `targets`/`ignore` pattern (layer name filtering)

3. **CompressionSession + State**
   - Holds model, calibration data enumerator, per-modifier state
   - Orchestrates modifier lifecycle

4. **Observer base + implementations**
   - `IObserver` interface: `Update(Tensor)`, `GetQuantParams()`
   - `MinMaxObserver` (static, per-tensor)
   - `PerChannelMinMaxObserver` (static, per output channel)
   - `MovingAverageMinMaxObserver` (EMA)

5. **Recipe YAML format + deserialisation**
   - `YamlDotNet` for parsing
   - `Recipe`, `Stage`, `ModifierConfig` POCOs
   - Polymorphic modifier type deserialisation
   - Validation: ordering constraints (AWQ before QuantizationModifier, etc.)

6. **Unit tests for all above** (no GPU required — use CPU tensors)

**Dependencies:** `TorchSharp-cpu` or `TorchSharp`, `YamlDotNet`

---

## Phase 2 — Simple Compression Algorithms (4–6 weeks)

**Goal:** Working RTN, SmoothQuant, and WANDA implementations against a small test model (GPT-2 or a 2-layer LLaMA clone).

### Tasks

1. **RTN `QuantizationModifier`**
   - Per-tensor and per-channel quantization
   - `Int8`, `Int4` (via FP32 fake-quant for now), `Float16` schemes
   - Observer integration for calibration-based activation quantization

2. **`SmoothQuantTransform`**
   - Forward hook attachment/removal
   - Per-channel activation max collection
   - Scale computation and weight application
   - Tests: verify `Y_original ≈ Y_smooth` within floating-point tolerance

3. **`WandaPruningModifier`**
   - Activation L2 norm collection (forward hooks)
   - Saliency scoring: `|w| × activation_norm`
   - Mask application (unstructured + 2:4 structured)

4. **`MagnitudePruningModifier`**
   - Trivial: sort by `|w|`, zero lowest fraction

5. **Integration test scaffold**
   - A minimal 2-layer decoder-only model in TorchSharp
   - Calibration data: 10 random integer sequences
   - Verify: compressed model perplexity doesn't explode vs baseline

**Dependencies:** Phase 1 + TorchSharp forward hook APIs

---

## Phase 3 — LLaMA Architecture Implementation (6–8 weeks)

**Goal:** Load a real Llama 3 model from HuggingFace safetensors and run forward passes.

### Tasks

1. **Config JSON parser**
   - Parse `config.json` from HuggingFace model repos
   - Map to `LlamaConfig` (hidden_size, num_heads, num_kv_heads, num_layers, vocab_size, etc.)

2. **LlamaForCausalLM in TorchSharp**
   - `LlamaRMSNorm`
   - `LlamaRotaryEmbedding` (RoPE)
   - `LlamaMLP` (gate_proj + up_proj + down_proj + SiLU activation)
   - `LlamaAttention` (q/k/v/o projections + grouped query attention)
   - `LlamaDecoderLayer` (attention + MLP + residuals)
   - `LlamaModel` (embedding + N decoder layers + final norm)
   - `LlamaForCausalLM` (model + lm_head)

3. **Weight loading**
   - `TorchSharp.PyBridge.load_safetensors()` + sharded checkpoint support
   - Name mapping: HuggingFace weight names → TorchSharp module parameter names

4. **Tokenizer integration**
   - `Microsoft.ML.Tokenizers` for LLaMA/BPE tokenization
   - Simple batched encoding + padding

5. **Validation**
   - Load Llama 3 8B, run 5 forward passes on sample text
   - Compare logit distributions to reference Python output (within FP32 tolerance)

**Estimated effort:** This is the highest-effort phase. LLaMA architecture is well-documented but ~500 lines of careful C# translation.

---

## Phase 4 — GPTQ and SparseGPT (6–8 weeks)

**Goal:** Calibrate and apply GPTQ W4A16 to Llama 3 8B; validate perplexity.

### Tasks

1. **Hessian accumulation**
   - Forward hook on each `Linear` layer collecting `H += X^T X`
   - Hessian offloading option (CPU after accumulation)
   - DDP-not-required: single-GPU only

2. **GPTQ core loop**
   - Cholesky decomposition + dampening
   - Block-wise column quantization with error propagation
   - Activation ordering support

3. **Quantization schemes**
   - INT8 per-tensor and per-channel (round-trip through TorchSharp primitives)
   - INT4 (simulate via FP32 fake-quant; pack for output)
   - FP8 (if LibTorch 2.10 exposes FP8 ops)

4. **SparseGPT variant**
   - Prune instead of quantize in the inner loop
   - 2:4 structured sparsity mask enforcement
   - Combined sparse + quantised (sparse mask preservation)

5. **Sequential calibration pipeline (manual)**
   - Layer-by-layer calibration for Llama:
     - Run embedding layer → cache activations to CPU
     - For each decoder block: move to GPU, accumulate Hessians, compress, move back
   - CPU RAM limit check before starting (estimate required memory)

6. **Accuracy validation**
   - Measure WikiText-2 perplexity before and after compression
   - Target: W4A16 GPTQ on Llama 3 8B ≤ +0.3 perplexity vs baseline

---

## Phase 5 — AWQ and Output Formats (4–6 weeks)

**Goal:** AWQ implementation + GGUF output for LLamaSharp validation.

### Tasks

1. **AWQ grid search implementation**
   - Forward hooks for activation stats
   - Grid search over scale candidates
   - Fake-quantize via `FakeQuantizeFunction` (custom autograd)
   - Permanent scale application

2. **Output: safetensors + quantization config JSON**
   - Write quantized weight tensors as safetensors (via `TorchSharp.PyBridge.save_safetensors()`)
   - Write `quantization_config.json` in compressed-tensors format
   - Validate: load output in Python with `compressed_tensors` library

3. **Output: GGUF (via llama.cpp tools)**
   - Write unquantized safetensors → call `llama.cpp/convert_hf_to_gguf.py` via subprocess
   - Call `llama.cpp/quantize` on the GGUF to apply GPTQ-compatible quantization

4. **End-to-end validation with LLamaSharp**
   - Load GGUF output with LLamaSharp
   - Run benchmark: tokens/sec vs baseline GGUF

---

## Phase 6 — CLI Tool + Polish (2–3 weeks)

**Goal:** A usable command-line tool distributed as a .NET global tool.

```bash
# compress.cs (file-based app) or dotnet tool
dotnet tool install --global LLMCompressorSharp

llmc compress \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --recipe gptq-w4a16 \
  --calibration-data wikitext-2 \
  --output ./compressed
```

### Tasks

1. File-based app entry point with `#:package LLMCompressorSharp` directives
2. Recipe file loading from YAML or built-in presets
3. Progress reporting (calibration samples processed, layer completion)
4. Hugging Face Hub token support (for gated models like Llama 3)
5. Memory estimate before compression (warn if insufficient RAM)
6. `dotnet pack` + NuGet publishing

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|---|---|---|
| 1: Infrastructure | 4–6 weeks | Modifier lifecycle, observer, recipe YAML |
| 2: Simple algorithms | 4–6 weeks | RTN, SmoothQuant, WANDA working on toy model |
| 3: LLaMA architecture | 6–8 weeks | Load + run real Llama 3 in TorchSharp |
| 4: GPTQ + SparseGPT | 6–8 weeks | W4A16 compression with validated perplexity |
| 5: AWQ + output | 4–6 weeks | AWQ + GGUF + safetensors output |
| 6: CLI polish | 2–3 weeks | Usable `llmc compress` command |
| **Total** | **~6–9 months** | Full v1 for LLaMA-family models |

---

## Technology Stack

```
LLMCompressorSharp (net10.0)
├── TorchSharp 0.107.0+       — tensor ops, nn modules, autograd, CUDA
├── TorchSharp-cuda-windows   — LibTorch 2.10.0 CUDA 12.8 binaries
├── TorchSharp.PyBridge 1.4.3 — safetensors load/save, HF checkpoint loading
├── Microsoft.ML.Tokenizers   — LLaMA/BPE tokenization
├── YamlDotNet                — recipe YAML parsing
└── System.Numerics.Tensors   — CPU SIMD for observer math (optional)

LLMCompressorSharp.Cli (net10.0)
└── System.CommandLine        — CLI argument parsing
```

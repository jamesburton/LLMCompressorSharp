# GPTQ — Hessian-Based Weight Quantization

**Paper:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)  
**Location:** `src/llmcompressor/modifiers/gptq/`  
**Class:** `GPTQModifier(Modifier, QuantizationMixin)`

---

## What It Does

GPTQ quantizes weights layer-by-layer using the inverse Hessian of the layer's inputs to minimise reconstruction error. For each weight matrix, it processes `block_size` columns at a time, quantizes them, then propagates the quantization error to remaining columns using the inverse Hessian — so later columns can partially compensate for earlier errors.

Result: much better accuracy than Round-To-Nearest (RTN) at 4-bit (W4A16), typically within ~0.1–0.5 perplexity points of full-precision.

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `block_size` | `int` | `128` | Columns processed per pass in the weight matrix |
| `dampening_frac` | `float \| None` | `0.01` | Diagonal dampening on Hessian for numerical stability |
| `actorder` | `ActivationOrdering \| Sentinel` | `Sentinel("static")` | Reorder columns by Hessian diagonal magnitude before processing |
| `offload_hessians` | `bool` | `False` | Store Hessian tensors on CPU (saves GPU memory, slower) |
| `targets` | `str \| list` | `"Linear"` | Which layer types to quantize |
| `ignore` | `list` | `["lm_head"]` | Layers to skip |
| `scheme` | `str` | — | Quantization scheme: `"W4A16"`, `"W8A16"`, etc. |

---

## Algorithm (step by step)

1. **Calibration phase:** For each target `Linear` layer, attach hooks that accumulate the Hessian `H = X^T X` from calibration samples. Optional: offload `H` to CPU after accumulation.

2. **Quantization phase:** For each layer:
   a. Add dampening: `H ← H + λ·diag(H)` to improve numerical stability.
   b. Optionally reorder columns by descending `diag(H)` (activation ordering).
   c. Compute inverse Hessian via **Cholesky decomposition**: `H_inv = chol(H)^-1`.
   d. For each block of `block_size` columns:
      - Fake-quantize the weight columns to the target precision.
      - Compute error: `E = (W_orig - W_quant) / diag(H_inv)^2`.
      - Update remaining columns: `W_remaining -= E · H_inv[block, remaining]`.
   e. Optionally restore original column order (if reordered).

3. **Freeze quantization:** Remove calibration hooks; freeze scale/zero_point parameters.

---

## Private State

| Field | Type | Description |
|---|---|---|
| `_hessians` | `Dict[Module, Tensor]` | Per-layer accumulated Hessian |
| `_num_samples` | `Dict[Module, Tensor]` | Sample count per layer |
| `_module_names` | `list` | Names of targeted modules |

---

## Recipes

**Minimal W4A16:**
```yaml
gptq_stage:
  gptq_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
```

**W4A16 with activation ordering:**
```yaml
gptq_stage:
  gptq_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
      block_size: 128
      dampening_frac: 0.01
      actorder: weight
```

**Python API:**
```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    block_size=128,
    dampening_frac=0.01,
)
```

---

## Supported Schemes

`W4A16`, `W8A16`, `W4AFP8`, `FP8_DYNAMIC`, `FP8_BLOCK`, `NVFP4`, `MXFP4`, `MXFP8` (and any `compressed-tensors` scheme preset).

---

## Accuracy Considerations

- **Calibration data:** 128–512 samples from the target domain; more representative = better accuracy.
- **Group size 128** (part of the W4A16 scheme): standard for W4A16. Smaller groups improve accuracy but increase model size.
- **Activation ordering** (`actorder: weight` or `actorder: activation`): reordering columns can improve accuracy at W3 or lower bits; marginal benefit at W4.
- **Hessian offloading** (`offload_hessians: True`): enables quantizing models larger than GPU VRAM at the cost of ~2–4× slower quantization.

---

## .NET Reimplementation Feasibility

All core operations are available in TorchSharp:
- Hessian accumulation: `tensor.mm(tensor.t())` (matmul) — ✅
- Cholesky decomposition: `torch.linalg.cholesky()` — ✅
- Triangular solve / inverse: `torch.linalg.solve_triangular()`, `torch.linalg.inv()` — ✅
- Block-wise weight update: indexing + matmul — ✅
- Forward hooks for Hessian capture: `register_forward_hook()` — ✅ (added 2024)

See [algorithm-mapping.md](../../llmcompressorsharp/algorithm-mapping.md) for full C# mapping.

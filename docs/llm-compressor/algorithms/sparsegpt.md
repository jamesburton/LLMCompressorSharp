# SparseGPT — Hessian-Based Pruning

**Paper:** "SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot" (Frantar & Alistarh, 2023)  
**Location:** `src/llmcompressor/modifiers/pruning/sparsegpt/`  
**Class:** `SparseGPTModifier(SparsityModifierBase)`

---

## What It Does

SparseGPT applies the same Hessian-based error propagation as GPTQ, but instead of quantizing weights, it *prunes* them (sets them to zero) to hit a target sparsity level. For each column block, the algorithm zeros out a fraction of weights (those with lowest saliency per the Hessian) and propagates the pruning error to remaining columns.

Supports:
- **Unstructured sparsity** — any individual weight can be pruned (`mask_structure: "0:0"`)
- **2:4 structured sparsity** — NVIDIA's hardware-accelerated format: exactly 2 non-zeros per 4 weights (`mask_structure: "2:4"`)
- **OWL** — Outlier Weighted Layerwise Sparsity: per-layer sparsity based on outlier statistics rather than uniform global sparsity

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `block_size` | `int` | `128` | Columns per pass |
| `dampening_frac` | `float \| None` | `0.01` | Hessian diagonal dampening |
| `preserve_sparsity_mask` | `bool` | `False` | Keep existing sparsity from a prior pruning pass |
| `offload_hessians` | `bool` | `False` | CPU offload for Hessian matrices |
| `sparsity` | `float` | — | Target global sparsity ratio (e.g. `0.5` = 50%) |
| `mask_structure` | `str` | `"0:0"` | `"0:0"` = unstructured; `"2:4"` = NVIDIA structured sparsity |
| `sparsity_profile` | `str \| None` | `None` | `"owl"` = Outlier Weighted Layerwise Sparsity |
| `owl_m` | `int` | — | OWL: number of outliers threshold per layer |
| `owl_lmbda` | `float` | — | OWL: lambda weighting |
| `targets` / `ignore` | — | — | Layer targeting |

---

## Algorithm

Identical infrastructure to GPTQ (Hessian accumulation + Cholesky + block-wise processing), but instead of quantizing:

1. For each column block, compute a **saliency score** per weight = `w² / H_inv_diag²`.
2. Zero out the bottom `(1 - sparsity_target) × block_size` weights (lowest saliency).
3. **Propagate error** to remaining columns (same formula as GPTQ).
4. Apply the sparsity mask permanently.

For **2:4 structured sparsity**, the selection is constrained to choose exactly 2 non-zeros per each group of 4 consecutive weights.

For **OWL**, the per-layer sparsity target is computed from the layer's outlier statistics before the main loop.

---

## Recipes

**50% unstructured sparsity:**
```yaml
sparsegpt_stage:
  sparsegpt_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      mask_structure: "0:0"
      targets: ["Linear"]
      ignore: ["lm_head"]
```

**2:4 structured sparsity:**
```yaml
sparsegpt_stage:
  sparsegpt_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      mask_structure: "2:4"
      targets: ["Linear"]
      ignore: ["lm_head"]
```

**OWL 50% average sparsity:**
```yaml
sparsegpt_stage:
  sparsegpt_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      sparsity_profile: owl
      owl_m: 18
      owl_lmbda: 0.08
      targets: ["Linear"]
      ignore: ["lm_head"]
```

**Sparse + quantised (SparseGPT + GPTQ):**
```yaml
sparsegpt_stage:
  sparsegpt_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      mask_structure: "2:4"
      targets: ["Linear"]
      ignore: ["lm_head"]

gptq_stage:
  gptq_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
      preserve_sparsity_mask: true
```

---

## .NET Reimplementation Feasibility

Same BLAS requirements as GPTQ — all available in TorchSharp. Additional requirements:
- Masking/zeroing: `tensor.masked_fill_()` — ✅
- 2:4 structured mask enforcement: custom selection logic in a loop — ✅ (no built-in, but implementable)
- OWL per-layer sparsity calculation: outlier statistics from activation hooks — ✅

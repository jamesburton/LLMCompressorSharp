# Other Algorithms — RTN, AutoRound, WANDA, SpinQuant, QuIP, Magnitude Pruning

---

## RTN (Round-to-Nearest) — `QuantizationModifier`

**Class:** `QuantizationModifier(Modifier, QuantizationMixin)`  
**Location:** `src/llmcompressor/modifiers/quantization/quantization/`

The simplest PTQ method: rounds each weight to the nearest quantization level. No calibration data required for weight-only schemes; optional calibration enables static activation quantization.

### Key Parameters

| Parameter | Description |
|---|---|
| `targets` | Layer types to quantize (default: `["Linear"]`) |
| `ignore` | Layers to exclude (e.g. `["lm_head"]`) |
| `scheme` | Preset name: `"W4A16"`, `"W8A16"`, `"W8A8"`, `"FP8_DYNAMIC"`, `"FP8_BLOCK"`, etc. |
| `config_groups` | Manual scheme → module mapping (alternative to `scheme`) |
| `kv_cache_scheme` | Separate `QuantizationArgs` for KV cache quantization |

### Lifecycle
1. `on_initialize` — attaches schemes to modules, adds disabled observer hooks
2. `on_start` — enables calibration
3. `on_event` — syncs observer stats across DDP; updates scale/zero_point
4. `on_end` — removes observers, freezes quantization
5. `on_finalize` — cleanup

### Scheme Presets (from `compressed-tensors`)
`W4A16`, `W8A16`, `W8A8`, `FP8_DYNAMIC`, `FP8_BLOCK`, `W4AFP8`, `NVFP4`, `MXFP4`, `MXFP8`

### KV Cache Quantization
```python
from compressed_tensors.quantization import QuantizationArgs

kv_cache_scheme = QuantizationArgs(
    num_bits=8, type="float", strategy="tensor", dynamic=False, symmetric=True
)
QuantizationModifier(scheme="W8A8", kv_cache_scheme=kv_cache_scheme)
```

---

## AutoRound — `AutoRoundModifier`

**Paper:** "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs" (Cheng et al., 2023)  
**Class:** `AutoRoundModifier(Modifier, QuantizationMixin)`  
**Location:** `src/llmcompressor/modifiers/autoround/`

### What It Does
Iterative rounding optimisation using signed gradient descent and block-wise MSE loss to fine-tune quantization *rounding decisions* (not the weights themselves). More accurate than GPTQ at the cost of significantly more computation.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `iters` | `200` | Tuning iterations per block |
| `enable_torch_compile` | `True` | Speed up tuning loop via `torch.compile` |
| `batch_size` | `8` | Calibration batch size |
| `lr` | — | Optimizer learning rate |
| `device_ids` | — | Device mapping for layer dispatch |

### Characteristics
- **Much slower than GPTQ:** 200 iterations × number of layers.
- **Higher accuracy** than GPTQ, especially at W4 and lower.
- `torch.compile` is used internally to speed up the inner loop — **no .NET equivalent**.

---

## WANDA — `WandaPruningModifier`

**Paper:** "A Simple and Effective Pruning Approach for Large Language Models" (Sun et al., 2023)  
**Class:** `WandaPruningModifier(SparsityModifierBase)`  
**Location:** `src/llmcompressor/modifiers/pruning/wanda/`

### What It Does
Faster alternative to SparseGPT. Pruning saliency = `|w_ij| · ‖x_j‖₂` (weight magnitude × input activation norm). No Hessian inversion needed — much faster calibration at a small accuracy cost vs SparseGPT.

### Key Parameters
Same as SparseGPT: `sparsity`, `mask_structure` (`"0:0"` or `"2:4"`), `sparsity_profile` (`"owl"`), `owl_m`, `owl_lmbda`, `targets`, `ignore`.

### Comparison to SparseGPT

| | SparseGPT | WANDA |
|---|---|---|
| Accuracy | Better | Slightly worse |
| Speed | Slower (Hessian inversion) | Much faster |
| Memory | Hessian matrices (offloadable) | Only activation norms |

---

## Magnitude Pruning — `MagnitudePruningModifier`

**Class:** `MagnitudePruningModifier(Modifier, LayerParamMasking)`  
**Location:** `src/llmcompressor/modifiers/pruning/magnitude/`

### What It Does
Prunes weights by absolute magnitude: `score = |w|`. The simplest pruning strategy; used in training-aware (QAT-style) sparsity scheduling rather than one-shot.

### Key Parameters

| Parameter | Description |
|---|---|
| `init_sparsity` | Starting sparsity (e.g. `0.0`) |
| `final_sparsity` | Target sparsity (e.g. `0.7`) |
| `targets` | Which layers to prune |
| `update_scheduler` | Scheduler type (`"cubic"` is standard) |
| `mask_structure` | `"0:0"` or `"2:4"` |

### Use Case
Training-aware progressive sparsification during fine-tuning, not one-shot compression. Not recommended for inference-only PTQ workflows (use SparseGPT or WANDA instead).

---

## SpinQuant — `SpinQuantModifier`

**Paper:** "SpinQuant: LLM Quantization with Learned Rotations" (Liu et al., 2023)  
**Class:** `SpinQuantModifier`  
**Location:** `src/llmcompressor/modifiers/transform/spinquant/`

### What It Does
Applies learned rotation matrices to weights and activations to reduce their dynamic range, making subsequent quantization more accurate. Offline-fused rotations (R1/R2) have zero inference overhead; online rotations (R3/R4) add small overhead.

### Key Parameters

| Parameter | Options | Description |
|---|---|---|
| `transform_type` | `"hadamard"`, `"random-hadamard"`, `"random-matrix"` | Matrix type; Hadamard is fastest but requires power-of-2 dims |
| `precision` | `fp32`, `bf16` | Precision for rotation computation |
| `transform_block_size` | int | Block-diagonal matrix size (defaults to `hidden_size` or `head_dim`) |
| `rotations` | list | Which rotation positions to apply (`R1`, `R2`, `R3`, `R4`) |

---

## QuIP / QuIP# — `QuIPModifier`

**Class:** `QuIPModifier`  
**Location:** `src/llmcompressor/modifiers/transform/quip/`

### What It Does
Incoherence-based rotation quantization related to SpinQuant. Applies U/V rotation matrices to reduce quantization error via incoherence processing — makes the weight matrix more "incoherent" so errors distribute more uniformly.

### Key Parameters
Same family as SpinQuant: `rotations` (`"v"`, `"u"`, or both), `transform_type`, `transform_block_size`, `precision`.

---

## Logarithmic Equalization

**Location:** `src/llmcompressor/modifiers/logarithmic_equalization/`

A channel-wise smoothing variant using log-scale equalisation rather than the linear smoothing of SmoothQuant. Can be used as the `algorithm: "log_equalization"` option in `SmoothQuantModifier`.

---

## iMatrix Transform

**Location:** `src/llmcompressor/modifiers/transform/imatrix/`  
**Observer:** `src/llmcompressor/observers/imatrix.py`

Computes an importance matrix (iMatrix) from calibration data to improve quantization accuracy, particularly useful for GGUF-format workflows (llama.cpp compatibility). The iMatrix weights the quantization step to prioritise channels that are more important to output quality.

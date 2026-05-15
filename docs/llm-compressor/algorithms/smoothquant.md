# SmoothQuant — Activation Smoothing

**Paper:** "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022)  
**Location:** `src/llmcompressor/modifiers/transform/smoothquant/base.py`  
**Class:** `SmoothQuantModifier`

> The old path `modifiers/smoothquant/base.py` is a deprecated shim.

---

## What It Does

Transformers have asymmetric quantization difficulty: activations have large outliers that are hard to quantize, while weights are more uniform. SmoothQuant migrates this difficulty by multiplying a per-channel scale factor into the weight matrix and dividing it out of the activations — keeping the mathematical output identical but making both activations and weights easier to quantize.

**Mathematical identity:** `Y = X · W = (X · diag(s)^-1) · (diag(s) · W)`

Where `s = max(|X|)^α / max(|W|)^(1−α)`. At α=0.5 (default): equal difficulty; at α=1: all difficulty in weights; at α=0: all difficulty in activations.

This is a **pre-processing transform** — it must precede a `QuantizationModifier` in the recipe.

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `smoothing_strength` | `float` | `0.5` | Alpha (α) controlling difficulty balance |
| `mappings` | `list[SmoothQuantMapping]` | — | Pairs of (activation layer, weight layer) to balance |
| `ignore` | `list` | — | Layers to skip |
| `num_calibration_steps` | `int \| None` | `None` | Steps to calibrate; `None` = full dataset |
| `calibration_function` | `Callable \| None` | `None` | Custom forward pass function |
| `algorithm` | `Literal` | `"smoothquant"` | `"smoothquant"` or `"log_equalization"` |

---

## Algorithm (step by step)

1. **Attach activation hooks** on the activation layers specified in `mappings`.
2. **Run calibration:** forward pass through `num_calibration_steps` batches, collecting per-channel absolute max values.
3. **DDP sync** (if multi-GPU): all-reduce the collected statistics.
4. **Compute scales:** `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)` per channel `j`.
5. **Apply scales:**
   - Divide weight columns by `s`: `W_smooth = W / s` (makes weights harder to quantize in those channels, easier in others).
   - Divide activation scale by `1/s` (effectively, the inverse scale is folded into the preceding layer's output projection weights).
6. **Remove hooks** and proceed to `QuantizationModifier`.

---

## Recipes

**SmoothQuant + W8A8 INT8:**
```yaml
smooth_quant_stage:
  smooth_quant_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.5
      ignore: ["lm_head"]

quantization_stage:
  quantization_modifiers:
    QuantizationModifier:
      targets: ["Linear"]
      scheme: W8A8
      ignore: ["lm_head"]
```

**Python:**
```python
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

---

## Practical Notes

- **Mappings are required** for correct smoothing. If not specified, the modifier attempts inference, but incorrect mappings degrade accuracy significantly.
- Standard mapping for decoder-only transformers (GPT/LLaMA style): pair each `q/k/v/out_proj` with the adjacent layer norm.
- SmoothQuant is most impactful for W8A8 quantization; for W4A16 (weight-only), AWQ is generally preferred.
- The `log_equalization` algorithm variant uses log-scale equalisation rather than linear smoothing.

---

## .NET Reimplementation Feasibility

All operations are straightforward:
- Forward hooks for activation collection: `register_forward_hook()` — ✅ (TorchSharp 2024)
- Per-channel max reduction: `torch.amax()` over batch/sequence dims — ✅
- Per-channel division of weight tensor: indexing + in-place divide — ✅
- DDP all-reduce (if needed): `torch.distributed` — ❌ (no DDP in TorchSharp; single-GPU only)

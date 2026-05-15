# AWQ — Activation-Aware Weight Quantization

**Paper:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)  
**Location:** `src/llmcompressor/modifiers/transform/awq/` (transform); `src/llmcompressor/modifiers/awq/` (compatibility shim)  
**Class:** `AWQModifier(Modifier)` in `transform/awq/base.py`

---

## What It Does

Identifies weight channels that correspond to large activation values (salient channels). These channels, if quantized naively, would contribute disproportionately to quantization error. AWQ rescales them: multiplying by a scale factor `s > 1` so the activations shrink (easier to represent) and the corresponding weight channel is divided by `s` (a mathematical identity). The scale is chosen to minimise MSE of the quantized output.

Unlike SmoothQuant (which targets activation quantization), AWQ is primarily for **weight-only quantization** (W4A16, W4A8).

AWQ is a **pre-processing transform** — recipe validation enforces that an `AWQModifier` must be followed by a `QuantizationModifier`.

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mappings` | `list[AWQMapping]` | — | Activation layers to smooth → balance layer mappings |
| `offload_device` | `str \| None` | CPU for MoE, else `None` | Device for activation caching |
| `duo_scaling` | `bool \| "both"` | `True` | Use both activation and weight stats for scaling |
| `n_grid` | `int` | `20` | Grid search points for optimal scale factor |
| `targets` / `ignore` | `list` | — | Layer targeting |

---

## Algorithm

1. **Attach activation cache hooks** on the activation layers in `mappings`.
2. **Calibration pass:** forward the model on calibration samples, collecting per-channel activation stats.
3. **Grid search:** for each of `n_grid` candidate scale ratios in `[0, 1]`:
   - Apply `s` to the smooth (activation-adjacent) layer's weights: `W_smooth ← W_smooth · diag(s)`
   - Apply `1/s` to the balance layer's weights: `W_balance ← W_balance · diag(1/s)`
   - Simulate quantization and measure output MSE
4. **Apply optimal scale:** the `s` that minimised MSE is permanently folded into the weights.
5. **Remove hooks.**

**Mathematical identity preserved:** `Y = (X · diag(s)^-1) · (diag(s) · W)` — activations shrink, weights grow in those channels, net output is unchanged.

---

## Recipes

**AWQ + W4A16:**
```yaml
awq_stage:
  awq_modifiers:
    AWQModifier:
      duo_scaling: true
      n_grid: 20

gptq_stage:
  gptq_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
```

**AWQ + RTN W4A16 (no calibration for quantization):**
```yaml
awq_stage:
  awq_modifiers:
    AWQModifier:
      duo_scaling: true

quantization_stage:
  quantization_modifiers:
    QuantizationModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
```

---

## Practical Notes

- AWQ is particularly effective for W3A16 and W4A16; it helps preserve accuracy on salient weight channels.
- `duo_scaling: true` uses both activation stats and weight magnitude in the grid search — generally better accuracy.
- `n_grid: 20` is the default; more grid points = more accurate scale search but slower calibration.
- For MoE models, `offload_device: "cpu"` reduces peak VRAM during activation caching.

---

## .NET Reimplementation Feasibility

- Forward hooks: ✅ (TorchSharp 2024)
- Per-channel activation stats: ✅ `torch.amax()`
- Grid search: simple loop over scale candidates — ✅
- Fake-quantize for MSE eval: requires `quantize_per_tensor`/`quantize_per_channel` — ✅ (added 0.107.0)
- In-place weight scaling: ✅

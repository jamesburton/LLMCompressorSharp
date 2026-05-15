# LLM Compressor — Deployment, Supported Models & Hardware

---

## vLLM Integration

LLM Compressor's output format is `compressed-tensors` (built on safetensors). vLLM natively reads this format with zero conversion.

### Workflow

```python
# 1. Compress
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

oneshot(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset="ultrachat-200k",
    recipe=GPTQModifier(scheme="W4A16"),
    output_dir="llama3-8b-w4a16-gptq",
)

# 2. Deploy with vLLM — no additional steps
from vllm import LLM
llm = LLM("llama3-8b-w4a16-gptq")
```

### What vLLM Gains

- INT4/INT8/FP8 weight loading → reduced GPU memory footprint
- Faster weight loading bandwidth
- Lower-precision tensor core utilisation (Ampere/Hopper/Blackwell hardware)
- KV cache quantization for longer context at the same memory budget

### Quantization Schemes by GPU Generation

| GPU Generation | Supported Schemes |
|---|---|
| Turing/Ampere | `W4A16`, `W8A16`, `W8A8-INT8` |
| Ampere | FP8 via emulation |
| Hopper (H100/H200) | `W8A8-FP8`, `FP8_DYNAMIC`, native FP8 |
| Hopper/Blackwell | `NVFP4`, `MXFP4`, `MXFP8` |

---

## Supported Model Architectures

### Architecture-Specific Patches (`src/llmcompressor/modeling/`)

These models have explicit code for sequential pipeline compatibility and MoE calibration:

| Model Family | Files | Notes |
|---|---|---|
| **Llama** | `llama4.py` | Llama 3.x, Llama 4 |
| **Qwen** | `qwen3_moe.py`, `qwen3_5_moe.py`, `qwen3_vl_moe.py`, `qwen3_next_moe.py` | Qwen3, Qwen3.5, Qwen-VL, Qwen-MoE |
| **DeepSeek** | `deepseek_v3.py`, `deepseekv32/` | DeepSeek-V3, V4-Flash |
| **Gemma** | `gemma4.py` | Gemma 4 |
| **Granite** | `granite4.py` | IBM Granite 4 |
| **GLM** | `glm4_moe.py`, etc. | GLM-4-MoE, GLM-4-MoE-Lite, GLM-MoE-DSA |
| **Generic MoE** | `afmoe.py`, `moe_context.py`, `MoECalibrationModule` | Any MoE model |

### Generally Supported (No Explicit Patch Needed)

Any HuggingFace `AutoModelForCausalLM`-loadable model works with:
- `QuantizationModifier` (RTN/SmoothQuant/AWQ + PTQ)
- `BasicPipeline`

Architecture patches are needed for:
- **Sequential pipeline** (model must be PyTorch FX traceable)
- **MoE calibration correctness** (without a `MoECalibrationModule`, experts may be calibrated incorrectly → NaNs)

### MoE Models — Important

For MoE models (DeepSeek-MoE, Mixtral, Qwen-MoE, etc.), you must import the model class from `llmcompressor.modeling` rather than bare `transformers`:

```python
from llmcompressor.modeling import AutoModelForCausalLM  # NOT from transformers

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3", ...)
```

Without this, the MoE routing during calibration may not cover all experts.

### Multimodal Support

- Vision-language: Qwen3-VL, LLaVA-style models
- Examples: `examples/multimodal_vision/`, `examples/multimodal_audio/`

---

## Hardware Requirements

### Python / Library Versions

| Dependency | Minimum |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.10.0 |
| Transformers | ≥ 4.56.1 |
| Datasets | ≥ 4.8.4 |
| Accelerate | ≥ 1.6.0 |
| compressed-tensors | ≥ 0.14.0 |
| NumPy | ≥ 2.0.0 |
| auto-round | ≥ 0.10.2 (AutoRound only) |

### GPU Requirements

- **Any CUDA GPU:** Basic quantization (RTN, SmoothQuant)
- **Sequential pipeline (default):** Enables compressing models larger than GPU VRAM — e.g., Llama 70B (>80GB) on a single A100 80GB — by loading one transformer block at a time
- **FP8:** Hopper (H100) or newer for native; emulated on older GPUs
- **NVFP4/MXFP4/MXFP8:** Blackwell (B100/B200) or Hopper for native execution
- **Multi-GPU (DDP):** For very large models (Llama 405B), Distributed Data Parallel compression across multiple GPUs

### Memory Patterns

| Technique | Requirement |
|---|---|
| Sequential pipeline | Only one transformer block on GPU at a time; activations cached to CPU |
| `offload_hessians=True` | Hessian matrices stored on CPU, loaded per-module |
| Disk offloading | Model shards stored on disk, loaded per-module; handles models exceeding GPU+CPU RAM |

---

## Output Format: `compressed-tensors`

The `compressed-tensors` library is an open specification layered on top of safetensors. It stores:
- Quantized weight tensors (INT4/INT8/FP8, packed or unpacked)
- Per-layer scale and zero-point tensors
- Sparsity masks (for pruned models)
- Quantization config metadata in `config.json`

To load compressed models outside vLLM: install the `compressed-tensors` Python package. The safetensors file format is language-agnostic and can be read directly from .NET via `TorchSharp.PyBridge.load_safetensors()`.

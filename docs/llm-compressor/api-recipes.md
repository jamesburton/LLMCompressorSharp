# LLM Compressor — Public API & Recipes

---

## `oneshot()` — Primary Entry Point

```python
from llmcompressor import oneshot

oneshot(
    model: str | PreTrainedModel,
    config_name: str | None = None,
    tokenizer: str | PreTrainedTokenizerBase | None = None,
    processor: str | ProcessorMixin | None = None,
    output_dir: str | None = None,
    log_dir: str | None = None,
    # recipe args:
    recipe: str | Modifier | list[Modifier] | None = None,
    recipe_args: dict | None = None,
    stages: list[str] | None = None,
    # dataset args:
    dataset: str | None = None,
    batch_size: int = 1,
    max_seq_length: int = 512,
    num_calibration_samples: int = 512,
    # model loading args:
    dtype: str | torch.dtype = "auto",
    device_map: str | dict = "auto",
    **kwargs,
) -> PreTrainedModel
```

**Steps:**
1. Load model + tokenizer from HuggingFace Hub or disk
2. Load calibration dataset
3. Initialize `CompressionSession` with the recipe
4. Run the appropriate pipeline (sequential / independent / data-free / basic)
5. Freeze quantization parameters
6. Return the modified model (call `model.save_pretrained(output_dir, save_compressed=True)` to persist)

---

## Typical Workflow

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset="ultrachat-200k",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("llama3-8b-w4a16-gptq", save_compressed=True)
tokenizer.save_pretrained("llama3-8b-w4a16-gptq")
```

---

## Recipe Formats

### YAML Recipe File

```yaml
# Basic schema:
# [stage_name]_stage:
#   [group_name]_modifiers:
#     ModifierType:
#       param: value

gptq_stage:
  gptq_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: W4A16
      ignore: ["lm_head"]
      block_size: 128
      dampening_frac: 0.01
```

**Multi-stage (SmoothQuant + W8A8):**
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

**Sparse + quantised:**
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

### Loading a Recipe File

```python
oneshot(model=model, dataset="...", recipe="path/to/recipe.yaml")
```

### Python Recipe Construction

```python
from llmcompressor.recipe import Recipe

# From YAML file path
recipe = Recipe.create_instance("recipe.yaml")

# From YAML string
recipe = Recipe.create_instance(yaml_string)

# From modifier instances (auto-converted)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    QuantizationModifier(scheme="W8A8"),
]

# From dict
recipe = Recipe.from_dict({...})
```

---

## Pipelines

Override with `oneshot(..., pipeline="basic")`.

| Pipeline | Use case | Behaviour |
|---|---|---|
| `"sequential"` | **Default** for GPTQ/SparseGPT | PyTorch FX partitions model into subgraphs; processes one subgraph at a time; activations cached to CPU |
| `"independent"` | Per-modifier isolation | Each modifier runs its own calibration epoch |
| `"datafree"` | No dataset needed | Dispatches model to GPU, fires lifecycle events only |
| `"basic"` | Model fits in single GPU VRAM | Disables sequential onloading; full model in memory |

---

## Key Classes

| Class | Import | Purpose |
|---|---|---|
| `oneshot` | `llmcompressor` | Main entry point |
| `Modifier` | `llmcompressor.modifiers.modifier` | Base class for all algorithms |
| `GPTQModifier` | `llmcompressor.modifiers.gptq` | GPTQ quantization |
| `QuantizationModifier` | `llmcompressor.modifiers.quantization` | RTN/PTQ/QAT quantization |
| `SmoothQuantModifier` | `llmcompressor.modifiers.transform.smoothquant` | SmoothQuant |
| `AWQModifier` | `llmcompressor.modifiers.transform.awq` | AWQ transform |
| `AutoRoundModifier` | `llmcompressor.modifiers.autoround` | AutoRound |
| `SparseGPTModifier` | `llmcompressor.modifiers.pruning.sparsegpt` | SparseGPT pruning |
| `WandaPruningModifier` | `llmcompressor.modifiers.pruning.wanda` | WANDA pruning |
| `MagnitudePruningModifier` | `llmcompressor.modifiers.pruning.magnitude` | Magnitude pruning |
| `SpinQuantModifier` | `llmcompressor.modifiers.transform.spinquant` | SpinQuant rotation |
| `Observer` | `llmcompressor.observers.base` | Calibration statistics base |
| `Recipe` | `llmcompressor.recipe` | Recipe parsing |
| `CompressionSession` | `llmcompressor.core.session` | Session lifecycle |
| `State` | `llmcompressor.core.state` | Compression state |

---

## Calibration Dataset Options

Pass to `oneshot(..., dataset=...)`:
- HuggingFace dataset name: `"ultrachat-200k"`, `"open-platypus"`, `"c4"`, `"wikitext-2-raw-v1"`
- A list of pre-tokenized samples
- A custom `DataLoader`
- `None` for data-free compression (requires `pipeline="datafree"`)

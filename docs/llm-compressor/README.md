# LLM Compressor — Overview & Architecture

**GitHub:** https://github.com/vllm-project/llm-compressor  
**Docs:** https://docs.vllm.ai/projects/llm-compressor  
**License:** Apache 2.0  
**Install:** `pip install llmcompressor`  
**Stats (May 2026):** ~3.2k stars, ~514 forks, 2,996 commits, actively maintained by the vLLM project.

---

## What It Is

LLM Compressor is a transformers-compatible Python library that applies post-training compression to large language models for deployment with [vLLM](https://github.com/vllm-project/vllm). It addresses four LLM deployment obstacles:

- **GPU/infrastructure costs** — 50–75% memory reduction via quantization
- **Response latency** — faster weight loading at lower precision
- **Request throughput** — lower-precision tensor cores
- **Energy consumption** — smaller data movement at inference

**Primary entry point:** `oneshot()` — a single function call that loads a model, applies a compression recipe, and saves in `compressed-tensors` format that vLLM reads natively.

---

## Architecture

### Module Layout (`src/llmcompressor/`)

| Module | Purpose |
|---|---|
| `entrypoints/` | `oneshot.py` — public API wrapper around the `Oneshot` class |
| `core/` | `CompressionSession`, `CompressionLifecycle`, `State` — lifecycle management |
| `modifiers/` | All compression algorithms as `Modifier` subclasses |
| `observers/` | Calibration statistics: MinMax, MSE, iMatrix |
| `pipelines/` | `sequential/`, `independent/`, `data_free/`, `basic/` |
| `recipe/` | `Recipe`, `recipe.py`, `metadata.py` — recipe parsing (YAML/JSON) |
| `transformers/` | HuggingFace Transformers integration |
| `modeling/` | Architecture-specific patches (Llama, Qwen, DeepSeek, GLM, Gemma, Granite, MoE) |
| `datasets/` | Calibration data loading utilities |
| `pytorch/` | PyTorch-specific implementations |
| `args/` | Argument handling |

---

### Core Abstraction: `Modifier`

Every algorithm is a `Modifier` (Pydantic model with `extra="forbid"`):

```python
class Modifier(ModifierInterface, HooksMixin):
    index: int
    group: str
    start: float | None
    end: float | None
    update: float | None
    # state flags
    initialized_: bool
    finalized_: bool
    started_: bool
    ended_: bool

    # lifecycle hooks (override in subclasses):
    def on_initialize(self, state, **kwargs) -> bool: ...
    def on_start(self, state, event, **kwargs): ...
    def on_update(self, state, event, **kwargs): ...
    def on_end(self, state, event, **kwargs): ...
    def on_finalize(self, state, **kwargs) -> bool: ...
    def on_event(self, state, event, **kwargs): ...
```

### Session / Lifecycle

`CompressionSession` orchestrates compression:
1. `initialize(recipe, model, data, optimizer)` — sets up all modifiers
2. `event(event_type)` — triggers modifier callbacks (batch, epoch, etc.)
3. `finalize()` — cleanup
4. `reset()` / `reset_stage()` — restart

`State` holds: model, teacher model, optimizer, loss, datasets (train/val/test/calibration), hardware info (device, DDP rank), current batch.

### Observer Abstraction

`Observer` computes quantization scales/zero-points from calibration data:

- `update_statistics_from_observed(tensor)` — abstract; updates internal stats
- `get_qparams()` → `{scale, zero_point, global_scale}`
- `forward(tensor)` — flattens and updates stats
- `fuse(observers)` — links observers for shared global_scale (TENSOR_GROUP strategy)
- `sync_activation_stats()` — DDP all-reduce synchronisation

**Concrete observers:**

| Observer | Strategy |
|---|---|
| `MemorylessMinMaxObserver` | Per-batch min/max only (no memory) |
| `StaticMinMaxObserver` | Accumulates across all calibration samples |
| `MinMaxObserver` | EMA with `averaging_constant=0.01` |
| `MemorylessMSEObserver` | Grid-search optimal range per batch |
| `MovingAverageMSEObserver` | EMA over MSE grid-search results |
| `iMatrixObserver` | Importance matrix observer |

---

### Pipelines

| Pipeline | Use | Description |
|---|---|---|
| `SequentialPipeline` (`"sequential"`) | Default for GPTQ/SparseGPT | Partitions model into subgraphs via PyTorch FX tracing; two-pass per subgraph (calibrate then capture output); activations cached to CPU |
| `IndependentPipeline` (`"independent"`) | Per-modifier isolation | Each modifier gets its own calibration epoch |
| `DataFreePipeline` (`"datafree"`) | No dataset needed | Dispatches model to GPU and fires lifecycle events |
| `BasicPipeline` (`"basic"`) | Model fits in VRAM | Disables sequential onloading; full model in memory |

Override: `oneshot(..., pipeline="basic")`

---

## Related Docs

- [Algorithms](./algorithms/) — detailed notes on each compression algorithm
- [API & Recipes](./api-recipes.md) — `oneshot()`, recipe YAML, Python API
- [Deployment](./deployment.md) — vLLM integration, supported models, hardware requirements

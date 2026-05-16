# Cache Conventions

LLMCompressorSharp must avoid duplicating model and dataset storage across the developer's machine. This document is the **canonical reference** that every loader/downloader in this repository follows.

**Principle:** Prefer reading from existing platform caches over re-downloading. Write into the HuggingFace cache by default so downloads are discoverable by other HF-aware tooling. Never put model bytes in our own private directory.

---

## 1. Cache Locations We Honour

### 1.1 HuggingFace (canonical default)

**Resolution precedence** (matches `huggingface_hub`):
1. `$HF_HUB_CACHE` — explicit hub-cache override
2. `$HF_HOME/hub` — HF root + `/hub` subdir
3. `$XDG_CACHE_HOME/huggingface/hub` — XDG base directory
4. `~/.cache/huggingface/hub` (Unix) / `%USERPROFILE%\.cache\huggingface\hub` (Windows) — fallback

**Layout:**
```
hub/
  models--{org}--{repo}/
    snapshots/{revision}/        ← symlink farm to blobs
      config.json
      model.safetensors            (or model.safetensors.index.json + shards)
      tokenizer.json
      tokenizer_config.json
    blobs/
      {sha256}                     ← actual file bytes
    refs/
      main                         ← current commit SHA
  datasets--{org}--{name}/         ← analogous structure for datasets
```

**Implementation in this repo:** `LLMCompressorSharp.Transformers.HuggingFaceCache` (Phase 0). All HF loaders write into `snapshots/{revision}/`. Downloads SHOULD go through `huggingface_hub`-compatible paths so Python tools see them too.

### 1.2 Ollama (read-only discovery)

**Default location:** `$OLLAMA_MODELS` if set, else `~/.ollama/models/` (Unix) or `%USERPROFILE%\.ollama\models\` (Windows).

**Layout:**
```
models/
  blobs/sha256-{hash}              ← content-addressed
  manifests/registry.ollama.ai/library/{model}/{tag}
                                   ← JSON manifest with layer digests
```

Ollama stores GGUF files in `blobs/`. Conversion from GGUF → safetensors (or vice versa) is non-trivial and may require an external tool (llama.cpp's `convert_hf_to_gguf.py` or `convert_gguf_to_hf.py`).

**LLMCompressorSharp approach:** Detect Ollama-cached models for **read-only reference** during compression validation (e.g., "compare my compressed output to the user's existing Ollama copy"). Do not write into the Ollama cache.

### 1.3 llama.cpp (read-only discovery)

**Default location:** No standard — varies by install. Common: `~/Models/`, `~/llama.cpp/models/`, or wherever the user explicitly placed GGUF files.

**LLMCompressorSharp approach:** Accept user-supplied paths. No automatic discovery.

### 1.4 PyTorch Hub

**Default location:** `$TORCH_HOME/hub` or `~/.cache/torch/hub/`.

LLMCompressorSharp does not currently use `torch.hub` paths.

### 1.5 vLLM downloads

vLLM uses the HuggingFace cache directly (it depends on `huggingface_hub`). Models compressed by LLMCompressorSharp and written into the HF cache are immediately discoverable by vLLM.

---

## 2. Rules for Loaders and Writers

### 2.1 Reading

When loading a model by HuggingFace repo id (e.g. `HuggingFaceTB/SmolLM2-135M`):

1. **Resolve the HF cache root** via the precedence above.
2. **Look up the local revision** in `models--{org}--{repo}/refs/main` (or the user-supplied revision).
3. **Locate the snapshot** at `models--{org}--{repo}/snapshots/{revision}/`.
4. **If missing**, download from HuggingFace Hub into the cache using the standard layout (blobs + symlinks).
5. **Optional fallback discovery:** if a same-named model exists in `~/.ollama/models/` or a user-configured `--external-cache` directory, log it and offer to convert/reuse rather than re-downloading. Default behaviour: download via HF.

When loading a dataset:
- Same precedence, replacing `models--` with `datasets--`.
- `huggingface_hub`-compatible.

### 2.2 Writing

When saving a compressed model:

1. **Default output location:** A new entry in the HF cache: `models--{org}--{repo}--compressed-{recipe-name}/snapshots/local-{timestamp}/`. This makes the result discoverable by HF-aware tooling.
2. **Explicit `--output` override:** If the user passes `--output ./somewhere`, write there instead. **Always** write a self-contained directory: `config.json`, weights (safetensors), `quantization_config.json`, tokenizer files copied from source.
3. **Never** scatter outputs across multiple directories. A compressed model is a single self-contained snapshot.

### 2.3 Cross-platform path handling

All paths are managed via `System.IO.Path.Combine` so Windows backslashes and Unix forward slashes resolve correctly. Environment variable lookups use `Environment.GetEnvironmentVariable` (cross-platform).

### 2.4 Symlinks

HuggingFace's cache uses symlinks from `snapshots/` to `blobs/`. On Windows this requires Developer Mode or admin elevation. If symlink creation fails, **fall back to copying** the blob into the snapshot — wasteful but correct. Log a warning so the user knows.

---

## 3. CLI Flags

The `llmc` CLI (Phase 6) will expose:

| Flag | Default | Purpose |
|---|---|---|
| `--cache-dir <path>` | unset | Override the resolved HF cache root for this invocation. Sets `HF_HUB_CACHE` for child processes. |
| `--external-cache <path>` | unset | Additional read-only directory checked before downloading. Repeatable. |
| `--output <path>` | (HF cache) | Output directory for the compressed model. |
| `--offline` | false | Refuse to download. Fail if a needed file isn't cached. |
| `--revision <ref>` | `main` | HF revision to load. |
| `--hf-token <token>` | `$HF_TOKEN` | Auth token for gated models. |

---

## 4. Test-Mode Caching

Unit and integration tests **must not write into the developer's real HF cache** by default. The `LLMCompressorSharp.Tests` project:

- For pure-CPU tests (CI): synthetic state only; never load actual model files.
- For integration tests that need a real model: read from `$LLMC_TEST_CACHE` if set (CI sets this to a workspace-scoped path); else use the developer's HF cache **read-only**; never download in tests — tests skip with a clear message if the model is missing.

---

## 5. Implementation Status

| Component | Phase | Status |
|---|---|---|
| `HuggingFaceCache` path resolver | Phase 0 | ✅ Shipped (v0.0.1-alpha) |
| HuggingFace download + snapshot population | Phase 3b | ⏳ Planned |
| Sharded safetensors loader | Phase 3b | ⏳ Planned |
| Compressed-output writer to HF cache | Phase 5 | ⏳ Planned |
| Ollama read-only discovery | Post-1.0 | ⏳ Backlog |
| `--external-cache` flag | Phase 6 | ⏳ Planned |

---

## 6. References

- HuggingFace cache layout: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Ollama model storage: https://github.com/ollama/ollama/blob/main/docs/faq.md#where-are-models-stored
- vLLM HF integration: https://docs.vllm.ai/en/latest/models/supported_models.html
- XDG Base Directory: https://specifications.freedesktop.org/basedir-spec/latest/

# LLMCompressorSharp — Design Spec

**Date:** 2026-05-15
**Status:** Approved
**Supersedes:** none

A .NET 10 / C# reimplementation of the Python [llm-compressor](https://github.com/vllm-project/llm-compressor) library, built on TorchSharp. Targets the LLaMA architecture family first; expands to other transformer architectures incrementally.

---

## 1. Goals and Non-Goals

### Goals

- A NuGet-distributable .NET library and CLI tool (`llmc`) that compresses LLaMA-family LLMs.
- Implement the core llm-compressor algorithms: RTN, SmoothQuant, AWQ, GPTQ, SparseGPT, WANDA, Magnitude Pruning.
- Use the **official TorchSharp NuGet** (currently 0.107.0, wraps LibTorch 2.10.0 / CUDA 12.8). No source fork unless a true blocker forces it.
- Real HuggingFace-hosted models in tests (SmolLM2-135M, TinyLlama-1.1B, Llama-3.2-1B, SmolLM2-135M-Instruct) — not hand-rolled clones — so we can verify outputs against reference values.
- Track every TorchSharp gap we hit (or notice) in `PR_TO_TORCHSHARP.md`, split into **Required** (we worked around it) and **Proposed** (would benefit the community).
- HuggingFace cache compatibility: store and resolve models from the standard `~/.cache/huggingface/hub/` layout so cached weights are shared with the Python ecosystem.

### Non-Goals (v0.6.0)

- Architectures beyond the LLaMA family (Qwen-MoE, DeepSeek, Granite, GLM, Gemma — deferred to post-1.0).
- Multi-GPU (DistributedDataParallel) compression. TorchSharp has no DDP; single-GPU + CPU/disk offload is the production path for ≤70B models.
- `torch.compile`-accelerated AutoRound. Implementable but unviably slow without compile; tracked as a known limitation.
- ONNX export from .NET (use ONNX Runtime in .NET for inference, but export via Python).
- A full `transformers` substitute. The `Transformers` project grows model-family by model-family as needed.

---

## 2. Architecture

### 2.1 Repository Layout

```
LLMCompressorSharp/                         ← repo root (renamed from llm-compressor-tests)
├── LLMCompressorSharp.slnx                 ← .NET 10 XML solution
├── README.md
├── PR_TO_TORCHSHARP.md                     ← upstream contribution tracker
├── Directory.Build.props                   ← common MSBuild props
├── Directory.Packages.props                ← central package versions
├── .editorconfig                           ← formatting + StyleCop config
├── global.json                             ← pin .NET 10 SDK
├── docs/
│   ├── INDEX.md
│   ├── llm-compressor/...                  ← research notes (existing)
│   ├── dotnet-10/...                       ← research notes (existing)
│   ├── torchsharp/...                      ← research notes (existing)
│   ├── llmcompressorsharp/...              ← feasibility + roadmap (existing)
│   └── superpowers/specs/2026-05-15-llmcompressorsharp-design.md  ← this file
├── src/
│   ├── LLMCompressorSharp.TorchExtensions/ ← gap-filling extensions
│   ├── LLMCompressorSharp.Core/            ← Modifier, Observer, Recipe, Session
│   ├── LLMCompressorSharp.Transformers/    ← LLaMA arch + HF loader + tokenizer
│   └── LLMCompressorSharp.Cli/             ← NuGet id LLMCompressorSharp, command llmc
└── tests/
    └── LLMCompressorSharp.Tests/           ← xUnit
```

### 2.2 Project Responsibilities

#### `LLMCompressorSharp.TorchExtensions`

Fills TorchSharp API gaps without forking. Every gap is annotated with `// TODO(PR_TO_TORCHSHARP <id>)` linking back to `PR_TO_TORCHSHARP.md`.

Contents:
- `FakeQuantizeFunction` — custom autograd via `SingleTensorFunction<T>` providing differentiable fake-quantization for AWQ inner loops.
- `MinMaxObserver`, `PerChannelMinMaxObserver`, `MovingAverageMinMaxObserver`, `MSEObserver` — calibration-stat collectors as TorchSharp-compatible modules.
- `Int4PackedTensor` helper — FP32-backed packed-INT4 simulation until upstream `QInt4` lands.
- `DisposeScopeExtensions` — ergonomic helpers for nested GPU memory scoping in calibration loops.
- `NvmlMemoryStats` — optional P/Invoke to `nvml.dll` for VRAM monitoring (`memory_allocated`, `memory_reserved`).
- Public APIs are PascalCase. Internal calls into TorchSharp keep snake_case at the boundary; consumers never see snake_case.

#### `LLMCompressorSharp.Core`

Algorithm framework, model-agnostic. References `TorchExtensions` and TorchSharp.

Contents:
- `IModifier` + `ModifierBase` lifecycle (`Initialize`, `OnStart`, `OnBatch`, `OnEnd`, `Finalize`).
- `CompressionSession` + `CompressionState` (model, calibration enumerator, per-modifier state).
- Algorithm implementations:
  - `RtnModifier` (per-tensor and per-channel RTN; INT8, INT4 fake-quant, FP16)
  - `SmoothQuantModifier` (forward hooks + channel scaling)
  - `AwqModifier` (grid search via `FakeQuantizeFunction`)
  - `GptqModifier` (Hessian + Cholesky + block-wise column quant + error propagation)
  - `SparseGptModifier` (same infrastructure with mask selection; unstructured + 2:4)
  - `WandaModifier` (saliency = |w| × ‖x‖₂; faster pruning)
  - `MagnitudePruningModifier`
- `Recipe` POCO + YAML deserialisation (YamlDotNet, polymorphic modifier type discriminator).
- Output writers: safetensors (via `TorchSharp.PyBridge.save_safetensors`) + `quantization_config.json` compatible with the `compressed-tensors` spec.
- Recipe validation: enforces modifier ordering (AWQ before QuantizationModifier, etc.).

#### `LLMCompressorSharp.Transformers`

Model architectures and loaders. This grows into our incremental `transformers` substitute. Phase 3 covers LLaMA only; later phases add Qwen, DeepSeek, etc.

Contents (v0.6.0):
- `Architectures/Llama/`:
  - `LlamaConfig`
  - `LlamaRMSNorm`
  - `LlamaRotaryEmbedding` (RoPE)
  - `LlamaMLP` (gate_proj + up_proj + down_proj + SiLU)
  - `LlamaAttention` (q/k/v/o projections + grouped query attention)
  - `LlamaDecoderLayer` (attention + MLP + residual)
  - `LlamaForCausalLM` (embedding + N decoder layers + RMSNorm + lm_head)
- `Loading/HuggingFaceLoader`:
  - Resolves models via standard HF cache layout (`$HF_HOME` or `~/.cache/huggingface/hub/`, directory pattern `models--{org}--{repo}/snapshots/{revision}/`)
  - On miss, downloads via Hub API + writes to the shared cache
  - Sharded safetensors support (reads `model.safetensors.index.json`)
- `Loading/LlamaConfigParser` — `config.json` → `LlamaConfig`.
- `Tokenization/LlamaTokenizer` — thin wrapper over `Microsoft.ML.Tokenizers`.
- `IArchitecture` interface for plugging in future families.

#### `LLMCompressorSharp.Cli`

The CLI tool. NuGet package id `LLMCompressorSharp` (short, matches the org's headline product); tool command `llmc` (so `dnx llmc compress ...` is concise).

Contents:
- Standard `.csproj` packaged as a .NET tool (`PackAsTool=true`, `ToolCommandName=llmc`). Not file-based — file-based apps can be packed as tools in .NET 10 but the multi-command structure benefits from a real project layout.
- `System.CommandLine` for argument parsing.
- Built-in recipe presets: `gptq-w4a16`, `smoothquant-w8a8`, `awq-w4a16`, `sparse-2-4`.
- Progress reporting (Spectre.Console).
- HF Hub token support for gated models (`$HF_TOKEN`).
- Memory pre-flight: estimate required VRAM/RAM before starting, warn if insufficient.

#### `LLMCompressorSharp.Tests`

xUnit. CPU-only by default; GPU tests opt-in via `[Trait("Category", "Gpu")]`.

Contents:
- Synthetic-tensor unit tests for observers, modifier lifecycle, recipe parsing.
- Integration tests against real HuggingFace models loaded into the standard HF cache.

### 2.3 Test Model Tiers

All four models share the LLaMA architecture and are ungated (no HF auth needed for CI).

| Tier | Models | CI category | Validates |
|---|---|---|---|
| Unit | None — synthetic tensors | default | Algorithm correctness on tiny tensors |
| Integration | `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolLM2-135M-Instruct` | default (CPU CI) | Loading, forward pass, compression end-to-end |
| Accuracy | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `unsloth/Llama-3.2-1B` | `Gpu` (gated) | Perplexity within tolerance vs FP16 baseline |
| Generation | `HuggingFaceTB/SmolLM2-135M-Instruct` | optional CPU / `Gpu` | Compressed model still produces coherent output |

Model weights are **not committed to git**. They live in the HF cache, shared with any other HF-based tools the developer uses. A `download-test-models.ps1` script primes the cache in CI.

---

## 3. TorchSharp Contribution Tracking

`PR_TO_TORCHSHARP.md` at repo root tracks every gap we hit or notice, split into:

- **Required** — we worked around it in `TorchExtensions`. We owe the upstream a PR once the workaround is stable and tested.
- **Proposed** — we noticed it would benefit the community but haven't needed it ourselves.

Each entry includes the files/changes a PR would touch in `dotnet/TorchSharp`, our local workaround location, and PR readiness (estimate to clean up for upstream).

Inline code references each entry with `// TODO(PR_TO_TORCHSHARP <id>)` so we can find every site when we prepare the actual PR.

Initial known entries:
- **R-001** FakeQuantize autograd function
- **R-002** `MinMaxObserver` / `PerChannelMinMaxObserver` modules
- **R-003** True packed `QInt4` dtype (likely fork-required — flagged for escalation)
- **P-001** `torch.cuda.memory_allocated()` / `memory_reserved()` exposure
- **P-002** `torch.fx`-style symbolic tracing (large scope — likely permanent gap)

### Fork Escalation Trigger

If an entry is `Required` *and* cannot be worked around in `TorchExtensions` (e.g., needs a new native binding), we:
1. Write `docs/llmcompressorsharp/fork-decision-<entry-id>.md` proposing the pivot.
2. Add `dotnet/TorchSharp` fork at `external/TorchSharp/` as a git submodule.
3. Build the patched NuGet locally and switch `Directory.Packages.props` to a local feed.
4. Continue work; submit the corresponding PR upstream and remove the fork once merged.

---

## 4. Build, CI & Conventions

### 4.1 Solution-level configuration

- `Directory.Build.props`: `TargetFramework=net10.0`, `LangVersion=14`, `Nullable=enable`, `ImplicitUsings=enable`, `TreatWarningsAsErrors=true`.
- `Directory.Packages.props`: central versions for TorchSharp 0.107.0+, TorchSharp.PyBridge 1.4.3, YamlDotNet, Microsoft.ML.Tokenizers, System.CommandLine, Spectre.Console, xUnit.
- `.editorconfig`: StyleCop rules enabled. **Explicit suppressions for snake_case at `Architectures/*` and `TorchExtensions/*`** so TorchSharp calls don't fight the linter. Public APIs use PascalCase elsewhere.
- `global.json`: pin to .NET 10 SDK feature band.
- `LLMCompressorSharp.slnx`: XML-format solution (preferred in .NET 10 over legacy `.sln`).

### 4.2 Dependency Choice

Project references use the backend-agnostic `TorchSharp` package. Consumers add either `TorchSharp-cpu` or one of the `TorchSharp-cuda-*` packages to choose a backend at deploy time. CI matrix runs CPU only; GPU job is self-hosted and opt-in.

### 4.3 GitHub Actions

- `ci.yml` — push/PR triggers; matrix `{windows-latest, ubuntu-latest}` × `{cpu}`; caches the HF model cache between runs; runs `dotnet test --filter Category!=Gpu`.
- `gpu-tests.yml` — manual dispatch + nightly; self-hosted runner with CUDA GPU; runs `--filter Category=Gpu`.
- `release.yml` — manual dispatch; tags + publishes NuGet packages to nuget.org.
- HF cache cache-key includes the SHA of the test-model fixture revision list so cache invalidates when fixtures change.

### 4.4 Versioning & Branching

- `main` always green.
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
- Branch naming: `feature/<phase>-<slug>` and `bugfix/<phase>-<slug>`. This is a GitHub repo (no Azure DevOps ticket numbers) so `<phase>` is the build-sequence phase number (`0`–`6`) or the issue number when one exists. Same `feature/` / `bugfix/` prefix discipline as the Azure DevOps repos to keep habits consistent.
- `MinVer` for git-tag-driven SemVer.
- 0.x while LLaMA is the only architecture. 1.0 when ≥2 model families ship and at least GPTQ W4A16 reaches accuracy parity with Python llm-compressor.

### 4.5 Coding Conventions

- Public APIs PascalCase per .NET conventions. `TorchExtensions` translates to snake_case at the boundary; consumers never see snake_case.
- XML doc comments required on all public types/members.
- `using` and `DisposeScope` audited in code review on every GPU tensor allocation.
- Calibration loops MUST be in `using (var scope = torch.NewDisposeScope())`. Reviewers reject PRs that violate this.

---

## 5. Build Sequence

Six phases mapped to `docs/llmcompressorsharp/roadmap.md`. Each phase ends with a `git tag` and a published NuGet pre-release.

| Phase | Duration | Deliverable | Tag |
|---|---|---|---|
| 0 — Repo bootstrap | 1 week | Solution skeleton, CI, project shells, `PR_TO_TORCHSHARP.md` template | v0.0.1-alpha |
| 1 — Infrastructure | 4–6 weeks | `TorchExtensions`, modifier lifecycle, observers, recipe YAML | v0.1.0-alpha |
| 2 — Simple algorithms | 4–6 weeks | RTN, SmoothQuant, WANDA, Magnitude on toy fixture | v0.2.0-alpha |
| 3 — LLaMA + HF loader | 6–8 weeks | Load SmolLM2-135M, forward pass parity with reference | v0.3.0-alpha |
| 4 — GPTQ + SparseGPT | 6–8 weeks | W4A16 perplexity within +0.3 of baseline on TinyLlama-1.1B | v0.4.0-alpha |
| 5 — AWQ + output | 4–6 weeks | AWQ + compressed-tensors output, validated in Python vLLM | v0.5.0-alpha |
| 6 — CLI + polish | 2–3 weeks | `dnx llmc compress`, recipe presets, docs, samples | v0.6.0 |

**Total: 27–37 weeks (~6–9 months) for v0.6.0 covering LLaMA-family models.**

### Decision Gates

- **End of Phase 1:** confirm `TorchExtensions` strategy is holding. If blocked, escalate to fork-submodule path (Section 3).
- **End of Phase 3:** confirm forward-pass numerical parity with Python `transformers` for SmolLM2-135M before committing to Phase 4 (Hessian accumulation requires correct activations).
- **End of Phase 4:** confirm GPTQ accuracy meets the +0.3 perplexity target before continuing to Phase 5.

---

## 6. Risks (Summary)

Full risk register lives in `docs/llmcompressorsharp/pitfalls.md`. Key items:

| Risk | Mitigation |
|---|---|
| GPU memory leaks from undisposed tensors | `DisposeScope` audited in every PR review; calibration tests assert non-growing VRAM |
| No HuggingFace `transformers` equivalent | Scope to LLaMA family in v0.6.0; grow `Transformers` project per architecture |
| P/Invoke overhead on small tensor ops | In-place operations (`add_`, `mm_`) by default in hot loops |
| No FX symbolic tracing → manual sequential pipeline | Explicit per-architecture calibration loops; documented in `Architectures/Llama/SequentialCalibration.cs` |
| Cholesky failures on ill-conditioned Hessian | Dampening parameter; pseudoinverse fallback; FP32 accumulation even for FP16 weights |
| Output format compatibility with vLLM | Validate roundtrip in CI by loading output in Python with `compressed_tensors` library |

---

## 7. Out of Scope (deferred to post-1.0)

- Multi-architecture support (Qwen-MoE, DeepSeek, Granite, GLM, Gemma)
- Multi-GPU compression (DDP)
- AutoRound (no `torch.compile` equivalent)
- SpinQuant, QuIP, iMatrix (lower-priority algorithms)
- ONNX export from .NET
- Live in-browser compression visualisation

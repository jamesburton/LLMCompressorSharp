# LLMCompressorSharp

A .NET 10 / C# library and CLI for compressing LLaMA-family large language models.
Reimplements the core algorithms from the Python
[llm-compressor](https://github.com/vllm-project/llm-compressor) project on top of
[TorchSharp](https://github.com/dotnet/TorchSharp).

> **Status:** Phase 0 — repo bootstrap. APIs are not stable until v1.0.

## Goals

- Pure-.NET compression pipeline: RTN, SmoothQuant, AWQ, GPTQ, SparseGPT, WANDA, Magnitude pruning.
- HuggingFace-cache compatible: shares model weights with the Python ecosystem.
- Output formats: safetensors + `compressed-tensors` config (vLLM-compatible) and ONNX (pure .NET export).
- Single CLI: `dnx llmc compress ...`

## Quick start (after Phase 6)

```bash
dotnet tool install -g LLMCompressorSharp
dnx llmc compress \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --recipe gptq-w4a16 \
  --output ./compressed
```

## Project layout

```
src/
  LLMCompressorSharp.TorchExtensions/   TorchSharp gap-fillers
  LLMCompressorSharp.Core/              Modifiers, observers, recipes
  LLMCompressorSharp.Transformers/      LLaMA architecture, HF loader
  LLMCompressorSharp.Cli/               The `llmc` tool
tests/
  LLMCompressorSharp.Tests/
docs/                                   Research notes and design
```

## Build

```powershell
dotnet build LLMCompressorSharp.slnx
dotnet test  LLMCompressorSharp.slnx
```

Requires .NET 10 SDK. CPU-only build works on any platform; CUDA-accelerated
compression requires a Windows or Linux machine with an NVIDIA GPU and the
relevant `TorchSharp-cuda-*` package referenced at deploy time.

## Documentation

- [`docs/INDEX.md`](docs/INDEX.md) — research notes (llm-compressor, .NET 10, TorchSharp)
- [`docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md`](docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md) — design spec
- [`PR_TO_TORCHSHARP.md`](PR_TO_TORCHSHARP.md) — upstream contribution tracker

## License

Apache-2.0.

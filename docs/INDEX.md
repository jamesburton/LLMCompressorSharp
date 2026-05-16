# LLM Compressor & .NET Ecosystem — Knowledge Base

Research notes covering llm-compressor (Python/PyTorch), .NET 10 platform capabilities, TorchSharp, and a feasibility study for **LLMCompressorSharp** — a potential .NET reimplementation.

---

## Contents

### [llm-compressor](./llm-compressor/README.md)
Python library by the vLLM project for post-training compression of large language models.

| File | Description |
|---|---|
| [README.md](./llm-compressor/README.md) | Overview, architecture, module layout |
| [algorithms/](./llm-compressor/algorithms/) | One file per compression algorithm |
| [algorithms/gptq.md](./llm-compressor/algorithms/gptq.md) | GPTQ — Hessian-based weight quantization |
| [algorithms/awq.md](./llm-compressor/algorithms/awq.md) | AWQ — Activation-aware weight quantization |
| [algorithms/sparsegpt.md](./llm-compressor/algorithms/sparsegpt.md) | SparseGPT — Hessian-based pruning |
| [algorithms/smoothquant.md](./llm-compressor/algorithms/smoothquant.md) | SmoothQuant — activation smoothing |
| [algorithms/other.md](./llm-compressor/algorithms/other.md) | RTN, AutoRound, WANDA, SpinQuant, QuIP, Magnitude Pruning |
| [api-recipes.md](./llm-compressor/api-recipes.md) | Public Python API, recipe YAML format, pipelines |
| [deployment.md](./llm-compressor/deployment.md) | vLLM integration, supported models, hardware |

### [dotnet-10](./dotnet-10/README.md)
.NET 10 platform capabilities relevant to ML/AI workloads.

| File | Description |
|---|---|
| [README.md](./dotnet-10/README.md) | Overview and summary |
| [language.md](./dotnet-10/language.md) | C# 14 new features |
| [sdk-tooling.md](./dotnet-10/sdk-tooling.md) | File-based apps, dnx, CLI improvements, NuGet |
| [performance.md](./dotnet-10/performance.md) | JIT, stack allocation, Native AOT |
| [numerics.md](./dotnet-10/numerics.md) | System.Numerics.Tensors, TensorPrimitives (stable in .NET 10) |

### [torchsharp](./torchsharp/README.md)
.NET bindings for LibTorch — the C++ runtime underlying PyTorch.

| File | Description |
|---|---|
| [README.md](./torchsharp/README.md) | Overview, version compatibility, PyTorch relationship |
| [api.md](./torchsharp/api.md) | Tensor ops, nn modules, autograd, optimizers |
| [quantization.md](./torchsharp/quantization.md) | Quantization primitive support (and gaps) |
| [serialization.md](./torchsharp/serialization.md) | State dict, PyBridge, TorchScript, AOT export |
| [limitations.md](./torchsharp/limitations.md) | Known gaps vs PyTorch |
| [alternatives.md](./torchsharp/alternatives.md) | ONNX Runtime, ML.NET, LLamaSharp, others |

### [llmcompressorsharp](./llmcompressorsharp/README.md)
Feasibility study and implementation plan for a .NET reimplementation.

| File | Description |
|---|---|
| [README.md](./llmcompressorsharp/README.md) | Executive summary and recommendation |
| [algorithm-mapping.md](./llmcompressorsharp/algorithm-mapping.md) | Algorithm-by-algorithm feasibility |
| [pitfalls.md](./llmcompressorsharp/pitfalls.md) | Critical risks and blockers |
| [roadmap.md](./llmcompressorsharp/roadmap.md) | Phased implementation plan |
| [cache-conventions.md](./llmcompressorsharp/cache-conventions.md) | Shared HF/Ollama cache rules — every loader follows this |

---

## Quick Reference

### Key Findings

- **llm-compressor** applies post-training quantization/pruning to LLMs via a `oneshot()` API and a recipe system. Core algorithms: GPTQ, AWQ, SparseGPT, SmoothQuant, RTN, AutoRound, WANDA.
- **.NET 10** ships `System.Numerics.Tensors` as a stable API with SIMD-accelerated `TensorPrimitives`. C# 14 adds extension operators on `Tensor<T>`. File-based apps replace `.csx` scripting.
- **TorchSharp** 0.107.0 wraps LibTorch 2.10.0 (CUDA 12.8). Has all BLAS/LAPACK ops, full autograd, forward hooks, and quantize/dequantize primitives — but lacks a `torch.quantization` pipeline, DDP, and model graph tracing.
- **LLMCompressorSharp** is feasible for the core algorithms (GPTQ, SmoothQuant, AWQ, SparseGPT), but the biggest gap is the absence of a .NET HuggingFace Transformers equivalent for model loading and architecture definitions.

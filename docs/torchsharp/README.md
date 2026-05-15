# TorchSharp — Overview

**GitHub:** https://github.com/dotnet/TorchSharp  
**NuGet:** `TorchSharp` (+ `TorchSharp-cpu` or `TorchSharp-cuda-windows`)  
**License:** MIT  
**Maintained by:** Microsoft / .NET Foundation  
**Latest (May 2026):** v0.107.0 wrapping LibTorch 2.10.0 (CUDA 12.8)

---

## What It Is

TorchSharp is a .NET library (C# + F#) that wraps **LibTorch** — the same C++ runtime underlying PyTorch — via P/Invoke. It is **not** a reimplementation; it calls the exact same kernels. The NuGet packages bundle `libtorch` binaries so no separate PyTorch installation is required.

**Design philosophy:** The API intentionally mirrors PyTorch's Python API (snake_case naming) to allow near-direct translation of Python code to C#. Named parameters use C# colon syntax instead of `=`.

**Companion packages (same repo):**
- `TorchVision` — image transforms, datasets (MNIST, CIFAR10), model zoo (ResNet, VGG, MobileNet, GoogLeNet, InceptionV3)
- `TorchAudio` — audio transforms, Wav2Vec2, Tacotron2, HuBERT

---

## Version Compatibility

| TorchSharp | LibTorch / PyTorch | CUDA | .NET targets |
|---|---|---|---|
| 0.107.0 (latest) | 2.10.0 | 12.8 | net8.0, netstandard2.0 |
| 0.105.x | 2.7.1 | 12.8 | net8.0, netstandard2.0 |
| 0.103.0 | 2.4.0 | 12.1 | net8.0, netstandard2.0 |
| 0.102.x | 2.2.1 | 12.1 | net6.0, netstandard2.0 |

> `netstandard2.0` means it also runs on .NET 4.6.1+ and .NET Core 2.0+, including .NET 10.

**Platform support:**
- Windows x64, ARM64 (CPU; ARM64 CPU added 0.107.0)
- Linux x64, ARM64
- macOS Apple Silicon (M-series, from 0.103.0)
- Intel macOS dropped at 0.103.0

**NuGet packages:**
- `TorchSharp` — core library (no native binaries)
- `TorchSharp-cpu` — bundles CPU LibTorch
- `TorchSharp-cuda-windows` — bundles CUDA 12.8 LibTorch for Windows
- `TorchSharp-cuda-linux` — bundles CUDA 12.8 LibTorch for Linux
- `TorchSharp.PyBridge` (separate, v1.4.3) — pickle/safetensors/HuggingFace checkpoint loading

---

## Key Capability Summary

| Capability | Status |
|---|---|
| Tensor creation, indexing, slicing | ✅ Full |
| BLAS/LAPACK (`torch.linalg.*`) | ✅ Full (matmul, cholesky, lu, qr, svd, inv, solve, etc.) |
| Neural network modules (`torch.nn`) | ✅ Full (Linear, Conv, LSTM, Transformer, etc.) |
| Autograd + custom backward | ✅ Full |
| Optimizers (12) | ✅ Full |
| LR schedulers (13) | ✅ Full |
| Forward/backward hooks | ✅ Added 2024 |
| CUDA device management | ✅ Basic (no memory stats) |
| Quantize/dequantize primitives | ✅ Added 0.107.0 |
| `torch.quantization` pipeline | ❌ Not implemented |
| DataParallel / DDP | ❌ Not implemented |
| `torch.compile` | ❌ Not available |
| Sparse tensor ops (beyond COO) | ❌ Partial only |
| ONNX export | ❌ Not built-in |
| TorchScript creation from .NET | ❌ Load only |

---

## Contents

- [api.md](./api.md) — Tensor ops, nn modules, autograd, optimizers, CUDA
- [quantization.md](./quantization.md) — Quantization support and gaps
- [serialization.md](./serialization.md) — State dict, PyBridge, TorchScript, AOTInductor
- [limitations.md](./limitations.md) — Known gaps vs PyTorch
- [alternatives.md](./alternatives.md) — ONNX Runtime, ML.NET, LLamaSharp, others

# Alternative .NET ML Libraries

---

## ONNX Runtime — `Microsoft.ML.OnnxRuntime`

**Version:** v1.26.0  
**NuGet:** `Microsoft.ML.OnnxRuntime` (CPU), `Microsoft.ML.OnnxRuntime.Gpu` (CUDA)  
**Frameworks:** .NET 9.0, .NET Standard 2.0; compatible with .NET 5–10

### Use Case

High-performance **inference** on ONNX models. Not a training library. Best choice when you have a pre-trained model (in any framework) and need fast inference in .NET.

### Backends

| Backend | Platform |
|---|---|
| CPU | All platforms |
| CUDA | Windows/Linux x64 |
| TensorRT | Windows/Linux (Nvidia) |
| DirectML | Windows (any GPU via D3D12) |
| CoreML | macOS / iOS |
| XNNPACK | Android / iOS |
| QNN (Qualcomm) | Mobile |
| OpenVINO | Intel |

### Quantization

Run quantized ONNX models (INT8, INT4) that were quantized externally (via ONNX Runtime quantization tools, llm-compressor, or other tools). No training-time quantization within .NET.

### Example

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using var session = new InferenceSession("model.onnx",
    SessionOptions.MakeSessionOptionWithCudaProvider(0));

var inputTensor = new DenseTensor<float>(data, new[] { 1, 128 });
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
};

using var results = session.Run(inputs);
var output = results[0].AsTensor<float>();
```

### Relationship to TorchSharp

Complementary. Typical flow: train/compress with TorchSharp (or Python/PyTorch), export to ONNX, deploy inference with ORT. ORT is generally faster for inference, especially with TensorRT or DirectML backends.

---

## ML.NET — `Microsoft.ML`

**Version:** v0.23.0 (+ `Microsoft.ML.TorchSharp` v0.23.0)  
**Target:** .NET 6+, .NET Framework 4.6.1+

### Use Case

High-level ML pipeline API for classification, regression, forecasting, anomaly detection. Traditional ML algorithms (FastTree, LightGBM, etc.) plus TensorFlow and ONNX model scoring.

### TorchSharp Integration

`Microsoft.ML.TorchSharp` bridges ML.NET with TorchSharp for NLP tasks:
- Named Entity Recognition (NER)
- Sentence similarity
- Question answering
- Uses pretrained BERT-like models via `Microsoft.ML.Tokenizers`

**Not suitable for:** Custom deep learning research, custom model architectures, or model compression workflows.

---

## LLamaSharp — `LLamaSharp` (SciSharp)

**NuGet:** `LLamaSharp`  
**Backend:** llama.cpp (compiled native library, separate from LibTorch)

### Use Case

Local LLM inference using GGUF-format quantized models. Wraps llama.cpp via P/Invoke.

### Supported Models

LLaMA 1/2/3, Mixtral, Phi-3/4, Gemma 2, Qwen 2/3, LLaVA, DeepSeek-R1/V3, and most GGUF-format models from HuggingFace.

### Quantization

**Runs** quantized models natively (INT4, INT8, Q2_K, Q4_K_M, Q8_0 GGUF quantization). Quantization is performed externally with llama.cpp tools (`quantize` CLI). LLamaSharp is for inference only.

### Backends

CPU, CUDA 11/12, Vulkan, Metal (Apple Silicon)

### Example

```csharp
using LLama;
using LLama.Common;

var modelParams = new ModelParams("model-q4_k_m.gguf")
{
    ContextSize = 2048,
    GpuLayerCount = 32
};

using var model = LLamaWeights.LoadFromFile(modelParams);
using var context = model.CreateContext(new ModelParams { ... });

var executor = new InteractiveExecutor(context);
var result = await executor.InferAsync("Hello, how are you?");
```

### Relationship to TorchSharp

Completely separate stack (llama.cpp, not LibTorch). LLamaSharp cannot be used for training or compression research. It is the best option for **consuming** GGUF-quantized models from .NET.

**For LLMCompressorSharp:** LLamaSharp could be used to validate compressed models at inference time (convert from compressed-tensors format → GGUF via llama.cpp tools, then benchmark with LLamaSharp).

---

## Torch.NET (SciSharp) — **Avoid**

Requires Python 3.7 + PyTorch installed (no bundled binaries). 63 total commits. Effectively abandoned. Use TorchSharp instead.

---

## TensorFlow.NET (`SciSharp.TensorFlow.Redist`) — **Limited**

.NET bindings for TensorFlow. Used by ML.NET's TensorFlow integration. Less actively developed than TorchSharp for research use. No advantage over TorchSharp for compression work.

---

## Summary Table

| Library | Training | Inference | Quantization | GPU | Best For |
|---|---|---|---|---|---|
| TorchSharp | ✅ | ✅ | Primitives only | CUDA/MPS | Research, custom models |
| ONNX Runtime | ❌ | ✅ | Run only | CUDA/DML/etc | Production inference |
| ML.NET | Limited | ✅ | Run only | CPU/CUDA (via TorchSharp) | Pipelines, traditional ML |
| LLamaSharp | ❌ | ✅ GGUF | Consumes | CUDA/Vulkan/Metal | Local LLM inference |
| Torch.NET | ⚠️ abandoned | ⚠️ | ❌ | — | Don't use |

**For LLMCompressorSharp:** TorchSharp for compression logic + LLamaSharp or ONNX Runtime for validation/benchmarking.

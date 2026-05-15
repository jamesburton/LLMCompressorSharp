# TorchSharp — Model Serialization

---

## Native TorchSharp Format

```csharp
// Save
model.save("model.dat");

// Load (model architecture must already be constructed)
model.load("model.dat");

// State dict
OrderedDict<string, Tensor> sd = model.state_dict();
model.load_state_dict(sd, strict: true, skip: null);

// Non-strict load (ignore missing keys)
model.load_state_dict(sd, strict: false);

// Selective load (skip specific keys)
model.load_state_dict(sd, strict: true, skip: new[] { "lm_head.weight" });
```

**Format notes:**
- Binary format using TorchSharp's own serialization (not Python pickle)
- Not directly loadable by Python's `torch.load`
- Supports models and tensors >2GB (from v0.102.0)
- Supports `Stream` and `BinaryReader`/`BinaryWriter` in addition to file paths

---

## PyTorch / HuggingFace Formats (via `TorchSharp.PyBridge`)

**NuGet:** `TorchSharp.PyBridge` (v1.4.3, separate package)

### Pickle (`.pth`) Format

```csharp
using TorchSharp.PyBridge;

// Load Python pickle .pth file (saved with torch.save(state_dict, path))
model.load_py("model.pth");

// Save in format loadable by Python torch.load
model.save_py("model.pth");
```

### SafeTensors Format

```csharp
// Load safetensors file (HuggingFace format)
model.load_safetensors("model.safetensors");

// Save as safetensors
model.save_safetensors("model.safetensors");

// Load a raw dict of tensors (without a model)
var tensors = TorchSharp.PyBridge.PythonInterop.load_safetensors("file.safetensors");
```

### HuggingFace Checkpoint Loading

```csharp
// Load a sharded HuggingFace checkpoint (model.safetensors.index.json + shards)
model.load_checkpoint("path/to/checkpoint/directory/");
```

**Dependency:** `TorchSharp.PyBridge` uses the `pickle` library (Irmen de Jong's implementation). It does **not** require Python to be installed.

---

## TorchScript (`.pt`)

TorchScript files are compiled in Python using `torch.jit.trace()` or `torch.jit.script()` and can be loaded for inference in TorchSharp.

```csharp
// Load a TorchScript .pt file
var scriptModule = torch.jit.load("model.pt");
var scriptModule = torch.jit.load("model.pt", device: new torch.Device("cuda:0"));

// From byte array
byte[] bytes = File.ReadAllBytes("model.pt");
var scriptModule = torch.jit.load(bytes);

// Run inference
var output = scriptModule.forward(input);

// Named method invocation
var output = scriptModule.invoke("encode", input);

// Access parameters and buffers
var namedParams = scriptModule.named_parameters();
var namedBuffers = scriptModule.named_buffers();

// Training / eval mode
scriptModule.train();
scriptModule.eval();

// Hooks on TorchScript
scriptModule.register_forward_hook((mod, input, output) => { ... });
```

**Limitations:**
- Cannot **create** TorchScript from .NET code — load only
- Traced modules lock behaviour to training vs eval mode at trace time
- No gradient computation through TorchScript modules
- `CompilationUnit` exists: can compile TorchScript source strings, but only for inference

---

## torch.export / AOTInductor (`.pt2`)

For models compiled with `torch._inductor.aoti_compile_and_package()` in Python:

```csharp
// Load .pt2 file (AOTInductor compiled)
var exportModule = torch.export.load("model.pt2");
var output = exportModule.forward(input1, input2);
```

**Restrictions:**
- Inference only — no training, no parameter access, no device movement after load
- Must be compiled with `torch._inductor.aoti_compile_and_package()`, **not** `torch.export.save()` alone
- Claims 30–40% better latency vs TorchScript in many benchmarks

---

## Comparison Table

| Format | Write from .NET | Read in .NET | Write from Python | Read in Python |
|---|---|---|---|---|
| TorchSharp native `.dat` | ✅ | ✅ | ❌ | ❌ |
| Pickle `.pth` | ✅ (PyBridge) | ✅ (PyBridge) | ✅ | ✅ |
| SafeTensors | ✅ (PyBridge) | ✅ (PyBridge) | ✅ | ✅ |
| HuggingFace checkpoint | ❌ | ✅ (PyBridge) | ✅ | ✅ |
| TorchScript `.pt` | ❌ | ✅ | ✅ | ✅ |
| AOTInductor `.pt2` | ❌ | ✅ | ✅ (inductor) | ✅ |
| ONNX | ❌ | ❌ (need ORT) | ✅ | ✅ |

---

## ONNX

TorchSharp has **no built-in ONNX export or import**. To use ONNX in .NET:

1. Export the model to ONNX in Python using `torch.onnx.export()`
2. Load it via **ONNX Runtime** (`Microsoft.ML.OnnxRuntime`) in .NET for inference

```csharp
using Microsoft.ML.OnnxRuntime;

using var session = new InferenceSession("model.onnx");
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
};
using var results = session.Run(inputs);
```

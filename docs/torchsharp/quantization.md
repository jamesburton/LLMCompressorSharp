# TorchSharp — Quantization Support

---

## Current State (v0.107.0)

TorchSharp 0.107.0 added quantized scalar types and the low-level quantization primitives. There is **no high-level `torch.quantization` pipeline** (no `prepare()`, `convert()`, `FakeQuantize`, `Observer`, or quantized layer variants).

---

## What Exists

### Quantized Dtypes

```csharp
ScalarType.QInt8    // 8-bit signed integer, per-tensor or per-channel
ScalarType.QUInt8   // 8-bit unsigned integer
ScalarType.QInt32   // 32-bit integer (used as accumulator)
```

### Quantize / Dequantize Primitives

```csharp
using static TorchSharp.torch;

// Per-tensor quantization
var q = quantize_per_tensor(
    input: floatTensor,
    scale: 0.025f,
    zero_point: 0L,
    dtype: ScalarType.QInt8
);

// Per-channel quantization (one scale/zero_point per output channel)
var qc = quantize_per_channel(
    input: floatTensor,
    scales: scalesPerChannel,     // 1-D tensor of float
    zero_points: zpPerChannel,    // 1-D tensor of int
    axis: 0,                      // which dimension is the channel dim
    dtype: ScalarType.QInt8
);

// Dequantize back to float32
var deq = dequantize(q);
// or
var deq = q.dequantize();
```

### Quantization Parameter Access

```csharp
float scale     = q.q_scale();
long zeroPoint  = q.q_zero_point();
var rawInt      = q.int_repr();                  // tensor of raw integer values

// Per-channel:
var scales      = q.q_per_channel_scales();
var zeroPoints  = q.q_per_channel_zero_points();
int axis        = (int)q.q_per_channel_axis().item<long>();

bool isQuant    = q.is_quantized;
```

### Direct Memory Access (0.107.0+)

```csharp
// Access raw storage pointer — enables custom quantization kernels
var storagePtr = q.storage().data_ptr();
```

---

## What Does NOT Exist

| PyTorch Feature | TorchSharp Status | Workaround |
|---|---|---|
| `torch.quantization.prepare()` | ❌ Missing | Manually attach hooks |
| `torch.quantization.convert()` | ❌ Missing | Manually call `quantize_per_tensor/channel` |
| `FakeQuantize` module | ❌ Missing | Implement via `autograd.Function` (see below) |
| Observer modules (`MinMaxObserver`, `PerChannelMinMaxObserver`) | ❌ Missing | Implement manually |
| `QuantizedLinear`, `QuantizedConv2d` | ❌ Missing | Call quantize ops manually |
| `fuse_modules()` | ❌ Missing | Apply fusion manually |
| INT4 / sub-byte quantization | ❌ Missing | Must use 8-bit + bit-packing |
| QAT (`prepare_qat()`) | ❌ Missing | Custom training loop |
| Issue tracking this | #1298 (open Apr 2024) | — |

---

## Implementing FakeQuantize in TorchSharp

For a compression workflow, `FakeQuantize` simulates quantization during calibration while maintaining a differentiable graph. Implement via a custom autograd function:

```csharp
public class FakeQuantizeFunction : SingleTensorFunction<FakeQuantizeFunction>
{
    public static Tensor Forward(
        AutogradContext ctx,
        Tensor x,
        Tensor scale,
        Tensor zeroPoint,
        int numBits)
    {
        ctx.save_for_backward(x, scale);
        int qMin = -(1 << (numBits - 1));
        int qMax =  (1 << (numBits - 1)) - 1;

        // Quantize: round(x / scale + zero_point), clamp
        var xInt = (x / scale).round_() + zeroPoint;
        xInt.clamp_(qMin, qMax);

        // Dequantize
        var xFakeQ = (xInt - zeroPoint) * scale;
        return xFakeQ;
    }

    public static Tensor Backward(AutogradContext ctx, Tensor gradOutput)
    {
        // Straight-through estimator: pass gradient through unchanged
        return gradOutput;
    }
}
```

---

## Implementing a MinMax Observer

```csharp
public class MinMaxObserver
{
    private float _runningMin = float.MaxValue;
    private float _runningMax = float.MinValue;

    public void Update(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        float batchMin = x.min().item<float>();
        float batchMax = x.max().item<float>();
        _runningMin = Math.Min(_runningMin, batchMin);
        _runningMax = Math.Max(_runningMax, batchMax);
    }

    public (float scale, long zeroPoint) GetQuantParams(int numBits = 8, bool symmetric = true)
    {
        if (symmetric)
        {
            float absMax = Math.Max(Math.Abs(_runningMin), Math.Abs(_runningMax));
            float scale = absMax / ((1 << (numBits - 1)) - 1);
            return (scale, 0L);
        }
        else
        {
            float range = _runningMax - _runningMin;
            float scale = range / ((1 << numBits) - 1);
            long zeroPoint = (long)Math.Round(-_runningMin / scale);
            return (scale, zeroPoint);
        }
    }
}
```

---

## Practical Quantization Workflow (Manual)

```csharp
// 1. Collect calibration statistics
var observer = new MinMaxObserver();
module.register_forward_hook((mod, input, output) =>
{
    observer.Update(output);
    return null;
});

// Run calibration
foreach (var batch in calibrationDataset)
{
    using var scope = torch.NewDisposeScope();
    model.forward(batch);
}

// 2. Compute quantization parameters
var (scale, zeroPoint) = observer.GetQuantParams(numBits: 8, symmetric: true);

// 3. Quantize weights
var wFloat = linear.weight;
var wQuant = torch.quantize_per_tensor(wFloat, scale, zeroPoint, ScalarType.QInt8);

// 4. Dequantize for inference (or use int8 kernel if available)
var wDequant = wQuant.dequantize();
```

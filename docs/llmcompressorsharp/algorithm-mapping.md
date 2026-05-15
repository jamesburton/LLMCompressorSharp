# LLMCompressorSharp — Algorithm-by-Algorithm C# Mapping

For each algorithm: the Python implementation approach, required TorchSharp APIs, and a C# sketch.

---

## RTN — Round-to-Nearest

**Python:** `QuantizationModifier` — attaches observer hooks, collects min/max stats, computes scale/zero_point, calls `quantize_per_tensor`.

**TorchSharp APIs needed:**
- `quantize_per_tensor()` ✅
- `quantize_per_channel()` ✅
- `register_forward_hook()` ✅
- `tensor.max()`, `tensor.min()`, `tensor.amax()` ✅

**C# sketch:**

```csharp
public class RtnQuantizer
{
    public void Quantize(Linear layer, int numBits = 8, bool symmetric = true)
    {
        using var w = layer.weight.detach();
        float absMax = w.abs().max().item<float>();
        float scale = absMax / ((1 << (numBits - 1)) - 1);

        var wQuant = torch.quantize_per_tensor(w, scale, 0L, ScalarType.QInt8);
        var wDequant = wQuant.dequantize();

        // Replace weight with dequantized version (simulate quantized inference)
        using (torch.no_grad())
        {
            layer.weight.copy_(wDequant);
        }
    }
}
```

**Difficulty:** Low. Two function calls.

---

## SmoothQuant

**Python:** Attaches forward hooks to collect per-channel activation max, computes per-channel scale `s = max(|X|)^α / max(|W|)^(1-α)`, applies `W *= s`, `W_prev /= s`.

**TorchSharp APIs needed:**
- `register_forward_hook()` ✅
- `tensor.abs().amax(dims)` ✅
- `tensor.pow(alpha)` ✅
- `tensor.mul_()` / `tensor.div_()` ✅

**C# sketch:**

```csharp
public class SmoothQuantTransform
{
    private readonly Dictionary<string, Tensor> _activationMax = new();

    public IDisposable AttachHooks(Module model, IEnumerable<string> targetLayers)
    {
        var handles = new List<IDisposable>();
        foreach (var (name, mod) in model.named_modules())
        {
            if (targetLayers.Contains(name))
            {
                var n = name;
                handles.Add(mod.register_forward_hook((_, _, output) =>
                {
                    using var scope = torch.NewDisposeScope();
                    var chMax = output.abs().amax(new long[] { 0, 1 });  // [channels]
                    if (_activationMax.TryGetValue(n, out var existing))
                        torch.max(existing, chMax, out var combined); // element-wise max
                    else
                        _activationMax[n] = chMax.MoveToOuterDisposeScope();
                    return null;
                }));
            }
        }
        return new CompositeDisposable(handles);
    }

    public void ApplySmoothing(
        Linear smoothLayer,
        Linear balanceLayer,
        Tensor activationMax,
        float alpha = 0.5f)
    {
        using var wBalance = balanceLayer.weight.detach();
        var wMax = wBalance.abs().amax(new long[] { 1 });  // per output channel

        var s = activationMax.pow(alpha) / wMax.pow(1 - alpha);

        using (torch.no_grad())
        {
            // Divide smooth layer's weight by s (makes activations harder, weights easier)
            smoothLayer.weight.div_(s.unsqueeze(0));
            // Multiply balance layer's weight by s
            balanceLayer.weight.mul_(s.unsqueeze(1));
        }
    }
}
```

**Difficulty:** Moderate. Hook management + per-channel tensor math.

---

## GPTQ

**Python:** Hessian `H = X^T X` accumulated from calibration. Cholesky inverse. Block-wise column quantization with error propagation.

**TorchSharp APIs needed:**
- `register_forward_hook()` ✅
- `torch.mm()`, `torch.matmul()` ✅
- `torch.linalg.cholesky()` ✅
- `torch.linalg.solve_triangular()` ✅
- `torch.linalg.inv()` ✅
- `quantize_per_tensor()` / custom fake-quant ✅
- `tensor.diagonal()`, `tensor.fill_diagonal_()` ✅
- `tensor.clone()`, `tensor[range, range]` ✅

**C# sketch (core loop):**

```csharp
public static void GptqQuantize(
    Tensor W,             // [outFeatures, inFeatures]
    Tensor H,             // [inFeatures, inFeatures] — accumulated Hessian
    float scale,
    long zeroPoint,
    int blockSize = 128,
    float dampeningFrac = 0.01f)
{
    int d = (int)H.shape[0];

    // Dampening
    var dampening = H.diagonal().mean().item<float>() * dampeningFrac;
    H.fill_diagonal_(H.diagonal() + dampening);

    // Cholesky inverse: H_inv = (L^-T)(L^-1) where L = chol(H)
    var L = torch.linalg.cholesky(H);
    var Linv = torch.linalg.solve_triangular(L, torch.eye(d), upper: false);
    var Hinv = torch.mm(Linv.t(), Linv);  // upper triangular inverse

    var Wq = W.clone();

    for (int i = 0; i < d; i += blockSize)
    {
        int end = Math.Min(i + blockSize, d);
        var wBlock   = W[TensorIndex.Colon, i..end];        // [out, blockSize]
        var wqBlock  = Wq[TensorIndex.Colon, i..end];
        var HinvBlock= Hinv[i..end, i..end];
        var HinvTail = Hinv[i..end, end..];

        // Quantize block
        for (int j = i; j < end; j++)
        {
            var col = W[TensorIndex.Colon, j];              // [out]
            var colQ = FakeQuantize(col, scale, zeroPoint); // round to nearest
            Wq[TensorIndex.Colon, j] = colQ;

            // Propagate error to remaining columns in this block
            if (j + 1 < end)
            {
                var err = (col - colQ) / HinvBlock[j - i, j - i];  // scalar per output
                W[TensorIndex.Colon, (j+1)..end] -= err.unsqueeze(1) * HinvBlock[j-i, (j-i+1)..];
            }
        }

        // Propagate error to remaining blocks
        if (end < d)
        {
            var blockErr = W[TensorIndex.Colon, i..end] - Wq[TensorIndex.Colon, i..end];
            W[TensorIndex.Colon, end..] -= torch.mm(blockErr, HinvTail);
        }

        W[TensorIndex.Colon, i..end] = Wq[TensorIndex.Colon, i..end];
    }
}
```

**Difficulty:** Moderate-Hard. Correct indexing and error propagation require careful implementation. Numerically sensitive; dampening value matters.

---

## SparseGPT

Same as GPTQ infrastructure, replacing fake-quantize with a **mask selection** step:

```csharp
// Instead of quantizing, prune lowest-saliency weights in block
// Saliency score per weight = w_ij^2 / H_inv[j,j]^2
var saliency = wBlock.pow(2) / HinvBlock.diagonal().pow(2).unsqueeze(0);
var threshold = saliency.topk((int)(blockSize * sparsity), largest: false).values.max();
var mask = saliency > threshold;
Wq[TensorIndex.Colon, i..end] = wBlock * mask;  // zero out pruned weights
```

For **2:4 structured sparsity**, constrain the mask to exactly 2 zeros per 4 consecutive weights:

```csharp
// Group columns into groups of 4, keep top-2 per group
for (int g = 0; g < blockSize; g += 4)
{
    var group = saliency[TensorIndex.Colon, g..(g+4)];
    var top2 = group.topk(2, dim: 1).indices;
    // Zero everything except top-2
}
```

---

## AWQ

```csharp
// Grid search over n_grid scale candidates
float bestMse = float.MaxValue;
Tensor bestScale = null;

for (int k = 1; k <= nGrid; k++)
{
    float alpha = k / (float)nGrid;
    var s = activationMax.pow(alpha) / wMax.pow(1 - alpha);
    s = s.clamp(min: 1e-4f);  // avoid division by zero

    // Apply scale and simulate quantization
    using var scope = torch.NewDisposeScope();
    var wScaled = weight * s.unsqueeze(0);
    var wQuant  = FakeQuantize(wScaled, ...);
    var wBack   = wQuant / s.unsqueeze(0);

    float mse = (wBack - weight).pow(2).mean().item<float>();
    if (mse < bestMse)
    {
        bestMse = mse;
        bestScale = s.MoveToOuterDisposeScope();
    }
}

// Apply best scale permanently
using (torch.no_grad())
{
    balanceLayer.weight.mul_(bestScale.unsqueeze(1));
    smoothLayer.weight.div_(bestScale.unsqueeze(0));
}
```

---

## WANDA

```csharp
// Calibration: collect per-column activation L2 norms via forward hooks
// activationNorm[j] = √(Σ_samples ||x_j||₂²)

// Pruning: saliency = |w_ij| * activationNorm[j]
var saliency = weight.abs() * activationNorm.unsqueeze(0);  // [out, in]
var flatSaliency = saliency.reshape(-1);
var threshold = flatSaliency.topk((int)(flatSaliency.shape[0] * sparsity),
                                   largest: false).values.max();
var mask = saliency > threshold;
using (torch.no_grad())
{
    weight.mul_(mask.to(weight.dtype));
}
```

**Difficulty:** Moderate. Simpler than GPTQ (no Cholesky), but correct L2 norm accumulation across batches requires care.

---

## Modifier Lifecycle (C# Design)

Maps cleanly to an interface + abstract base class:

```csharp
public interface IModifier
{
    void Initialize(CompressionState state);
    void OnStart(CompressionState state);
    void OnBatch(CompressionState state, int batchIndex);
    void OnEnd(CompressionState state);
    void Finalize(CompressionState state);
}

public abstract class ModifierBase : IModifier
{
    public string[]? Targets { get; init; }
    public string[]? Ignore  { get; init; }
    // ... default implementations

    protected IEnumerable<(string name, Linear layer)> GetTargetLayers(Module model)
        => model.named_modules()
                .Where(m => Targets?.Contains(m.name) != false
                         && Ignore?.Contains(m.name) != true
                         && m.module is Linear)
                .Select(m => (m.name, (Linear)m.module));
}
```

---

## Recipe YAML (C# with YamlDotNet)

```yaml
stages:
  - name: smooth_quant
    modifiers:
      - type: SmoothQuant
        smoothing_strength: 0.5
  - name: gptq
    modifiers:
      - type: GPTQ
        scheme: W4A16
        block_size: 128
        ignore: ["lm_head"]
```

```csharp
[YamlRoot("stages")]
public class Recipe
{
    public List<Stage> Stages { get; set; } = new();
}

public class Stage
{
    public string Name { get; set; } = "";
    public List<ModifierConfig> Modifiers { get; set; } = new();
}

// Polymorphic deserialization using a custom YamlDotNet type discriminator
```

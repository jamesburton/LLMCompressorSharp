# TorchSharp — API Reference

---

## Tensor Operations

### Creation

```csharp
using static TorchSharp.torch;

// Basics
var t = zeros(3, 4);          // 3×4 float32 zeros
var o = ones(3, 4, dtype: ScalarType.Float16);
var r = rand(3, 4);            // uniform [0,1)
var n = randn(3, 4);           // standard normal
var i = arange(0, 10, step: 2);
var l = linspace(0.0, 1.0, 100);
var e = eye(4);

// From data
var t2 = tensor(new float[] { 1, 2, 3, 4 }).reshape(2, 2);
var t3 = frombuffer<float>(buffer);

// Type factory
var half = zeros(3, 4, dtype: ScalarType.Float16);
var bf16 = zeros(3, 4, dtype: ScalarType.BFloat16);
var qint8 = zeros(3, dtype: ScalarType.QInt8);   // added 0.107.0
```

### Dtypes Available

`Bool`, `Byte` (uint8), `Int8`, `Int16`, `Int32`, `Int64`, `Float16` (Half), `BFloat16`, `Float32`, `Float64`, `ComplexFloat32`, `ComplexFloat64`, `QInt8`, `QUInt8`, `QInt32`

### Indexing and Slicing

```csharp
var t = randn(4, 4);
var row = t[0];                   // first row (TensorIndex.Single(0))
var col = t[TensorIndex.Colon, 1];  // all rows, column 1
var sub = t[1..3, 0..2];          // range slice
var mask = t[t > 0];              // boolean masking
var sel = t.index_select(0, tensor(new long[] { 0, 2 }));
```

### BLAS / LAPACK (`torch.linalg`)

```csharp
// Matrix multiply
var C = mm(A, B);               // 2D only
var C = matmul(A, B);           // broadcasting ND
var C = bmm(A, B);              // batched 2D

// Decompositions
var (L, info) = linalg.cholesky(A);
var (Q, R)    = linalg.qr(A);
var (U, S, Vh)= linalg.svd(A);
var (vals, vecs) = linalg.eigh(A);  // symmetric eigen
var lu        = linalg.lu(A);

// Solve
var x = linalg.solve(A, b);
var x = linalg.solve_triangular(L, b, upper: false);

// Inversion / pseudoinverse
var Ainv = linalg.inv(A);
var Apinv = linalg.pinv(A);

// Norms / determinant
float norm  = linalg.norm(A).item<float>();
float det   = linalg.det(A).item<float>();
int rank    = linalg.matrix_rank(A).item<long>(); // returns tensor
```

### Reduction Operations

```csharp
var s = t.sum();
var m = t.mean();
var mx = t.max();               // returns (values, indices)
var mn = t.min();
var am = t.amax(dim: 0);        // per-row max
var ai = t.argmax(dim: 1);      // per-row argmax
var nm = t.norm(p: 2.0);
var all_true = t.all();
var any_true = t.any();
```

### Scatter / Gather

```csharp
var gathered = t.gather(1, index);           // gather along dim 1
t.scatter_(1, index, src);                   // in-place scatter
var out_t = t.scatter_add(0, index, src);    // scatter + reduce add
```

### In-Place Operations (Preferred for Performance)

All operations have `_` variants that modify the tensor in-place and return `this`:

```csharp
t.add_(other);        // t += other (no new Tensor object)
t.mul_(2.0f);
t.relu_();
t.fill_(0.0f);
t.clamp_(min: -1.0f, max: 1.0f);
t.div_(scale);
```

> **Performance note:** In-place ops are significantly faster than out-of-place because they avoid allocating a new C# `Tensor` wrapper object on each call.

### Data Access

```csharp
float scalar = t.item<float>();          // single-element tensor
var arr = t.data<float>().ToArray();     // copy to managed array
ReadOnlySpan<float> span = t.data<float>();  // span over native memory
unsafe {
    float* ptr = (float*)t.data_ptr();   // raw pointer (0.107.0+)
}
```

---

## Neural Network Modules (`torch.nn`)

### Defining a Module

```csharp
using TorchSharp;
using static TorchSharp.torch.nn;

public class QuantLinear : Module<Tensor, Tensor>
{
    private readonly Linear _linear;
    private Tensor _scale;
    private Tensor _zero_point;

    public QuantLinear(long inFeatures, long outFeatures)
        : base("QuantLinear")
    {
        _linear = Linear(inFeatures, outFeatures);
        _scale = zeros(outFeatures);
        _zero_point = zeros(outFeatures, dtype: ScalarType.Int32);
        RegisterComponents();  // REQUIRED — auto-registers all private fields
    }

    public override Tensor forward(Tensor input) => _linear.forward(input);
}
```

> `RegisterComponents()` must be called at the end of every constructor to auto-register submodules and parameters.

### Available Layers

**Linear:** `Linear`, `Bilinear`  
**Conv:** `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`  
**Recurrent:** `RNN`, `LSTM`, `GRU` (+ cell variants)  
**Transformer:** `Transformer`, `TransformerEncoderLayer`, `TransformerDecoderLayer`, `MultiheadAttention`  
**Norm:** `LayerNorm`, `BatchNorm1d/2d/3d`, `GroupNorm`, `InstanceNorm1d/2d/3d`  
**Pool:** `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`  
**Dropout:** `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`  
**Embedding:** `Embedding`, `EmbeddingBag`  
**Container:** `Sequential`, `ModuleList`, `ModuleDict`  
**Activations (28):** `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `CELU`, `ELU`, `SELU`, `Mish`, `Hardswish`, `Hardsigmoid`, `LeakyReLU`, `PReLU`, ...  
**Loss (21):** `CrossEntropyLoss`, `MSELoss`, `BCELoss`, `NLLLoss`, `KLDivLoss`, `HuberLoss`, ...

### Module API

```csharp
module.parameters()          // IEnumerable<Parameter>
module.named_parameters()    // IEnumerable<(string, Parameter)>
module.state_dict()          // OrderedDict<string, Tensor>
module.load_state_dict(dict, strict: true, skip: null)
module.save(path)
module.load(path)
module.train()               // set training mode
module.eval()                // set eval mode
module.zero_grad()
module.apply(fn)             // depth-first visitor
module.to(device)
module.to(dtype: ScalarType.Float16)
```

### Forward Hooks

```csharp
// Register a hook that fires after each forward pass
var handle = module.register_forward_hook((mod, input, output) =>
{
    // Capture activations for calibration
    activationStats.Update(output);
    return null;  // return null to leave output unchanged
});

// Pre-hook (fires before forward)
var preHandle = module.register_forward_pre_hook((mod, input) =>
{
    // Inspect/modify inputs
    return null;
});

handle.Dispose();  // remove hook
```

---

## Autograd

```csharp
// Enable gradient tracking
var w = randn(4, 4, requires_grad: true);
var x = randn(4, 1);

var y = mm(w, x).sum();
y.backward();

Tensor grad = w.grad;

// Gradient context managers
using (var noGrad = no_grad())
{
    // operations here don't build a graph
}

using (var inferMode = inference_mode())
{
    // inference-mode: like no_grad but stronger
}

// Compute explicit gradients
var grads = autograd.grad(
    outputs: new[] { loss },
    inputs: new[] { w },
    retain_graph: false
);

// Custom backward function
public class MyFunction : SingleTensorFunction<MyFunction>
{
    public static Tensor Forward(AutogradContext ctx, Tensor input, Tensor weight)
    {
        ctx.save_for_backward(input, weight);
        return mm(input, weight);
    }

    public static Tensor Backward(AutogradContext ctx, Tensor gradOutput)
    {
        var (input, weight) = ctx.get_saved_tensors();
        return mm(gradOutput, weight.t());
    }
}

var result = MyFunction.apply(input, weight);
```

---

## Optimizers

```csharp
var model = new MyModel();
var optimizer = optim.AdamW(model.parameters(), lr: 1e-4, weight_decay: 0.01);
var scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 100);

for (int step = 0; step < maxSteps; step++)
{
    optimizer.zero_grad();
    var loss = model.forward(x);
    loss.backward();
    optimizer.step();
    scheduler.step();
}
```

**Available optimizers:** SGD, Adam, AdamW, Adamax, NAdam, RAdam, Adagrad, Adadelta, ASGD, RMSprop, Rprop, LBFGS  
**LR schedulers:** LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, PolynomialLR, LinearLR, ConstantLR, SequentialLR, MultiplicativeLR

---

## CUDA / Device Management

```csharp
bool hasCuda = cuda.is_available();
int gpuCount = cuda.device_count();

// Move tensor to GPU
var t_gpu = t.cuda();
var t_gpu = t.to(new device("cuda:0"));

// Synchronize
cuda.synchronize();

// Apple Silicon
bool hasMPS = mps_is_available();
var t_mps = t.mps();

// Thread control
set_num_threads(8);
set_num_interop_threads(4);

// Flash attention
backends.cuda.enable_flash_sdp(true);
```

---

## Memory Management

TorchSharp tensors hold native GPU/CPU memory. The .NET GC does not track VRAM pressure — **explicit disposal is required** to avoid OOM.

```csharp
// Option 1: using statement
using var t = randn(1000, 1000).cuda();

// Option 2: DisposeScope (preferred for loops)
using (var scope = NewDisposeScope())
{
    for (int i = 0; i < batches; i++)
    {
        var batch = GetBatch(i);
        var output = model.forward(batch);
        // All tensors created in this scope are disposed when scope exits
    }
}

// Option 3: Manual disposal + GC hint
t.Dispose();
GC.Collect();
GC.WaitForPendingFinalizers();
```

> **Rule:** Any tensor that exits a training loop iteration as a temporary should be disposed. Use `DisposeScope` for calibration loops.

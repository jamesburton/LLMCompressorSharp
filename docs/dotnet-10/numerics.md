# .NET 10 — System.Numerics.Tensors & TensorPrimitives

The `System.Numerics.Tensors` NuGet package is **stable in .NET 10** (no longer `[Experimental]`). It provides N-dimensional typed tensors and a comprehensive library of SIMD-accelerated span-based operations.

**NuGet:** `System.Numerics.Tensors`

---

## Types

### `Tensor<T>`

N-dimensional tensor backed by managed memory. Indexed by `ReadOnlySpan<nint>` or a `TensorIndex[]`.

```csharp
// Create
var t = Tensor.Create<float>(new nint[] { 3, 4 });  // 3x4 float tensor
var t2 = Tensor.Create<float>([1.0f, 2.0f, 3.0f, 4.0f], [2, 2]);

// Access
float val = t[1, 2];
t[0, 3] = 5.0f;

// Slice (zero-copy in .NET 10)
ReadOnlyTensorSpan<float> row = t[0, ..];
```

### `TensorSpan<T>` / `ReadOnlyTensorSpan<T>`

Like `Span<T>` but N-dimensional. Can wrap managed, native, or stack memory. Zero-copy slicing (new in .NET 10).

```csharp
// Stack allocation of small tensor
Span<float> buffer = stackalloc float[16];
TensorSpan<float> span = new TensorSpan<float>(buffer, [4, 4], []);

// Wrap unmanaged pointer
unsafe {
    TensorSpan<float> native = new TensorSpan<float>(ptr, [rows, cols], []);
}
```

### Interfaces (New in .NET 10)

| Interface | Description |
|---|---|
| `IReadOnlyTensor` | Non-generic: exposes `Lengths`, `Strides` without knowing element type |
| `IReadOnlyTensor<TSelf, T>` | Generic read-only tensor contract |
| `ITensor<TSelf, T>` | Read-write tensor contract |

---

## `TensorPrimitives` — Operation Catalogue

Every operation is hardware-accelerated via SIMD (AVX2, AVX-512, NEON) where available. All operations have both a typed `float` overload and a generic `<T>` overload constrained to the appropriate `INumber<T>` or `IBinaryInteger<T>` interface.

### Arithmetic

```csharp
TensorPrimitives.Add(x, y, destination);        // x + y
TensorPrimitives.Subtract(x, y, destination);
TensorPrimitives.Multiply(x, y, destination);
TensorPrimitives.Divide(x, y, destination);
TensorPrimitives.Negate(x, destination);
TensorPrimitives.Abs(x, destination);
TensorPrimitives.AddMultiply(x, y, z, dest);    // (x + y) * z
TensorPrimitives.MultiplyAdd(x, y, z, dest);    // x * y + z (FMA)
TensorPrimitives.FusedMultiplyAdd(x, y, z, dest);
```

### Reduction (returns a scalar)

```csharp
float sum  = TensorPrimitives.Sum(x);
float prod = TensorPrimitives.Product(x);
float max  = TensorPrimitives.Max(x);
float min  = TensorPrimitives.Min(x);
float dot  = TensorPrimitives.Dot(x, y);            // dot product
float norm = TensorPrimitives.Norm(x);              // L2 norm (‖x‖₂)
float sumSq = TensorPrimitives.SumOfSquares(x);
float avg  = TensorPrimitives.Average(x);
int idxMax = TensorPrimitives.IndexOfMax(x);
int idxMin = TensorPrimitives.IndexOfMin(x);
float cos  = TensorPrimitives.CosineSimilarity(x, y);
float dist = TensorPrimitives.Distance(x, y);       // L2 distance
```

### Trigonometric & Exponential

```csharp
TensorPrimitives.Sin(x, dest);    TensorPrimitives.Cos(x, dest);
TensorPrimitives.Exp(x, dest);    TensorPrimitives.Log(x, dest);
TensorPrimitives.Log2(x, dest);   TensorPrimitives.Sqrt(x, dest);
TensorPrimitives.Pow(x, y, dest); // element-wise x^y
TensorPrimitives.Tanh(x, dest);   TensorPrimitives.Sigmoid(x, dest);
TensorPrimitives.SinCos(x, sin, cos);  // compute both simultaneously
// ... full trig suite: Asin, Acos, Atan, Atan2, Sinh, Cosh, etc.
```

### ML-Specific

```csharp
TensorPrimitives.SoftMax(x, dest);       // softmax over span
TensorPrimitives.Sigmoid(x, dest);
TensorPrimitives.SigmoidDerivative(x, dest);
TensorPrimitives.Lerp(x, y, t, dest);   // linear interpolation
```

### Rounding / Clamping

```csharp
TensorPrimitives.Clamp(x, min, max, dest);
TensorPrimitives.Round(x, dest);
TensorPrimitives.Ceiling(x, dest);
TensorPrimitives.Floor(x, dest);
```

### Comparison (element-wise)

```csharp
TensorPrimitives.GreaterThan(x, y, dest);      // float comparison → bool span
TensorPrimitives.LessThan(x, y, dest);
TensorPrimitives.Equals(x, y, dest);
```

### Bitwise (integer types)

```csharp
TensorPrimitives.BitwiseAnd(x, y, dest);
TensorPrimitives.ShiftLeft(x, count, dest);
TensorPrimitives.PopCount(x, dest);
```

### Conversion

```csharp
TensorPrimitives.ConvertToSingle(source, dest);   // int/double/Half → float
TensorPrimitives.ConvertToHalf(source, dest);     // float → Half (float16)
TensorPrimitives.ConvertToDouble(source, dest);
TensorPrimitives.ConvertToInt32(source, dest);
```

---

## C# 14 Extension Operators on `Tensor<T>`

With C# 14, `Tensor<T>` supports operator overloads when `T` implements the relevant `INumber<T>` interfaces:

```csharp
Tensor<float> a = Tensor.Create<float>([1, 2, 3], [3]);
Tensor<float> b = Tensor.Create<float>([4, 5, 6], [3]);

var c = a + b;    // element-wise add
var d = a * 2.0f; // scalar multiply
var e = a - b;
```

`Tensor<bool>` does **not** support arithmetic operators.

---

## `Half` (float16) Support

`System.Half` works with all `TensorPrimitives` generic overloads. Critical for quantization workflows (FP16 weight storage with FP32 accumulation):

```csharp
Half[] weights16 = ...; // FP16 weights
float[] weights32 = new float[weights16.Length];
TensorPrimitives.ConvertToSingle(weights16, weights32);
```

---

## Limitations vs TorchSharp for Compression

| Need | System.Numerics.Tensors | TorchSharp |
|---|---|---|
| Element-wise ops (add, mul, exp...) | ✅ SIMD-accelerated | ✅ GPU-accelerated |
| Large matrix multiply (GEMM) | ❌ No in-box BLAS | ✅ cuBLAS via libtorch |
| Cholesky decomposition | ❌ Not included | ✅ `torch.linalg.cholesky` |
| GPU/CUDA | ❌ CPU only | ✅ CUDA kernels |
| Autograd | ❌ Not included | ✅ Full autograd |
| bfloat16 | ❌ Not in BCL | ✅ `ScalarType.BFloat16` |
| N-dimensional named axes | Partial (via `TensorSpan`) | ✅ Full named-dim support |

**Recommendation:** Use `TensorPrimitives` for SIMD-accelerated CPU preprocessing (observer stats, scale computation, small reductions) and TorchSharp for the heavy lifting (Hessian accumulation, Cholesky, GPU tensor ops).

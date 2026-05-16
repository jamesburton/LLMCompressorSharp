# Phase 3a: LLaMA Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the LLaMA decoder-only transformer architecture as a tree of TorchSharp `Module`s — `LlamaForCausalLM` composed of `LlamaDecoderLayer`s (Attention + MLP + residuals) with `LlamaRMSNorm`, `LlamaRotaryEmbedding`, and grouped-query attention. Tests use synthetic configs (small hidden size, few layers/heads) on CPU. **Shape and sanity correctness only — numerical parity against the Python reference comes in Phase 3c.**

**Architecture:**

```
LlamaForCausalLM (Module<Tensor, Tensor>)
   ├── embed_tokens : nn.Embedding [vocab_size, hidden_size]
   ├── layers : ModuleList of LlamaDecoderLayer × num_hidden_layers
   │     ├── input_layernorm : LlamaRMSNorm
   │     ├── self_attn : LlamaAttention
   │     │     ├── q_proj, k_proj, v_proj, o_proj : nn.Linear
   │     │     └── rotary_emb : LlamaRotaryEmbedding (shared by Q and K)
   │     ├── post_attention_layernorm : LlamaRMSNorm
   │     └── mlp : LlamaMLP
   │           ├── gate_proj, up_proj, down_proj : nn.Linear
   │           └── SiLU activation
   ├── norm : LlamaRMSNorm (final)
   └── lm_head : nn.Linear (or tied to embed_tokens.weight)
```

**No KV cache in Phase 3a** — compression-time forward passes are full-sequence (no autoregressive generation), so we skip cache management. Phase 3c may add it for tokenizer-driven generation tests.

**Tech Stack:** TorchSharp 0.107.0 (Module, nn.Linear, nn.Embedding, nn.ModuleList, RegisterComponents). C# 14 / .NET 10.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (Transformers/Architectures/Llama)

---

## File Structure

```
src/LLMCompressorSharp.Transformers/
├── LLMCompressorSharp.Transformers.csproj         ← already exists
├── (delete) PlaceholderMarker.cs                  ← removed in Task 10
├── Architectures/
│   └── Llama/                                     ← NEW namespace
│       ├── LlamaConfig.cs                         ← POCO hyperparameters
│       ├── LlamaRMSNorm.cs                        ← Module<Tensor, Tensor>
│       ├── LlamaRotaryEmbedding.cs                ← Module producing (cos, sin) caches
│       ├── LlamaMLP.cs                            ← Module<Tensor, Tensor>
│       ├── LlamaAttention.cs                      ← Module with multi-arg forward
│       ├── LlamaDecoderLayer.cs                   ← Module with multi-arg forward
│       └── LlamaForCausalLM.cs                    ← Module<Tensor, Tensor>

tests/LLMCompressorSharp.Tests/
└── Transformers/
    └── Architectures/
        └── Llama/
            ├── LlamaRMSNormTests.cs
            ├── LlamaRotaryEmbeddingTests.cs
            ├── LlamaMLPTests.cs
            ├── LlamaAttentionTests.cs
            ├── LlamaDecoderLayerTests.cs
            └── LlamaForCausalLMTests.cs
```

**Responsibility per file:**

- `LlamaConfig` — POCO matching HuggingFace `config.json` field names (snake_case → PascalCase): `HiddenSize`, `IntermediateSize`, `NumHiddenLayers`, `NumAttentionHeads`, `NumKeyValueHeads` (for GQA), `VocabSize`, `MaxPositionEmbeddings`, `RopeTheta`, `RmsNormEps`, `HiddenAct` (only "silu" supported), `TieWordEmbeddings`. Computed: `HeadDim = HiddenSize / NumAttentionHeads`.

- `LlamaRMSNorm` — `output = (x * rsqrt(mean(x²) + eps)) * weight`. Weight shape `[hidden]`. Forward returns same shape as input.

- `LlamaRotaryEmbedding` — Pre-computes `inv_freq[i] = 1 / theta^(2i / head_dim)` at construction. Forward takes `[batch, seq, num_heads, head_dim]` query/key tensor + `position_ids [batch, seq]` and returns rotated tensor. Exposes `Apply(q, k, position_ids) -> (q_rot, k_rot)` as the canonical interface.

- `LlamaMLP` — `output = down_proj(silu(gate_proj(x)) * up_proj(x))`. All projections are bias-less `nn.Linear`.

- `LlamaAttention` — Grouped-query attention. `q_proj` produces `num_heads * head_dim`, `k_proj` and `v_proj` produce `num_kv_heads * head_dim`. K and V are repeated `num_heads / num_kv_heads` times before the attention dot product (`repeat_kv` helper). Causal mask applied. No KV cache.

- `LlamaDecoderLayer` — pre-norm pattern: `x = x + attn(norm1(x), mask, pos_ids)`; `x = x + mlp(norm2(x))`.

- `LlamaForCausalLM` — input tokens `[batch, seq]` → logits `[batch, seq, vocab]`. Builds causal mask + position_ids internally. Tied embeddings supported via `TieWordEmbeddings` config flag.

**Out of scope:**
- KV cache (Phase 3c if needed for generation)
- Weight loading from HuggingFace (Phase 3b)
- Tokenizer (Phase 3c)
- Numerical parity with Python `transformers` (Phase 3c)
- Quantized inference (the architecture is FP32; modifiers compress it post-hoc)

---

## Prerequisites & Conventions

- Phase 2b is merged. Tag `v0.2.1-alpha`. 119 tests passing on `main`.
- `.editorconfig` already exempts `src/LLMCompressorSharp.Transformers/Architectures/**.cs` from SA1300/SA1303/SA1311 — snake_case TorchSharp calls land naturally in this namespace.
- Branch off `main` as `feature/3a-llama-architecture`.
- TorchSharp imports: `using TorchSharp; using static TorchSharp.torch; using static TorchSharp.torch.nn; using TorchSharp.Modules;`.
- StyleCop patterns from prior phases apply.
- Tests use small synthetic configs (e.g., `HiddenSize=8, NumHiddenLayers=2, NumAttentionHeads=2, NumKeyValueHeads=1`) to keep CPU runs fast.

---

### Task 1: Branch + baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
Expected: clean tree, HEAD `3c8bed3`, tags through `v0.2.1-alpha`.

- [ ] **Step 2:** `git checkout -b feature/3a-llama-architecture`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"` — expect 119 passing.

No commit.

---

### Task 2: Add `LlamaConfig`

**Files:**
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaConfig.cs`

- [ ] **Step 1: Write the config**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Hyperparameters for a LLaMA-family decoder-only transformer.
/// </summary>
/// <remarks>
/// Field names match the HuggingFace <c>config.json</c> keys with PascalCase substitution.
/// Phase 3b adds a JSON parser; Phase 3a tests construct instances directly.
/// </remarks>
public sealed class LlamaConfig
{
    /// <summary>Gets or sets the residual stream dimension. Required.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Gets or sets the MLP hidden (inner) dimension. Required.</summary>
    public int IntermediateSize { get; set; }

    /// <summary>Gets or sets the number of decoder layers. Required.</summary>
    public int NumHiddenLayers { get; set; }

    /// <summary>Gets or sets the number of attention query heads. Required.</summary>
    public int NumAttentionHeads { get; set; }

    /// <summary>Gets or sets the number of KV heads for grouped-query attention. Equals <see cref="NumAttentionHeads"/> for MHA.</summary>
    public int NumKeyValueHeads { get; set; }

    /// <summary>Gets or sets the vocabulary size. Required.</summary>
    public int VocabSize { get; set; }

    /// <summary>Gets or sets the maximum position embedding length. Default: 2048.</summary>
    public int MaxPositionEmbeddings { get; set; } = 2048;

    /// <summary>Gets or sets the RoPE theta base. Default: 10000.</summary>
    public float RopeTheta { get; set; } = 10000f;

    /// <summary>Gets or sets the RMSNorm epsilon. Default: 1e-5.</summary>
    public float RmsNormEps { get; set; } = 1e-5f;

    /// <summary>Gets or sets the hidden activation. Phase 3a only supports "silu".</summary>
    public string HiddenAct { get; set; } = "silu";

    /// <summary>Gets or sets a value indicating whether <c>lm_head.weight</c> shares storage with <c>embed_tokens.weight</c>.</summary>
    public bool TieWordEmbeddings { get; set; } = true;

    /// <summary>Gets the per-head dimension <c>HiddenSize / NumAttentionHeads</c>.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Validates that required fields are present and consistent.</summary>
    /// <exception cref="ArgumentException">If any required field is missing or inconsistent.</exception>
    public void Validate()
    {
        if (HiddenSize <= 0)
        {
            throw new ArgumentException($"HiddenSize must be positive (got {HiddenSize}).");
        }

        if (NumAttentionHeads <= 0 || HiddenSize % NumAttentionHeads != 0)
        {
            throw new ArgumentException(
                $"NumAttentionHeads={NumAttentionHeads} must divide HiddenSize={HiddenSize} evenly.");
        }

        if (NumKeyValueHeads <= 0 || NumAttentionHeads % NumKeyValueHeads != 0)
        {
            throw new ArgumentException(
                $"NumKeyValueHeads={NumKeyValueHeads} must divide NumAttentionHeads={NumAttentionHeads} evenly.");
        }

        if (NumHiddenLayers <= 0 || IntermediateSize <= 0 || VocabSize <= 0)
        {
            throw new ArgumentException("NumHiddenLayers, IntermediateSize, and VocabSize must be positive.");
        }

        if (!string.Equals(HiddenAct, "silu", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Phase 3a only supports HiddenAct='silu' (got '{HiddenAct}').");
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaConfig.cs
git commit -m "feat(transformers): add LlamaConfig hyperparameters"
```

---

### Task 3: TDD `LlamaRMSNorm`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRMSNormTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRMSNorm.cs`

- [ ] **Step 1: Write the failing tests**

Create `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRMSNormTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Shape and basic-math tests for <see cref="LlamaRMSNorm"/>.
/// </summary>
public class LlamaRMSNormTests
{
    [Fact]
    public void Forward_PreservesInputShape()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-5f);
        using var input = randn(2, 4, 8);
        using var output = norm.forward(input);

        output.shape.Should().Equal(new long[] { 2, 4, 8 });
    }

    [Fact]
    public void Forward_WithUnitWeight_ProducesUnitRmsRow()
    {
        // RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight
        // With weight = 1 and eps tiny, mean(y²) per row ≈ 1.
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-8f);
        using var input = randn(1, 1, 8);
        using var output = norm.forward(input);

        // mean(y²) along the last dim should be ≈ 1.0
        using var sq = output.pow(2);
        using var meanSq = sq.mean(new long[] { -1L }, keepdim: false);
        meanSq.cpu().item<float>().Should().BeApproximately(1f, 1e-3f);
    }

    [Fact]
    public void Forward_WithZeroInput_ProducesZeroOutput()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 4, eps: 1e-5f);
        using var input = zeros(1, 1, 4);
        using var output = norm.forward(input);

        var arr = output.cpu().data<float>().ToArray();
        arr.Should().AllSatisfy(v => v.Should().BeApproximately(0f, 1e-5f));
    }

    [Fact]
    public void Weight_IsLearnableAndExposed()
    {
        using var norm = new LlamaRMSNorm(hiddenSize: 8, eps: 1e-5f);
        var parameters = norm.parameters().ToList();
        parameters.Should().HaveCount(1);
        parameters[0].shape.Should().Equal(new long[] { 8 });
    }

    [Fact]
    public void Constructor_NonPositiveHiddenSize_Throws()
    {
        var act = () => new LlamaRMSNorm(hiddenSize: 0, eps: 1e-5f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
```

- [ ] **Step 2: Verify CS0246 build failure**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaRMSNormTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRMSNormTests.cs
git commit -m "test(transformers): add failing LlamaRMSNorm tests"
```

- [ ] **Step 4: Implement `LlamaRMSNorm`**

Create `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRMSNorm.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Root Mean Square LayerNorm — the normalization used in LLaMA-family decoders.
/// </summary>
/// <remarks>
/// <c>y = x · rsqrt(mean(x², dim=-1) + eps) · weight</c>. No bias term.
/// </remarks>
public sealed class LlamaRMSNorm : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly float eps;

    /// <summary>Initializes a new instance of the <see cref="LlamaRMSNorm"/> class.</summary>
    /// <param name="hiddenSize">The last-dim size of the input tensor.</param>
    /// <param name="eps">Epsilon added to the mean-square term for numerical stability.</param>
    public LlamaRMSNorm(int hiddenSize, float eps)
        : base(nameof(LlamaRMSNorm))
    {
        if (hiddenSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        }

        this.eps = eps;
        this.weight = Parameter(ones(hiddenSize));
        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        // Cast to FP32 for numerical stability; cast back at the end if needed.
        var inputDtype = x.dtype;
        using var x32 = x.to(ScalarType.Float32);
        using var variance = x32.pow(2).mean(new long[] { -1L }, keepdim: true);
        using var stabilized = variance.add(eps);
        using var rstd = stabilized.rsqrt();
        using var normalized = x32.mul(rstd);
        using var result = normalized.mul(this.weight);
        return result.to(inputDtype);
    }
}
```

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaRMSNormTests"`
Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRMSNorm.cs
git commit -m "feat(transformers): implement LlamaRMSNorm"
```

---

### Task 4: TDD `LlamaRotaryEmbedding`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRotaryEmbeddingTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRotaryEmbedding.cs`

The RoPE math: given `head_dim` (must be even), compute `inv_freq[i] = 1 / theta^(2i / head_dim)` for `i in [0, head_dim/2)`. For each position `m`, compute `freq[m, i] = m * inv_freq[i]`. Concatenate `freq` with itself to form a `[..., head_dim]` rotation, then `cos = cos(freq)`, `sin = sin(freq)`. Rotation: `rotate_half(x) = cat([-x[..., half:], x[..., :half]])`; `x_rot = x * cos + rotate_half(x) * sin`.

- [ ] **Step 1: Write the failing tests**

Create `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRotaryEmbeddingTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaRotaryEmbedding"/> — RoPE rotation.
/// </summary>
public class LlamaRotaryEmbeddingTests
{
    [Fact]
    public void Apply_PreservesShapes()
    {
        // [batch=1, num_heads=2, seq=4, head_dim=8]
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 2, 4, 8);
        using var k = randn(1, 2, 4, 8);
        using var positions = arange(4).unsqueeze(0); // [1, 4]

        var (qRot, kRot) = rope.Apply(q, k, positions);
        using (qRot)
        using (kRot)
        {
            qRot.shape.Should().Equal(new long[] { 1, 2, 4, 8 });
            kRot.shape.Should().Equal(new long[] { 1, 2, 4, 8 });
        }
    }

    [Fact]
    public void Apply_PreservesVectorMagnitude()
    {
        // Rotation preserves L2 norm per vector.
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 1, 4, 8);
        using var k = q.clone();
        using var positions = arange(4).unsqueeze(0);

        var (qRot, _) = rope.Apply(q, k, positions);
        using (qRot)
        {
            using var origNorm = q.pow(2).sum(new long[] { -1L });
            using var rotNorm = qRot.pow(2).sum(new long[] { -1L });
            var origArr = origNorm.cpu().data<float>().ToArray();
            var rotArr = rotNorm.cpu().data<float>().ToArray();
            for (var i = 0; i < origArr.Length; i++)
            {
                rotArr[i].Should().BeApproximately(origArr[i], 1e-3f);
            }
        }
    }

    [Fact]
    public void Apply_PositionZero_IsIdentity()
    {
        // At position 0, cos = 1 and sin = 0, so rotation is identity.
        using var rope = new LlamaRotaryEmbedding(headDim: 8, maxPositionEmbeddings: 32, theta: 10000f);
        using var q = randn(1, 1, 1, 8);
        using var k = q.clone();
        using var positions = zeros(1, 1, dtype: ScalarType.Int64);

        var (qRot, _) = rope.Apply(q, k, positions);
        using (qRot)
        {
            using var diff = (qRot - q).abs().max();
            diff.cpu().item<float>().Should().BeLessThan(1e-5f);
        }
    }

    [Fact]
    public void Constructor_OddHeadDim_Throws()
    {
        var act = () => new LlamaRotaryEmbedding(headDim: 7, maxPositionEmbeddings: 32, theta: 10000f);
        act.Should().Throw<ArgumentException>().WithMessage("*even*");
    }
}
```

- [ ] **Step 2: Verify failure, commit**

Run targeted test. Expected: build fails (CS0246).

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaRotaryEmbeddingTests.cs
git commit -m "test(transformers): add failing LlamaRotaryEmbedding tests"
```

- [ ] **Step 3: Implement `LlamaRotaryEmbedding`**

Create `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRotaryEmbedding.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Rotary Position Embedding (RoPE). Pre-computes the inverse-frequency table once at
/// construction; <see cref="Apply"/> rotates query and key tensors by their token positions.
/// </summary>
public sealed class LlamaRotaryEmbedding : Module
{
    private readonly Tensor invFreq;

    /// <summary>Initializes a new instance of the <see cref="LlamaRotaryEmbedding"/> class.</summary>
    /// <param name="headDim">Per-head dimension; must be even.</param>
    /// <param name="maxPositionEmbeddings">Maximum sequence length the table supports.</param>
    /// <param name="theta">RoPE base frequency (LLaMA default 10000; SmolLM2 100000).</param>
    public LlamaRotaryEmbedding(int headDim, int maxPositionEmbeddings, float theta)
        : base(nameof(LlamaRotaryEmbedding))
    {
        if (headDim <= 0 || (headDim % 2) != 0)
        {
            throw new ArgumentException($"headDim must be a positive even integer (got {headDim}).", nameof(headDim));
        }

        // inv_freq[i] = 1 / theta^(2i / head_dim) for i in [0, head_dim/2)
        using var indices = arange(0, headDim, 2, dtype: ScalarType.Float32);
        using var exponents = indices / headDim;
        using var powers = exponents.mul(MathF.Log(theta)).exp();
        this.invFreq = powers.reciprocal();
        this.register_buffer("inv_freq", this.invFreq);
    }

    /// <summary>
    /// Applies RoPE rotation to query and key tensors.
    /// </summary>
    /// <param name="q">Query tensor of shape <c>[batch, num_heads, seq, head_dim]</c>.</param>
    /// <param name="k">Key tensor of shape <c>[batch, num_kv_heads, seq, head_dim]</c>.</param>
    /// <param name="positionIds">Position indices, shape <c>[batch, seq]</c>.</param>
    /// <returns>Rotated <c>(q, k)</c>.</returns>
    public (Tensor Q, Tensor K) Apply(Tensor q, Tensor k, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(positionIds);

        // Compute cos/sin for the supplied positions.
        // positionIds shape: [batch, seq]; produce [batch, seq, head_dim].
        using var posFloat = positionIds.to(ScalarType.Float32);
        using var freqs = posFloat.unsqueeze(-1).matmul(this.invFreq.unsqueeze(0)); // [batch, seq, half]
        using var emb = cat(new[] { freqs, freqs }, dim: -1); // [batch, seq, head_dim]
        using var cos = emb.cos().unsqueeze(1).to(q.dtype); // broadcast across heads
        using var sin = emb.sin().unsqueeze(1).to(q.dtype);

        var qRot = ApplyRotation(q, cos, sin);
        var kRot = ApplyRotation(k, cos, sin);
        return (qRot, kRot);
    }

    private static Tensor ApplyRotation(Tensor x, Tensor cos, Tensor sin)
    {
        using var rotated = RotateHalf(x);
        using var cosPart = x.mul(cos);
        using var sinPart = rotated.mul(sin);
        return cosPart.add(sinPart);
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var headDim = (int)x.shape[^1];
        var half = headDim / 2;
        using var firstHalf = x.narrow(-1, 0, half);
        using var secondHalf = x.narrow(-1, half, half);
        using var negSecond = secondHalf.neg();
        return cat(new[] { negSecond, firstHalf }, dim: -1);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaRotaryEmbeddingTests"`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaRotaryEmbedding.cs
git commit -m "feat(transformers): implement LlamaRotaryEmbedding (RoPE)"
```

---

### Task 5: TDD `LlamaMLP`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaMLPTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaMLP.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaMLP"/>.
/// </summary>
public class LlamaMLPTests
{
    [Fact]
    public void Forward_PreservesInputShape()
    {
        using var mlp = new LlamaMLP(hiddenSize: 8, intermediateSize: 32);
        using var x = randn(2, 4, 8);
        using var y = mlp.forward(x);
        y.shape.Should().Equal(new long[] { 2, 4, 8 });
    }

    [Fact]
    public void Parameters_ThreeLinearLayersNoBias()
    {
        using var mlp = new LlamaMLP(hiddenSize: 8, intermediateSize: 32);
        var named = mlp.named_parameters().ToList();
        // gate_proj.weight, up_proj.weight, down_proj.weight — 3 weights, no biases
        named.Should().HaveCount(3);
        var names = named.Select(p => p.name).ToList();
        names.Should().Contain(n => n.Contains("gate_proj"));
        names.Should().Contain(n => n.Contains("up_proj"));
        names.Should().Contain(n => n.Contains("down_proj"));
        names.Should().NotContain(n => n.Contains("bias"));
    }

    [Fact]
    public void Constructor_NonPositiveSize_Throws()
    {
        var act = () => new LlamaMLP(hiddenSize: 0, intermediateSize: 32);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaMLPTests.cs
git commit -m "test(transformers): add failing LlamaMLP tests"
```

- [ ] **Step 3: Implement `LlamaMLP`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// LLaMA gated MLP: <c>down_proj(silu(gate_proj(x)) * up_proj(x))</c>.
/// </summary>
public sealed class LlamaMLP : Module<Tensor, Tensor>
{
    private readonly Linear gate_proj;
    private readonly Linear up_proj;
    private readonly Linear down_proj;

    /// <summary>Initializes a new instance of the <see cref="LlamaMLP"/> class.</summary>
    /// <param name="hiddenSize">Outer (residual stream) dimension.</param>
    /// <param name="intermediateSize">Inner MLP dimension.</param>
    public LlamaMLP(int hiddenSize, int intermediateSize)
        : base(nameof(LlamaMLP))
    {
        if (hiddenSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        }

        if (intermediateSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(intermediateSize), intermediateSize, "intermediateSize must be positive.");
        }

        this.gate_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
        this.up_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
        this.down_proj = Linear(intermediateSize, hiddenSize, hasBias: false);
        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        using var gate = this.gate_proj.forward(x);
        using var up = this.up_proj.forward(x);
        using var activated = silu(gate);
        using var gated = activated.mul(up);
        return this.down_proj.forward(gated);
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaMLPTests"
```
Expected: 3 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaMLP.cs
git commit -m "feat(transformers): implement LlamaMLP"
```

---

### Task 6: TDD `LlamaAttention` with grouped-query attention

This is the most complex module. Reference behaviour:
1. Project `x` to q (`[batch, seq, num_heads * head_dim]`), k and v (`[batch, seq, num_kv_heads * head_dim]`).
2. Reshape to `[batch, num_heads, seq, head_dim]` and `[batch, num_kv_heads, seq, head_dim]`.
3. Apply RoPE to q and k.
4. Repeat k and v to match num_heads via `repeat_interleave` (the GQA step).
5. Compute scaled dot-product attention: `attn = softmax(q · k^T / sqrt(head_dim) + mask)`.
6. `out = attn · v` reshape to `[batch, seq, hidden]`, project via o_proj.

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaAttentionTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaAttention.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaAttention"/> with grouped-query attention.
/// </summary>
public class LlamaAttentionTests
{
    private static LlamaConfig MakeConfig()
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 1,
            NumAttentionHeads = 4,
            NumKeyValueHeads = 2,  // GQA: 2 KV heads grouped into 4 Q heads
            VocabSize = 100,
            MaxPositionEmbeddings = 16,
            RopeTheta = 10000f,
            RmsNormEps = 1e-5f,
            HiddenAct = "silu",
            TieWordEmbeddings = false,
        };
    }

    [Fact]
    public void Forward_PreservesShape()
    {
        var config = MakeConfig();
        using var attn = new LlamaAttention(config);
        using var x = randn(2, 4, config.HiddenSize);
        using var positions = arange(4).unsqueeze(0).expand(2, 4);

        using var y = attn.forward(x, attentionMask: null, positionIds: positions);
        y.shape.Should().Equal(new long[] { 2, 4, config.HiddenSize });
    }

    [Fact]
    public void Parameters_FourLinearProjectionsNoBias()
    {
        using var attn = new LlamaAttention(MakeConfig());
        var named = attn.named_parameters().ToList();
        named.Select(p => p.name).Should().Contain(n => n.Contains("q_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("k_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("v_proj"));
        named.Select(p => p.name).Should().Contain(n => n.Contains("o_proj"));
        named.Should().HaveCount(4); // 4 weights, no biases
    }

    [Fact]
    public void Forward_CausalMask_PreventsAttendingToFutureTokens()
    {
        // With identical q tokens but seq positions 0, 1, 2, the token at position 0 must produce
        // the same output regardless of what comes after it (because the causal mask zeros out
        // future positions).
        var config = MakeConfig();
        using var attn = new LlamaAttention(config);
        using var input = randn(1, 3, config.HiddenSize);
        using var positions = arange(3).unsqueeze(0);

        // Run with full sequence
        using var fullOut = attn.forward(input, attentionMask: null, positionIds: positions);

        // Run with just the first token
        using var firstOnly = input.narrow(1, 0, 1);
        using var firstPositions = positions.narrow(1, 0, 1);
        using var prefixOut = attn.forward(firstOnly, attentionMask: null, positionIds: firstPositions);

        // The first-token output should be approximately equal in both runs (causal mask isolates it).
        using var fullFirstToken = fullOut.narrow(1, 0, 1);
        using var diff = (fullFirstToken - prefixOut).abs().max();
        diff.cpu().item<float>().Should().BeLessThan(1e-4f);
    }

    [Fact]
    public void Constructor_HeadDimDoesNotDivideHidden_Throws()
    {
        var config = MakeConfig();
        config.NumAttentionHeads = 3; // 8 / 3 doesn't divide evenly
        var act = () => new LlamaAttention(config);
        act.Should().Throw<ArgumentException>();
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaAttentionTests.cs
git commit -m "test(transformers): add failing LlamaAttention tests"
```

- [ ] **Step 3: Implement `LlamaAttention`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// Grouped-query multi-head self-attention with RoPE and a causal mask.
/// </summary>
public sealed class LlamaAttention : Module
{
    private readonly LlamaConfig config;
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly LlamaRotaryEmbedding rotary_emb;
    private readonly int kvGroupSize;

    /// <summary>Initializes a new instance of the <see cref="LlamaAttention"/> class.</summary>
    /// <param name="config">Architecture config (validated).</param>
    public LlamaAttention(LlamaConfig config)
        : base(nameof(LlamaAttention))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        this.config = config;

        this.kvGroupSize = config.NumAttentionHeads / config.NumKeyValueHeads;

        this.q_proj = Linear(config.HiddenSize, config.NumAttentionHeads * config.HeadDim, hasBias: false);
        this.k_proj = Linear(config.HiddenSize, config.NumKeyValueHeads * config.HeadDim, hasBias: false);
        this.v_proj = Linear(config.HiddenSize, config.NumKeyValueHeads * config.HeadDim, hasBias: false);
        this.o_proj = Linear(config.NumAttentionHeads * config.HeadDim, config.HiddenSize, hasBias: false);

        this.rotary_emb = new LlamaRotaryEmbedding(
            headDim: config.HeadDim,
            maxPositionEmbeddings: config.MaxPositionEmbeddings,
            theta: config.RopeTheta);

        RegisterComponents();
    }

    /// <summary>Self-attention forward pass.</summary>
    /// <param name="x">Input hidden states <c>[batch, seq, hidden]</c>.</param>
    /// <param name="attentionMask">Additive mask <c>[batch, 1, seq, seq]</c> or null (causal mask is built internally).</param>
    /// <param name="positionIds">Position indices <c>[batch, seq]</c>.</param>
    /// <returns>Output hidden states <c>[batch, seq, hidden]</c>.</returns>
    public Tensor forward(Tensor x, Tensor? attentionMask, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(positionIds);

        var batch = x.shape[0];
        var seq = x.shape[1];
        var numHeads = (long)this.config.NumAttentionHeads;
        var numKvHeads = (long)this.config.NumKeyValueHeads;
        var headDim = (long)this.config.HeadDim;

        // Project to Q/K/V
        using var qFlat = this.q_proj.forward(x);
        using var kFlat = this.k_proj.forward(x);
        using var vFlat = this.v_proj.forward(x);

        // Reshape to [batch, num_heads, seq, head_dim]
        using var qHeads = qFlat.view(batch, seq, numHeads, headDim).transpose(1, 2);
        using var kHeads = kFlat.view(batch, seq, numKvHeads, headDim).transpose(1, 2);
        using var vHeads = vFlat.view(batch, seq, numKvHeads, headDim).transpose(1, 2);

        // RoPE
        var (qRot, kRot) = this.rotary_emb.Apply(qHeads, kHeads, positionIds);

        // GQA: repeat K/V to match num_heads
        using (qRot)
        using (kRot)
        {
            using var kRepeated = RepeatKv(kRot, this.kvGroupSize);
            using var vRepeated = RepeatKv(vHeads, this.kvGroupSize);

            // Scaled dot-product attention: attn = softmax(q @ k^T / sqrt(head_dim) + mask) @ v
            using var attnScores = qRot.matmul(kRepeated.transpose(-2, -1));
            using var scale = tensor(1f / MathF.Sqrt(headDim)).to(attnScores.dtype);
            using var scaled = attnScores.mul(scale);

            // Build causal mask: lower-triangular ones, inverted; or apply additive mask if provided.
            using var causalMask = BuildCausalMask(seq, x.device, x.dtype);
            using var masked = attentionMask is null
                ? scaled.add(causalMask)
                : scaled.add(causalMask).add(attentionMask);

            using var probs = softmax(masked, dim: -1);
            using var attnOut = probs.matmul(vRepeated); // [batch, num_heads, seq, head_dim]

            // Reshape back to [batch, seq, hidden]
            using var transposed = attnOut.transpose(1, 2).contiguous();
            using var combined = transposed.view(batch, seq, numHeads * headDim);
            return this.o_proj.forward(combined);
        }
    }

    private static Tensor RepeatKv(Tensor kv, int repeats)
    {
        if (repeats == 1)
        {
            return kv.alias();
        }

        // kv shape: [batch, num_kv_heads, seq, head_dim] → [batch, num_kv_heads * repeats, seq, head_dim]
        var batch = kv.shape[0];
        var numKv = kv.shape[1];
        var seq = kv.shape[2];
        var hd = kv.shape[3];
        using var expanded = kv.unsqueeze(2).expand(batch, numKv, repeats, seq, hd);
        return expanded.reshape(batch, numKv * repeats, seq, hd);
    }

    private static Tensor BuildCausalMask(long seq, torch.Device device, ScalarType dtype)
    {
        // Lower-triangular mask: 0 for allowed positions, -inf for disallowed.
        using var ones = torch.ones(seq, seq, device: device);
        using var triu = ones.triu(diagonal: 1); // 1s above the diagonal
        using var minusInf = triu.mul(float.NegativeInfinity);
        return minusInf.to(dtype);
    }
}
```

> **Implementation note:** the use of `.alias()` for the no-repeat path is to return a tensor that can be `using`-disposed without invalidating the original. If TorchSharp lacks `.alias()`, use `.clone()` (slower) or restructure with a manual wrapper.

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaAttentionTests"
```
Expected: 4 tests pass.

If `Forward_CausalMask_PreventsAttendingToFutureTokens` fails: print the actual difference. The most common bug is using `seq_kv` instead of `seq_q` somewhere in the mask construction.

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaAttention.cs
git commit -m "feat(transformers): implement LlamaAttention with GQA and RoPE"
```

---

### Task 7: TDD `LlamaDecoderLayer`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaDecoderLayerTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaDecoderLayer.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaDecoderLayer"/>.
/// </summary>
public class LlamaDecoderLayerTests
{
    private static LlamaConfig MakeConfig()
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 1,
            NumAttentionHeads = 4,
            NumKeyValueHeads = 2,
            VocabSize = 100,
            MaxPositionEmbeddings = 16,
            RopeTheta = 10000f,
            RmsNormEps = 1e-5f,
            HiddenAct = "silu",
            TieWordEmbeddings = false,
        };
    }

    [Fact]
    public void Forward_PreservesShape()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        using var x = randn(2, 4, config.HiddenSize);
        using var positions = arange(4).unsqueeze(0).expand(2, 4);

        using var y = layer.forward(x, attentionMask: null, positionIds: positions);
        y.shape.Should().Equal(new long[] { 2, 4, config.HiddenSize });
    }

    [Fact]
    public void Forward_ResidualConnection_OutputIsNotIdenticalToInput()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        using var x = randn(1, 2, config.HiddenSize);
        using var positions = arange(2).unsqueeze(0);

        using var y = layer.forward(x, attentionMask: null, positionIds: positions);
        using var diff = (y - x).abs().max();
        diff.cpu().item<float>().Should().BeGreaterThan(1e-5f);
    }

    [Fact]
    public void NamedSubmodules_ContainExpectedComponents()
    {
        var config = MakeConfig();
        using var layer = new LlamaDecoderLayer(config);
        var names = layer.named_modules().Select(m => m.name).ToList();
        names.Should().Contain(n => n.Contains("input_layernorm"));
        names.Should().Contain(n => n.Contains("self_attn"));
        names.Should().Contain(n => n.Contains("post_attention_layernorm"));
        names.Should().Contain(n => n.Contains("mlp"));
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaDecoderLayerTests.cs
git commit -m "test(transformers): add failing LlamaDecoderLayer tests"
```

- [ ] **Step 3: Implement `LlamaDecoderLayer`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// One LLaMA decoder block: pre-norm + self-attention + residual; pre-norm + MLP + residual.
/// </summary>
public sealed class LlamaDecoderLayer : Module
{
    private readonly LlamaRMSNorm input_layernorm;
    private readonly LlamaAttention self_attn;
    private readonly LlamaRMSNorm post_attention_layernorm;
    private readonly LlamaMLP mlp;

    /// <summary>Initializes a new instance of the <see cref="LlamaDecoderLayer"/> class.</summary>
    /// <param name="config">Architecture configuration.</param>
    public LlamaDecoderLayer(LlamaConfig config)
        : base(nameof(LlamaDecoderLayer))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();

        this.input_layernorm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.self_attn = new LlamaAttention(config);
        this.post_attention_layernorm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.mlp = new LlamaMLP(config.HiddenSize, config.IntermediateSize);
        RegisterComponents();
    }

    /// <summary>Forward pass.</summary>
    /// <param name="x">Input hidden states <c>[batch, seq, hidden]</c>.</param>
    /// <param name="attentionMask">Optional additive attention mask.</param>
    /// <param name="positionIds">Position indices <c>[batch, seq]</c>.</param>
    /// <returns>Output hidden states <c>[batch, seq, hidden]</c>.</returns>
    public Tensor forward(Tensor x, Tensor? attentionMask, Tensor positionIds)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(positionIds);

        // Pre-norm self-attention with residual
        using var attnInput = this.input_layernorm.forward(x);
        using var attnOut = this.self_attn.forward(attnInput, attentionMask, positionIds);
        using var afterAttn = x.add(attnOut);

        // Pre-norm MLP with residual
        using var mlpInput = this.post_attention_layernorm.forward(afterAttn);
        using var mlpOut = this.mlp.forward(mlpInput);
        return afterAttn.add(mlpOut);
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaDecoderLayerTests"
```
Expected: 3 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaDecoderLayer.cs
git commit -m "feat(transformers): implement LlamaDecoderLayer"
```

---

### Task 8: TDD `LlamaForCausalLM`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaForCausalLMTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaForCausalLM.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Architectures.Llama;

/// <summary>
/// Tests for <see cref="LlamaForCausalLM"/>.
/// </summary>
public class LlamaForCausalLMTests
{
    private static LlamaConfig MakeConfig(bool tied = true)
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 2,
            NumAttentionHeads = 4,
            NumKeyValueHeads = 2,
            VocabSize = 100,
            MaxPositionEmbeddings = 16,
            RopeTheta = 10000f,
            RmsNormEps = 1e-5f,
            HiddenAct = "silu",
            TieWordEmbeddings = tied,
        };
    }

    [Fact]
    public void Forward_ProducesLogitsOfShape_BatchSeqVocab()
    {
        var config = MakeConfig();
        using var model = new LlamaForCausalLM(config);
        using var inputIds = randint(low: 0, high: config.VocabSize, new long[] { 2, 5 }, dtype: ScalarType.Int64);

        using var logits = model.forward(inputIds);
        logits.shape.Should().Equal(new long[] { 2, 5, config.VocabSize });
    }

    [Fact]
    public void NamedSubmodules_ContainExpectedTopLevelComponents()
    {
        var config = MakeConfig();
        using var model = new LlamaForCausalLM(config);
        var names = model.named_modules().Select(m => m.name).ToList();
        names.Should().Contain(n => n.Contains("embed_tokens"));
        names.Should().Contain(n => n.Contains("layers"));
        names.Should().Contain(n => n.Contains("norm"));
        names.Should().Contain(n => n.Contains("lm_head"));
    }

    [Fact]
    public void TieWordEmbeddings_True_SharesEmbeddingAndLmHeadWeights()
    {
        var config = MakeConfig(tied: true);
        using var model = new LlamaForCausalLM(config);

        var embed = model.named_parameters()
            .First(p => p.name.Contains("embed_tokens.weight"))
            .parameter;
        var lmHead = model.named_parameters()
            .First(p => p.name.Contains("lm_head.weight"))
            .parameter;

        // For tied embeddings, the tensors share underlying storage; modifying one shows in the other.
        // Verify via comparing data pointers (or shapes + element-equality after modification).
        embed.shape.Should().Equal(lmHead.shape);
        using var diff = (embed - lmHead).abs().max();
        diff.cpu().item<float>().Should().BeLessThan(1e-6f);
    }

    [Fact]
    public void TieWordEmbeddings_False_HasSeparateLmHead()
    {
        var config = MakeConfig(tied: false);
        using var model = new LlamaForCausalLM(config);

        var paramNames = model.named_parameters().Select(p => p.name).ToList();
        // We should see distinct parameters; lm_head weights need not equal embed_tokens.
        paramNames.Should().Contain(n => n.Contains("lm_head.weight"));
        paramNames.Should().Contain(n => n.Contains("embed_tokens.weight"));
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Architectures/Llama/LlamaForCausalLMTests.cs
git commit -m "test(transformers): add failing LlamaForCausalLM tests"
```

- [ ] **Step 3: Implement `LlamaForCausalLM`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// LLaMA causal language model: embed_tokens → N × LlamaDecoderLayer → final RMSNorm → lm_head.
/// </summary>
public sealed class LlamaForCausalLM : Module<Tensor, Tensor>
{
    private readonly LlamaConfig config;
    private readonly Embedding embed_tokens;
    private readonly ModuleList<LlamaDecoderLayer> layers;
    private readonly LlamaRMSNorm norm;
    private readonly Linear lm_head;

    /// <summary>Initializes a new instance of the <see cref="LlamaForCausalLM"/> class.</summary>
    /// <param name="config">Architecture configuration.</param>
    public LlamaForCausalLM(LlamaConfig config)
        : base(nameof(LlamaForCausalLM))
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        this.config = config;

        this.embed_tokens = Embedding(config.VocabSize, config.HiddenSize);

        var decoderLayers = new List<LlamaDecoderLayer>(config.NumHiddenLayers);
        for (var i = 0; i < config.NumHiddenLayers; i++)
        {
            decoderLayers.Add(new LlamaDecoderLayer(config));
        }

        this.layers = ModuleList<LlamaDecoderLayer>(decoderLayers.ToArray());
        this.norm = new LlamaRMSNorm(config.HiddenSize, config.RmsNormEps);
        this.lm_head = Linear(config.HiddenSize, config.VocabSize, hasBias: false);

        if (config.TieWordEmbeddings)
        {
            // Make lm_head.weight share storage with embed_tokens.weight.
            this.lm_head.weight = this.embed_tokens.weight;
        }

        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor inputIds)
    {
        ArgumentNullException.ThrowIfNull(inputIds);

        var batch = inputIds.shape[0];
        var seq = inputIds.shape[1];

        using var positionIds = arange(seq, device: inputIds.device, dtype: ScalarType.Int64)
            .unsqueeze(0)
            .expand(batch, seq);

        using var hiddenStates = this.embed_tokens.forward(inputIds);
        var current = hiddenStates;
        var ownsCurrent = false;

        foreach (var layer in this.layers)
        {
            var next = layer.forward(current, attentionMask: null, positionIds: positionIds);
            if (ownsCurrent)
            {
                current.Dispose();
            }

            current = next;
            ownsCurrent = true;
        }

        try
        {
            using var normalized = this.norm.forward(current);
            return this.lm_head.forward(normalized);
        }
        finally
        {
            if (ownsCurrent)
            {
                current.Dispose();
            }
        }
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaForCausalLMTests"
```
Expected: 4 tests pass.

If `TieWordEmbeddings_True_SharesEmbeddingAndLmHeadWeights` fails because TorchSharp's `Linear.weight` setter doesn't accept replacement (some versions are read-only): in the constructor, use `register_parameter("weight", this.embed_tokens.weight)` on the `lm_head` module, or expose the lm_head weight by manual matmul instead of `Linear`. Adapt to actual API.

```powershell
git add src/LLMCompressorSharp.Transformers/Architectures/Llama/LlamaForCausalLM.cs
git commit -m "feat(transformers): implement LlamaForCausalLM"
```

---

### Task 9: Cleanup and full verification

**Files:**
- Delete: `src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`

- [ ] **Step 1: Verify the placeholder no longer exists**

Run: `Test-Path src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`
If `True`, run `Remove-Item src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`. If `False`, skip — the placeholder was already removed in Phase 0 task 13.

- [ ] **Step 2: Full clean run**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected:
- Build: 0 errors, 0 warnings
- Tests: **~138 passing** (119 prior + 5 RMSNorm + 4 RotaryEmbedding + 3 MLP + 4 Attention + 3 DecoderLayer + 4 ForCausalLM = 142; minor adjustments for [Theory] row counts)

- [ ] **Step 3: Commit placeholder removal (only if needed)**

```powershell
git rm src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs
git commit -m "chore(transformers): remove PlaceholderMarker (LLaMA architecture present)"
```

If the placeholder was already gone, skip the commit.

---

### Task 10: Merge and tag — STOP here

Controller handles the merge to `main` and the `v0.3.0-alpha` tag.

---

## Self-Review Notes

**Spec coverage:** every Architectures/Llama item from spec §2.2 is implemented (LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM). HuggingFace loader and tokenizer deferred to 3b/3c per the split.

**Out of scope (per the plan-split):**
- HF cache resolver (already exists from Phase 0, but the full HuggingFaceLoader with download + sharded safetensors is Phase 3b)
- `LlamaConfigParser` (JSON → LlamaConfig) — Phase 3b
- `LlamaTokenizer` — Phase 3c
- Numerical parity with Python `transformers` — Phase 3c (requires real weights)
- KV cache for autoregressive generation — deferred

**Type consistency:**
- All modules inherit `Module` (non-generic) or `Module<Tensor, Tensor>` from TorchSharp.
- `LlamaRotaryEmbedding.Apply(q, k, positionIds)` returns a named tuple; used consistently in `LlamaAttention`.
- `LlamaAttention.forward(x, attentionMask, positionIds)` and `LlamaDecoderLayer.forward(x, attentionMask, positionIds)` have identical signatures — the layer passes through directly.
- `LlamaConfig.Validate()` is called in every module that consumes a config (defensive but cheap).
- Snake_case field names (`q_proj`, `k_proj`, etc.) match HuggingFace naming so Phase 3b's weight loader can map them 1:1.

**Known risks:**
- **`Linear.weight` setter for tied embeddings.** TorchSharp may or may not allow direct replacement. The fallback (matmul against `embed_tokens.weight^T` instead of a `Linear`) is documented in Task 8.
- **`Tensor.alias()` in `RepeatKv`.** May not exist; fall back to `.clone()` or `.detach()` if the build fails.
- **`Embedding`'s parameter name.** TorchSharp's `Embedding` exposes `weight` — verify via `named_parameters()` output during tests; should be `embed_tokens.weight` once registered under that name.
- **`arange(seq, device, dtype)` overload.** May differ — the alternative is `arange(seq).to(device).to(dtype)`.

**Commit count target:** ~13 commits across 10 tasks (TDD pairs for each module).

# Phase 3b: HuggingFace Loader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load LLaMA-family models from the standard HuggingFace cache into `LlamaForCausalLM`. Parse `config.json` → `LlamaConfig`. Read single-file and sharded safetensors. Map HuggingFace's `model.layers.N.*` weight names onto our parameter hierarchy. No network downloads in this phase — those come later or via the existing `huggingface_hub` CLI. Tests use synthetic safetensors fixtures.

**Architecture:**

```
LLMCompressorSharp.Transformers.Loading/
   ├── LlamaConfigParser              : config.json bytes → LlamaConfig
   ├── HuggingFaceWeightNameMapper    : HF weight name → our parameter name (strip "model." prefix)
   ├── SafetensorsManifestParser      : model.safetensors.index.json → Dict<weightName, shardFile>
   ├── ModelWeightsLoader             : merges all shards in a snapshot dir into a single state-dict
   ├── HuggingFaceLoader              : orchestrator (repoId, revision) → LlamaConfig + state-dict
   └── LlamaModelLoader               : applies a state-dict onto a LlamaForCausalLM instance
```

All loaders are **cache-resolved-only** in Phase 3b: they look in `HuggingFaceCache.GetSnapshotPath(...)` and fail with an actionable message if files are missing. Phase 6 adds a downloader.

**Tech Stack:** `System.Text.Json` for `config.json`, `TorchSharp.PyBridge.Safetensors.LoadStateDict` for safetensors I/O (already proven in Phase 2a's `SafetensorsWriter`). Builds on Phase 3a's `LlamaForCausalLM` + `LlamaConfig` and Phase 0's `HuggingFaceCache`.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (`Loading/HuggingFaceLoader`, `Loading/LlamaConfigParser`)  
**Reference convention:** `docs/llmcompressorsharp/cache-conventions.md`

---

## File Structure

```
src/LLMCompressorSharp.Transformers/
└── Loading/                                  ← NEW
    ├── LlamaConfigParser.cs
    ├── HuggingFaceWeightNameMapper.cs
    ├── SafetensorsManifestParser.cs
    ├── ModelWeightsLoader.cs
    ├── LlamaModelLoader.cs
    └── HuggingFaceLoader.cs

tests/LLMCompressorSharp.Tests/
└── Transformers/
    └── Loading/                              ← NEW
        ├── LlamaConfigParserTests.cs
        ├── HuggingFaceWeightNameMapperTests.cs
        ├── SafetensorsManifestParserTests.cs
        ├── ModelWeightsLoaderTests.cs
        ├── LlamaModelLoaderTests.cs
        └── HuggingFaceLoaderTests.cs
```

**Responsibility per file:**

- `LlamaConfigParser` — `Parse(string json) → LlamaConfig` and `LoadFromFile(string path) → LlamaConfig`. Field mapping is explicit (snake_case JSON keys → PascalCase properties). Unknown keys are ignored (so HF additions like `attention_dropout` don't fail parsing). Required fields throw with clear messages.

- `HuggingFaceWeightNameMapper` — Strips the `model.` prefix that HF uses to wrap their `LlamaModel` inside `LlamaForCausalLM`. Static method `MapName(string hfName) → string ourName`. Also exposes `MapDictionary(IDictionary<string, Tensor>) → Dictionary<string, Tensor>` that returns a remapped copy.

- `SafetensorsManifestParser` — `Parse(string json) → SafetensorsManifest` record `(IReadOnlyDictionary<string, string> WeightMap, long? TotalSize)`. Manifest format documented at https://huggingface.co/docs/safetensors/index#format.

- `ModelWeightsLoader` — Given a snapshot directory, locates either `model.safetensors` (single file) or `model.safetensors.index.json` + shards (sharded). Returns a single merged `IDictionary<string, Tensor>` keyed by the original HF names. **Disposable** because the loaded tensors need lifetime management.

- `LlamaModelLoader` — `Load(LlamaForCausalLM model, IDictionary<string, Tensor> hfWeights, bool strict = true)`. Iterates over the model's `named_parameters()`, looks up the HF name (via `HuggingFaceWeightNameMapper.UnmapName(ourName)`), copies the tensor data via `parameter.copy_(hfWeights[hfName])`. Validates shape match. With `strict: true`, fails if any model parameter has no source weight or if any source weight is unused.

- `HuggingFaceLoader` — Top-level orchestrator: `Load(string repoId, string? revision = "main", IEnvironment? environment = null) → LoadedLlamaModel`. Resolves the snapshot directory via `HuggingFaceCache.GetSnapshotPath`, parses `config.json`, constructs `LlamaForCausalLM(config)`, calls `ModelWeightsLoader` + `LlamaModelLoader`. Returns a `LoadedLlamaModel` record `(LlamaForCausalLM Model, LlamaConfig Config, string SnapshotPath)`. Throws `HuggingFaceLoadException` with actionable diagnostic on every failure mode (cache miss, malformed config, shape mismatch).

- `HuggingFaceLoadException` — Specific exception type so callers can distinguish loader failures from generic IOException.

---

## Prerequisites & Conventions

- Phase 3a is merged. Tag `v0.3.0-alpha`. 142 tests passing on `main`.
- `Microsoft.Extensions.Logging.Abstractions` is NOT a dependency — keep loaders dependency-free; use `throw` for diagnostics.
- Branch off `main` as `feature/3b-huggingface-loader`.
- Tests build synthetic safetensors via `TorchSharp.PyBridge.Safetensors.SaveStateDict` in `xunit.IClassFixture` setup, then exercise the loaders against the temp file.

---

### Task 1: Branch + baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
Expected: clean tree, HEAD `76a6405`, tags through `v0.3.0-alpha`.

- [ ] **Step 2:** `git checkout -b feature/3b-huggingface-loader`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"` — expect 142 passing.

No commit.

---

### Task 2: TDD `LlamaConfigParser`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/LlamaConfigParserTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/LlamaConfigParser.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="LlamaConfigParser"/> — JSON to LlamaConfig conversion.
/// </summary>
public class LlamaConfigParserTests
{
    [Fact]
    public void Parse_SmolLM2Config_ExtractsAllRequiredFields()
    {
        var json = @"
{
  ""architectures"": [""LlamaForCausalLM""],
  ""hidden_size"": 576,
  ""intermediate_size"": 1536,
  ""num_hidden_layers"": 30,
  ""num_attention_heads"": 9,
  ""num_key_value_heads"": 3,
  ""vocab_size"": 49152,
  ""max_position_embeddings"": 8192,
  ""rope_theta"": 100000.0,
  ""rms_norm_eps"": 1e-5,
  ""hidden_act"": ""silu"",
  ""tie_word_embeddings"": true
}";

        var config = LlamaConfigParser.Parse(json);

        config.HiddenSize.Should().Be(576);
        config.IntermediateSize.Should().Be(1536);
        config.NumHiddenLayers.Should().Be(30);
        config.NumAttentionHeads.Should().Be(9);
        config.NumKeyValueHeads.Should().Be(3);
        config.VocabSize.Should().Be(49152);
        config.MaxPositionEmbeddings.Should().Be(8192);
        config.RopeTheta.Should().BeApproximately(100000f, 0.1f);
        config.RmsNormEps.Should().BeApproximately(1e-5f, 1e-7f);
        config.HiddenAct.Should().Be("silu");
        config.TieWordEmbeddings.Should().BeTrue();
    }

    [Fact]
    public void Parse_IgnoresUnknownFields()
    {
        var json = @"
{
  ""hidden_size"": 16,
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100,
  ""attention_dropout"": 0.1,
  ""some_future_field"": ""whatever""
}";

        var act = () => LlamaConfigParser.Parse(json);
        act.Should().NotThrow();
    }

    [Fact]
    public void Parse_MissingRequiredField_Throws()
    {
        // hidden_size omitted
        var json = @"
{
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100
}";

        var act = () => LlamaConfigParser.Parse(json);
        act.Should().Throw<HuggingFaceLoadException>()
            .WithMessage("*hidden_size*");
    }

    [Fact]
    public void Parse_DefaultsApplied_WhenOptionalFieldsAbsent()
    {
        var json = @"
{
  ""hidden_size"": 16,
  ""intermediate_size"": 64,
  ""num_hidden_layers"": 2,
  ""num_attention_heads"": 4,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 100
}";

        var config = LlamaConfigParser.Parse(json);

        config.MaxPositionEmbeddings.Should().Be(2048);
        config.RopeTheta.Should().BeApproximately(10000f, 1f);
        config.RmsNormEps.Should().BeApproximately(1e-5f, 1e-7f);
        config.HiddenAct.Should().Be("silu");
        config.TieWordEmbeddings.Should().BeTrue();
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        var act = () => LlamaConfigParser.Parse("not json {");
        act.Should().Throw<HuggingFaceLoadException>();
    }

    [Fact]
    public void LoadFromFile_NonExistentPath_Throws()
    {
        var act = () => LlamaConfigParser.LoadFromFile("/nonexistent/path/config.json");
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void LoadFromFile_ReadsAndParses()
    {
        var tmp = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tmp, @"{
  ""hidden_size"": 8,
  ""intermediate_size"": 32,
  ""num_hidden_layers"": 1,
  ""num_attention_heads"": 2,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 50
}");
            var config = LlamaConfigParser.LoadFromFile(tmp);
            config.HiddenSize.Should().Be(8);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}
```

- [ ] **Step 2: Verify CS0246 build failure**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaConfigParserTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/LlamaConfigParserTests.cs
git commit -m "test(transformers): add failing LlamaConfigParser tests"
```

- [ ] **Step 4: Implement `HuggingFaceLoadException`**

Create `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoadException.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Thrown when <see cref="HuggingFaceLoader"/> or any of its helpers fails to locate, parse,
/// or apply a HuggingFace model resource.
/// </summary>
public sealed class HuggingFaceLoadException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="HuggingFaceLoadException"/> class.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    public HuggingFaceLoadException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="HuggingFaceLoadException"/> class with an inner cause.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    /// <param name="innerException">Underlying exception.</param>
    public HuggingFaceLoadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
```

- [ ] **Step 5: Implement `LlamaConfigParser`**

Create `src/LLMCompressorSharp.Transformers/Loading/LlamaConfigParser.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.Json;
using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parses HuggingFace <c>config.json</c> into a <see cref="LlamaConfig"/>.
/// </summary>
/// <remarks>
/// Unknown fields are ignored so HF-side additions don't break loading. Required fields throw
/// <see cref="HuggingFaceLoadException"/> with the offending field name in the message.
/// </remarks>
public static class LlamaConfigParser
{
    /// <summary>Parses a JSON string into a <see cref="LlamaConfig"/>.</summary>
    /// <param name="json">The <c>config.json</c> contents.</param>
    /// <returns>The parsed config (validated).</returns>
    /// <exception cref="HuggingFaceLoadException">If the JSON is malformed or missing required fields.</exception>
    public static LlamaConfig Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HuggingFaceLoadException("config.json is not valid JSON.", ex);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new HuggingFaceLoadException("config.json root must be a JSON object.");
            }

            var config = new LlamaConfig
            {
                HiddenSize = RequireInt(root, "hidden_size"),
                IntermediateSize = RequireInt(root, "intermediate_size"),
                NumHiddenLayers = RequireInt(root, "num_hidden_layers"),
                NumAttentionHeads = RequireInt(root, "num_attention_heads"),
                NumKeyValueHeads = OptionalInt(root, "num_key_value_heads") ?? RequireInt(root, "num_attention_heads"),
                VocabSize = RequireInt(root, "vocab_size"),
                MaxPositionEmbeddings = OptionalInt(root, "max_position_embeddings") ?? 2048,
                RopeTheta = OptionalFloat(root, "rope_theta") ?? 10000f,
                RmsNormEps = OptionalFloat(root, "rms_norm_eps") ?? 1e-5f,
                HiddenAct = OptionalString(root, "hidden_act") ?? "silu",
                TieWordEmbeddings = OptionalBool(root, "tie_word_embeddings") ?? true,
            };

            config.Validate();
            return config;
        }
    }

    /// <summary>Loads and parses a <c>config.json</c> file.</summary>
    /// <param name="path">Path to the config file.</param>
    /// <returns>The parsed config.</returns>
    /// <exception cref="HuggingFaceLoadException">If the file is missing, unreadable, or malformed.</exception>
    public static LlamaConfig LoadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new HuggingFaceLoadException($"config.json not found at '{path}'.");
        }

        string json;
        try
        {
            json = File.ReadAllText(path);
        }
        catch (IOException ex)
        {
            throw new HuggingFaceLoadException($"Failed to read config.json from '{path}'.", ex);
        }

        return Parse(json);
    }

    private static int RequireInt(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            throw new HuggingFaceLoadException($"config.json is missing required field '{field}'.");
        }

        if (prop.ValueKind != JsonValueKind.Number)
        {
            throw new HuggingFaceLoadException($"config.json field '{field}' must be a number.");
        }

        return prop.GetInt32();
    }

    private static int? OptionalInt(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.Number ? prop.GetInt32() : null;
    }

    private static float? OptionalFloat(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.Number ? (float)prop.GetDouble() : null;
    }

    private static string? OptionalString(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.String ? prop.GetString() : null;
    }

    private static bool? OptionalBool(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };
    }
}
```

- [ ] **Step 6: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaConfigParserTests"`
Expected: 7 tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoadException.cs src/LLMCompressorSharp.Transformers/Loading/LlamaConfigParser.cs
git commit -m "feat(transformers): implement LlamaConfigParser + HuggingFaceLoadException"
```

---

### Task 3: TDD `HuggingFaceWeightNameMapper`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/HuggingFaceWeightNameMapperTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceWeightNameMapper.cs`

The mapping rule: HF wraps everything in a `LlamaModel` subobject so all decoder weights are prefixed `model.`. Our `LlamaForCausalLM` flattens this. Mapping:

| HF name | Our name |
|---|---|
| `model.embed_tokens.weight` | `embed_tokens.weight` |
| `model.layers.0.self_attn.q_proj.weight` | `layers.0.self_attn.q_proj.weight` |
| `model.layers.0.input_layernorm.weight` | `layers.0.input_layernorm.weight` |
| `model.norm.weight` | `norm.weight` |
| `lm_head.weight` | `lm_head.weight` (no prefix — HF doesn't put lm_head under `model.`) |

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="HuggingFaceWeightNameMapper"/>.
/// </summary>
public class HuggingFaceWeightNameMapperTests
{
    [Theory]
    [InlineData("model.embed_tokens.weight", "embed_tokens.weight")]
    [InlineData("model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight")]
    [InlineData("model.layers.15.mlp.gate_proj.weight", "layers.15.mlp.gate_proj.weight")]
    [InlineData("model.layers.0.input_layernorm.weight", "layers.0.input_layernorm.weight")]
    [InlineData("model.layers.0.post_attention_layernorm.weight", "layers.0.post_attention_layernorm.weight")]
    [InlineData("model.norm.weight", "norm.weight")]
    [InlineData("lm_head.weight", "lm_head.weight")]
    public void MapName_StripsModelPrefix(string hfName, string expected)
    {
        HuggingFaceWeightNameMapper.MapName(hfName).Should().Be(expected);
    }

    [Theory]
    [InlineData("embed_tokens.weight", "model.embed_tokens.weight")]
    [InlineData("layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight")]
    [InlineData("norm.weight", "model.norm.weight")]
    [InlineData("lm_head.weight", "lm_head.weight")]
    public void UnmapName_AddsModelPrefix_ExceptLmHead(string ourName, string expected)
    {
        HuggingFaceWeightNameMapper.UnmapName(ourName).Should().Be(expected);
    }

    [Fact]
    public void MapDictionary_RemapsAllKeys()
    {
        using var t1 = zeros(2, 3);
        using var t2 = zeros(4);
        var hfDict = new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = t1,
            ["lm_head.weight"] = t2,
        };

        var mapped = HuggingFaceWeightNameMapper.MapDictionary(hfDict);

        mapped.Should().ContainKey("embed_tokens.weight");
        mapped.Should().ContainKey("lm_head.weight");
        mapped.Should().NotContainKey("model.embed_tokens.weight");
    }

    [Fact]
    public void MapName_NullOrEmpty_Throws()
    {
        var act = () => HuggingFaceWeightNameMapper.MapName(null!);
        act.Should().Throw<ArgumentException>();
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/HuggingFaceWeightNameMapperTests.cs
git commit -m "test(transformers): add failing HuggingFaceWeightNameMapper tests"
```

- [ ] **Step 3: Implement the mapper**

Create `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceWeightNameMapper.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Maps between HuggingFace's <c>model.*</c>-prefixed parameter names and our flattened
/// <c>LlamaForCausalLM</c> parameter hierarchy.
/// </summary>
public static class HuggingFaceWeightNameMapper
{
    private const string HfModelPrefix = "model.";
    private const string LmHeadName = "lm_head.weight";

    /// <summary>Maps an HF parameter name to our equivalent.</summary>
    /// <param name="hfName">HuggingFace name (e.g. <c>model.layers.0.self_attn.q_proj.weight</c>).</param>
    /// <returns>The corresponding name in our parameter hierarchy.</returns>
    public static string MapName(string hfName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(hfName);
        return hfName.StartsWith(HfModelPrefix, StringComparison.Ordinal)
            ? hfName[HfModelPrefix.Length..]
            : hfName;
    }

    /// <summary>Maps one of our parameter names back to HF's convention.</summary>
    /// <param name="ourName">Our parameter name (e.g. <c>layers.0.self_attn.q_proj.weight</c>).</param>
    /// <returns>The corresponding HF name.</returns>
    public static string UnmapName(string ourName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ourName);
        // lm_head is not prefixed by HF (it lives outside their LlamaModel wrapper).
        if (string.Equals(ourName, LmHeadName, StringComparison.Ordinal))
        {
            return ourName;
        }

        return HfModelPrefix + ourName;
    }

    /// <summary>Builds a new dictionary with all keys remapped via <see cref="MapName"/>.</summary>
    /// <param name="hfDict">Source dictionary keyed by HF names.</param>
    /// <returns>New dictionary keyed by our names; tensor values are shared (not copied).</returns>
    public static Dictionary<string, Tensor> MapDictionary(IDictionary<string, Tensor> hfDict)
    {
        ArgumentNullException.ThrowIfNull(hfDict);
        var result = new Dictionary<string, Tensor>(hfDict.Count, StringComparer.Ordinal);
        foreach (var (key, value) in hfDict)
        {
            result[MapName(key)] = value;
        }

        return result;
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~HuggingFaceWeightNameMapperTests"
```
Expected: 12 tests pass (4 [Fact] + 8 [Theory] rows).

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/HuggingFaceWeightNameMapper.cs
git commit -m "feat(transformers): implement HuggingFaceWeightNameMapper"
```

---

### Task 4: TDD `SafetensorsManifestParser`

A sharded safetensors manifest looks like:

```json
{
  "metadata": {"total_size": 4711234567},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.20.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors"
  }
}
```

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/SafetensorsManifestParserTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifest.cs` (record)
- Create: `src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifestParser.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="SafetensorsManifestParser"/>.
/// </summary>
public class SafetensorsManifestParserTests
{
    [Fact]
    public void Parse_ProducesWeightMapAndTotalSize()
    {
        var json = @"
{
  ""metadata"": {""total_size"": 4711234567},
  ""weight_map"": {
    ""model.embed_tokens.weight"": ""model-00001-of-00002.safetensors"",
    ""lm_head.weight"": ""model-00002-of-00002.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);

        manifest.TotalSize.Should().Be(4711234567L);
        manifest.WeightMap.Should().HaveCount(2);
        manifest.WeightMap["model.embed_tokens.weight"].Should().Be("model-00001-of-00002.safetensors");
        manifest.WeightMap["lm_head.weight"].Should().Be("model-00002-of-00002.safetensors");
    }

    [Fact]
    public void Parse_TotalSizeOptional_NullWhenAbsent()
    {
        var json = @"
{
  ""weight_map"": {
    ""x.weight"": ""shard.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);
        manifest.TotalSize.Should().BeNull();
        manifest.WeightMap.Should().HaveCount(1);
    }

    [Fact]
    public void Parse_DistinctShardsList()
    {
        var json = @"
{
  ""weight_map"": {
    ""a"": ""model-00001-of-00003.safetensors"",
    ""b"": ""model-00002-of-00003.safetensors"",
    ""c"": ""model-00001-of-00003.safetensors"",
    ""d"": ""model-00003-of-00003.safetensors""
  }
}";

        var manifest = SafetensorsManifestParser.Parse(json);
        manifest.DistinctShardFiles.Should().BeEquivalentTo(new[]
        {
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        });
    }

    [Fact]
    public void Parse_MissingWeightMap_Throws()
    {
        var json = @"{""metadata"": {""total_size"": 100}}";
        var act = () => SafetensorsManifestParser.Parse(json);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*weight_map*");
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        var act = () => SafetensorsManifestParser.Parse("not json");
        act.Should().Throw<HuggingFaceLoadException>();
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/SafetensorsManifestParserTests.cs
git commit -m "test(transformers): add failing SafetensorsManifestParser tests"
```

- [ ] **Step 3: Implement**

Create `src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifest.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parsed contents of a <c>model.safetensors.index.json</c> sharded-checkpoint manifest.
/// </summary>
/// <param name="WeightMap">Map from weight name to the shard filename containing it.</param>
/// <param name="TotalSize">Optional total size in bytes across all shards (HF metadata).</param>
public sealed record SafetensorsManifest(
    IReadOnlyDictionary<string, string> WeightMap,
    long? TotalSize)
{
    /// <summary>Gets the distinct set of shard files referenced by <see cref="WeightMap"/>.</summary>
    public IReadOnlyCollection<string> DistinctShardFiles =>
        new HashSet<string>(this.WeightMap.Values, StringComparer.Ordinal);
}
```

Create `src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifestParser.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.Json;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parses HuggingFace <c>model.safetensors.index.json</c> manifests for sharded checkpoints.
/// </summary>
public static class SafetensorsManifestParser
{
    /// <summary>Parses a JSON string into a <see cref="SafetensorsManifest"/>.</summary>
    /// <param name="json">The manifest contents.</param>
    /// <returns>The parsed manifest.</returns>
    /// <exception cref="HuggingFaceLoadException">If the JSON is malformed or missing <c>weight_map</c>.</exception>
    public static SafetensorsManifest Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HuggingFaceLoadException("safetensors manifest is not valid JSON.", ex);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (!root.TryGetProperty("weight_map", out var weightMapElement)
                || weightMapElement.ValueKind != JsonValueKind.Object)
            {
                throw new HuggingFaceLoadException("safetensors manifest is missing 'weight_map'.");
            }

            var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var entry in weightMapElement.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.String)
                {
                    throw new HuggingFaceLoadException(
                        $"weight_map entry for '{entry.Name}' must be a string filename.");
                }

                weightMap[entry.Name] = entry.Value.GetString()!;
            }

            long? totalSize = null;
            if (root.TryGetProperty("metadata", out var meta)
                && meta.ValueKind == JsonValueKind.Object
                && meta.TryGetProperty("total_size", out var sizeProp)
                && sizeProp.ValueKind == JsonValueKind.Number)
            {
                totalSize = sizeProp.GetInt64();
            }

            return new SafetensorsManifest(weightMap, totalSize);
        }
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SafetensorsManifestParserTests"
```
Expected: 5 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifest.cs src/LLMCompressorSharp.Transformers/Loading/SafetensorsManifestParser.cs
git commit -m "feat(transformers): implement SafetensorsManifestParser"
```

---

### Task 5: TDD `ModelWeightsLoader`

Loads either single-file or sharded safetensors from a snapshot directory.

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/ModelWeightsLoaderTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/ModelWeightsLoader.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using TorchSharp.PyBridge;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="ModelWeightsLoader"/> — single-file and sharded safetensors loading.
/// </summary>
public class ModelWeightsLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public ModelWeightsLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"llmc-loader-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }

    [Fact]
    public void Load_SingleFileSafetensors_ReturnsAllTensors()
    {
        // Arrange: write a model.safetensors with two tensors
        using var t1 = tensor(new float[] { 1f, 2f, 3f });
        using var t2 = tensor(new float[] { 4f, 5f });
        var saveDict = new Dictionary<string, Tensor>
        {
            ["embed.weight"] = t1,
            ["lm_head.weight"] = t2,
        };
        var path = Path.Combine(_tempDir, "model.safetensors");
        Safetensors.SaveStateDict(path, saveDict);

        // Act
        var loaded = ModelWeightsLoader.LoadFromSnapshot(_tempDir);

        // Assert
        loaded.Should().HaveCount(2);
        loaded.Should().ContainKey("embed.weight");
        loaded.Should().ContainKey("lm_head.weight");
        loaded["embed.weight"].cpu().data<float>().ToArray().Should().Equal(new float[] { 1f, 2f, 3f });

        foreach (var t in loaded.Values)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void Load_ShardedSafetensors_MergesAcrossShards()
    {
        // Arrange: two shards plus index file
        using var t1 = tensor(new float[] { 1f, 2f });
        using var t2 = tensor(new float[] { 3f, 4f });
        Safetensors.SaveStateDict(
            Path.Combine(_tempDir, "model-00001-of-00002.safetensors"),
            new Dictionary<string, Tensor> { ["a"] = t1 });
        Safetensors.SaveStateDict(
            Path.Combine(_tempDir, "model-00002-of-00002.safetensors"),
            new Dictionary<string, Tensor> { ["b"] = t2 });

        File.WriteAllText(Path.Combine(_tempDir, "model.safetensors.index.json"), @"
{
  ""metadata"": {""total_size"": 32},
  ""weight_map"": {
    ""a"": ""model-00001-of-00002.safetensors"",
    ""b"": ""model-00002-of-00002.safetensors""
  }
}");

        // Act
        var loaded = ModelWeightsLoader.LoadFromSnapshot(_tempDir);

        // Assert
        loaded.Should().HaveCount(2);
        loaded["a"].cpu().data<float>().ToArray().Should().Equal(new float[] { 1f, 2f });
        loaded["b"].cpu().data<float>().ToArray().Should().Equal(new float[] { 3f, 4f });

        foreach (var t in loaded.Values)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void Load_NeitherSingleNorSharded_Throws()
    {
        var act = () => ModelWeightsLoader.LoadFromSnapshot(_tempDir);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*model.safetensors*");
    }

    [Fact]
    public void Load_NonExistentDirectory_Throws()
    {
        var act = () => ModelWeightsLoader.LoadFromSnapshot("/nonexistent/path/12345");
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/ModelWeightsLoaderTests.cs
git commit -m "test(transformers): add failing ModelWeightsLoader tests"
```

- [ ] **Step 3: Implement**

Create `src/LLMCompressorSharp.Transformers/Loading/ModelWeightsLoader.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Loads model weights from a HuggingFace snapshot directory, handling both single-file and
/// sharded safetensors layouts.
/// </summary>
public static class ModelWeightsLoader
{
    private const string SingleFile = "model.safetensors";
    private const string ShardManifestFile = "model.safetensors.index.json";

    /// <summary>Loads all weights from a snapshot directory into a merged dictionary.</summary>
    /// <param name="snapshotDir">The HF cache snapshot directory.</param>
    /// <returns>A dictionary mapping weight name (HF convention) to tensor.</returns>
    /// <exception cref="HuggingFaceLoadException">If the directory is missing or no weight file is present.</exception>
    public static IDictionary<string, Tensor> LoadFromSnapshot(string snapshotDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDir);

        if (!Directory.Exists(snapshotDir))
        {
            throw new HuggingFaceLoadException($"Snapshot directory not found: '{snapshotDir}'.");
        }

        var singleFilePath = Path.Combine(snapshotDir, SingleFile);
        var manifestPath = Path.Combine(snapshotDir, ShardManifestFile);

        if (File.Exists(singleFilePath))
        {
            return LoadSingleFile(singleFilePath);
        }

        if (File.Exists(manifestPath))
        {
            return LoadSharded(snapshotDir, manifestPath);
        }

        throw new HuggingFaceLoadException(
            $"Neither '{SingleFile}' nor '{ShardManifestFile}' found in snapshot '{snapshotDir}'.");
    }

    private static IDictionary<string, Tensor> LoadSingleFile(string path)
    {
        try
        {
            return Safetensors.LoadStateDict(path);
        }
        catch (Exception ex) when (ex is not HuggingFaceLoadException)
        {
            throw new HuggingFaceLoadException($"Failed to load safetensors file '{path}'.", ex);
        }
    }

    private static IDictionary<string, Tensor> LoadSharded(string snapshotDir, string manifestPath)
    {
        string manifestJson;
        try
        {
            manifestJson = File.ReadAllText(manifestPath);
        }
        catch (IOException ex)
        {
            throw new HuggingFaceLoadException($"Failed to read manifest '{manifestPath}'.", ex);
        }

        var manifest = SafetensorsManifestParser.Parse(manifestJson);

        var merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var shardFile in manifest.DistinctShardFiles)
        {
            var shardPath = Path.Combine(snapshotDir, shardFile);
            if (!File.Exists(shardPath))
            {
                throw new HuggingFaceLoadException(
                    $"Shard file '{shardFile}' referenced by manifest is missing in '{snapshotDir}'.");
            }

            IDictionary<string, Tensor> shardWeights;
            try
            {
                shardWeights = Safetensors.LoadStateDict(shardPath);
            }
            catch (Exception ex) when (ex is not HuggingFaceLoadException)
            {
                throw new HuggingFaceLoadException($"Failed to load shard '{shardPath}'.", ex);
            }

            foreach (var (name, tensor) in shardWeights)
            {
                if (merged.ContainsKey(name))
                {
                    throw new HuggingFaceLoadException(
                        $"Weight '{name}' appears in more than one shard.");
                }

                merged[name] = tensor;
            }
        }

        return merged;
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~ModelWeightsLoaderTests"
```
Expected: 4 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/ModelWeightsLoader.cs
git commit -m "feat(transformers): implement ModelWeightsLoader (single + sharded)"
```

---

### Task 6: TDD `LlamaModelLoader`

Applies a state dict (HF-named) onto a `LlamaForCausalLM` instance, with shape validation and strict-mode unused-key detection.

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/LlamaModelLoaderTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/LlamaModelLoader.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for <see cref="LlamaModelLoader"/>.
/// </summary>
public class LlamaModelLoaderTests
{
    private static LlamaConfig MakeConfig()
    {
        return new LlamaConfig
        {
            HiddenSize = 8,
            IntermediateSize = 32,
            NumHiddenLayers = 1,
            NumAttentionHeads = 2,
            NumKeyValueHeads = 2,
            VocabSize = 50,
            MaxPositionEmbeddings = 16,
            HiddenAct = "silu",
            TieWordEmbeddings = false,
        };
    }

    private static Dictionary<string, Tensor> BuildHfWeights(LlamaConfig config)
    {
        // Build a complete HF-named weight dictionary for the given config.
        // Untied: lm_head separate. Layer 0 has 4 attention projs, 3 MLP projs, 2 layernorms.
        // Plus embed_tokens and the final norm.
        var headProj = config.NumAttentionHeads * config.HeadDim; // = HiddenSize
        var kvProj = config.NumKeyValueHeads * config.HeadDim;
        return new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = randn(config.VocabSize, config.HiddenSize),
            ["model.layers.0.self_attn.q_proj.weight"] = randn(headProj, config.HiddenSize),
            ["model.layers.0.self_attn.k_proj.weight"] = randn(kvProj, config.HiddenSize),
            ["model.layers.0.self_attn.v_proj.weight"] = randn(kvProj, config.HiddenSize),
            ["model.layers.0.self_attn.o_proj.weight"] = randn(config.HiddenSize, headProj),
            ["model.layers.0.mlp.gate_proj.weight"] = randn(config.IntermediateSize, config.HiddenSize),
            ["model.layers.0.mlp.up_proj.weight"] = randn(config.IntermediateSize, config.HiddenSize),
            ["model.layers.0.mlp.down_proj.weight"] = randn(config.HiddenSize, config.IntermediateSize),
            ["model.layers.0.input_layernorm.weight"] = randn(config.HiddenSize),
            ["model.layers.0.post_attention_layernorm.weight"] = randn(config.HiddenSize),
            ["model.norm.weight"] = randn(config.HiddenSize),
            ["lm_head.weight"] = randn(config.VocabSize, config.HiddenSize),
        };
    }

    [Fact]
    public void Load_PopulatesAllParameters_WithMatchingShapes()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        try
        {
            using var model = new LlamaForCausalLM(config);

            LlamaModelLoader.Load(model, hfWeights);

            // Spot check: embed_tokens.weight should match the source.
            var loadedEmbed = model.named_parameters()
                .First(p => p.name.Contains("embed_tokens.weight"))
                .parameter;
            using var diff = (loadedEmbed - hfWeights["model.embed_tokens.weight"]).abs().max();
            diff.cpu().item<float>().Should().BeLessThan(1e-6f);
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_StrictMode_MissingWeight_Throws()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        hfWeights.Remove("model.layers.0.self_attn.q_proj.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: true);
            act.Should().Throw<HuggingFaceLoadException>().WithMessage("*q_proj*");
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_NonStrictMode_MissingWeight_DoesNotThrow()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        hfWeights.Remove("model.layers.0.self_attn.q_proj.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: false);
            act.Should().NotThrow();
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_ShapeMismatch_Throws()
    {
        var config = MakeConfig();
        var hfWeights = BuildHfWeights(config);
        // Replace embed_tokens with wrong shape
        hfWeights["model.embed_tokens.weight"].Dispose();
        hfWeights["model.embed_tokens.weight"] = randn(config.VocabSize + 1, config.HiddenSize);
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights);
            act.Should().Throw<HuggingFaceLoadException>().WithMessage("*shape*");
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_TiedEmbeddings_DoesNotRequireLmHeadInSource()
    {
        // When TieWordEmbeddings = true, the model's lm_head.weight aliases embed_tokens.weight,
        // and the HF source omits lm_head.weight.
        var config = MakeConfig();
        config.TieWordEmbeddings = true;
        var hfWeights = BuildHfWeights(config);
        hfWeights["lm_head.weight"].Dispose();
        hfWeights.Remove("lm_head.weight");
        try
        {
            using var model = new LlamaForCausalLM(config);
            var act = () => LlamaModelLoader.Load(model, hfWeights, strict: true);
            act.Should().NotThrow();
        }
        finally
        {
            foreach (var t in hfWeights.Values)
            {
                t.Dispose();
            }
        }
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/LlamaModelLoaderTests.cs
git commit -m "test(transformers): add failing LlamaModelLoader tests"
```

- [ ] **Step 3: Implement**

Create `src/LLMCompressorSharp.Transformers/Loading/LlamaModelLoader.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Applies a HuggingFace-named state dict onto a <see cref="LlamaForCausalLM"/>.
/// </summary>
public static class LlamaModelLoader
{
    private const string LmHeadKey = "lm_head.weight";
    private const string EmbedTokensKey = "model.embed_tokens.weight";

    /// <summary>
    /// Copies weights from <paramref name="hfWeights"/> into the model's parameters.
    /// </summary>
    /// <param name="model">The target model.</param>
    /// <param name="hfWeights">Source state dict keyed by HuggingFace names.</param>
    /// <param name="strict">When true, throws if any model parameter has no source weight.</param>
    /// <exception cref="HuggingFaceLoadException">On missing weights (strict) or shape mismatches.</exception>
    public static void Load(LlamaForCausalLM model, IDictionary<string, Tensor> hfWeights, bool strict = true)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(hfWeights);

        var missing = new List<string>();

        using (no_grad())
        {
            foreach (var (paramName, parameter) in model.named_parameters())
            {
                var hfName = HuggingFaceWeightNameMapper.UnmapName(paramName);

                if (!hfWeights.TryGetValue(hfName, out var source))
                {
                    // For tied embeddings, lm_head.weight aliases embed_tokens.weight — HF omits it.
                    // We skip the lm_head parameter; populating embed_tokens propagates to it through aliasing.
                    if (paramName.EndsWith(LmHeadKey, StringComparison.Ordinal)
                        && hfWeights.ContainsKey(EmbedTokensKey))
                    {
                        continue;
                    }

                    missing.Add(hfName);
                    continue;
                }

                if (!ShapesEqual(parameter.shape, source.shape))
                {
                    throw new HuggingFaceLoadException(
                        $"Shape mismatch for '{hfName}': model expects [{string.Join(",", parameter.shape)}] "
                        + $"but source has [{string.Join(",", source.shape)}].");
                }

                parameter.copy_(source.to(parameter.device));
            }
        }

        if (strict && missing.Count > 0)
        {
            throw new HuggingFaceLoadException(
                $"Missing weights in source for: {string.Join(", ", missing)}.");
        }
    }

    private static bool ShapesEqual(long[] a, long[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaModelLoaderTests"
```
Expected: 5 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/LlamaModelLoader.cs
git commit -m "feat(transformers): implement LlamaModelLoader (apply HF state dict)"
```

---

### Task 7: TDD `HuggingFaceLoader` orchestrator

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Loading/HuggingFaceLoaderTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Loading/LoadedLlamaModel.cs` (record)
- Create: `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoader.cs`

- [ ] **Step 1: Write the failing tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using TorchSharp.PyBridge;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Transformers.Loading;

/// <summary>
/// Tests for the <see cref="HuggingFaceLoader"/> orchestrator end-to-end.
/// </summary>
public class HuggingFaceLoaderTests : IDisposable
{
    private readonly string _cacheRoot;

    public HuggingFaceLoaderTests()
    {
        _cacheRoot = Path.Combine(Path.GetTempPath(), $"llmc-hfloader-test-{Guid.NewGuid():N}", "hub");
        Directory.CreateDirectory(_cacheRoot);
    }

    public void Dispose()
    {
        try
        {
            var parent = Directory.GetParent(_cacheRoot)?.FullName;
            if (!string.IsNullOrEmpty(parent) && Directory.Exists(parent))
            {
                Directory.Delete(parent, recursive: true);
            }
        }
        catch
        {
            // best-effort
        }
    }

    private sealed class FakeEnvironment : IEnvironment
    {
        public string? HfHubCache { get; init; }

        public string? HfHome { get; init; }

        public string? XdgCacheHome { get; init; }

        public string UserProfile { get; init; } = "/home/test";
    }

    [Fact]
    public void Load_FromSynthesizedSnapshot_ProducesPopulatedModel()
    {
        // Arrange: synthesize a minimal SmolLM2-like snapshot in the temp cache.
        var repoId = "FakeOrg/FakeRepo";
        var revision = "main";
        var snapshotDir = Path.Combine(_cacheRoot, HuggingFaceCache.GetRepoFolderName(repoId), "snapshots", revision);
        Directory.CreateDirectory(snapshotDir);

        // Write config.json
        File.WriteAllText(Path.Combine(snapshotDir, "config.json"), @"
{
  ""hidden_size"": 8,
  ""intermediate_size"": 32,
  ""num_hidden_layers"": 1,
  ""num_attention_heads"": 2,
  ""num_key_value_heads"": 2,
  ""vocab_size"": 50,
  ""max_position_embeddings"": 16,
  ""rope_theta"": 10000.0,
  ""rms_norm_eps"": 1e-5,
  ""hidden_act"": ""silu"",
  ""tie_word_embeddings"": false
}");

        // Write model.safetensors with the full HF state dict.
        var saveDict = new Dictionary<string, Tensor>
        {
            ["model.embed_tokens.weight"] = randn(50, 8),
            ["model.layers.0.self_attn.q_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.k_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.v_proj.weight"] = randn(8, 8),
            ["model.layers.0.self_attn.o_proj.weight"] = randn(8, 8),
            ["model.layers.0.mlp.gate_proj.weight"] = randn(32, 8),
            ["model.layers.0.mlp.up_proj.weight"] = randn(32, 8),
            ["model.layers.0.mlp.down_proj.weight"] = randn(8, 32),
            ["model.layers.0.input_layernorm.weight"] = randn(8),
            ["model.layers.0.post_attention_layernorm.weight"] = randn(8),
            ["model.norm.weight"] = randn(8),
            ["lm_head.weight"] = randn(50, 8),
        };
        try
        {
            Safetensors.SaveStateDict(Path.Combine(snapshotDir, "model.safetensors"), saveDict);

            var env = new FakeEnvironment { HfHubCache = _cacheRoot };

            // Act
            var result = HuggingFaceLoader.Load(repoId, revision, env);

            using (result.Model)
            {
                // Assert
                result.Config.HiddenSize.Should().Be(8);
                result.SnapshotPath.Should().Be(snapshotDir);

                // Forward pass should run with the loaded weights.
                using var inputIds = randint(0, 50, new long[] { 1, 3 }, dtype: ScalarType.Int64);
                using var logits = result.Model.forward(inputIds);
                logits.shape.Should().Equal(new long[] { 1, 3, 50 });
            }
        }
        finally
        {
            foreach (var t in saveDict.Values)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void Load_MissingSnapshot_Throws()
    {
        var env = new FakeEnvironment { HfHubCache = _cacheRoot };
        var act = () => HuggingFaceLoader.Load("NoSuch/Repo", "main", env);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void Load_MissingConfigJson_Throws()
    {
        var repoId = "Org/Repo";
        var snapshotDir = Path.Combine(_cacheRoot, HuggingFaceCache.GetRepoFolderName(repoId), "snapshots", "main");
        Directory.CreateDirectory(snapshotDir);
        // Don't write config.json

        var env = new FakeEnvironment { HfHubCache = _cacheRoot };
        var act = () => HuggingFaceLoader.Load(repoId, "main", env);
        act.Should().Throw<HuggingFaceLoadException>().WithMessage("*config.json*");
    }
}
```

- [ ] **Step 2: Verify failure, commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Loading/HuggingFaceLoaderTests.cs
git commit -m "test(transformers): add failing HuggingFaceLoader tests"
```

- [ ] **Step 3: Implement**

Create `src/LLMCompressorSharp.Transformers/Loading/LoadedLlamaModel.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// The result of loading a LLaMA model from HuggingFace.
/// </summary>
/// <param name="Model">The instantiated model with weights loaded.</param>
/// <param name="Config">The parsed config.</param>
/// <param name="SnapshotPath">The on-disk snapshot directory the model was loaded from.</param>
public sealed record LoadedLlamaModel(
    LlamaForCausalLM Model,
    LlamaConfig Config,
    string SnapshotPath);
```

Create `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoader.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Top-level orchestrator for loading a LLaMA model from the standard HuggingFace cache.
/// </summary>
/// <remarks>
/// Follows the rules in <c>docs/llmcompressorsharp/cache-conventions.md</c>: resolves the cache
/// root via <see cref="HuggingFaceCache"/>, looks up the snapshot directory, parses config.json,
/// and loads safetensors (single-file or sharded). Does not download — assumes the model is
/// already present in the cache. Phase 6 adds a downloader.
/// </remarks>
public static class HuggingFaceLoader
{
    /// <summary>Loads a model from the HuggingFace cache.</summary>
    /// <param name="repoId">Repo id in <c>org/repo</c> form.</param>
    /// <param name="revision">Revision or branch; default <c>main</c>.</param>
    /// <param name="environment">Environment provider for cache resolution; defaults to <see cref="SystemEnvironment.Instance"/>.</param>
    /// <returns>The loaded model + config + snapshot path.</returns>
    /// <exception cref="HuggingFaceLoadException">On any cache miss, parse, or shape error.</exception>
    public static LoadedLlamaModel Load(
        string repoId,
        string? revision = "main",
        IEnvironment? environment = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        var rev = revision ?? "main";
        var env = environment ?? SystemEnvironment.Instance;

        var cacheRoot = HuggingFaceCache.ResolveCacheRoot(env);
        var snapshotDir = HuggingFaceCache.GetSnapshotPath(cacheRoot, repoId, rev);

        if (!Directory.Exists(snapshotDir))
        {
            throw new HuggingFaceLoadException(
                $"Snapshot for '{repoId}@{rev}' not found at '{snapshotDir}'. "
                + "Run `huggingface-cli download {repoId} --revision {rev}` or set HF_HUB_CACHE.");
        }

        var configPath = Path.Combine(snapshotDir, "config.json");
        if (!File.Exists(configPath))
        {
            throw new HuggingFaceLoadException(
                $"config.json not found in snapshot '{snapshotDir}' for repo '{repoId}@{rev}'.");
        }

        var config = LlamaConfigParser.LoadFromFile(configPath);
        var model = new LlamaForCausalLM(config);

        IDictionary<string, TorchSharp.torch.Tensor>? weights = null;
        try
        {
            weights = ModelWeightsLoader.LoadFromSnapshot(snapshotDir);
            LlamaModelLoader.Load(model, weights);
        }
        catch
        {
            model.Dispose();
            throw;
        }
        finally
        {
            if (weights is not null)
            {
                foreach (var t in weights.Values)
                {
                    t.Dispose();
                }
            }
        }

        return new LoadedLlamaModel(model, config, snapshotDir);
    }
}
```

- [ ] **Step 4: Run tests, commit**

```
dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~HuggingFaceLoaderTests"
```
Expected: 3 tests pass.

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/LoadedLlamaModel.cs src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoader.cs
git commit -m "feat(transformers): implement HuggingFaceLoader orchestrator"
```

---

### Task 8: Full verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Full clean run**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```
Expected: 0 errors, 0 warnings, **~178 tests passing** (142 + 7 + 12 + 5 + 4 + 5 + 3 = 178).

- [ ] **Step 2: File inventory check**

```powershell
Get-ChildItem -Recurse -Filter "*.cs" -Path src/LLMCompressorSharp.Transformers/Loading/ | ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\', '') } | Sort-Object
```
Expected 7 files: HuggingFaceLoadException, LlamaConfigParser, HuggingFaceWeightNameMapper, SafetensorsManifest, SafetensorsManifestParser, ModelWeightsLoader, LlamaModelLoader, HuggingFaceLoader, LoadedLlamaModel — that's 9. Adjust expectation if I miscounted.

No commit — verification only.

---

### Task 9: STOP — controller handles merge + tag

Tag will be `v0.3.1-alpha`.

---

## Self-Review Notes

**Spec coverage:** every Loading/* item from spec §2.2:
- `HuggingFaceLoader` orchestrator → Task 7
- HF cache layout resolution → already in Phase 0's `HuggingFaceCache`; used in Task 7
- Sharded safetensors → Tasks 4, 5
- `LlamaConfigParser` → Task 2

**Out of scope (Phase 3c / Phase 6):**
- Network downloads — Phase 6's CLI delegates to `huggingface-cli download` or implements HTTP fetch
- Tokenizer loading — Phase 3c
- Compressed-tensors output writer — Phase 5

**Type consistency:**
- `HuggingFaceLoadException` used by all loaders for uniform diagnostics
- `LoadedLlamaModel` exposes `Model`, `Config`, `SnapshotPath`
- `HuggingFaceWeightNameMapper.UnmapName` handles the `lm_head` special case (no `model.` prefix)
- `LlamaModelLoader.Load(model, hfWeights, strict)` matches the test signatures

**Known risks:**
- `parameter.copy_(source)` may need `source.to(parameter.device)` if source is CPU and parameter is GPU. The implementation handles this.
- `TorchSharp.PyBridge.Safetensors.SaveStateDict` and `LoadStateDict` — already proven in Phase 2a.
- `Safetensors.SaveStateDict` may not exist by that exact name — Phase 2a's `SafetensorsWriter` used it. If the API moved, adapt.
- Tied embeddings: when source omits `lm_head.weight`, the loader skips it; embed_tokens populates the tied parameter automatically through aliasing.

**Commit count target:** ~14 commits (7 TDD pairs).

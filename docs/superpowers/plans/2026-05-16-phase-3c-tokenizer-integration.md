# Phase 3c: Tokenizer + Real-Model Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `Microsoft.ML.Tokenizers` so that tokenizer files from a HuggingFace snapshot are loaded transparently. Add a test-model download script that populates the standard HF cache with SmolLM2-135M (the 135M LLaMA-architecture model selected during brainstorming). End-to-end integration tests load the real model + tokenizer via our `HuggingFaceLoader`, encode a prompt, run a forward pass, and verify the output is sensible. Tests skip cleanly when the model is not cached (CI, offline development).

**Architecture:**

```
LLMCompressorSharp.Transformers/
├── Tokenization/
│   ├── LlamaTokenizer.cs              ← wraps Microsoft.ML.Tokenizers
│   └── TokenizerLoadException.cs
├── Loading/
│   └── HuggingFaceLoader.cs           ← MODIFY: optionally return tokenizer too
└── (LoadedLlamaModel record updated to include optional Tokenizer)
```

Scripts:
```
scripts/
└── download-test-models.ps1            ← populates HF cache from huggingface.co
```

**Tech Stack:** `Microsoft.ML.Tokenizers` 1.0.2 (already referenced from Phase 0; supports loading HF tokenizer.json), `System.Net.Http.HttpClient` for the download script (via PowerShell `Invoke-WebRequest`).

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (`Tokenization/LlamaTokenizer`)  
**Reference convention:** `docs/llmcompressorsharp/cache-conventions.md`

---

## File Structure

```
src/LLMCompressorSharp.Transformers/
├── Tokenization/                              ← NEW
│   ├── LlamaTokenizer.cs
│   └── TokenizerLoadException.cs
└── Loading/
    ├── LoadedLlamaModel.cs                    ← MODIFY: add optional Tokenizer field
    └── HuggingFaceLoader.cs                   ← MODIFY: add LoadWithTokenizer overload

tests/LLMCompressorSharp.Tests/
└── Transformers/
    ├── Tokenization/                          ← NEW
    │   └── LlamaTokenizerTests.cs             ← skip-if-not-cached integration tests
    └── Integration/
        └── SmolLM2EndToEndTests.cs            ← skip-if-not-cached full forward pass

scripts/                                      ← NEW
└── download-test-models.ps1
```

**Responsibility per file:**
- `LlamaTokenizer` — Constructor takes a snapshot directory; locates `tokenizer.json` and constructs a `Microsoft.ML.Tokenizers.Tokenizer` via the appropriate factory. Exposes `Encode(string text) → int[]`, `Decode(IEnumerable<int>) → string`, and `Tokenizer` (underlying object, for advanced callers).
- `TokenizerLoadException` — distinct from `HuggingFaceLoadException` so tokenizer failures are diagnosable separately.
- `LoadedLlamaModel` — record extended with optional `LlamaTokenizer? Tokenizer { get; init; }` field. Backward compatible (default null).
- `HuggingFaceLoader.LoadWithTokenizer(repoId, revision, env)` — new overload that loads model + tokenizer in one call. Existing `Load(...)` retained.
- `download-test-models.ps1` — downloads SmolLM2-135M (config.json, tokenizer.json, special_tokens_map.json, tokenizer_config.json, model.safetensors) into the HF cache snapshot for `HuggingFaceTB/SmolLM2-135M`. Idempotent (skips files already present).

---

## Prerequisites & Conventions

- Phase 3b is merged. Tag `v0.3.1-alpha`. 179 tests passing on `main`.
- `Microsoft.ML.Tokenizers` 1.0.2 is in `Directory.Packages.props` and referenced by `LLMCompressorSharp.Transformers.csproj`. Already proven loadable.
- Branch off `main` as `feature/3c-tokenizer-integration`.
- Integration tests check `File.Exists` on tokenizer.json / config.json before running; skip with a clear xunit `Skip = "..."` message when missing.

---

### Task 1: Branch + baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
Expected: clean tree, HEAD `884acc8`, tags through `v0.3.1-alpha`.

- [ ] **Step 2:** `git checkout -b feature/3c-tokenizer-integration`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
Expected: 179 passing.

No commit.

---

### Task 2: Investigate `Microsoft.ML.Tokenizers` 1.0.2 API

**Files:** (no file changes — research only)

- [ ] **Step 1: Inspect the API surface for HF tokenizer.json loading**

```powershell
$xmlPath = Get-ChildItem -Path "$env:USERPROFILE\.nuget\packages\microsoft.ml.tokenizers" -Filter "*.xml" -Recurse | Where-Object { $_.FullName -match 'lib\\net' } | Select-Object -First 1 -ExpandProperty FullName
if ($xmlPath) {
    Get-Content $xmlPath | Select-String -Pattern "(BpeTokenizer|TiktokenTokenizer|LlamaTokenizer|FromFile|FromStream|class Tokenizer)" -Context 0,2 | Select-Object -First 60
}
```

Identify the factory method that loads a HuggingFace `tokenizer.json`. Common patterns in 1.0+:
- `BpeTokenizer.Create(Stream vocabAndMerges, ...)` — for HF-style BPE JSON
- `TiktokenTokenizer.CreateForModel(...)` — for OpenAI tiktoken
- `LlamaTokenizer.Create(string sentencePiecePath, ...)` — for SentencePiece (NOT what SmolLM2 uses)

For SmolLM2 (which uses HF's BPE in `tokenizer.json`), the right factory is typically `BpeTokenizer.Create(Stream tokenizerJson)`. Verify the exact static method name.

- [ ] **Step 2: Document the findings**

Note the chosen factory method and parameters; the implementation in Task 4 references it directly.

No commit.

---

### Task 3: Add `TokenizerLoadException`

**Files:**
- Create: `src/LLMCompressorSharp.Transformers/Tokenization/TokenizerLoadException.cs`

- [ ] **Step 1: Write the exception**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers.Tokenization;

/// <summary>
/// Thrown when a tokenizer cannot be loaded — missing files, malformed JSON, or unsupported tokenizer type.
/// </summary>
public sealed class TokenizerLoadException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="TokenizerLoadException"/> class.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    public TokenizerLoadException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="TokenizerLoadException"/> class with an inner cause.</summary>
    /// <param name="message">Diagnostic describing what failed.</param>
    /// <param name="innerException">Underlying exception.</param>
    public TokenizerLoadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Tokenization/TokenizerLoadException.cs
git commit -m "feat(tokenization): add TokenizerLoadException"
```

---

### Task 4: TDD `LlamaTokenizer`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Transformers/Tokenization/LlamaTokenizerTests.cs`
- Create: `src/LLMCompressorSharp.Transformers/Tokenization/LlamaTokenizer.cs`

- [ ] **Step 1: Write the tests**

The tests split into two groups:
1. **Constructor validation** — runs without any cached model (always exercised).
2. **Round-trip encode/decode** — skip if SmolLM2-135M tokenizer files not present in HF cache.

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Tokenization;
using Xunit;

namespace LLMCompressorSharp.Tests.Transformers.Tokenization;

/// <summary>
/// Tests for <see cref="LlamaTokenizer"/>.
/// </summary>
public class LlamaTokenizerTests
{
    private const string SmolLm2RepoId = "HuggingFaceTB/SmolLM2-135M";
    private const string SmolLm2Revision = "main";

    private static string? GetSnapshotDir()
    {
        try
        {
            var cacheRoot = HuggingFaceCache.ResolveCacheRoot(SystemEnvironment.Instance);
            var dir = HuggingFaceCache.GetSnapshotPath(cacheRoot, SmolLm2RepoId, SmolLm2Revision);
            return Directory.Exists(dir) && File.Exists(Path.Combine(dir, "tokenizer.json")) ? dir : null;
        }
        catch
        {
            return null;
        }
    }

    [Fact]
    public void Constructor_NullSnapshotDir_Throws()
    {
        var act = () => new LlamaTokenizer(null!);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_NonExistentDir_Throws()
    {
        var act = () => new LlamaTokenizer("/nonexistent/path/no-tokenizer-here");
        act.Should().Throw<TokenizerLoadException>().WithMessage("*not found*");
    }

    [Fact]
    public void Constructor_DirWithoutTokenizerJson_Throws()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"llmc-tokenizer-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var act = () => new LlamaTokenizer(tempDir);
            act.Should().Throw<TokenizerLoadException>().WithMessage("*tokenizer.json*");
        }
        finally
        {
            Directory.Delete(tempDir);
        }
    }

    [Fact]
    public void Encode_SmolLM2_RoundTripsKnownText()
    {
        var snapshotDir = GetSnapshotDir();
        Skip.IfNot(snapshotDir is not null,
            $"SmolLM2-135M tokenizer not cached. Run scripts/download-test-models.ps1 to enable this test.");

        using var tokenizer = new LlamaTokenizer(snapshotDir!);
        const string text = "Hello, world!";

        var ids = tokenizer.Encode(text);
        ids.Should().NotBeEmpty();

        var decoded = tokenizer.Decode(ids);
        // BPE tokenizers may add or drop leading whitespace; assert the meaningful content survives.
        decoded.Trim().Should().Contain("Hello");
        decoded.Should().Contain("world");
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmptyOrSpecialTokens()
    {
        var snapshotDir = GetSnapshotDir();
        Skip.IfNot(snapshotDir is not null,
            $"SmolLM2-135M tokenizer not cached.");

        using var tokenizer = new LlamaTokenizer(snapshotDir!);
        var ids = tokenizer.Encode(string.Empty);
        ids.Should().NotBeNull();
    }
}
```

> **xunit v3 Skip API:** The static `Skip.IfNot(condition, reason)` is available in xunit.v3 1.0+. If the build complains, use `Skip.If(!condition, reason)` — the negated form. If neither helper exists, the implementer should add `using Xunit.Sdk;` or use the `[Fact(Skip = "...")]` attribute pattern (requires compile-time `string`, so the path check has to move to a `Theory` with `MemberData`).

- [ ] **Step 2: Verify CS0246 build failure**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaTokenizerTests"`
Expected: BUILD FAILS.

- [ ] **Step 3: Commit failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/Transformers/Tokenization/LlamaTokenizerTests.cs
git commit -m "test(tokenization): add failing LlamaTokenizer tests"
```

- [ ] **Step 4: Implement `LlamaTokenizer`**

Create `src/LLMCompressorSharp.Transformers/Tokenization/LlamaTokenizer.cs`. The exact API depends on Task 2 findings; below is a likely shape:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using Microsoft.ML.Tokenizers;

namespace LLMCompressorSharp.Transformers.Tokenization;

/// <summary>
/// Wraps a HuggingFace tokenizer.json from a snapshot directory into a LLaMA-family tokenizer.
/// </summary>
public sealed class LlamaTokenizer : IDisposable
{
    private const string TokenizerJsonName = "tokenizer.json";

    private readonly Tokenizer _tokenizer;

    /// <summary>Initializes a new instance of the <see cref="LlamaTokenizer"/> class.</summary>
    /// <param name="snapshotDir">Directory containing <c>tokenizer.json</c>.</param>
    /// <exception cref="TokenizerLoadException">If the directory or tokenizer file is missing or malformed.</exception>
    public LlamaTokenizer(string snapshotDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDir);
        if (!Directory.Exists(snapshotDir))
        {
            throw new TokenizerLoadException($"Snapshot directory not found: '{snapshotDir}'.");
        }

        var tokenizerPath = Path.Combine(snapshotDir, TokenizerJsonName);
        if (!File.Exists(tokenizerPath))
        {
            throw new TokenizerLoadException(
                $"tokenizer.json not found in snapshot '{snapshotDir}'. "
                + "Download the model with `huggingface-cli download <repo>` first.");
        }

        try
        {
            using var stream = File.OpenRead(tokenizerPath);
            _tokenizer = BpeTokenizer.Create(stream);
        }
        catch (Exception ex) when (ex is not TokenizerLoadException)
        {
            throw new TokenizerLoadException($"Failed to load tokenizer from '{tokenizerPath}'.", ex);
        }
    }

    /// <summary>Gets the underlying <see cref="Microsoft.ML.Tokenizers.Tokenizer"/> for advanced use.</summary>
    public Tokenizer UnderlyingTokenizer => _tokenizer;

    /// <summary>Encodes <paramref name="text"/> to a list of token ids.</summary>
    /// <param name="text">The input text.</param>
    /// <returns>The token ids.</returns>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = _tokenizer.EncodeToIds(text);
        return result.ToArray();
    }

    /// <summary>Decodes a sequence of token ids back to text.</summary>
    /// <param name="ids">The token ids.</param>
    /// <returns>The decoded text.</returns>
    public string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return _tokenizer.Decode(ids) ?? string.Empty;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // Microsoft.ML.Tokenizers.Tokenizer is not IDisposable; nothing to clean up here.
        // The Dispose method exists so LoadedLlamaModel can dispose tokenizer + model together symmetrically.
    }
}
```

> **API adaptations:** if `BpeTokenizer.Create(Stream)` doesn't exist, try `BpeTokenizer.Create(string vocabPath, string mergesPath)` and skip the test (no merges file from tokenizer.json alone). Or check for `Tokenizer.CreateFromFile(string)` — newer ML.Tokenizers versions added this. The exact factory name varies; **adapt without changing the public API of `LlamaTokenizer`**.

- [ ] **Step 5: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaTokenizerTests"`
Expected: 3 always-on tests pass; 2 skip-conditional tests either skip or pass.

If `BpeTokenizer.Create` from JSON throws an unsupported-format error, the implementer should:
1. Examine the SmolLM2 `tokenizer.json` structure (the `model` field typically reports `"type": "BPE"`)
2. Try the `Tokenizer` static factories: `Tokenizer.CreateFromFile`, `TiktokenTokenizer.Create*`, etc.
3. If absolutely no method in ML.Tokenizers 1.0.2 can read HF tokenizer.json directly, **report BLOCKED** with the specific error. We can pivot to a tokenizer.model SentencePiece file (LLaMA-1/2 style) but SmolLM2 doesn't ship one.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Tokenization/LlamaTokenizer.cs
git commit -m "feat(tokenization): implement LlamaTokenizer wrapper"
```

---

### Task 5: Add `download-test-models.ps1` script

**Files:**
- Create: `scripts/download-test-models.ps1`

- [ ] **Step 1: Write the script**

```powershell
# Copyright (c) James Burton. Licensed under the Apache-2.0 license.

<#
.SYNOPSIS
    Downloads test models (SmolLM2-135M) into the shared HuggingFace cache.

.DESCRIPTION
    Populates the standard HuggingFace cache layout (see docs/llmcompressorsharp/cache-conventions.md)
    so Phase 3c integration tests can verify end-to-end loading. Idempotent — skips files already on disk.

    Default cache location resolution:
      1. $env:HF_HUB_CACHE
      2. $env:HF_HOME/hub
      3. $env:XDG_CACHE_HOME/huggingface/hub
      4. ~/.cache/huggingface/hub

.EXAMPLE
    pwsh scripts/download-test-models.ps1

.EXAMPLE
    pwsh scripts/download-test-models.ps1 -Repo "HuggingFaceTB/SmolLM2-135M-Instruct"
#>

[CmdletBinding()]
param(
    [string] $Repo = "HuggingFaceTB/SmolLM2-135M",
    [string] $Revision = "main",
    [string[]] $Files = @(
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors"
    )
)

$ErrorActionPreference = 'Stop'

function Resolve-HfHubCache {
    if ($env:HF_HUB_CACHE) { return $env:HF_HUB_CACHE }
    if ($env:HF_HOME) { return (Join-Path $env:HF_HOME "hub") }
    if ($env:XDG_CACHE_HOME) { return (Join-Path $env:XDG_CACHE_HOME "huggingface/hub") }
    return (Join-Path $HOME ".cache/huggingface/hub")
}

function Get-RepoFolder {
    param([string] $repoId)
    $parts = $repoId -split '/'
    if ($parts.Count -ne 2) {
        throw "Repo id must be in 'org/repo' form. Got: '$repoId'."
    }
    return "models--$($parts[0])--$($parts[1])"
}

$cacheRoot = Resolve-HfHubCache
$snapshotDir = Join-Path $cacheRoot (Join-Path (Get-RepoFolder $Repo) (Join-Path "snapshots" $Revision))

New-Item -ItemType Directory -Force -Path $snapshotDir | Out-Null

Write-Host "Cache root:   $cacheRoot"
Write-Host "Snapshot dir: $snapshotDir"
Write-Host "Repo:         $Repo @ $Revision"
Write-Host ""

$baseUrl = "https://huggingface.co/$Repo/resolve/$Revision"
$downloaded = 0
$skipped = 0
$missing = @()

foreach ($file in $Files) {
    $targetPath = Join-Path $snapshotDir $file
    if (Test-Path $targetPath) {
        $size = (Get-Item $targetPath).Length
        Write-Host "  [skip]  $file ($size bytes already present)"
        $skipped++
        continue
    }

    $url = "$baseUrl/$file"
    Write-Host "  [get]   $file ..." -NoNewline
    try {
        Invoke-WebRequest -Uri $url -OutFile $targetPath -UseBasicParsing
        $size = (Get-Item $targetPath).Length
        Write-Host " done ($size bytes)"
        $downloaded++
    }
    catch {
        Write-Host " FAILED ($($_.Exception.Message))"
        Remove-Item -Force -ErrorAction SilentlyContinue $targetPath
        $missing += $file
    }
}

Write-Host ""
Write-Host "Summary: $downloaded downloaded, $skipped already present, $($missing.Count) missing."
if ($missing.Count -gt 0) {
    Write-Warning "Missing files: $($missing -join ', '). Some integration tests will skip."
    exit 2
}

Write-Host "OK — model is ready for Phase 3c integration tests."
exit 0
```

- [ ] **Step 2: Commit**

```powershell
git add scripts/download-test-models.ps1
git commit -m "scripts: add download-test-models.ps1 for SmolLM2-135M into HF cache"
```

- [ ] **Step 3 (optional): Attempt the download in this environment**

Run: `pwsh scripts/download-test-models.ps1`

If it succeeds, the integration tests in Task 4 (and Task 7) will exercise real model loading. If it fails (no network), tests skip — no commit needed and the script is still useful for developers.

If the download succeeds, **proceed to Step 4**. If it fails, skip ahead — the script lands as docs.

- [ ] **Step 4 (if download succeeded): Verify integration tests run**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~LlamaTokenizerTests"`
Expected: 5 tests pass (3 always-on + 2 skip-conditional, now running).

---

### Task 6: Extend `LoadedLlamaModel` with optional tokenizer

**Files:**
- Modify: `src/LLMCompressorSharp.Transformers/Loading/LoadedLlamaModel.cs`

- [ ] **Step 1: Replace the record definition**

Edit `LoadedLlamaModel.cs` to add a `Tokenizer` member with `init`-only assignment. Use `record { }` body so the `LlamaTokenizer` import doesn't break consumers who didn't construct a tokenizer.

Replace the entire file contents with:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;
using LLMCompressorSharp.Transformers.Tokenization;

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
    string SnapshotPath)
{
    /// <summary>Gets the optional tokenizer loaded alongside the model.</summary>
    public LlamaTokenizer? Tokenizer { get; init; }
}
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
Expected: 179 + recent additions still pass.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/LoadedLlamaModel.cs
git commit -m "feat(transformers): add optional Tokenizer to LoadedLlamaModel"
```

---

### Task 7: Extend `HuggingFaceLoader` with `LoadWithTokenizer`

**Files:**
- Modify: `src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoader.cs`

- [ ] **Step 1: Add the overload**

Use the Edit tool to add this method to `HuggingFaceLoader` (after the existing `Load` method, before the closing brace of the class):

```csharp
    /// <summary>
    /// Loads a model from the HuggingFace cache, including its tokenizer.
    /// </summary>
    /// <param name="repoId">Repo id in <c>org/repo</c> form.</param>
    /// <param name="revision">Revision or branch; default <c>main</c>.</param>
    /// <param name="environment">Environment provider for cache resolution; defaults to <see cref="SystemEnvironment.Instance"/>.</param>
    /// <returns>The loaded model + config + snapshot path + tokenizer.</returns>
    /// <exception cref="HuggingFaceLoadException">If the model or its snapshot cannot be loaded.</exception>
    /// <exception cref="TokenizerLoadException">If the tokenizer cannot be loaded.</exception>
    public static LoadedLlamaModel LoadWithTokenizer(
        string repoId,
        string? revision = "main",
        IEnvironment? environment = null)
    {
        var loaded = Load(repoId, revision, environment);
        try
        {
            var tokenizer = new LlamaTokenizer(loaded.SnapshotPath);
            return loaded with { Tokenizer = tokenizer };
        }
        catch
        {
            loaded.Model.Dispose();
            throw;
        }
    }
```

Add `using LLMCompressorSharp.Transformers.Tokenization;` to the file's imports.

- [ ] **Step 2: Build**

Run: `dotnet build src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/Loading/HuggingFaceLoader.cs
git commit -m "feat(transformers): add HuggingFaceLoader.LoadWithTokenizer overload"
```

---

### Task 8: End-to-end integration test (skip-if-not-cached)

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Integration/SmolLM2EndToEndTests.cs`

- [ ] **Step 1: Write the test**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using LLMCompressorSharp.Transformers.Loading;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Integration;

/// <summary>
/// End-to-end integration tests against the real SmolLM2-135M model. Skipped when the model
/// is not present in the local HuggingFace cache.
/// </summary>
public class SmolLM2EndToEndTests
{
    private const string RepoId = "HuggingFaceTB/SmolLM2-135M";
    private const string Revision = "main";

    private static bool IsCached()
    {
        try
        {
            var cacheRoot = HuggingFaceCache.ResolveCacheRoot(SystemEnvironment.Instance);
            var dir = HuggingFaceCache.GetSnapshotPath(cacheRoot, RepoId, Revision);
            return Directory.Exists(dir)
                && File.Exists(Path.Combine(dir, "config.json"))
                && File.Exists(Path.Combine(dir, "tokenizer.json"))
                && (File.Exists(Path.Combine(dir, "model.safetensors"))
                    || File.Exists(Path.Combine(dir, "model.safetensors.index.json")));
        }
        catch
        {
            return false;
        }
    }

    [Fact]
    public void LoadSmolLM2_FromHfCache_ProducesUsableModel()
    {
        Skip.IfNot(IsCached(),
            "SmolLM2-135M not present in HF cache. Run scripts/download-test-models.ps1 first.");

        var loaded = HuggingFaceLoader.LoadWithTokenizer(RepoId, Revision);
        using (loaded.Model)
        {
            loaded.Config.HiddenSize.Should().Be(576);
            loaded.Config.NumHiddenLayers.Should().Be(30);
            loaded.Config.VocabSize.Should().Be(49152);
            loaded.Tokenizer.Should().NotBeNull();

            // Tokenize a short prompt
            var ids = loaded.Tokenizer!.Encode("The capital of France is");
            ids.Should().NotBeEmpty();

            // Run a forward pass
            using var inputIds = tensor(ids.Select(i => (long)i).ToArray()).reshape(1, ids.Length);
            using var logits = loaded.Model.forward(inputIds);

            logits.shape.Should().Equal(new long[] { 1, ids.Length, loaded.Config.VocabSize });

            // The argmax of the last-token logits should be a valid token id (not NaN/Inf).
            using var lastLogits = logits.select(1, ids.Length - 1).squeeze(0);
            using var argmax = lastLogits.argmax();
            var nextTokenId = argmax.cpu().item<long>();
            nextTokenId.Should().BeGreaterOrEqualTo(0L);
            nextTokenId.Should().BeLessThan(loaded.Config.VocabSize);
        }
    }

    [Fact]
    public void LoadSmolLM2_ForwardPassProducesFinitelogits()
    {
        Skip.IfNot(IsCached(),
            "SmolLM2-135M not present in HF cache.");

        var loaded = HuggingFaceLoader.LoadWithTokenizer(RepoId, Revision);
        using (loaded.Model)
        {
            var ids = loaded.Tokenizer!.Encode("Hello");
            using var inputIds = tensor(ids.Select(i => (long)i).ToArray()).reshape(1, ids.Length);
            using var logits = loaded.Model.forward(inputIds);

            // All values must be finite (no NaN/Inf).
            using var isFinite = logits.isfinite();
            using var allFinite = isFinite.all();
            allFinite.cpu().item<bool>().Should().BeTrue();
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SmolLM2EndToEndTests"`
Expected: 2 tests skip if model not cached, OR pass if cached.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Integration/SmolLM2EndToEndTests.cs
git commit -m "test(integration): add SmolLM2-135M end-to-end forward pass (skip if not cached)"
```

---

### Task 9: Full verification

**Files:** (no file changes — verification only)

- [ ] **Step 1: Clean build + test**

```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected:
- 0 errors, 0 warnings
- ~184 tests passing or skipped (179 + 3 always-on tokenizer + 2 skip-conditional tokenizer + 2 skip-conditional e2e). The non-conditional count is **184**; skipped count depends on whether the model is downloaded.

- [ ] **Step 2: No commit — verification only**

---

### Task 10: STOP — controller handles merge + tag

Tag will be `v0.3.2-alpha`.

---

## Self-Review Notes

**Spec coverage:**
- `LlamaTokenizer` wrapper → Task 4
- Real-model integration → Task 8
- Cache convention compliance (shared HF cache, never duplicate) → Tasks 4, 5, 8 all use `HuggingFaceCache`

**Out of scope:**
- KV cache for autoregressive generation
- Numerical-parity comparison vs Python `transformers` reference logits — that requires saving a reference vector from a Python script, which is a separate effort
- Sentencepiece tokenizers (LLaMA-1/2 style) — Phase 4 or beyond
- Phase 6's `--cache-dir`, `--external-cache`, `--offline` flags

**Type consistency:**
- `LlamaTokenizer` is `IDisposable` even though `Microsoft.ML.Tokenizers.Tokenizer` isn't — defensive future-proofing.
- `LoadedLlamaModel.Tokenizer` is `init`-only nullable — backward compatible.
- `HuggingFaceLoader.LoadWithTokenizer` returns the same `LoadedLlamaModel` record with `Tokenizer` set.

**Known risks:**
- **`Microsoft.ML.Tokenizers` factory method names** vary across 0.x and 1.0+ — Task 2 explicitly does API research first.
- **`Skip.IfNot` xunit v3 API** — may need adjustment based on the actual xunit v3 1.0.0 surface.
- **Network downloads** in Task 5 may fail in restricted environments. The script's exit code 2 + warning makes this discoverable; integration tests skip cleanly when files are missing.
- **`tensor.isfinite()`** in Task 8 may not exist in TorchSharp 0.107.0. Fallback: `(logits == logits).all().item<bool>()` (NaN != NaN trick).

**Commit count target:** ~11 commits (3 always-on + script + record + loader + integration test).

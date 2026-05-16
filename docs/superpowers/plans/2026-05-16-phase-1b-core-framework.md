# Phase 1b: Core Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the `LLMCompressorSharp.Core` library with the modifier lifecycle framework (`IModifier`, `ModifierBase`), the orchestrator (`CompressionSession`, `CompressionState`), and the recipe system (`Recipe` POCO, YamlDotNet parser, validator with ordering rules). All TDD'd. Phase 2 algorithm modifiers will plug into this framework without further infrastructure work.

**Architecture:**

```
CompressionSession (orchestrator)
   ├── owns CompressionState (named weights, sample data, hardware info)
   ├── runs registered IModifier instances through their lifecycle:
   │     Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize
   └── ModifierBase provides Targets/Ignore filtering + helper accessors

Recipe (YAML schema)
   ├── stages: ordered list of Stage
   ├── each Stage: ordered list of ModifierConfig (polymorphic via "type" discriminator)
   └── RecipeBuilder turns configs → concrete IModifier via ModifierRegistry

RecipeValidator
   ├── ordering rule: AWQ must precede a Quantization-stage modifier
   ├── (extensible — Phase 2 algorithms add their own rules)
```

The framework intentionally avoids any TorchSharp `Module` coupling at this layer. `CompressionState` holds named weights as `IReadOnlyDictionary<string, Tensor>` — Phase 3 will add a helper that extracts the dictionary from a `Module`.

**Tech Stack:** .NET 10 / C# 14, TorchSharp 0.107.0 (only `Tensor` type touched at this layer), YamlDotNet 16.3.0, xUnit v3, FluentAssertions.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md` §2.2 (Core responsibilities)

---

## File Structure

```
src/LLMCompressorSharp.Core/
├── LLMCompressorSharp.Core.csproj                ← already exists
├── (delete) PlaceholderMarker.cs                 ← removed in Task 14
├── Modifiers/
│   ├── IModifier.cs                              ← lifecycle interface
│   ├── ModifierBase.cs                           ← abstract base with Targets/Ignore
│   ├── ModifierLifecycleException.cs             ← thrown on protocol violations
│   └── TargetMatcher.cs                          ← static name-pattern matching helpers
├── Compression/
│   ├── CompressionSession.cs                     ← orchestrator
│   ├── CompressionState.cs                       ← named weights, sample data, RNG seed
│   └── SessionStatus.cs                          ← enum (NotStarted, Running, Completed, Failed)
└── Recipes/
    ├── Recipe.cs                                 ← root POCO
    ├── Stage.cs
    ├── ModifierConfig.cs                         ← abstract base; type discriminator
    ├── ModifierRegistry.cs                       ← Register<TConfig, TModifier> registry
    ├── RecipeParser.cs                           ← YamlDotNet wrapper
    ├── RecipeValidator.cs                        ← ordering rules
    └── RecipeBuilder.cs                          ← config → IModifier instance

tests/LLMCompressorSharp.Tests/
├── Modifiers/
│   ├── ModifierBaseTests.cs
│   └── TargetMatcherTests.cs
├── Compression/
│   └── CompressionSessionTests.cs
└── Recipes/
    ├── RecipeParserTests.cs
    ├── RecipeValidatorTests.cs
    └── RecipeBuilderTests.cs
```

**Responsibility per file:**
- `IModifier.cs` — the 5-method lifecycle contract.
- `ModifierBase.cs` — abstract base that enforces lifecycle ordering (e.g. throws if OnBatch fires before OnStart), holds `Targets` / `Ignore` patterns, and exposes a `GetTargetedWeights(CompressionState)` helper.
- `ModifierLifecycleException.cs` — thrown when a modifier is misused (e.g., `OnBatch` before `Initialize`).
- `TargetMatcher.cs` — static name-pattern matching: a name matches if any `targets` pattern matches **and** no `ignore` pattern matches. Patterns support wildcards (`*` → glob).
- `CompressionState.cs` — mutable bag containing named weights, optional calibration sample tensor for the current batch, current batch index, and a deterministic RNG seed.
- `CompressionSession.cs` — runs registered modifiers through `Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize`. Handles exceptions by transitioning to `SessionStatus.Failed`.
- `Recipe.cs` / `Stage.cs` — POCOs.
- `ModifierConfig.cs` — abstract; YAML `type` discriminator picks the concrete subclass; configs are immutable records.
- `ModifierRegistry.cs` — maps a string type name to a (`ModifierConfig` subtype, factory delegate). Phase 2 modifiers register here.
- `RecipeParser.cs` — `Parse(string yamlText)` returns a `Recipe`. Uses YamlDotNet's `IObjectFactory` for polymorphic dispatch.
- `RecipeValidator.cs` — enforces ordering invariants. Initially: AWQ before any Quantization stage modifier. Phase 2 extends.
- `RecipeBuilder.cs` — `Build(Recipe) → IReadOnlyList<IModifier>` by consulting `ModifierRegistry`.

**Out of scope:**
- Concrete modifier configs (e.g., `GptqConfig`) and modifier classes (`GptqModifier`) — Phase 2.
- Output writers — Phase 5.
- TorchSharp `Module` integration — Phase 3.

---

## Prerequisites & Conventions

- Phase 1a is merged; tag `v0.1.0-alpha` is in place. 45 tests passing on `main`.
- The `Core` project references `TorchExtensions`. `TorchSharp` and `TorchSharp.PyBridge` are referenced as packages.
- Branch off `main` as `feature/1b-core-framework`.
- Same StyleCop patterns as Phase 1a: blank line before inline `//`, "Initializes a new instance of" for constructors, multi-line array initializers, properties before methods.
- `.editorconfig` snake_case suppressions apply to `TorchExtensions/**.cs` and `Transformers/Architectures/**.cs`, but **NOT** to `Core/**.cs` — Core uses standard .NET PascalCase throughout.

---

### Task 1: Create the working branch and verify baseline

- [ ] **Step 1:** `git status --short && git log --oneline -1 && git tag | findstr alpha`
Expected: clean working tree; HEAD is `a8e3fed` (Phase 1a docs commit); `v0.1.0-alpha` tag present.

- [ ] **Step 2:** `git checkout -b feature/1b-core-framework`

- [ ] **Step 3:** `dotnet test LLMCompressorSharp.slnx --configuration Release --filter "Category!=Gpu"`
Expected: 45 tests pass.

No commit for this task.

---

### Task 2: Add `SessionStatus` enum and `CompressionEvent` enum

**Files:**
- Create: `src/LLMCompressorSharp.Core/Compression/SessionStatus.cs`
- Create: `src/LLMCompressorSharp.Core/Compression/CompressionEvent.cs`

- [ ] **Step 1: Create `SessionStatus.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Lifecycle status of a <see cref="CompressionSession"/>.
/// </summary>
public enum SessionStatus
{
    /// <summary>The session has been constructed but <c>Run</c> has not yet been called.</summary>
    NotStarted,

    /// <summary>The session is currently running modifiers.</summary>
    Running,

    /// <summary>The session ran to completion and all modifiers finalized successfully.</summary>
    Completed,

    /// <summary>A modifier threw or the session was aborted before completion.</summary>
    Failed,
}
```

- [ ] **Step 2: Create `CompressionEvent.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Lifecycle events that <see cref="IModifier"/> implementations can observe.
/// </summary>
public enum CompressionEvent
{
    /// <summary>Fired once at the start of a session, before any batches.</summary>
    Initialize,

    /// <summary>Fired once after Initialize, before the first batch.</summary>
    Start,

    /// <summary>Fired for each calibration batch.</summary>
    Batch,

    /// <summary>Fired once after the last batch.</summary>
    End,

    /// <summary>Fired once at the very end, after End, for cleanup.</summary>
    Finalize,
}
```

> `CompressionEvent` is not used by `IModifier` in Phase 1b (we expose separate methods per event for type safety). It exists for future use by `Recipe`-level hooks and for Phase 2 algorithms that share a single `on_event` dispatcher.

- [ ] **Step 3: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Compression/
git commit -m "feat(core): add SessionStatus and CompressionEvent enums"
```

---

### Task 3: Add `CompressionState`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Compression/CompressionState.cs`

- [ ] **Step 1: Write `CompressionState`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Mutable state shared with every <see cref="IModifier"/> across a compression session.
/// </summary>
/// <remarks>
/// Phase 1b exposes the named-weights dictionary and per-batch sample. Phase 3 will add a
/// helper to extract <see cref="NamedWeights"/> from a TorchSharp <c>Module</c>; until then,
/// callers populate it directly (typically from a safetensors load).
/// </remarks>
public sealed class CompressionState
{
    private readonly Dictionary<string, Tensor> _namedWeights;

    /// <summary>Initializes a new instance of the <see cref="CompressionState"/> class.</summary>
    /// <param name="namedWeights">The initial named-weights dictionary. The session takes ownership.</param>
    /// <param name="rngSeed">Deterministic seed for any RNG-using modifier.</param>
    public CompressionState(IReadOnlyDictionary<string, Tensor> namedWeights, long rngSeed = 0L)
    {
        ArgumentNullException.ThrowIfNull(namedWeights);
        _namedWeights = new Dictionary<string, Tensor>(namedWeights);
        RngSeed = rngSeed;
    }

    /// <summary>Gets the named-weights dictionary. Modifiers may replace values in place.</summary>
    public IDictionary<string, Tensor> NamedWeights => _namedWeights;

    /// <summary>Gets or sets the current calibration batch tensor, or <see langword="null"/> outside a batch.</summary>
    public Tensor? CurrentBatch { get; set; }

    /// <summary>Gets or sets the zero-based index of the current calibration batch.</summary>
    public int CurrentBatchIndex { get; set; }

    /// <summary>Gets the deterministic RNG seed assigned to this session.</summary>
    public long RngSeed { get; }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Compression/CompressionState.cs
git commit -m "feat(core): add CompressionState"
```

---

### Task 4: Add `IModifier` interface

**Files:**
- Create: `src/LLMCompressorSharp.Core/Modifiers/IModifier.cs`

- [ ] **Step 1: Write the interface**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// A compression algorithm exposed as a lifecycle of hooks.
/// </summary>
/// <remarks>
/// Implementations should be stateless across sessions: any state accumulated during one run
/// must be reset in <see cref="Initialize"/> and disposed in <see cref="Finalize"/>.
///
/// <para>Lifecycle: <c>Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize</c>.</para>
/// </remarks>
public interface IModifier
{
    /// <summary>A short identifier used in logs and recipe round-trips (e.g. "GPTQ", "SmoothQuant").</summary>
    string Name { get; }

    /// <summary>Called once before any batches. Allocate observers, hooks, and accumulators.</summary>
    /// <param name="state">The session state.</param>
    void Initialize(CompressionState state);

    /// <summary>Called once after Initialize and before the first batch.</summary>
    /// <param name="state">The session state.</param>
    void OnStart(CompressionState state);

    /// <summary>Called for each calibration batch.</summary>
    /// <param name="state">The session state. <see cref="CompressionState.CurrentBatch"/> is non-null.</param>
    void OnBatch(CompressionState state);

    /// <summary>Called once after the last batch. The modifier should produce its compressed result here.</summary>
    /// <param name="state">The session state.</param>
    void OnEnd(CompressionState state);

    /// <summary>Called once after OnEnd, regardless of success. Dispose any tensors and detach hooks.</summary>
    /// <param name="state">The session state.</param>
    void Finalize(CompressionState state);
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Modifiers/IModifier.cs
git commit -m "feat(core): add IModifier lifecycle interface"
```

---

### Task 5: Add `ModifierLifecycleException` and `TargetMatcher`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Modifiers/ModifierLifecycleException.cs`
- Create: `src/LLMCompressorSharp.Core/Modifiers/TargetMatcher.cs`

- [ ] **Step 1: Create `ModifierLifecycleException.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Thrown when a modifier's lifecycle methods are invoked in an invalid order
/// (e.g. <c>OnBatch</c> before <c>Initialize</c>).
/// </summary>
public sealed class ModifierLifecycleException : InvalidOperationException
{
    /// <summary>Initializes a new instance of the <see cref="ModifierLifecycleException"/> class.</summary>
    /// <param name="message">Description of the protocol violation.</param>
    public ModifierLifecycleException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ModifierLifecycleException"/> class with an inner exception.</summary>
    /// <param name="message">Description of the protocol violation.</param>
    /// <param name="innerException">The underlying cause.</param>
    public ModifierLifecycleException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
```

- [ ] **Step 2: Create `TargetMatcher.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.RegularExpressions;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Name-pattern matching for selecting which named weights a modifier targets.
/// </summary>
/// <remarks>
/// A name matches if <paramref name="targets"/> is empty (match-all) or any target pattern matches,
/// and no <paramref name="ignore"/> pattern matches. Patterns are glob-style — <c>*</c> matches any
/// sequence of characters, including path separators (dots, slashes).
/// </remarks>
public static class TargetMatcher
{
    /// <summary>Returns true if <paramref name="name"/> matches the target/ignore patterns.</summary>
    /// <param name="name">The fully qualified weight or module name (e.g. <c>model.layers.0.q_proj.weight</c>).</param>
    /// <param name="targets">Target patterns; empty means "match any".</param>
    /// <param name="ignore">Ignore patterns; any match excludes the name.</param>
    /// <returns>True if the name should be processed.</returns>
    public static bool Matches(string name, IReadOnlyList<string> targets, IReadOnlyList<string> ignore)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentNullException.ThrowIfNull(ignore);

        foreach (var pattern in ignore)
        {
            if (MatchesPattern(name, pattern))
            {
                return false;
            }
        }

        if (targets.Count == 0)
        {
            return true;
        }

        foreach (var pattern in targets)
        {
            if (MatchesPattern(name, pattern))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Filters <paramref name="names"/> through <see cref="Matches"/>.</summary>
    /// <param name="names">Candidate names.</param>
    /// <param name="targets">Target patterns.</param>
    /// <param name="ignore">Ignore patterns.</param>
    /// <returns>The subset that matches.</returns>
    public static IEnumerable<string> Filter(
        IEnumerable<string> names,
        IReadOnlyList<string> targets,
        IReadOnlyList<string> ignore)
    {
        ArgumentNullException.ThrowIfNull(names);
        foreach (var n in names)
        {
            if (Matches(n, targets, ignore))
            {
                yield return n;
            }
        }
    }

    private static bool MatchesPattern(string name, string pattern)
    {
        if (string.IsNullOrEmpty(pattern))
        {
            return false;
        }

        if (!pattern.Contains('*'))
        {
            return string.Equals(name, pattern, StringComparison.Ordinal);
        }

        var regex = "^" + Regex.Escape(pattern).Replace("\\*", ".*") + "$";
        return Regex.IsMatch(name, regex);
    }
}
```

- [ ] **Step 3: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Modifiers/ModifierLifecycleException.cs src/LLMCompressorSharp.Core/Modifiers/TargetMatcher.cs
git commit -m "feat(core): add ModifierLifecycleException and TargetMatcher"
```

---

### Task 6: TDD `TargetMatcher`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Modifiers/TargetMatcherTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Modifiers;
using Xunit;

namespace LLMCompressorSharp.Tests.Modifiers;

/// <summary>
/// Tests for <see cref="TargetMatcher"/> — glob-style name pattern filtering.
/// </summary>
public class TargetMatcherTests
{
    [Fact]
    public void Matches_EmptyTargets_MatchesAnything()
    {
        TargetMatcher.Matches("anything", Array.Empty<string>(), Array.Empty<string>())
            .Should().BeTrue();
    }

    [Fact]
    public void Matches_ExactTarget_MatchesOnlyExactName()
    {
        var targets = new[] { "Linear" };
        TargetMatcher.Matches("Linear", targets, Array.Empty<string>()).Should().BeTrue();
        TargetMatcher.Matches("Linear2", targets, Array.Empty<string>()).Should().BeFalse();
    }

    [Theory]
    [InlineData("model.layers.0.q_proj", "model.layers.*.q_proj", true)]
    [InlineData("model.layers.15.q_proj", "model.layers.*.q_proj", true)]
    [InlineData("model.layers.0.k_proj", "model.layers.*.q_proj", false)]
    [InlineData("model.lm_head", "model.lm_head", true)]
    public void Matches_GlobStarMatchesAnySegment(string name, string pattern, bool expected)
    {
        TargetMatcher.Matches(name, new[] { pattern }, Array.Empty<string>()).Should().Be(expected);
    }

    [Fact]
    public void Matches_IgnoreWins_OverTargetMatch()
    {
        var targets = new[] { "*" };
        var ignore = new[] { "lm_head" };
        TargetMatcher.Matches("lm_head", targets, ignore).Should().BeFalse();
        TargetMatcher.Matches("other", targets, ignore).Should().BeTrue();
    }

    [Fact]
    public void Filter_ReturnsOnlyMatchingNames()
    {
        var names = new[] { "layer.0", "layer.1", "lm_head", "embeddings" };
        var matched = TargetMatcher.Filter(names, new[] { "layer.*" }, Array.Empty<string>()).ToArray();
        matched.Should().Equal("layer.0", "layer.1");
    }

    [Fact]
    public void Matches_NullName_Throws()
    {
        var act = () => TargetMatcher.Matches(null!, Array.Empty<string>(), Array.Empty<string>());
        act.Should().Throw<ArgumentNullException>();
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~TargetMatcherTests"`
Expected: 9 tests pass (5 [Fact] + 4 [Theory] rows).

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Modifiers/TargetMatcherTests.cs
git commit -m "test(core): add TargetMatcher pattern matching tests"
```

---

### Task 7: Add `ModifierBase` abstract class with lifecycle enforcement

**Files:**
- Create: `src/LLMCompressorSharp.Core/Modifiers/ModifierBase.cs`

- [ ] **Step 1: Write `ModifierBase`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Abstract base class for modifiers that adds lifecycle ordering enforcement,
/// target/ignore filtering, and a helper to enumerate targeted weights.
/// </summary>
/// <remarks>
/// Subclasses override the <c>On...</c> hooks. <see cref="Initialize"/> through
/// <see cref="Finalize"/> are sealed: they enforce the protocol before calling
/// the hook.
/// </remarks>
public abstract class ModifierBase : IModifier
{
    private bool _initialized;
    private bool _started;
    private bool _ended;
    private bool _finalized;

    /// <summary>Initializes a new instance of the <see cref="ModifierBase"/> class.</summary>
    /// <param name="name">A short identifier (e.g. "GPTQ").</param>
    /// <param name="targets">Name patterns of weights to target; <see langword="null"/> = all.</param>
    /// <param name="ignore">Name patterns to exclude; <see langword="null"/> = none.</param>
    protected ModifierBase(string name, IReadOnlyList<string>? targets = null, IReadOnlyList<string>? ignore = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Targets = targets ?? Array.Empty<string>();
        Ignore = ignore ?? Array.Empty<string>();
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>Gets the configured target patterns.</summary>
    public IReadOnlyList<string> Targets { get; }

    /// <summary>Gets the configured ignore patterns.</summary>
    public IReadOnlyList<string> Ignore { get; }

    /// <inheritdoc />
    public void Initialize(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was initialized twice.");
        }

        _initialized = true;
        OnInitialize(state);
    }

    /// <inheritdoc />
    public void OnStart(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureInitialized();
        if (_started)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was started twice.");
        }

        _started = true;
        OnStartCore(state);
    }

    /// <inheritdoc />
    public void OnBatch(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureStarted();
        if (_ended)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' received a batch after OnEnd.");
        }

        OnBatchCore(state);
    }

    /// <inheritdoc />
    public void OnEnd(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        EnsureStarted();
        if (_ended)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was ended twice.");
        }

        _ended = true;
        OnEndCore(state);
    }

    /// <inheritdoc />
    public void Finalize(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was finalized before being initialized.");
        }

        if (_finalized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was finalized twice.");
        }

        _finalized = true;
        OnFinalizeCore(state);
    }

    /// <summary>Implements the modifier's <see cref="IModifier.Initialize"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected abstract void OnInitialize(CompressionState state);

    /// <summary>Implements the modifier's <see cref="IModifier.OnStart"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnStartCore(CompressionState state)
    {
    }

    /// <summary>Implements the modifier's <see cref="IModifier.OnBatch"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnBatchCore(CompressionState state)
    {
    }

    /// <summary>Implements the modifier's <see cref="IModifier.OnEnd"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected abstract void OnEndCore(CompressionState state);

    /// <summary>Implements the modifier's <see cref="IModifier.Finalize"/> hook.</summary>
    /// <param name="state">The session state.</param>
    protected virtual void OnFinalizeCore(CompressionState state)
    {
    }

    /// <summary>Enumerates the names from <paramref name="state"/> that match <see cref="Targets"/> minus <see cref="Ignore"/>.</summary>
    /// <param name="state">The session state.</param>
    /// <returns>The targeted names in dictionary-enumeration order.</returns>
    protected IEnumerable<string> GetTargetedNames(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        return TargetMatcher.Filter(state.NamedWeights.Keys, Targets, Ignore);
    }

    /// <summary>Enumerates targeted (name, weight) pairs from the state.</summary>
    /// <param name="state">The session state.</param>
    /// <returns>The targeted (name, tensor) pairs.</returns>
    protected IEnumerable<KeyValuePair<string, Tensor>> GetTargetedWeights(CompressionState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        foreach (var name in GetTargetedNames(state))
        {
            yield return new KeyValuePair<string, Tensor>(name, state.NamedWeights[name]);
        }
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was used before Initialize.");
        }
    }

    private void EnsureStarted()
    {
        EnsureInitialized();
        if (!_started)
        {
            throw new ModifierLifecycleException($"Modifier '{Name}' was used before OnStart.");
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Modifiers/ModifierBase.cs
git commit -m "feat(core): add ModifierBase with lifecycle enforcement"
```

---

### Task 8: TDD `ModifierBase`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Modifiers/ModifierBaseTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Modifiers;

/// <summary>
/// Tests for <see cref="ModifierBase"/> — lifecycle enforcement and target filtering.
/// </summary>
public class ModifierBaseTests
{
    [Fact]
    public void Lifecycle_HappyPath_AllHooksFireInOrder()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();

        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnBatch(state);
        modifier.OnBatch(state);
        modifier.OnEnd(state);
        modifier.Finalize(state);

        modifier.EventLog.Should().Equal("Initialize", "Start", "Batch", "Batch", "End", "Finalize");
    }

    [Fact]
    public void OnBatch_BeforeInitialize_ThrowsLifecycleException()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        var act = () => modifier.OnBatch(state);
        act.Should().Throw<ModifierLifecycleException>()
            .WithMessage("*before Initialize*");
    }

    [Fact]
    public void Initialize_Twice_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        var act = () => modifier.Initialize(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*initialized twice*");
    }

    [Fact]
    public void OnEnd_BeforeOnStart_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        var act = () => modifier.OnEnd(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*before OnStart*");
    }

    [Fact]
    public void OnBatch_AfterOnEnd_Throws()
    {
        var modifier = new RecordingModifier();
        var state = CreateState();
        modifier.Initialize(state);
        modifier.OnStart(state);
        modifier.OnEnd(state);
        var act = () => modifier.OnBatch(state);
        act.Should().Throw<ModifierLifecycleException>().WithMessage("*after OnEnd*");
    }

    [Fact]
    public void GetTargetedNames_FiltersByTargetsAndIgnore()
    {
        var modifier = new RecordingModifier(targets: new[] { "layer.*" }, ignore: new[] { "layer.5" });
        var state = new CompressionState(new Dictionary<string, Tensor>
        {
            ["layer.0"] = zeros(1),
            ["layer.5"] = zeros(1),
            ["layer.7"] = zeros(1),
            ["other"] = zeros(1),
        });

        var matched = modifier.GetTargetedNamesPublic(state).ToArray();
        matched.Should().BeEquivalentTo("layer.0", "layer.7");
    }

    [Fact]
    public void Name_Required_Throws_OnEmpty()
    {
        var act = () => new RecordingModifier(name: " ");
        act.Should().Throw<ArgumentException>();
    }

    private static CompressionState CreateState()
    {
        return new CompressionState(new Dictionary<string, Tensor>());
    }

    private sealed class RecordingModifier : ModifierBase
    {
        public RecordingModifier(string name = "Recording", IReadOnlyList<string>? targets = null, IReadOnlyList<string>? ignore = null)
            : base(name, targets, ignore)
        {
        }

        public List<string> EventLog { get; } = new();

        public IEnumerable<string> GetTargetedNamesPublic(CompressionState state) => GetTargetedNames(state);

        protected override void OnInitialize(CompressionState state) => EventLog.Add("Initialize");

        protected override void OnStartCore(CompressionState state) => EventLog.Add("Start");

        protected override void OnBatchCore(CompressionState state) => EventLog.Add("Batch");

        protected override void OnEndCore(CompressionState state) => EventLog.Add("End");

        protected override void OnFinalizeCore(CompressionState state) => EventLog.Add("Finalize");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~ModifierBaseTests"`
Expected: 7 tests pass.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Modifiers/ModifierBaseTests.cs
git commit -m "test(core): add ModifierBase lifecycle tests"
```

---

### Task 9: Add `CompressionSession` orchestrator

**Files:**
- Create: `src/LLMCompressorSharp.Core/Compression/CompressionSession.cs`

- [ ] **Step 1: Write `CompressionSession`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Orchestrates a list of <see cref="IModifier"/> instances through the
/// <c>Initialize → OnStart → (OnBatch × N) → OnEnd → Finalize</c> lifecycle.
/// </summary>
/// <remarks>
/// Modifiers run in declaration order at every lifecycle phase. If any modifier throws,
/// the session marks itself <see cref="SessionStatus.Failed"/> and still attempts to run
/// <see cref="IModifier.Finalize"/> on previously-initialized modifiers.
/// </remarks>
public sealed class CompressionSession
{
    private readonly IReadOnlyList<IModifier> _modifiers;
    private readonly List<IModifier> _initialized = new();

    /// <summary>Initializes a new instance of the <see cref="CompressionSession"/> class.</summary>
    /// <param name="modifiers">The modifiers to run, in execution order.</param>
    public CompressionSession(IReadOnlyList<IModifier> modifiers)
    {
        ArgumentNullException.ThrowIfNull(modifiers);
        _modifiers = modifiers;
    }

    /// <summary>Gets the current session status.</summary>
    public SessionStatus Status { get; private set; } = SessionStatus.NotStarted;

    /// <summary>Gets the exception that caused failure, if any.</summary>
    public Exception? Failure { get; private set; }

    /// <summary>
    /// Runs the full lifecycle: initialize all modifiers, start, run calibration batches,
    /// end, and finalize.
    /// </summary>
    /// <param name="state">The session state. <see cref="CompressionState.CurrentBatch"/> is set by the session per batch.</param>
    /// <param name="batches">The calibration batches to iterate. May be empty.</param>
    /// <returns>The final <see cref="Status"/>.</returns>
    public SessionStatus Run(CompressionState state, IEnumerable<Tensor> batches)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(batches);

        if (Status != SessionStatus.NotStarted)
        {
            throw new InvalidOperationException("CompressionSession.Run may only be called once.");
        }

        Status = SessionStatus.Running;

        try
        {
            foreach (var m in _modifiers)
            {
                m.Initialize(state);
                _initialized.Add(m);
            }

            foreach (var m in _modifiers)
            {
                m.OnStart(state);
            }

            var index = 0;
            foreach (var batch in batches)
            {
                state.CurrentBatch = batch;
                state.CurrentBatchIndex = index;
                foreach (var m in _modifiers)
                {
                    m.OnBatch(state);
                }

                index++;
            }

            state.CurrentBatch = null;

            foreach (var m in _modifiers)
            {
                m.OnEnd(state);
            }

            Status = SessionStatus.Completed;
        }
        catch (Exception ex)
        {
            Status = SessionStatus.Failed;
            Failure = ex;
        }
        finally
        {
            FinalizeAll(state);
        }

        return Status;
    }

    private void FinalizeAll(CompressionState state)
    {
        foreach (var m in _initialized)
        {
            try
            {
                m.Finalize(state);
            }
            catch (Exception finalizeEx) when (Status == SessionStatus.Failed)
            {
                // Swallow secondary finalize errors when we're already in Failed state — keep the original Failure.
                _ = finalizeEx;
            }
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Compression/CompressionSession.cs
git commit -m "feat(core): add CompressionSession orchestrator"
```

---

### Task 10: TDD `CompressionSession`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Compression/CompressionSessionTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Compression;

/// <summary>
/// Tests for <see cref="CompressionSession"/> orchestrator.
/// </summary>
public class CompressionSessionTests
{
    [Fact]
    public void Run_WithNoBatches_FiresInitStartEndFinalizeOnAllModifiers()
    {
        var a = new RecordingModifier("A");
        var b = new RecordingModifier("B");
        var session = new CompressionSession(new IModifier[] { a, b });

        var status = session.Run(NewState(), Enumerable.Empty<Tensor>());

        status.Should().Be(SessionStatus.Completed);
        a.EventLog.Should().Equal("Initialize", "Start", "End", "Finalize");
        b.EventLog.Should().Equal("Initialize", "Start", "End", "Finalize");
    }

    [Fact]
    public void Run_WithBatches_FiresOnBatchPerCalibrationSample()
    {
        var m = new RecordingModifier("A");
        var session = new CompressionSession(new IModifier[] { m });
        var batches = new[] { ones(2, 2), ones(2, 2), ones(2, 2) };

        session.Run(NewState(), batches);

        m.EventLog.Should().Equal(
            "Initialize", "Start", "Batch", "Batch", "Batch", "End", "Finalize");
        m.BatchIndices.Should().Equal(0, 1, 2);
    }

    [Fact]
    public void Run_ModifiersRunInDeclarationOrderPerPhase()
    {
        var a = new RecordingModifier("A");
        var b = new RecordingModifier("B");
        var observed = new List<string>();
        a.OnPhase = phase => observed.Add($"A.{phase}");
        b.OnPhase = phase => observed.Add($"B.{phase}");

        var session = new CompressionSession(new IModifier[] { a, b });
        session.Run(NewState(), new[] { ones(1) });

        observed.Should().Equal(
            "A.Initialize", "B.Initialize",
            "A.Start", "B.Start",
            "A.Batch", "B.Batch",
            "A.End", "B.End",
            "A.Finalize", "B.Finalize");
    }

    [Fact]
    public void Run_WhenModifierThrowsDuringBatch_TransitionsToFailedAndStillFinalizes()
    {
        var failing = new RecordingModifier("Failing")
        {
            ThrowOnBatch = new InvalidOperationException("boom"),
        };
        var session = new CompressionSession(new IModifier[] { failing });

        var status = session.Run(NewState(), new[] { ones(1) });

        status.Should().Be(SessionStatus.Failed);
        session.Failure.Should().BeOfType<InvalidOperationException>().Which.Message.Should().Be("boom");
        failing.EventLog.Should().Contain("Finalize");
    }

    [Fact]
    public void Run_CallingTwice_Throws()
    {
        var session = new CompressionSession(Array.Empty<IModifier>());
        session.Run(NewState(), Enumerable.Empty<Tensor>());
        var act = () => session.Run(NewState(), Enumerable.Empty<Tensor>());
        act.Should().Throw<InvalidOperationException>().WithMessage("*only be called once*");
    }

    [Fact]
    public void Run_SetsCurrentBatchAndClearsItAfterEnd()
    {
        var sawBatch = false;
        Tensor? batchAfterEnd = null;
        var m = new RecordingModifier("A")
        {
            BatchObserver = s => { sawBatch = s.CurrentBatch is not null; },
            EndObserver = s => { batchAfterEnd = s.CurrentBatch; },
        };

        var session = new CompressionSession(new IModifier[] { m });
        session.Run(NewState(), new[] { ones(1) });

        sawBatch.Should().BeTrue();
        batchAfterEnd.Should().BeNull();
    }

    private static CompressionState NewState()
    {
        return new CompressionState(new Dictionary<string, Tensor>());
    }

    private sealed class RecordingModifier : ModifierBase
    {
        public RecordingModifier(string name)
            : base(name)
        {
        }

        public List<string> EventLog { get; } = new();

        public List<int> BatchIndices { get; } = new();

        public Action<string>? OnPhase { get; set; }

        public Action<CompressionState>? BatchObserver { get; set; }

        public Action<CompressionState>? EndObserver { get; set; }

        public Exception? ThrowOnBatch { get; set; }

        protected override void OnInitialize(CompressionState state)
        {
            EventLog.Add("Initialize");
            OnPhase?.Invoke("Initialize");
        }

        protected override void OnStartCore(CompressionState state)
        {
            EventLog.Add("Start");
            OnPhase?.Invoke("Start");
        }

        protected override void OnBatchCore(CompressionState state)
        {
            EventLog.Add("Batch");
            BatchIndices.Add(state.CurrentBatchIndex);
            BatchObserver?.Invoke(state);
            OnPhase?.Invoke("Batch");
            if (ThrowOnBatch is not null)
            {
                throw ThrowOnBatch;
            }
        }

        protected override void OnEndCore(CompressionState state)
        {
            EventLog.Add("End");
            EndObserver?.Invoke(state);
            OnPhase?.Invoke("End");
        }

        protected override void OnFinalizeCore(CompressionState state)
        {
            EventLog.Add("Finalize");
            OnPhase?.Invoke("Finalize");
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~CompressionSessionTests"`
Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Compression/CompressionSessionTests.cs
git commit -m "test(core): add CompressionSession orchestrator tests"
```

---

### Task 11: Add `ModifierConfig`, `Stage`, `Recipe`, `ModifierRegistry`

**Files:**
- Create: `src/LLMCompressorSharp.Core/Recipes/ModifierConfig.cs`
- Create: `src/LLMCompressorSharp.Core/Recipes/Stage.cs`
- Create: `src/LLMCompressorSharp.Core/Recipes/Recipe.cs`
- Create: `src/LLMCompressorSharp.Core/Recipes/ModifierRegistry.cs`

- [ ] **Step 1: Create `ModifierConfig.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Polymorphic base for serialized modifier configurations.
/// </summary>
/// <remarks>
/// Subclasses are registered with <see cref="ModifierRegistry"/> under a short type name
/// used as the YAML <c>type:</c> discriminator.
/// </remarks>
public abstract class ModifierConfig
{
    /// <summary>Gets the short type identifier (matches the YAML <c>type:</c> field).</summary>
    public abstract string Type { get; }

    /// <summary>Gets or sets the target name patterns. Null is treated as "match all".</summary>
    public IReadOnlyList<string>? Targets { get; set; }

    /// <summary>Gets or sets the ignore name patterns. Null is treated as "ignore none".</summary>
    public IReadOnlyList<string>? Ignore { get; set; }
}
```

- [ ] **Step 2: Create `Stage.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// A named stage within a <see cref="Recipe"/>. Stages execute in declaration order and
/// each stage's modifiers run sequentially within it.
/// </summary>
public sealed class Stage
{
    /// <summary>Gets or sets the stage's human-readable name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Gets or sets the modifier configurations belonging to this stage.</summary>
    public IList<ModifierConfig> Modifiers { get; set; } = new List<ModifierConfig>();
}
```

- [ ] **Step 3: Create `Recipe.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// A compression recipe: an ordered list of stages, each containing modifier configurations.
/// </summary>
public sealed class Recipe
{
    /// <summary>Gets or sets the stages of this recipe.</summary>
    public IList<Stage> Stages { get; set; } = new List<Stage>();

    /// <summary>Enumerates every modifier config across all stages in order.</summary>
    /// <returns>The (stage-index, stage-name, modifier-config) triples.</returns>
    public IEnumerable<(int StageIndex, string StageName, ModifierConfig Config)> EnumerateModifiers()
    {
        for (var i = 0; i < Stages.Count; i++)
        {
            var stage = Stages[i];
            foreach (var m in stage.Modifiers)
            {
                yield return (i, stage.Name, m);
            }
        }
    }
}
```

- [ ] **Step 4: Create `ModifierRegistry.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Registry mapping YAML <c>type:</c> discriminators to (config subtype, factory) pairs.
/// </summary>
/// <remarks>
/// Phase 2 modifier libraries register their configs here at startup, e.g.:
/// <code>ModifierRegistry.Register&lt;GptqConfig, GptqModifier&gt;("GPTQ", c =&gt; new GptqModifier(c));</code>
///
/// Phase 1b ships an empty registry plus the registration API and lookup methods. Tests use
/// their own registrations via the public <see cref="Register"/> method.
/// </remarks>
public static class ModifierRegistry
{
    private static readonly Dictionary<string, Registration> Registry = new(StringComparer.Ordinal);

    /// <summary>Registers a config + factory pair under a YAML type discriminator.</summary>
    /// <typeparam name="TConfig">The concrete <see cref="ModifierConfig"/> subtype.</typeparam>
    /// <param name="typeName">The YAML <c>type:</c> string (case-sensitive).</param>
    /// <param name="factory">Builds an <see cref="IModifier"/> from a config instance.</param>
    public static void Register<TConfig>(string typeName, Func<TConfig, IModifier> factory)
        where TConfig : ModifierConfig
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        ArgumentNullException.ThrowIfNull(factory);
        Registry[typeName] = new Registration(typeof(TConfig), c => factory((TConfig)c));
    }

    /// <summary>Removes a previously-registered type.</summary>
    /// <param name="typeName">The discriminator to unregister.</param>
    /// <returns>True if removed.</returns>
    public static bool Unregister(string typeName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        return Registry.Remove(typeName);
    }

    /// <summary>Clears all registrations. Intended for test isolation only.</summary>
    public static void Clear() => Registry.Clear();

    /// <summary>Resolves a registration by type discriminator.</summary>
    /// <param name="typeName">The discriminator string.</param>
    /// <returns>The registration if found, else <see langword="null"/>.</returns>
    public static Registration? Resolve(string typeName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        return Registry.TryGetValue(typeName, out var reg) ? reg : null;
    }

    /// <summary>A type registration.</summary>
    /// <param name="ConfigType">The concrete config CLR type.</param>
    /// <param name="Factory">A factory that turns a config into a modifier.</param>
    public sealed record Registration(Type ConfigType, Func<ModifierConfig, IModifier> Factory);
}
```

- [ ] **Step 5: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Recipes/
git commit -m "feat(core): add Recipe POCOs and ModifierRegistry"
```

---

### Task 12: Add `RecipeParser` with YamlDotNet polymorphic dispatch

**Files:**
- Create: `src/LLMCompressorSharp.Core/Recipes/RecipeParser.cs`

- [ ] **Step 1: Write `RecipeParser.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Globalization;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Parses YAML recipe documents into <see cref="Recipe"/> instances, dispatching modifier
/// configs through <see cref="ModifierRegistry"/> via the YAML <c>type:</c> discriminator.
/// </summary>
public static class RecipeParser
{
    /// <summary>Parses a YAML string into a <see cref="Recipe"/>.</summary>
    /// <param name="yaml">The recipe YAML text.</param>
    /// <returns>The parsed recipe.</returns>
    /// <exception cref="RecipeParseException">If the YAML is malformed or references an unregistered modifier type.</exception>
    public static Recipe Parse(string yaml)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(yaml);

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .WithTypeConverter(new ModifierConfigConverter())
            .Build();

        try
        {
            return deserializer.Deserialize<Recipe>(yaml) ?? new Recipe();
        }
        catch (YamlException ex)
        {
            throw new RecipeParseException("Failed to parse recipe YAML.", ex);
        }
    }

    private sealed class ModifierConfigConverter : IYamlTypeConverter
    {
        public bool Accepts(Type type) => type == typeof(ModifierConfig);

        public object? ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
        {
            // Read the mapping into a dictionary so we can pick the type discriminator.
            parser.Consume<MappingStart>();
            var fields = new Dictionary<string, object?>();
            string? typeName = null;
            while (!parser.TryConsume<MappingEnd>(out _))
            {
                var key = parser.Consume<Scalar>().Value;
                if (parser.Current is Scalar s)
                {
                    parser.MoveNext();
                    if (key == "type")
                    {
                        typeName = s.Value;
                    }
                    else
                    {
                        fields[key] = s.Value;
                    }
                }
                else if (parser.Current is SequenceStart)
                {
                    parser.Consume<SequenceStart>();
                    var items = new List<string>();
                    while (!parser.TryConsume<SequenceEnd>(out _))
                    {
                        items.Add(parser.Consume<Scalar>().Value);
                    }

                    fields[key] = items;
                }
                else
                {
                    // Skip unknown nested structures gracefully.
                    parser.SkipThisAndNestedEvents();
                }
            }

            if (string.IsNullOrEmpty(typeName))
            {
                throw new RecipeParseException("Each modifier must specify a `type:` field.");
            }

            var registration = ModifierRegistry.Resolve(typeName)
                ?? throw new RecipeParseException(
                    $"Modifier type '{typeName}' is not registered. Did you call ModifierRegistry.Register<...>?");

            var config = (ModifierConfig)Activator.CreateInstance(registration.ConfigType)!;
            PopulateConfig(config, fields);
            return config;
        }

        public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
        {
            throw new NotSupportedException("Recipe serialization is not implemented in Phase 1b.");
        }

        private static void PopulateConfig(ModifierConfig config, Dictionary<string, object?> fields)
        {
            foreach (var (key, raw) in fields)
            {
                var prop = config.GetType().GetProperty(ToPascalCase(key));
                if (prop is null || !prop.CanWrite)
                {
                    continue;
                }

                if (raw is List<string> list)
                {
                    prop.SetValue(config, list);
                }
                else if (raw is string s)
                {
                    var converted = ConvertScalar(s, prop.PropertyType);
                    prop.SetValue(config, converted);
                }
            }
        }

        private static string ToPascalCase(string snake)
        {
            if (string.IsNullOrEmpty(snake))
            {
                return snake;
            }

            var sb = new System.Text.StringBuilder(snake.Length);
            var nextUpper = true;
            foreach (var c in snake)
            {
                if (c == '_')
                {
                    nextUpper = true;
                    continue;
                }

                sb.Append(nextUpper ? char.ToUpper(c, CultureInfo.InvariantCulture) : c);
                nextUpper = false;
            }

            return sb.ToString();
        }

        private static object? ConvertScalar(string s, Type targetType)
        {
            if (targetType == typeof(string))
            {
                return s;
            }

            if (targetType == typeof(int) || targetType == typeof(int?))
            {
                return int.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(long) || targetType == typeof(long?))
            {
                return long.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(float) || targetType == typeof(float?))
            {
                return float.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(double) || targetType == typeof(double?))
            {
                return double.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(bool) || targetType == typeof(bool?))
            {
                return bool.Parse(s);
            }

            // Last resort: leave as string.
            return s;
        }
    }
}

/// <summary>Thrown when <see cref="RecipeParser"/> cannot parse a recipe YAML document.</summary>
public sealed class RecipeParseException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="RecipeParseException"/> class.</summary>
    /// <param name="message">Diagnostic message.</param>
    public RecipeParseException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="RecipeParseException"/> class with an inner exception.</summary>
    /// <param name="message">Diagnostic message.</param>
    /// <param name="innerException">The underlying YAML error.</param>
    public RecipeParseException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
```

> **YamlDotNet API note:** The exact `IYamlTypeConverter.ReadYaml` signature can vary between versions. YamlDotNet 16.x uses `(IParser parser, Type type, ObjectDeserializer rootDeserializer)` and `WriteYaml(IEmitter, object?, Type, ObjectSerializer)`. If the build fails because the signature differs, look at YamlDotNet's actual interface definition (via IntelliSense or NuGet package XML docs) and adapt. The core logic — read the mapping, find `type:`, dispatch via `ModifierRegistry`, populate the config — stays the same regardless.

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Recipes/RecipeParser.cs
git commit -m "feat(core): add RecipeParser with polymorphic modifier dispatch"
```

---

### Task 13: TDD `RecipeParser`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Recipes/RecipeParserTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeParser"/> YAML deserialization with polymorphic modifier dispatch.
/// </summary>
public class RecipeParserTests : IDisposable
{
    public RecipeParserTests()
    {
        // Reset the global registry to a known state before each test
        ModifierRegistry.Clear();
        ModifierRegistry.Register<TestModifierConfig>("TestModifier", c => new TestModifier(c));
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
    }

    [Fact]
    public void Parse_SingleStage_SingleModifier_ProducesRecipe()
    {
        var yaml = @"
stages:
  - name: quant_stage
    modifiers:
      - type: TestModifier
        scale: 0.5
";

        var recipe = RecipeParser.Parse(yaml);

        recipe.Stages.Should().HaveCount(1);
        recipe.Stages[0].Name.Should().Be("quant_stage");
        recipe.Stages[0].Modifiers.Should().HaveCount(1);
        var cfg = recipe.Stages[0].Modifiers[0].Should().BeOfType<TestModifierConfig>().Subject;
        cfg.Scale.Should().Be(0.5f);
        cfg.Type.Should().Be("TestModifier");
    }

    [Fact]
    public void Parse_ParsesTargetsAndIgnoreLists()
    {
        var yaml = @"
stages:
  - name: stage
    modifiers:
      - type: TestModifier
        targets: [layer.0, layer.1]
        ignore: [lm_head]
        scale: 1.0
";
        var recipe = RecipeParser.Parse(yaml);
        var cfg = (TestModifierConfig)recipe.Stages[0].Modifiers[0];
        cfg.Targets.Should().Equal("layer.0", "layer.1");
        cfg.Ignore.Should().Equal("lm_head");
    }

    [Fact]
    public void Parse_MultipleStages_ExecutionOrderPreserved()
    {
        var yaml = @"
stages:
  - name: first
    modifiers:
      - type: TestModifier
        scale: 0.1
  - name: second
    modifiers:
      - type: TestModifier
        scale: 0.2
";
        var recipe = RecipeParser.Parse(yaml);
        recipe.Stages.Should().HaveCount(2);
        recipe.Stages[0].Name.Should().Be("first");
        recipe.Stages[1].Name.Should().Be("second");
    }

    [Fact]
    public void Parse_UnregisteredModifierType_Throws()
    {
        var yaml = @"
stages:
  - name: s
    modifiers:
      - type: NotRegistered
        scale: 1.0
";
        var act = () => RecipeParser.Parse(yaml);
        act.Should().Throw<RecipeParseException>().WithMessage("*NotRegistered*not registered*");
    }

    [Fact]
    public void Parse_MissingTypeField_Throws()
    {
        var yaml = @"
stages:
  - name: s
    modifiers:
      - scale: 1.0
";
        var act = () => RecipeParser.Parse(yaml);
        act.Should().Throw<RecipeParseException>().WithMessage("*type*");
    }

    [Fact]
    public void Parse_EnumerateModifiers_YieldsStageIndexAndName()
    {
        var yaml = @"
stages:
  - name: a
    modifiers:
      - type: TestModifier
        scale: 0.1
      - type: TestModifier
        scale: 0.2
  - name: b
    modifiers:
      - type: TestModifier
        scale: 0.3
";
        var recipe = RecipeParser.Parse(yaml);
        var enumerated = recipe.EnumerateModifiers().ToArray();
        enumerated.Should().HaveCount(3);
        enumerated[0].StageIndex.Should().Be(0);
        enumerated[0].StageName.Should().Be("a");
        enumerated[2].StageIndex.Should().Be(1);
        enumerated[2].StageName.Should().Be("b");
    }

    private sealed class TestModifierConfig : ModifierConfig
    {
        public override string Type => "TestModifier";

        public float Scale { get; set; }
    }

    private sealed class TestModifier : ModifierBase
    {
        public TestModifier(TestModifierConfig config)
            : base("TestModifier", config.Targets, config.Ignore)
        {
            Scale = config.Scale;
        }

        public float Scale { get; }

        protected override void OnInitialize(CompressionState state)
        {
        }

        protected override void OnEndCore(CompressionState state)
        {
        }
    }
}
```

> **Note:** `ModifierRegistry` is process-global. The tests use `Clear()` in setup/teardown to maintain isolation. If multiple test classes run in parallel and both clear the registry, they will interfere. For Phase 1b the tests in this file are the only registry consumers — Phase 2 will need to revisit isolation.

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~RecipeParserTests"`
Expected: 6 tests pass.

If a YamlDotNet API mismatch breaks the build, fix `RecipeParser.cs` to match the actual interface signature, then re-run. Do **not** weaken the test assertions — they capture the required behaviour.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Recipes/RecipeParserTests.cs
git commit -m "test(core): add RecipeParser YAML deserialization tests"
```

---

### Task 14: Add `RecipeValidator` with extensible ordering rules

**Files:**
- Create: `src/LLMCompressorSharp.Core/Recipes/RecipeValidator.cs`

- [ ] **Step 1: Write `RecipeValidator.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Validates that a <see cref="Recipe"/> obeys cross-modifier ordering constraints.
/// </summary>
/// <remarks>
/// Rules are registered via <see cref="AddRule(IRecipeRule)"/>. Phase 1b ships a single
/// rule — <see cref="MustPrecedeRule"/> — which requires a modifier of one type to
/// precede a modifier of another type within the recipe. Phase 2 adds rules for AWQ
/// preceding QuantizationModifier and similar.
/// </remarks>
public sealed class RecipeValidator
{
    private readonly List<IRecipeRule> _rules = new();

    /// <summary>Adds a rule. Rules execute in registration order.</summary>
    /// <param name="rule">The rule to add.</param>
    public void AddRule(IRecipeRule rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        _rules.Add(rule);
    }

    /// <summary>Validates <paramref name="recipe"/> against all registered rules.</summary>
    /// <param name="recipe">The recipe to validate.</param>
    /// <returns>The set of violations; empty when the recipe is valid.</returns>
    public IReadOnlyList<string> Validate(Recipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        var violations = new List<string>();
        foreach (var rule in _rules)
        {
            rule.Check(recipe, violations);
        }

        return violations;
    }
}

/// <summary>A validation rule executed by <see cref="RecipeValidator"/>.</summary>
public interface IRecipeRule
{
    /// <summary>Inspects <paramref name="recipe"/> and appends violation messages to <paramref name="violations"/>.</summary>
    /// <param name="recipe">The recipe under test.</param>
    /// <param name="violations">Mutable list to which violation messages are appended.</param>
    void Check(Recipe recipe, IList<string> violations);
}

/// <summary>
/// Rule: every modifier of type <see cref="SuccessorType"/> must be preceded by at least one
/// modifier of type <see cref="PredecessorType"/> within the same recipe.
/// </summary>
public sealed class MustPrecedeRule : IRecipeRule
{
    /// <summary>Initializes a new instance of the <see cref="MustPrecedeRule"/> class.</summary>
    /// <param name="predecessorType">The required earlier modifier's type discriminator.</param>
    /// <param name="successorType">The later modifier's type discriminator.</param>
    public MustPrecedeRule(string predecessorType, string successorType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(predecessorType);
        ArgumentException.ThrowIfNullOrWhiteSpace(successorType);
        PredecessorType = predecessorType;
        SuccessorType = successorType;
    }

    /// <summary>Gets the required predecessor type.</summary>
    public string PredecessorType { get; }

    /// <summary>Gets the constrained successor type.</summary>
    public string SuccessorType { get; }

    /// <inheritdoc />
    public void Check(Recipe recipe, IList<string> violations)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentNullException.ThrowIfNull(violations);

        var seenPredecessor = false;
        foreach (var (_, _, config) in recipe.EnumerateModifiers())
        {
            if (config.Type == PredecessorType)
            {
                seenPredecessor = true;
            }
            else if (config.Type == SuccessorType && !seenPredecessor)
            {
                violations.Add(
                    $"Modifier '{SuccessorType}' must be preceded by a '{PredecessorType}' modifier.");
            }
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Recipes/RecipeValidator.cs
git commit -m "feat(core): add RecipeValidator with MustPrecedeRule"
```

---

### Task 15: TDD `RecipeValidator`

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/Recipes/RecipeValidatorTests.cs`

- [ ] **Step 1: Write the tests**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Recipes;
using Xunit;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeValidator"/> and <see cref="MustPrecedeRule"/>.
/// </summary>
public class RecipeValidatorTests
{
    [Fact]
    public void Validate_EmptyRecipe_ReportsNoViolations()
    {
        var validator = new RecipeValidator();
        var recipe = new Recipe();

        var violations = validator.Validate(recipe);
        violations.Should().BeEmpty();
    }

    [Fact]
    public void MustPrecedeRule_PredecessorBeforeSuccessor_NoViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("AWQ", 1), ("Quantization", 1));

        validator.Validate(recipe).Should().BeEmpty();
    }

    [Fact]
    public void MustPrecedeRule_SuccessorWithoutPredecessor_ReportsViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 1));

        validator.Validate(recipe)
            .Should().ContainSingle()
            .Which.Should().Contain("'Quantization' must be preceded by a 'AWQ'");
    }

    [Fact]
    public void MustPrecedeRule_SuccessorBeforePredecessor_ReportsViolation()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 1), ("AWQ", 1));

        validator.Validate(recipe).Should().ContainSingle();
    }

    [Fact]
    public void MustPrecedeRule_MultipleSuccessorsWithoutPredecessor_ReportsMultiple()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("AWQ", "Quantization"));
        var recipe = BuildRecipe(("Quantization", 2));

        validator.Validate(recipe).Should().HaveCount(2);
    }

    [Fact]
    public void Validate_MultipleRules_AllRun()
    {
        var validator = new RecipeValidator();
        validator.AddRule(new MustPrecedeRule("A", "B"));
        validator.AddRule(new MustPrecedeRule("C", "D"));

        var recipe = BuildRecipe(("B", 1), ("D", 1));
        validator.Validate(recipe).Should().HaveCount(2);
    }

    private static Recipe BuildRecipe(params (string Type, int Count)[] entries)
    {
        var recipe = new Recipe();
        foreach (var (type, count) in entries)
        {
            var stage = new Stage { Name = type };
            for (var i = 0; i < count; i++)
            {
                stage.Modifiers.Add(new GenericConfig(type));
            }

            recipe.Stages.Add(stage);
        }

        return recipe;
    }

    private sealed class GenericConfig : ModifierConfig
    {
        public GenericConfig(string type)
        {
            TypeName = type;
        }

        public override string Type => TypeName;

        private string TypeName { get; }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~RecipeValidatorTests"`
Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/Recipes/RecipeValidatorTests.cs
git commit -m "test(core): add RecipeValidator tests"
```

---

### Task 16: Add `RecipeBuilder` + integration test

**Files:**
- Create: `src/LLMCompressorSharp.Core/Recipes/RecipeBuilder.cs`
- Create: `tests/LLMCompressorSharp.Tests/Recipes/RecipeBuilderTests.cs`

- [ ] **Step 1: Write `RecipeBuilder.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Materializes a <see cref="Recipe"/> into the ordered list of <see cref="IModifier"/>
/// instances that a <see cref="Compression.CompressionSession"/> will run.
/// </summary>
public static class RecipeBuilder
{
    /// <summary>Builds modifiers from a recipe, using factories from <see cref="ModifierRegistry"/>.</summary>
    /// <param name="recipe">The parsed recipe.</param>
    /// <returns>Modifiers in declaration order (stage order, then modifier order within a stage).</returns>
    /// <exception cref="RecipeParseException">If a referenced modifier type is unregistered.</exception>
    public static IReadOnlyList<IModifier> Build(Recipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        var built = new List<IModifier>();
        foreach (var (_, _, config) in recipe.EnumerateModifiers())
        {
            var registration = ModifierRegistry.Resolve(config.Type)
                ?? throw new RecipeParseException(
                    $"Modifier type '{config.Type}' is not registered. Did you call ModifierRegistry.Register<...>?");
            built.Add(registration.Factory(config));
        }

        return built;
    }
}
```

- [ ] **Step 2: Write `RecipeBuilderTests.cs`**

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.Core.Recipes;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Recipes;

/// <summary>
/// Tests for <see cref="RecipeBuilder"/> — recipe → modifier list materialization,
/// plus the end-to-end Recipe → CompressionSession integration.
/// </summary>
public class RecipeBuilderTests : IDisposable
{
    public RecipeBuilderTests()
    {
        ModifierRegistry.Clear();
        ModifierRegistry.Register<EchoConfig>("Echo", c => new EchoModifier(c));
    }

    public void Dispose()
    {
        ModifierRegistry.Clear();
    }

    [Fact]
    public void Build_ProducesModifiersInDeclarationOrder()
    {
        var yaml = @"
stages:
  - name: a
    modifiers:
      - type: Echo
        tag: first
  - name: b
    modifiers:
      - type: Echo
        tag: second
      - type: Echo
        tag: third
";
        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);
        modifiers.Should().HaveCount(3);
        modifiers.Cast<EchoModifier>().Select(e => e.Tag).Should().Equal("first", "second", "third");
    }

    [Fact]
    public void Build_UnregisteredType_Throws()
    {
        // Construct a recipe whose config points at an unregistered type discriminator.
        ModifierRegistry.Unregister("Echo");

        var recipe = new Recipe
        {
            Stages =
            {
                new Stage
                {
                    Name = "stage",
                    Modifiers = { new EchoConfig { Tag = "x" } },
                },
            },
        };

        var act = () => RecipeBuilder.Build(recipe);
        act.Should().Throw<RecipeParseException>().WithMessage("*Echo*not registered*");
    }

    [Fact]
    public void EndToEnd_RecipeRunsAllModifiersThroughSession()
    {
        var yaml = @"
stages:
  - name: only
    modifiers:
      - type: Echo
        tag: alpha
      - type: Echo
        tag: beta
";
        var recipe = RecipeParser.Parse(yaml);
        var modifiers = RecipeBuilder.Build(recipe);

        var session = new CompressionSession(modifiers);
        var state = new CompressionState(new Dictionary<string, Tensor>());

        var status = session.Run(state, new[] { ones(1), ones(1) });

        status.Should().Be(SessionStatus.Completed);
        modifiers.Cast<EchoModifier>()
            .Select(m => m.Log)
            .Should().AllSatisfy(l => l.Should().Equal("Init", "Start", "Batch", "Batch", "End", "Finalize"));
    }

    private sealed class EchoConfig : ModifierConfig
    {
        public override string Type => "Echo";

        public string Tag { get; set; } = string.Empty;
    }

    private sealed class EchoModifier : ModifierBase
    {
        public EchoModifier(EchoConfig config)
            : base("Echo", config.Targets, config.Ignore)
        {
            Tag = config.Tag;
        }

        public string Tag { get; }

        public List<string> Log { get; } = new();

        protected override void OnInitialize(CompressionState state) => Log.Add("Init");

        protected override void OnStartCore(CompressionState state) => Log.Add("Start");

        protected override void OnBatchCore(CompressionState state) => Log.Add("Batch");

        protected override void OnEndCore(CompressionState state) => Log.Add("End");

        protected override void OnFinalizeCore(CompressionState state) => Log.Add("Finalize");
    }
}
```

- [ ] **Step 3: Run tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~RecipeBuilderTests"`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```powershell
git add src/LLMCompressorSharp.Core/Recipes/RecipeBuilder.cs tests/LLMCompressorSharp.Tests/Recipes/RecipeBuilderTests.cs
git commit -m "feat(core): add RecipeBuilder + end-to-end integration test"
```

---

### Task 17: Cleanup, full verification, prepare for merge

**Files:**
- Delete: `src/LLMCompressorSharp.Core/PlaceholderMarker.cs`

- [ ] **Step 1: Delete the placeholder**

Run: `Remove-Item src/LLMCompressorSharp.Core/PlaceholderMarker.cs`

- [ ] **Step 2: Full clean test run**

Run:
```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```
Expected: 0 errors, 0 warnings, **~82 tests passing** (45 prior + 9 TargetMatcher + 7 ModifierBase + 6 Session + 6 Parser + 6 Validator + 3 Builder = 82).

- [ ] **Step 3: Commit placeholder removal**

```powershell
git rm src/LLMCompressorSharp.Core/PlaceholderMarker.cs
git commit -m "chore(core): remove PlaceholderMarker (real APIs now present)"
```

- [ ] **Step 4: Tag prep — STOP here**

The controller (main session) handles the merge to `main`, the `v0.1.1-alpha` tag, and the push to `origin`. Stop after Step 3.

---

## Self-Review Notes

**Spec coverage** — every Core item from spec §2.2:
- `IModifier` + `ModifierBase` lifecycle → Tasks 4, 7, 8
- `CompressionSession` + `CompressionState` → Tasks 3, 9, 10
- Recipe POCO + YAML deserialisation → Tasks 11, 12, 13
- Recipe validation (ordering) → Tasks 14, 15
- `RecipeBuilder` end-to-end → Task 16

**Out of scope** (deferred):
- Concrete modifier configs/classes (`GptqConfig`, `GptqModifier`, etc.) → Phase 2
- TorchSharp `Module` integration → Phase 3
- Output writers → Phase 5

**Type consistency:**
- `IModifier.Name` matches `ModifierBase.Name` (string)
- `ModifierBase.Targets`/`Ignore` (`IReadOnlyList<string>`) matches `ModifierConfig.Targets`/`Ignore` (`IReadOnlyList<string>?`) — nullability is the only difference (config allows null which the base coerces to empty)
- `CompressionState.NamedWeights` (`IDictionary<string, Tensor>`) consumed by `ModifierBase.GetTargetedWeights` and by `CompressionSession.Run` parameter
- `ModifierRegistry.Resolve` returns `Registration?` — both `RecipeParser` and `RecipeBuilder` null-check the result and throw `RecipeParseException` on miss

**Known risks**:
- **YamlDotNet `IYamlTypeConverter` API** — Task 12 sketches the signature but flags that the actual 16.x interface should be verified by IntelliSense / package XML before committing. Adapt without weakening behaviour.
- **Process-global `ModifierRegistry`** — Tasks 13 + 16 use `Clear()`/`Dispose` for isolation, but xUnit may parallelize across classes. If parallelism causes flake, add `[Collection("Registry")]` to both test classes to serialise them.
- **`Tensor` lifetime in `CompressionState.NamedWeights`** — the session does not own tensor lifetimes; callers must dispose. Phase 3 will revisit this with `Module`-derived states.
- **StyleCop SA1402 (one type per file)** — Tasks 12 and 14 each place multiple public types in a single file for readability (`RecipeParser` + `RecipeParseException`; `RecipeValidator` + `IRecipeRule` + `MustPrecedeRule`). If SA1402 fires as an error, the implementer should split each public type into its own file matching its name — that's the convention Phase 1a established. `ModifierRegistry.Registration` is a nested `record` and exempt from SA1402.

**Commit count target:** ~15 commits across 17 tasks.

# Phase 0: Repo Bootstrap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the LLMCompressorSharp .NET 10 solution skeleton with five project shells, CI pipeline, HuggingFace cache integration, and the `PR_TO_TORCHSHARP.md` tracker — culminating in a green smoke test that loads TorchSharp and creates a tensor.

**Architecture:** Five C# 14 / .NET 10 projects in one `.slnx` solution: `TorchExtensions` (gap-fillers), `Core` (algorithm framework), `Transformers` (architectures + HF loader), `Cli` (`llmc` tool), `Tests` (xUnit). Central package management via `Directory.Packages.props`. GitHub Actions for CI on Windows + Ubuntu CPU matrix. HF model cache stored in the standard `~/.cache/huggingface/hub/` layout for sharing with the Python ecosystem.

**Tech Stack:** .NET 10 SDK, C# 14, TorchSharp 0.107.0+ (CPU), TorchSharp.PyBridge 1.4.3, YamlDotNet, Microsoft.ML.Tokenizers, Microsoft.ML.OnnxRuntime, System.CommandLine, Spectre.Console, xUnit, MinVer (SemVer from git tags), StyleCop.Analyzers.

**Reference spec:** `docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md`

---

## File Structure

```
LLMCompressorSharp/                                ← repo root
├── .gitattributes                                 ← Create  enforce LF in .cs/.csproj/.yml/.md
├── .gitignore                                     ← Modify  add .NET, .vs, bin, obj, .idea
├── .editorconfig                                  ← Create  formatting + StyleCop SA suppressions
├── global.json                                    ← Create  pin .NET 10 SDK
├── Directory.Build.props                          ← Create  common MSBuild
├── Directory.Packages.props                       ← Create  central package versions
├── LLMCompressorSharp.slnx                        ← Create  XML solution
├── README.md                                      ← Create  project readme
├── PR_TO_TORCHSHARP.md                            ← Create  upstream PR tracker
├── stylecop.json                                  ← Create  StyleCop rules
├── .github/
│   └── workflows/
│       ├── ci.yml                                 ← Create  CPU CI on Windows + Ubuntu
│       ├── gpu-tests.yml                          ← Create  manual + nightly GPU
│       └── release.yml                            ← Create  manual NuGet publish
├── src/
│   ├── LLMCompressorSharp.TorchExtensions/
│   │   ├── LLMCompressorSharp.TorchExtensions.csproj    ← Create
│   │   └── PlaceholderMarker.cs                          ← Create  one-line type so the asm is non-empty
│   ├── LLMCompressorSharp.Core/
│   │   ├── LLMCompressorSharp.Core.csproj                ← Create
│   │   └── PlaceholderMarker.cs                          ← Create
│   ├── LLMCompressorSharp.Transformers/
│   │   ├── LLMCompressorSharp.Transformers.csproj        ← Create
│   │   ├── HuggingFaceCache.cs                           ← Create  cache path resolver (HF-compatible)
│   │   └── PlaceholderMarker.cs                          ← Create
│   └── LLMCompressorSharp.Cli/
│       ├── LLMCompressorSharp.Cli.csproj                 ← Create  PackAsTool=true, ToolCommandName=llmc
│       └── Program.cs                                    ← Create  prints version + does TorchSharp smoke test
└── tests/
    └── LLMCompressorSharp.Tests/
        ├── LLMCompressorSharp.Tests.csproj               ← Create
        ├── SmokeTests.cs                                 ← Create  TorchSharp loads + tensor created
        └── HuggingFaceCacheTests.cs                      ← Create  cache path resolution
```

**Responsibility per file:**
- `Directory.Build.props` — every project inherits these MSBuild props (target framework, lang version, nullability, warnings-as-errors, analyzers).
- `Directory.Packages.props` — central package versions; no `<PackageReference Version="…">` allowed in individual csproj files.
- `LLMCompressorSharp.slnx` — XML-format .NET 10 solution file listing all five projects.
- `stylecop.json` — StyleCop.Analyzers configuration (project name, company name, file header template).
- `.editorconfig` — formatter rules + StyleCop suppressions for the `Architectures/` and `TorchExtensions/` boundaries (where TorchSharp snake_case calls land).
- `PlaceholderMarker.cs` files — minimal `internal sealed class PlaceholderMarker { }` so each empty assembly compiles. Real types replace these in Phase 1+.
- `HuggingFaceCache.cs` — the **only** non-trivial code shipped in Phase 0. Resolves the HF cache directory respecting `$HF_HOME`, `$XDG_CACHE_HOME`, and the platform default. Has full unit tests.
- `Program.cs` (Cli) — single command `version` that prints the package version and creates a 3×3 TorchSharp tensor to confirm the native library loads.
- `SmokeTests.cs` — verifies the CLI binary path and that TorchSharp's CPU library is usable from the test assembly.

---

## Prerequisites & Conventions

Before starting, the engineer should know:

- **.NET 10 SDK** must be installed. Check with `dotnet --version` (should report `10.0.x`).
- **TorchSharp** uses snake_case method names mirroring PyTorch (e.g. `torch.zeros`, `tensor.add_`). This is intentional and StyleCop must be configured to ignore SA1300/SA1303 violations on TorchSharp call sites.
- **Central package management:** every package version is in `Directory.Packages.props`. Individual `.csproj` files use `<PackageReference Include="Name" />` only — no `Version` attribute.
- **xUnit** is the test framework; `[Fact]` for tests, `[Trait("Category", "Gpu")]` for GPU-gated tests, `[Theory]` + `[InlineData]` for parameterised tests.
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`).
- **Branch naming:** `feature/0-<slug>` for this phase. Current branch may need to be created from `main` before starting Task 1.
- **Path separators in commands below** use backslashes (Windows) since the target shell is PowerShell. On Linux/macOS, replace `\` with `/`.

---

### Task 1: Create the working branch and verify .NET 10

**Files:** (no file changes — branch + env check)

- [ ] **Step 1: Verify .NET 10 SDK is installed**

Run: `dotnet --version`
Expected output: starts with `10.` (e.g. `10.0.100`). If not, install .NET 10 SDK before continuing.

- [ ] **Step 2: Verify clean working tree on `main`**

Run: `git status --short`
Expected: empty output (no untracked or modified files). If `LLM-COMPRESSOR.md` is shown as untracked, leave it — it's a scratch file we'll not commit.

Run: `git log --oneline -3`
Expected: recent commits include `afbd423 docs(spec): add Phase 5.5 pure-.NET ONNX export` and `ad29c1a docs: research notes and LLMCompressorSharp design spec`.

- [ ] **Step 3: Create the working branch**

Run: `git checkout -b feature/0-repo-bootstrap`
Expected: `Switched to a new branch 'feature/0-repo-bootstrap'`

---

### Task 2: Establish `.gitattributes` and update `.gitignore`

**Files:**
- Create: `.gitattributes`
- Modify: `.gitignore` (append .NET-specific entries)

- [ ] **Step 1: Create `.gitattributes` enforcing LF line endings for source**

Create file `.gitattributes` with content:

```gitattributes
* text=auto eol=lf

*.cs           text eol=lf
*.csproj       text eol=lf
*.slnx         text eol=lf
*.props        text eol=lf
*.targets      text eol=lf
*.json         text eol=lf
*.yml          text eol=lf
*.yaml         text eol=lf
*.md           text eol=lf
*.editorconfig text eol=lf
*.gitattributes text eol=lf
*.gitignore    text eol=lf

*.dll binary
*.pdb binary
*.png binary
*.jpg binary
*.safetensors binary
*.onnx binary
```

- [ ] **Step 2: Inspect current `.gitignore` to avoid duplicating entries**

Run: `Get-Content .gitignore | Select-String -Pattern '^(bin|obj|\.vs|\.idea|\.vscode)$'`
Expected: may show some entries already present. We only append what's missing.

- [ ] **Step 3: Append .NET entries to `.gitignore`**

Append to `.gitignore`:

```gitignore

# === .NET ===
bin/
obj/
*.user
*.suo
*.userprefs

# Build output
artifacts/
publish/
TestResults/

# IDE
.vs/
.vscode/
.idea/

# Rider
*.DotSettings.user

# MinVer / package output
*.nupkg
*.snupkg
```

- [ ] **Step 4: Verify staged content is correct**

Run: `git diff .gitignore`
Expected: shows the appended block. No prior entries should be duplicated.

- [ ] **Step 5: Commit**

```powershell
git add .gitattributes .gitignore
git commit -m "chore: add .gitattributes and .NET .gitignore entries"
```

---

### Task 3: Pin .NET 10 SDK with `global.json`

**Files:**
- Create: `global.json`

- [ ] **Step 1: Find the installed .NET 10 SDK version**

Run: `dotnet --list-sdks`
Expected: lists installed SDKs. Note the exact .NET 10 version string (e.g. `10.0.100`).

- [ ] **Step 2: Create `global.json` pinning the SDK feature band**

Create file `global.json`:

```json
{
  "sdk": {
    "version": "10.0.100",
    "rollForward": "latestFeature",
    "allowPrerelease": false
  }
}
```

If your installed version is different (e.g. `10.0.101`), replace `10.0.100` with that string.

`rollForward: latestFeature` means any 10.0.1xx SDK satisfies the pin; 10.0.2xx or 10.1.x do not. This protects against unexpected SDK upgrades breaking the build.

- [ ] **Step 3: Verify the pin works**

Run: `dotnet --version`
Expected: matches a version satisfying the pin (e.g. `10.0.100`).

- [ ] **Step 4: Commit**

```powershell
git add global.json
git commit -m "chore: pin .NET 10 SDK in global.json"
```

---

### Task 4: Create `Directory.Build.props` and `Directory.Packages.props`

**Files:**
- Create: `Directory.Build.props`
- Create: `Directory.Packages.props`

- [ ] **Step 1: Create `Directory.Build.props`**

Create file `Directory.Build.props`:

```xml
<Project>

  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <LangVersion>14</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <NoWarn>$(NoWarn);CS1591</NoWarn>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>

    <!-- Repository metadata for NuGet packages -->
    <Authors>James Burton</Authors>
    <Company>LLMCompressorSharp</Company>
    <Product>LLMCompressorSharp</Product>
    <Copyright>Copyright (c) James Burton</Copyright>
    <RepositoryUrl>https://github.com/jamesburton/LLMCompressorSharp</RepositoryUrl>
    <RepositoryType>git</RepositoryType>
    <PackageLicenseExpression>Apache-2.0</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/jamesburton/LLMCompressorSharp</PackageProjectUrl>
    <PublishRepositoryUrl>true</PublishRepositoryUrl>
    <EmbedUntrackedSources>true</EmbedUntrackedSources>
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>

    <!-- Deterministic builds -->
    <ContinuousIntegrationBuild Condition="'$(GITHUB_ACTIONS)' == 'true'">true</ContinuousIntegrationBuild>
  </PropertyGroup>

  <!-- StyleCop applied to all projects -->
  <ItemGroup>
    <PackageReference Include="StyleCop.Analyzers" PrivateAssets="all" />
    <AdditionalFiles Include="$(MSBuildThisFileDirectory)stylecop.json" Link="stylecop.json" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Create `Directory.Packages.props`**

Create file `Directory.Packages.props`:

```xml
<Project>

  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
  </PropertyGroup>

  <ItemGroup>
    <!-- Build & analyzers -->
    <PackageVersion Include="StyleCop.Analyzers" Version="1.2.0-beta.556" />
    <PackageVersion Include="MinVer" Version="6.0.0" />

    <!-- ML / Tensor -->
    <PackageVersion Include="TorchSharp" Version="0.107.0" />
    <PackageVersion Include="TorchSharp-cpu" Version="0.107.0" />
    <PackageVersion Include="TorchSharp.PyBridge" Version="1.4.3" />
    <PackageVersion Include="Microsoft.ML.OnnxRuntime" Version="1.26.0" />
    <PackageVersion Include="Microsoft.ML.Tokenizers" Version="1.0.2" />
    <PackageVersion Include="System.Numerics.Tensors" Version="10.0.0" />

    <!-- CLI -->
    <PackageVersion Include="System.CommandLine" Version="2.0.0-rc.1.26052.1" />
    <PackageVersion Include="Spectre.Console" Version="0.50.0" />

    <!-- Recipe parsing -->
    <PackageVersion Include="YamlDotNet" Version="16.3.0" />

    <!-- Testing -->
    <PackageVersion Include="Microsoft.NET.Test.Sdk" Version="17.13.0" />
    <PackageVersion Include="xunit.v3" Version="1.0.0" />
    <PackageVersion Include="xunit.runner.visualstudio" Version="3.0.0" />
    <PackageVersion Include="coverlet.collector" Version="6.0.4" />
    <PackageVersion Include="FluentAssertions" Version="7.0.0" />
  </ItemGroup>

</Project>
```

> If a listed version no longer resolves (NuGet feed has moved on), upgrade to the latest stable and note the change in the commit message.

- [ ] **Step 3: Commit**

```powershell
git add Directory.Build.props Directory.Packages.props
git commit -m "chore: add Directory.Build.props and central package versions"
```

---

### Task 5: Create `stylecop.json` and `.editorconfig`

**Files:**
- Create: `stylecop.json`
- Create: `.editorconfig`

- [ ] **Step 1: Create `stylecop.json`**

Create file `stylecop.json`:

```json
{
  "$schema": "https://raw.githubusercontent.com/DotNetAnalyzers/StyleCopAnalyzers/master/StyleCop.Analyzers/StyleCop.Analyzers/Settings/stylecop.schema.json",
  "settings": {
    "documentationRules": {
      "companyName": "LLMCompressorSharp",
      "copyrightText": "// Copyright (c) James Burton. Licensed under the Apache-2.0 license.",
      "xmlHeader": false,
      "fileNamingConvention": "stylecop"
    },
    "orderingRules": {
      "usingDirectivesPlacement": "outsideNamespace",
      "systemUsingDirectivesFirst": true
    },
    "namingRules": {
      "allowedHungarianPrefixes": [],
      "allowCommonHungarianPrefixes": true
    }
  }
}
```

- [ ] **Step 2: Create `.editorconfig`**

Create file `.editorconfig`:

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
indent_style = space
insert_final_newline = true
trim_trailing_whitespace = true

[*.{cs,csproj,props,targets,slnx}]
indent_size = 4

[*.{json,yml,yaml,md}]
indent_size = 2

# C# language conventions
[*.cs]
dotnet_sort_system_directives_first = true
csharp_new_line_before_open_brace = all
csharp_new_line_before_else = true
csharp_new_line_before_catch = true
csharp_new_line_before_finally = true
csharp_indent_case_contents = true
csharp_indent_switch_labels = true
csharp_style_var_for_built_in_types = true:suggestion
csharp_style_var_when_type_is_apparent = true:suggestion
csharp_style_var_elsewhere = true:silent
csharp_style_expression_bodied_methods = when_on_single_line:suggestion
csharp_style_expression_bodied_properties = true:suggestion
csharp_style_expression_bodied_accessors = true:suggestion

# StyleCop: suppress documentation requirements for internal/private types in Phase 0
[*.cs]
dotnet_diagnostic.SA1633.severity = none   # File header (we use a different convention)
dotnet_diagnostic.SA1101.severity = none   # Prefix local calls with this
dotnet_diagnostic.SA1309.severity = none   # Field names must not begin with underscore
dotnet_diagnostic.SA1310.severity = none   # Field names must not contain underscore
dotnet_diagnostic.SA1413.severity = none   # Use trailing comma in multi-line initializers (opinion)

# Allow snake_case at the TorchSharp boundary
[src/LLMCompressorSharp.TorchExtensions/**.cs]
dotnet_diagnostic.SA1300.severity = none   # Element should begin with upper-case letter
dotnet_diagnostic.SA1303.severity = none   # Const field names should begin with upper-case letter
dotnet_diagnostic.SA1311.severity = none   # Static readonly fields should begin with upper-case letter

[src/LLMCompressorSharp.Transformers/Architectures/**.cs]
dotnet_diagnostic.SA1300.severity = none
dotnet_diagnostic.SA1303.severity = none
dotnet_diagnostic.SA1311.severity = none
```

- [ ] **Step 3: Commit**

```powershell
git add stylecop.json .editorconfig
git commit -m "chore: add StyleCop config and .editorconfig"
```

---

### Task 6: Create the solution file

**Files:**
- Create: `LLMCompressorSharp.slnx`

- [ ] **Step 1: Create the solution skeleton**

Create file `LLMCompressorSharp.slnx`:

```xml
<Solution>
  <Folder Name="/src/">
    <Project Path="src/LLMCompressorSharp.TorchExtensions/LLMCompressorSharp.TorchExtensions.csproj" />
    <Project Path="src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj" />
    <Project Path="src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj" />
    <Project Path="src/LLMCompressorSharp.Cli/LLMCompressorSharp.Cli.csproj" />
  </Folder>
  <Folder Name="/tests/">
    <Project Path="tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj" />
  </Folder>
  <Folder Name="/Solution Items/">
    <File Path=".editorconfig" />
    <File Path=".gitattributes" />
    <File Path=".gitignore" />
    <File Path="Directory.Build.props" />
    <File Path="Directory.Packages.props" />
    <File Path="global.json" />
    <File Path="README.md" />
    <File Path="PR_TO_TORCHSHARP.md" />
    <File Path="stylecop.json" />
  </Folder>
</Solution>
```

> The referenced .csproj files don't exist yet — that's expected. Solution validation is deferred to Task 11 (first `dotnet build`).

- [ ] **Step 2: Commit**

```powershell
git add LLMCompressorSharp.slnx
git commit -m "chore: add LLMCompressorSharp.slnx solution skeleton"
```

---

### Task 7: Create the `TorchExtensions` project shell

**Files:**
- Create: `src/LLMCompressorSharp.TorchExtensions/LLMCompressorSharp.TorchExtensions.csproj`
- Create: `src/LLMCompressorSharp.TorchExtensions/PlaceholderMarker.cs`

- [ ] **Step 1: Create the project file**

Create file `src/LLMCompressorSharp.TorchExtensions/LLMCompressorSharp.TorchExtensions.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <RootNamespace>LLMCompressorSharp.TorchExtensions</RootNamespace>
    <AssemblyName>LLMCompressorSharp.TorchExtensions</AssemblyName>
    <PackageId>LLMCompressorSharp.TorchExtensions</PackageId>
    <Description>TorchSharp extension helpers used by LLMCompressorSharp — fills API gaps until upstream lands matching support.</Description>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="TorchSharp" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Create the placeholder marker**

Create file `src/LLMCompressorSharp.TorchExtensions/PlaceholderMarker.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.TorchExtensions;

/// <summary>
/// Placeholder type so the assembly is non-empty before Phase 1.
/// Replaced by real public APIs in subsequent phases.
/// </summary>
internal sealed class PlaceholderMarker
{
}
```

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.TorchExtensions/
git commit -m "feat(torch-extensions): add project shell"
```

---

### Task 8: Create the `Core` project shell

**Files:**
- Create: `src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj`
- Create: `src/LLMCompressorSharp.Core/PlaceholderMarker.cs`

- [ ] **Step 1: Create the project file**

Create file `src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <RootNamespace>LLMCompressorSharp.Core</RootNamespace>
    <AssemblyName>LLMCompressorSharp.Core</AssemblyName>
    <PackageId>LLMCompressorSharp.Core</PackageId>
    <Description>Core compression algorithms (RTN, GPTQ, SparseGPT, SmoothQuant, AWQ, WANDA, Magnitude) and the modifier/recipe framework.</Description>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\LLMCompressorSharp.TorchExtensions\LLMCompressorSharp.TorchExtensions.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="TorchSharp" />
    <PackageReference Include="TorchSharp.PyBridge" />
    <PackageReference Include="YamlDotNet" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Create the placeholder marker**

Create file `src/LLMCompressorSharp.Core/PlaceholderMarker.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core;

/// <summary>
/// Placeholder type so the assembly is non-empty before Phase 1.
/// Replaced by IModifier, ModifierBase, CompressionSession etc. in Phase 1.
/// </summary>
internal sealed class PlaceholderMarker
{
}
```

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Core/
git commit -m "feat(core): add project shell"
```

---

### Task 9: Create the `Transformers` project shell with `HuggingFaceCache`

This is the only Phase 0 project that ships real (tested) code: the HuggingFace cache path resolver. We TDD it.

**Files:**
- Create: `src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj`
- Create: `src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`
- Create: `src/LLMCompressorSharp.Transformers/HuggingFaceCache.cs` (deferred to Task 12 — TDD'd from the test side)

- [ ] **Step 1: Create the project file**

Create file `src/LLMCompressorSharp.Transformers/LLMCompressorSharp.Transformers.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <RootNamespace>LLMCompressorSharp.Transformers</RootNamespace>
    <AssemblyName>LLMCompressorSharp.Transformers</AssemblyName>
    <PackageId>LLMCompressorSharp.Transformers</PackageId>
    <Description>Transformer architectures (starting with LLaMA), HuggingFace cache + loader, and tokenizer integration for LLMCompressorSharp.</Description>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\LLMCompressorSharp.Core\LLMCompressorSharp.Core.csproj" />
    <ProjectReference Include="..\LLMCompressorSharp.TorchExtensions\LLMCompressorSharp.TorchExtensions.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="TorchSharp" />
    <PackageReference Include="TorchSharp.PyBridge" />
    <PackageReference Include="Microsoft.ML.Tokenizers" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Create the placeholder marker**

Create file `src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Placeholder type so the assembly is non-empty before Phase 3.
/// Replaced by LlamaForCausalLM etc. in Phase 3.
/// </summary>
internal sealed class PlaceholderMarker
{
}
```

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/
git commit -m "feat(transformers): add project shell"
```

---

### Task 10: Create the `Cli` project shell

**Files:**
- Create: `src/LLMCompressorSharp.Cli/LLMCompressorSharp.Cli.csproj`
- Create: `src/LLMCompressorSharp.Cli/Program.cs`

- [ ] **Step 1: Create the project file**

Create file `src/LLMCompressorSharp.Cli/LLMCompressorSharp.Cli.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <RootNamespace>LLMCompressorSharp.Cli</RootNamespace>
    <AssemblyName>llmc</AssemblyName>
    <PackageId>LLMCompressorSharp</PackageId>
    <PackAsTool>true</PackAsTool>
    <ToolCommandName>llmc</ToolCommandName>
    <Description>LLMCompressorSharp command-line tool. Compresses LLaMA-family LLMs via GPTQ, AWQ, SparseGPT, SmoothQuant, WANDA, and Magnitude pruning.</Description>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\LLMCompressorSharp.Core\LLMCompressorSharp.Core.csproj" />
    <ProjectReference Include="..\LLMCompressorSharp.Transformers\LLMCompressorSharp.Transformers.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="TorchSharp" />
    <PackageReference Include="TorchSharp-cpu" />
    <PackageReference Include="System.CommandLine" />
    <PackageReference Include="Spectre.Console" />
  </ItemGroup>

</Project>
```

> `TorchSharp-cpu` is referenced *only* in the CLI project — it bundles the LibTorch native binaries. Library projects depend on the API-only `TorchSharp` package so they can be consumed by callers who choose their own backend (CPU/CUDA/MPS).

- [ ] **Step 2: Create `Program.cs` with version + smoke command**

Create file `src/LLMCompressorSharp.Cli/Program.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Reflection;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Cli;

/// <summary>
/// Entry point for the <c>llmc</c> command-line tool.
/// </summary>
public static class Program
{
    /// <summary>
    /// Runs the CLI. Phase 0 implements <c>version</c> and <c>smoke</c> subcommands only;
    /// the full <c>compress</c> command lands in Phase 6.
    /// </summary>
    /// <param name="args">Command-line arguments.</param>
    /// <returns>Process exit code. 0 on success, non-zero on failure.</returns>
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] == "--help" || args[0] == "-h")
        {
            PrintHelp();
            return 0;
        }

        return args[0] switch
        {
            "version" => RunVersion(),
            "smoke" => RunSmoke(),
            _ => UnknownCommand(args[0]),
        };
    }

    private static int RunVersion()
    {
        var version = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "0.0.0-unknown";
        Console.WriteLine($"llmc {version}");
        return 0;
    }

    private static int RunSmoke()
    {
        Console.WriteLine("LLMCompressorSharp smoke test...");
        var t = zeros(3, 3);
        Console.WriteLine($"Created TorchSharp tensor shape=[{string.Join(",", t.shape)}] dtype={t.dtype}");
        t.Dispose();
        Console.WriteLine("OK");
        return 0;
    }

    private static int UnknownCommand(string name)
    {
        Console.Error.WriteLine($"Unknown command: {name}");
        PrintHelp();
        return 1;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("Usage: llmc <command>");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  version    Print the installed llmc version.");
        Console.WriteLine("  smoke      Run a minimal TorchSharp smoke test.");
    }
}
```

- [ ] **Step 3: Commit**

```powershell
git add src/LLMCompressorSharp.Cli/
git commit -m "feat(cli): add llmc tool shell with version and smoke commands"
```

---

### Task 11: Verify the solution builds (first compile)

**Files:** (no file changes — build verification only)

- [ ] **Step 1: Restore packages**

Run: `dotnet restore LLMCompressorSharp.slnx`
Expected: completes without errors. NuGet downloads TorchSharp, StyleCop, etc.

If you see `NU1008: Projects that use central package version management should not define the version on the PackageReference items`, find any csproj with a `Version="…"` attribute and remove the version (it must come from `Directory.Packages.props`).

If you see `NU1604: Project dependency does not contain an inclusive lower bound`, ensure `Directory.Packages.props` contains a `<PackageVersion>` for every package referenced in any csproj.

- [ ] **Step 2: Build the solution**

Run: `dotnet build LLMCompressorSharp.slnx --no-restore`
Expected: `Build succeeded` with no errors and no warnings (warnings would be elevated to errors by `TreatWarningsAsErrors`).

If StyleCop warnings appear, either fix them or extend the suppression list in `.editorconfig`. Do **not** add `<NoWarn>` entries to silence StyleCop globally.

- [ ] **Step 3: Run the smoke command from the built CLI**

Run: `dotnet run --project src/LLMCompressorSharp.Cli/LLMCompressorSharp.Cli.csproj -- smoke`
Expected output:
```
LLMCompressorSharp smoke test...
Created TorchSharp tensor shape=[3,3] dtype=Float32
OK
```

If TorchSharp fails to load native binaries (`DllNotFoundException` on `LibTorchSharp`), confirm `TorchSharp-cpu` is referenced in `LLMCompressorSharp.Cli.csproj`.

- [ ] **Step 4: Commit nothing — this is a verification step**

No file changes. Proceed to Task 12.

---

### Task 12: TDD `HuggingFaceCache` — failing test first

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj`
- Create: `tests/LLMCompressorSharp.Tests/HuggingFaceCacheTests.cs`

- [ ] **Step 1: Create the test project file**

Create file `tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <RootNamespace>LLMCompressorSharp.Tests</RootNamespace>
    <AssemblyName>LLMCompressorSharp.Tests</AssemblyName>
    <IsPackable>false</IsPackable>
    <IsPublishable>false</IsPublishable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\..\src\LLMCompressorSharp.TorchExtensions\LLMCompressorSharp.TorchExtensions.csproj" />
    <ProjectReference Include="..\..\src\LLMCompressorSharp.Core\LLMCompressorSharp.Core.csproj" />
    <ProjectReference Include="..\..\src\LLMCompressorSharp.Transformers\LLMCompressorSharp.Transformers.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" />
    <PackageReference Include="xunit.v3" />
    <PackageReference Include="xunit.runner.visualstudio" />
    <PackageReference Include="coverlet.collector" />
    <PackageReference Include="FluentAssertions" />
    <PackageReference Include="TorchSharp" />
    <PackageReference Include="TorchSharp-cpu" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Write the failing tests for `HuggingFaceCache`**

Create file `tests/LLMCompressorSharp.Tests/HuggingFaceCacheTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.Transformers;
using Xunit;

namespace LLMCompressorSharp.Tests;

/// <summary>
/// Tests for <see cref="HuggingFaceCache"/> path resolution. The cache root must follow the
/// standard HF layout so weights are shared with the Python ecosystem (huggingface_hub, transformers, datasets).
/// </summary>
public class HuggingFaceCacheTests
{
    [Fact]
    public void ResolveCacheRoot_WhenHfHomeIsSet_UsesHfHomeSlashHub()
    {
        var env = new TestEnvironment
        {
            HfHome = "/custom/hf",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/custom/hf", "hub"));
    }

    [Fact]
    public void ResolveCacheRoot_WhenHfHubCacheIsSet_TakesPrecedenceOverHfHome()
    {
        var env = new TestEnvironment
        {
            HfHome = "/custom/hf",
            HfHubCache = "/explicit/cache",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be("/explicit/cache");
    }

    [Fact]
    public void ResolveCacheRoot_WhenXdgCacheHomeIsSet_UsesXdgSlashHuggingfaceSlashHub()
    {
        var env = new TestEnvironment
        {
            XdgCacheHome = "/xdg/cache",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/xdg/cache", "huggingface", "hub"));
    }

    [Fact]
    public void ResolveCacheRoot_WithNoEnvVars_FallsBackToUserProfileCache()
    {
        var env = new TestEnvironment
        {
            UserProfile = "/home/test-user",
        };

        var result = HuggingFaceCache.ResolveCacheRoot(env);

        result.Should().Be(Path.Combine("/home/test-user", ".cache", "huggingface", "hub"));
    }

    [Theory]
    [InlineData("HuggingFaceTB/SmolLM2-135M", "models--HuggingFaceTB--SmolLM2-135M")]
    [InlineData("meta-llama/Llama-3.2-1B", "models--meta-llama--Llama-3.2-1B")]
    [InlineData("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")]
    public void GetRepoFolderName_MatchesHuggingFaceLayout(string repoId, string expectedFolder)
    {
        HuggingFaceCache.GetRepoFolderName(repoId).Should().Be(expectedFolder);
    }

    [Fact]
    public void GetSnapshotPath_ComposesRootRepoAndRevision()
    {
        var path = HuggingFaceCache.GetSnapshotPath(
            cacheRoot: "/cache/huggingface/hub",
            repoId: "HuggingFaceTB/SmolLM2-135M",
            revision: "abc123def");

        path.Should().Be(Path.Combine(
            "/cache/huggingface/hub",
            "models--HuggingFaceTB--SmolLM2-135M",
            "snapshots",
            "abc123def"));
    }

    [Fact]
    public void GetRepoFolderName_WithInvalidRepoId_Throws()
    {
        var act = () => HuggingFaceCache.GetRepoFolderName("invalid-no-slash");
        act.Should().Throw<ArgumentException>().WithMessage("*org/repo*");
    }

    private sealed class TestEnvironment : IEnvironment
    {
        public string? HfHubCache { get; init; }

        public string? HfHome { get; init; }

        public string? XdgCacheHome { get; init; }

        public string UserProfile { get; init; } = "/home/default";
    }
}
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj`
Expected: build fails with `CS0246: The type or namespace name 'HuggingFaceCache' could not be found` and `CS0246: 'IEnvironment'`.

This is the expected RED state for TDD. Proceed to Task 13 to make the tests pass.

- [ ] **Step 4: Commit the failing tests**

```powershell
git add tests/LLMCompressorSharp.Tests/
git commit -m "test(transformers): add failing HuggingFaceCache resolution tests"
```

---

### Task 13: Implement `HuggingFaceCache` and `IEnvironment` to make tests pass

**Files:**
- Create: `src/LLMCompressorSharp.Transformers/IEnvironment.cs`
- Create: `src/LLMCompressorSharp.Transformers/SystemEnvironment.cs`
- Create: `src/LLMCompressorSharp.Transformers/HuggingFaceCache.cs`

- [ ] **Step 1: Create the `IEnvironment` abstraction (testability seam)**

Create file `src/LLMCompressorSharp.Transformers/IEnvironment.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Provides access to environment-level inputs that <see cref="HuggingFaceCache"/> needs:
/// HuggingFace environment variables and the current user's home directory. Abstracted to
/// keep cache-resolution tests hermetic.
/// </summary>
public interface IEnvironment
{
    /// <summary>Value of the <c>HF_HUB_CACHE</c> environment variable, or <see langword="null"/>.</summary>
    string? HfHubCache { get; }

    /// <summary>Value of the <c>HF_HOME</c> environment variable, or <see langword="null"/>.</summary>
    string? HfHome { get; }

    /// <summary>Value of the <c>XDG_CACHE_HOME</c> environment variable, or <see langword="null"/>.</summary>
    string? XdgCacheHome { get; }

    /// <summary>The current user's profile directory (e.g. <c>~</c> on Unix, <c>%USERPROFILE%</c> on Windows).</summary>
    string UserProfile { get; }
}
```

- [ ] **Step 2: Create the default `IEnvironment` implementation**

Create file `src/LLMCompressorSharp.Transformers/SystemEnvironment.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Default <see cref="IEnvironment"/> backed by <see cref="System.Environment"/>.
/// </summary>
public sealed class SystemEnvironment : IEnvironment
{
    /// <summary>Singleton instance.</summary>
    public static readonly SystemEnvironment Instance = new();

    private SystemEnvironment()
    {
    }

    /// <inheritdoc />
    public string? HfHubCache => Environment.GetEnvironmentVariable("HF_HUB_CACHE");

    /// <inheritdoc />
    public string? HfHome => Environment.GetEnvironmentVariable("HF_HOME");

    /// <inheritdoc />
    public string? XdgCacheHome => Environment.GetEnvironmentVariable("XDG_CACHE_HOME");

    /// <inheritdoc />
    public string UserProfile => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
}
```

- [ ] **Step 3: Create `HuggingFaceCache`**

Create file `src/LLMCompressorSharp.Transformers/HuggingFaceCache.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Transformers;

/// <summary>
/// Resolves paths inside the standard HuggingFace cache layout so weights are shared with
/// other HuggingFace-aware tooling (huggingface_hub, transformers, datasets).
/// </summary>
/// <remarks>
/// Resolution precedence (matches huggingface_hub):
/// <list type="number">
///   <item><c>HF_HUB_CACHE</c> — explicit override of the hub cache root.</item>
///   <item><c>HF_HOME</c> — root for all HF caches; hub cache lives at <c>$HF_HOME/hub</c>.</item>
///   <item><c>XDG_CACHE_HOME</c> — XDG base directory; <c>$XDG_CACHE_HOME/huggingface/hub</c>.</item>
///   <item>User profile fallback: <c>~/.cache/huggingface/hub</c>.</item>
/// </list>
/// </remarks>
public static class HuggingFaceCache
{
    /// <summary>
    /// Returns the path to the HuggingFace hub cache root for the supplied environment.
    /// </summary>
    /// <param name="environment">Environment provider; pass <see cref="SystemEnvironment.Instance"/> in production.</param>
    /// <returns>Absolute path to the hub cache root directory.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="environment"/> is <see langword="null"/>.</exception>
    public static string ResolveCacheRoot(IEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);

        if (!string.IsNullOrEmpty(environment.HfHubCache))
        {
            return environment.HfHubCache;
        }

        if (!string.IsNullOrEmpty(environment.HfHome))
        {
            return Path.Combine(environment.HfHome, "hub");
        }

        if (!string.IsNullOrEmpty(environment.XdgCacheHome))
        {
            return Path.Combine(environment.XdgCacheHome, "huggingface", "hub");
        }

        return Path.Combine(environment.UserProfile, ".cache", "huggingface", "hub");
    }

    /// <summary>
    /// Returns the folder name for a HuggingFace repo, matching the layout used by huggingface_hub.
    /// </summary>
    /// <param name="repoId">Repo identifier in <c>org/repo</c> form (e.g. <c>HuggingFaceTB/SmolLM2-135M</c>).</param>
    /// <returns>Folder name (e.g. <c>models--HuggingFaceTB--SmolLM2-135M</c>).</returns>
    /// <exception cref="ArgumentException">If <paramref name="repoId"/> is not in <c>org/repo</c> form.</exception>
    public static string GetRepoFolderName(string repoId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var parts = repoId.Split('/');
        if (parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
        {
            throw new ArgumentException(
                $"Repo id must be in 'org/repo' form. Got: '{repoId}'.",
                nameof(repoId));
        }

        return $"models--{parts[0]}--{parts[1]}";
    }

    /// <summary>
    /// Returns the absolute path to a given snapshot of a repo inside the cache.
    /// </summary>
    /// <param name="cacheRoot">Cache root from <see cref="ResolveCacheRoot"/>.</param>
    /// <param name="repoId">Repo identifier in <c>org/repo</c> form.</param>
    /// <param name="revision">Revision SHA or tag name.</param>
    /// <returns>Absolute snapshot directory path.</returns>
    public static string GetSnapshotPath(string cacheRoot, string repoId, string revision)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);

        return Path.Combine(cacheRoot, GetRepoFolderName(repoId), "snapshots", revision);
    }
}
```

- [ ] **Step 4: Replace the `Transformers` placeholder with a real `internal` doc-comment**

We'd already created a placeholder marker. Now that `HuggingFaceCache` exists, the placeholder is no longer needed:

Delete file `src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`.

Run: `Remove-Item src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj`
Expected: all 9 `HuggingFaceCacheTests` pass (the 6 `[Fact]`s + 3 `[Theory]` data rows).

If any test fails, inspect the failure message — most likely a path-separator mismatch or a missing `ArgumentNullException`.

- [ ] **Step 6: Commit**

```powershell
git add src/LLMCompressorSharp.Transformers/IEnvironment.cs
git add src/LLMCompressorSharp.Transformers/SystemEnvironment.cs
git add src/LLMCompressorSharp.Transformers/HuggingFaceCache.cs
git rm src/LLMCompressorSharp.Transformers/PlaceholderMarker.cs
git commit -m "feat(transformers): implement HuggingFaceCache path resolver"
```

---

### Task 14: Add a TorchSharp smoke test

**Files:**
- Create: `tests/LLMCompressorSharp.Tests/SmokeTests.cs`

- [ ] **Step 1: Write the smoke test**

Create file `tests/LLMCompressorSharp.Tests/SmokeTests.cs`:

```csharp
// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests;

/// <summary>
/// Smoke tests verifying the toolchain is functional: TorchSharp loads its native library
/// and can allocate a CPU tensor.
/// </summary>
public class SmokeTests
{
    [Fact]
    public void TorchSharp_CanCreateCpuTensor()
    {
        using var t = zeros(2, 3);

        t.shape.Should().Equal(2, 3);
        t.device_type.Should().Be(DeviceType.CPU);
    }

    [Fact]
    public void TorchSharp_CanRunSimpleMatmul()
    {
        using var a = ones(2, 3);
        using var b = ones(3, 4);
        using var c = matmul(a, b);

        c.shape.Should().Equal(2, 4);
        c[0, 0].ToSingle().Should().Be(3.0f);
    }
}
```

- [ ] **Step 2: Run the smoke tests**

Run: `dotnet test tests/LLMCompressorSharp.Tests/LLMCompressorSharp.Tests.csproj --filter "FullyQualifiedName~SmokeTests"`
Expected: both tests pass. Test discovery may take 10–20s on first run while NuGet caches warm up.

If `TorchSharp_CanCreateCpuTensor` fails with `DllNotFoundException`, confirm the test project references `TorchSharp-cpu` (Task 12 step 1).

- [ ] **Step 3: Commit**

```powershell
git add tests/LLMCompressorSharp.Tests/SmokeTests.cs
git commit -m "test: add TorchSharp CPU smoke tests"
```

---

### Task 15: Add `PR_TO_TORCHSHARP.md` template

**Files:**
- Create: `PR_TO_TORCHSHARP.md`

- [ ] **Step 1: Create the tracker file**

Create file `PR_TO_TORCHSHARP.md`:

```markdown
# PR_TO_TORCHSHARP — Upstream Contribution Tracker

Tracks every TorchSharp gap we hit or notice, split into **Required** (we worked
around it; we owe a PR) and **Proposed** (would benefit the community; we
haven't needed it ourselves yet).

Inline code references each entry with `// TODO(PR_TO_TORCHSHARP <id>)` so we
can find every site when we prepare the actual PR.

---

## Entry Template

```markdown
### [<id>] <Short title>

**Status:** Required | Proposed
**Category:** Quantization | Autograd | Memory | Tracing | Other
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `path/to/file.cs` (action)

**Why upstream cares:** <one paragraph>

**Our workaround:** `LLMCompressorSharp.TorchExtensions.<Type>` (file)

**PR readiness:** <estimate + blockers>
```

---

## Required

### [R-001] FakeQuantize autograd function

**Status:** Required — pending implementation in `LLMCompressorSharp.TorchExtensions` (Phase 1).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/NN/Quantization/FakeQuantize.cs` (new)
- `src/TorchSharp/Tensor/torch.cs` (add `torch.fake_quantize_per_tensor_affine` static)
- `test/TorchSharpTest/TestQuantization.cs` (new tests)

**Why upstream cares:** Mirrors `torch.fake_quantize_per_tensor_affine` from PyTorch. Bridges a key gap to a full `torch.quantization` story without requiring a higher-level API. Useful for any quantization-aware workflow, not only ours.

**Our workaround:** `LLMCompressorSharp.TorchExtensions.FakeQuantizeFunction` (custom autograd via `SingleTensorFunction<T>`). To be implemented in Phase 1.

**PR readiness:** Self-contained; STE backward is standard. Estimate: half-day cleanup once our workaround is stable.

---

### [R-002] MinMax observers as TorchSharp modules

**Status:** Required — pending implementation in `LLMCompressorSharp.TorchExtensions` (Phase 1).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/NN/Quantization/Observers/MinMaxObserver.cs` (new)
- `src/TorchSharp/NN/Quantization/Observers/PerChannelMinMaxObserver.cs` (new)
- `src/TorchSharp/NN/Quantization/Observers/MovingAverageMinMaxObserver.cs` (new)
- `test/TorchSharpTest/TestObservers.cs` (new tests)

**Why upstream cares:** Mirrors `torch.ao.quantization.observer.*`. Required to build any calibration workflow.

**Our workaround:** Observer hierarchy in `LLMCompressorSharp.TorchExtensions.Observers.*`. To be implemented in Phase 1.

**PR readiness:** Self-contained per observer. Estimate: 1 day cleanup per observer once stable.

---

### [R-003] True packed `QInt4` dtype

**Status:** Required — likely fork-required (blocks pure-extensions strategy).
**Category:** Quantization
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- Native LibTorch bindings (Pinvoke layer)
- `src/TorchSharp/Tensor/torch.ScalarType.cs`
- `src/TorchSharp/Tensor/Tensor.factories.cs`

**Why upstream cares:** Real 4-bit packed storage is essential for production memory savings. Currently `QInt8`/`QUInt8` are the only packed quantized dtypes.

**Our workaround:** `LLMCompressorSharp.TorchExtensions.Int4PackedTensor` — FP32-backed simulation. Provides correctness but no memory benefit. To be implemented in Phase 1.

**PR readiness:** Blocked on native binding work. Likely escalates to fork-submodule path per the design spec (Section 3). To be re-evaluated end of Phase 4 when actual int4 storage matters for accuracy validation.

---

## Proposed

### [P-001] Expose CUDA memory stats

**Status:** Proposed
**Category:** Memory
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- `src/TorchSharp/Tensor/torch.cuda.cs` (add `memory_allocated`, `memory_reserved`, `memory_stats`)

**Why upstream cares:** Mirrors `torch.cuda.memory_allocated()` etc. Useful for any application needing to monitor VRAM during long-running calibration / training.

**Our workaround:** Optional P/Invoke to NVML in `LLMCompressorSharp.TorchExtensions.NvmlMemoryStats`. Works but is platform-specific (Linux/Windows with NVIDIA driver). To be implemented in Phase 1.

**PR readiness:** Small. Estimate: 1–2 days once we have a working internal version to validate against.

---

### [P-002] FX-style symbolic tracing

**Status:** Proposed — large scope, likely permanent gap.
**Category:** Tracing
**Source files we'd add/modify in `dotnet/TorchSharp`:**
- New `src/TorchSharp/Fx/` namespace
- Substantial — would require ahead-of-time C# code analysis or runtime hook instrumentation

**Why upstream cares:** PyTorch's `torch.fx` enables many ecosystem tools (model rewriting, ONNX export, llm-compressor's sequential pipeline). A .NET equivalent would unlock similar tooling.

**Our workaround:** Manual per-architecture sequential calibration loops (e.g. `Architectures/Llama/SequentialCalibration.cs`). Avoids tracing entirely. To be implemented in Phase 4.

**PR readiness:** Out of scope for upstream contribution in the foreseeable future. Documented here as a known gap.

---

## How To Use This File

1. When you hit a TorchSharp limitation in code, add an entry under the right section.
2. Mark the inline workaround with `// TODO(PR_TO_TORCHSHARP <id>)` so we can find all call sites when the PR is ready.
3. Once a workaround is stable + tested, raise the corresponding PR upstream. Link the PR URL back here when opened.
4. When a PR merges, remove the inline TODOs and replace the workaround with the upstream API.
```

- [ ] **Step 2: Commit**

```powershell
git add PR_TO_TORCHSHARP.md
git commit -m "docs: add PR_TO_TORCHSHARP tracker with initial entries"
```

---

### Task 16: Create the project README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create the README**

Create file `README.md`:

```markdown
# LLMCompressorSharp

A .NET 10 / C# library and CLI for compressing LLaMA-family large language models.
Reimplements the core algorithms from the Python
[llm-compressor](https://github.com/vllm-project/llm-compressor) project on top of
[TorchSharp](https://github.com/dotnet/TorchSharp).

> **Status:** Phase 0 — repo bootstrap. APIs are not stable until v1.0.

## Goals

- Pure-.NET compression pipeline: RTN, SmoothQuant, AWQ, GPTQ, SparseGPT, WANDA, Magnitude pruning.
- HuggingFace-cache compatible: shares model weights with the Python ecosystem.
- Output formats: safetensors + `compressed-tensors` config (vLLM-compatible) and ONNX (pure .NET export).
- Single CLI: `dnx llmc compress ...`

## Quick start (after Phase 6)

```bash
dotnet tool install -g LLMCompressorSharp
dnx llmc compress \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --recipe gptq-w4a16 \
  --output ./compressed
```

## Project layout

```
src/
  LLMCompressorSharp.TorchExtensions/   TorchSharp gap-fillers
  LLMCompressorSharp.Core/              Modifiers, observers, recipes
  LLMCompressorSharp.Transformers/      LLaMA architecture, HF loader
  LLMCompressorSharp.Cli/               The `llmc` tool
tests/
  LLMCompressorSharp.Tests/
docs/                                   Research notes and design
```

## Build

```powershell
dotnet build LLMCompressorSharp.slnx
dotnet test  LLMCompressorSharp.slnx
```

Requires .NET 10 SDK. CPU-only build works on any platform; CUDA-accelerated
compression requires a Windows or Linux machine with an NVIDIA GPU and the
relevant `TorchSharp-cuda-*` package referenced at deploy time.

## Documentation

- [`docs/INDEX.md`](docs/INDEX.md) — research notes (llm-compressor, .NET 10, TorchSharp)
- [`docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md`](docs/superpowers/specs/2026-05-15-llmcompressorsharp-design.md) — design spec
- [`PR_TO_TORCHSHARP.md`](PR_TO_TORCHSHARP.md) — upstream contribution tracker

## License

Apache-2.0.
```

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "docs: add project README"
```

---

### Task 17: Add the CI workflow (`ci.yml`)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the directory if needed**

Run: `New-Item -ItemType Directory -Force .github/workflows | Out-Null`

- [ ] **Step 2: Write the CI workflow**

Create file `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-test:
    name: ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, ubuntu-latest]

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0   # MinVer needs full history for version computation

      - name: Setup .NET 10
        uses: actions/setup-dotnet@v4
        with:
          global-json-file: global.json

      - name: NuGet cache
        uses: actions/cache@v4
        with:
          path: ~/.nuget/packages
          key: ${{ runner.os }}-nuget-${{ hashFiles('**/Directory.Packages.props', '**/*.csproj') }}
          restore-keys: |
            ${{ runner.os }}-nuget-

      - name: HuggingFace cache
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/huggingface/hub
            ~/AppData/Local/huggingface/hub
          key: ${{ runner.os }}-hf-v1
          restore-keys: |
            ${{ runner.os }}-hf-

      - name: Restore
        run: dotnet restore LLMCompressorSharp.slnx

      - name: Build
        run: dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release

      - name: Test (CPU only)
        run: >
          dotnet test LLMCompressorSharp.slnx
          --no-build --configuration Release
          --filter "Category!=Gpu"
          --logger "trx;LogFileName=test-results.trx"
          --results-directory artifacts/test-results

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.os }}
          path: artifacts/test-results/**/*.trx
          if-no-files-found: warn
```

- [ ] **Step 3: Commit**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: add CPU build + test workflow for Windows and Ubuntu"
```

---

### Task 18: Add the GPU test workflow stub

**Files:**
- Create: `.github/workflows/gpu-tests.yml`

- [ ] **Step 1: Write the GPU workflow**

Create file `.github/workflows/gpu-tests.yml`:

```yaml
name: GPU Tests

on:
  workflow_dispatch:
  schedule:
    - cron: '0 3 * * *'   # nightly at 03:00 UTC

jobs:
  gpu-tests:
    name: GPU tests (self-hosted)
    runs-on: [self-hosted, cuda]
    if: ${{ vars.ENABLE_GPU_TESTS == 'true' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup .NET 10
        uses: actions/setup-dotnet@v4
        with:
          global-json-file: global.json

      - name: NuGet cache
        uses: actions/cache@v4
        with:
          path: ~/.nuget/packages
          key: gpu-nuget-${{ hashFiles('**/Directory.Packages.props', '**/*.csproj') }}

      - name: HuggingFace cache
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface/hub
          key: gpu-hf-v1

      - name: Restore
        run: dotnet restore LLMCompressorSharp.slnx

      - name: Build
        run: dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release

      - name: Test (GPU)
        run: >
          dotnet test LLMCompressorSharp.slnx
          --no-build --configuration Release
          --filter "Category=Gpu"
          --logger "trx;LogFileName=gpu-test-results.trx"
          --results-directory artifacts/test-results

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: gpu-test-results
          path: artifacts/test-results/**/*.trx
          if-no-files-found: warn
```

> `vars.ENABLE_GPU_TESTS` is a repository variable that gates this workflow. Set it to `true` in the repo settings once a self-hosted CUDA runner is registered. Until then the workflow runs but the job is skipped — no scheduled noise.

- [ ] **Step 2: Commit**

```powershell
git add .github/workflows/gpu-tests.yml
git commit -m "ci: add GPU tests workflow stub (self-hosted, gated by repo var)"
```

---

### Task 19: Add the release workflow stub

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write the release workflow**

Create file `.github/workflows/release.yml`:

```yaml
name: Release

on:
  workflow_dispatch:
    inputs:
      tag:
        description: 'Release tag (e.g. v0.1.0-alpha). Must be pre-created on main.'
        required: true

jobs:
  publish:
    name: Pack and publish to NuGet
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Checkout tag
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.inputs.tag }}

      - name: Setup .NET 10
        uses: actions/setup-dotnet@v4
        with:
          global-json-file: global.json

      - name: Restore
        run: dotnet restore LLMCompressorSharp.slnx

      - name: Build
        run: dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release

      - name: Pack
        run: >
          dotnet pack LLMCompressorSharp.slnx
          --no-build --configuration Release
          --output artifacts/packages

      - name: Push to NuGet
        env:
          NUGET_API_KEY: ${{ secrets.NUGET_API_KEY }}
        run: >
          dotnet nuget push 'artifacts/packages/*.nupkg'
          --api-key $NUGET_API_KEY
          --source https://api.nuget.org/v3/index.json
          --skip-duplicate
```

> `NUGET_API_KEY` is configured as a repository secret. Until the first 0.1.0-alpha tag exists this workflow is dispatch-only and inert.

- [ ] **Step 2: Commit**

```powershell
git add .github/workflows/release.yml
git commit -m "ci: add manual NuGet release workflow"
```

---

### Task 20: Add MinVer for git-tag-driven versioning

**Files:**
- Modify: `Directory.Build.props` (already lists MinVer in central versions; this task wires it in)

- [ ] **Step 1: Add the MinVer PackageReference to the central build props**

Open `Directory.Build.props` and add this `ItemGroup` after the StyleCop one:

```xml
  <!-- Versioning from git tags via MinVer; applies to library + CLI projects.
       Test projects opt out (see Directory.Build.props at tests/). -->
  <ItemGroup Condition="'$(IsPackable)' != 'false'">
    <PackageReference Include="MinVer" PrivateAssets="all" />
  </ItemGroup>

  <PropertyGroup>
    <MinVerTagPrefix>v</MinVerTagPrefix>
    <MinVerDefaultPreReleaseIdentifiers>alpha.0</MinVerDefaultPreReleaseIdentifiers>
  </PropertyGroup>
```

The opening `<Project>` and existing `<PropertyGroup>` content stays unchanged.

- [ ] **Step 2: Verify the wiring builds cleanly (MinVer runs with no tag yet)**

Run: `dotnet build src/LLMCompressorSharp.Core/LLMCompressorSharp.Core.csproj --configuration Release`
Expected: build succeeds. With no tag in the repo yet, MinVer falls back to `0.0.0-alpha.0.X` (X is the height above the first commit). That's fine — the real version is produced at Task 22 once the tag exists.

If MinVer errors out, confirm `Directory.Build.props` contains the MinVer `ItemGroup` and `PropertyGroup` blocks added in Step 1.

- [ ] **Step 3: Commit the MinVer wiring**

```powershell
git add Directory.Build.props
git commit -m "chore: wire MinVer for git-tag SemVer"
```

---

### Task 21: Full-solution verification

**Files:** (no file changes — full verification of the phase)

- [ ] **Step 1: Clean restore + build + test**

Run:
```powershell
dotnet restore LLMCompressorSharp.slnx
dotnet build LLMCompressorSharp.slnx --no-restore --configuration Release
dotnet test LLMCompressorSharp.slnx --no-build --configuration Release --filter "Category!=Gpu"
```

Expected:
- Restore: no errors
- Build: 0 errors, 0 warnings
- Test: all tests pass (6 `HuggingFaceCacheTests` `[Fact]` + 3 `[Theory]` data rows + 2 `SmokeTests` = **11 tests minimum**)

- [ ] **Step 2: Verify the CLI tool packs and the smoke command works from the packed tool**

Run:
```powershell
dotnet pack src/LLMCompressorSharp.Cli/LLMCompressorSharp.Cli.csproj --configuration Release --output artifacts/packages
dotnet tool install --add-source artifacts/packages --tool-path artifacts/tool LLMCompressorSharp
artifacts/tool/llmc smoke
```

Expected output from `llmc smoke`:
```
LLMCompressorSharp smoke test...
Created TorchSharp tensor shape=[3,3] dtype=Float32
OK
```

If the tool install fails with `error NU1100`, confirm the .nupkg in `artifacts/packages` is named `LLMCompressorSharp.<version>.nupkg` (the package id is `LLMCompressorSharp` not `LLMCompressorSharp.Cli`).

- [ ] **Step 3: Clean up the test install**

Run: `Remove-Item -Recurse -Force artifacts`

- [ ] **Step 4: Commit nothing — this is verification only**

No file changes. Proceed to Task 22.

---

### Task 22: Tag and merge

**Files:** (no file changes — release ritual)

- [ ] **Step 1: Confirm the working tree is clean**

Run: `git status --short`
Expected: empty.

- [ ] **Step 2: Show the bootstrap commit graph for review**

Run: `git log --oneline main..HEAD`
Expected: ~20 commits between `feature/0-repo-bootstrap` and `main`, one per task with conventional-commit prefixes.

- [ ] **Step 3: Open a PR (humans review)**

This is a manual step — push the branch and open a PR via the GitHub UI.

```powershell
git push -u origin feature/0-repo-bootstrap
```

Once the PR is approved and CI is green, merge it.

- [ ] **Step 4: After merge, tag the merge commit on main**

```powershell
git checkout main
git pull
git tag -a v0.0.1-alpha -m "Phase 0 — repo bootstrap"
git push origin v0.0.1-alpha
```

- [ ] **Step 5: Verify Phase 0 is complete**

Open Phase 1 plan (to be written next: `docs/superpowers/plans/2026-06-XX-phase-1-infrastructure.md`).

Phase 0 deliverables checklist:
- ✅ Solution skeleton with 5 projects
- ✅ Central package management
- ✅ StyleCop + .editorconfig (with TorchSharp boundary suppressions)
- ✅ `HuggingFaceCache` with full unit tests (HF-cache-compatible)
- ✅ CLI tool with `version` and `smoke` commands
- ✅ TorchSharp CPU smoke tests
- ✅ CI workflow (CPU on Windows + Ubuntu)
- ✅ GPU test workflow stub
- ✅ Release workflow stub
- ✅ MinVer git-tag SemVer
- ✅ `PR_TO_TORCHSHARP.md` with 5 initial entries
- ✅ Project README
- ✅ First public tag `v0.0.1-alpha`

---

## Self-Review Notes

The plan covers every spec section that targets Phase 0:

- §2.1 Repository Layout — Tasks 6, 7, 8, 9, 10, 12
- §2.2 Project Responsibilities — Tasks 7–10 create the shells; HF cache in Task 13 is the only "real code" Phase 0 ships
- §2.3 Test Model Tiers — Tasks 12+13 (cache path layout); actual model downloads are Phase 3
- §3 TorchSharp Contribution Tracking — Task 15
- §4.1 Solution-level configuration — Tasks 3, 4, 5, 6
- §4.2 Dependency Choice — Tasks 7–10 (csproj structure)
- §4.3 GitHub Actions — Tasks 17, 18, 19
- §4.4 Versioning & Branching — Tasks 1 (branch), 20 (MinVer), 22 (tag)
- §4.5 Coding Conventions — Task 5 (.editorconfig + StyleCop boundary suppressions)
- §5 Build Sequence (Phase 0 row) — entire plan

Phase 0 explicitly does **not** cover:
- Real `Modifier` / `Observer` / `Recipe` types (Phase 1)
- TorchExtensions actual gap-fillers (Phase 1)
- LLaMA architecture (Phase 3)
- Any compression algorithm (Phase 2 onward)

Subsequent phases get their own plans written as we approach them, per the decision-gate model in the spec.

# .NET 10 SDK Tooling — File-Based Apps, dnx, CLI, NuGet

---

## File-Based Apps (Major New Feature)

A single `.cs` file can now be a complete .NET application — **no `.csproj` required**. Native AOT is enabled by default. This is Microsoft's first-party replacement for the `.csx` scripting gap.

### Running

```bash
dotnet run hello.cs          # compile and run
dotnet hello.cs              # shorthand
dotnet run hello.cs -- arg1  # pass arguments
echo 'Console.WriteLine("hi");' | dotnet run -   # stdin
```

### Configuration via `#:` Directives

Directives go at the top of the `.cs` file, before any code:

```csharp
#:package TorchSharp-cpu@0.107.0
#:package YamlDotNet@*
#:project ../MyLibrary/MyLibrary.csproj
#:property PublishAot=false
#:property TargetFramework=net10.0
#:sdk Microsoft.NET.Sdk.Web
```

| Directive | Purpose |
|---|---|
| `#:package Name@version` | Add a NuGet package reference |
| `#:package Name@*` | Use the latest available version |
| `#:project path/to/proj.csproj` | Reference another project |
| `#:property Key=Value` | Set an MSBuild property |
| `#:sdk SDK.Name` | Change the SDK (default: `Microsoft.NET.Sdk`) |

### Publishing

```bash
dotnet publish compress.cs    # produces native AOT executable in artifacts/
```

Opt out of AOT: `#:property PublishAot=false`

### Convert to Project

```bash
dotnet project convert compress.cs    # creates compress.csproj + src/ directory
```

### Shebang (Unix)

```csharp
#!/usr/bin/env -S dotnet --
Console.WriteLine("Hello from a .NET shell script!");
```

### Build Caching

The SDK caches output based on file content, directives, SDK version, and implicit build files — subsequent runs of the same file are instant.

### Limitations

- Single file only (no multi-file scripting without project conversion)
- AOT by default: may break packages relying on reflection (`#:property PublishAot=false` to opt out)
- No interactive REPL (batch compilation only)
- Concurrent instances of the same file can cause build cache contention

---

## `dnx` — dotnet Tool Exec Shorthand

The name `dnx` has been reused (the original DNX from 2015 is long gone) for a new purpose in .NET 10: a shorthand for `dotnet tool exec`.

```bash
dnx <tool-name> [args]
# equivalent to:
dotnet tool exec <tool-name> [args]
```

**One-shot execution:** Downloads and runs a .NET tool without installing it permanently.

```bash
dnx dotnet-ef migrations add Initial
dnx dotnet-format
```

---

## CLI Improvements

### Noun-First Command Aliases

New alternative syntax (both old and new forms work):

```bash
# New (noun-first)
dotnet package add Newtonsoft.Json
dotnet package list
dotnet package remove Newtonsoft.Json
dotnet reference add ../OtherProject/OtherProject.csproj
dotnet reference list
dotnet reference remove ...

# Old (still works)
dotnet add package Newtonsoft.Json
dotnet list package
...
```

### Machine-Readable Command Descriptions

```bash
dotnet --cli-schema         # emit JSON schema for the entire CLI command tree
dotnet build --cli-schema   # emit JSON schema for just the 'build' command
```

### Interactive Mode

`--interactive` is now the **default** in interactive terminal sessions. Use `--interactive false` to disable (e.g., in CI pipelines).

### Tab Completion

Native completion scripts for all major shells:

```bash
dotnet completions script bash      > ~/.local/share/bash-completion/completions/dotnet
dotnet completions script powershell >> $PROFILE
dotnet completions script zsh
dotnet completions script fish
dotnet completions script nushell
```

---

## .NET Tools Enhancements

### Platform-Specific Tools

A single NuGet package can now bundle binaries for multiple RIDs:
- Framework-dependent agnostic/specific
- Self-contained
- Trimmed
- AOT

Include an `any` RID fallback (framework-dependent DLL) alongside platform-specific AOT binaries.

---

## NuGet — Package Pruning

### `PrunePackageReference` (Default for .NET 10 Targets)

Automatically removes packages from the dependency graph that are provided by the .NET runtime itself. Opt-in in .NET 9 SDK; **enabled by default** when targeting .NET 10+.

```xml
<!-- Disable if needed: -->
<RestoreEnablePackagePruning>false</RestoreEnablePackagePruning>
```

**Benefits:**
- Faster restores, less disk usage
- Fewer false positives from NuGet Audit
- Smaller `.deps.json`

**Impact on ML projects:** Packages like `System.Numerics.Vectors`, `System.Memory`, `System.Runtime.CompilerServices.Unsafe` will be pruned from the dependency graph when targeting .NET 10 since they are runtime-provided. Remove explicit `PackageReference` for these (NuGet will warn with NU1510).

---

## MSBuild in-process .NET tasks

.NET MSBuild tasks can now run inside Visual Studio and `msbuild.exe` (previously only worked with `dotnet` CLI), using:

```xml
<UsingTask TaskName="MyTask" AssemblyFile="..." Runtime="NET" TaskFactory="TaskHostFactory" />
```

---

## Container Publishing

Console apps can now create container images without explicitly enabling the feature:

```bash
dotnet publish /t:PublishContainer
```

```xml
<!-- Set image format -->
<ContainerImageFormat>OCI</ContainerImageFormat>
```

---

## Testing

`dotnet test` now natively supports `Microsoft.Testing.Platform` (MTP) without additional flags. Enable via `global.json`:

```json
{
  "test": {
    "runner": "Microsoft.Testing.Platform"
  }
}
```

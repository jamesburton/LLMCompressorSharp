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

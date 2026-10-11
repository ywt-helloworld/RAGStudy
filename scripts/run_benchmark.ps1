param(
    [string]$Config = "config.local.json",
    [string]$Questions = "初赛验证集dev/初赛验证集dev.json",
    [string[]]$Documents = @("初赛验证集dev/初赛规则集rules1.json", "初赛验证集dev/复赛规则集"),
    [string]$Output = "artifacts/dev.answers.json",
    [string]$Index = "",
    [switch]$Lexical,
    [switch]$RetrieveOnly,
    [switch]$RebuildIndex,
    [int]$Limit = 0
)
$ErrorActionPreference = "Stop"
$projectPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectPath
try {
    $exePath = Join-Path $projectPath "build/Release/RAGCLI.exe"
    if (-not (Test-Path -LiteralPath $exePath)) { throw "Build first: cmake --build build --config Release" }
    $modeName = if ($Lexical) { "lexical" } else { "hybrid" }
    if ($Index) {
        $indexPath = $Index
    } else {
        $documentPaths = @($Documents | ForEach-Object {
            if ([System.IO.Path]::IsPathRooted($_)) { [System.IO.Path]::GetFullPath($_) }
            else { [System.IO.Path]::GetFullPath((Join-Path $projectPath $_)) }
        } | Sort-Object)
        $configData = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        $identity = ($documentPaths -join "`n") + "`n" + $configData.models.embedding + "`n" + $configData.embedding.n_ctx + ":" + $configData.embedding.n_batch
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try { $hash = [System.BitConverter]::ToString($hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($identity))).Replace("-", "").Substring(0, 12).ToLowerInvariant() }
        finally { $hasher.Dispose() }
        $indexPath = "artifacts/corpus.$hash.$modeName.json"
    }
    $modeArgs = if ($Lexical) { @("--lexical") } else { @() }
    if ($RebuildIndex -or -not (Test-Path -LiteralPath $indexPath)) {
        & $exePath --config $Config --build @Documents --db $indexPath @modeArgs
        if ($LASTEXITCODE -ne 0) { throw "Ingestion failed" }
    }
    $batchArgs = @("--config", $Config, "--batch", $Questions, "--db", $indexPath, "--output", $Output, "--resume") + $modeArgs
    if ($RetrieveOnly) { $batchArgs += "--retrieve-only" }
    if ($Limit -gt 0) { $batchArgs += @("--limit", "$Limit") }
    & $exePath @batchArgs
    if ($LASTEXITCODE -ne 0) { throw "Batch run failed; inspect results and rerun to retry failed rows" }
} finally {
    Pop-Location
}

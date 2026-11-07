param(
    [Parameter(Mandatory=$true)][string]$TestBin,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][string]$SourceRevision,
    [int]$Seconds = 30,
    [int]$Bars = 1024,
    [int]$Repetitions = 3
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite paired evidence' }
if ($Seconds -lt 1 -or $Seconds -gt 120 -or $Bars -lt 16 -or $Bars -gt 1024 -or $Repetitions -lt 1 -or $Repetitions -gt 3) { throw 'Invalid bounded campaign' }
$binary = (Resolve-Path -LiteralPath $TestBin).Path
if ((Split-Path -Leaf $binary) -notmatch '^moosik-[a-z0-9]+\.exe$') { throw 'Expected the exact release-test executable, not the player' }
New-Item -ItemType Directory -Path $OutputDir -ErrorAction Stop | Out-Null
Get-FileHash -LiteralPath $binary -Algorithm SHA256 | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'test-binary.json')
$arms = @('aslt1','cqt12','aslt9','cqt48','cqt192')
$expected = @{}
for ($rep = 1; $rep -le $Repetitions; $rep++) {
    $order = 0..($arms.Length - 1) | ForEach-Object { $arms[($_ + $rep - 1) % $arms.Length] }
    foreach ($arm in $order) {
        $contenders = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(moosik.*|llama.*|cargo|rustc|cqt-speed-probe)\.exe$' }
        if ($contenders) { throw "Contending process present; nothing killed: $($contenders.ProcessId)" }
        $env:MOOSIK_GPU = '0'
        $env:MOOSIK_CQT_PAIR_ARM = $arm
        $env:MOOSIK_CQT_PAIR_SECONDS = [string]$Seconds
        $env:MOOSIK_CQT_PAIR_BARS = [string]$Bars
        $env:MOOSIK_CQT_PAIR_SR = '48000'
        $env:MOOSIK_CQT_PAIR_WORKERS = '6'
        $env:MOOSIK_CQT_PAIR_MODE = 'joint'
        $env:MOOSIK_CQT_PAIR_SOURCE_REV = $SourceRevision
        $lines = & $binary 'spectrum::cqt_pair_bench::production_pair' --exact --ignored --nocapture --test-threads=1 2>&1
        $exitCode = $LASTEXITCODE
        $lines | ForEach-Object { $_.ToString() } | Set-Content -LiteralPath (Join-Path $OutputDir "$rep-$arm.log")
        if ($exitCode -ne 0) { throw "Paired run failed ($exitCode): $rep/$arm; evidence retained" }
        $jsonLines = @($lines | ForEach-Object {
            $line = $_.ToString()
            $marker = $line.IndexOf('CQT_PAIR {', [StringComparison]::Ordinal)
            if ($marker -ge 0) { $line.Substring($marker + 9) }
        })
        if ($jsonLines.Count -ne 1) { throw 'Exactly one result is required' }
        $record = $jsonLines[0] | ConvertFrom-Json
        if ($record.arm -ne $arm -or $record.bars -ne $Bars -or $record.input_frames -ne ($Seconds * 48000) -or
            $record.selected_workers -ne 6 -or !$record.disk_readback_all_codes_equal -or
            $record.mix_file_bytes -le 0 -or $record.lr_file_bytes -le 0 -or $record.wall_seconds -le 0) { throw 'Invalid paired result' }
        $identity = $record.code_plane_sha256 -join ','
        if ($expected.ContainsKey($arm) -and $expected[$arm] -ne $identity) { throw "Repeated output changed: $arm" }
        $expected[$arm] = $identity
        $record | Add-Member -NotePropertyName repetition -NotePropertyValue $rep
        $record | Add-Member -NotePropertyName recorded_at -NotePropertyValue (Get-Date -Format o)
        $record | ConvertTo-Json -Depth 12 -Compress | Add-Content -LiteralPath (Join-Path $OutputDir 'results.jsonl')
        Write-Output "$rep/$arm wall=$($record.wall_seconds) cache=$($record.mix_file_bytes + $record.lr_file_bytes)"
    }
}

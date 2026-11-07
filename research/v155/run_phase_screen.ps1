param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][string]$SourceRevision
)
$ErrorActionPreference = 'Stop'
if ($SourceRevision -notmatch '^[0-9a-f]{40}$') { throw 'Expected a full source revision' }
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite phase evidence' }
$binary = (Resolve-Path -LiteralPath $Probe).Path
New-Item -ItemType Directory -Path $OutputDir -ErrorAction Stop | Out-Null
Get-FileHash -LiteralPath $binary -Algorithm SHA256 | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'probe.json')
$previous = $env:MOOSIK_CQT_PHASES
try {
    foreach ($bpo in @(48, 192)) {
        $expected = $null
        # Two coarse profiles around an uninstrumented observation. The single
        # off run is an overhead screen, not a statistically robust speed gate.
        $ordinal = 0
        foreach ($enabled in @($true, $false, $true)) {
            $ordinal++
            $contenders = @(Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match '^(moosik.*|llama.*|cargo|rustc|cqt-(speed|phase).*)\.exe$'
            })
            if ($contenders.Count) { throw "Contender present; nothing stopped: $($contenders.ProcessId)" }
            $env:MOOSIK_CQT_PHASES = if ($enabled) { '1' } else { '0' }
            $lines = @(& $binary pool6 48000 30 1024 joint 24000 $bpo 2>&1 | ForEach-Object { $_.ToString() })
            $exitCode = $LASTEXITCODE
            $lines | Set-Content -LiteralPath (Join-Path $OutputDir "$bpo-$ordinal.log")
            if ($exitCode -ne 0) { throw "Phase probe failed: $bpo/$ordinal ($exitCode)" }
            $outputs = @($lines | Where-Object { $_.StartsWith('{') })
            $phases = @($lines | Where-Object { $_.StartsWith('CQT_PHASES ') })
            if ($outputs.Count -ne 1 -or $phases.Count -ne [int]$enabled) { throw 'Unexpected probe/phase record count' }
            $r = $outputs[0] | ConvertFrom-Json
            if ($r.arm -ne 'pool6' -or $r.sr -ne 48000 -or $r.seconds -ne 30 -or
                $r.bars -ne 1024 -or $r.joint -ne $true -or $r.max_hz -ne 24000 -or
                $r.bpo -ne $bpo -or $r.frames -ne 5400 -or $r.stats.max_workers -gt 6 -or
                $r.stats.workspace_bytes -gt (256 * 1024 * 1024) -or
                [double]::IsNaN($r.wall_seconds) -or [double]::IsInfinity($r.wall_seconds) -or $r.wall_seconds -le 0) {
                throw 'Unexpected geometry, timing or workspace result'
            }
            $identity = $r.codes_fnv64 -join ','
            if ($null -ne $expected -and $identity -ne $expected) { throw 'Profile on/off changed code identity aid' }
            $expected = $identity
            $phase = if ($enabled) { $phases[0].Substring('CQT_PHASES '.Length) | ConvertFrom-Json } else { $null }
            [ordered]@{
                source_revision = $SourceRevision; bpo = $bpo; ordinal = $ordinal
                profile_enabled = $enabled; result = $r; phases = $phase
            } | ConvertTo-Json -Depth 20 -Compress | Add-Content -LiteralPath (Join-Path $OutputDir 'results.jsonl')
            "BPO$bpo run$ordinal phases=$enabled wall=$($r.wall_seconds)"
        }
    }
} finally {
    $env:MOOSIK_CQT_PHASES = $previous
}

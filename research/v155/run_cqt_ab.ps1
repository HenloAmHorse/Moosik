param(
    [Parameter(Mandatory=$true)][string]$Baseline,
    [Parameter(Mandatory=$true)][string]$Candidate,
    [Parameter(Mandatory=$true)][string]$BaselineRevision,
    [Parameter(Mandatory=$true)][string]$CandidateRevision,
    [Parameter(Mandatory=$true)][string]$OutputDir
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDir) { throw 'Refusing to overwrite A/B evidence' }
if ($BaselineRevision -notmatch '^[0-9a-f]{40}$' -or $CandidateRevision -notmatch '^[0-9a-f]{40}$') { throw 'Full revisions required' }
$binaries = @{ baseline = (Resolve-Path -LiteralPath $Baseline).Path; candidate = (Resolve-Path -LiteralPath $Candidate).Path }
if ($binaries.baseline -eq $binaries.candidate) { throw 'Two separately frozen executables required' }
New-Item -ItemType Directory -Path $OutputDir -ErrorAction Stop | Out-Null
$provenance = foreach ($arm in @('baseline','candidate')) {
    [ordered]@{arm=$arm;revision=if($arm -eq 'baseline'){$BaselineRevision}else{$CandidateRevision};path=$binaries[$arm];sha256=(Get-FileHash -LiteralPath $binaries[$arm] -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$provenance | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'probes.json')
$previous = $env:MOOSIK_CQT_PHASES
$expected = @{}
try {
    $env:MOOSIK_CQT_PHASES = '0'
    foreach ($bpo in @(12,48,192)) {
        foreach ($rep in 1..3) {
            $order = if ($rep % 2) { @('baseline','candidate') } else { @('candidate','baseline') }
            foreach ($arm in $order) {
                $contenders = @(Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -match '^(moosik.*|llama.*|cargo|rustc|cqt-(speed|phase).*)\.exe$'
                })
                if ($contenders.Count) { throw "Contender present; nothing stopped: $($contenders.ProcessId)" }
                $lines = @(& $binaries[$arm] pool6 48000 30 1024 joint 24000 $bpo 2>&1 | ForEach-Object { $_.ToString() })
                $exitCode = $LASTEXITCODE
                $lines | Set-Content -LiteralPath (Join-Path $OutputDir "$bpo-$rep-$arm.log")
                if ($exitCode -ne 0) { throw "A/B run failed: $bpo/$rep/$arm ($exitCode)" }
                if ($lines.Count -ne 1 -or !$lines[0].StartsWith('{')) { throw 'Expected exactly one unprofiled result' }
                $r = $lines[0] | ConvertFrom-Json
                if ($r.arm -ne 'pool6' -or $r.sr -ne 48000 -or $r.seconds -ne 30 -or
                    $r.bars -ne 1024 -or $r.joint -ne $true -or $r.max_hz -ne 24000 -or
                    $r.bpo -ne $bpo -or $r.frames -ne 5400 -or $r.stats.max_workers -gt 6 -or
                    $r.stats.workspace_bytes -gt (256 * 1024 * 1024) -or
                    [double]::IsNaN($r.wall_seconds) -or [double]::IsInfinity($r.wall_seconds) -or $r.wall_seconds -le 0 -or
                    @($r.codes_fnv64).Count -ne 3 -or @($r.codes_fnv64 | Where-Object { $_ -notmatch '^[0-9a-f]{16}$' }).Count) {
                    throw 'Unexpected geometry, identity aid, timing or workspace'
                }
                $identity = $r.codes_fnv64 -join ','
                if ($expected.ContainsKey($bpo) -and $expected[$bpo] -ne $identity) { throw 'Cross-arm code identity aid changed' }
                $expected[$bpo] = $identity
                [ordered]@{arm=$arm;bpo=$bpo;repetition=$rep;recorded_at=(Get-Date -Format o);result=$r} |
                    ConvertTo-Json -Depth 10 -Compress | Add-Content -LiteralPath (Join-Path $OutputDir 'results.jsonl')
                "BPO$bpo rep$rep $arm wall=$($r.wall_seconds)"
            }
        }
    }
} finally {
    $env:MOOSIK_CQT_PHASES = $previous
}

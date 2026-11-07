param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite evidence: $Output" }
# 1024 is the product's maximum CQT column count. ASLT's 2x grid does not
# apply to genuine CQT; 2048 below is explicitly an extra core stress case.
$cases = @(
    @{sr=48000;seconds=120;bars=1024;max_hz=24000;scope='production_geometry'},
    @{sr=96000;seconds=30;bars=1024;max_hz=24000;scope='production_geometry'},
    @{sr=48000;seconds=30;bars=2048;max_hz=20000;scope='core_stress_not_product'}
)
foreach ($case in $cases) {
    $expected = $null
    foreach ($rep in 1..3) {
        $arms = if ($rep % 2) { @('frozen','serial','pool6') } else { @('pool6','serial','frozen') }
        foreach ($arm in $arms) {
            $contenders = @(Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match '^(moosik.*|llama.*|cargo|rustc|cqt-speed-probe)\.exe$'
            })
            if ($contenders.Count) { throw "Contending player/model/build/probe present; nothing killed: $($contenders.ProcessId)" }
            $row = & $Probe $arm $case.sr $case.seconds $case.bars joint $case.max_hz
            if ($LASTEXITCODE -ne 0) { throw "Probe failed: $arm $($case.sr) $($case.seconds)" }
            $contenders = @(Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match '^(moosik.*|llama.*|cargo|rustc|cqt-speed-probe)\.exe$'
            })
            if ($contenders.Count) {
                Write-Output $row
                throw "Contender appeared during run; record not accepted and nothing killed: $($contenders.ProcessId)"
            }
            $record = $row | ConvertFrom-Json -ErrorAction Stop
            $hi = [math]::Min($case.max_hz, $case.sr / 2.0)
            if ($record.arm -ne $arm -or $record.sr -ne $case.sr -or
                $record.seconds -ne $case.seconds -or $record.bars -ne $case.bars -or
                -not $record.joint -or $record.min_hz -ne 20 -or $record.max_hz -ne $hi -or
                $record.max_hz_requested -ne $case.max_hz -or $record.bpo -ne 48 -or
                $record.window -ne 'Blackman' -or $record.fps -ne 180 -or
                $record.frames -ne ($case.seconds * 180) -or
                [double]::IsNaN($record.wall_seconds) -or [double]::IsInfinity($record.wall_seconds) -or
                $record.wall_seconds -le 0 -or
                $record.stats.workspace_bytes -gt (256 * 1024 * 1024) -or
                $record.codes_fnv64.Count -ne 3) { throw 'Invalid extended probe record' }
            $identity = $record.codes_fnv64 -join ','
            if ($null -eq $expected) { $expected = $identity }
            if ($identity -ne $expected) { throw "Code identity changed: $arm $($case.sr) $($case.seconds)" }
            $record | Add-Member -NotePropertyName repetition -NotePropertyValue $rep
            $record | Add-Member -NotePropertyName recorded_at -NotePropertyValue (Get-Date -Format o)
            $record | Add-Member -NotePropertyName geometry_scope -NotePropertyValue $case.scope
            $record | ConvertTo-Json -Depth 6 -Compress | Add-Content -LiteralPath $Output
        }
    }
}

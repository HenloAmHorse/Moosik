param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite evidence: $Output" }
foreach ($case in @(@(44100,2),@(48000,2),@(96000,2),@(48000,10),@(48000,30))) {
    foreach ($mode in @('mono','joint')) {
        $expected = $null
        foreach ($rep in 1..3) {
            $arms = if ($rep % 2) { @('frozen','serial','pool6') } else { @('pool6','serial','frozen') }
            foreach ($arm in $arms) {
                $contenders = Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -match '^(moosik.*|llama.*|cargo|rustc)\.exe$'
                }
                if ($contenders) { throw "Contending player/model/build present; nothing killed: $($contenders.ProcessId)" }
                $row = & $Probe $arm $case[0] $case[1] 1024 $mode
                if ($LASTEXITCODE -ne 0) { throw "Probe failed: $arm $case $mode" }
                $record = $row | ConvertFrom-Json -ErrorAction Stop
                if ($record.arm -ne $arm -or $record.sr -ne $case[0] -or $record.seconds -ne $case[1] -or
                    $record.bars -ne 1024 -or $record.joint -ne ($mode -eq 'joint') -or
                    $record.frames -ne ($case[1] * 180) -or $record.wall_seconds -le 0 -or
                    $record.stats.workspace_bytes -gt (256 * 1024 * 1024)) { throw 'Invalid probe record' }
                $identity = $record.codes_fnv64 -join ','
                if ($null -eq $expected) { $expected = $identity }
                if ($identity -ne $expected) { throw "Code identity changed: $arm $case $mode" }
                $record | Add-Member -NotePropertyName repetition -NotePropertyValue $rep
                $record | Add-Member -NotePropertyName recorded_at -NotePropertyValue (Get-Date -Format o)
                $record | ConvertTo-Json -Depth 6 -Compress | Add-Content -LiteralPath $Output
            }
        }
    }
}

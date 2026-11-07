param([Parameter(Mandatory=$true)][string]$EvidenceDir)
$ErrorActionPreference = 'Stop'
$records = @(Get-Content -LiteralPath (Join-Path $EvidenceDir 'results.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$arms = @('aslt1', 'cqt12', 'aslt9', 'cqt48', 'cqt192')
$inputHash = '3d7d03fd21cfffe10f488428efa8eecfe5c09a1176413a5c252a0bed22461746'
$revision = '833871f9fb56a1aa7f7aa8f1888a512b7bbcea19'
if ($records.Count -ne 15) { throw 'Expected exactly fifteen measured records' }
$seen = @{}
for ($index = 0; $index -lt $records.Count; $index++) {
    $r = $records[$index]
    $rep = [int][math]::Floor($index / 5) + 1
    $arm = $arms[($index % 5 + $rep - 1) % 5]
    if ($r.schema -ne 'moosik-v155-pair-v1' -or $r.arm -ne $arm -or $r.repetition -ne $rep -or
        $r.runner_source_revision -ne $revision -or $r.build_label -ne 'prereview 833871f' -or
        $r.input_sha256 -ne $inputHash -or $r.input_sample_rate -ne 48000 -or
        $r.input_frames -ne 1440000 -or $r.bars -ne 1024 -or $r.audio_seconds -ne 30 -or
        $r.nominal_frame_rate -ne 180 -or $r.mode -ne 'joint' -or
        $r.selected_workers -ne 6 -or $r.observed_pool_workers -ne 6 -or
        $r.gpu_disabled -ne $true -or $r.disk_readback_all_codes_equal -ne $true -or
        $r.comparison -ne 'selectable-settings-not-matched-response' -or
        $r.process_peak_scope -ne 'lifetime-including-setup-not-analysis-only' -or
        $r.timer_scope -ne 'production-decode-statistics-analysis-encode-publication; excludes-fixture-pool-readback') {
        throw "Unexpected result identity/configuration at record $index"
    }
    foreach ($value in @($r.wall_seconds, $r.cpu_seconds, $r.actual_frame_rate)) {
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) { throw 'Invalid finite timing/rate' }
    }
    if ($r.mix_file_bytes -le 0 -or $r.lr_file_bytes -le 0 -or
        $r.resources_after.peak_working_set_bytes -lt $r.resources_after.working_set_bytes -or
        $r.resources_after.peak_commit_bytes -lt $r.resources_after.commit_bytes) { throw 'Invalid byte measurement' }
    $isAslt = $arm.StartsWith('aslt')
    $expectedFrames = if ($isAslt) { 5393 } else { 5400 }
    $expectedRate = if ($isAslt) { 48000.0 / 267.0 } else { 180.0 }
    if ($r.output_frames -ne $expectedFrames -or [math]::Abs($r.actual_frame_rate - $expectedRate) -gt 1e-10) { throw 'Unexpected frame clock' }
    if (@($r.code_plane_sha256).Count -ne 3 -or @($r.code_plane_sha256 | Where-Object { $_ -notmatch '^[0-9a-f]{64}$' }).Count -ne 0) { throw 'Invalid three-plane digest' }
    # Join per record, not after PowerShell has flattened a collection of arrays.
    $identity = $r.code_plane_sha256 -join ','
    if ($seen.ContainsKey($arm) -and $seen[$arm] -ne $identity) { throw "Repeated code planes changed: $arm" }
    $seen[$arm] = $identity
}
$summary = foreach ($arm in $arms) {
    $group = @($records | Where-Object arm -eq $arm)
    if ($group.Count -ne 3) { throw "Missing repetitions: $arm" }
    $wall = @($group.wall_seconds | Sort-Object)
    $cpu = @($group.cpu_seconds | Sort-Object)
    $sizes = @($group | ForEach-Object { $_.mix_file_bytes + $_.lr_file_bytes } | Sort-Object)
    [ordered]@{
        arm = $arm
        wall_seconds_min = $wall[0]; wall_seconds_median = $wall[1]; wall_seconds_max = $wall[2]
        cpu_seconds_median = $cpu[1]
        cache_file_bytes_min = $sizes[0]; cache_file_bytes_max = $sizes[2]
        process_peak_working_set_bytes_max = ($group.resources_after.peak_working_set_bytes | Measure-Object -Maximum).Maximum
        process_peak_commit_bytes_max = ($group.resources_after.peak_commit_bytes | Measure-Object -Maximum).Maximum
        code_plane_sha256 = $group[0].code_plane_sha256
    }
}
[ordered]@{
    schema = 'moosik-v155-pair-summary-v1'
    measured_runs = $records.Count
    source_revision = $revision
    input_sha256 = $inputHash
    scope = 'one-synthetic-workload; selectable-settings-not-matched-response; optimized-test-binary-not-GUI'
    arms = @($summary)
} | ConvertTo-Json -Depth 8

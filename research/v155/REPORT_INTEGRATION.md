# 1.5.5 — genuine CQT integration and prereview

Historical first-prereview report. The next candidate's labelled live preview
and bounded requested-output acceleration are described in
[REPORT_SPEED_PREVIEW.md](REPORT_SPEED_PREVIEW.md); f9a6999 itself is unchanged.

2026-10-19, Windows x86_64 MSVC, Rust/Cargo 1.94.0, i7-9700F.
Implementation: `f9a69999d8d97d032e2f3c07ecc204a65ed50369` on
`codex/v155-cqt-research`. Base release `330b8ac` (1.5.4).
Recovery `refs/recovery/codex-v155-integration-start-20261019` pins `f0b97c0`.

Status: **prereview frozen, ready for the owner's normal-use check**. This is a
development candidate, not a published release. All owned build/test jobs exited.
Main/public repository remain outside the authorized write scope.

## Product delta

The previously isolated CQT bank is now a product analysis method, with a
separate enum payload and persistent config, not a renamed FFT weighting.
`src/spectrum/cqt.rs` owns the f64 grouped overlap-save core; `cqt_pipeline.rs`
adapts it to existing decode/statistics/cache/publication; `cqt_ui.rs` owns the
new controls. Existing monolithic spectrum code is only wired at its seams.

The definition and limits are authoritative in [docs/CQT.md](../../docs/CQT.md).
In particular:

- Full centered inverse-frequency windows, four selectable window shapes,
  no capped bass, sparse threshold, decimation or recursive approximation.
- Support BPO is independent of the log display sampling. Default48BPO,
  Blackman,180fps is not a claim of universally best quality. Full20Hz support
  is approximately3.44s: frequency selectivity costs temporal localization.
- Row k is labelled k/fps; sample center floor(k*sr/fps), including fractional
  hops. Fresh results, reload, repeat, hover and Whole track use that clock.
- Joint CQT does two convolutions and forms Mix from the complex mean before
  magnitudes. Kernel transforms and input transforms are shared where applicable.
- Returned planes use the production `aslt_code` mapping, not a new test-only
  quantizer. These are display codes, not dBFS/phase/invertible/pro-analysis data.
- Preprocess only, with no temporary FFT result labelled True CQT. UI disables
  live mode and hides irrelevant FFT/ASLT controls; public live entry points
  independently refuse a True-CQT FFT substitute.
- New CQT identities contain revision/BPO/window/fps/clock/grid policies. No
  global namespace/schema bump. Literal old FFT/ASLT record tests stay exact.
- Reuse existing store publication/pins/cancellation; no cache rewrite. Existing
  source matching and cleanup guarantees/limitations remain the same.

The actual live FFT workspace optimization from `8c08d4d` is also in this
candidate. It removes four measured stage allocations after warmup and preserves
tested complex/norm bits; see [REPORT_BATCH2.md](REPORT_BATCH2.md). No new ASLT
arithmetic, translation, audio-backend or dependency changes in this integration.

## Correctness evidence

- Independent direct f64 reference re-derives Q with exp_m1, uses conventional
  window coordinates and compensated real/imaginary sums. It does not call the
  optimized kernel/window builder. Actual production complex coefficients are
  observed before magnitude (wrong phase/sign cannot hide behind equal levels).
- Complex responses within3e-12 absolute tolerance on tested fixtures; zero
  differing production codes on the tested44.1/48/96kHz, four-window, silence,
  impulse, tone, noise and sweep/gated-tone configurations. Not a proof over all
  inputs or a screenshot/pixel-comparison claim.
- Fractional179.5fps and multi-block edges; full-window gain/no edge boost;
  arbitrary stereo and exact antiphase cancellation; homogeneous actual work
  counters proving two channel convolution sets rather than a third Mix pass.
- Actual FFT scratch controls workspace admission. Batch-size changes leave
  tested codes unchanged. Resource refusal and callback cancellation return no
  partial core output. The product tests force pre-work abort and resource
  refusal through production job/service entry points and check no cache head.
- Private generated stereo WAV through on_play/start_preprocess/service, exact
 180fps at48kHz, cache publication, fresh channels, reload without a worker,
  repeat retaining the same frames/revision, and Whole track snapshot metadata.
- Headless egui click on the painted method selector checks actual mode/style
  changes, remembered config, absent irrelevant controls and preserved ASLT scale.
  Profile tests cover old Cqt, new active/inactive configs, invalid sanitization
  and valid clocks outside the UI slider's normal edit range.
- CQT-only Nyquist-clamped display labels/ISO weights/EQ geometry and f64-derived
  Whole track center labels; old-method geometry remains unchanged.

These are generated/headless tests, not a desktop audio-device or visual-quality
approval. Use [OWNER_CHECKLIST_zh-TW.md](OWNER_CHECKLIST_zh-TW.md) for that boundary.

## Gates

Local raw evidence: `C:/Users/PC/Programming/moosik-v155-home/integration/`.

- Debug filter `cargo test --locked --offline cqt -- --nocapture`:
  **26 passed /0 failed /1 ignored**, after final semantic changes. The ignored
  item is the explicit bounded performance screen.
- `cargo clippy --locked --offline --all-targets --all-features`: exit0;
  **28 diagnostic-message multiset identical** to the batch2/release baseline,
  not merely an equal count. `warnings-final.txt` and both raw logs retained.
- `cargo test --release --locked --offline --all-targets` (default features):
  **1386 passed / 0 failed / 41 ignored**, 32.08 s execution on committed code.
  Do not compare that count arithmetically with batch2's all-feature command:
  diagnostic-feature conditions change which tests are ignored.
- `cargo check --locked --offline --no-default-features`: exit0, five existing
  warnings (new unused result/stat warnings eliminated before this final run).
- `cargo build --release --locked --offline` (normal/default features): exit0,
  one pre-existing GPU-method dead-code warning. Label `prereview f9a6999`.
  PE FileVersion `1.5.5`, ProductVersion `1.5.5 prereview f9a6999` verified
  before freezing; no stale target executable copied after a failed build.
- `git diff --check` clean. Only new modules formatted; existing global rustfmt
  waiver is not redefined or silently repaired.

## Cost screen

The ignored test includes layout, f64 FFT planning, kernels, convolution, mapping
and code allocation. Excludes signal creation, decoding/statistics, cache I/O,
display/UI and device work. Single observations, no confidence intervals or
before/after product speedup claim. Runs sequentially, not during another build.

An initial pre-final48kHz/1024bars/48BPO/180fps two-second screen observed3.233s
mono,4.630s joint. Left codes matched the mono result exactly. Fixed setup is
large on such a short signal; this is not a whole-track throughput ratio.
The final matrix on committed code, normal/default release test build:

| Sample rate | Signal | Mono | Joint Mix+L+R |
|---|---:|---:|---:|
| 44.1 kHz | 2 s | 2.897 s | 4.134 s |
| 48 kHz | 2 s | 3.308 s | 4.544 s |
| 96 kHz | 2 s | 7.009 s | 10.136 s |
| 48 kHz | 10 s | 7.967 s | 13.553 s |

All use 1024 display samples, 48 support BPO, Blackman, 20 Hz–20 kHz and180fps.
Every joint Left plane is code-identical to its corresponding mono run; actual
column counts are1024/2048, not a fabricated intended-work count. Final test
completed8 timing rows in53.60s. Raw `cqt-matrix-final.log` includes FFT call
counts, explicit payloads and deterministic non-cryptographic code digests.
Source inspection before this matrix found only the owned cargo/test jobs, no
owner player or local model; no concurrent build ran during the measurements.

**Performance limit:** the ten-second joint case still exceeds its audio duration.
This first full-support f64 single-worker CQT is not sold as an already optimized
stereo-throughput replacement for ASLT. Short high-frequency kernels doing dense
full inverses, and the large fixed kernel preparation, are concrete next profiling
candidates. Direct/pruned/adaptive routing needs its own numerical-route/code gates;
do not use this result to weaken output requirements or silently change defaults.

No hard whole-song deadline or claim that True CQT is faster than ASLT/FFT.
Runtime work is reported as kernel/input/inverse FFT counts separately, not
summed into a fictitious operation total.256MiB is explicit dense payload, not
measured peak process RAM; decoded audio/planner/allocator overhead is excluded.

## Corrections encountered

Initial root adapter compilation had a logging macro format error and the
frozen old-FFT reference match needed an explicit unreachable new-method arm.
Neither was shipped or hidden. The initial resource fixture assumed zero FFT
scratch; it failed and was repaired using the real plan's requirement. First
clippy found two unused result/stat fields; production now consumes and logs
them instead of adding dead-code allowances. Read-only cross-review caught the
ordinary-view Nyquist axis mismatch and a missed FFT-control visibility branch;
both received integration fixes/tests. Whole track centers now derive from the
same f64 expression as the producer, rounded only for the existing f32 label API.

## Deliberate limits / next boundary

Single-worker CPU CQT. No forced GPU, multirate/sparse/pruned CQT or causal live
variant. Explicit payload limits/refusal, not full RSS governance. Cancellation
does not interrupt an individual rustfft call, allocation or plane zero-fill.
Existing cache format additionally caps500,000 rows; the adapter refuses that
before starting the core. No retained phase/chord/spatial feature is implied.
No Linux/hardware/desktop pixel/soak campaign this round; no soak for its own sake.

Keep this candidate isolated until the owner's normal-use check. No merge,
push, tag, release, public-repository operation or owner-profile mutation by the
research process. Tests use generated private directories and suppress profile
writes. The exe itself is a normal player and will save settings if the owner
runs it; the checklist recommends backing up spectrum.json for old-exe rollback.

## Frozen artifact / protected state

`C:/Users/PC/Programming/moosik-v155-home/artifacts/Moosik-1.5.5-prereview-f9a6999.exe`

- 67,597,312 bytes.
- SHA-256 `7be7555cc2560cff98c21bcc3e31b3fd7fb1994d9633d217fce312bd786c981f`.
- Frozen from implementation commit `f9a6999`; this later report-only commit does
  not change product sources, package metadata or the frozen executable.
- Checklist copied alongside it. The executable was not launched on the owner's
  profile. No downloaded model, runtime, GPU or audio-device test this round.
- Final local main/origin-main `330b8ac`; shared checkout HEAD `71be39b` and
  status count15; original stash preserved. No remote operation; public repo
  never contacted. Candidate code commit has no tag. Approximately89GiB free.
- Sessions35734/84185 (targeted debug),44775 (initial lint/check),67833
  (full release/matrix/final check) and75325 (build/freeze) exited. Initial
  failed compile/test attempts and the short pre-final screen are described
  above; they are not counted as clean final gates.

The final local evidence manifest is
`C:/Users/PC/Programming/moosik-v155-home/MANIFEST_INTEGRATION.json`; no compiler
cache, model weights or old release assets are included in that manifest.

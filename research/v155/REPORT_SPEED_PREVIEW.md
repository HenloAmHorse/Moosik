# 1.5.5 — exact requested-output CQT and labelled live transition

Subsequent timing: [measured addendum](REPORT_TIMING.md), 2026-10-20 after
the owner closed the player. This software-freeze report remains historical.

2026-10-20, Windows x86_64 MSVC, Rust/Cargo 1.94.0, i7-9700F.
Starting HEAD8545b17; recovery codex-v155-speed-preview-start-20261019.
This report supersedes the no-preview/single-worker statements in
REPORT_INTEGRATION.md for the new candidate only. The old f9a6999 executable
and its evidence remain frozen. No merge, push, tag, release or public action.

## Status

This software batch is complete: gates passed and the new prereview is frozen.
It is ready for the owner's normal-use check, not authorized for release.
Formal speed comparisons have NOT run: the owner's f9a6999 player
(PID1020, started2026-10-19 23:00:36) remained open throughout this round.
It was not closed, paused or otherwise controlled. Do not treat correctness
test duration or theoretical work reduction as a measured speedup.

## Product changes

1. Pending True CQT supplies an explicitly labelled LIVE FFT PREVIEW on ordinary
   plots. It uses existing FFT constant-Q weighting, log display sampling and
   the live PCM rate/Nyquist, not the pending analysis's possibly different
   sample rate. No usable PCM means an honest unavailable status, not an old
   frozen picture. Whole track never turns a live FFT frame into stored CQT.
2. Live FFT preview size/window controls are separate from CQT identity and do
   not restart CQT. Completed CQT clears preview/history and uses the CQT clock
   and frequency grid. Loaded CQT cannot be overwritten through the live entry
   points. Production-entry tests hold a real worker, show FFT, then install CQT.
3. Same CQT FFT planner, sizes, group-wide half-width, block starts and source
   transforms. Only requested inverse outputs are evaluated where supported.
   Unsupported/dense/duplicate schedules or optional workspace refusal retain
   full execution; there is no sparse threshold, downsampling or shorter filter.
4. f64 AVX256/512 requested leaves mirror the original full graph's operations,
   constants, FMA order and lane addresses. Existing full bodies and all f32
   code are unchanged. Other leaves and requests outside1..4 use full+gather.
5. Per-filter inverses can run on the existing owner-selected analysis pool.
   Source, slot products/scratch/requests, coefficient slabs and kernel batch
   are admitted together under256MiB explicit vector payload. These replace,
   not accompany, the serial workspace. No implicit global-pool work.
6. Window values are computed once and temporarily retained in the already
   charged kernel vector. Sum order, normalization and carrier arithmetic stay
   unchanged. FFT/ASLT analysis semantics/cache identities are unchanged.

## Exactness evidence and its boundary

- Independent frozen f9a6999 source is retained in frozen_cqt_f9a.rs; a SHA256
  prefix guard pins its original body. The appended test-only accessor exposes
  that body's own coefficient observer. It does not call candidate helpers.
-24cases:44.1/48/96kHz, four windows, mono/joint, fractional clocks and arbitrary
  or antiphase stereo compare normalized complex real/imaginary **bits** and
  every shipped display code against that frozen body.
- Separate48-configuration requested-vs-whole matrix, plus independent direct
  f64 definition checks retained from the first prereview.
- Explicit1/2/6-thread pools and2/8/64MiB budgets compare complex bits and code
  planes against frozen code. Tight budgets really exercise serial/batched
  fallback; ample budgets assert the parallel path was used. Helper tests cover
  uneven batches, duplicate clocks, cancellation-hidden partial slabs and reuse.
- Vendor leaf gate: singleton positions on12fixtures/both directions,
  mixed-row sets, old Mul product/in-place product, pre-write argument failures,
  and326,912sorted pairs. Finite/signed-zero/infinity bits and NaN-ness are
  compared; NaN payload/sign and FP status flags are outside the API contract.
- Wrong-lane production mutation was rejected by the singleton gate. It was
  restored before all final gates. This is not a proof for every floating-point
  input or every unsupported architecture. No screenshot/pixel gate was run.
  Equal stored codes and unchanged geometry support display equality after
  analysis; the pending preview deliberately changes the old blank behavior.

## Measurements and reproduction

No new throughput percentage or lower process-RAM claim is established here.
The256MiB figure is explicit vector payload, not RSS: decoded inputs, allocator
overhead, rustfft plans, small metadata and Rayon internals are excluded.

research/v155/speed_probe provides a standalone release comparator with frozen,
serial requested-output and explicit1/2/6-worker arms. Its timer includes plans,
kernels, transforms, code mapping and returned planes; fixture generation and
pool construction are excluded. Decode, cache IO, render, playback and model
work are excluded. FNV code digests are identity aids, not proofs; the in-app
gates compare complete code/complex arrays.
The probe's release profile matches the app's opt-level3/thinLTO/one-codegen-unit/
panic-abort settings; the first build before this profile correction was never
used for timing. The leaf gate matches optimization/LTO/codegen settings (Cargo
test does not use panic-abort).
The probe also uses the app's2024edition; its first rebuild after the core's
let-chain cleanup caught and corrected the old2021manifest before any timings.

The run.ps1 campaign alternates order over3repetitions,44.1/48/96kHz2s and
48kHz10/30s,1024columns,48BPO/Blackman/180fps,mono/joint. It refuses to start a
new case while a player/model/build process exists and never kills contenders.
It is prepared, not executed. Long-track/2048-column/end-to-end timings,
process RAM/commit and CPU/power profiling remain unmeasured.

## Preserved limitations

CQT is centered/preprocess only; long bass support still smears time.12-bit
display cache is not an invertible/phase/dBFS professional analysis archive.
The full FFT call/large allocation is not interruptible. No GPU/multirate CQT,
audio backend, translation model or dependency-version change. No live audio
device or owner profile was used by the tests. The owner still needs to inspect
desktop transitions and normal use.
The shared worker-setting editor still lives under the Superlet panel; this
round reuses that setting but does not duplicate GPU/ASLT controls into CQT.
Actual allocator refusal was not fault-injected; budget refusal and normal
allocation/cancellation cleanup are tested, not a complete OOM-recovery proof.

## Development failures preserved

The first vendor test invocation could not resolve its offline wasm-bindgen-test
dev dependency; the isolated public-AVX-planner gate avoids vendor dev deps.
The new leaf dispatch initially omitted the64-point fallback method. Root's
initial frozen gate import/type annotation, kernel storage type inference and
standalone module path also failed compilation before correction. Logs remain
under moosik-v155-home/round4. Initial clippy found3new warnings, corrected in
code rather than waived. No broad formatting of old spectrum/vendor files.
The frozen test module explicitly allows dead_code so its original unused UI
helpers/result fields can remain verbatim; the standalone probe also retains
unused included public helpers. These are disclosed test/research allowances,
not production-candidate lint waivers.

The first complete debug suite ran1396pass/1fail/41ignored. The only failure
was an old capability assertion that f64 256/512 could not prune, precisely
the capability intentionally added here. It is updated to assert both f64
directions and the unchanged128-point fallback. f32 assertions and all numeric
oracles stay intact. Later final-gate results supersede that failed attempt.
Log max parallel slots is allocated parallel capacity, not measured active
concurrency; zero denotes the serial route. Requested-output call counts do
not mean every nested leaf avoided its own dense fallback.

A fresh read-only agent audit caught a pending-preview UI grid defect:
opening Whole track switched the ordinary plot's rate back to the analysis
rate, caching ISO weights that would be stale on close (44.1k analysis vs96k
live). The correction keeps pending presentation on the live rate even while
hidden and syncs hidden route changes. A real UI-toggle test covers96k→32k
live against unchanged44.1k analysis, ISO weights, no snapshot/head/worker.
The agent otherwise found no concrete blocker in parallel workspace accounting,
serial folding, cancellation or pool boundaries; that was a static audit, not
another execution gate. The full debug binary already running at discovery
predates this correction; final release/targeted debug cover the newer tree.

## Final gate disposition

Runtime code commit20c0b020740fb1af72763e09a85a4107bf1a720e; prior runtime
commits8121367(core/preview) andeba8835(f64 leaves). No runtime source edits
after20c0b02; subsequent report/harness changes are not application changes.

| Gate | Result |
|---|---|
| Full default debug on8121367, before the final UI-only correction |1397passed /0failed /41ignored |
| Targeted default debug CQT on20c0b02, including new UI regression |38passed /0failed /1ignored |
| Full all-target/all-feature release on20c0b02 |1399passed /0failed /45ignored |
| No-default-feature check on20c0b02 |exit0,5existing warnings |
| All-target/all-feature clippy on20c0b02 |exit0;19unique file/message identities unchanged |
| Isolated default release leaf gate, thinLTO/one codegen unit |5passed, including326912pairs |

The full release inventory has one enabled diagnostic test and four ignored
diagnostic experiments beyond the normal feature set; the new UI test also
explains the one-test increase beyond the earlier full debug binary.
Clippy JSON emits53records while the old rendered baseline prints28blocks
because duplicates are suppressed there. Both have the same19unique
file/message identities, with zero additions/removals. This is not a claim
that53JSON records are the same measure as28rendered blocks; raw logs and
round4/clippy-comparison.json are retained.

No new formal timing, physical-memory, long soak, live-device or pixel-render
claim is made by these gates. The ordinary-feature build/artifact disposition
is recorded below.

## Frozen prereview and next step

Ordinary/default-feature release build on20c0b02 exited0, with one existing
gpu.rs dead-code warning. No RUSTFLAGS or CARGO_ENCODED_RUSTFLAGS override.
PE FileVersion1.5.5, ProductVersion1.5.5 prereview20c0b02 (the actual resource
string has spaces around the label/hash). This was not launched by the agent.

- File: artifacts/speed-preview-20c0b02/Moosik-1.5.5-prereview-20c0b02.exe
- Size:67,643,904bytes.
- SHA256:54c626e3f0d90559d8f13680dad5226489852dfd18e767ca8a9861fda47cae56.
- The copy and normal-build source exe have identical SHA256.
- Oldf9a6999 exe is not overwritten. Its frozen SHA256 remains
  7be7555cc2560cff98c21bcc3e31b3fd7fb1994d9633d217fce312bd786c981f.

The CLI probe was rebuilt with the corrected app-matching profile/edition.
Its guarded90-run paired campaign is ready but unexecuted. All owned compiler,
test and probe processes retired. The owner player alone remained at handoff;
no process was stopped to manufacture a quiet machine. Close the old player
before starting the new one for normal use. A subsequent clean-machine
comparison is the remaining performance research item, not a completed result.

The checklist and current definition accompany the new exe. Report-only and
research-harness closeout commits may follow20c0b02; runtime source is unchanged.
Main/local origin-main330b8ac, original shared checkout71be39b with15status
entries and the original stash remain untouched by this work. No remote/public
repository action, merge, tag or release.

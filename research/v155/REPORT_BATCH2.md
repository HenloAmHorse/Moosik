# 1.5.5 research — batch 2

2026-10-19. Owner asked whether to continue or start manual testing. Continued;
**no owner test yet**, because true CQT has no integrated UI/cache route.

Status: **batch complete**. Product workspace change accepted as an isolated
candidate after the gates below. CQT is still research-only; no prereview exe.

## Commits and product scope

- `8c08d4d`: actual live FFT workspace reuse, plus three targeted product tests.
- `b16a17c`: shared-input CQT FFT and initial window-quality research.
- Later report/scripts revision extends the window study to approximately matched
  Hann half-power bandwidth, not just equal nominal support.

Only `src/spectrum.rs` and its new test module change product code. No version,
settings serialization, cache identity/format, translation, UI or ASLT arithmetic
changes. Main remains the released `330b8ac`; no merge/push/tag/release.

## Actual live FFT patch

Reuse the sample snapshot vector, complex Mix input and FFT scratch. The snapshot
still leaves the tap lock before windowing/FFT. Channels reuse the same scratch
sequentially, with separate channel norms so the Mix measurement is not destroyed.
Each geometry rebuild resizes buffers and resets channel state. Padded input tails
are reset on every transform; FFT scratch itself need not be cleared.

The old allocating `bins_to_bars` wrapper now exists only for tests; production
Mix/channels both already have norms. No dead-code `allow` was introduced.

Evidence:
- Three new debug product tests pass. The independent FFT expression is copied
  from release330b8ac, not a call through the new helper.
- Every complex coefficient and norm bit matches across4 FFT sizes (up/down),
  all4 supported windows, changing signals and a silent tail. Channel-first use,
  Mix norm preservation and short-input/snap behavior are checked.
- Actual production FFT helpers allocate **zero times after warmup**, measured
  with the existing thread-local allocator probe. **Not** a claim that whole UI
  ticks, bar vectors or waterfall row creation allocate nothing.
- Eight existing frozen mapping/downstream live-route tests pass debug.
- Seven feed/FFT-domain tests pass debug.
- Three existing ASLT/cache/channel/waterfall downstream tests pass debug. The
  full requested-output leaf check took245.66s for the suite; do not repeat it
  indiscriminately in debug just to inflate gate counts.

Reusable buffers retain capacity, including after downsizing. This trades a
small bounded-by-supported-FFT-size resident workspace for repeated allocations;
no measured whole-player RAM or CPU saving is asserted. Batch1's ~7–10% stage
microbenchmark was not an entire app benchmark, and must not be marketed as one.

## Shared-input CQT FFT prototype

`grouped_fft.rs`: same dense f64 Hann definition as batch1, no sparsification,
downsampling or interpolation. Group by transform size; share each input FFT
across a batch of kernels, with separate inverse per kernel. Kernels and temporary
work are batched under an explicit buffer-payload allowance. Lower allowance
may repeat input FFTs; insufficient allowance returns an error, not bad data.

**This allowance is not a total-process memory cap.** It includes kernel spectra,
input/product buffers, exact scratch length, and a conservative transient reserve
for dense kernel construction. It excludes planner internals, source audio,
output coefficient matrix, vector headers and allocator overhead. No RSS peak was
measured. Group planner lifetime is bounded to one size, not accumulated forever.
Do not repeat the earlier M1 mistake of calling partial accounting a RAM limit.

18 rows, six configurations, three repetitions each, alternating method order:
4s chirp+noise,55Hz..8kHz,180 frames/s, single-thread f64 CPU. Setup, planning,
kernel construction, allocation and output extraction included in both arms.
Baseline is the NEW CQT's per-bin full convolution, **not shipped ASLT/FFT**.

| rate / BPO | explicit allowance | per-bin median | grouped median | speedup |
|---|---:|---:|---:|---:|
|48k /24|32MiB|546.26ms|337.22ms|1.62x|
|48k /24|64MiB|541.15ms|335.52ms|1.61x|
|48k /48|32MiB|1468.88ms|858.62ms|1.71x|
|48k /48|64MiB|1470.35ms|853.21ms|1.72x|
|96k /24|32MiB|1158.40ms|750.79ms|1.54x|
|96k /24|64MiB|1160.19ms|757.00ms|1.53x|

Do not add/multiply these ratios by batch1's ratios: different bank ranges and
baselines. The result does not establish optimal route thresholds. Direct short
kernels and sliding sums remain possible arms, not all necessarily winners.

Observed max complex discrepancy vs per-bin FFT **2.50e-16**, zero changed research
codes. Independent direct f64 reference also checked by tests. Batch-limit test
requires identical complex bits between768KiB/8MiB runs with different batch
counts, and verifies too-small refusal. This is stronger evidence of batch-only
invariance than merely observing identical magnitudes, but still not a pixel or
production quantizer gate.

Counters are incremented at actual forward/inverse calls. Examples:48k/24 used
1437 input FFTs,173 kernel FFTs,22950 inverses.48k/48 reduced input FFTs735->723
with32->64MiB, while inverse count remained22392. The inverse work has not magically
disappeared. No GPU, multithreading, stereo reuse, cancellation or streaming sink
was implemented in this research function; these remain integration requirements.

## Window quality: do not pick a default on speed alone

`two_tone_quality.py`: real-tone coefficient formula, with independent finite-sum
checks at selected phases. Strong440Hz tone amplitude1, weak neighboring tone
amplitude0.01 (-40dB),24 bins/octave,5 separations and4 relative phases. Initial
20-row equal-support table retained separately; final40-row table includes an
approximately equal Hann half-power-bandwidth policy. It does not assert exact
equal ENBW, uncertainty product or equal perceptual resolution.

At equal nominal support and2 grid-bin separation, Hann leaks about-35.86dB of the
strong tone into the weak bin, greater than the weak tone itself. Four tested
relative phases give combined magnitude roughly0.0061..0.0261 for a0.01 weak
tone. At6 grid-bin separation,4-term Blackman-Harris leaks about-103.66dB, but its
broader mainlobe makes close neighbors worse. Different bandwidth normalization
also changes temporal support; there is no universal best window in this table.

These are analytical synthetic model results, not listening tests, real-music
harmonic transcription, professional accuracy certification or comparison with
ASLT. Window defaults and parameter semantics are not settled yet.

## Gate evidence / pending limits

Product on the b16a17c tree (same product sources as8c08d4d), with test-build label
`1.5.5 research`, no custom rustflags:
- Full release all-targets/all-features: **1361 passed,0 failed,44 ignored**.
  This is release1.5.4's1358 plus the3 new tests. Execution34.23s after6m05s
  compilation. No new full debug DSP suite or soak was run.
- No-default check succeeds,5 emitted existing warnings.
- Clippy all-targets/all-features: **28 normalized diagnostics**, header/file
  multiset identical to the released1.5.4 baseline (not just equal counts).
- `git diff --check` clean; the only formatted product file was the new test
  module. No blanket rustfmt of the existing source.
- Targeted debug results above are in addition to the release run, not independent
  proof of pixel equivalence. No GUI, live playback or audio-device measurement.

Research crate: **12 passed debug and release** (11 CQT+1 isolated live FFT),
clippy `-D warnings` clean. Two new grouped tests passed on first run. An initial
test budget was corrected before execution because the planned512KiB admitted
no kernel;768KiB correctly forces multiple batches. No failed result was discarded.

Research measurements were run after the app debug downstream job exited, before
the app release compile began. Later correctness-only research tests ran alongside
the app compile; their wall times are not performance measurements. No owner
player/model process was running for the CQT measurements. Ordinary OS background
load was not controlled. Raw evidence at
`C:/Users/PC/Programming/moosik-v155-home/batch2/`.

Next: settle CQT configuration/normalization and window choice against stronger
fixtures, compare actual production level/quantization and display replay, add
bounded streaming stereo/cancellation core, then preprocess/cache/UI integration.
No need for owner testing or long soak of the current prototypes.

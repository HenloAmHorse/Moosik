# 1.5.5: common audio formats and the missing CQT/ASLT comparison

2026-10-20, Windows x86_64, i7-9700F, Rust/Cargo 1.94.0.
Research branch only. No merge, push, tag, release, public-repository action,
audio-device launch, downloaded recording or owner-profile experiment.

## Outcome

Implemented common audio-format import/decoding consistency, including native
mono/stereo Opus. This is **not** universal codec/profile support, a YouTube
client or a video renderer. Details: [Audio formats](../../docs/AUDIO_FORMATS.md).

The owner's CQT speed concern is real relative to ASLT n=1. The new end-to-end
screen finds CQT48 about 4.07x its wall time on this fixture. CQT's files are
smaller here, not larger, while its peak working set is higher. These are
selectable settings, **not matched response/quality**. CQT's long bass response
has not been shortened or hidden behind display smoothing.

## Sources and provenance

- Start: `3f098add4fa55b6ad728c1d70b5a7b1694397183`, clean.
- Recovery: `refs/recovery/codex-v155-format-cqt-start-20261020`.
- Format implementation and paired harness:
  `833871f9fb56a1aa7f7aa8f1888a512b7bbcea19`.
- Separate Opus seek integer-clock correction:
  `de4ec763ba9a9a06cf28ac029355c82abb4685d4`.
- Timing uses the frozen **833871f default-feature release test executable**,
  SHA-256 `541d0a2b1f32fad4a81d6772e7d1c77561f3aea8ac5119843cce62e53028a568`.
  The later correction changes only seek targets/tests. No CQT/ASLT/preprocess
  source differs between these commits; no measured arm seeks its input.

Private evidence root:
`C:/Users/PC/Programming/moosik-v155-home/formats-20261020/`.
Raw runs: `pair-30s/results.jsonl` and all 15 stdout/stderr logs. Five 1s/32bar
smoke records are separate, not included in the performance table. Earlier
117 CQT-core comparisons remain in `REPORT_TIMING.md`; they were not repeated.

## What changed for formats

The picker and folder importer used different narrow extension lists despite
existing AAC/MP4 readers. One case-insensitive inventory now exposes supported
containers and aliases. ALAC, AIFF/AIFF-C and CAF are enabled explicitly.
Information reports the probed codec/lossless property; M4A does not imply AAC,
and WAV does not imply uncompressed PCM. Native preparation no longer refuses
an ALAC stream merely because the container omits a channel field that the
priming decode can establish.

Opus uses the bundled reference libopus through a small original adapter. Its
fixed packet buffers, header gain, mono/stereo mapping admission and reset/error
ownership are explicit. It is registered in a shared opt-in Symphonia registry,
so rodio playback/preprocess and native output cannot silently diverge on codec
availability. There is no runtime FFmpeg process or extra Opus DLL installation.

Registering a codec was insufficient. Ogg now owns pre-skip, positive initial
granules and EOS padding; Matroska owns CodecDelay and signed block padding,
including laces. The decoder applies trims once. Rodio now retains a selected
audio track, filters foreign packets, continues past fully trimmed packets,
reports fractional duration correctly and uses the real time base. Its generic
Read+Seek wrapper supplies file length and restores the original cursor.

Correctness-first Opus seeks decode/discard from the known beginning with
bounded packet memory. This avoids cold predictive decoder state, works without
Cues and reproduces decoded samples, but can cost more on long seeks. Indexed
preroll is not claimed. The follow-up replaces f64 seconds-to-frame conversion
with integer nanoseconds: even 9ms at48k can otherwise land one frame early.

Native decode is f32/48k for Opus. The existing shared/analysis rodio i16
conversion is **unchanged**, not represented as a full-precision source archive.
AAC-LC is playable but existing MP4 edit-list/encoder-delay handling is not
sample-exact against the synthetic source; its tail is explicitly not gated as
exact. HE-AAC, WMA, APE, WavPack and Opus multistream mappings are not newly added.
General chained/multiplexed Opus variants remain unverified.

## Decode evidence

Checked-in original synthetic fixtures include 1.137s stereo48k tones in Opus,
WebM, ALAC/M4A, AAC-LC/M4A, AIFF and CAF; 3ms EOS-only Opus; 2.5ms packets with
fully trimmed startup; and a synthetic video-first WebM with copied Opus audio.
No commercial recording is included. Tests need no FFmpeg or device.

Independent private FFmpeg f32 decoding of ordinary Opus and WebM gives exactly
109152 interleaved samples; worst absolute difference is `7.4505806e-8`.
This is a finite numeric oracle comparison, not a universal bit identity claim.
ALAC, AIFF and CAF also decode109152 samples. Production entry tests publish
Mix and L/R caches from all six new fixture families.

Tests also exercise positive initial Ogg positions, longer pre-skip, repeated
forward/backward native/shared seeks, exact9/18/71/282ms cases, malformed header/
packet/trim rejection, output-gain reset, whole-block padding across laces,
and audio selection when a video track precedes it.

## End-to-end performance screen

Identical deterministic 30s stereo48k PCM16 fixture, 1024 columns, nominal180fps,
joint Mix/L/R, six selected workers, GPU off. Three cyclically rotated repeats
per arm; one fresh process/private cache/calibration per run. All15 accepted,
no failed/excluded measured runs, all published disk codes match memory and
all repeated per-arm code-plane SHA-256s agree. Fixture SHA-256:
`3d7d03fd21cfffe10f488428efa8eecfe5c09a1176413a5c252a0bed22461746`.

Timer includes actual decode, statistics, analysis, normalization, encoding and
store publication. It excludes fixture generation, selected-pool prewarm,
validation readback and UI/audio playback. This is an optimized **test build**,
with its test allocator instrumentation and unwind policy, not a measured normal
player session. Leaf-diagnostic atomics are off. Peaks are captured process
lifetime peaks through analysis completion, including setup, not resettable
analysis-only allocations and not peaks of subsequent readback verification.

| Arm | Wall min / median / max (s) | Median CPU seconds | Mix+LR files (MiB) | Maximum observed peak working set (MiB) |
|---|---:|---:|---:|---:|
| ASLT n=1 | 2.160 / 2.166 / 2.178 | 11.141 | 21.51 | 115.34 |
| ASLT n=9 | 16.309 / 16.483 / 16.487 | 94.641 | 21.26 | 120.38 |
| Full CQT12 | 6.597 / 6.636 / 6.755 | 14.766 | 16.08 | 177.61 |
| Full CQT48 | 8.777 / 8.813 / 9.007 | 24.938 | 18.43 | 320.35 |
| Full CQT192 | 20.446 / 20.467 / 20.486 | 41.453 | 17.19 | 342.66 |

ASLT settings: q_ratio2, requested max_window0.25s, spread0.5, Blend0.5. CQT:
full uncapped Blackman, Log centres. Qs/windows/axes are not equivalent. ASLT
produces5393 rows at179.7752809fps; CQT5400 at180fps. Their common stored format
holds magnitude-level codes, not f64 complex coefficients. File envelopes can
differ by a few bytes between fresh generations; this is not a code-plane change.
The table is one synthetic workload, not a real-music/hardware generalization.

## What this establishes and what it does not

- The owner's rough n=3–5 comparison is plausible as a runtime impression on
  ordinary support settings; do not turn that into an equivalent-quality label.
- Full CQT48 is faster than n=9 here but substantially slower than n=1. Full
  CQT192 is slower than both. There is no claim CQT must always be cheaper.
- No cache-size penalty appears on this fixture. Compression is content-
  dependent, so this does not prove that for every song.
- CPU/wall ratios indicate roughly2–3 core-equivalents averaged over CQT versus
  roughly5–6 for ASLT, despite the same witnessed six-worker pool. That suggests
  implementation/scheduling/serial-phase headroom; it is **not attribution** to
  a particular phase. Serial kernel preparation and source FFTs are visible
  candidates in the source, not yet measured explanations.
- Read-only scheduling review finds a concrete uneven-batch underfill: the
  current ceil-sized partition assigns7 kernels/6 slots to4 tasks, and13/6 to5.
  A balanced allocation-free split is worth measuring, but7 kernels still need
  two kernel-times on the critical path; fuller task coverage is not a speedup
  measurement. The planner already requires `batch_size >= slots`, so the
  initial suspicion of one-kernel full batches occupying six slots was false.
  First-feasible slot admission also does not score repeated source FFTs across
  batches. Profile those costs before tuning the planner or changing arithmetic.
- The mathematical long support explains temporal spreading, not all wall time.
  Shortening it requires an explicit response tradeoff, not a repaint fix.

## Next bounded CQT work

1. Measure planning/kernel preparation, shared source transforms, inverse work,
   folding and cache phases separately on the surviving representative workload.
   Pursue exact-operation parallelism/reuse before precision concessions. Do not
   rerun the full old campaign or a soak to fill time.
2. Compare actual bandwidth/impulse and gated-tone responses with the ASLT
   controls, not nominal Q numbers. Use long-enough fixtures for bass support.
3. Preserve Full CQT. A separately named responsive/variable-Q option with an
   explicit bass-window/bandwidth control is a plausible usability candidate;
   it is not implemented or silently substituted by this round.
4. Direct short-kernel and octave/multirate candidates remain experiments.
   Different operation order/precision/anti-aliasing is not automatically exact.
   Require the declared complex/reference/code gates and a distinct numerical
   identity if the accepted contract changes. Future professional consumers
   must not mistake the display cache for a complex/phase archive.

## Gates / closeout

- Full debug all targets/all features:1411 pass,0 fail,46 ignored.
- Frozen833 release default features:1410 pass,0 fail,42 ignored.
- Opus leaf:7/7; MKV:7/7 including4 new padding tests; actual patched rodio
  module:8/8 through a private test facade. The facade preserves its local error
  as boxed Other; production retains its concrete rodio error conversion.
- Clippy19 unique(file,message) identities, matching20c0b02; no-default check
  exit0 with5 existing warnings. No whole-repository formatting applied.
- Final de4 release default-feature validation:1410 pass,0 fail,42 ignored,
  31.63s test execution after compilation. Targeted final debug audio tests:
  14 pass,0 fail,1 unrelated ignored; private decoder facade8/8 again.
- Final de4 clippy and no-default checks also exit0:19 unique(file,message)
  warning identities identical to20c0b02,5 existing no-default warnings.
  A first warning-comparison command attempted to parse two Cargo progress
  lines as JSON; the corrected fail-fast comparison explicitly accepts those
  progress lines and confirms an empty identity-set difference.

Initial failures are retained: CMake auto-selected an unsupported VS2026
generator; selected installed VS2022 explicitly. A channel-layout assumption,
an initialization-seek panic/unknown wrapper size, a moved iterator, trait import
and the test-facade's duplicate error type were found and repaired. Do not call
those first attempts green. The extra summary command initially flattened hash
arrays incorrectly; corrected grouping validates all15 records.

The old frozen20c0b02 and f9a6999 player hashes were rechecked unchanged. No new
GUI player bundle is distributed by this report. The source notice inventory
needs full regeneration before binary distribution/release. Main/localorigin
remain330b8ac, the shared checkout's15 changes and stash are preserved.

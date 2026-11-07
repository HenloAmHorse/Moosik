# CQT follow-up: response and end-to-end cost

Read-only design review, 2026-10-20, against `3f098ad`.
No experiment in this document has been run. Existing timing evidence remains
`REPORT_TIMING.md`; its 117 runs compare CQT implementations, not CQT to ASLT.

## What the owner's report means

Two different problems need separate answers:

1. **Analysis completion:** elapsed decode, statistics, transform, encoding and
   publication time. Faster convolution can improve this.
2. **Temporal spreading:** the coefficients respond over a long centred window.
   Computing those coefficients faster cannot shorten that response.

The current transform genuinely uses every requested display centre. At 1024
columns, selecting 12 bins/octave still computes 1024 filters: BPO changes the
support Q, not filter count. Its low BPO result can therefore be both densely
sampled and broad in frequency. This is not the same as a native 12-filter-per-
octave bank followed by a display resample.

`src/spectrum/cqt.rs:102-108,371-396` gives full support approximately `Q/f`,
where `Q = 1/(2^(1/BPO)-1)`. Calculated, not measured temporal widths:

| BPO | Q | 20 Hz full support | 50 Hz | 100 Hz | 1 kHz |
|---|---:|---:|---:|---:|---:|
| 12 | 16.817 | 0.841 s | 0.336 s | 0.168 s | 0.0168 s |
| 50 | 71.636 | 3.582 s | 1.433 s | 0.716 s | 0.0716 s |
| 192 | 276.498 | 13.825 s | 5.530 s | 2.765 s | 0.2765 s |

These are not update intervals, group delays, onset errors or measured FWHM.
For preprocessed centred windows, future samples participate too. Increasing
stored frame rate cannot remove this spreading.

ASLT has a different contract: `effective_q_at` and `window_seconds_at`
(`aslt.rs:413-461`) limit support using `max_window_s`, subject to its minimum
three-cycle member. Its Gaussian member's Q is not numerically comparable to
CQT's cosine-sum-window support Q. ASLT `2x grid` means `q_ratio = 2`, **not
2048 columns**. Its `Blend(0.5)` frequency centres are also not CQT's log centres.

## First bounded experiment: answer the comparison the owner actually asked

Add a new ignored app test, rather than modifying the completed timing probe.
Reuse the private calibration/pool, Win32 process counters and production-entry
pattern in `src/spectrum/aslt_r11e2e.rs:53-138`. Call `preprocess_file`
(`spectrum.rs:4711`) for all arms so decode/statistics and publication are real.
Use one named test per process to isolate peak working-set and commit counters.
Do not run through an audio device or the owner's profile.

### Repair the measurement trap before copying that harness

The old R11 test still obtains file sizes from `track.spectrumcache` and
`stereo::path_for(track.spectrumcache)` at lines 97-98. Current `test_store`
opens `<parent>/v2`, and publication writes generation `.mix`/`.lr` files there.
The old size reads can return zero through `unwrap_or(0)` despite successful
publication, and its old cleanup leaves the new generation files behind.
This is a stale research harness, not evidence of a product cache defect.

The new harness must:

- Require `meta.published.is_some()` and a successful store, not silently time
  a run whose cache publication failed.
- Resolve the actual head/generation using `published_heads`/`head_files`
  (`spectrum.rs:4030-4060`) or the returned publication identity.
- Require exactly one published head in each fresh private arm directory.
- Measure positive Mix and L/R payload sizes separately, plus namespace total
  and coordination overhead. Decode and validate the actual disk payloads.
- Use a uniquely owned temporary directory and retire that entire exact
  private directory after releasing message pins. Never clean by a broad glob.

### Two comparisons, not one misleading leaderboard

**A. Actual selectable settings.** Same synthetic WAV, sample rate, 1024
columns, nominal 180 fps, six selected CPU workers, same channel request:

- ASLT n=1 and n=9, q_ratio=2, max_window_s=0.25, spread=0.5, Blend(0.5).
- Full CQT Blackman, BPO 12, 48/50 and 192 as explicitly labelled settings.

This answers practical speed/size, not equal-quality efficiency. Do not claim
the two frequency axes or windows are matched. Scope the first pass to 48 kHz
30 s joint, then 48 kHz 120 s joint and 96 kHz 30 s joint for surviving arms.
One short mono case checks joint overhead. Three rotated repetitions per
retained comparison are enough; do not repeat the old 117-run campaign.

**B. Matched response.** Use ASLT Log, same physical frequency centres and a
common physical clock for the response-only probe. At 30/50/100/250/1000/4000/
10000 Hz measure actual steady-tone -3 dB and -6 dB widths, off-centre tone
suppression, impulse-envelope width and gated-tone attack/release. Measure
side-lobes or two-tone separation too: equal mainlobe width is not equal
quality. ASLT n=9 is nonlinear after the per-member magnitude, so a single
linear impulse-response description does not establish all its behaviour.

Use the independent CQT direct f64 definition (`cqt/tests.rs:60-89`) and ASLT's
actual member construction, with central signals padded by real fixture
duration on both sides. A 2 s fixture cannot characterize a 13.8 s bass window
away from boundaries. For long-window low-frequency response, use a small
selected-frequency probe rather than running 1024 filters on every fixture.

Compare kernel response before display EMA/weighting. Separately replay cached
rows with smoothing disabled to check the UI/clock; do not confuse coefficient
response with display smoothing. Production clocks differ: ASLT rounds an
integer hop (`aslt.rs:3153-3161`), while CQT uses nominal fractional frame
centres (`cqt.rs:205-227`). At 48 kHz ASLT nominal 180 means 48000/267 fps;
do not demand equal row counts or compare row indices as equal times.

### Required records

Each record pins revision, input hash, descriptor, mode, actual sample/row/
column count, frame clock, worker witness, output hashes and actual cache
bytes. Record wall time and CPU seconds, peak process commit/working set,
and the existing explicit-vector workspace separately. Report RTF as
wall/audio-duration, not wall/rounded-row-duration.

Add test-only coarse spans around decode, statistics, transform and encode/
publish. Fine CQT counters may additionally separate per-size planning/kernel
preparation, input FFT, inverse/gather and code mapping. Never derive phase
durations from UI progress percentages or sum overlapping worker times as wall
time. Avoid per-cell timers, which would alter the hot loop being measured.

## Candidate order and numerical contracts

### E1: direct versus FFT for short high-frequency kernels

Current CQT routes all bins through FFT groups (`cqt.rs:371-396,573-880`), even
short treble kernels; optional parallel inverse work begins at N >= 4096
(`cqt/parallel.rs:20`). ASLT already has an empirically selected direct/FFT
split (`aslt.rs:1533-1549`). At 180 stored frames/s, computing only requested
short dot products may be cheaper than even the pruned FFT route.

First benchmark a small kernel-length/hop grid, including setup, mono/joint,
short/long inputs and the existing six-worker pool. Share kernel construction
between L/R and derive Mix from complex responses before magnitudes. Preserve
full support, gain, sign, zero-outside-file convention and frame centres.
Do not import ASLT's truncated-edge renormalization: CQT currently uses full
window normalization with zero-valued out-of-file samples.

This is **mathematically equivalent, not floating-bit equivalent**: direct
summation and FFT change rounding. Admit no automatic production switch on an
error norm alone. Compare complex responses to the independent f64 reference,
and require zero differing v4 codes against the retained route on the stated
corpus. Include near-code-boundary, silence, cancellation, anti-phase stereo,
short-file and block-edge cases. If codes differ, retain it as a separately
identified research route; do not quietly relabel it an exact optimization.

### E2: a responsive analysis profile, explicitly not uncapped constant Q

This is needed to address the owner's temporal complaint. An optimization
which preserves the existing full CQT cannot do so.

Keep existing Full CQT available and cache-distinct. Evaluate a clearly named
**Variable-Q / Responsive** profile with an explicit maximum support (initial
probe 0.25 s) and/or positive bandwidth floor. For a hard support maximum,
`T(f) = min(Q/f, T_max)` is a straightforward window definition: constant Q
above a crossover, approximately constant bandwidth below it. It gives up
low-frequency discrimination for shorter transient spreading and often less
work. A bandwidth-floor VQT is a different smooth definition; compare response
instead of claiming the formulas are interchangeable.

Changing BPO alone cannot preserve treble detail while shortening only bass.
This profile is a user-visible analysis change, not a cache-compatible speed
patch. Its descriptor must include the support/bandwidth rule, values, clock,
normalization and numerical revision. Do not change an existing Full CQT
profile's meaning when it reloads. Show useful presets and measured tradeoffs
after the response survey, not a slider whose name implies more is always
better. Source audio playback is unaffected.

### E3: octave multirate CQT, only after those comparisons

Lower octaves can be evaluated at successively reduced rates, shrinking the
number of samples in equally long physical windows. A real anti-alias chain,
phase/delay alignment, resampled kernel definition, guard-band rule and common
output clock are required. Low-pass then downsample, not sample skipping.
Do not mix separately decimated L/R clocks; Mix must still derive before
magnitudes. Pay for filtering and resampling in the timing.

This is a high-leverage algorithmic candidate, but generally **not identical
to the current full-rate discrete convolution**. Define an independent
multirate reference and separate analysis identity; do not promise old caches,
old bits or universal pixel equality. Test alias leakage with out-of-band
tones, octave-boundary continuity, phase cancellation, impulses and fractional
frame-centre alignment. Preserve an exact-full-rate fallback where its
definition is requested. Multirate reduces compute, not the physical support
duration; it alone will not fix bass temporal smearing.

### Lower-priority exact reuse

Prepared kernel spectra/plans depend on settings, not song samples, so a
bounded same-settings bank could reduce setup on subsequent *different* songs.
It does not improve the first song or fix temporal response. The current
per-size planner intentionally dies after each group to bound residency; do
not undo that with an unbounded global cache. Prove the complete key (sample
rate, exact centres, BPO, window, transform length/numerical revision), charge
resident vectors and account plan storage honestly before considering it.

Current CQT already shares source FFTs across filters in a size group and uses
`process_pruned_product_with_scratch` (`cqt.rs:324-336`). Neither is an untried
new idea, and changing display bar gap does not remove analysis work.

## Stop points

1. Land no CQT numerical/default change merely because the format work builds.
2. Produce the paired baseline and measured response table first.
3. Screen E1 and E2 independently; only the winning bounded candidates get
   production-integration work and an owner-labelled prereview.
4. Start E3 only with a numerical/output contract accepted for its new mode;
   never smuggle an approximation behind the current exact-display promise.
5. No repeat soak, GPU campaign, owner-media download, merge, push, tag or
   release is needed for this design pass.

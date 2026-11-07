# 1.5.5 implementation proposal after the first screen

Status: proposal; no production source changed. This is not a release promise.

## First integrate the small, exact FFT workspace change

`src/spectrum.rs::process_realtime` currently allocates a sample snapshot, complex
input and norms each tick; rustfft's `process` owns a temporary scratch allocation.
The isolated `rt_fft_probe` reproduces those four allocations and compares all
complex/norm bits against persistent buffers and `process_with_scratch`.

Product patch should add reusable sample/complex/scratch storage; take the same
sample snapshot under the same short tap lock, then release the lock before FFT.
Fill the same coefficients/products in the same order. Keep `last_fft_norms`
available to its current consumers; do not borrow/overwrite it for a channel FFT.
Existing `ch_scratch` can use persistent FFT scratch; channel norms need their own
reusable storage. Buffer sizes must follow `rebuild_fft`, sample-rate/size changes,
short input and cancellation. No new FFT plan per frame. No long-lived references
into the mutable tap and no changes to smoothing/snap/waterfall semantics.

Acceptance: existing frozen-reference routes, actual Mix/L/R/Split snapshots,
every waterfall row and peak input, all mapping modes and geometry changes.
Allocation counter scoped to actual accepted ticks, not just this isolated probe.
Time the application before claiming less CPU; the FFT stage is only one part of
the UI and this change is not expected to multiply whole-app speed.

## True CQT: specification before a dropdown

Keep the old `Cqt` serialized value/id 2 as FFT weighting. Add an independent
transform configuration and revision; do not invalidate unrelated 1.5.4 caches.
The store's descriptor/identity is the right place to bind new result-affecting
settings. Do not redesign the 1.5.3 store while adding a transform.

Minimum explicit fields: transform revision, effective sample rate, frequency
bounds and grid convention, bins-per-octave or independent Q/density if offered,
window definition, centered/causal timestamp convention, edge policy, exact frame
centres/rate, optional support cap and the fact that it changes Q. Separate
display-only settings from analysis identity. Canonical serialization, validation
and source snapshot checks must remain shared with existing code.

Do not silently turn 2048 display columns into extremely long bass filters.
Classic musical density (24/48 bins per octave) and dense display grids should
be described separately. Oversampling a fixed-Q bank adds sample points, not new
resolving power. A capped constant-Q bank is variable-Q in its capped region.

Hann is the first **test definition**, not an approved quality default. Hamming
and Blackman-family windows have different mainlobe/sidelobe tradeoffs. Evaluate
strong/weak neighboring tones and time localization before choosing defaults;
do not maximize a pretty, sharp picture at the expense of hiding leakage.

### Algorithm order

1. Add a standalone `src/spectrum/cqt/` module, independent of egui/cache. Complex
   coefficient core, metadata and cancellation first. Keep slow independent f64
   reference in tests, with formula/dimension checks outside the optimized builder.
2. Baseline dense/direct evaluation and bounded per-bin work; retain one semantic
   reference. Grouped FFT convolution then amortizes shared source FFTs and LR
   kernel setup across bins of the same block size. Explicitly budget kernel
   spectra + product/scratch + source blocks + outputs, not just tap counts.
3. Three-/five-/seven-resonator sliding sums are promising for cosine windows.
   They are mathematically equivalent in exact arithmetic, NOT bit-identical
   floating-point replacements. Real cache-code comparison and difficult
   quantization-boundary fixtures are required; a tiny absolute error is not
   permission to change the owner's display contract. Reject an unproven route
   rather than normalizing deviations away. Rebase/seek/long-tail behavior needs
   explicit tests; constant state per bin excludes necessary source history.
4. The first FFT screen is single-bin and f64, includes planning. Build a bounded
   shared-input bank before choosing it or rejecting it on these microbenchmarks.
   Do not confuse its performance with the shipped ASLT pruned-inverse route.
5. Multirate/sparse/octave algorithms remain separately named experiments. Proper
   anti-alias filtering is mandatory; same-looking tones do not prove no loss.
   GPU follows an end-to-end cost census, not a utilization target.

### Stereo, time and professional consumers

For this linear CQT, form `M=(C_L+C_R)/2` and `S=(C_L-C_R)/2` before magnitudes;
the existing ASLT member-folding rule must not be changed by analogy. Mix/L/R
therefore need two channel transforms, not a third Mix transform. A future Side
plane can also be computed then, but persistence/capacity is a separate decision.
Magnitude-only caches cannot later reconstruct arbitrary phase relationships.

Centered CQT needs future samples. Preprocess may read them; real-time cannot
pretend that displaying a delayed coefficient is zero latency. State warmup,
lookahead, seek reset and timestamp policy explicitly, and do not compare an old
FFT trailing window with a centered CQT as if their transients shared a time axis.
All downstream `wholetrack`, waterfall and spectrum axes must use the new route's
metadata, not assume FFT hop or ASLT settings.

## Bounded next batch

1. Actual app FFT workspace patch + affected reference tests and scoped timing.
2. CQT window/geometry quality fixture matrix and a bounded grouped FFT prototype.
3. Compare CQT direct / grouped / sliding on the same definition, with actual
   production f32 level conversion and quantizer. No pixel-perfect claim yet.
4. Integrate preprocess only after the choice survives that gate. Add Mix/L/R,
   isolated caches, progress/cancellation and a labelled prereview artifact.
5. Real-time CQT follows latency and memory evidence; never hold the UI thread
   hostage to a multi-second low-frequency computation.

No owner hardware test needed for the current research-only files. No reason for
a 10-hour soak yet: algorithm errors need targeted fixtures first.

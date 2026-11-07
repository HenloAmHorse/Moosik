# 1.5.5 research — batch 1

2026-10-19. **Research screen completed; 1.5.5 is not implemented or ready to test.**
Measured probe source: `6b34309`. This report and the integration proposal are a
later documentation commit. Production source, product Cargo files, docs, main,
released artifacts and the owner's profile were not changed.

Evidence root: `C:/Users/PC/Programming/moosik-v155-home/`.
Final measurements: `final/*.csv`, derived `final/summary.json`; previous screens
remain in `raw/`, mutation sources/results in `mutants/` and `final-mutants/`.

## 1. What is established

1. Current `BarMappingMode::Cqt` is FFT-magnitude weighting; its cached gather plan
   is already optimized. That is not the true CQT now requested.
2. A genuine frequency-dependent Hann-window transform has an independent f64
   compensated direct oracle, cached direct implementation, dense overlap-save
   comparator, and a three-rectangle sliding candidate in this research crate.
3. The sliding candidate can save substantial low-frequency work by reusing
   overlapping input across time. It loses in the treble, so a blanket replacement
   is rejected. This does **not** accelerate Gaussian ASLT by the same identity.
4. Existing live FFT workspace reuse is a separate small, bit-preserving candidate.
   It has been proved only in an isolated reproduction, not integrated into Moosik.

## 2. Definition and quality boundaries

See CHARTER for the full equation, sample clock, gain and edge convention.
This is a **symmetric Hann test definition**, not an approved CQT default or a
claim of superior quality to ASLT. No coefficient sparsification or downsampling.

For the Q/density coupling used here, the calculated nominal support at 20 Hz is
1.706 s at 24 bins/octave, 6.900 s at 96 bins/octave, and **14.799 s** if literal
2048-bin density is requested across 20 Hz–20 kHz. Half that support is centered
lookahead. Integer sample rounding adds up to two samples to full support.
These are not the owner's capped 0.25 s ASLT settings. A support cap breaks the
strict Q in the affected region; oversampling a lower-Q bank is another honest
option, but neither can be advertised as free resolving power.

Window response screen (`window_response.py`, analytical DTFT, h=4096, offset
step 0.001 full-window DFT bins, sidelobes searched out to 20 bins):

| window | full -3dB width (bins) | first minimum offset | peak sidelobe (dB) | sliding rectangles |
|---|---:|---:|---:|---:|
| Hann | 1.442 | 2.000 | -31.47 | 3 |
| Hamming | 1.304 | 2.000 | -42.68 | 3 |
| Blackman | 1.644 | 3.000 | -58.11 | 5 |
| Blackman-Harris 4-term | 1.900 | 4.000 | -92.01 | 7 |

These are model response measurements, not a perceptual quality ranking. More
sidelobe rejection can widen the mainlobe; choosing a fast Hann just to win a
benchmark would be the wrong default-selection process. The analytical DTFT was
checked against an independently summed small window before producing the table.

## 3. Complete-bank CPU screen

i7-9700F, 8 cores, rustc 1.94.0, Windows x86_64 MSVC, optimized f64, single-thread
bank traversal. Synthetic chirp plus seeded noise, 20 Hz–16 kHz, 180 frames/s.
Three repetitions, alternating direct/hybrid order inside each bin. Hybrid uses
sliding only <=1 kHz, a fixed threshold chosen before the bank runs, not a fitted
planner. Plans are generated/released one bin at a time. No app/profile/device use.

| rate / density / duration | bins | direct processing | hybrid processing | shared plan cost | ratio including plan |
|---|---:|---:|---:|---:|---:|
| 48 kHz / 24 BPO / 4 s | 232 | 1846.41 ms | 542.30 ms | 69.82 ms | 3.13x |
| 48 kHz / 48 BPO / 2 s | 463 | 3693.20 ms | 983.91 ms | 279.44 ms | 3.14x |
| 96 kHz / 24 BPO / 2 s | 232 | 1853.71 ms | 542.75 ms | 139.22 ms | 2.92x |

Each entry is a median of three. Plan cost is added once to either arm, since the
hybrid still builds a direct kernel for the control/fallback. This is **against a
new, simple direct CQT**, NOT the shipped FFT or optimized ASLT. No 3x product
speed claim. Very short inputs include edge-zero regions, especially with large
support; they do not establish sustained real-music throughput. Output traversal
and numerical/code comparison are outside both timers; initialization/rebases
are inside the hybrid timer. No measured RSS/commit peak in this batch.

The 135-row single-bin screen covers 3 rates x 3 densities x 5 frequencies x
3 repetitions. It includes a 2048-density example, unlike the bank screen. It
establishes the low/high-frequency crossover, not a universal route threshold.

Dense FFT comparator: 36 rows (2 rates x 2 densities x 3 frequencies x 3 reps),
four seconds each. It ranges from ~4.25x faster than direct at one low-frequency
case to ~5.27x slower at one short high-frequency case. Planning, kernel FFT,
allocation and all blocks included. Source FFT sharing across a full bank is
NOT implemented, so the result cannot decide the final grouped algorithm.

## 4. Correctness evidence and what it does NOT prove

- **10 tests pass debug and release**: nine CQT tests, one live-FFT test.
- CQT fixtures: silence, impulse, multi-tone, noise, chirp+noise, gated tone;
  44.1/48/96 kHz; edges/outside track; phase/gain; fractional sample hops;
  long loud-to-silence tail and rebases; true complex stereo cancellation.
- An analytical impulse and stationary tone frequency response check protects
  against a reference and optimized route merely sharing a sign/gain error.
- Final 180 CQT timing rows: maximum complex error **3.56e-14** relative to cached
  direct; no differing **research f64 quantizer** codes on these fixtures.
- Five compiled mutants rejected by named tests: reversed phase, wrong incoming
  index, missing Hann sideband, wrong gain, rounded-hop drift. Both original and
  final formatted-source mutant campaigns are retained. Compile errors are not
  counted as killed mutants.
- Clippy of the isolated crate: **zero warnings with -D warnings**.

There is **no pixel gate**, production level/quantizer call, actual cache-byte
comparison, actual whole-track/waterfall replay or GUI interaction test here.
The recurrence is mathematically equivalent but numerically different. Tiny
errors are not proof that no quantization boundary can ever move. Independent
reference comparisons are selective fixtures, not an exhaustive correctness proof.
No real-time scheduling, latency, playback, GPU, long full-bank run or ASLT
regression measurement is claimed. No statement of "highest possible quality".

## 5. Existing real-time FFT candidate

The isolated f32 probe reproduces sample copy -> windowed complex buffer ->
rustfft -> magnitudes from `process_realtime`. Reused storage preserves every
complex/norm bit across 5 sizes x 8 changing-input ticks, including a silent tail.
Measured dynamic allocation calls per warm transform: **4 -> 0**. These are
allocated bytes, not resident-memory savings: 160/320/640 KiB per transform at
4096/8192/16384 input samples. Persistent buffers retain memory instead of freeing
it every tick; the app tradeoff still needs measurement.

Final timing: 2000 warmup iterations per method, 5 alternating repetitions, each
1000 transforms, allocation counting disabled during timers:

| input samples | baseline median | reused median | scope |
|---|---:|---:|---|
| 4096 | 34.99 us | 28.46 us | strong baseline trend; no robust speed claim |
| 8192 | 65.88 us | 59.56 us | ~9.6% off this isolated stage |
| 16384 | 140.19 us | 130.28 us | ~7.1% off this isolated stage |

4096 baseline fell 47.03 -> 32.01 us across the final repetitions despite warmup;
the earlier screen had the same trend. Cause unconfirmed. No runs were removed.
This is not 7–10% off total player CPU; at 180 Hz the 8192 case saves only ~1.14
milliseconds of one core per second for this stage. Bars/mapping, lock contention,
L/R transforms, waterfall/render and frame scheduling are not timed here.

## 6. Reproduction

From this worktree, with build output outside the repository:

```powershell
$env:CARGO_TARGET_DIR='C:\Users\PC\Programming\moosik-v155-home\target'
cargo test --offline --locked --manifest-path research/v155/Cargo.toml
cargo test --offline --locked --manifest-path research/v155/Cargo.toml --release
cargo clippy --offline --locked --manifest-path research/v155/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --manifest-path research/v155/Cargo.toml --release
# cqt-probe.exe: geometry | bench | bank | fft
# rt-fft-probe.exe: no arguments
python research/v155/summarize_all.py C:/Users/PC/Programming/moosik-v155-home/final
python research/v155/window_response.py
# mutation_check.py --evidence <a NEW, private output directory>
```

Cargo.lock is for this isolated probe only; the application lock was not changed.
Dependencies are the existing vendored RustFFT and cached transitive packages;
all cargo commands used offline. Code at 6b34309 was rebuilt and all timing lanes
rerun in sequence for final evidence. Timed campaign ended 12:43:33 +08:00.
No overlapping owned build/test/model job during those timers. Ordinary machine
background services were not controlled; no CPU affinity or frequency lock.

## 7. Corrections kept visible / next work

First expanded probe failed to compile because Rust parsed a cast adjacent to `<`
as generics; parentheses fixed it. First clippy pass found range_contains and
items-after-test-module; both corrected without an allow. Initial and final
timings are retained. Formatting touched only the new research crate, not the app.

Proceed with the bounded next batch in INTEGRATION_PROPOSAL: actual app FFT
workspace test/patch, window-quality matrix, bounded grouped-input FFT bank,
then production code/display gates before CQT preprocess/UI integration. Do not
start a long soak or force a GPU route to manufacture more activity.

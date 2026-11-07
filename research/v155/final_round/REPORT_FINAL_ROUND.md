# 1.5.5 final performance round and compression verdict

2026-10-22, Claude. Branch `claude/v155-final-20261022` from `bcff980`
(recovery ref `refs/recovery/claude-v155-final-base-20261022`).
i7-9700F, Windows, release test builds, no player/model/compiler running
during timing. Synthetic inputs only. Raw lines are in this directory.

## Live-path candidates

| Candidate | Verdict | Evidence |
|---|---|---|
| 1. Exact per-side L/R FFT reuse | **Accepted** (`052ba13`) | paired probe below; wide screen |
| 2. Reuse mapped-bar output buffers | **Rejected** | 2048-bar collect 1.52 µs vs refill 1.49 µs: about 0.04 µs per tick, under 0.1% of the cheapest tick |
| 3. Shared edges in cold Flat/Gaussian geometry | **Accepted** (`360f2b2`) | cold build about 1.95× faster, bit-identical |

### 1. Per-side L/R reuse

Each side keeps its own norms and the exact window they came from. Reuse needs
the side's whole newest window to match bit for bit, a snapshot of the current
length and norms of the current padded length. Rebuilds clear both sides. The
complex scratch is still shared, so the test-only reference route disables
reuse. Smoothing and output run every tick. The production channel-FFT counter
counts only real transforms.

Paired probe (`lr-paired.jsonl`): the same binary with only channel reuse
switched off as the control. Mix reuse was on in both arms. CQT weighting,
15 alternating rounds of 64 ticks; ratio = control / candidate.

| Geometry | Fresh input (median ratio, IQR) | Two ticks per publication |
|---|---|---|
| 4096 / 44.1 kHz / 1024 / 2x | 1.000 (0.992–1.004) | 1.25× |
| 8192 / 48 kHz / 2048 / 2x | 0.993 (0.991–0.994) | 1.24× |
| 8192 / 48 kHz / 2048 / 8x | 0.982 (0.974–0.990) | 1.39× |
| 16384 / 96 kHz / 2048 / 8x | 0.990 (0.979–1.002) | 1.47× |

- **Fresh input:** costs up to about 2%, at most about 17 µs per joint tick. That is the comparison plus the snapshot copy.
- **Repeated windows:** save 20–32%.
- **All-identical ticks:** the wide screen's all-identical case (all 162 groups, `live-screen.jsonl`) is up to 6× faster. That is what a paused tap or bursty device periods produce.
- **Break-even:** about 5–7% of ticks repeated.
- **When repeats happen:** the tap publishes 512-frame batches, about 86–188 per second. The analyser runs at 60 fps by default, up to 240.
  - At 60 fps with steady publication, repeats are rare, so the cost (at most 2%) applies.
  - At higher fps, with bursty device periods, or while paused, the saving applies.

This is the same trade-off Codex's accepted Mix reuse already makes.

### 3. Cold geometry

Bar i's upper edge and bar i+1's lower edge are the same `freq_at` call on the
same `f32`, so n + 1 edges are computed once. The probe showed that 2n edge
inversions are about 95–99% of a cold Flat/Gaussian build.

| Bars / FFT / pad | Before (ms) | After (ms, 2 runs) |
|---|---|---|
| 1024 / 4096 / 2 | 3.57–3.59 | 1.80–1.83 |
| 2048 / 8192 / 8 | 7.16–7.41 | 3.65–3.87 |
| 2048 / 16384 / 8 | 7.37–7.47 | 3.68–3.80 |

The untouched CQT-weighting build is unchanged within ±2%; it serves as the
control. A cold build happens on settings changes, new geometry and slider
drags, where it rebuilds every tick. At 2048 bars that is now below a 144 Hz
frame (6.9 ms) instead of above it.

These are sequential before/after runs, not paired runs. The effect (about 2×)
is far outside run-to-run spread (under 2%).

### Exactness gates

- **Debug tests** (focused filter `realtime_padding live_workspace realtime_route channel overlap`): 124 passed, 0 failed, 2 ignored.
- **New test:** `each_channel_reuses_only_its_own_identical_window`. It covers:
  - held and changed sides;
  - swapped sides;
  - left equal to the old right window;
  - signed zero;
  - a same-size window rebuild;
  - a padding change;
  - cleared norms.
- **Mutants caught:**
  - comparing against side 0's snapshot fails at step 1;
  - dropping the norm-length guard panics on cleared norms;
  - an off-by-one shared edge fails 5 frozen-route/geometry tests.
- All mutants were restored before committing.

## Compression lane: closed, current v4 writer kept

What v4 is: 12-bit codes and a per-row frequency delta, which resets each row. The delta is wrapped to 12 bits and zigzagged, then split into a low-byte plane and a packed high-nibble plane, each compressed as one LZ4 block. The sidecar embeds two v4 payloads.

Only `lz4_flex` and `flate2` are available offline. Deflate decodes several times slower than LZ4, so it was not tried.

Screen (`codec-screen.jsonl`, `cache/codec_screen.rs`):
- **Inputs:** real production planes from 30 s of synthetic music-like, noise and silence audio, through FFT CQT-weighting, FFT Gaussian, ASLT and true CQT 48, at 1024 and 2048 bars.
- **Variants:** four reversible LZ4 predictors in one implementation:
  - frequency, which is v4's own predictor and reproduces v4's size exactly;
  - time;
  - gradient (left + up − up-left);
  - MED.
- **Checks:** every variant round-tripped exactly.

Sizes relative to v4:

| Predictor | Size |
|---|---|
| Gradient (best) | 0.85–0.89 (ASLT), 0.91–0.95 (true CQT), 0.95–1.01 (FFT) |
| Time | 0.93–1.18 |
| MED | 0.88–1.02 |
| Silence (all) | 1.000 |

Cost, gradient against the same-implementation frequency control:
- encode +12–40% (median +17%);
- decode +10–14% (median +13%).

Its extra per-cell work and the row-to-row dependency are inherent. Matching shipped v4 speed would need a separately optimised decoder, which would speed up v4 just as much. It would also need a new wire version for the Mix and the sidecar.

The rule allows no regression in either direction, and the cheapest stage already regressed in both. So v4 stays, no format or identity changed, and no full publication/load benchmark was needed.

- **Lead (not measured):** v4 decode spends far more than LZ4's own throughput on per-cell reconstruction. A same-format decoder speed-up could be a future, separate item.

## Not done / limits

- Native player CPU, cadence, audio, pixels and Linux remain untested.
- Real device publication cadence was inferred from code (512-frame batches), not measured.

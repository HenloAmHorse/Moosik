# 1.5.5: genuine CQT and measured performance research

Base: released `330b8acbea0a115f0d79e36053443e54a02386b4` (1.5.4).
Branch: `codex/v155-cqt-research`. This directory is research, not compiled by
the application. No product version/cache/UI changes until evidence supports them.

## Contract

- Do not modify main, the owner's dirty checkout/profile or the public repository.
- Do not merge, push, tag or release. Preserve earlier research and recovery refs.
- Preserve existing FFT and ASLT semantics. New CQT is a new transform, not a
  renamed FFT bin weighting; it needs a distinct persistent identity.
- No output-fidelity concession hidden as an optimization. New CQT candidates
  compare to their own independently evaluated definition, not to old FFT bars.
- Complex coefficients precede magnitudes, so Mix can be formed from `(L+R)/2`
  before magnitude without a third transform. No claim that ASLT's nonlinear
  geometric mean can be mixed after aggregation.
- Measure costs before long experiments. No soak to fill time. Stop rejected
  candidates; preserve their negative results.
- All timing claims name workload, precision, inclusion/exclusion and repetitions.
  Synthetic CPU microbenchmarks do not establish whole-app speed or GPU benefits.

## Stages and stop points

1. **Definition / geometry:** inspect existing consumers and identity. Define
   per-frequency windows, timestamps, edge handling, gain and Q. Separate the
   displayed bar grid from measured resolving power. Quantify bass support and
   real-time lookahead. Capping a constant-Q window makes the capped region
   variable-Q, and must be named as such.
2. **Independent reference / fixtures:** slow f64 direct evaluation, complex
   coefficients, analytical impulse/tone checks, edges, stereo cancellation,
   silence/noise/sweeps. Independently test frequency and time responses.
3. **Algorithm screen:** precomputed direct filters; Hann three-resonator sliding
   evaluation; grouped dense FFT convolution; then sparse/multirate candidates
   only with explicit error gates. Sparse thresholds and decimation are not exact
   merely because their errors look small. Bound drift, workspace and cancellation.
4. **Implementation proposal:** decide kernel/window and parameter semantics from
   the measured evidence. Preprocess integration first, distinct cache identity,
   Mix/L/R/Split, exact metadata/whole-track propagation. No reuse of `Cqt`/id 2.
5. **Performance screen:** 44.1/48/96 kHz, 12/24/48/96 bins per octave plus
   owner-like 1024/2048-column grids, 180 fps. Include n=1/n=9 ASLT controls only
   when making product comparisons. Account for setup, stereo, RAM, commit,
   cancellation and worker limits; GPU end-to-end, not isolated kernel timings.
6. **Realtime / existing paths:** allocations and copies in live Mix/L/R paths,
   scratch lifetime, measured p50/p95/p99 and frame deadlines, not only FPS.
   No silent change to smoothing, amplitude scales, phase or history consumers.
7. **Handoff:** decision table, raw evidence, reproducible commands, limitations,
   proposal/experimental code and a bounded next batch. Owner checklist only once
   an integrated prereview artifact is ready.

## Primary references and scope

- Judith C. Brown (1991), *Calculation of a constant Q spectral transform*:
  https://www.ee.columbia.edu/~dpwe/papers/Brown91-cqt.pdf . Geometric frequencies
  and inverse-frequency window support motivate the definition. The initial
  prototype uses an explicitly specified symmetric Hann, not Brown's Hamming
  implementation. Window choice is not claimed identical to that paper.
- Schorkhuber / Klapuri (2010), *Constant-Q Transform Toolbox for Music Processing*:
  https://zenodo.org/records/849741 . Efficient octave/downsampling family to
  investigate, not automatically an exact replacement for a full-rate kernel.
  PDF fetch failed in the initial pass; no detailed implementation claim rests on it.
- Holighaus et al., *A framework for invertible, real-time constant-Q transforms*:
  https://arxiv.org/abs/1210.0084 . Nonstationary Gabor / sliced alternatives.
  Invertibility is useful future work, not promised by this analysis-only prototype.

## Initial research definition (not a product default)

`f[k]=f_min*2^(k/B)`, `Q=1/(2^(1/B)-1)`,
`h[k]=ceil(Q*sample_rate/(2*f[k]))`, support `j=-h..h`.
`w[j]=0.5+0.5*cos(pi*j/h)`. Complex coefficient:
`2/sum(w) * sum(x[t+j]*w[j]*exp(-i*2*pi*f[k]*j/sample_rate))`.
Samples outside the track are zero; normalization is the full window (no edge
boost). Frame centres are `floor(frame*sample_rate/fps)`, not an accumulated rounded
hop. Gain is unit positive-frequency sinusoid amplitude away from boundaries;
finite negative-frequency leakage is measured, not assumed absent. `Q` here sets
support, not the Hann's measured -3dB bandwidth. Centered online evaluation needs
lookahead `h/sample_rate`; a causal variant changes timestamp/latency semantics.

Optional maximum support is a separate **capped/variable-Q** experiment. 1024 bars
do not mandate 1024 independent musical bins. Density and display aggregation
must be explicit; arbitrary frequency scales are not automatically constant-Q.

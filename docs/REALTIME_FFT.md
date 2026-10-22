# Live FFT and display density (1.5.5)

The Bars control supports **16–2048 bars**. The default remains 1024. A window
too narrow to show them separately folds source bars at draw time; increasing
the setting does not manufacture screen pixels. In Superlet, bar density also
affects target Q, so 2048 can cost more than twice the analysis of 1024 when
support is not already capped. Stored-analysis identities include bar count:
existing 1024-bar caches are unchanged, not silently upsampled into 2048.

## Independent live padding

In Real-time, or the **Live FFT preview** fold while a pre-process analysis is
pending, **Live zero-padding** offers 1x, 2x, 4x and 8x. This is separate from
**Pre-process zero-padding**. In older builds the live path always used 2x,
even when the pre-process control showed a larger number.

Padding samples the same windowed transform more densely. It does not lengthen
the input window, resolve arbitrarily close tones, or bypass the time/frequency
trade-off. Higher factors cost more CPU and workspace. FFT window length and
shape still control the independent resolving power and temporal response.

A live-only change does not restart an analysis, invalidate its cache, change
playback, or modify CQT/Superlet settings. Active live histories are reset because
their bin geometry changed. A completed, paused cache and its channel display
stay in place. Saved live settings are restored independently; older profiles
without this field retain the shipped 2x live behavior.

The raw half-spectrum, bars, left/right, octave meters and rolling spectrogram
all use the actual live FFT length for frequency mapping. The auxiliary meters
retain their existing 2x-window level reference, separately from bin spacing, so
choosing denser padding does not introduce a spurious level drop. This preserves
the old scale; it is not a new calibration or a dBFS claim.

## Exact geometry reuse

The FFT constant-Q weighting already retained its frequency gather plan. Flat
and Gaussian overlap now retain their frequency boundaries and weights too.
Only audio-independent geometry moves out of the tick: interpolation, products,
ascending accumulation, division and level conversion keep the same arithmetic.
Changing bar count, padded length, rate, frequency range, frequency-scale
parameters or weighting selects the corresponding geometry. Interpolation and
equal-loudness corrections still use the current setting on every tick.
Building that geometry evaluates each shared bar edge once (n + 1 frequency-scale
inversions instead of 2n), with the same expression and therefore the same bits.
A cold 2048-bar Blend 0.5 build took about 3.6–3.9 ms instead of 7.1–7.5 ms in
the final-round probe; this is a one-off on a settings change or new geometry.

The live PCM source and display ticks run on independent clocks. If the entire
current input window is bit-for-bit identical to the retained snapshot and the
current FFT/norm buffers are valid, Mix reuses those FFT results. This is an
exact comparison, including signed zero, not a hash or similarity threshold.
A window/size/padding rebuild clears the snapshot; a cleared norm buffer forces
a transform. Mapping, smoothing, peaks and history still advance on every
accepted tick. Left and right do the same, each against its own snapshot and
norms (never the other side's or the Mix's); their smoothing still runs every
tick. The benefit depends on the source publication cadence (512-frame batches,
about 86–188 per second) against the analyser rate (60 fps by default, up to
240): fresh PCM on every tick cannot gain, and pays the exact comparison and
snapshot copy instead — under 2% of a joint tick in the paired probe.

## Memory and verification limits

The cache reader retains its previous **512,000,000-cell** ceiling; supporting
2048 bars does not double its permitted allocation. Other existing file-byte,
frame and transform-workspace limits remain. Long, dense analyses can be refused
instead of silently reducing bars, frames or precision.

Headless FFT and analyser tests are not whole-player CPU measurements or native
GUI/pixel tests. Same-configuration bit comparisons are numerical regression
gates; comparing 2x with 8x is comparing different sampled FFT configurations.
The performance report records its exact scope and baseline rather than
attributing previously shipped workspace or norm-dedup savings to this round.

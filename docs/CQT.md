# True CQT (1.5.5)

Select **Spectrum → Analysis → Analysis method → True CQT**. This is separate
from **FFT · constant-Q weighting**, which keeps its old meaning and caches.
The new method is pre-process only. Selecting it switches to Pre-process;
FFT-only visualizations switch to Waterfall. While analysis is pending, ordinary
plots show a clearly labelled **LIVE FFT PREVIEW**, using the existing FFT
constant-Q weighting and the live PCM sample rate. It is not a CQT, is never
stored as one, and is unavailable without usable live PCM (e.g. native DSD/DoP).
Whole track waits for completed CQT instead. Arrival resets preview history and
switches to the stored CQT clock/grid. Live FFT preview controls do not invalidate
the CQT cache. If no matching cache exists, use the normal Re-analyze action.
Playback audio is not changed.

## Definition

For each displayed frequency `f`, `B` is the support-resolution setting:

```
Q = 1 / (2^(1/B) - 1)
h = ceil(Q * sample_rate / (2*f))
C(f,t) = 2/sum(w) * sum(x[t+j] * w[j] * exp(-i*2*pi*f*j/sample_rate)), j=-h..h
```

The Hann, Hamming, Blackman and four-term Blackman–Harris windows span this full
symmetric support. There is no maximum-window truncation. `Q` describes support,
not the measured −3 dB bandwidth; windows have different mainlobes/sidelobes.
Outside the track, samples are zero, with the full-window normalization retained.
Edges therefore fade; missing samples are not renormalized into a louder result.

The `bar_count` frequencies are log cell centres across
`min_freq .. min(max_freq, sample_rate/2)`. Bar count samples the picture;
**bins/octave sets filter support, not the number of displayed bars**. A denser
picture does not itself resolve more tones. The default is 48 bins/octave,
Blackman, 180 analysis frames/s. At 20 Hz that is about 3.44 s of full support,
with roughly 1.72 s on either side of the centre: sharp bass is not instant bass.
No setting is universally highest quality for every purpose.

Frame `k` is labelled `k/fps`; its sample centre is `floor(k*sample_rate/fps)`.
Integer spacings may alternate. In particular, 48 kHz at 180 fps does **not**
become a drifting fixed 266-sample-hop clock. Whole track discloses the nominal
time and actual sample centre. Raising fps samples the same windows more often;
it does not shorten their temporal response.

## Bass response: Full or Responsive

Long bass support is the definition, not an unfinished optimization: at 20 Hz
Full CQT's support is about 0.84 / 3.58 / 13.8 s at 12 / 50 / 192 bins/octave.
**Bass response → Responsive** selects a separate variable-Q transform (VQT,
Schörkhuber et al. 2014) with a bass limit `t` (selection default 0.25 s, 0.05–10 s
accepted):

```
bandwidth = f/Q + 1/t        support = 1/(f/Q + 1/t)   (< Q/f and < t)
h = ceil(sample_rate / (2*(f/Q + 1/t)))
```

Everything else (window, gain, centring, clock, edges, Mix, engine) is the same.
The owner approved Responsive for release with the 0.25 s selection default.
Existing saved Responsive limits, including 0.5 s, are not changed. Selecting
Full and then Responsive again initializes the limit to the selection default;
there is not a separate remembered limit for each bass-response mode.
Where `Q/f` is much shorter than `t` (the treble) it approaches Full CQT; in the
bass it reacts sooner and separates nearby pitches less. It is a different
method with its own identity (`map=vqt`, `vqt-rev`, `bass-limit`) and cache;
Full CQT records are byte-identical to before and are never reused for it.

Measured on stored codes (before display smoothing), 48 kHz, Blackman, 40 Hz
gated tone and a one-semitone pair
([report](../research/v155/REPORT_BASS_RESPONSE.md)):

| Setting | ±time at −20 dB | −3 dB width | semitone dip |
|---|---:|---:|---:|
| Full 48 | 0.36 s | 36 cents | 24 dB |
| Full 192 | 1.59 s | <12 cents | 45 dB |
| Responsive 192, 0.5 s | 0.094 s | 132 cents | 3.3 dB |
| Responsive 192, 0.25 s | 0.050 s | 288 cents | 0.2 dB |
| ASLT n=1 (owner's usual) | 0.037 s | 351 cents | 0 dB |

These measured 40 Hz settings show a time–frequency trade-off (time × width is
about 13–15 here), not a universal optimum for all transforms. None of the
listed settings is simultaneously fastest and sharpest in the bass. At 640 Hz, Responsive 192
still resolves a semitone by about 47 dB, where Full 12 does not resolve it.

## Implementation, channels and display

Dense f64 overlap-save convolution shares a source FFT between filters of the
same transform size and shares the kernel between Left/Right. It is not a sparse,
recursive, thresholded or downsampled approximation. The independent direct-f64
oracle tests compare actual complex responses and the shipped quantized codes;
passing those fixtures is not a universal floating-point equality proof.

A joint pass transforms Left and Right only. Mix is `abs((C_left+C_right)/2)`
before magnitude, not the average of magnitudes, so opposite polarities cancel.
The CPU implementation evaluates only needed inverse-FFT outputs where the
existing plan supports exact requested-output execution, keeping its operation
graph, FFT size and block geometry. Unsupported/dense/duplicate schedules keep
the whole inverse. Independent frozen-prereview tests compare actual complex
bits as well as codes; these finite fixture gates are not a universal proof.
Window values are reused without changing their arithmetic.

Independent filter inverses can run on the existing selected analysis-worker
pool. Explicit source, product, scratch, requested-output, coefficient and
kernel buffers are admitted together under the same 256 MiB allowance; tight
budgets reduce the batch/worker count or keep the serial route, not the quality.
Calls outside an explicit analysis pool do not create global-pool work. Planner
internals and Rayon scheduling metadata remain outside the payload allowance.
The same bounded route now admits transforms down to 512 points; smaller
transforms remain serial. This changes where independent work runs, not its
arithmetic or support. It can finish sooner by using more of the selected
workers, rather than reducing instantaneous CPU usage. See the
[prereview measurements](../research/v155/REPORT_PREVIEW_PHASES.md) for the
specific workloads and limits of the speed claims.
For admitted groups, independent kernel construction/forward FFTs also reuse
these slots' existing scratch, joining before source transforms borrow it.
The ordered bank, each kernel's arithmetic and the payload allowance are
unchanged; no-workspace groups remain serial. The separate
[kernel-preparation follow-up](../research/v155/REPORT_KERNEL_PREPARATION.md)
records the incremental paired measurements and cancellation/exactness gates.
GPU and multirate CQT are not implemented. New speedups need paired whole-workload
measurements; a pruning-capable call counter alone is not a speedup measurement.

Finished magnitudes use the existing Superlet display level reference and v4
12-bit level codes. Those are visualization levels, **not calibrated dBFS or an
invertible/phase-preserving analysis archive**. Whole track inspects those stored
cells exactly, subject to the existing display aggregation policy. Future phase,
harmonic or spatial tools need an explicitly suitable analysis data path rather
than treating this compressed display cache as lossless complex data.

## Memory, cancellation and compatibility

- Explicit dense FFT/kernel/scratch payload is limited to 256 MiB; returned code
  planes to 1536 MiB total; each FFT to 2²² points. These are allocation-payload
  bounds, **not peak process RAM**. Decoded inputs, allocator overhead and rustfft
  plan internals are outside them; only one size group's plans are retained.
- Oversized requests fail with a reason instead of shortening bass windows,
  lowering BPO, or substituting FFT weighting. Very high BPO/low frequencies or
  long stereo tracks can hit these limits. Reduce an explicit setting if wanted.
- Cancellation is checked during fills and between FFT work. A single rustfft
  call, allocation or code-plane zero-fill is not interruptible; no hard latency
  guarantee is claimed. Cancelled/refused work publishes no partial cache head.
- True CQT has its own method/revision/window/BPO/fps/clock/grid identity. FFT and
  Superlet keys are unchanged; this feature does not retire their existing caches.
- The old `Cqt` profile value still means FFT weighting. New settings remember
  True CQT independently and hide inapplicable FFT/ASLT controls. Do not run an
  older executable against a new profile without a settings backup: old builds
  cannot understand a newly added enum variant.

Released in 1.5.5.

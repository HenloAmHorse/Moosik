# 1.5.5 bass response: Full CQT, Responsive (variable-Q) and ASLT

2026-10-21, Claude takeover round. Source: branch `claude/v155-bass-20261021` from `6903b8a`.
Feature commit `64f6a92`; probe `src/spectrum/bass_response.rs`; raw lines in
`research/v155/bass_response/bass_response.jsonl` (one JSON object per arm × fixture).

## Question

The owner reports slow-moving bass, especially at high bins/octave. Is that the
transform's definition or an implementation defect, and what can be offered
without silently weakening Full CQT?

## Method

- The production pre-process entry (`preprocess_file`): decode, analysis, encode
  and private-cache publication. Mono content (L = R), Mix plane, 1024 bars,
  20 Hz–24 kHz, 48 kHz, 180 fps, six workers, GPU off. Release test build.
- The stored code plane is measured: 12-bit levels, 80 dB over 4095 codes, so
  code differences are dB differences. This is the definition plus quantization,
  **before display smoothing/history**. Display effects are not included here.
- Frame times are each frame's actual centre sample (ASLT `k·hop`, CQT
  `floor(k·sr/fps)`).
- Fixtures, 40 s, at least 15 s of silence around every event (more than half the
  longest support, 6.9 s):
  - gate: 40/160/640/2560 Hz tones, 0.2 FS each, on 15–25 s. Gives the onset
    lead and release in time (time from the true edge to −20/−40 dB of the
    steady level; symmetric windows make lead = release), and the steady-state
    frequency width (contiguous bars within 3/20 dB of the peak, in cents).
  - pair: tones one semitone apart at 40, 160 and 640 Hz. Dip = weaker peak
    minus the lowest level between them (0 = not resolved).
  - click: **invalid as designed, kept in the raw lines.** A single sample has
    too little energy in a narrow band (about −100 dB at 40 Hz for Full 192),
    so every bass bar sat at the code floor and its "widths" are meaningless.
    The gate's step response already gives the temporal response.
- ASLT is the owner's usual setting: Blend(0.5) scale, q_ratio 2, max window
  0.25 s, spread 0.5, n = 1 or 9. CQT/VQT use Blackman. These are selectable
  settings, not matched settings; the widths below are what make a match.
- Width resolution is limited by the 1024-bar grid (about 12 cents at this
  range for the log grid; "0" means narrower than one bar step). ASLT's
  Blend grid is finer in the bass and coarser in the treble than the log grid.

## Findings

1. **Full CQT behaves exactly as defined.** The −20 dB half-spread is about
   0.2 × support and the −3 dB width about 1.6/support Hz, within a few
   percent across 12/48/192. At 40 Hz: ±0.083 s (12), ±0.36 s (48), ±1.59 s
   (192). The slow bass at high BPO is the definition, not a defect.
2. **The measured settings exhibit a time–frequency trade-off.** At 40 Hz, the
   −20 dB spread (s) × −3 dB width (cents) is about 13 for ASLT n=1, 13 for
   Full 12, 12.8 for Full 48 and 14–15 for the Responsive settings. This is
   one fixture/window/grid comparison, not proof of a universal frontier for
   all transforms, signals or response metrics. ASLT reacts
   faster because its 0.25 s maximum window blurs bass pitch by about
   3.5 semitones (351 cents), not because it escapes the trade-off.
3. **Responsive (VQT) primarily shortens low-frequency support.** Its support
   is `1/(f/Q + 1/t)`. It changes every frequency; high frequencies approach
   Full only when `Q/f` is much shorter than `t`.
   - At 40 Hz, Responsive 192 at 0.5 s responds like Full 12 (±0.094 s,
     132 cents).
   - At 640 Hz it still resolves a semitone by 47 dB, where Full 12 shows no
     dip at all.
   - At 0.25 s it approaches ASLT n=1 in the bass (±0.05 s, 288 cents).
4. **Full CQT is unchanged.** Its code path keeps the original expression,
   and its records are byte-identical. Production Mix/L/R plane SHA-256 for
   12/48/192 (`full_cqt_identity.jsonl`) equal the values recorded for
   `6081ef6` in `kernel-prep-20261021/PRODUCTION_COMPARISON.json`.
5. **Responsive runs were also quicker to compute** (shorter kernels):
   192 BPO took about 3–4 s per 40 s mono fixture, against about 11 s for
   Full 192. These are single uncontrolled observations, including test
   start-up, **not a benchmark**.

## Policy (owner decision received 2026-10-21)

- Full CQT stays the default and is unchanged.
- "Bass response: Responsive" is an explicit, separately cached choice. The
  slider covers 0.1–2 s; profiles accept 0.05–10 s. The default limit when
  selected is now 0.25 s. Existing saved limits remain unchanged. Selecting
  Full and then Responsive again initializes the limit to that default.
- The owner approved retaining Responsive in the formal release. The measured
  0.5 s rows below remain evidence for that configuration, not the new default.
- Further CQT research is paused; next work targets other modes and live FFT.

## Limits

- One window (Blackman), one sample rate, synthetic steady tones and edges.
  This does not include display smoothing, real music, transients with
  harmonics, or listening tests.
- The VQT is the standard bandwidth-offset form with full windows. It is not a
  multirate, GPU or sparse implementation.
- Gates for this round are recorded in the handoff, not here.

## Tables (generated from the raw lines)

### 40 Hz

| Setting | ±s at −20 dB | ±s at −40 dB | −3 dB width (cents) | −20 dB width (cents) | semitone dip (dB) |
|---|---:|---:|---:|---:|---:|
| ASLT n=1 | 0.0368 | 0.0702 | 350.8225 | 958.8834 | 0.0 |
| ASLT n=9 | 0.0257 | 0.0479 | 448.6569 | 1257.4759 | 0.0 |
| Full 12 | 0.0833 | 0.1444 | 155.8297 | 395.5676 | 2.3248 |
| Responsive 12, 0.25 s | 0.0333 | 0.0555 | 443.5152 | 1090.8076 | 0.0 |
| Full 48 | 0.3555 | 0.6055 | 35.9607 | 83.9083 | 24.1856 |
| Responsive 48, 0.5 s | 0.0778 | 0.1333 | 179.8035 | 431.5283 | 1.7973 |
| Responsive 48, 0.25 s | 0.0444 | 0.0778 | 311.6593 | 779.1483 | 0.0781 |
| Full 192 | 1.5889 | 2.5278 | 0.0 | 11.9869 | 45.4799 |
| Responsive 192, 0.5 s | 0.0944 | 0.1611 | 131.8559 | 359.6069 | 3.2821 |
| Responsive 192, 0.25 s | 0.05 | 0.0833 | 287.6855 | 707.2269 | 0.1758 |

### 160 Hz

| Setting | ±s at −20 dB | ±s at −40 dB | −3 dB width (cents) | −20 dB width (cents) | semitone dip (dB) |
|---|---:|---:|---:|---:|---:|
| ASLT n=1 | 0.0368 | 0.0702 | 73.2274 | 219.6975 | 2.1294 |
| ASLT n=9 | 0.0257 | 0.0479 | 102.5191 | 293.2313 | 0.6447 |
| Full 12 | 0.0167 | 0.0333 | 167.8166 | 395.5676 | 0.0 |
| Responsive 12, 0.25 s | 0.0111 | 0.0222 | 227.751 | 563.3842 | 0.0 |
| Full 48 | 0.0889 | 0.15 | 23.9738 | 95.8952 | 17.9341 |
| Responsive 48, 0.5 s | 0.0444 | 0.0778 | 71.9214 | 167.8166 | 3.1453 |
| Responsive 48, 0.25 s | 0.0278 | 0.0555 | 95.8952 | 263.7117 | 0.2735 |
| Full 192 | 0.3667 | 0.6111 | 0.0 | 11.9869 | 47.4921 |
| Responsive 192, 0.5 s | 0.0778 | 0.1333 | 35.9607 | 95.8952 | 14.5153 |
| Responsive 192, 0.25 s | 0.0444 | 0.0722 | 71.9214 | 179.8035 | 2.6569 |

### 640 Hz

| Setting | ±s at −20 dB | ±s at −40 dB | −3 dB width (cents) | −20 dB width (cents) | semitone dip (dB) |
|---|---:|---:|---:|---:|---:|
| ASLT n=1 | 0.0368 | 0.0702 | 11.2001 | 56.0004 | 47.3553 |
| ASLT n=9 | 0.0257 | 0.0479 | 11.2001 | 67.1706 | 31.3162 |
| Full 12 | 0.0 | 0.0055 | 155.8297 | 407.5545 | 0.0 |
| Responsive 12, 0.25 s | 0.0 | 0.0055 | 179.8035 | 431.5283 | 0.0 |
| Full 48 | 0.0222 | 0.0333 | 23.9738 | 95.8952 | 14.2418 |
| Responsive 48, 0.5 s | 0.0167 | 0.0278 | 47.9476 | 107.8821 | 7.4628 |
| Responsive 48, 0.25 s | 0.0111 | 0.0222 | 47.9476 | 131.8559 | 3.79 |
| Full 192 | 0.0889 | 0.15 | 0.0 | 11.9869 | 45.4799 |
| Responsive 192, 0.5 s | 0.0444 | 0.0778 | 0.0 | 23.9738 | 47.2186 |
| Responsive 192, 0.25 s | 0.0278 | 0.0555 | 23.9738 | 47.9476 | 47.6093 |

### 2560 Hz

| Setting | ±s at −20 dB | ±s at −40 dB | −3 dB width (cents) | −20 dB width (cents) | semitone dip (dB) |
|---|---:|---:|---:|---:|---:|
| ASLT n=1 | 0.0257 | 0.0479 | 0.0 | 9.8993 | — |
| ASLT n=9 | 0.0257 | 0.0424 | 0.0 | 9.8993 | — |
| Full 12 | 0.0 | 0.0 | 155.8297 | 395.5676 | — |
| Responsive 12, 0.25 s | 0.0 | 0.0 | 155.8297 | 407.5545 | — |
| Full 48 | 0.0055 | 0.0055 | 35.9607 | 83.9083 | — |
| Responsive 48, 0.5 s | 0.0 | 0.0055 | 35.9607 | 95.8952 | — |
| Responsive 48, 0.25 s | 0.0 | 0.0055 | 35.9607 | 95.8952 | — |
| Full 192 | 0.0222 | 0.0389 | 0.0 | 11.9869 | — |
| Responsive 192, 0.5 s | 0.0167 | 0.0278 | 0.0 | 23.9738 | — |
| Responsive 192, 0.25 s | 0.0111 | 0.0222 | 0.0 | 23.9738 | — |

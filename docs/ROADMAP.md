# Roadmap

The owner fixed the next three versions. Each is a scope, not a date.

## 1.5.4 — interaction, follow, lyrics, local translation (released 2026-10-19)

- Whole-track spectrogram interaction and native-window performance.
- Correct continuous playback follow and intuitive controls.
- Lyric quality-of-life improvements already made (text size, setup card,
  model reuse).
- Translation adapter corrections.
- Better downloadable local translation models.
- Hardware-aware model and configuration recommendations.
- Explicit lower-RAM 30B GPU route; seeded drafts, saved custom instructions,
  feedback revisions, locked rows and bounded session history.

## 1.5.5 — real CQT and another performance campaign (released 2026-10-22)

- A real constant-Q transform analysis, not an FFT mapped onto a
  constant-Q-like axis under a new name.
- Another performance campaign across FFT, CQT and ASLT, for both
  pre-processing and real-time.
- Reuse the improvements already shown to work (planning, buffers,
  scheduling, data movement, storage, rendering) where they genuinely apply.
- Shipped: centred full-support CQT and a Responsive variable-Q option, each
  with its own cache identity, stereo and Whole track integration; up to 2048
  bars; independent live padding; exact live FFT buffer, geometry and window
  reuse; common audio formats with native Opus. [Scope and limitations](CQT.md).

## 1.6.0 — stereo and polyphonic analysis (not started)

- Stereo / spatial analysis.
- Polyphonic pitch, harmonic and chord analysis.
- Keep tools inside Spectrum with progressively disclosed controls rather
  than adding another top-level analysis window. These are plans, not shipped
  harmonic/chord or physical-room reconstruction features.

1.6.0 remains unimplemented. No chord/spatial framework is added in anticipation
of it; the CQT display cache contains level codes, not retained complex phase.

## Internal batch numbers

Working folders and branches such as `moosik-155-*` or `claude/155-*` carry
an internal **batch** number (the 155th work batch), not a product version.
"Batch 155" produced work for 1.5.4; it is not version 1.5.5. Folders for
1.5.4 work from here on are named `moosik-v154-*`.

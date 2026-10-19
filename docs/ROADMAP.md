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

## 1.5.5 — real CQT and another performance campaign (not started)

- A real constant-Q transform analysis, not an FFT mapped onto a
  constant-Q-like axis under a new name.
- Another performance campaign across FFT, CQT and ASLT, for both
  pre-processing and real-time.
- Reuse the improvements already shown to work (planning, buffers,
  scheduling, data movement, storage, rendering) where they genuinely apply.

## 1.6.0 — stereo and polyphonic analysis (not started)

- Stereo / spatial analysis.
- Polyphonic pitch, harmonic and chord analysis.
- Keep tools inside Spectrum with progressively disclosed controls rather
  than adding another top-level analysis window. These are plans, not shipped
  harmonic/chord or physical-room reconstruction features.

Nothing for 1.5.5 or 1.6.0 is built ahead of time; no general framework is
added in anticipation of them.

## Internal batch numbers

Working folders and branches such as `moosik-155-*` or `claude/155-*` carry
an internal **batch** number (the 155th work batch), not a product version.
"Batch 155" produced work for 1.5.4; it is not version 1.5.5. Folders for
1.5.4 work from here on are named `moosik-v154-*`.

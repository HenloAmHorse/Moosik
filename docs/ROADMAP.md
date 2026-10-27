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

## 1.6.0 — redesigned interface, stereo and polyphonic analysis (released 2026-10-27)

A large release in two parts.

**Interface.** A designed look and motion for the whole player: one player
with three modes (Listen, Library, Analyze). Analyze keeps the spectrum's full
resolution (up to 2048 bars, Superlet, CQT) and its existing colour ramps; the
redesign changes the frame around it, never the rendering. Done in this
order:

1. Inventory the current interface: every window, control and state, a
   screenshot matrix across languages and widths, CJK font issues, and a
   baseline of per-frame CPU/GPU cost.
2. Choose a visual direction, then a design system (colour, type, spacing,
   components, motion timings), then high-fidelity key screens.
3. Build one screen in egui, compare it with the design and measure it.
4. Only if measurement shows a need: research partial redraw on the vendored
   egui/eframe (cached layers for unchanged regions, dirty-rectangle
   presentation). A player must stay light on resources.
5. Apply the design screen by screen.

It stays native: egui first, Slint only if an egui prototype fails; no web UI.
Animation runs while something is interacting or already redrawing; an idle
window does not redraw.

**Analysis.**

- **Done:** a Stereo panel (mid/side scope, width, balance, correlation and
  ten octave bands) and a Harmony panel (chord now and next, key, note wheel)
  inside Analyze, not another top-level window. Chords come from a Rust port
  of lv-chordia, downloaded on request, with confidence and user correction;
  without it the panel guesses live. See [HARMONY.md](HARMONY.md).
- **Not in 1.6.0:** note-by-note (polyphonic pitch) transcription. A
  fine-tuned note model exists in research but is not shipped; publishing its
  weights needs the owner's decision. Score transcription (chord charts,
  MIDI, MusicXML, tabs) is deferred until recognition is solid. No
  physical-room reconstruction is claimed.
- The owner accepted the chord model's accuracy for this release and is not
  pursuing more of it for now.

## 1.6.1 — YouTube source (planned, timing open)

- Search, stream and download from YouTube and YouTube Music through yt-dlp,
  installed as an optional component the user downloads, with one-click
  updates (YouTube changes often break it).
- Audio stays as delivered (Opus or AAC, lossy) and is labelled lossy; Moosik
  already decodes both natively.
- The 1.6.0 Library design reserves a place for sources; nothing is built for
  YouTube in 1.6.0.

## 1.6.2 — Pro audio tools (planned)

- **What a file really is:** lossy-transcode ("fake lossless") detection from
  the spectrum's cut-off, upsampled ("fake hi-res") and padded ("fake 24-bit")
  detection, and FLAC MD5 verification of the decoded audio.
- **Measurement:** true peak (ITU-R BS.1770, oversampled), loudness range,
  PLR/PSR, DR and crest factor, clipping and inter-sample over counts marked on
  the timeline, a goniometer beside the correlation meter (the 1.6.0 Stereo
  panel already has a mid/side scope), DC offset and polarity.
- **Listening:** headphone EQ from AutoEq profiles into the parametric EQ,
  crossfeed, and room correction from REW or FIR files. All off unless switched
  on; with them off, output stays bit-perfect.
- **Comparing:** level-matched A/B, ABX blind tests with their statistics, and
  a spectrum difference between two masters of one song.
- **Musical tools**, once recognition is solid: tempo and beats, sections, and
  speed or pitch change for practice.

## Internal batch numbers

Working folders and branches such as `moosik-155-*` or `claude/155-*` carry
an internal **batch** number (the 155th work batch), not a product version.
"Batch 155" produced work for 1.5.4; it is not version 1.5.5. Folders for
1.5.4 work from here on are named `moosik-v154-*`.

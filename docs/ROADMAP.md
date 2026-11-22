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

## 1.6.1 — YouTube source (released 2026-11-08)

Shipped: search (YouTube and YouTube Music songs), play, queue and download
through yt-dlp with Deno, both optional components fetched on request; songs
play while they download; music videos behind Listen and the spectrum, 480p
to 2160p (see [YOUTUBE.md](YOUTUBE.md)). Also: QQ Music as a third lyrics
source and lyrics looked up as a song starts, spectrum bars drawn to each
song's loudness, and the experimental
[exact NVIDIA assist](ASLT_EXACT_GPU.md) for the superlet pre-process (off
by default). The scope as planned:

- Search, stream and download from YouTube and YouTube Music through yt-dlp,
  installed as an optional component the user downloads, with one-click
  updates (YouTube changes often break it).
- Audio stays as delivered (Opus or AAC, lossy) and is labelled lossy; Moosik
  already decodes both natively.
- The 1.6.0 Library design reserves a place for sources; nothing is built for
  YouTube in 1.6.0.

## 1.6.2 — Pro audio tools, faster analysis, smaller caches (released 2026-11-23)

Shipped: the pro audio tools (see [PRO_AUDIO.md](PRO_AUDIO.md)), and two
experimental options that are off by default: High-accuracy GPU for the
superlet pre-process, and a smaller, lossless form for new analysis cache
files. Of **Less disk** below, that cache form and Settings → Storage
shipped: what each part takes, and removing models, unfinished model
downloads and the YouTube cache. Also shipped: the DR figure in Measure, a
headphone EQ that searches AutoEq and saves for other apps, 32 EQ bands,
a Pro tools section in Settings, and exact speed-ups across the analysis
(see the changelog). The goniometer is the existing Phase view. The scope
as planned:

- **Less disk:** what the analysis cache, YouTube's cache and videos,
  downloads and models take, and trimming it.
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

## 1.6.3 — A real library (proposed; the owner has not chosen yet)

A draft for the owner to cut or reorder. Moosik's Library is still built
around playlists; this release would make it a music library.

- **Watched folders:** choose music folders once; Moosik scans them in the
  background (reusing the analysis cache's file identity, so a moved or
  retagged file is noticed) and keeps up with changes.
- **Browse by album, artist and genre:** an album grid with covers, artist
  pages, a library-wide search, and Play / Queue from any of them.
- **Smart playlists:** most played, recently added, never played, by
  loudness or tempo (from 1.6.2's measurements), kept up to date.
- **YouTube, continued:** import a YouTube or YouTube Music playlist into a
  Moosik playlist, play while fetching instead of after, and a "more like
  this" radio from a result.
- **Practice, continued:** loop a section or a number of bars (from 1.6.2's
  beats and sections), a count-in, and an optional click on the beat.
- **Notes (if the owner decides to publish the weights):** the fine-tuned
  note model behind a download button like the chord model's, and a piano
  roll under the spectrum.
- **Housekeeping:** the AccessKit misrouting of the classic toolbar's
  transport buttons; an offscreen screen test for every mode in both themes.

Not planned for 1.6.3: chord charts, MIDI, MusicXML or tabs (score
transcription stays deferred), and any web interface.

## Internal batch numbers

Working folders and branches such as `moosik-155-*` or `claude/155-*` carry
an internal **batch** number (the 155th work batch), not a product version.
"Batch 155" produced work for 1.5.4; it is not version 1.5.5. Folders for
1.5.4 work from here on are named `moosik-v154-*`.

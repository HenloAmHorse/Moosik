# 1.6.0 interface inventory

2026-10-23, Claude. Source: `claude/v160-ui` at 1.5.5 (`dac995c`) plus the
child-window AccessKit patch. Screens were captured from a sandboxed profile
with eight synthetic tracks (no owner content), driven only through UI
Automation. Screenshots, UIA dumps and the driver live outside the repository
in `moosik-v160-home` (`shots/matrix`, `steps/`, `scripts/ui.ps1`).

The interface text is English only; it has no UI translation. The language
axis is therefore *content*: titles, artists and lyrics in English, Japanese,
Traditional and Simplified Chinese and Korean, plus deliberately overlong
strings.

## Windows and surfaces

| Surface | Kind | What is on it |
|---|---|---|
| Main | root window, 816×639 default | now-playing header; saved-playlist combo and Save As; playlist title and Clear All; filter field; six sort buttons; playlist rows (drag handle, cover, number, title, artist, duration, play-count option); waveform seek bar with times; transport; toolbar; five status lines |
| Spectrum Analyzer | child window | mode (Real-time / Pre-process); nine views; six channel views; Stereo by default; loudness (Flat / ISO 226); palette; EQ; Debug; Bars slider and field; Gap slider and field; panels Analysis, Peak Hold, Album Art, Cache; plot with LUFS and fps readouts; analysing progress with Abort; Whole track view; inline help paragraphs |
| Lyrics | child window | Find lyrics, Search…, Paste…, Sync…, Save, Translate…; Follow; text size slider; offset field; lines with current-line highlight; Japanese: Furigana / Romanization / Both / Original; Korean: Romanization; translation sheet; sync editor (Stamp, Back, Clear all times, + Line); search and paste panels; empty state |
| Tags | child window | tag type; 12 fields; All tags disclosure; Save / Revert |
| Info | child window | File, History, Stream, Tags, Inferred sections as label/value grids |

### Main-window popups

| Opened by | Contents |
|---|---|
| 🎨 Look | Theme Dark/Light; font filter and system font list; text size with Apply/Reset; tint UI with album-art accent; show play counts |
| 🗎 Log | session log controls |
| 🔊 RG | ReplayGain Off/Track/Album; prevent clipping |
| 🔈▾ | when exact output is impossible (Strict/Automatic/HQ); output device list with capabilities; Rescan; native DSD (ASIO) device; PCM capability probes |
| 🔖 | Bookmark current position, bookmarks |
| 💤 | sleep timer Off/15/30/45/60/90 minutes/End of track |
| playlist combo | saved playlists |
| 💾 Save As | inline name field, Save, Cancel |
| cover hover | enlarged album art |

### Control vocabulary in use

Icon-only transport buttons (40×36); text buttons with an emoji prefix; segmented
choices drawn as selectable labels (Mode, View, Channels, Loudness, sort, theme);
checkboxes; sliders paired with a numeric drag field; one combo box style; text
fields; disclosure triangles; label/value grids; progress bar; waveform seek bar;
popups anchored under toolbar buttons; red/amber status text.

## Measured cost (baseline for later work)

i7-9700F, RTX 2060 SUPER, 170 Hz monitor. Process CPU over 15 s, as a percentage
of one core (machine share = ÷8). Frame counts from `MOOSIK_FRAME_LOG`.

| State | CPU (% of a core) | GPU 3D | Frames/s |
|---|---|---|---|
| Stopped or paused, main window only | 0 | 0% | 0 (event-driven) |
| Playing, main window only | 11.6–12.1 | 0.3–0.5% | 59 |
| Playing, main + lyrics | 13.5 | 2.3% | — |
| Stopped or paused, spectrum open | 7.6–7.7 | 1.0–1.2% | 30 + 30 (by design) |
| Playing, spectrum Real-time bars 1024 | 36.8 | 3.3% | 170 + 170 |
| Playing, spectrum Spectro | 39.2 | 5.5% | 170 + 170 |
| Playing, spectrum bars 2048 | 35.9 | 3.4% | 170 + 170 |
| Playing, Whole track | 39.4 | 4.1% | 170 + 170 |

Max FPS defaults to the monitor's refresh rate (`monitor_refresh_hz` at
start-up; about 170 Hz here), so 170 frames/s while the spectrum plays is the
configured rate. Paused and stopped states were first measured wrongly: the
driver's pause and stop never took effect (see finding 16), so those rows were
really playback. They were re-measured with a posted Space key and an idle
stop; the table above holds the corrected values.

Main-window UI at 60 frames/s costs about 1.7–2 ms of CPU per frame; with the
spectrum open, the two windows together cost about 2.2 ms per frame.

These are whole-process numbers. They do not separate UI building,
tessellation, painting and analysis yet; that split is the measurement step
before any partial-redraw work.

## Findings to carry into the design

Behaviour (cost):

1. Idle behaviour is correct: a paused or stopped player does not redraw, and
   an open but static spectrum is capped at 30 frames/s as the code intends.
2. A playing spectrum costs about a third of a core at 170 Hz (both windows
   redraw together). This is the main target for later frame-cost work.
3. Frame pacing sleeps rather than following the display, which matters for
   smooth motion later.

Layout:

4. Below about 1300 px the bottom toolbar overlaps the transport row: Look, Log,
   RG and Bit-Perfect are drawn over the volume slider and the LUFS readout.
5. Long titles run into the artist column; nothing truncates or fades.
6. Five lines of technical status text sit permanently under the transport.
7. The now-playing header has no cover art and leaves most of its height empty.
8. The spectrum controls wrap unpredictably, an explanatory paragraph is
   inlined between control rows, and at the default spectrum size the palette
   combo is cut off.
9. Lyrics and spectrum toolbars wrap into a second row without grouping.
10. Popups anchor to toolbar buttons that are themselves overlapping, so they
    appear in unexpected places.

Visual:

11. Icons are emoji, so their style, weight and colour vary by glyph and
    platform.
12. Selected states are a small blue chip; hierarchy is mostly weight-free text
    in two sizes.
13. In the light theme the spectrum plot stays dark while its chrome turns light.

Typography (CJK):

14. Japanese fonts are first in the fallback chain. All Han characters therefore
    take Japanese forms. When a Simplified character is missing, it falls
    through to a Chinese font *within the same string*: in 风吹过的旧站台, 过 and
    远 visibly change face, weight and baseline. Fixing it needs a language
    choice per text run (track or lyric language), not a different global order.

Accessibility:

15. Before this branch only the main window exposed an accessibility tree. The
    vendored eframe now gives child windows their own AccessKit adapter
    (`moosik_child_accesskit`), which screen readers and the automation both use.
16. While playing, invoking the transport buttons (pause, stop, previous, next)
    through UI Automation delivers the Click to a different AccessKit node. The
    buttons' egui ids are stable (logged every pass), so the misrouting lies
    between the UIA provider and the tree updates. Screen-reader users would
    hit the same thing. The bookmark button in the same row works.

## Not covered yet

- Text selection, drag-reordering, multi-select and the cover hover popup need
  pointer input, which the driver deliberately does not fake.
- The A–B repeat state was not captured. Its button was not found by name
  during the run, although an earlier tree dump showed it.
- Filter text can't be set through UIA because egui's text field ignores it, so
  there is no filtered-playlist capture.
- DSD playback, bit-perfect exclusive output, and the translation workbench
  with a local model installed are not covered.
- Linux and macOS are not covered.

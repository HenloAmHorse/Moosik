# Changelog

## [0.3.0] - 2026-04-24

### Spectrum Analyzer — Peak Hold
- **New feature: Peak Hold** — available in Bars style; a thin marker line sits at the highest level each bar has reached
- **Three decay modes**
  - *Linear* — marker falls at a constant rate after the hold window expires
  - *Gravity* — marker accelerates downward, giving a natural "drop" feel; configurable acceleration
  - *Fade Out* — marker stays at its position and fades out with exponential alpha decay; smoother and more organic than linear alpha would be
- **Configurable hold time** — 10–1000ms slider
- **Configurable fall speed** — logarithmic slider; same value produces equivalent fade duration in Linear and Fade Out modes
- **Configurable peak thickness** — 1–6 physical pixels; rendered as a mesh rectangle that exactly matches the bar's pixel width, so it displays correctly at any bar gap including 1px-wide bars
- **Configurable color** — full color picker including alpha
- **Peak floor clamping** — in Linear and Gravity modes the marker cannot fall below the current bar height; no visual overlap between bar and peak
- **Fade Out correctness fixes** — per-bar peak resets when alpha fully decays so historical highs don't block future peaks from displaying; FadeOut markers on all bars fade to the same alpha at the same time (draw-time sync)

### Spectrum Analyzer — Analysis Cache
- **Correct cache key** — FFT size, window function, min Hz, and max Hz are now part of the cache filename; previously changing these would silently reuse a cache computed with different settings
- **Reanalysis warning wired up** — the ⚠ "no cache for current settings" banner now also triggers when FFT size, window function, or frequency range changes (previously only pad factor, overlap, bar mapping, interpolation, and bar count triggered it)
- **Clear All button** — single click deletes every `.spectrumcache` file in `~/.moosik/cache/`; shown inline next to the cache stats line
- **Cache-exists highlight** — FFT size, window, interpolation, zero-padding, overlap, and bar mapping buttons turn green when a cache already exists for that combination with the current other settings; no filesystem calls per frame (backed by a `HashSet` refreshed every 2 s)

## [0.2.1] - 2026-04-20

### Album Art
- **Thumbnails on by default** — playlist thumbnails are now enabled out of the box
- **Hover preview** — hold the cursor over a playlist thumbnail for 1 second to see a 512px popup of the full art; disappears when the cursor moves away

### Spectrum Analyzer
- **60 fps default** — corrected the default frame rate back to 60 fps
- **Window function descriptions** — hovering Hann / Hamming / Blackman / Flat-top now shows a plain-English explanation of each window's trade-offs

### Internal
- `spectrum.rs` split into `spectrum/eq.rs` and `spectrum/art.rs` — no user-facing changes

## [0.2.0] - 2026-04-18

### Album Art

- **Playlist thumbnails** — each playlist row shows a 28×28 thumbnail of the track's embedded cover art, displayed alongside the track checkbox
- **Spectrum overlay — Transparent mode** — album art is rendered behind the spectrum at a configurable opacity; fit mode is selectable (Contain, Cover, Stretch)
- **Spectrum overlay — Mask mode** — spectrum bars act as a cut-out window into the album art; each bar is textured with the region of the art it covers; brightness can be fixed or follow bar magnitude dynamically
- **Art Settings panel** in the spectrum window — collapses under "Album Art"; supports global settings and per-track overrides (same pattern as EQ presets)
- **Spectrum placeholder** — optional ♪ glyph shown when a track has no embedded art
- **Settings persistence** — art display preferences saved to `~/.moosik/art_settings.json`

## [0.1.0] - 2026-04-04

Initial public release.

<p align="center">
  <img src="screenshots/player.png" alt="Player UI" width="700"/>
</p>

<p align="center">
  <img src="screenshots/spectrum.png" alt="Spectrum analyzer with parametric EQ overlay" width="700"/>
</p>

### Spectrum Analyzer
- Pre-processed + real-time hybrid mode: full-track FFT analysis runs in the background while real-time FFT feeds the display; display switches seamlessly between the two
- Seven visualization styles: Bars, Line, Filled Area, Waterfall, Spectrogram, Octave Bands, Phasescope
- CQT (Constant-Q Transform) bar mapping — equal relative frequency resolution per bar across the full spectrum
- Flat Overlap and Gaussian bar mapping modes also available
- FFT zero-padding up to 16× for sub-bin frequency resolution
- Six window functions: Hann, Hamming, Blackman, Flat Top, and more
- Overlap up to 87.5% for high temporal resolution
- Six sub-bin interpolation modes: None, Linear, Catmull-Rom, PCHIP, Akima, Lanczos
- Auto FFT size scaling: adapts to the track's sample rate to maintain a consistent ~85ms analysis window
- Analysis caching: pre-processed frames cached to disk; switching settings reloads from cache instantly; "needs re-analyze" banner shown when cache is unavailable for current settings
- Cache size and file count display in settings panel
- Flat and ISO 226:2003 equal-loudness weighting (40 phon)
- Configurable frequency range, bar count, smoothing, bar gap

### Parametric EQ
- Up to 16 bands per track
- Band types: Peaking, Low Shelf, High Shelf, High Pass, Low Pass, Notch
- Biquad IIR filters using Audio EQ Cookbook formulas, applied in real time via a rodio Source wrapper
- Draggable nodes directly on the spectrum: click empty space to add a band at that frequency/gain, drag to adjust, right-click to remove
- EQ overlay modes: Curve (response curve drawn over bars), Apply (bar heights reflect EQ gain), Both
- Bake to cache: re-analyze the current track with EQ applied and store the result

### EQ Presets
- Global presets (available for all tracks) and song-specific presets (per file path)
- Auto-load on track change: loads last-used preset for that track; falls back to default global preset; falls back to empty
- Modified indicator: preset name shows `*` when live bands differ from the saved state
- Update and Discard buttons when a preset has been modified
- Pending-switch prompt when switching presets with unsaved changes: Save & switch / Discard & switch / Cancel
- Save As New with name input and scope selector (Global / Song)
- Rename preset inline
- Duplicate preset
- Delete preset with inline confirmation
- Set as Default (★) for global presets — used as the fallback for tracks with no last-used preference
- Presets persisted as JSON at `~/.moosik/eq_presets.json`

### Player
- Audio playback via rodio + symphonia (MP3, FLAC, OGG, WAV, AAC, and more)
- Waveform seek bar with click-to-seek and live position display
- Volume control
- Track metadata: title, artist, album, duration via lofty
- Album cover art display
- CJK font fallback (Japanese, Chinese, Korean tags display correctly on all platforms)
- Momentary LUFS display
- Stereo correlation meter
- Chord detection and timeline overlay

# Changelog

## [1.0.0] - 2026-06-15

First stable release. Adds true bit-perfect output (with WASAPI exclusive mode
on Windows and a device picker), and resolves the long-standing high-CPU draw
of the spectrum window. Everything below lands in 1.0.

### Bit-Perfect Output (new)
- **Native-rate, bit-transparent playback** — a dedicated symphonia decode thread (full sample precision, 24-bit safe) feeds a lock-free ring buffer; the output callback only copies samples and converts to the device's native format with exact power-of-two scaling, so integer sources round-trip unchanged
- **Windows: WASAPI exclusive mode** — shared-mode WASAPI only accepts the mixer's configured format (e.g. a DAC pinned to 384 kHz rejects 44.1 kHz tracks), so the bit-perfect path opens the device in exclusive mode like foobar2000's WASAPI output; device capabilities are probed in exclusive mode too, format candidates (16/24/24-in-32/32-bit int, 32-bit float) are chosen to match the source bit depth, and polling mode is used to avoid the USB-audio stutter of event-driven exclusive streams
- **Linux/macOS** — direct cpal output at the exact rate; selecting an ALSA `hw:` device bypasses the PipeWire/Pulse resampling shims
- **Device detection & selection** — output devices are scanned in the background with their capabilities (supported rates, sample formats, max channels); pick one from the 🔈▾ menu next to the 💎 button, or stay on the system default; selection is persisted to `~/.moosik/bitperfect.json`
- **Fast seeking** — container-level symphonia seek instead of decode-and-discard
- **Sample-accurate position** — elapsed time is derived from frames actually delivered to the device, not wall-clock time
- **Graceful fallback** — if the device rejects a track's format the player drops back to normal mode (or rolls back a device switch) with a clear status message listing what the device supports
- **Volume warning** — the 💎 tooltip warns when volume is below 100%, since rescaling breaks bit-perfectness
- EQ is intentionally bypassed in bit-perfect mode; the EQ panel shows a notice and the spectrum stops simulating EQ gain

### Player
- **Gapless playback** — consecutive tracks now play back-to-back with no silence, on both paths. Normal mode appends the next track onto the same output before the current one ends (rodio plays queued sources seamlessly); bit-perfect chains the next file's decode into the same device stream, and rolls the display over on frame-exact track boundaries (no drift across a long album). Bit-perfect stays gapless as long as the next track shares the current sample rate and channel count — a format change still re-opens the device, as it must. The next track is prebuffered a few seconds ahead; changing what plays next (seek, loop mode, new selection) discards the prebuffer cleanly.
- **OS media integration** — hardware media keys (play/pause/next/previous/stop) and the system now-playing panel now work: MPRIS on Linux, System Media Transport Controls on Windows, the Now Playing center on macOS. Title/artist/album/duration and live playback position/state are published to the OS; transport buttons (and the lock-screen scrubber) drive playback. Uses the pure-Rust D-Bus backend on Linux, so no `libdbus` system library is needed to build.
- **ReplayGain loudness normalization** — new 🔊 RG menu with Off / Track / Album modes and a clip-prevention toggle. Reads ReplayGain tags when present; for untagged files it falls back to Moosik's own measured integrated LUFS (normalising to the −18 LUFS reference), so it works even on libraries that were never scanned. Applied live (updates as the loudness scan completes mid-track) and **bypassed in bit-perfect mode**, since a gain change would break bit-perfectness — the menu says so. Setting persists in `~/.moosik/replaygain.json`.

### Performance — Spectrum window CPU
- **Frame limiter** — with the spectrum window open, the app was repainting at ~900 fps (an immediate child viewport requests a repaint every frame, which overrode `request_repaint_after`), burning ~30% CPU regardless of the Max FPS setting. The render loop is now hard-capped to the target frame rate by parking the UI thread to the frame deadline (the thread sleeps, it does not spin), so CPU scales with Max FPS as expected
- **Windows timer resolution** — raised to 1 ms at startup (`timeBeginPeriod`) so the limiter's sleep is accurate on high-refresh displays; without it Windows' ~15.6 ms default would clamp the cap to ~64 fps
- **Max FPS up to 240** — raised from 120 for high-refresh monitors
- **Honest F3 instrumentation** — the overlay now shows the real repaint rate and per-frame draw cost (the old "FPS" line tracked only the FFT throttle and couldn't reveal the true repaint rate)

### Spectrum Analyzer
- **FFT auto-size** targets a ≥100 ms analysis window (8192 @ 44.1/48 kHz, 16384 @ 96 kHz, 32768 @ 192 kHz); 32768 is selectable manually

### Fixed
- **UI freeze on FLAC seeks** — the rodio seek fallback (used for FLAC, which symphonia can't seek in-place through rodio) decoded-and-discarded every sample up to the target *on the UI thread*. Deep into a hi-res file that meant tens of millions of samples and a multi-second "not responding" freeze — most visibly when toggling bit-perfect **off** mid-track (which restarts the track and seeks back to position). The decode-and-discard now runs on a background thread and the seeked stream is installed when ready, so the UI stays responsive; pause/stop/track-change during the seek are handled.
- **Toggling bit-perfect off mid-track no longer hitches** — the toggle-off restart used to play the track from 0 and *then* seek back to the current position, which briefly emitted audio from the start and did decode work on the UI thread. It now opens and positions the stream entirely on a background thread (no play-from-0 blip, no UI-thread decode).
- **Silent rodio output after using bit-perfect** — on Windows, opening the device in WASAPI exclusive mode suspends the long-lived shared-mode (rodio) output stream, and cpal never recovered it, so normal-mode playback was mute for the rest of the session once bit-perfect had run. The output stream is now recreated before the next normal-mode track whenever a bit-perfect stream has held the device.
- **Crash log** — panics on any thread are now recorded to `~/.moosik/crash.log` (the release build aborts on panic, so a background-thread panic would otherwise vanish with no trace).

### Removed
- **Chord detection overlay** — the triad-template matcher was unreliable on anything beyond simple major/minor material (no 7th/sus/extended chords, weak major-vs-minor discrimination, no harmonic suppression), so it's been dropped. Key detection (Krumhansl-Schmuckler), which shares the chromagram front-end, stays.

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
  <img src="screenshots/player.png" alt="UI" width="700"/>
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

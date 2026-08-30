# Moosik

<p align="center">
  <img src="assets/icon.png" width="120" alt="Moosik icon"/>
</p>

A desktop music player with bit-perfect output (PCM and DSD), a professional-grade spectrum analyzer, and a parametric EQ, built in Rust.

### The icon

The waveform is a [sinc function](https://en.wikipedia.org/wiki/Sinc_function) — the mathematical foundation of sampling theory and the ideal low-pass filter. The axes are scaled at **8.539:1** (≈ πe), cropped at x = ±8.539 and centered vertically at y = 0.444 ± 1 (1.444 ≈ e^(1/e), the maximum of x^(1/x)).

The background is **Eigengrau** (#16161d) — the color the human brain perceives in total darkness.

The waveform is **Kugelblitz** (#94b1ff) — the theoretical RGB of an infinite-temperature blackbody radiator, which emits all frequencies equally. A perfectly flat spectrum. The EQ ideal.

<p align="center">
  <img src="screenshots/player.png" alt="UI" width="700"/>
</p>

<p align="center">
  <img src="screenshots/spectrum.png" alt="Spectrum analyzer with parametric EQ overlay" width="700"/>
</p>

## Features

### Bit-Perfect Output

> **What "bit-perfect" claims here.** The sample and payload words Moosik hands
> the Windows audio driver are the source's own words, unmodified. That is the
> whole claim. The USB link, the driver's internals and the DAC are past the
> boundary of anything this process can observe — a DAC's rate display, its DSD
> lock light and its temperature are not evidence, and verifying the physical
> end of the chain needs capture equipment on the link, not a front panel.
> Nothing in Moosik has been validated that way.

Moosik distinguishes four things, and uses these words everywhere — badge,
tooltip, panel and log:

| State | Meaning | Badge |
|---|---|---|
| **Payload exact** | The representation handed to the driver is the source's own. Every valid integer bit preserved; Float32 words bit-identical; DSD payload bits the file's own bits. | 💎 green |
| **Value exact** | The representation changed, but every decoded sample landed exactly on the destination lattice. Signed zero and NaN payloads do not survive, so this is *not* bit-perfect. | ◇ hollow |
| **Processed** | At least one known alteration: rounding, clipping, gain, EQ, ReplayGain, resampling or DSD decimation. | ⚠ amber |
| **Unverified** | Nothing Moosik does alters the audio, but the platform path downstream cannot be proved transparent. | ⚠ amber |

Only **payload exact** ever renders the green diamond. Protocol framing — DoP
markers, and the protocol-defined silence inserted for pause, device prefill
and a bounded drain — is carrier grammar, not a change to the source frames.

- **Native-rate, exact playback** — toggle the 💎 button and audio is sent to the device at the file's exact sample rate, decoded at full precision, and packed into the device's negotiated format with shifts and byte copies. No float multiply, no rescaling, no saturating cast: integer sources reach the driver with every bit intact, and 32-bit float sources reach it as their own IEEE-754 bit pattern
- **Exact formats only — no silent narrowing** — the device format is chosen from a matrix that admits nothing lossy, and there is no last-chance candidate. Integer sources of 1–16 bits may use `16i`, packed `24i`, 24-in-32 or `32i`; 17–24 bits may use the three wider ones; 25–32 bits may use `32i` alone; a 32-bit float may use `32f` alone. A 24-bit file will never negotiate a 16-bit format, and a 32-bit integer will never pass through a float. If no format can carry the source, the open **fails with the reason** instead of quietly converting
- **Unprovable sources are labelled, not badged** — a 64-bit float source has no exact representation in any available device format. Under **Strict** it stops and says so; under **Automatic** or **HQ** it plays through the same Q1.31 conversion the Float32 route uses, marked **Processed** and never value-exact. The narrowing to 32-bit float happens in the decoder, before that conversion sees a sample, so the value-exactness scan is looking at numbers that have already been altered and its verdict cannot mean what it means elsewhere — the label says both steps, and the value-exact claim is refused to such a source whatever the scan finds
- **Float32 files get an honest conversion, not a redefinition** — a DAC advertising "32-bit" almost always means 32-bit *integer*, and a Float32 file has no exact route to one. Under **Strict** Moosik stops and says so. Under **Automatic** it tries the file's own representation first — raw `32f` exclusive — and falls back to a single named conversion to Q1.31. `I32` is never added to the raw Float32 exact set: the conversion is named, not smuggled
  - The conversion **plays everything**. A sample that lands on the Q1.31 lattice converts exactly; one that does not is rounded, deterministically. It is labelled **Processed** from the moment it opens
  - *Value-exact* is a property of the **material**, not a route you can ask for — it is common in 16- and 24-bit masters exported as floats. Moosik claims it only when a **complete decode of the whole track** says every sample was exactly representable *and* the running conversion has not had to round one. Either alone is not enough: a cached verdict can be stale, and the conversion has only seen what has played
  - The verdict cache is durable, and it is advisory. Every packet is re-checked as it is converted, so a stale entry can cost you a label and can never put a rounded sample on the wire unannounced
  - Value-exact is **not** bit-perfect and never shows the diamond. Signed zero and NaN payloads do not survive the change of representation
- **Choose what happens when exact is impossible** — the 🔈▾ menu offers **Strict** (never transform; stop with the reason, and never fall back to the shared mixer, which has a volume stage, an EQ and a resampler in it), **Automatic** (the file's own representation first, then the conversion) and **HQ** (go straight to the conversion, because that is what asking for it means). The choice applies per track and **never changes your bit-perfect preference** — one unsupported file no longer disables exact output for everything after it
- **Volume is locked where there is no volume** — an exact route has no software gain stage between the decoder and the device, so the slider is disabled and reads `100% · locked on this route`, with a tooltip naming the route. The keyboard, the slider and every other input share one predicate; your saved volume is untouched and returns with normal playback. EQ and ReplayGain are outside the output path entirely
- **A track that fails is not a track that finished** — a read error, a decode error, a dead backend or a driver reset stops playback once, with the reason, and nothing reopens it. Only a track that reached the end of its source advances the playlist or restarts under Repeat One. And a track has not reached its end until the device has played what it was already given, so the next one does not begin over the tail of the last
- **DoP gapless has one exception, stated** — a DSD file whose frame count is odd ends on a half-full carrier frame, which is completed with DSD silence. Carrying that across a boundary would pair bytes from two recordings inside one carrier frame, so Moosik does not: that boundary re-opens the device instead. The cost is one gap on odd-length DSD files
- **An integrity fault removes the claim** — a dropout, a torn frame, a decode or read error, a backend write failure, a driver reset or overload, or a late render callback. Any of them means the device received something that was not in the file, so the badge drops to ⚠ with the reason for the rest of the session; only a genuinely new session can claim again. A backend whose thread has died is never reused
- **Stopping actually stops** — stop releases every exclusive and native handle and clears the claim, so a stopped player neither shows a stale diamond nor holds your DAC open against other applications. Switching from an exact route to ordinary playback releases the device first
- **Windows: WASAPI exclusive mode** — opens the device exclusively, like foobar2000's WASAPI output, bypassing the Windows mixer; a DAC pinned to e.g. 384 kHz in the control panel still plays 44.1/48/96/192 kHz tracks at their native rate. The format the driver replies with is read back and checked field by field — rate, channels, channel mask, block alignment, container bits, valid bits and subtype — so a driver that accepts a request and then describes something else is refused rather than believed. A driver that replies in the legacy `WAVEFORMATEX` form carries no channel mask and no valid-bit count; that silence is not read as agreement, so those routes are confined to stereo with valid bits filling the container, where nothing the structure omits could be hiding anything
- **Linux / macOS** — cpal asks for the exact format, and the same exact-only policy applies (no narrowing, no cross-family conversion). But what happens after the request belongs to ALSA, PipeWire, Pulse, `dmix` or CoreAudio, and this process cannot see which route it got. A direct `hw:` device is the way to avoid the mixing shims — and Moosik still reports **Exactness unverified** there rather than a green diamond, because requesting a format is not the same as measuring one
- **Device picker** — the 🔈▾ menu lists every output device with its real capabilities (supported rates, sample formats, channels), probed in the background; pick one or stay on the system default. Your choice is remembered. The probe runs the *same* two steps playback runs — ask, then check the reply field by field — so the list does not advertise a rate that the open would then refuse
- **Honest fallbacks, scoped to one track** — if a device can't play a track's format, playback drops to shared mode with the reason, and the badge changes to ⚠ rather than staying green. Two things have to be true: the endpoint must have *answered the format query and said no*, and your policy must permit processing. A bad `MOOSIK_BP_FORMAT` value, an unreadable file, a failed seek, a device that is unplugged or held by another application, and any failure inside Moosik's own backend all stop with their own message instead of quietly taking a lossier route. EQ is bypassed on the exact routes (and the EQ panel says so); on a processed route the persistent output panel shows the live volume, EQ and ReplayGain state, so what is actually in the path is always readable
- **The panel says what is playing, and stays put** — source, any transform, any carrier, transport and endpoint, negotiated output format, and the live processing chain. DSD is reported as DSD with its DoP carrier shown separately, rather than as the 24-bit PCM the carrier happens to be. Ordinary status messages cannot overwrite it

### DSD Playback
- **DSF & DSDIFF** — `.dsf`/`.dff` files are first-class: header-parsed stream
  properties, ID3v2 tags (title/artist/album, ReplayGain, embedded cover art)
  through the same pipeline as every other format
- **Bit-perfect DoP** — DSD64/128/256 play as DSD-over-PCM (DoP 1.1): the raw
  1-bit stream packed untouched into 24-bit words at the carrier rate
  (176.4/352.8/705.6 kHz), integer-only device formats, volume/EQ/ReplayGain
  hard-bypassed. The status line shows the exact mode, e.g.
  *💎 DSD128 via DoP · 352.8 kHz · 2ch · 24i excl*
- **Gapless DSD** — same-rate DSD tracks hand off with no device re-open and
  an unbroken DoP marker sequence: the marker phase is owned by the render
  thread and counts straight through a file boundary, a seek and a pause
- **Every DoP frame is a valid DoP frame** — the device prefill, a pause, an
  underrun, the gap after a seek and the final drain all emit DSD silence
  carrying a correct marker, never a frame of PCM zeros, which a DoP DAC reads
  as loss of lock. A pause consumes no source audio at all
- **Automatic PCM fallback, for device limitations only** — if the failure is
  classified as a device limitation, such as a DAC that can't take the DoP
  carrier rate, the bitstream is decimated to high-rate PCM and played through
  the normal path instead (volume/EQ/ReplayGain apply), with a clear status
  notice. Every other failure stops with a visible error rather than quietly
  processing the audio: an invalid `MOOSIK_BP_FORMAT` override, a request that
  contradicts itself, and any file, parse or seek failure all report the real
  problem instead of decimating around it
- **Analyzer support** — spectrum, waveform, LUFS/DR and spectral ceiling run
  on a decimated PCM feed (176.4 kHz default, 352.8 kHz selectable for a
  faster-reacting spectrum); playback stays raw bits over DoP
- **Native DSD** (experimental, on by default — build with
  `--no-default-features` to exclude it) — pick a native output in the 🔈
  menu and DSD plays as raw native DSD with no DoP carrier ceiling,
  unlocking **bit-perfect DSD512** — and likely **DSD1024** too: the rate
  is negotiated with the driver directly rather than picked from a fixed
  list, so nothing in the code stops at 512 (not hardware-verified at 1024
  yet). A native output that can't be opened falls back to DoP, and only a
  DoP or output failure classified as a device limitation continues on to
  decimated PCM.
  - **Windows (x86_64): ASIO** — pick your DAC's ASIO driver under
    "Native DSD (ASIO)". No Steinberg SDK required. Hardware-verified on
    an SMSL C200Pro.
  - **Linux: ALSA** — pick your DAC's direct `hw:` device under
    "Native DSD (ALSA)"; playback uses the kernel's native DSD formats
    (`DSD_U32_BE`/`U16`/`U8`), the same route MPD uses. The card's driver
    must advertise DSD formats, and the device must be free — PipeWire /
    PulseAudio hold `hw:` devices, so reserve or release the card first.
    Not yet hardware-verified (the transport is exercised end-to-end in
    tests; a real DAC report would be very welcome).
  - Driver behavior varies by vendor on both platforms, so a new DAC may
    surface new quirks the first time it's tried — if yours does, the
    console log (run from a terminal) shows each negotiation step
  - macOS has no native-DSD transport (CoreAudio is DoP-only), so DSD
    plays via DoP there

### Spectrum Analyzer
- **Pre-processed + real-time hybrid** — full-track analysis runs in the background while real-time FFT feeds the display during playback; seamlessly switches between the two
- **Multiple visualization styles** — Bars, Line, Filled Area, Waterfall, Spectrogram, Octave Bands, Phasescope
- **Selectable palette** — seven colour ramps (Classic, Kugelblitz, Ice, Magma, Aurora, Mono, and an Album-accent ramp that follows the cover art), picked from the control row with a gradient swatch preview and applied across bars, octave bands, line, filled, spectrogram, and waterfall
- **Settings persist** — the view settings (style, bars, window, interpolation, padding, overlap, bar mapping, frequency range, …) restore across launches (`~/.moosik/spectrum.json`)
- **Peak Hold** — configurable marker that tracks the highest level per bar; three decay modes (Linear, Gravity, Fade Out), hold time, fall speed, thickness, and color all adjustable
- **CQT bar mapping** — Constant-Q Transform mapping gives each bar the same relative frequency resolution regardless of pitch, just like professional analyzers
- **Adaptive Superlet Transform** — an alternative to the FFT that analyses the waveform directly with sets of Morlet wavelets, taking the geometric mean of their responses. Where an FFT uses one window width for the whole spectrum, this varies it per frequency, which is what a log-spaced bar grid actually wants. Pre-process only; the live view keeps using the FFT. Five quality presets plus a Custom mode exposing the window budget, sharpness, wavelet count and spread, with a live readout comparing each against the FFT
  - *Bass*: resolves detail no FFT setting reaches — at 60 Hz it separates tones 1.5 Hz apart that a 171 ms FFT window renders as a single blob. The cost is honest and unavoidable: constant-Q at the bottom of hearing needs windows measured in seconds, so bass responds more slowly. The window budget is a slider, not a hidden curve
  - *Treble*: strictly better, for free — 24 ms of temporal smearing at 10 kHz against the FFT's 171 ms, with no loss of frequency detail the display can show
  - *Around 500 Hz – 2 kHz*: roughly a wash. An FFT is genuinely good there, and nothing pretends otherwise
  - Analysis runs 1–17 minutes per five-minute track depending on preset, with a live ETA, a progress bar and an Abort button. Long wavelets are convolved in the frequency domain, which is 4–5× faster and numerically identical (measured worst-case difference 2×10⁻⁷)
- **Configurable FFT** — up to 16× zero-padding, six window functions (Hann, Hamming, Blackman, Flat Top), up to 87.5% overlap
- **Six interpolation modes** — None, Linear, Catmull-Rom, PCHIP, Akima, Lanczos
- **Auto FFT size scaling** — adapts to the track's sample rate to maintain a consistent analysis window
- **Refresh-rate-aware frame cap** — Max FPS defaults to the monitor's refresh rate (up to 240) so the animation is as smooth as the display allows, adjustable in the settings
- **Analysis caching** — pre-processed frames cached to disk; settings buttons highlight green when a cache exists for that combination; "Clear All" button; reanalysis warning fires on any cache-key change. A size budget (4 GB by default) evicts least-recently-used caches after each analysis, since a superlet track at 180 fps costs 45–80 MB and cannot be packed much smaller — the low byte of every stored value is quantisation noise, so no lossless coder beats about 12 %
- **GPU acceleration** — the superlet pre-process offloads its large convolutions to the GPU through wgpu (Vulkan or Metal, so AMD, Intel, NVIDIA and Apple are all covered). Measured against the same build with the device off: High 1.23×, Ultra 1.71×, Extreme 1.32×. Fast and Standard are unchanged by design — their kernels never reach a size where a device beats the cores
  - *Calibrated per machine, above a fixed floor.* Blocks below 2^17 samples never reach the device at all — that eligibility floor is a conservative constant compiled in, not a measurement. Above it the crossover is a property of one device against one core count, so it is settled per machine rather than assumed: a probe on first run picks a starting point, then real analyses A/B the two routes on the eligible sizes and adapt on evidence. A device that turns out slower gets switched off by itself. Settings expose Auto / Always / Off, a Re-measure button, and the per-size table
  - *Analysis core budget* — a slider, defaulting to two fewer threads than the machine has, so the output thread keeps its deadline while a track is being analysed
- **Bar spacing** — where the bars land, independent of how many there are
  - *Bass width*, a slider from wider-than-log through **log** (the classic analyser axis) to **ERB** (constant bars per auditory filter, Glasberg & Moore 1990). Not a resolution control: a tone's blur in bars and a bassline's travel in bars both scale with bar density, so the ratio is constant across the slider — what changes is how much screen the bass gets to move across. Below a capped window it costs nothing
  - *Zoom*, a lens anywhere on the axis with adjustable centre, width and strength. Bars are taken from the rest of the spectrum, never invented. Unlike bass width this is not free above ~1 kHz, so the panel quotes the cost multiplier
  - Every setting is the normalised integral of a strictly positive density, so the axis cannot fold back on itself or push a bar off the display at any setting
- **Loudness** — flat or ISO 226:2003 equal-loudness weighting, applied at display time so toggling it is instant

### Parametric EQ
- **Up to 16 bands** — Peaking, Low Shelf, High Shelf, High Pass, Low Pass, Notch
- **Biquad IIR filters** (Audio EQ Cookbook) — applied in real time via a `rodio` Source wrapper
- **Draggable nodes on the spectrum** — click to add a band, drag horizontally for frequency, drag vertically for gain, right-click to remove
- **EQ overlay modes** — Curve (response curve drawn over spectrum), Apply (bar heights reflect EQ gain), Both

### EQ Presets
- **Global and song-specific presets** — two separate dropdowns, one active at a time
- **Auto-load on track change** — loads the last-used preset for the track, falls back to the default global preset, then empty
- **Modified indicator** — preset name shows `*` when bands have been changed; Update and Discard buttons appear
- **Pending-switch prompt** — switching presets while modified asks Save & switch / Discard & switch / Cancel
- **Full preset management** — Save As New, Rename, Duplicate, Delete (with confirmation), Set as Default (★)
- **Persistent** — stored as JSON in `~/.moosik/eq_presets.json`

### Album Art
- **Playlist thumbnails** — 28×28 cover art thumbnail in every playlist row; hover for 1 second to see a 512px preview
- **Transparent overlay** — art rendered behind the spectrum at adjustable opacity; Contain / Cover / Stretch fit modes
- **Mask mode** — spectrum bars act as a cut-out window into the art, each bar textured with the art region it covers; brightness can track bar magnitude dynamically or be fixed
- **Art Settings panel** — collapsible section in the spectrum window; global settings with optional per-track overrides
- **Spectrum placeholder** — configurable ♪ glyph when a track has no embedded art
- **Persistent** — settings stored in `~/.moosik/art_settings.json`

### Lyrics
- **🎤 Lyrics window** — its own window, so it can live on a second screen. The current line highlights and centres itself, sung lines dim, and a slider sets the text size
- **Reads `.lrc` beside the track, or the file's own tags** (`USLT` / Vorbis `LYRICS` / iTunes atom)
- **Online lookup via [LRCLIB](https://lrclib.net), then NetEase Cloud Music** — no account, no API key, and both serve *synced* lyrics, not just plain text. Two sources because one wasn't enough: on a real set of thirteen failing tracks LRCLIB returned zero hits for nine, since a Japanese/Vocaloid library is largely absent from it. NetEase had synced lyrics for nearly all of them
- **A source that can't answer never reports "no lyrics"** — a rate limit or an outage says nothing about whether the lyrics exist, so it's reported as such rather than sending you off to transcribe by hand
- **Matching built for messy tags** — the lookup is a ladder (exact → without album → structured search on normalised fields → normalised title), filtering and ranking on duration throughout. Normalisation folds full-width Latin, strips bracketed decoration (`【初音ミク】…【オリジナルMV】`), cuts at `feat.`, and keeps only the first artist credit. This is what finds a track whose artist field holds the producer where the database has the vocalist
- **Manual search** — type anything, see the hits with synced-vs-plain marked, pick one
- **Tap-along sync editor** — play the song and press Space as each line starts; a plain sheet becomes a timed one in a few minutes. Re-times existing sheets too, and resumes at the first untimed line
- **Sidecar-only writes** — lyrics are saved as `.lrc`; the audio file is never modified, which also makes this safe for DSD

### Tag Editing
- **🏷 Tags window** — edit title, artist, album, album artist, track/disc numbers, year, genre, composer and comment
- **Full tag viewer** — every key the file carries, including ones the editor doesn't touch (which are preserved on save)
- **Safe writes** — the change goes to a copy which then replaces the original by rename, so an interrupted write can't damage the file; the result is read back and verified before it's accepted
- **DSD is read-only** — DSF keeps its ID3 blob behind a header pointer, and a bad write there costs the audio, not the metadata

### Theme & Appearance
- **Cohesive theme, light or dark** — built from the app's own identity colors (the **Eigengrau** #16161d base and **Kugelblitz** #94b1ff accent): a tinted surface ramp, rounded corners, and accent-coloured selection and hover, applied across both the egui widgets and the custom-painted surfaces (seek bar, playlist rows, status text). A light variant is a click away
- **Per-track accent** — the now-playing bar, current-row marker, and seek fill take on an accent sampled from the current track's cover art (blended toward the brand blue so it never clashes); art-less or grayscale covers fall back to the brand accent
- **🎨 Look menu** — optional, persisted appearance controls (`~/.moosik/appearance.json`), all defaulting to the shipped look:
  - **Theme** — dark (default) or light
  - **Font** — pick any font installed on the system. Families are read from each file's own name table, so the list reads *Fira Code Medium* rather than `FiraCode-Medium`, and every face in a `.ttc` collection is offered; a filter box keeps a few hundred families navigable. CJK fallbacks stay installed behind whatever you pick, so a Latin-only font won't tofu your Japanese tags
  - **Text size** — a 70–160% UI-scale slider (applied on click), scaling all text and chrome
  - **Accent source** — album-art accent or the fixed brand accent

### Player
- **Gapless playback** — consecutive tracks play with no silence between them, in both normal and bit-perfect mode. An exact stream stays gapless only while the next track still fits it exactly — same endpoint, rate, channel count, channel layout and sample family, and a container wide enough for its precision. Anything else re-opens the device at the boundary rather than narrowing the next track to fit
- **Playlist search** — a 🔍 box filters the playlist by title, artist, or album as you type (`Ctrl+F` to focus, `Esc` to clear)
- **Column sorting** — sort by Title / Artist / Album / Time, plus **Plays** and **Recent** from play history; click a column to sort, click again to reverse
- **Sleep timer** — stop after 15–90 minutes or at the end of the current track
- **A-B repeat** — loop a section of the current track, with markers on the seek bar
- **Bookmarks** — save and jump back to positions per track
- **Play statistics** — per-track play count and last-played, shown in the Info window and (optionally) as a column in the playlist
- Waveform seek bar with click-to-seek
- Volume control (volume and loop mode persist across launches)
- Metadata display (title, artist, album, cover art) via `lofty`
- CJK font fallback (Japanese, Chinese, Korean tags display correctly)
- **OS media integration** — hardware media keys and the system now-playing panel (MPRIS on Linux, SMTC on Windows, Now Playing on macOS): play/pause/next/previous/stop, live title/artist/album and playback position
- **ReplayGain** — loudness normalization (Track / Album), reading ReplayGain tags when present and falling back to Moosik's own measured LUFS for untagged files; clip-prevention toggle; bypassed in bit-perfect mode
- Momentary LUFS display
- Stereo correlation meter
- **Session log** — every run normally creates a timestamped file in `~/.moosik/logs/` (creation can fail; the 🗎 Log button then says why), recording the machine, GPU and analysis timings and a full backtrace on any panic. **WASAPI exclusive** and **native ASIO / ALSA DSD** additionally log the device and the format, rate and buffer they negotiated; normal shared-mode playback and non-Windows CPAL PCM/DoP do not yet report theirs. The **🗎 Log** button opens the folder, and older sessions are pruned back to roughly ten
  - *Before sharing one:* a log can contain your user name and file paths, track file names, audio device / GPU / driver details, and panic backtraces. Read it first — there is no redaction step yet
  - Retention is per-process: running two copies of Moosik at once can prune a log the other is still writing

## Building

Requires Rust (stable, edition 2024).

On Linux, the audio backend (cpal/ALSA) needs the ALSA development headers:

```sh
sudo apt install libasound2-dev   # Debian/Ubuntu
```

Then:

```sh
git clone https://github.com/HenloAmHorse/Moosik
cd Moosik
cargo build --release
./target/release/moosik
```

### Dependencies

All pulled automatically via Cargo:

| Crate | Purpose |
|---|---|
| `eframe` / `egui` | Immediate-mode GUI |
| `rodio` | Audio playback (normal mode) |
| `cpal` | Direct device output (bit-perfect mode, Linux/macOS) |
| `wasapi` | WASAPI exclusive-mode output (bit-perfect mode, Windows) |
| `symphonia` | Audio decoding (MP3, FLAC, OGG, WAV, AAC, …) |
| `rtrb` | Lock-free ring buffer (decode → output) |
| `rustfft` | FFT engine |
| `wgpu` / `pollster` / `bytemuck` | GPU compute for the superlet pre-process |
| `rayon` | Parallel analysis |
| `lofty` | Tag / metadata reading |
| `souvlaki` | OS media-key / now-playing integration (MPRIS / SMTC / macOS) |
| `image` | Album art decoding |
| `serde` / `serde_json` | Preset / settings persistence |

## Platform Support

"Status" here means what has actually been run, which is not the same as what
is believed to work.

| Platform | Status | Output backend | Exactness claimed |
|---|---|---|---|
| Windows | Builds, tests pass, hardware-verified for native ASIO DSD | WASAPI exclusive mode; native ASIO DSD | Yes — as far as the audio driver |
| Linux | Builds and tests pass; no DAC exercised | cpal (a direct ALSA `hw:` device avoids the mixing shims); native ALSA DSD | No — reported as *unverified*, because the platform route is not observable from here |
| macOS | Neither built nor tested | cpal — exact format requested, CoreAudio device behaviour decides the rest | No — same reason |

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

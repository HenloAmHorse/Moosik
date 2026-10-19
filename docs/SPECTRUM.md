# Spectrum analyser and EQ

The analyser in detail — visualisation styles and channel views, the adaptive
superlet transform and where it beats an FFT, GPU offload and how it is
calibrated, bar spacing, caching — and the parametric EQ that draws on top of
it. Split out of the README; nothing here has been shortened.

See also [BIT_PERFECT.md](BIT_PERFECT.md) and [FEATURES.md](FEATURES.md).

## Spectrum Analyzer

- **Pre-processed + real-time hybrid** — full-track analysis runs in the background while real-time FFT feeds the display during playback; seamlessly switches between the two
- **Multiple visualization styles** — Bars, Line, Filled Area, Waterfall, Spectrogram, Octave Bands, Phasescope
- **Channel views** — Mix (the default), Left, Right, Split (stacked on one shared scale), Overlay (cyan left over magenta right), and **Diff**, which plots right-minus-left per band on a ±20 dB axis about a centre line — the thing to look at for balance, panning or a crossfeed network, and the one series the other views cannot show. Diff's arrangement is a preference and persists: horizontal or vertical, swap which channel sits on which side, flip the frequency axis. Available live, and in Pre-process for tracks analysed in stereo. Where they are not available the toolbar says which reason — a mono track, more channels than two, no live PCM, a visualisation with no per-channel form, or a cache with no channels beside it — rather than leaving a greyed-out button to be discovered
- **Waterfall history** — the retained span in **seconds** (0.2–8, default 0.67), not a fixed row count, which would mean a different amount of time on every machine. In Real-time the figure is nominal: it is sized from Max FPS, which is a ceiling requested of the analyser rather than a measured rate, so a machine that cannot keep up retains *more* time than the slider says
- **Selectable palette** — seven colour ramps (Classic, Kugelblitz, Ice, Magma, Aurora, Mono, and an Album-accent ramp that follows the cover art), picked from the control row with a gradient swatch preview and applied across bars, octave bands, line, filled, spectrogram, and waterfall
- **Settings persist** — the view settings (style, bars, window, interpolation, padding, overlap, bar mapping, frequency range, …) restore across launches (`~/.moosik/spectrum.json`)
- **Peak Hold** — configurable marker that tracks the highest level per bar; three decay modes (Linear, Gravity, Fade Out), hold time, fall speed, thickness, and color all adjustable
- **FFT constant-Q weighting** — gathers a single FFT's magnitudes with frequency-dependent weights. It is not a true constant-Q transform and does not create detail absent from that FFT. A separate true CQT is planned for 1.5.5; old FFT caches will not be relabelled as CQT results
- **Adaptive Superlet Transform** — an alternative to the FFT that analyses the waveform directly with sets of Morlet wavelets, taking the geometric mean of their responses. Where an FFT uses one window width for the whole spectrum, this varies it per frequency, which is what a log-spaced bar grid actually wants. Pre-process only; the live view keeps using the FFT. Five quality presets plus a Custom mode exposing the window budget, sharpness, wavelet count and spread, with a live readout comparing each against the FFT
  - Frequency and time response depend on the actual kernel, frequency grid, wavelet count and window budget. Longer bass windows trade temporal localization for frequency selectivity; a denser display is not proof of greater resolving power
  - Pre-processing time is workload- and hardware-dependent. The UI provides an ETA, progress and Abort. Frequency-domain convolution and shared/pruned work reduce cost where applicable; see the measured release comparison rather than treating an old timing range or one fixture as a universal speed/accuracy claim
- **Configurable FFT** — up to 16× zero-padding, six window functions (Hann, Hamming, Blackman, Flat Top), up to 87.5% overlap
- **Six interpolation modes** — None, Linear, Catmull-Rom, PCHIP, Akima, Lanczos
- **Auto FFT size scaling** — adapts to the track's sample rate to maintain a consistent analysis window
- **Refresh-rate-aware frame cap** — Max FPS defaults to the monitor's refresh rate (up to 240) so the animation is as smooth as the display allows, adjustable in the settings
- **Analysis caching** — pre-processed frames cached to disk; settings buttons highlight green when a cache exists for that combination; "Clear All" button; reanalysis warning fires on any cache-key change. A size budget (4 GB by default) evicts least-recently-used caches after each analysis, a track's mono file and its channel sidecar together
  - *Format.* New caches store each level as one of 4096 steps on the fixed 80 dB axis — a quantiser step of 80/4095 = 0.0195 dB, worst error half that — packed and then compressed. Over 11 real tracks at the owner's settings that came to 429.9 MB against 608.1 MB for the previous 16-bit format, about 29 %; **that figure is from the study encoder used to evaluate the scheme and has not been re-measured end to end with the production encoder.** Synthetic fixtures of the production encoder range from −3 % to −32 % with the noise content of the material
  - *Existing caches are left alone.* The older formats still load and keep the 16-bit precision they were written with. Nothing is converted or deleted, and only a fresh analysis writes the new format
  - *Exact identity.* A new cache is named by a hash of every setting that changes the analysis, at full precision, and carries those settings — with the sample rate it was decoded at and the frame hop — in its header. It is reused only when that record is exactly the current settings for the track; a cache with a missing or different record is analysed over. Two settings that differ in the third decimal place no longer share a file
  - *Legacy analyses.* Caches written before exact identity name their settings only rounded (hundredths, tenths, whole hertz), and nothing records what they were really analysed at. They are never loaded automatically, not even one whose rounded name is today's settings: the track is analysed at your settings straight away, and the notice under the controls offers the legacy analysis, labelled **"Legacy analysis — original settings not verified."**, with the settings from its file name described as nominal. "Use this legacy analysis" stops the running analysis and shows it for this track only; repeating the track keeps it, changing a setting or the track ends it, and "Return to my settings" goes back. Your saved settings never change. Legacy analyses are offered only in Superlet mode and only while your own analysis of the track is not on screen. They are not deleted on purpose, but the cache size limit can remove them like any other cache; the one on screen is not removed while shown
  - *A sidecar is two more planes.* A track analysed in stereo has three planes on disk rather than one. Adding channels to a track that already has a valid mix cache keeps that cache rather than rewriting it — a rewrite would coarsen a 16-bit cache to 12 for no reason the user asked for
- **GPU acceleration** — the superlet pre-process offloads its large convolutions to the GPU through wgpu (Vulkan or Metal, so AMD, Intel, NVIDIA and Apple are all covered). Measured against the same build with the device off: High 1.23×, Ultra 1.71×, Extreme 1.32×. Fast and Standard are unchanged by design — their kernels never reach a size where a device beats the cores
  - *Calibrated per machine, above a fixed floor.* Blocks below 2^17 samples never reach the device at all — that eligibility floor is a conservative constant compiled in, not a measurement. Above it the crossover is a property of one device against one core count, so it is settled per machine rather than assumed: a probe on first run picks a starting point, then real analyses A/B the two routes on the eligible sizes and adapt on evidence. A device that turns out slower gets switched off by itself. Settings expose Auto / Always / Off, a Re-measure button, and the per-size table
  - *Analysis core budget* — a slider, defaulting to two fewer threads than the machine has, so the output thread keeps its deadline while a track is being analysed
- **Bar spacing** — where the bars land, independent of how many there are
  - *Bass width*, a slider from wider-than-log through **log** (the classic analyser axis) to **ERB** (constant bars per auditory filter, Glasberg & Moore 1990). Not a resolution control: a tone's blur in bars and a bassline's travel in bars both scale with bar density, so the ratio is constant across the slider — what changes is how much screen the bass gets to move across. Below a capped window it costs nothing
  - *Zoom*, a lens anywhere on the axis with adjustable centre, width and strength. Bars are taken from the rest of the spectrum, never invented. Unlike bass width this is not free above ~1 kHz, so the panel quotes the cost multiplier
  - Every setting is the normalised integral of a strictly positive density, so the axis cannot fold back on itself or push a bar off the display at any setting
- **Loudness** — flat or ISO 226:2003 equal-loudness weighting, applied at display time so toggling it is instant

### Channels in Pre-process

- **Off by default** — *Stereo by default*, under Channels in the Pre-process
  view, decides what future analyses produce. It is a preference about
  generating data, not about showing it: a track already analysed in stereo
  keeps its channels whichever way it is set, and selecting a channel view is
  what loads them. Turning it off deletes nothing
- **Three separate actions** when a track has no channels — analyse them for
  *this track*, *from now on* (the only one that changes the setting, and it
  names where to change it back), or *not now*
- **The mix is derived, not analysed again.** `M = (L + R)/2` on the complex
  per-member responses, before magnitude and before the geometric mean. That is
  the mix exactly, since the transform is linear. It is not the average of two
  finished spectra — two channels in opposite polarity are both loud and sum to
  silence, and only the complex form knows that. Every route does it this way,
  the GPU included
- **Cost** — two channel convolution sets rather than three analyses. On one
  track on one machine, a joint analysis took 2.26× a mix-only one; that is a
  single measurement and varies with content, settings and machine
- **Loaded on demand** — the sidecar is read on a worker when a channel view
  asks for it, and released on return to Mix. The file stays, so asking again
  reads rather than re-analyses

### Whole track (Pre-process)

The finished analysis of the whole track as one time–frequency picture, drawn
from the stored cells (no smoothing, no ISO 226 weighting). Opening it never
starts an analysis.

| On the plot | Does |
|---|---|
| Wheel | zoom time: around the playhead while following (following carries on), around the pointer otherwise |
| Shift + wheel | zoom frequency, around the pointer (following carries on) |
| Drag | pan (does not seek; sets following aside) |
| Click | seek playback to that time; paused stays paused |
| Middle click | fit the whole track (does not seek) |
| Overview strip | click or drag to move the time window |

The toolbar has **Fit track**, **Time ±** and **Freq ±** for the same
without a mouse wheel or middle button (Time ± zooms around the playhead while
following, around the middle otherwise), and **Mouse help** lists the
gestures.

- **Follow playback** (on by default, remembered) moves the time window with
  the playhead every frame — not a page at a time — keeping the zoom and the
  frequency range, with the playhead a third of the way across. Zooming in
  time while following zooms around the playhead, which keeps its place, and
  following carries on; a frequency zoom does not touch it either. A drag or
  the overview strip sets it aside; **Resume follow** picks it up again at the
  current zoom. Pausing stops the scrolling; a click seeks and following
  carries on from there. At the whole-track zoom there is nothing to scroll.
- **What a pixel is.** Magnified, the one stored cell under it; reduced, the
  largest cell of exactly the frames and bars its pixel covers. That holds for
  every frame drawn while following or panning, not only once things stop:
  the picture is kept as columns, each keyed by the frames it covers, and a
  column is reused only when its frames are the same. The drawn time window
  sits on the pixel grid (within half a pixel of where it was asked to be).
- **Preparing a view.** Larger zooms, distant seeks and long pans can need
  background work. If a complete picture at the same size is already visible,
  it stays together with its own axes and readout until the replacement is
  ready; all planes then switch together. The corner says **Preparing view…
  showing previous view**. Wheel input continues to accumulate and following
  stays enabled. The old image is neither stretched nor relabelled. First
  opening, resizing, changing the source or changing the palette can still
  show **loading** when there is no compatible complete picture to retain.
- **Repeat keeps the view.** Playing the same track again, with its analysis
  unchanged, restarts playback only: the zoom, the frequency range and the
  picture stay (following, if on, scrolls back to the start at the same zoom).
  Another track, changed audio, other settings or a newer analysis are loaded
  as before.

## Parametric EQ

- **Up to 16 bands** — Peaking, Low Shelf, High Shelf, High Pass, Low Pass, Notch
- **Biquad IIR filters** (Audio EQ Cookbook) — applied in real time via a `rodio` Source wrapper
- **Draggable nodes on the spectrum** — click to add a band, drag horizontally for frequency, drag vertically for gain, right-click to remove
- **EQ overlay modes** — Curve (response curve drawn over spectrum), Apply (bar heights reflect EQ gain), Both

## EQ Presets

- **Global and song-specific presets** — two separate dropdowns, one active at a time
- **Auto-load on track change** — loads the last-used preset for the track, falls back to the default global preset, then empty
- **Modified indicator** — preset name shows `*` when bands have been changed; Update and Discard buttons appear
- **Pending-switch prompt** — switching presets while modified asks Save & switch / Discard & switch / Cancel
- **Full preset management** — Save As New, Rename, Duplicate, Delete (with confirmation), Set as Default (★)
- **Persistent** — stored as JSON in `~/.moosik/eq_presets.json`

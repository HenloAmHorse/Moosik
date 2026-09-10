# Moosik

<p align="center">
  <img src="assets/icon.png" width="120" alt="Moosik icon"/>
</p>

A desktop music player with bit-perfect output (PCM and DSD), a
professional-grade spectrum analyzer, and a parametric EQ, built in Rust.

<p align="center">
  <img src="screenshots/player.png" alt="UI" width="700"/>
</p>

<p align="center">
  <img src="screenshots/spectrum.png" alt="Spectrum analyzer with parametric EQ overlay" width="700"/>
</p>

## Build

Requires Rust (stable, edition 2024). There are no prebuilt binaries — building
it is how you run it.

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

## What it does

- **Bit-perfect output** — the sample words handed to the driver are the
  source's own, unmodified. Exact formats only: a 24-bit file never negotiates a
  16-bit one, and if no format can carry the source the open fails with the
  reason instead of quietly converting
- **DSD** — `.dsf` and `.dff` as first-class formats, over DoP or native ASIO
  (Windows) and ALSA (Linux)
- **Spectrum analyzer** — pre-processed and real-time, seven visualisation
  styles, an adaptive superlet transform that beats an FFT at both ends of the
  spectrum, GPU offload, and per-channel views including a left/right difference
  plot. Channels work from the cache as well as live: a stereo analysis derives
  the mix from the two channels' complex responses rather than analysing it a
  third time, and stores them in a sidecar beside the cache
  ([docs/SPECTRUM.md](docs/SPECTRUM.md))
- **Parametric EQ** — up to 16 bands, dragged directly on the spectrum, with
  global and per-song presets
- **Lyrics** — synced `.lrc`, read from the file or looked up online, with a
  tap-along editor. Never writes to the audio file
- **Tag editing** — with verified, non-destructive writes
- **The rest** — gapless playback, ReplayGain, album-art overlays and mask mode,
  A-B repeat, bookmarks, sleep timer, play statistics, OS media keys, LUFS and
  correlation metering, and a light or dark theme that takes its accent from the
  cover art

### What "bit-perfect" claims here

The words Moosik hands the audio driver are the source's own. That is the whole
claim. The USB link, the driver's internals and the DAC are past the boundary of
anything this process can observe — a DAC's rate display and its lock light are
not evidence. Verifying the physical end of the chain needs capture equipment on
the link, and nothing in Moosik has been validated that way.

The four states this distinguishes, and the rules for each, are in
[docs/BIT_PERFECT.md](docs/BIT_PERFECT.md).

## Reading further

| | |
|---|---|
| [docs/BIT_PERFECT.md](docs/BIT_PERFECT.md) | the exactness contract in full, and DSD |
| [docs/SPECTRUM.md](docs/SPECTRUM.md) | the analyser, the superlet transform, and the EQ |
| [docs/FEATURES.md](docs/FEATURES.md) | album art, lyrics, tags, appearance, player |
| [docs/ICON.md](docs/ICON.md) | what the icon is |
| [CHANGELOG.md](CHANGELOG.md) | what changed, and why |

## Platform Support

"Status" here means what has actually been run, which is not the same as what
is believed to work.

| Platform | Status | Output backend | Exactness claimed |
|---|---|---|---|
| Windows | Builds, tests pass, hardware-verified for native ASIO DSD | WASAPI exclusive mode; native ASIO DSD | Yes — as far as the audio driver |
| Linux | Builds and tests pass; no DAC exercised | cpal (a direct ALSA `hw:` device avoids the mixing shims); native ALSA DSD | No — reported as *unverified*, because the platform route is not observable from here |
| macOS | Neither built nor tested | cpal — exact format requested, CoreAudio device behaviour decides the rest | No — same reason |

## Dependencies

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

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

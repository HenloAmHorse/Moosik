# Bit-perfect output, and what it claims

The exactness contract, in full: the four states Moosik distinguishes, the
format-negotiation policy, what happens when an exact route is impossible, and
what the claim does *not* cover. Split out of the README, which needed to be
readable in a sitting; nothing here has been shortened.

See also [SPECTRUM.md](SPECTRUM.md) for the analyser and
[FEATURES.md](FEATURES.md) for everything else.

## Bit-Perfect Output

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

## DSD Playback

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

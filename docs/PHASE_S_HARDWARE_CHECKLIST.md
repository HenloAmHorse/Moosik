# Phase S — owner hardware checklist

Everything below needs a real device, a real track and a pair of ears. None of
it is covered by the automated suite, which can only show that the mechanism is
wired correctly, never that the result looks or sounds right.

Base: `81862c345e533fc42e96aff2a18afd97e8069575` (v1.4.5).
Branch: `claude/1.5-spectrum-channels`.

**The artifact's identity is not in this file, and cannot be.** The build
embeds Git metadata, so writing a hash here and then committing it produces a
table that describes a binary the commit can no longer produce — it invalidates
itself. An earlier revision of this document did exactly that.

The authoritative identity is in the handoff report for the final commit, and
in `artifact-manifest.txt` beside the built binary, which is gitignored for the
same reason. Check the hash there before you start:

```
sha256sum target/release/moosik.exe
git rev-parse HEAD
```

If it does not match, rebuild — and close Moosik first:

```
rm -f target/release/moosik.exe
cargo build --release --locked --all-features
```

A build performed while the player is running silently leaves the old file in
place; `cargo` reports `failed to remove file ... (os error 5)` and the binary
you then test is a previous revision. Record the hash with your results, because
a result against an unidentified binary cannot be reproduced.

The commit range under test is `81862c3..HEAD` — nothing outside
`src/spectrum*`, `src/bitperfect/mod.rs` and a handful of engine teardown lines
in `src/main.rs` is touched.

Three things to know before starting, because all of them will look like bugs:

1. **Cached playback should feel like 1.4.5 again.** Its appearance was
   previously fixed at a hard-coded 0.5 per repaint, and an earlier revision of
   this branch replaced that with the live control's saved 0.75 — which is
   174 ms to 95%, against roughly 24 ms before, and is why the spectrum was
   reported as laggy. Cached playback now has its own setting, defaulting to
   0.125, which is ~24 ms at every analysis rate. If it still feels wrong, the
   Smoothing slider is the answer and its tooltip quotes the milliseconds.
2. **Left/Right are disabled in Pre-process mode.** The cache holds one mono
   row per frame. Switch to Real-time to exercise them.
3. **Left/Right have never actually worked before this build.** An earlier
   round shipped the controls but the display lease was revoked a moment after
   every track started, so they drew nothing. If you tested them on an earlier
   build and saw an empty plot, that is what you were seeing — check §5 and §6
   first.
4. **There are now two Smoothing values, one per mode.** Live keeps the
   per-tick behaviour it has always had, so it is unchanged from 1.4.5 —
   including that its response follows Max FPS. Cached is quoted at a 60 Hz
   reference and converted for the interval actually crossed, so it is the same
   on every machine. They persist independently; the slider shows whichever
   mode you are in.

---

## 1. Mix at 180 fps matches existing behaviour

Play a familiar track in Pre-process, mapping Superlet, `pre_fps` 180, Channels
= Mix. Compare against a v1.4.5 build side by side.

- Motion should be *smoother*, not different in character: all 180 rows a
  second now reach the temporal filter instead of every third one. They are not
  all *drawn* — the plot still shows one value per repaint — but the rows in
  between now influence it rather than being thrown away.
- Leave Smoothing at its default. On a ~180 Hz analysis the default produces
  alpha 0.5 per row, which is the number 1.4.5 hard-coded per repaint on a
  machine running at about that rate — so it should read as the same spectrum.
  Note that this is a coincidence of *your* rates: 1.4.5's response was whatever
  its repaint rate happened to make it, and had no single value to reproduce.
- Open the debug overlay (F3). `Src rows` should sit near `pre_fps` (~180/s),
  *not* near the `Tick rate`. On a v1.4.5 build the same number was effectively
  a third of that. `Src rows` counts rows consumed by the reducer, which is the
  quantity this change is about; it is not a frame rate and is not comparable
  to `REPAINT`.

**Fail if:** the spectrum stutters, lags playback, or the bass drifts against
the beat.

## 2. Left-only and right-only material shows isolation

Needs a test recording with content in one channel only (a hard-panned track,
or a generated file).

- Channels = Left on left-only material: full plot, silent on Right.
- Channels = Right on the same material: silent.
- Split: one populated half, one flat half, and the labels `L`/`R` correct.
- Overlay: one curve visible, the other on the floor. Confirm cyan is left.
- Diff: a bar hard against one end of the ±20 dB axis, on the side the content
  is panned to — magenta above for right, cyan below for left. On balanced
  material it should sit close to the centre line, which is the reading the view
  exists for.

**Reaching them at all:** they are **Real-time only** — the cached analysis
stores one mono row per frame — and they need Bars, Line or Filled. The toolbar
says which of those is missing next to the control.

**Fail if:** both channels show the same thing on hard-panned material — that
is the exact failure mode this feature exists to avoid.

## 3. Split and Overlay stay smooth at 180 fps

Real-time mode, `max_fps` 180, a busy track.

- No tearing, no stalling, no visible difference in cadence between the two
  halves of Split.
- Resize the window while running; both halves must track together.

## 4. CPU cost, Mix versus Split/Overlay

Task Manager or Process Explorer, same track, same window size, 60 s each:

| view | CPU % | notes |
|---|---|---|
| Mix | | baseline |
| Left | | one extra pair of transforms |
| Split | | should match Left |
| Overlay | | should match Left |

Mix performs no channel FFT — that much is asserted by a test that counts them
— but its total cost is *not* known to be unchanged, and this table is how that
gets established rather than assumed. Pre-process now folds in rows it used to
discard, which is real arithmetic that was not happening before. Measure Mix on
this build against Mix on a v1.4.5 build, in both modes.

The channel views should cost roughly one extra real-time FFT pair per frame.
The overlay's `Channels` line reports the transform rate directly.

**Fail if:** Mix costs materially more than it did in v1.4.5. A small
Pre-process increase is expected and should be quantified here, not waved
through.

## 5. Shared PCM and bit-perfect PCM

Both taps publish the channel count and feed the same buffer.

- Shared (normal) output: Left/Right enabled and correct.
- Bit-perfect PCM, WASAPI exclusive: same.
- Bit-perfect at a high rate (176.4/192 kHz) so the FFT size auto-scales to
  16384 or 32768 — this is what the old 8192-frame stereo cap could not serve.
- Watch for any callback-integrity warning in the log throughout.

## 6. Transport transitions

For each of: seek forward, seek backward, pause/resume, next track, gapless
transition, repeat-one loop wrap.

- The spectrum snaps to the new position rather than smearing from the old one.
- Peak markers do not hang from the previous position.
- In a channel view, L/R do not briefly show the previous track.
- The waterfall does not gain a block of duplicated rows.

Gapless specifically, on the shared route:

- Let a track run into the next one without touching anything. The spectrum
  must keep showing **A** until the moment the audio changes — the successor is
  handed to the mixer up to two seconds early, and taking the display then would
  be visible as the plot changing before the sound does.
- At the rollover it must transfer once, not flicker between the two.
- Queue a next track, then skip past it before it plays. The display must not
  have been disturbed by the track that never played.

On the bit-perfect route a gapless hand-off between compatible tracks keeps the
same output stream, and therefore the same tap: Left/Right must keep working
straight through the boundary without a blank frame.

**Fail if:** a seek produces a visible replay of the intervening audio, or the
plot changes track before the audio does.

## 6b. Mode switching

- Play in Pre-process for a while, switch to Real-time, leave it 10 s, switch
  back. The spectrum must resume at the current position, not replay the
  seconds you spent in the other mode.
- The waterfall must start empty after each switch rather than continuing a
  history taken on a different time base.
- Do the same while **paused**: the switch must take effect immediately, not on
  the next play.

## 6c. Stopping

- Press stop while a channel view is showing. Left/Right must go blank and the
  controls must disable, giving "no live PCM" as the reason — not freeze on the
  last frame of the track that just ended.

## 7. Mono and native DSD give an honest reason

- Play a mono file: Left/Right/Split/Overlay disabled; hovering says the track
  is mono.
- Play a DSD file on a native DSD path (ASIO or ALSA DSD): the controls say
  there is no live PCM to analyse. **Specifically check that it does not show
  the previous track's channels** — start a stereo PCM track first, let it run,
  then switch straight to native DSD.
- If you have any surround material: the reason should name the channel count.

## 8. Smoothing at 0, 0.5, 0.75, 0.97

In both Real-time and Pre-process, all four mappings:

- **0** — no smearing at all; bars land exactly where the analysis put them.
- **0.5 / 0.75** — progressively softer.
- **0.97** — very slow, but still moving; not frozen.
- Changing the slider must **not** trigger reanalysis and must not invalidate a
  cache. If a progress bar appears, that is a bug.

## 9. Same decay at 60 and 180 max FPS

The headline property. Same track, same position, same smoothing:

- Set `max_fps` 60, watch a decaying peak.
- Set `max_fps` 180, watch the same passage.

The decay should take the same wall-clock time in both. On v1.4.5 the 180 fps
case decayed roughly three times as fast.

Repeat for 144 and 240 if a monitor allows.

## 10. No audio dropout, no callback-integrity warning

Throughout everything above, and especially during 3, 4 and 5:

- No underruns reported.
- No allocation or blocking warnings from the render path.
- Spectrum window open the whole time, at the highest `max_fps` the machine
  offers.

---

## 11. The Diff view draws what it says

Diff needs a live stereo tap, so: Real-time, a Bars/Line/Filled style, and a
stereo track playing. Pick something with obvious movement between the channels.

**Orientation.** Settings → *Diff layout* → Horizontal. Frequency runs left to
right along the bottom with hertz labels under it; the difference runs up and
down with signed decibel labels — `+10`, `0`, `-10` — down the left-hand margin
and a brighter line across the middle at zero. Switch to Vertical: the two swap
outright. Frequency now runs down the left with hertz labels beside it, the
difference runs left to right with its decibel labels along the bottom, and the
zero line is vertical down the middle.

What must **not** appear in either: a `-70`, `-80` or `-40` gridline. Those
belong to the ordinary level axis, and seeing one means the wrong axes are being
drawn.

**Flip freq.** Tick it. Every frequency label must move to the mirrored position
*and* the bars must move with them — pick a band you can see moving, note which
end it is at, and check it is at the other end afterwards. A label that stays put
while the bars move is the bug this was written to catch.

**Swap L/R.** Tick it. A band that was above the line moves below it, and the end
labels swap so the one naming `L` follows the left channel. **The colours must
not move.** Left is cyan and right is magenta in every arrangement: if swapping
the sides recolours a band, the key is lying. Check the end labels are coloured
to match — the end labelled `L` in cyan, `R` in magenta, whichever end each is
at.

**Fallback.** With Diff selected, stop playback, or switch to Pre-process, or
pick a mono track. The plot must fall back to the ordinary Mix spectrum **with
ordinary axes** — `0` to `-70` gridlines down the left, frequency along the
bottom — not a Mix spectrum with difference axes over it. The Channels row says
why Diff is unavailable.

**EQ overlay.** Turn the EQ overlay on and select Diff. The curve and its
draggable nodes disappear and a line at the bottom of the plot says so. Switch to
any other view: the overlay comes straight back, with the same bands and gains.
Nothing about the EQ should have changed — check a band you had set is still set.

**Frequency labels on a warped scale.** Set the frequency scale to ERB or a
Blend/Lens tilt, in Mix. The hertz labels should sit under the bars they name; a
1 kHz label should land on the bar a 1 kHz tone lights up. Previously the bars
were warped and the labels were not.

**Labels land on their bars at 44.1 kHz.** This one needs a 44.1 kHz track and
the default 24 kHz maximum — the case where the ceiling is above Nyquist, which
48 kHz material hides. Play a 1 kHz tone, or find a strong tonal peak you can
identify, and check the tick names the bar the peak is in. It used to sit about
six pixels to the right on a 900 px plot, in Mix and in Diff alike. Then switch
the same track to 20 kHz maximum and confirm nothing moves relative to the bars.

**Raised minimum frequency.** Set Min Hz to 500. The 50, 100 and 200 Hz labels
must **disappear**, not stack up against the left-hand end. Anything piled on the
endpoint is the old clamping behaviour. Check in Mix and in both Diff
orientations, with Flip freq on and off.

**Endpoint labels are whole.** In Horizontal Diff, the `+20 dB` label at the top
of the difference axis must be completely readable — not sliced by the top of the
plot area. In Vertical Diff, the same for the label at the right-hand end. Turn
Swap L/R on and check both again: the labels change which channel they name, and
both must still be complete. Resize the window narrow and short and check once
more; the labels should stay inside rather than getting trimmed.

## Not covered by this phase

Recorded so the gaps are known rather than discovered:

- **Accessibility.** No screen-reader testing has been done on the new
  Channels control.
- **CJK input** near the new control has not been re-checked.
- **Peak hold and album-art masking are Mix-only.** Selecting a channel view
  drops both; whether that is acceptable is an owner call.
- **Overlay ignores the Bars/Filled style selector** and always draws lines,
  because overlapping filled areas hide whichever is drawn second.
- **Pre-process stereo** is deliberately out of scope; it needs the
  channel-aware v4 cache.
- **The waterfall advances once per analysis update, and its depth is a
  setting.** One row per cached row consumed in Pre-process, one per accepted
  analyser tick in Real-time. An intermediate revision capped this at 60 rows/s,
  which on a ~180 Hz display scrolled three times slower than 1.4.5 and threw
  away two of every three analysed rows; that cap is gone.

  **Waterfall history** in the spectrum settings sets the span in seconds
  (0.2–8, default 0.67, persisted). The ring is a fixed number of rows, so the
  span is that count divided by the rate rows arrive at; the count is shown
  beside the slider. Check that the span you pick holds when you change mode or
  Max FPS — the rate changes underneath it and the row count is meant to follow.

  **In Real-time the duration is nominal.** It is sized from Max FPS, which is a
  ceiling requested of the analyser rather than a measured rate. If your machine
  does not reach it, the same rows cover more time and the history reaches
  further back than the slider says — the label beside the slider says
  "nominal" and quotes the rate as an upper bound. In Pre-process it is
  measured: every cached row is consumed.

  What to check on hardware: at a Max FPS your machine comfortably exceeds, a
  0.67 s setting should look like roughly 0.67 s of history. At a Max FPS it
  cannot reach — try 240 with a heavy superlet pre-process running — expect the
  history to reach *further* back than the slider says, not less far. That is
  the nominal behaviour working as described, not a regression.

  The ring is held between 32 and 2048 rows. At the ends of the slider a bound
  binds and the span stops matching the request; the label then shows the span
  the ring actually holds instead.

  A stall still loses rows; nothing is synthesised to fill one.

---

## The artifact under test

Deliberately not recorded here. The build embeds Git metadata, so a hash
committed to a tracked file describes a binary that the commit containing it
can no longer produce.

Authoritative sources, in order:

1. the handoff report for the final commit;
2. `target/release/artifact-manifest.txt`, written beside the binary by the
   build step and gitignored.

Both carry the same fields: HEAD, absolute path, size, SHA-256 and the PE
file and product versions.

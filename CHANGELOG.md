# Changelog

## [Unreleased]

## [1.4.5] - 2026-09-01

### Fixed

- **The spectrum keeps its frame rate while the player window is minimized
  (Windows).** eframe sleeps the event loop for 10 ms whenever the window it
  has just painted is minimized. The spectrum is an immediate viewport, so it
  is painted inside the player window's pass — minimizing the player throttled
  a window that was still on screen, from roughly 170 FPS to roughly 80. Moosik
  now vendors the exact published `eframe 0.31.1` and waives that sleep only
  while a visible, non-minimized descendant, linked through an unbroken chain
  of immediate viewports, still depends on the minimized viewport's pass.
  Anything unknown keeps the sleep. Minimizing the player when no visible,
  non-minimized immediate descendant remains behaves exactly as before, as does
  every non-Windows build. The rule is generic, not spectrum-specific: the
  lyrics, tags and track-info windows are immediate viewports too, and keep
  their frame rate on the same terms. See `vendor/eframe-0.31.1/PATCH.md`.

  Minimizing only the spectrum while the player stays visible still costs what
  it did before: an immediate viewport shares its parent's repaints, so the
  work continues whether or not the child is on screen. That is existing
  behaviour, not a regression, and it is not addressed here.

## [1.4.4] - 2026-08-30

Output-truth repair, on top of 1.4.3.

1.4.3 made the diamond a measurement instead of a preference. This makes the
*rest* of the player agree with it — the lifecycle, the decoder, the backends —
and adds an honest answer for Float32 files on integer-only DACs.

The Float32 route is hardware-verified on an SMSL C200 Pro: the DAC rejects
`32f` in exclusive mode and accepts `32i`, and the file now plays continuously
through the Q1.31 conversion, labelled amber Processed, with no close/reopen
loop and no fault. That is one route, one DAC, one file, on Windows.
Nothing else here has been heard. **The evidence boundary for the rest of this
release is: automated tests, run on Windows and on Linux, at the seams named in
the closing section — and nothing else.** No other DAC, no ASIO driver, no ALSA
hardware, no macOS, and no listening. Several repairs sit *below* that boundary
because every path to them opens a sound card; the closing section names each
one and what stands in for a test there.

**Corrections to what an earlier draft of these notes claimed.** Several
described work as finished that was not, and one described a route that could
not work. They are kept rather than edited away, because the pattern is the
point: every one of them was a true sentence about the *shape* of a repair
standing in for a false one about its effect.

- The Float32 ladder was described as "raw `32f`, then a conversion checked
  sample by sample, then a rounded conversion". Those last two were not two
  rungs: both request identical 32-bit integer output, so the first always won
  the negotiation and the second was unreachable — and the runtime guard then
  *faulted* any ordinary off-grid file that reached it, which is the common
  case and exactly the material the route existed to carry. There is one
  integer rung now, described below.
- Routing was described as reflecting the active session. It did not: it read
  three request flags, so with the preference on, a session running on the
  shared mixer was treated as a device stream.
- "A durable Q31 verdict cache" was listed as not implemented. It is now.
- "No Linux or macOS build was attempted — the ALSA repair is reasoned from the
  types, not compiled" was true when written. The Linux build is now compiled
  and tested; macOS still is not.
- Reading the integrity state was described as one coherent read once each
  field carried a generation stamp. Stamping settles whose evidence a field is
  and says nothing about when it was read; three stamped loads still describe
  an instant that never held. Corrected in the entry below, and the read now
  retries against a count of publications.
- The value-exactness verdict was described as checked against the audio clock.
  It was checked against the *folded copy* of that clock, which lags the device
  by up to a frame — which is the whole window the check exists to close. It
  reads the live stream now, before and after.
- The restart was described as one transaction. The transaction was real, and
  the route still opened at position zero and seeked afterwards, so the
  listener heard the beginning of the track before the sentence's guarantee had
  anything to guard.
- The rollback after a failed device change was described as fixed by reporting
  its error instead of discarding it. There was no error to report: the
  rollback re-read state the failed attempt had already set to `Stopped` and
  returned success without doing anything.
- The status line was described as owned. Ownership was necessary and not
  sufficient: the now-playing line stored the whole rendered line as its
  fingerprint rather than the headline, and the ASIO probe wrote text without
  an owner at all and inherited whatever was there.
- "Everything else below is verified by automated tests on Windows and Linux"
  was too strong in one direction and too vague in the other. Replaced with an
  explicit evidence boundary above and a list, in the closing section, of what
  sits below it.

Four words are used everywhere, in code, badge, tooltip, panel and log:
**payload exact** (the driver got the source's own words), **value exact** (the
representation changed but no number did), **processed** (something was
altered), **unverified** (nothing Moosik did altered it, but the platform path
downstream cannot be proved). Only payload exact is green.

### Fixed

- **Reading the integrity state could describe two tracks at once.** The UI
  asked whether the audio generation was still the one it had folded and then
  made three separate calls for the rounding count, the recoverable code and
  the fatal one. A check placed before three reads cannot be about what happens
  between them, and the device crosses a gapless boundary at the sample: the
  set of facts that came back could be an off-grid count from the incoming
  track beside a dropout from the outgoing one, with nothing anywhere able to
  notice. There is one call now and every field in it is read by the
  generation's own stamp; the per-field accessors are deleted rather than left
  unused, so the pattern cannot be rebuilt out of parts.

  Stamping settled *whose* evidence each field was and nothing about *when*.
  An earlier draft of this entry called the result one read of one instant; it
  was three atomic loads that could not name the wrong track and could still
  describe a moment that never held — a terminal code beside an empty
  recoverable slot, which reads as a track that died with nothing having gone
  wrong.

  **The first attempt at a fix was itself wrong, and two of the claims made
  for it are withdrawn.** It counted publications and bumped the counter
  *after* each store, so the store was visible before the count moved: a
  reader could load the count, read a field a writer had already changed, load
  the count again unchanged, and call that a snapshot. "An unchanged count is
  a true snapshot" was false, and the exact schedule it fails on is the one it
  was built for — a dropout and the write failure after it, both stored,
  neither counted. The second claim, that the last attempt could read the
  terminal record first "because nothing may be published against a generation
  after its fatal record", was also false: a decode thread unwinding after a
  write failure, and both native callbacks, revoke afterwards.

  What is there now is a window rather than a count. Every writer announces
  that it is *about to* change something before it changes it and announces
  again when it is done, as two independent counters — not one odd/even
  counter, because writers do not exclude each other and two overlapping
  publications leave a single counter even in the middle of the second one. A
  reader that sees the two equal, reads, and then sees the first unchanged has
  read across an interval no writer was inside. When it cannot find such an
  interval it answers from the union of what its attempts saw and marks the
  answer unsettled, and the one caller that grants a claim on this evidence —
  the value-exact label — refuses to act on an unsettled read. The same
  protocol on PCM, ASIO and ALSA: one function all three call, exercised
  through all three public adapters.
- **A value-exactness verdict could be applied to the wrong file.** The scan
  recorded the session generation, which is the UI's counter and does not move
  at a gapless hand-off until the UI folds one — so a verdict computed for the
  track that was playing when the scan began could be printed against the track
  that had since replaced it. It records the audio generation too, and needs
  both.

  The audio generation it compared against was the folded copy, which lags the
  device by up to a frame — so an earlier draft's claim that the verdict is
  checked against the audio clock was true only of a clock that had not caught
  up yet. It reads the live stream now, before and after the evidence the
  verdict rests on, so a hand-off during the check is caught rather than
  straddled. A verdict is a label on the session, not a transient: had it gone
  on the wrong track it would have stayed there.
- **A new open showed the previous session's evidence.** `begin_open` left the
  last track's source and the last session's dropout in place, so "Verifying
  output…" appeared beside a specific claim about a file that was no longer
  playing, on a route that did not exist yet. A failure did the same for a route
  that never opened at all.
- **A failed reopen said "Verifying output…" for the rest of the session.** A
  seek that got as far as reopening the output and then failed left the
  transport in its opening state, because the branch that publishes a stopped
  state only fires when no handle is held — and the handle the reopen was about
  to replace was still there. The typed seek reason is published instead.
- **A dropout was invisible on processed and unverified routes.** The integrity
  line was shown only when the fidelity was already faulted, which is the one
  case where the headline mentions it anyway; the routes where the loss was the
  only record of it were exactly the ones that hid it. A processed route keeps
  its badge — a dropout does not change what it is doing to the audio — and the
  loss is in the detail on every live route.
- **The status line went on saying things that had stopped being true.** It
  compared badges rather than sentences, so a dropout replaced by the backend
  write failure that ended the track — two headlines under one `Faulted` badge
  — was never refreshed. And it kept ownership through a stop, so a stopped
  track had "Playing: …" put back by the next tick.

  The line has an owner now, and the text and the owner are one value with
  private fields and three ways in: the now-playing line, which carries the
  fidelity headline it was rendered from and is the only thing that may be
  regenerated; a message the listener has just been given, which is nobody's to
  overwrite; or nobody. Two things that an owner alone did not fix have gone
  with it. The now-playing line stored the *whole rendered line* as its
  fingerprint, title included, which the live headline can never equal — so it
  regenerated on every tick rather than never, the same defect wearing the
  opposite disguise. And the ASIO PCM probe wrote the text and left the owner
  alone, so a probe result run during playback inherited the now-playing line's
  claim and was wiped by the next tick; there is now no way to write the text
  without saying who it belongs to.
- **A backend that died while paused was not noticed until Resume.** The
  completion poll lived inside the playing branch, which is right for advancing
  a playlist and wrong for noticing a failure: a paused session still holds an
  exclusive device, and a driver reset while paused left the player sitting on
  it. Auto-advance is unchanged — a paused track has not ended — but a failure
  is not an ending.
- **A restart played the beginning of the track before going where it was
  asked.** Opening and seeking were two operations: every route opened at zero
  and the caller seeked afterwards, so a bit-perfect toggle or an output-device
  change mid-song emitted the opening of the track first — audibly, every time
  — and if the seek then failed, that opening was all the listener got, from a
  device the player was still holding.

  An earlier draft called this fixed by making the two halves one transaction.
  The transaction was real and it released what it took, but the audio at
  position zero was emitted before anything could undo it, so the sentence
  described half the repair. The position is an argument to the open now. The
  PCM and DoP routes hand it to `prepare`, which seeks the container before a
  device is touched; the native route hands it to its own opener; the decimated
  fallback starts at the byte offset; and the shared mixer holds the sink
  paused across the append and the seek, because a `rodio` sink plays the
  moment a source is appended to it. There is no second half, so there is
  nothing left that can fail after audio has started.
- **A paused restart played before it paused.** Pausing is a flag the callback
  reads; starting the output is what lets the callback run. Every route set the
  flag *after* the start, or let the caller set it a statement later, so a
  bit-perfect toggle or a device change made while paused emitted the opening
  of the track — on an exclusive DAC, the first sound after a silent gap — and
  then went quiet. Nothing about the final state records the difference: an
  engine that played for a moment and then paused looks exactly like one that
  never played, which is why it survived four passes of tests that checked the
  final state.

  The intent is an argument to the open now, next to the position, and the
  ordering is one function every route goes through.

  The first attempt at this was worse than it looked. The ordering function was
  real and correct, but only the exact-output route called it: the DoP route and
  both native DSD routes still resumed unconditionally on the line before their
  own starting call, and the test that was said to cover them drove a recorder
  those three routes never touch — a passing test for a fix three of four routes
  had not received. All four now come up through the same adapter, and each is
  tested over the stream type it actually holds, asked at the instant its own
  output step runs whether its own flag is set.

  The renderers had the other half of it, and installing the pause correctly
  did not close it. All three read `paused` before they contend for the session
  slot, so a renderer already past that read could be handed a session the
  pause installed behind it — and it played that session's first buffer, which
  on an exclusive device is the sound a paused restart exists to prevent. Each
  now reads the flag again once it holds the slot: `pause` publishes with
  `Release` and the installing thread takes the same mutex, so acquiring it is
  what makes the second read reliable. Nothing is consumed before it, so a
  paused hand-off takes no frame, moves no position, starts no drain, and puts
  each transport's own silence on the wire — PCM zero, DoP's marked `0x69`,
  native DSD idle. A `try_lock` that missed because the UI thread was
  mid-install is no longer counted as a dropout either; it was, and that turned
  the hand-off itself into a revoked exactness claim. The schedule is forced in
  a test on each of the three renderers rather than argued about.
- **A route that failed to start could become the current track.** The PCM path
  recorded it before the stream started, so a failed open left the engine naming
  a file it had never played — which is the record `resume` reopens after a
  failed seek.
- **Four callers of the restart each interpreted the result differently.** Some
  set the play state, some did not, one discarded the error outright, and none
  agreed about the pause state or the position. There is one outcome and one
  description of what applying it means. Paused stays paused.
- **The rollback after a failed device change did nothing at all.** Reporting
  the rollback's error instead of discarding it — which an earlier draft
  described as the fix — made no difference, because there was no error to
  report: the rollback called the ordinary restart, which reads live state, and
  the attempt it was rolling back had just set that state to `Stopped`. The
  restart's first statement was therefore `return Ok(())`. The listener was
  told the previous output had taken the track back. It had never been asked.

  A restart now carries its own request — which track, where in it, whether it
  was paused — captured before the attempt. The rollback replays that request,
  so it restores the position and the pause state and not merely the track, and
  when the previous output will not have it back either, both reasons are
  reported.
- **Starting a track released nothing when it failed.** Every other open was a
  transaction; `play_index` was not, and it had a second copy of the policy
  ladder besides. A DSD open that failed after the ASIO driver had been loaded
  published the typed reason and left the driver loaded and the endpoint held
  by a player that had stopped. Starting, restarting, the three device seeks
  and the decimated-DSD rebuild now go through one open-attempt transaction:
  if the body fails, the sink is stopped and dropped, the exact stream
  released, the native session closed, the scan retired, the current track
  cleared — a route that never played is not what is playing — and the
  failure's own typed reason published.
- **A rebuild that could not happen was reported as a seek that landed.** The
  decimated-DSD in-place rebuild and the background seek worker both returned
  `Ok(elapsed)` on failure: the position playback happened to stop at, handed
  back as though the decoder had moved there. The bar moved, the spectrum
  followed, and the only sign that nothing was playing was that nothing was
  playing.

- **A superseded decoder could make itself current.** The rollover minted its
  generation unconditionally, so a decode thread parked between taking its
  queued successor and rolling came back after a new track had started,
  allocated a number *newer* than the replacement session's, and stored it as
  the decode clock. Every ownership check downstream then compared against that
  number and passed: its boundary, its end-of-source, its fault and its
  rounding evidence all reached a session they had nothing to do with. A stop
  flag cannot close this — it is read before the same window. The rollover is a
  compare-exchange from the generation the caller believes it owns.
- **A hand-off could be half-published across a reset.** Publication was one
  check followed by four separate stores, so starting a new track in the middle
  of it cleared a queue the publisher then pushed into and zeroed a count the
  publisher then incremented. The result was a boundary belonging to a dead
  track sitting in a live session's queue behind a pending count that never
  came back down — and a pending count that never comes down is a session that
  can never end cleanly. Claim and stores now happen under one lock that the
  reset also takes, so a publisher is either wholly published and then wholly
  cleared, or wholly rejected.
- **A stale decoder could erase its successor's end-of-source, or its rounding
  evidence.** Both were a check followed by a store, and a thread parked
  between the two stored anyway. The end-of-source case left the track that had
  genuinely finished unable to end — the session sat at almost-complete until
  the drain's wall clock failed it, and a failure never advances a playlist.
  The rounding case gave the new track the old one's off-grid count, which is
  the number its value-exact label is decided by. Both are compare-exchanges
  conditional on what was read.
- **The reason a track stopped could be replaced by the dropout before it.**
  The public fatal accessor fell through to the recoverable record when there
  was no terminal one, so the UI pushed a dropout into the one first-wins field
  that holds the reason a session ended — and the backend write failure that
  actually stopped the track arrived at an occupied slot and was discarded. The
  listener was told the music stopped because of a dropout. The fatal accessor
  is fatal only.
- **A dropout left the diamond up.** With the two records separated, the
  recoverable one needed somewhere visible to be. A session that dropped out
  cannot go on describing itself as exact — the DAC played something that was
  not in the file — so the badge goes amber while the music keeps playing, and
  a fatal reason arriving later takes the headline with the earlier loss kept
  in the detail.
- **A dropout in the incoming track was charged to the one that had just
  finished.** The render thread crosses a gapless boundary at the sample and
  the UI finds out on its next frame; integrity was polled before that boundary
  was folded. The track that ended cleanly was marked as having lost its claim
  and the track that really dropped out started clean a moment later — both
  surfaces wrong, in opposite directions. Boundaries are folded first, and the
  stream also publishes which audio generation its evidence belongs to so that
  a poll running ahead of the fold skips rather than mis-stamps.
- **A cached status line went on showing a diamond the session had lost.** The
  now-playing line embeds the fidelity headline in a stored string; it now
  records **the headline it was rendered from** — the whole sentence from
  `Presentation`, not the badge — and is refreshed when that changes. (This
  entry said "the badge it was built with". Recording the badge is what the
  first version of the repair did, and it is the thing later work had to
  undo: a dropout replaced by the write failure that ended the track is two
  sentences under one `Faulted` badge, and a badge comparison leaves the first
  one on screen.)
- **Turning bit-perfect on during an unsupported track threw the setting
  away.** The restart had no fallback policy — it propagated the error, and the
  toggle flipped itself back. Starting the same track from the playlist fell
  back to the mixer and kept the setting, so one question had two answers and
  the worse one belonged to the path the listener reached by asking. Both go
  through one reducer now, and it cannot touch the preference because it is not
  given it: under Automatic or HQ the track falls back for itself, under Strict
  it stops and says why, and in both cases the next track starts again from the
  top of the ladder.
- **A DoP route recorded what it was playing before it knew.** The current
  track was published at the top of `start_dop` with an unknown source, so a
  successful open reported nothing about a file it had just parsed and a failed
  one reported a track that never played — which is the record `resume` reopens
  after a failed seek. It is published after the stream starts, with the DSD
  source, and a failed open publishes nothing.
- **A decimated DSD track rolled over as an identity transform.** The
  gapless rollover set the transform in the fidelity and not in the processing
  record, so the one surface whose job is to list what is being done to the
  audio said nothing was.

- **Two tracks could be handed the same generation number.** Every stamp check
  in the audio path compares a generation, and a generation was arithmetic: the
  decode clock advanced by adding one to itself, the playback clock by adding
  one to whatever it was about to overwrite. With a track queued the decode
  clock runs ahead, so a seek or a new track — which takes the *playback* clock
  and adds one — was handed the number the queued decoder was still using. Every
  check then compared them, found them equal, and let the superseded thread
  through: its end-of-source became the new track's, its rounding evidence
  decided the new track's badge, a fault it had parked was raised against a file
  the listener never asked for, and a boundary it published described a track
  that was no longer queued behind anything. Generations come from one process
  allocator now — handed out once, used by one session, never seen again — so a
  replacement outranks both clocks without anyone comparing them.
- **A gapless hand-off could be crossed before it existed.** The realtime
  descriptor the render thread reads was published *before* the metadata
  describing the track, so the device could cross a boundary whose source,
  transform and path had not been written down, hand the UI a token for it, and
  leave the UI popping an empty queue: the token spent, the description gone for
  the rest of the track. The pending count went up last, so a crossing could
  arrive before it and give back a boundary that had never been counted — which
  wrapped to `u64::MAX` and left "a track is still queued" true forever, so the
  session could never end cleanly again. Publication is ordered now — metadata,
  count, generation, then the frame that claims it — and the UI never spends a
  token without metadata to spend it on.
- **A dead backend stopped being asked how the track ended.** The predicate that
  routes "how did this end?" to the object holding the session also asked
  whether that object still worked, so the moment a WASAPI backend died or an
  ASIO driver requested a reset the engine stopped asking the only thing that
  knew. Completion fell through to the shared-sink branch, found no sink, and
  answered "still running": the track never ended, the playlist never advanced,
  and the exclusive handle stayed open for the life of the process. The failure
  had been recorded correctly, in a place nobody was reading. Ownership and
  liveness are separate questions now; liveness decides stream reuse and nothing
  else.
- **Native DSD never wrote down what it was playing.** ASIO and ALSA opened a
  device, played a file, and left the engine's record of the current track
  holding whatever the last other route had put there — and that record is what
  `resume` reopens after a failed seek. Both publish the path, the parsed DSD
  source and the starting position now.
- **The shared mixer described files by its own internal format.** Any track the
  exact route had not already prepared was published as 32-bit float at the
  sink's channel count, because that is what `rodio` converts everything to: a
  24-bit FLAC the DAC had refused was reported to the listener as a float
  source. The container is asked instead, and an unreadable one is unknown —
  which is a real answer with a word for it.
- **A gapless rollover on the shared mixer could claim the audio was
  untouched.** It asked the function that decides whether a *device* route
  preserved the samples, and on the mixer the answer is known in advance: it
  converts to float and applies the volume control, the equaliser and
  ReplayGain. A track whose channel layout could not be checked came back
  Unverified — the badge meaning nothing was altered — of exactly that path, and
  a DSD file on the decimating fallback came back as an identity transform of a
  path that had just resampled it. The route travels with the queued track now,
  decided where it is appended.
- **Turning bit-perfect on or off mid-track restarted the wrong kind of
  route.** The choice between the instant restart and the full reopen was made
  from two global preferences rather than from the track. A PCM file with the
  toggle off and an ASIO driver configured for DSD took the reopen path — a deep
  hi-res seek on the UI thread, which is the freeze the shortcut exists to
  prevent — and a DSD file with the toggle off and no native driver was handed
  to a decoder that cannot read a DSD container at all, so a track playing a
  moment earlier failed to restart. One planner answers for both, and it asks
  about the track.
- **"Seeking to 3:20…" stayed on screen after the seek landed.** It is there to
  say the position shown is a request and not a landing, and nothing took it
  down — so it was still up at the one moment it was false.
- **Frames were counted before the device had them.** The listener's position —
  and the elapsed clock, the seek bar and the gapless boundary arithmetic built
  on it — moved forward when audio left the ring rather than when the device
  accepted it. An ASIO callback that found a null buffer pointer for one channel
  wrote a partial frame and still counted the whole one; an ALSA period was
  counted in full before the first `writei` attempt, including the periods a
  dying device never took. Both count on acceptance now, and ALSA re-offers
  whatever a partial write left behind.
- **A dropout in the last seconds of a track was recorded against nothing.** The
  drain begins where the session slot is given up, so the ALSA writer has no
  session to name the track with from that point and was using zero — a number
  no session holds. An XRUN during a drain, which is exactly where one is most
  likely, left the route still claiming to be exact. A drain records whose it
  is.
- **The reason a track stopped could be hidden by the dropout before it.** The
  recoverable loss and the terminal reason were pushed to the same first-wins
  field, loss first because it happened first — so the reason that ended the
  audio arrived at a slot already taken and was thrown away. A track that
  dropped out and then had its write rejected said "a dropout" and never said
  the device had stopped accepting audio. They are two fields: the badge is what
  ended the track, the loss is in the detail beside it.

- **A track that failed reopened itself, forever.** Every fatal path — a read
  error, a decode error, a dead backend, a panicked thread, a driver reset —
  set the same `finished` flag the end of a track sets, because that flag's job
  was "stop waiting for audio". The auto-advance read it as "the track ended"
  and started the next one, which under Repeat One is the *same* file. A user's
  log shows one deterministically failing track opened hundreds of times in a
  row. Ending and failing are now different states, and only *ending* advances
  anything; a failure stops once, keeps its reason on screen, and is reopened
  by nothing but the user.
- **A track was reported finished while the device was still playing it.** The
  end of the ring is not the end of the audio: up to a full exclusive buffer
  was still unplayed, so the next track began over the tail of the last one.
  Completion now waits for a bounded drain — on the WASAPI/CPAL path and, since
  they had the same defect, on native ASIO and ALSA too, where the buffer the
  driver still holds is a DSD bitstream the DAC is locked to.
- **A shared-mode decode that stopped in the middle advanced the playlist.**
  `rodio`'s decoder yields `None` for a corrupt frame exactly as it does for
  the end of a file, so an empty sink could not tell "played out" from "gave
  up" — and "played out" is what it was taken as. A track that ends well short
  of its declared length is now a failure, which is the one thing that never
  advances.
- **Every visible fact rolled over up to a second early at a gapless
  boundary.** The generation advanced when the *decoder* crossed, and the
  decoder runs up to a full ring ahead of the device. The badge, the source and
  the transform therefore described the incoming track while the outgoing one
  was still audible. There are two clocks now — one for decoding, one for
  playback — and everything the listener sees follows the second.
- **A decode error in the queued track stopped the track that was playing.**
  Same cause: a fault raised while the decoder was a ring ahead was attributed
  to the audible track, which then halted partway through for something wrong
  with a file that had not started. Such a fault now waits until the device
  reaches the track it belongs to.
- **Rounding evidence was per stream, not per track.** The incoming track's
  rounding was counted against the outgoing one — the track still being heard,
  and the track whose value-exact label the count decides. Each of the two
  tracks in flight has its own.
- **A fault and the generation it belonged to were two separate atomics.** A
  reader could see the new code beside the old generation and discard a live
  fault, or the old code beside the new generation and report one the current
  session never had. Each stamped value is one word now. The native backends
  had no generation at all, so a fault from the track that just ended was still
  readable against the one that replaced it.
- **A truncated DSD file opened the device before it was refused.** The feeder
  caught it, but by then an ASIO engine was running and the DAC was locked to a
  DSD stream, so the failure arrived as a fault on a route already claiming to
  be exact. It is refused at the source, before any device is touched.
- **A DSD seek that failed said nothing.** The DoP fallback arm discarded its
  error with an `if ... .is_ok()` and no `else`: the player went silent at the
  old position, which is indistinguishable from a seek that worked into
  silence. It reports through the same typed channel as every other seek.
- **The odd-tail rule was judged on the wrong file.** Whether a DSD track ends
  on a half-full carrier frame was recorded when a DoP session *opened*, so
  after one gapless hand-off it still described whichever file had opened the
  stream. It travels with the boundary.
- **A Q1.31 stream could not carry a second Float32 track.** The reuse check
  asked whether an integer stream could carry Float32 — true of the source,
  irrelevant to the question, since nothing is sending it floats. On an
  integer-only DAC every track boundary in a Float32 playlist therefore forced
  a full device close and reopen instead of a gapless hand-off.
- **One off-grid track withheld the value-exact label from every track after
  it** for the life of the stream, because the rounding latch was per stream
  rather than per track. It is cleared at each hand-off, along with the
  generation, the pending scan, and the source, transform and fidelity — which
  used to roll over only in part, so a new track kept the old one's badge.
- **A stale positive verdict could outlive the evidence.** A cached
  value-exact result was checked once, when it arrived; the conversion goes on
  producing evidence for the whole track. A file rewritten between its scan and
  its playback now costs a label instead of producing a false one.
- **A DSF file cut off inside a block row played fabricated audio.** A row is
  `channels` runs of `block_size` bytes, so the bytes present belong to the
  first channels; dividing the byte count by the channel count pretended the
  shortfall was shared evenly. Stereo lost the right channel; multichannel
  played zero-filled scratch as the recording's own bits. A short row is now
  refused, and a truncated DSD file is refused before a device is opened rather
  than discovered mid-track on a route already claiming to be exact.
- **A failed background seek said nothing.** A device that would not open left
  the pending seek in place and retried it every frame forever; a container
  that refused the seek cleared it with no message and no state, so the player
  stopped mid-track with the old position still on the slider. Seeks now report
  a typed outcome, and the shared route's state is published only once a sink
  is actually installed.
- **The output panel described the processing chain the track started with.**
  Enabling an EQ band, or switching the EQ off, changed the chain and nothing
  recomputed the description.
- **A shared fallback described every track as Float32**, because that is the
  mixer's internal format. A 24-bit FLAC the DAC had refused was reported to
  the user as a float source; the format established by decoding is kept and
  used.
- **The ASIO callback table was leaked on every open**, and every early return
  after `createBuffers` released the driver without disposing its buffers —
  while it still held a pointer to a table about to be freed.
- **The realtime allocation test warmed its paths before arming the probe**, so
  a buffer whose reservation did not take grew on its *first* flush and the
  measurement reported zero. `reserve` takes an amount additional to the
  current length, not a target capacity, and the tap's request was computed
  from capacity — a buffer reused across streams was left under-reserved.

- **A stopped player kept its diamond and kept your DAC.** `stop()` ended the
  session but left the exclusive/native handle open and the state saying
  `streaming`, so the badge described a device nothing was being sent to and no
  other application could open it. Stop now releases every handle and clears the
  claim; the request and the policy survive untouched.
- **Switching from an exact route to ordinary playback did neither.** The shared
  path was reached with the WASAPI-exclusive stream still open and its exact
  state still published — so `rodio` failed in a way that looked like a missing
  device. Handles are released before shared output opens, and the shared route
  publishes itself as processed with the live volume, EQ and ReplayGain state.
- **One unsupported track disabled exact output for the whole session.** A
  single file the DAC could not take switched the feature off for everything
  after it, until the user noticed. Fallback is per track now and never touches
  the preference — the policy reducer that decides it is not given the
  preference to touch. (The named function in the original text no longer
  performs the fallback; the sentence described 1.4.2's code and stopped being
  a description of anything.)
- **A persisted preference rendered as an achievement at startup.** Loading
  `enabled = true` set the flag and left the session state alone, so the two
  disagreed from the first frame. A request now shows as a request.
- **Arrow-key volume overwrote your saved level on an exact route.** The slider
  was disabled; the keyboard was not, and wrote `self.volume` before consulting
  anything. Every input path now shares one predicate, and the value is only
  written if the engine accepted the change.
- **Failed seeks moved the display and not the audio.** `format.seek`'s result
  was discarded, so a container that refused the seek carried on decoding from
  wherever it was while the UI, spectrum and elapsed time all moved. Seeks are
  typed failures and the time base follows the position actually reached.
- **A decoder reset was treated as end of file.** `SymError::ResetRequired`
  means "rebuild the decoder", not "the track ended" — so tracks stopped
  wherever a reset happened, silently, still claiming to be exact. Reset and
  read errors are now typed faults, and a damaged first packet fails the open
  rather than starting a session that has already lost audio.
- **Precision was checked once and then assumed.** A clean priming packet could
  establish a 24-bit plan and a later packet with content in the low eight bits
  would reach a 24-bit writer intact. Every packet now goes through the same
  validator — family, rate, channels, layout and effective precision against the
  plan the device was opened for — and a packet that fails publishes nothing.
- **Impossible bit depths wrapped into plausible ones.** `bits_per_sample` was
  cast from `u32` to `u8` before it was checked, so 256 became 0 and 257 became
  1, and the result then steered the output format.
- **Truncated DSD files ended cleanly.** The declared audio length was clamped
  to the file size and forgotten, so a truncated file looked like a complete
  shorter one — at a gapless boundary, indistinguishable from a track finishing.
  Both lengths are kept and running short is reported with the counts.
- **Multichannel DSD claimed a layout it had not read.** DSF `channelType` and
  DSDIFF `CHNL` identifiers are parsed into real speaker masks; a type that
  disagrees with the channel count, or an unknown identifier, leaves the layout
  unknown. Above stereo, an unknown layout withholds the exact claim.
- **WASAPI validated the channel mask against a default instead of the source.**
  The default for three channels is FL/FR/FC, so a FL/FR/LFE source was
  negotiated, accepted and "validated" as FL/FR/FC — LFE content to the centre
  speaker, with the diamond lit. The source's own mask is requested and checked.
- **A changed default endpoint was invisible.** The reuse key was built from the
  open stream and compared back to it, so it could only ever match. The target
  endpoint is resolved independently before every reuse decision, and identity
  is the endpoint's stable ID rather than its friendly name.
- **A dead backend was reused.** A render thread that exited on a write error
  left the handle looking alive, so the next track was handed a stream with
  nobody behind it and played silence. Backend death is published separately
  from session faults and prevents reuse.
- **ASIO started on whatever was in the driver's buffers.** Both halves are now
  prefilled with correct DSD idle before `ASIOStart`, without consuming source,
  and the open waits a bounded 1500 ms for a first callback — `ASE_OK` only
  means the driver accepted the request. Every buffer pointer is validated; a
  null used to be skipped in silence, so one channel played nothing.
- **LSB1 drivers got MSB1 silence.** Audio bytes were reversed for an LSB1
  driver but padding, lead-in, tail, pause and underrun silence were left at a
  fixed `0x69`. `0x69` reversed is `0x96`, and they are not the same byte.
- **Every ASIO callback logged.** `bufferSwitchTimeInfo`, `sampleRateDidChange`
  and `asioMessage` each formatted a string and took the logging mutex on the
  thread with the hardest deadline in the process. They publish atomics now, and
  driver events — rate change, reset, resync, overload, buffer-size change — are
  handled as faults rather than noted as trivia.
- **Realtime paths could allocate.** The CPAL callback resized its scratch; the
  spectrum tap could grow its own batches and both shared analyser buffers from
  inside a render callback, because it appended everything and trimmed
  afterwards — which is exactly when a `Vec` grows. Both are now bounded, with
  an allocation-counted test over the complete render-and-tap chain.
- **The Linux default-feature build did not compile.** The ALSA session took
  `rtrb::Consumer<u8>` while the call site passed a `FrameConsumer<u8>`. It now
  consumes whole byte-frames through the frame ring, like every other backend.
- **A decode thread that failed to start left a silent session claiming to be
  exact.** Spawn results were discarded with `.ok()`. Every start returns a
  `Result`, the session is installed only after a successful spawn, and a guard
  turns a thread that ends without finishing — including by panic — into a
  visible fault.

- **The playback clock and the terminal state were two atomics.** Every
  transition was a check against one followed by a write to the other — which
  is two transitions with a window between them, and the window is wide enough
  for the thing it was meant to prevent. Track A's decode thread reads the
  generation, finds its own, is descheduled, and wakes to write its failure
  into a word that now belongs to track B; B fails before its first sample,
  blaming a file it never opened. Installing a generation and ending one are
  now the same operation on the same word, each a compare-exchange, on the PCM
  path and on both native backends.
- **One dropout stopped the playlist.** Integrity and completion were one
  thing, so a track that dropped out was a track that had failed — and a
  failure never advances. They are separate: a dropout, a missed callback lock
  and an off-grid sample cost the badge and nothing else, while a decode error,
  a dead backend, a session that never primed and a torn output ring end the
  session. A torn ring is on the fatal side and this list had it on the other:
  a dropout is a hole and the audio resumes, but a ring holding a non-multiple
  of the channel count can no longer say which channel its head belongs to, so
  every sample after it plays on the wrong one for the rest of the track. An ASIO overload has its own code rather than borrowing the
  survivable one it is not.
- **A drain that outran the UI thread ended a track that had not started.** The
  ring empties past a gapless boundary before anyone has popped it, so "the
  ring is dry" and "there is nothing left to play" are different statements
  whenever a successor is queued. The last track of a gapless run was reported
  finished before it began.
- **The decode guard carried the generation its thread was born in.** A panic
  three tracks into a gapless run was stamped stale and vanished — the decoder
  was gone and nothing anywhere said so. It follows the track the loop is on,
  and a fault for a track that has not started waits for the device rather than
  stopping the one being heard.
- **A background seek reported four different failures as one.** A file that
  could not be opened, bytes that could not be decoded, a thread that would not
  start and a container that refused all arrived as `None` and were reported as
  "this file could not be seeked", which for three of them is untrue. The
  spawn failure was discarded outright with `.ok()`. Each says which it was.
- **A seek past the end of the audio was reported as a landing.** The worker
  stopped discarding samples when the decoder ran out and handed it back
  anyway, sitting at EOF, while the caller published the target it had asked
  for — the slider moved to a position the file does not contain and the track
  ended at once. The result carries where the decoder actually is.
- **A seek that failed left the engine in a half-state.** No sink, the clock
  frozen, and the slider showing a place nothing was playing from — which to a
  listener is indistinguishable from a seek that worked into a silent passage.
  Nothing is committed until the sink exists, and every failure puts the
  position back where playback was.
- **A shared-mixer track that stopped short claimed a read error.** `rodio`'s
  decoder yields `None` for a corrupt frame exactly as it does for the end of a
  file, so nothing had observed a read error; a truncated file, a decoder
  giving up and a wrong duration tag are identical from there. The inference is
  named for what it is, and its rule is a function of values rather than a
  method on a `Sink`, so it can be tested without an audio device.
- **Both native drains counted the wrong units.** `ASIOGetBufferSize` answers
  in 1-bit DSD samples and the callback fills bytes, so the ASIO drain waited
  for sixteen buffers instead of two. ALSA had it inverted — a period is PCM
  frames of `bps` bytes each — so on `DSD_U32` it ended after half a period,
  which on DSD is the DAC losing its lock mid-hand-off. Both count byte-frames
  now, and both count what the device actually accepted rather than what it was
  offered.
- **Neither native drain was bounded in time.** A drain is advanced by
  callbacks, so a driver that goes quiet mid-drain leaves the track at
  almost-finished for the life of the process. The wall clock is checked from
  the UI thread, and an expired drain is a failure — the device did not play
  what it was holding.
- **A started ASIO driver had no owner.** Four exits between `ASIOStart` and
  the host thread returning each had to remember to tear down, and a panic
  remembered nothing: the driver was never stopped and never released, so the
  DAC kept running and the device stayed held until the process ended.
- **A driver reporting a rate change left its stream reusable.** Reset, resync
  and buffer-size change all refuse reuse; this one did not, so the next track
  was handed to a device that had already said it was running at a different
  speed.
- **A DFF sound chunk that is not whole byte-frames was accepted.** DFF
  interleaves one byte per channel, so a remainder belongs to the first
  channels of a frame whose remaining channels are missing —
  `data_len / channels` divided the shortfall evenly between them, rotating
  every channel from that point and reporting a length the file does not have.
  The same fabrication the DSF block reader already refuses.

### Added

- **Strict / Automatic / HQ output policy**, in the 🔈▾ menu and persisted with
  `serde(default)` so existing settings load unchanged. Strict never transforms
  and stops with the reason — including refusing to be moved onto the shared
  mixer, which has a volume stage, an EQ and a resampler in it. Automatic tries
  the source's own representation first and falls back to the conversion. HQ
  goes straight to the conversion, because that is what asking for it means.
  The choice applies per track and never changes your bit-perfect preference.
- **One Float32 → Q1.31 conversion**, which plays everything. A sample that
  lands on the Q1.31 lattice converts exactly; one that does not is rounded,
  deterministically — NaN to zero, clamped at both ends, `f64` multiply,
  ties-to-even, no dither. It is labelled **Processed** from the moment it
  opens.
- **Value-exactness as an observation, not a route.** Many Float32 files are 16-
  or 24-bit masters exported as floats, and for those the conversion changes
  representation and not one number. That is a property of the *material*, so
  it is discovered rather than negotiated for: the label is applied only when a
  complete decode of the whole track says every sample was representable *and*
  the running conversion has not had to round one. Either alone is not enough —
  a scan can be stale, and the conversion has only seen what has played. Never
  green: signed zero and NaN payloads do not survive a change of
  representation.
- **A durable verdict cache**, so a track proved once is not decoded in full
  again on the next launch. Written atomically through a temporary file and a
  rename, capped, and pruned least-recently-used. A missing, unreadable,
  truncated or foreign-schema file is a cold cache, never an error. It stays
  advisory: every packet is re-checked as it is converted, so a stale entry can
  cost a label and can never put a rounded sample on the wire unannounced.
- **A policy-controlled route for 64-bit float sources.** They have no exact
  representation in any device format that exists. Strict stops and says so;
  Automatic and HQ play them through the same Q1.31 conversion as Float32.
  The label names both halves of what happened — "Float64 → Float32 → Q1.31,
  narrowed then rounded ties-to-even and clamped" — because the narrowing to
  32-bit float happens in the decoder, before this route sees a sample, and
  that is where the numbers change. It follows that such a source can never be
  **value exact**, whatever the scan finds: the scan is looking at numbers that
  have already been altered. That is now enforced rather than assumed.

  The reason shown alongside it names the source rather than the DAC. It used
  to say the device had no Float32 exclusive format, which is why a *Float32*
  source takes this route and is not why this one does: a 64-bit float source
  would be narrowed on hardware that does not exist yet, and blaming the DAC
  invited the listener to go looking for one that would fix it.

  Three fixtures cover the route, on synthetic files: one puts a 64-bit float
  WAV through `prepare` and the decode loop and checks that each sample
  arrives as the `f32` nearest the original double; one runs the same file
  through the Q1.31 conversion the processed policies choose and checks the
  payloads that reached the ring against the two roundings in order, ending
  cleanly; one covers the ladder and the published fidelity under Strict,
  Automatic and HQ. The value-exact rule is checked separately, as a
  function of its four conditions.
- **An ASIO PCM capability probe**, in the ASIO menu and labelled a diagnostic.
  It opens a driver, asks what it would do with ordinary PCM — channels, buffer
  range, rates, sample type — and closes it again. It never creates buffers,
  never starts the driver and never sets a rate. Moosik does not play PCM over
  ASIO and this does not change that; it answers the question that has to be
  settled before a renderer is worth writing.
- **One authoritative session state** behind every surface, with generations, so
  a scan or a fault from a superseded session is discarded instead of applied
  to its successor.
- **A persistent output panel**: source, transform, carrier, transport and
  endpoint, negotiated format, and the live processing chain. DSD is reported as
  DSD with its DoP carrier shown separately, rather than as the 24-bit PCM the
  carrier happens to be.

### Not implemented, and so not exposed

Stated plainly rather than left to be discovered:

- **PCM/Float-to-DSD "HQ" output.** The gates for it include measured passband
  ripple and out-of-band noise spectra per preset and rate, and a long-run
  realtime budget on target hardware. None of that can be produced without the
  hardware, and shipping a modulator whose behaviour has not been measured
  would be the thing the gate exists to prevent. There is no setting, no route
  and no claim.
- **ASIO PCM playback.** The capability probe above exists; the renderer does
  not. There is no ASIO PCM setting, no ASIO PCM transport and no ASIO PCM
  claim anywhere — native ASIO DSD is unaffected.
- **Carrying an odd DSD tail across a gapless boundary.** A DoP carrier frame
  holds two DSD bytes per channel, so a file with an odd frame count ends on a
  half-full frame completed with DSD silence. Pairing the outgoing file's last
  byte with the incoming file's first would put two recordings inside one
  carrier frame; instead that boundary is declared non-gapless and re-opens the
  device. The cost is one track gap on odd-length DSD files.
- **Hardware validation beyond one acceptance run.** The Float32 route was
  verified on an SMSL C200 Pro: one `32f` rejection, one `32i` open, continuous
  playback of the whole track, amber Processed throughout, no reopen loop. That
  is the complete list of what hardware has confirmed.

  Nothing else here rests on hardware. Specifically, and because each of these
  is easy to read as more than it is:

  - The **drain bounds** on WASAPI, ASIO and ALSA are derived from buffer and
    period sizes that no real driver has been observed reporting in this build.
    The units are now right by construction and by test; whether the resulting
    wait matches a particular DAC's actual latency is unmeasured.
  - The **ASIO ownership and teardown order** is proved against a fake function
    table. That is a contract, not a driver: it shows Moosik calls `stop`,
    `disposeBuffers` and `release` in that order on every path including a
    panic, and shows nothing about how any real driver responds.
  - The **IO-format and rate readback** decide a badge, not audio. A driver
    that implements neither gets amber Unverified, which is a statement about
    the absence of evidence.
  - The **native DSD paths** — DoP carriers, truncation refusal, the odd tail,
    both containers — are exercised on synthetic files only.
  - The **failure-halt behaviour** is exercised through production reducers and
    forced thread schedules. The schedules are real — barriers, and hooks that
    park a thread *inside* `Shared::transition` and inside `fault_decoding`,
    rather than a sequential model. Two of them were written first without a
    hook and passed against the broken code, which is why the hooks exist: a
    rendezvous on either side of a call does not enter the window the defect
    lives in. Each of the schedule tests has been run against a mutant of the
    code it covers and fails against it. The hardware they stand in for is
    still not present.
  - The **ASIO process globals** (`ACTIVE`, `ENGAGED`) are written by several
    tests, and the harness runs tests in parallel. They are serialised through
    one scoped guard now. Before it, `driver_events_fault_the_session_and_never_log`
    failed on `ev_rate_changed` about one full-suite run in several hundred —
    another test's `EngineClaim::drop` had nulled `ACTIVE` between the store
    and the callback — and passed every time it was run alone. Worse than the
    flake: a `Fake` lives on the stack of the test that built it, so
    publishing its address without holding the guard published a pointer
    another thread could dereference after the frame had gone.
  - The **shared-mixer shortfall rule** has a three-second tolerance chosen as
    a judgement about how wrong duration tags usually are. It is not a
    measurement.

  macOS is still neither built nor tested.
- **A formatting pass.** `cargo fmt --all -- --check` exits non-zero on this
  tree and has since before this work began. Running it would rewrite most of
  the source and bury the changes above in a diff nobody can review, so it is
  deliberately deferred and recorded here as an outstanding debt rather than
  quietly satisfied. The count is a number of `Diff in` sections: 1104 at
  `940e58c`, where 1.4.4 was first prepared, and 1116 before this closing pass
  began. The final figure is in the release checklist below.

### How the gates were run

Recorded because "the tests pass" is not a claim anyone can check.

**Windows 11**, `rustc 1.94.0 (4a4ef493e 2026-03-02)`,
`cargo 1.94.0 (85eff7c80 2026-01-15)`, host `x86_64-pc-windows-msvc`,
`rustup` default toolchain `stable-x86_64-pc-windows-msvc`:

> An earlier draft of this section recorded 1.98.0 for Windows. That was the
> *container's* compiler, copied across when the two tables were written
> together; the Windows toolchain on this machine is and was 1.94.0. The
> numbers below were produced by 1.94.0 whatever the previous text said, and
> the two platforms are not on the same compiler.

```
cargo test   --locked --all-features
cargo test   --locked --no-default-features
cargo check  --locked --no-default-features
cargo clippy --locked --all-targets --all-features
cargo build  --locked --release --all-features
cargo fmt    --all -- --check
git diff --check
```

**Linux**, in a `rust:1-bookworm` container with `libasound2-dev` and
`libudev-dev` installed — `rustc 1.98.0 (88d9e12ae 2026-08-18)`,
`cargo 1.98.0 (797e8a9bc 2026-08-05)`, host `x86_64-unknown-linux-gnu`,
whatever the image happens to ship. The working tree is mounted read-only at
`/src` and copied to `/work`, so the Windows `target/` is never reachable:

```
cargo check  --locked --all-targets --all-features
cargo check  --locked --no-default-features
cargo clippy --locked --all-targets --all-features
cargo test   --locked --all-features
cargo test   --locked --no-default-features
cargo build  --locked --release --all-features
```

macOS: neither command was run, because there is no machine to run it on.

**What they reported**, on the commit these notes ship with:

| | Windows | Linux |
|---|---|---|
| `test --all-features` | 430 passed, 0 failed, 16 ignored | 407 passed, 0 failed, 16 ignored |
| `test --no-default-features` | 394 passed, 0 failed, 16 ignored | 389 passed, 0 failed, 16 ignored |
| `check --no-default-features` | exit 0 | exit 0 |
| `check --all-targets --all-features` | — | exit 0 |
| `clippy --all-targets --all-features` | exit 0, 27 warnings | exit 0, 84 warnings |
| `build --release --all-features` | exit 0 | exit 0 |
| `fmt --all -- --check` | **non-zero: 1206 sections — waived, not passed** | not run |

`cargo fmt` is a **waiver and not a pass**. It exits non-zero, it has done so
since 1.4.3, and the debt has grown with the comment volume: 1104 → 1116 →
1121 → 1150 → 1163 → 1183 → 1190 → 1196 → 1201 → 1206. Running the formatter would
produce a diff far larger than the work it would be mixed into, and mixing them
would make both unreviewable.

Both clippy runs were compared against the same command on an unpacked copy of
the immediately preceding commit, `0582f41`, and both are **identical** —
Windows 27, Linux 84, the same messages in the same numbers, with none in any
file this release touched last. The Windows set is also the set at `252032c`,
where the comparison started.

The two platform figures are not comparable to each other and are not being
compared: the container ships `rustc 1.98.0` and Windows has 1.94.0, four
releases of new lints (`chunks_exact_to_as_chunks`, `manual_checked_ops`,
`identity_op` and others that did not exist in 1.94), plus dead-code warnings
for the Windows-only backends that Linux does not compile.

> An earlier draft of this paragraph recorded 71 for Linux. That number came
> from a different grep in the container script — the run loop printed only
> lines matching `warning: unused`, and the total was taken from a summary
> line rather than from a count of warnings. Counted the same way on both
> commits, it is 84 on each.

**Test inventories and why the two platforms differ.** Windows lists 446
tests and Linux 423, of which 16 are ignored on both and are the same 16.
The difference is `cfg`, not coverage: 42 tests exist only on Windows
(`bitperfect::asio_dsd` 35, `bitperfect::wasapi_out` 5, `log` 1, and the native
ASIO route's pause-ordering test in `main`) and 19 only on Linux
(`bitperfect::alsa_dsd` 17, `bitperfect::cpal_out` 1, and the ALSA mirror of
that same test). ASIO is a Windows
API and ALSA is a Linux one, so each backend's tests compile on one platform and
not the other; the shared PCM, DoP, decode, fidelity and lifecycle tests run on
both.

42 − 19 = 23, which is 446 − 423. The `cfg` gates are
`cfg(all(windows, feature = "asio-dsd"))` and
`cfg(all(target_os = "linux", feature = "alsa-dsd"))` for the two backend
suites; `wasapi_out` and `cpal_out` are the two shared-path backends, and the
`log` test needs Windows file-locking semantics to make a delete fail. The
all-features/no-default difference is those same backend suites plus the
`asio-pcm-probe` diagnostics.

**The Linux matrix has caught five defects that Windows could not**, and they
are recorded because that is the interesting part of running it.

The fifth is this pass's. Three new evidence tests moved a stream into the
thread that reads it, which does not compile on Linux: `Backend` holds a
`cpal::Stream`, deliberately neither `Send` nor `Sync`, so a `BpStream` cannot
cross a thread boundary there. On Windows the WASAPI handle is `Send` and all
three built and passed — a test that exists on one platform and not the other,
found only because the whole matrix is run. The stream is now built inside the
reading thread.

A Linux step also failed twice on this pass for a reason that is not a result:
`check --all-targets --all-features` could not download a crate from
`crates.io`, once for `flate2` and once for `bitflags`. Both were re-run on the
same tree and passed. Recorded rather than quietly dropped.

Three of them belong to one pass. Two ALSA tests were still reading the
combined fault accessor and calling its answer "the fault" after that accessor
became fatal-only — the same edit had been made to every Windows-visible test,
and these two are compiled only on Linux. The third was worse: a test that
burnt exactly one generation and then asserted the next one shared its parity.
The allocator is process-wide and the suite runs in parallel, so how many
numbers another test takes in between is not knowable; the assertion held on
Windows by scheduling luck and failed on Linux, where more tests are compiled
and the contention is higher. It now rolls until the condition it is about is
actually met.

The fourth was the pass before that: `alsa_dsd::null_device_end_to_end`, an
end-to-end run against the `null` device, reported 92.9 ms of elapsed time on a
100 ms clip because a change to *when* frames are counted had stopped crediting
the last, partly-full period of a track.

Each time, the whole matrix was re-run on the corrected tree. An earlier phase had one Linux step fail on a
`crates.io` download timeout and re-run on the same tree; that is still true of
the run those notes described.

The 16 ignored tests are ignored on both platforms and are the same 16 in
every run. Counted from `--ignored --list` rather than from memory, because
the previous version of this paragraph was wrong in three of its five figures:

| Area | Count | What they need |
|---|---|---|
| `lyrics::lrclib`, `lyrics::netease` | 5 | a live lyrics service |
| `spectrum::aslt`, `spectrum::cache_key`, `spectrum::freq_scale` | 5 | long recordings and survey runs |
| `spectrum::gpu`, `spectrum::gpu_calib` | 3 | a GPU adapter |
| `tags` | 2 | a writable music library |
| `fonts` | 1 | an installed system font |

None of them covers anything in this release.

**The full Windows all-features suite was run 25 consecutive times** in the
default parallel configuration, on the source tree these notes ship with — the
only change made after it was to this file. The count per run is the one in the
table above: 423 passed, 0 failed, 25 times out of 25, with each run's output
kept and discarded only on a pass — so a failure would have left its log
behind. An earlier version of this paragraph still said 361, which was the
count three commits before it; the number was not re-read when the suite grew.

**An earlier attempt at that run was 22 of 25**, and it is recorded because
discarding it would make the clean one look like the only thing that happened.
Runs 2, 3 and 4 of that attempt did not report the expected line. Their output
was not kept — the loop only counted — so *what* failed in them is not known
and is not being guessed at — **and no cause is inferred for them here.** An
earlier version of this paragraph offered one, on the strength of a linker
error seen in the same session; that is a coincidence in time, not evidence
about those three runs, and it is withdrawn. The loop was re-run on the same
tree with each run's output kept, so that a repeat would be diagnosable rather
than merely counted, and it came out 25 of 25 with nothing to diagnose.

That is the honest shape of it: one attempt with three failures that were not
captured and remain unexplained, and one attempt that was clean and would have
produced evidence if it had not been.

Twenty-five clean runs is evidence about one hazard and not a general claim.
Before the ASIO global isolation the failure being chased appeared roughly once
in several hundred runs; twenty-five runs says that specific hazard is gone. It
says nothing about interleavings nobody has forced, and no claim is made that
the concurrent paths here are exhaustively covered. What is covered is what is
listed below, and nothing else.

**What is not covered.** Some of the repairs here are reasoned from the types
and the comments beside them, and nothing automated proves them, because every
path that reaches them opens a sound card first:

* that a PCM or DoP route which fails to start publishes no current track —
  the ordering is right in the source and the mutation that reverses it fails
  no test, because no test reaches `start_bp` or `start_dop`;
* that `Engine::completion` reaches a dead handle, which needs a handle that
  can die;
* that the ALSA writer loop credits what `writei` accepted, which needs a
  device that accepts partially;
* that each device route inside `play_file_at` opens *at* the target — the
  argument is threaded through in the source, and what a test can hold is that
  `open_track` asks for the target and never for zero;
* that the shared route stays silent until it has seeked — the sink is paused
  across the append and released after `try_seek`, and constructing a `rodio`
  sink needs an output device;
* that the three device branches of `Engine::seek_to` release what they took;
  the transaction is the same one, but the only failing seek a test can reach
  without hardware is the decimated-DSD rebuild, which is the one driven;
* that the two `rodio` routes — the shared mixer and the decimated-DSD rebuild
  — pause before their own append. They take the flag in the same place for the
  same reason as the four device routes, but constructing a `rodio` sink needs
  an output device, so no test executes that code. The four routes that hold a
  stream of their own are covered: the exact-output and DoP routes over a
  `BpStream`, and the two native DSD sessions over an `AsioDsdStream` and an
  `AlsaDsdStream`, each asked at the instant its output step runs whether its
  own flag is set. That correction is described above; the first attempt at it
  proved a generic recorder that three of those four routes did not call.

Two further boundaries are worth naming because they are not "no device":

* **The shared stream on the WASAPI-exclusive path is preempted, not
  released.** That is the intended rule for exclusive mode, and it is now
  written down beside the flag that records it — but no test exercises it, and
  nothing here claims every held resource is released on that path.
* **`App::stop` clearing the status line** is asserted on `StatusLine::clear`,
  not on the caller. That is the gap the brief names, and it is still a gap:
  `MoosikApp` cannot be constructed, and a wrapper function would be the same
  assertion wearing a different name.

And a boundary of a different kind: **`MoosikApp` cannot be constructed in a
unit test.** It takes an `eframe` creation context and reads the user's own
configuration and playlists on the way up, so the app methods a listener
actually triggers — `play_index`, `select_bp_device`, `toggle_bit_perfect`,
`restart_current_track`, the per-frame tick — are proved at the seams they
call, not as whole methods. Those seams are production code that the callers
have no alternative to: `open_track`, `switch_output`, `StatusLine`,
`restart_effect`, `Engine::open_attempt`, `Engine::abort_open` and
`Engine::halt`. What is not proved is the wiring between an app method and its
seam. That wiring is **not** always a single call, and saying it was is how
two of this release's defects survived a phase: the bit-perfect toggle and the
output-device selection each contain multi-step orchestration — read a
setting, restart, decide what the status line says, roll back if it failed —
and both of them were wrong in the part that is not the call. The two
decisions that were wrong are now reducers the callers ask, with a source
check that the callers have not grown a second copy; what remains uncovered is
the sequencing around them.

They are listed because a release note that says "tested on Windows and Linux"
without saying what could not be tested is the kind of claim this release
exists to stop making.

**Every schedule-dependent repair is checked against a mutant of the code it
covers** — the fix reverted, the test expected to fail, the fix restored. That
is the only way to tell a forced schedule from a test that would pass against
anything, and it has caught five tests written for this release that proved
nothing: two that rendezvoused outside the window a race lives in, one whose
seek targets could not produce the misalignment it was checking for, one that
asserted on a field it had just written rather than on the decision that reads
it, and one that spawned a competing thread without establishing that it had
been reached, so the interleaving it meant to force was a race it usually won.

The forced schedules, each parking a thread at a named hook inside the window
its race lives in:

| Hook | What is parked there |
|---|---|
| `Transition` | a stale writer inside a terminal transition |
| `DeferFault` | a decoder between finding its track queued and parking a fault |
| `Successor` | a real PCM or DoP decode loop, between taking its queued successor and claiming the clock |
| `BoundaryPublish(0..3)` | a publisher after the claim, and after each of the three stores |
| `DecodeEof` | a decoder between its liveness check and publishing end-of-source |
| `OffGrid` | a decoder between its liveness check and publishing rounding evidence |
| `Evidence(1..2)` | a gapless boundary crossing between two fields of one evidence read |
| `Evidence(3)` | a dropout and the failure after it, published inside the final ordered attempt at reading evidence |
| `ResetContend` | a reset immediately before it contends for the publication lock, so a publisher is released on a handshake rather than a sleep |

The mutants, each a rule reverted and its test confirmed to fail: the
render-owned boundary crossing; the stale drain latch; deferred-fault
monotonicity; per-session frame accounting; the shared seek resume; the native
fatal-over-recoverable override; the ASIO drain counting its own final buffer;
the fidelity ordering; counting frames before the buffers are filled; a drain
expiring against the live clock; an XRUN attributed to generation zero; both
integrity records pushed to one slot; an unconditional rollover; a reset
outside the publication lock; a check-then-store end-of-source; a
validate-then-store rounding count; the fatal accessor falling back to
recoverable evidence; a rollover keeping its predecessor's loss; a dropout
leaving the diamond up; a policy-blind open reducer; and a rollover leaving the
processing record identity.

Then, for the corrective pass that followed — the publication window, pause
intent and caller truth — seventeen more, every one caught: removing the begin
marker, the end marker, the fold, the retry and the quiescence check; bypassing
the PCM and ASIO public adapters; letting the value-exact caller ignore whether
its read settled; pausing a route after it has started; dropping the pause
intent on the way into the openers; telling the fallback rung to play; a seek
worker that ignores cancellation; both untyped error mappings; a toggle that
treats a selected track as a playing one; a paused session that does not halt
on a failure; and a toggle caller that decides for itself instead of asking.

Then, for the evidence, lifecycle and status work in this release: reading
evidence without retrying when a publication lands between the loads; the final
attempt reading the recoverable records before the terminal one; that attempt
answering from the last read rather than the union of what it saw; each of the
four things an open releases, dropped one at a time — the sink, the exact
stream, the native session and the scan — plus the typed reason, the cleared
current track, and the undo itself; the decimated-DSD rebuild reporting a
failure as a landing; opening at zero instead of at the target; skipping the
rollback; rolling back to a different position; discarding the rollback's own
error; a seek landing taking the status line when it had put nothing up; the
now-playing line storing the rendered line rather than its headline; a message
written without an owner; and the refresh firing for a player that has
stopped.

## [1.4.3] - 2026-08-25

Bit-perfect output stops being a preference and starts being a measurement.

The short version: the diamond used to come from the toggle. If you had it
switched on, it was green — while a DSD file was being decimated to processed
PCM, while a 24-bit track rode a 16-bit stream, while the volume slider sat at
80%. This release derives it from what the output path is actually doing, and
where the answer is "not exactly", it says so and says why.

One boundary, stated once and meant everywhere below: **exact means the sample
and payload words Moosik hands the Windows audio driver are the source's own
words.** What the USB link, the driver and the DAC do after that is outside
anything this process can see. A DAC's rate display, its DSD lock light and its
temperature are not evidence of anything, and nothing here claims otherwise.

### Fixed

- **A 32-bit integer source lost its lowest eight bits, silently.** The output
  ring carried `f32`, which has 24 bits of mantissa, so every 32-bit sample was
  rounded before a device ever saw it — and the diamond stayed green while it
  happened. The ring now carries a canonical `u32` payload: left-aligned signed
  integers, or raw `f32` bits, with the meaning fixed by the decoded source
  family. Packing to a device format is shifts and byte copies, never a float
  multiply and never a saturating cast.

- **A 24-bit file could negotiate a 16-bit device format and still be called
  bit-perfect.** Each source width had a five-format fallback order ending in
  `16i`, so a device offering nothing better truncated eight bits per sample
  and reported success. There is no longer any fallback candidate that is not
  exact for the source: 1–16-bit integers may use `16i`, packed `24i`,
  24-in-32 or `32i`; 17–24-bit may use the three wider ones; 25–32-bit may use
  `32i` and nothing else; a 32-bit float may use `32f` and nothing else. If no
  device format can carry the source, the open fails with the reason rather
  than negotiating something quieter. `MOOSIK_BP_FORMAT` obeys the same matrix
  — it can pick among the exact formats and can no longer authorise a
  narrowing one.

- **A 64-bit float source claimed exactness it cannot have.** No available
  device format carries one without rounding. It now plays through normal
  output, labelled *Exactness unverified*, with the reason attached.

- **The negotiated format was believed rather than checked.** The stored format
  came from the candidate that was *offered*. A driver that accepted a request
  and then described something else was taken at its word, which puts frames of
  the wrong width on the wire. Rate, channel count, channel mask, block
  alignment, container bits, valid bits and integer/float subtype are now all
  read back from the driver's reply and compared; a mismatch rejects the
  candidate and moves on.

- **Torn frames could rotate every channel for the rest of a track.** The ring
  was drained a sample at a time until it ran dry, which could stop between a
  left and a right sample. The leftover then became the *first* channel of the
  next callback, and every channel stayed shifted until some later odd-sized
  read happened to shift it back. The ring is now frame-atomic on both sides —
  whole interleaved frames in, whole frames out, capacity a multiple of the
  channel count — on the WASAPI PCM, WASAPI DoP and native ASIO DSD paths
  alike. A short read now leaves a whole frame behind instead of half of one.

- **DoP emitted PCM zeros wherever the decoder had not pre-packed silence.**
  The marker travelled through the ring, so only silence the *decode* thread
  knew about in advance — the lead-in and the tail — carried a valid DoP
  marker. The device prefill, a pause, an underrun, the gap after a seek and
  the final drain all sent frames of zeros, which a DoP DAC reads as loss of
  lock. Marker ownership has moved to the render thread: one phase counter for
  the life of the stream, one marker per complete channel frame, and marked DSD
  silence generated wherever payload runs out. No all-zero DoP frame can be
  emitted. A pause now also consumes no source at all, where it used to destroy
  audio by turning it into silence.

- **The volume slider appeared to work on a path that has no volume.** There is
  no multiply between the decoder and the device on an exact route, so a slider
  that moved was reporting a state the audio could not be in. While an exact
  PCM, DoP or native-DSD route is open the slider is disabled and reads
  `100% · locked for bit-perfect output`. Your saved volume is untouched and
  returns the moment normal playback resumes.

- **Reusing an open stream could narrow the next track.** The check was the
  requested device, the sample rate and the channel count. A 16-bit track
  followed by a 24-bit track at the same rate reused the 16-bit stream and
  truncated the second one — and the gapless queue used an even looser test
  than the reopen path, so the boundary case was the worst case. Reuse now
  compares the resolved endpoint, the route, the rate, the channel count, the
  channel layout, the source family and the negotiated container's valid bits;
  a wider integer container may still carry a narrower integer source, because
  that packing is exact. Both paths ask the same question, and the decode
  thread asks it again at the moment of the swap.

- **A read or decode error mid-track looked like the end of the file.** It was
  logged and then treated as a clean end, so the track simply stopped early and
  the session still claimed to have carried it exactly. Read errors, decode
  errors and a source that changes rate, layout or sample family mid-stream are
  now typed integrity faults.

- **Native ASIO DSD checked one channel and assumed the rest.** The bit order
  is applied once to the whole interleaved stream, which is only correct if
  every selected channel wants the same order — and a driver is free to report
  per-channel types. All selected output channels are now queried, and a mixed
  configuration is refused before the stream starts rather than played with one
  channel inverted. `DSD Int8 NER8` is also refused: it is 8-bit data at one
  sample per byte, an eighth of the data rate this path produces, and it used
  to be treated as MSB1.

### Changed

- **The diamond has five states, and they say what they are in words.** Colour
  is never the only signal:

  ```
  ◇ Bit-perfect off
  ◇ Bit-perfect requested — no active output stream
  ◇ Verifying exclusive output…
  💎 Exact output path · WASAPI Exclusive / DoP / Native DSD
  ⚠ Not bit-perfect · <reason>
  ⚠ Exactness unverified · <reason>
  ✕ Bit-perfect unavailable · <reason>
  ```

  A stored preference on its own is never green. Neither is a route that is
  still being negotiated — the window between pressing play and the driver
  accepting a format is exactly when a claim is least justified, and a stale
  diamond used to survive it.

- **A persistent output panel that transient messages cannot overwrite.**
  While something is playing, three lines stay put: the source format, the
  device and negotiated output format, and what is in the processing path.
  There used to be one status string that every subsystem wrote to, so the line
  describing the open device survived until the next tag save wanted to say
  something.

- **Any integrity fault revokes the claim for the rest of the session.** A
  dropout or a render callback that could not reach its session in time. A
  dropout means the DAC played silence that was not in the file; no later good
  buffer undoes that, and only a new session can claim again. A torn frame, a
  decode or read error and a backend write failure are listed here in error —
  they do not revoke a claim, they end the session, and 1.4.4 separated the
  two.

- **Native DSD is never described as "via DoP".** They are different transports
  with different ceilings, and the ASIO route exists precisely because it is
  not DoP.

- **Linux and macOS no longer claim exactness.** cpal asks for the exact
  format, and on a direct `hw:` device it very likely gets it — but ALSA,
  PipeWire, Pulse, `dmix` and CoreAudio can each resample or mix on the way to
  the hardware, and this process cannot see which route it got. The format
  policy there is now exact-only in the same way as on Windows (no narrowing,
  no cross-family conversion, `F64` refused), but the route reports
  *Exactness unverified* rather than green. Native ALSA DSD is treated the same
  way for this release.

- **Source precision is measured, not guessed.** `prepare` now decodes one
  packet before returning, so the family comes from the buffer symphonia
  actually produces rather than from a container hint that some formats do not
  provide. A declared depth narrower than the decoded storage width — 24-bit
  FLAC in 32-bit slots — is believed only if the samples honour it; content
  below the declared point widens the requirement instead. Nothing is lost to
  the probe: that packet is the first audio into the ring.

### Internal

- **Realtime threads publish faults as a `u8`.** No formatting, no allocation,
  no logging mutex and no filesystem on a thread holding a hard deadline; a
  non-realtime poller turns the code into words. The ASIO callback's remaining
  `mlog!` is gone, and a test reads the source to keep it gone. Another test
  measures the callback with a counting allocator and requires zero allocations
  in steady state — and calibrates the probe first, so "zero" means something.

- **New modules.** `bitperfect::format` (canonical payloads, the exactness
  matrix, the byte writers, DoP marker state), `bitperfect::frame_ring`
  (the frame-atomic ring), `bitperfect::state` (route, integrity and the reuse
  predicate). The DoP encoder's marker packer is gone — the marker is not a
  property of the audio.

### Not covered

Stated so it is not mistaken for coverage:

- **No hardware was validated for this release.** Every claim above is about
  what this process hands the driver, verified by tests. The DAC end has not
  been re-checked since 1.4.2, and doing so properly needs capture equipment on
  the USB link, not a DAC's front panel.
- **FLAC is not in the decoder fixtures.** The suite builds WAV (16-, 24- and
  32-bit integer and 32-bit float), DSF and DFF fixtures byte by byte; encoding
  FLAC would need a dependency this release does not add. The FLAC path is
  covered only through the shared canonical conversion, not end to end.
- **Linux and macOS builds are unverified here.** `cpal_out.rs` and the ALSA
  DSD backend are not compiled on Windows, where this was developed and tested.

## [1.4.2] - 2026-08-21

### Removed

- **EQ "Bake to Cache".** The control never did what it said. It computed an EQ
  fingerprint, deleted an `_eq…spectrumcache` file that nothing ever wrote, and
  then ran an ordinary analysis — which recomputes its own cache path without
  the EQ suffix and applies no EQ at any stage. Neither half of its tooltip was
  true, and it has been that way since the feature was announced. It is removed
  rather than left in place while a correct version is built.

  Nothing that worked is lost: real-time EQ on the normal PCM path, the DSD
  PCM-fallback EQ, presets, the response curve, and the Apply/Both bar overlay
  are all unchanged, as is EQ bypass on the PCM bit-perfect, native-DSD, and
  DoP routes.

### Internal

- **The analysis suite could fail intermittently.** Two tests steered each other
  through a process-wide GPU-routing threshold: one rewrote it while another
  demanded bit-exact equality between consecutive analyses, so under a parallel
  run one call went to the device and one to the cores — and those agree to a
  tolerance, not bit for bit. The routing decision is now passed in rather than
  read from a mutable global, and the mutable test setter is gone. Production
  routing behaviour is unchanged.

### Fixed

- **`MOOSIK_BP_FORMAT` could destroy DoP.** The override was applied before the
  DoP container check, so forcing `16i` truncated the 24-bit DoP word and `32f`
  re-encoded it as floating point — in both cases the DAC stops seeing DoP
  markers while the route still reports success. DoP now accepts a forced
  `24i`, `24i32` or `32i` and refuses `16i`/`32f` before the device is opened,
  rather than silently substituting a different format.

  Two consequences worth stating outright. An **unrecognised** value (a typo)
  is now an error as well — it used to be logged and ignored, which handed you a
  different format than the one you pinned. And a DSD file that fails for a
  *configuration* reason, or because the file itself will not open or parse, no
  longer falls back to decimated PCM: only an error currently classified as
  `Device` — a coarse backend/output-open category — may use that legacy
  automatic fallback. Processing audio in answer to a mistyped environment
  variable is not a fallback, it is a wrong answer.
- **The dropout diagnostic could cause dropouts.** The WASAPI render thread runs
  at MMCSS "Pro Audio" priority with a hard deadline, and 1.4.1 logged from
  inside its loop — formatting a string, taking a global mutex, and writing and
  flushing to disk, including on every change to the dropout count. That loop
  no longer logs at all; counters are accumulated and reported after the
  deadline is released. *(The ASIO driver callbacks and the CPAL error callback
  still log synchronously; that needs a lock-free breadcrumb transport and is
  not yet done.)*
- **Session logs could overwrite each other.** File names carried whole seconds
  only and were opened with truncation, so two instances started in the same
  second shared one file. Names now include the process id and are created
  exclusively.
- **Ten logs meant eleven.** Retention ran before the new file was created, so
  the documented ten became eleven — and a good log was deleted before finding
  out whether its replacement could be created. Retention now runs after a
  successful create, counts the current session, and ignores anything outside
  Moosik's current and accepted legacy reserved session-log filename schemas.

  Retention remains **per-process**: two copies of Moosik running at once can
  each prune the other's live log. Doing that safely needs a lock the operating
  system releases on crash, which is not in this release.
- **The Log button could point at nothing.** A failed create was discarded while
  the path was published anyway. Failures are now reported and the disabled
  button carries the reason.
- **A failed DoP seek moved the display anyway.** The seek result was discarded,
  so the spectrum advanced and the position readout settled on a target the
  decoder never reached. The seek is now applied to the source *before* the
  device or any engine state is touched, so a rejected target is not committed:
  the spectrum is not advanced, the seek bar snaps back to the actual elapsed
  position, and the reason appears as a status message rather than only in the
  log.

  DoP only. The decimated-PCM DSD fallback still ignores its own seek failure,
  and normal-mode seeks fail asynchronously in a worker; both remain open.

## [1.4.1] - 2026-08-17

A build on a second machine misbehaved in four different ways at once, with
nothing to show for any of it. So this release is a session log, and then the
bugs the log found — including one that killed the process outright.

### There is a log now

Every run writes a file to `~/.moosik/logs/`, ten kept. The **🗎 Log** button
next to 🎨 Look opens the folder; no console needed, which matters because the
build that needs it most is the one somebody double-clicked.

Every line is flushed as it is written rather than buffered, since a buffer
loses precisely the lines that describe the failure. Timestamps count from
process start — when reading a session back, what matters is that the stall
began 4 s after the analysis did, not what o'clock it was. The banner records
version, OS, cores, exe path and build profile before anything can fail, so
even a log with one line in it says what it ran on.

Panics record a forced backtrace. The previous crash log wrote a single line
saying something had died, without the path that reached it.

If you hit something odd, the log is the bug report.

### Fixed

- **Playing a 44.1 kHz track could kill the app.** `max_freq` defaults to
  24 kHz, above Nyquist for CD-rate audio, and the bin lookup bounded the top
  of its range but not the bottom — so a bar past Nyquist indexed off the end
  of the spectrum. Invisible on 48 kHz material, where Nyquist is *exactly*
  the default ceiling and nothing above it is ever asked for. Bars above
  Nyquist now read the topmost bin, which is what the constant-Q path has
  always done with them.
- **Bass appeared a second time at the top of the real-time display.** The tap
  fed interleaved channels into a buffer the analyser reads as one mono
  stream, making it an N-times sample-and-hold: every partial at f/N with a
  mirror image at Nyquist − f/N. Frames are averaged down now, as the
  pre-process decoder always did. Momentary LUFS was reading the same
  interleaved stream and is corrected with it.
- **Bit-perfect output could be noise, played fast, at a volume the slider did
  not affect.** Negotiation offered packed 24-bit before 24-in-32; Realtek
  reports support for the 6-byte frame and mishandles it. 24-in-32 is equally
  bit-exact and is now preferred. Negotiation also logs every candidate and
  refuses outright any format whose rate, channels or block alignment disagree
  with what was requested.
- **The app could start in the wrong theme.** egui defaults to following the
  system theme and re-applies its own visuals over ours; on a machine set to
  Light that landed on the first frame, showing stock light styling while the
  settings panel still read Dark.
- **Opening the spectrum settings could freeze the window.** The GPU probe ran
  under a lock held across six block sizes timed on both routes — 35.7 s on a
  four-core machine — and adapter enumeration ran on first touch. Both could
  land inside a repaint. A warm-up thread does the work at startup; the panel
  reports what is known and never blocks.
- **The analysis progress readout was missing during a first analysis** — the
  one case where it is most wanted. It keyed on whether cached frames existed
  while the centred notice keyed on whether the plot was silent, and a first
  run satisfied neither.
- **Dropouts are no longer counted while the ring refills after a seek.** That
  gap is the one you asked for by jumping. Every dropout observed in testing
  so far was this and nothing else, which made the counter noise.

### For diagnosing a device

- `MOOSIK_BP_FORMAT=16i|24i|24i32|32i|32f` pins the output format, so a format
  problem can be bisected on the machine that has it rather than the one with
  the compiler.
- `MOOSIK_BP_TRACE=1` logs each write: frames available, samples wanted and
  received, byte count against expected, and peak sample value.

## [1.4.0] - 2026-08-16

Two things: the pre-process learned to use the GPU, and the display stopped
assuming there is one right way to lay out a spectrum. Plus the honest
accounting that made both possible, and three bugs that had been quietly
wrong for a while.

### The pre-process uses the GPU

Measured per 5 minutes of audio at 1024 bars, against the same build with the
device switched off:

| preset   | before | after |       |
|----------|--------|-------|-------|
| Fast     | 0.7    | 0.7   | —     |
| Standard | 1.3    | 1.3   | —     |
| High     | 3.2    | 2.6   | 1.23x |
| Ultra    | 6.0    | 3.5   | 1.71x |
| Extreme  | 12.7   | 9.6   | 1.32x |

Fast and Standard are unchanged by design: their kernels never reach a block
size where a device beats eight cores, so they keep the CPU route.

Worth recording how this went, because the numbers do not show it. The first
working version made Extreme *slower* — 19.4 minutes against 12.7 — and stayed
that way through three fixes aimed squarely at the device. Per-phase timings
then showed it spending 0.11 s per chunk on the GPU and 0.46 s on the CPU work
that followed. The device had never been the bottleneck. What was: a parallel
loop over three bars on an eight-core machine, a full-signal transform being
redone per wavelet, kernels built one at a time on a single thread, and
dispatches that waited for the assembly instead of running beside it.
`MOOSIK_GPU_TRACE=1` prints those timings and stays in the tree.

The threshold between GPU and CPU is **measured on each machine, not
hardcoded** — it is a property of one device against one core count, and a card
that is slow but working would clear a fixed gate and never fall back. A probe
on first run picks a starting point; real analyses then A/B the two routes and
settle it on evidence, switching the device off by itself if it loses.

- Settings: GPU **Auto / Always / Off**, a **Re-measure** button, the per-size
  calibration table, and what the last run cost.
- Settings: an **analysis core budget**, defaulting to two fewer than the
  machine has.
- `MOOSIK_GPU=0` forces the CPU route, which is unchanged and is what every
  number above is measured against.

### Bar spacing is now a choice

Where the bars land, independent of how many there are. The count never
changes; every setting redistributes the same bars.

- **Bass width** — a slider from wider-than-log, through **log** (the classic
  analyser axis, still the default), to **ERB** (constant bars per auditory
  filter, Glasberg & Moore 1990).
- **Zoom** — a lens anywhere on the axis, with centre, width and strength.

The bass slider is not a resolution control, which is the surprising part. A
tone's blur in bars and a bassline's travel in bars both scale with bar
density, so the ratio between them is the same at every setting — 6.3 on log,
6.6 on ERB. What changes is how much screen the bass gets to move across.
Below a capped analysis window it is also free: the window fixes resolution in
Hz, so widening the bass adds no work at all.

The lens is not free. Above roughly a kilohertz nothing caps the window, so
concentrating bars there raises the resolution actually demanded. The panel
quotes the cost multiplier rather than leaving it to be discovered.

Every setting is the normalised integral of a strictly positive density, so no
combination can fold the axis back on itself or push a bar off the display —
that is structural, not a clamp. Selecting `log` delegates to the original
functions rather than reimplementing them, so it is not merely equivalent to
the previous behaviour, it is that behaviour; a test asserts exact equality
against the pre-scale code path.

### Audio output holds its deadline under load
- **The render thread registers with MMCSS as a "Pro Audio" task.** It was an
  ordinary-priority poll loop competing with a rayon pool that takes every core
  for minutes during a pre-process; losing that race by more than the device
  buffer is audible.
- **The worker pool leaves two threads free** rather than taking every hardware
  thread.
- **Dropouts are counted and shown.** A render call that comes up short with
  the decoder still running — not a track boundary, not a pause — is silence
  the DAC played, and the status line now says so. No test in this suite can
  observe a dropout, so this is the only evidence the process can produce about
  its own output.

### Fixed
- **ISO 226 weighting did nothing in pre-process mode**, which is the default.
  The weights were folded into the frames during analysis, but `loudness_mode`
  is not part of the cache key — so switching it changed nothing and
  re-analysing loaded the same file back. Whichever mode happened to be
  selected when a track was first analysed was the mode it kept, permanently.
  It is now applied at display time, one add per bar, and toggling is instant.
- **ISO 226 pinned everything from 2 kHz to 10 kHz to the ceiling.** Referenced
  to 1 kHz the curve asks for up to +22 dB there, and bar heights arrive
  normalised, so a positive correction has nowhere to go. The positive lobe is
  now clamped: 1 kHz keeps full height, that region reads as it does with
  weighting off, and the tilt below 1 kHz — the informative half — survives
  intact.
- **The debug overlay cost about half the frame rate.** It asked for the
  adapter name once a frame, and that built a fresh wgpu instance and
  enumerated adapters every time.
- **Analysis progress is visible without opening a panel.** It now draws on the
  plot itself whenever a track is being analysed. The old indicator only
  appeared when there was nothing else to draw, so re-analysing a track that
  already had frames showed nothing anywhere obvious.
- **The ETA notices when it is wrong.** It averaged over the whole run, which
  assumes the work ahead costs what the work behind did — it does not, since
  the short-kernel bars come last and cost more wall clock per tap. It now
  measures recent throughput, so the last stretch stops overrunning its own
  estimate quite so confidently.
- The Analysis settings panel is no longer called "FFT Settings". It has not
  been FFT-only since the superlet path landed.

## [1.3.1] - 2026-08-15

Everything 1.3.0 added, put through a real library — and mostly what that
found. Thirteen tracks that could not be matched, a font picker that silently
did nothing, a scrollbar in the middle of the window, and a waterfall doing
about a hundred times more work than it needed to.

### Lyrics: found the ones that were missing
- **A second source (NetEase Cloud Music) alongside LRCLIB.** Measured against
  a real set of thirteen failing tracks, LRCLIB returned *zero* hits for nine
  of them: a Japanese and Vocaloid library is largely absent from it, and no
  amount of better matching invents a record that is not there. NetEase had
  synced lyrics for effectively all of them.
- Genius and UtaTen were considered and rejected: neither publishes lyrics
  through an API (Genius's deliberately excludes them), so the only route is
  scraping HTML that breaks on any redesign — and neither carries timestamps,
  so even a successful scrape yields an unsynced sheet that still needs timing
  by hand. The paste box already covers that, without a scraper to maintain.
- **Wrong lyrics are now refused.** `SWIPE×SWIPE` was being given the lyrics of
  an unrelated song called `Swipe`: a synced sheet with a plausible duration
  outscored the nothing it earned on the title, and no rule required the title
  to agree at all. An automatic match must now clear a bar — and a matching
  title alone does not clear it, since covers and same-name songs both pass
  that. Wrong lyrics applied confidently are worse than none, because nothing
  tells you to go looking.
- **Matching works on Japanese now.** Word-token overlap is meaningless without
  spaces — every CJK title is a single token, so the score was 1 or 0 with
  nothing between. Replaced with a character-based measure that needs no word
  segmenter. Artist comparison also learned that a credit list is a superset,
  not a different artist: a tag of `TAK` against a database row of
  `TAK/Hatsune Miku` was being refused.
- **Length disagreements no longer lose a match.** A flat ±8 s gate threw away
  a record whose title *and* artist were identical, because a database recorded
  the track 9 s longer than the file. The gate now widens when everything else
  corroborates, and stays narrow when it does not.
- **A source that cannot answer never reports "no lyrics".** NetEase rate-limits
  by replying HTTP 200 with the real status buried in the body, which reads as
  an empty result to anything checking only the transport. Being told a track
  has no lyrics, when the truth was to wait a minute, sends you off to
  transcribe something for nothing.
- **"📋 Paste…"** takes lyrics from anywhere for the tracks no database has.
  Plain text drops straight into the sync editor; LRC keeps its timing.

### Appearance
- **Choose the UI font.** Families are read from each font file's own name
  table rather than guessed from filenames, so the list says *Fira Code Medium*
  rather than `FiraCode-Medium`; collections are handled, so every face in a
  `.ttc` is offered. A filter box, because a machine can easily carry a couple
  of hundred families. CJK fallbacks are reinstalled behind whatever is chosen,
  so picking a Latin-only font does not turn Japanese tags into tofu.

### Performance
- **The waterfall uploads only the rows that changed.** It had been rebuilding
  and re-uploading its entire texture whenever one new row arrived — at 1024
  bars that is 1024×120 pixels re-palettised and pushed to the GPU, at the
  frame rate, to change 1024 of them. Now a ring buffer with partial uploads,
  which took waterfall mode from about 20% CPU to about 10% at 180 fps.

### Fixes
- The lyrics scrollbar sat stranded in the middle of the window, against the
  longest line rather than the window edge.
- The font picker never applied anything. A dropdown nested inside a menu opens
  its own overlay, so clicking it counts as clicking *outside* the menu — the
  menu closed and swallowed the click before it reached anything.
- A NetEase track id could appear in the lyrics pane where the song should be.

## [1.3.0] - 2026-08-13

Two things the spectrum could not do before: analyse the waveform directly
with wavelets instead of mapping FFT bins, and play native DSD on Linux.
Plus lyrics and tag editing, so the player stops needing a second app.

### Lyrics
- **New window (🎤 Lyrics).** Its own viewport, so it can sit on a second
  screen while the player stays where it is. The current line highlights and
  centres itself, sung lines dim, and there is a text-size slider — a lyrics
  view on another monitor is read from across the room.
- **Reads a `.lrc` beside the track, or the file's own tags.** Sheets are
  written as sidecars, never into the audio file. That is what every other
  player reads, so a fix here is a fix everywhere; it is hand-editable; and it
  keeps DSD safe, where the ID3 blob sits behind a header pointer.
- **"Find lyrics" looks the track up on LRCLIB, then NetEase Cloud Music** —
  neither needs an account or an API key, and both serve *synced* lyrics rather
  than plain text.
- **Two sources because one was not enough.** Against a real set of thirteen
  failing tracks, LRCLIB returned *zero* hits for nine of them — a Japanese and
  Vocaloid library is mostly absent from it, and no amount of better matching
  invents a record that is not there. NetEase has synced lyrics for effectively
  all of them.
- Genius and UtaTen were considered and rejected: neither publishes lyrics
  through an API (Genius's deliberately excludes them), so the only route is
  scraping HTML that breaks on any redesign — and neither carries timestamps,
  so even a successful scrape gives an unsynced sheet that still needs timing
  by hand. The paste box already covers that, without a scraper to maintain.
- **A source that cannot answer never reports "no lyrics".** NetEase rate-limits
  by replying HTTP 200 with the real status buried in the body, which reads as
  an empty result to anything checking only the transport. Telling someone their
  track has no lyrics, when the truth was to wait a minute, would send them off
  to transcribe by hand for nothing.
- **The matching is the part that matters.** Fetching is easy; finding the
  right record is where a paid lyrics plugin fails on a well-known song. Tags
  carry decoration the database does not — `【初音ミク】タイトル【オリジナルMV】`,
  a producer where the database has the vocalist, full-width punctuation,
  `feat.` chains. So the lookup is a ladder: exact match, exact without the
  album, structured search on normalised fields, then normalised title alone,
  with duration filtering and ranking at every rung because duration is the one
  field decoration cannot corrupt.
- **"Search…" is the manual override** — type anything, see what came back
  with synced-versus-plain marked, pick one. No automatic match is right every
  time, and being unable to correct it by hand is what makes a lyrics feature
  feel broken.
- **"Sync…" times a sheet by tapping along.** Play the song and press Space as
  each line starts; Back undoes a stamp, a timestamp click seeks the player to
  it, lines can be edited and reordered. It re-times an existing sheet as
  happily as a plain one, and resumes at the first untimed line so an
  interrupted sync picks up where it stopped. This is what makes the plain-text
  fallback worth having — a sheet the database only has unsynced is a few
  minutes of tapping from a proper one.
- An offset control nudges timing on a sheet that is close but not aligned.

### Tag editing
- **New window (🏷 Tags):** the common fields, plus a viewer for every tag the
  file carries — including keys the editor does not touch, which are preserved
  on save.
- **Writes go to a copy which then replaces the original by rename**, so an
  interrupted write cannot leave a half-rewritten file. The result is read back
  and compared before it is accepted: a tag that silently did not take is worse
  than an error, because the user believes it worked.
- **DSD is read-only here** and says so. DSF stores its ID3 blob behind a
  header pointer, and the player reaches it with a read-only trick; writing
  through that path would cost the audio rather than the metadata.
- Blanking a field removes the key rather than storing an empty one, which is
  not the same thing to other readers.

### Adaptive Superlet Transform — a spectrum that isn't an FFT (pre-process only)
- **New bar mapping: Superlet.** Instead of mapping FFT bins onto bars, it
  analyses the waveform directly with sets of Morlet wavelets and takes the
  geometric mean of their responses (Moca et al., *Nature Communications*
  2021). An FFT uses one window width for the whole spectrum; a log bar grid
  wants a different width at every frequency, and that mismatch is what this
  fixes. Pre-process only — the live view has about 2 ms per frame and a
  superlet needs orders of magnitude more, so real-time keeps its FFT.
- **What it actually buys, measured against the 171 ms FFT it replaces:**
  - *Bass* — resolves detail no FFT setting can reach. At 60 Hz it separates
    two tones 1.5 Hz apart that the FFT renders as one blob (σ 0.33 Hz vs
    3.58 Hz). The price is real and stated plainly: constant-Q at the bottom
    of hearing needs windows measured in seconds, so bass responds more
    slowly. That trade is a slider, not a hidden curve.
  - *Treble* — strictly better at no cost: 24 ms of smearing at 10 kHz
    against the FFT's 171 ms, losing no frequency detail 1024 bars can draw.
  - *Roughly 500 Hz – 2 kHz* — a wash. An FFT is genuinely hard to beat
    there and the presets don't pretend otherwise.
- **Five presets plus Custom.** Presets are stated as window budgets rather
  than abstract orders; Custom exposes window, sharpness (relative to the bar
  grid), wavelet count and spread, with a live readout comparing each setting
  against the FFT. Analysis costs roughly 1–17 minutes per five-minute track.
- **Long wavelets are convolved in the frequency domain** — overlap-save,
  4–5× faster, and numerically identical to the direct route (worst measured
  difference 2×10⁻⁷ end-to-end). Frames at the very start and end of a track
  stay on the direct path, because their windows are renormalised over the
  samples that exist and a convolution cannot express that.
- **Frame rate is now stated in time, not samples** — 180 fps by default.
  Besides matching high-refresh displays, this stops cost growing with the
  square of the sample rate, which had made a 176.4 kHz DSD analysis cost 16×
  a 44.1 kHz one instead of 4×.
- **Progress, ETA and Abort.** Analysis reports three named stages
  (Decoding, Loudness/key, Transform) with a time estimate that starts from a
  throughput model and switches to the observed rate once the work is
  underway. Abort stops within a second and never writes a partial cache.
- **A repeat no longer restarts the analysis.** Looping a track used to
  cancel and relaunch on every pass, so a run lasting longer than the track
  could never finish. A genuine track change still cancels.

### Superlet: sharpness comes from the bar grid, and nothing else
- **"Follow vibrato" is gone.** It shortened the window in bands whose pitch
  was moving, on the reasoning that a sweep already sets the width there so
  extra sharpness is wasted. The reasoning holds; the rendering does not. A
  wavelet is normalised so a *tone* reads the same at any bandwidth, which
  means broadband content in the same band reads proportional to the square
  root of that bandwidth. Give one band a different sharpness from its
  neighbours and it draws as a bright stripe across the waterfall — up to
  9.5 dB on real material, where the estimator also fired on far more bands
  than intended.
- No per-band gain can correct it: the correction a tone needs is 0 dB and
  the correction noise needs is not. The only sharpness reduction small
  enough to hide the step is too small to save any time, so the feature has
  no working setting and was removed rather than defaulted off. `aslt.rs`
  keeps the measurement as a test, so anything that reintroduces per-band
  sharpness variation has to answer it first.

### Cache management
- **Size budget with LRU eviction** (4 GB by default, adjustable, 0 disables)
  runs after each analysis, with a "Trim now" button and an over-budget
  warning. Superlet caches run 45–80 MB per track and no amount of packing
  changes that — the low byte of every stored value is quantisation noise, so
  no lossless coder beats about 12 % and even truncating to a depth finer
  than a 4K pixel only reaches 14 %. Bounding the total is the real control.
- **Cache format v3** takes that 12 % anyway: values are delta-coded across
  frequency and split into byte planes before LZ4, so the compressor is no
  longer defeated by noise sitting between every pair of compressible bytes.
  Existing v2 caches still load.

### Native DSD via ALSA — Linux joins the DSD512 party (experimental, on by default)
- **Raw native DSD on Linux** — pick a direct `hw:` device under the 🔈
  menu's new "Native DSD (ALSA)" section and DSD plays through the kernel's
  native DSD formats (`DSD_U32_BE`/`U16_BE`/`U8`, plus the `_LE` layouts),
  no DoP carrier and no carrier-rate ceiling — the same transport MPD and
  HQPlayer use. Format and rate are negotiated with the driver directly
  (resampling explicitly disabled, exact-rate or bust), so DSD512 — and
  DSD1024 — work wherever the card's driver does.
- **Same architecture as the Windows ASIO path, same behavior** — the two
  backends share the ring-feed loop and the session model: pause holds the
  DAC on DSD-marked silence (no lock drop), seeks are sample-accurate,
  position comes from byte-frames actually delivered, automatic fallback
  ordering stays native → DoP → decimated PCM, and every negotiation step
  logs to the console (`[alsa-dsd]` prefix) so driver quirks are diagnosable
  from a terminal.
- **Costs nothing to carry** — cpal already links alsa-lib into every Linux
  build, so the feature adds no new dependency; like the ASIO path it's
  inert until a device is explicitly selected, and `--no-default-features`
  removes it entirely.
- **Exercised end-to-end in tests** (negotiation, the writer thread,
  position accounting and finish detection all run against ALSA's `null`
  device in CI), but not yet validated on a physical DAC — reports welcome.
  Note for PipeWire/PulseAudio systems: native DSD needs exclusive `hw:`
  access, so the sound server must release the card first.

## [1.2.1] - 2026-07-16

Native DSD output via ASIO — hardware-validated, driving the DAC's own
driver directly for the DSD512 headroom DoP's carrier rate can't reach.
On by default so more real-world drivers get exercised.

### Native DSD via ASIO (experimental, on by default)
- **Raw native DSD to ASIO drivers** — a new output path hands the
  untouched 1-bit stream straight to the DAC vendor's ASIO driver
  (`kAsioSetIoFormat` DSD mode), with no DoP carrier and therefore no
  carrier-rate ceiling: **DSD512 plays bit-perfect** wherever the driver
  supports it. Selected per-driver in the 🔈 menu ("Native DSD (ASIO)"),
  persisted, with automatic fallback ordering: native → DoP → decimated PCM.
- **No Steinberg SDK needed** — the driver interface is declared by hand for
  x86_64 (where the ASIO thiscall ABI quirk doesn't exist), so the feature
  builds with stock Rust.
- **On by default** (still Windows x86_64 only, still gated to
  `cfg(windows)` everywhere it's used — a no-op on Linux/macOS regardless).
  With only one DAC hardware-validated so far, the goal is more real-world
  drivers exercising the path so undiscovered quirks surface and get fixed
  faster. Still inert until a user explicitly picks a driver in the 🔈
  menu — nothing changes for anyone who doesn't. Build with
  `--no-default-features` for a binary with no ASIO/COM code at all.
- Pause holds the DAC on DSD-marked silence (no lock drop), seeks are
  sample-accurate, position comes from the same frame-counting the other
  bit-perfect paths use, and the driver is released cleanly when a PCM
  track needs the device back (and vice versa — WASAPI exclusive is
  released before ASIO opens).
- **Verified on real hardware** (SMSL C200Pro, USB DAC ASIO driver): the DAC
  locks into native DSD mode with clean audio, play/pause/seek all correct.
  ASIO driver behavior varies by vendor, so other DACs/drivers may need
  further quirk-fixing the first time they're tried — one session per file
  for now (no native gapless yet), stereo-focused. Status line reads e.g.
  *💎 DSD512 native (22.5792 MHz) · 2ch → ASIO: <driver>*.
- **DSD1024 (and beyond) should work too, untested** — nothing in the native
  path is hardcoded to DSD64/128/256/512: the rate comes straight from the
  file header and is negotiated with the driver's own `canSampleRate`/
  `setSampleRate`, rate labels are computed from a formula (not a lookup
  table), and the buffer/ring math has no rate ceiling. It should play
  bit-perfect on any driver that accepts the rate in DSD mode — just not
  hardware-validated yet, since DSD1024 gear hasn't been tested against it.

### Fixes
- **Launch-time "device is no longer available" console error** — the rodio
  output stream opened eagerly at startup, and the background device scan's
  WASAPI *exclusive-mode* format probing invalidated it, tripping its error
  callback. The stream now opens lazily on first normal-mode playback (and
  is rebuilt after any exclusive session), so nothing holds the device while
  the scan runs.
- **Native ASIO silently losing to DoP** — `start_asio_native` never released
  the WASAPI-exclusive stream, so the DAC's own ASIO driver couldn't open
  hardware the DoP path still held, failed, and the error was swallowed when
  DoP succeeded. The exclusive stream is now released before ASIO opens
  (mirroring the existing ASIO-before-WASAPI release), the native failure is
  always logged to the console, drivers get a real window handle at init
  (ASIO4ALL refuses a null one), an extra rate convention (bit rate ÷16) is
  probed, and every negotiation step logs its result so driver quirks are
  diagnosable from the console.
- **`kAsioCanDoIoFormat`/`kAsioSetIoFormat` misreported as failing** —
  `ASIOFuture` signals success with a dedicated code (`0x3f4847a0`,
  distinct from the ordinary `ASE_OK`), which was being read as an error and
  made capable drivers look like they didn't support DSD mode.
- **Crash right after `start()` on the first real driver test** — ASIO
  expresses DSD buffer size in 1-bit *samples*, not bytes, so an Int8 buffer
  actually holds `bufferSize ÷ 8` bytes per channel; the fill callback was
  writing the full sample count as bytes, overrunning the driver's buffer
  allocation by 8× and corrupting its heap on the very first callback.
  Diagnosed from the driver's own (adjacent, 8×-too-close) buffer pointers
  in the debug log; every negotiation and callback step now logs to the
  console (`[asio-dsd]` prefix) to make future driver quirks this
  traceable.

## [1.2.0] - 2026-07-15

DSD support. `.dsf`/`.dff` files play bit-perfect over DoP (DSD64/128/256),
fall back to decimated PCM on devices that can't take the DoP carrier rate,
get full spectrum/loudness analysis via a separate decimated feed, and play
gapless with the rest of an album. Verified end-to-end on real DSD hardware.
DSD512 (native ASIO) remains a future stretch goal — DoP tops out at DSD256
on any device with a sub-1.4 MHz PCM ceiling.

### DSD hardening, round 2 — session-routing correctness
A second pass, prompted by the first: every place that had gated behavior on
the *global* 💎 toggle instead of what the current session is actually doing
was the same bug shape. Added `Engine::on_bp_stream()` as the single source
of truth (true only for a real bit-perfect device session — PCM bit-perfect
or DSD via DoP; false for DSD fallback even with the toggle on) and routed
everything through it:
- **ReplayGain silently inert on DSD fallback** — with 💎 on globally, RG
  computed unity gain even for a DSD track playing decimated PCM on the
  ordinary sink, where a gain multiply is completely safe. Loudness
  normalization now works on fallback DSD exactly like any PCM track.
- **EQ overlay showed the wrong spectrum** — the "is EQ actually in the audio
  path" flag driving the plotted-bars EQ overlay was only updated on an
  explicit toggle click, never per-frame, so it could show unmodified bars
  during real DoP EQ-bypass or (worse) EQ-modified bars during fallback
  where EQ genuinely is applied. Now synced every frame from the actual
  session.
- **Gapless could freeze the UI while audio kept playing** — with 💎 on
  globally, a DSD-fallback session's queued next track (appended to the
  rodio sink, same as normal gapless) had no boundary signal reaching the
  UI: the frame-exact bp path never fires for fallback, and the time-based
  fallback path was excluded by the toggle. Title/position would stick on
  the old track while rodio silently moved on to the next. Fixed at the
  same time as the equivalent `flush_gapless` case (a stale next-track
  reference could survive an A-B-repeat/seek during fallback).

### DSD hardening (integrity audit + fallback)
- **Automatic PCM fallback** — when a device can't take the DoP carrier rate
  (laptop outputs, budget interfaces), the DSD bitstream is decimated to
  high-rate PCM and played through the normal path: volume, EQ and ReplayGain
  apply, the realtime spectrum works, seeks are instant, and consecutive DSD
  tracks still play gapless. The status line says why and at what rate. DSD
  files are never simply unplayable.
- **DAC unlock tail** — a DoP session now also *ends* with DSD-marked silence
  (mirroring the warm-up), so the DAC never loses DSD lock mid-drain on plain
  zeros at the end of the last track.
- Audit fixes: the time-based gapless rollover could race the frame-exact DoP
  boundary (early title/position switch at a DSD track seam); picking a new
  output device mid-DSD-track did nothing until the next track (DSD uses the
  device regardless of the 💎 toggle); with 💎 on, a fallback session would
  have routed pause/position/finish to the idle device stream — session
  routing now has a single source of truth.

### DSD playback polish
- **Gapless DSD** — consecutive DSD tracks at the same rate now hand off with
  no device re-open and no gap, the way an album (SACD rip, live set) should
  play. The DoP marker phase is carried across the file boundary so the
  0x05/0xFA alternation stays intact — the DAC never sees a glitch at the
  seam. Reuses the same frame-exact boundary machinery as PCM gapless.
- **DAC warm-up** — a DoP session now leads with ~24 ms of DoP-marked silence
  so the DAC locks into DSD mode before the first audio sample, guarding
  against a start-of-track transient. (Only at session start, never between
  gapless tracks.)
- **Clean DSD↔PCM transitions** — switching between a DSD and a PCM track
  always re-opens the device with the right format; fixed a latent case where
  a PCM track at a DoP carrier rate (e.g. 176.4 kHz after DSD64) could have
  wrongly reused the DoP stream, leaving volume and the analyzer bypassed.

### DSD analysis
- **Spectrum, waveform, LUFS, DR and spectral ceiling now work for DSD** —
  a two-stage decimator (byte-table FIR over the raw bits ÷8, then half-band
  ÷2 cascades) renders the audible band as PCM for the analyzer while
  stripping the ultrasonic modulator noise that isn't music. Playback is
  untouched — the DAC still gets the raw bits over DoP.
- **Selectable analysis rate** — 176.4 kHz default (every DSD rate divides
  into it exactly), or 352.8 kHz for a faster-reacting spectrum: twice the
  frame rate at the same FFT size, at the cost of a larger cache and longer
  analysis. Picked in ⚙ FFT Settings (the row appears when a DSD track is
  loaded), persisted, and part of the cache key — caches for both rates can
  coexist, and existing PCM cache files stay valid untouched.
- 48k-family DSD (2.8 MHz × 48k multiples) lands on its own exact
  sub-multiples (192/384 kHz) instead of resampling.
- During DoP playback the display is driven purely by the pre-processed
  analysis (there is no PCM signal to tap in real time — the stream is DoP
  words); momentary LUFS and the correlation meter stay dark. (Fallback
  playback is ordinary decimated PCM, so all of this — including realtime
  mode — works exactly like a normal track.)
- **"Analyzing… %" is now visible on the plot itself** — a DSD track's first
  analysis used to look like a dead spectrum (nothing to show until the
  background pass lands, and the only progress readout hid inside the 🗄
  Cache chip). The plot now overlays "Analyzing DSD… N%" with a progress
  bar whenever it would otherwise be blank, Real-time mode on a DSD track
  explains itself instead of staying silently empty, and PCM tracks now get
  the live-FFT fallback while their first analysis runs (the hybrid the
  README always promised) instead of a blank panel.

### DSD playback
- **DSD64/128/256 play back bit-perfect via DoP** — `.dsf`/`.dff` files open
  the bit-perfect output at the DoP carrier rate (176.4 / 352.8 / 705.6 kHz)
  in a 24-bit-or-wider integer device format; the status line reads e.g.
  "💎 DSD128 via DoP · 352.8 kHz · 2ch · 24i excl → device". Play/pause,
  seek, and stop all work; sample-accurate position comes from the same
  frame-counting the PCM bit-perfect path already used.
- **Hard bit-perfect, no exceptions** — unlike PCM bit-perfect (which still
  applies volume below 100%, with a warning), DSD volume and EQ are fully
  bypassed: any scaling would corrupt the packed marker bits, not just
  reduce precision. The 💎 toggle is disabled while a DSD track plays (DSD
  is always bit-perfect via DoP, independent of that PCM-only setting).
- **`.dsf` / `.dff` recognized** — both DSD containers parse natively (Sony
  DSF and Philips DSDIFF up to 8 channels; DST-compressed DSDIFF is rejected
  with a clear message). Files can be added via the file picker and folder
  scan.
- **Full metadata** — stream properties (rate, channels, duration, throughput)
  come from the DSD header; titles, ReplayGain and embedded cover art come from
  the containers' ID3v2 tags through the same pipeline as every other format.
- **DSD-aware Info window** — sample rate shown as "DSD64 — 2.8224 MHz" style,
  1-bit depth labelled, throughput and uncompressed-size math correct for
  1-bit streams.
- Gapless is never used across a DSD boundary in either direction — a DoP
  session is always exactly one file; the next track starts fresh instead.
- Spectrum analysis for DSD tracks isn't computed yet (the decimated PCM
  feed lands in a later phase) — the Info window says so plainly instead of
  spinning forever.

### Internal
- **Normalised bitstream reader** — streams either container as MSB-first
  channel-interleaved bytes with seek support, the shared feed for the DoP
  packer and the future analyzer decimator.
- **DoP encoder** — packs the DSD bitstream into 24-bit DoP words (alternating
  `0x05`/`0xFA` markers, 16 DSD bits per sample, carrier = DSD rate ÷ 16) with
  correct marker phase across reads/seeks, and DSD-silence (`0x69`) padding
  for stream tails.
- Device format negotiation (WASAPI exclusive and cpal) is restricted to
  integer PCM only in DoP mode — never float, never 16-bit — so the DAC
  always sees the literal packed bit pattern.
- All new code covered by unit tests, including a proof that the DoP word
  encoding round-trips bit-exactly through the existing power-of-two device
  writers, which is what let the whole bit-perfect ring-buffer/backend
  pipeline be reused unchanged for DSD.

## [1.1.2] - 2026-07-14

Daily-use conveniences and per-track history.

### Player
- **Sleep timer** (💤) — stop playback after 15 / 30 / 45 / 60 / 90 minutes or
  at the end of the current track; the button lights up and its tooltip shows
  the time remaining.
- **A-B repeat** — cycle A → B → off to loop a section of the current track,
  with A/B markers drawn on the seek bar. Gapless prebuffering and auto-advance
  are suppressed while a loop is active.
- **Bookmarks** (🔖) — save the current position per track and jump back to it
  later; persisted to `~/.moosik/bookmarks.json`.
- **Play statistics** — per-track play count and last-played time, persisted to
  `~/.moosik/stats.json`. Shown in the Info window, as an optional **▶N column**
  in the playlist (toggle in the 🎨 Look menu), and as two new sort columns —
  **Plays** (most-played first) and **Recent** (most-recently-played first).

## [1.1.1] - 2026-07-13

Follow-up polish to the 1.1.0 appearance work: a tidier spectrum window, a lot
more persistence, a light theme, and playlist search + sorting.

### Spectrum window
- **Palette moved into the window** — the palette picker now lives on the
  spectrum window's own control row (next to Mode / View / Loudness) instead of
  the main Look menu, and each option shows a gradient **swatch** previewing its
  low→high ramp (the Album-accent palette previews the current cover tint).
- **Tidier controls, bigger plot** — the FFT Settings, Peak Hold, and Album Art
  sections collapsed from three stacked headers into a single wrap-around row of
  toggle chips; each body renders full-width only when open, and the row wraps to
  more lines only when the window is too narrow. Bar gap folded onto the bars
  row, and the pre-process cache controls (Clear Cache / Re-analyze / Cache size
  / Clear All) moved into a new 🗄 Cache chip. The "no cache" warning and the
  cache-hit highlighting stay live.
- **View settings persist** — mode, style, loudness, bar count/gap, window,
  smoothing, frequency range, interpolation, padding, overlap, bar mapping, the
  section chip states, and the full **Peak Hold** config now restore across
  launches (`~/.moosik/spectrum.json`). FFT size is intentionally excluded — it
  auto-scales per track.

### Appearance
- **Light theme** — a Dark / Light toggle in the 🎨 Look menu (persisted). The
  whole UI resolves per theme, including the hand-painted chrome (playlist rows,
  seek bar, waveform, now-playing, status text) and every custom-coloured label
  (section chips, cache / album-art text, the bit-perfect / ReplayGain / EQ
  status colours) — so text is dark-on-light and light-on-dark instead of the
  old low-contrast greys and pale mint on white. The accent deepens on light and
  the cover-derived per-track accent is dimmed to read against the pale chrome.

### Player
- **Playlist search / filter** — a 🔍 box filters the playlist by title, artist,
  or album as you type (**Ctrl+F** to focus, **Esc** to clear); play / select /
  current-track stay correct, and drag-reorder is disabled while a filter is
  active.
- **Column sorting** — sort the playlist by Title / Artist / Album / Time; click
  a column to sort, click again to reverse (▲/▼). The playing track is followed
  to its new position; manual drag-reorder still builds a custom order.
- **Volume & loop mode persist** — both now restore on launch
  (`~/.moosik/player.json`) instead of resetting to 80% / Repeat All.

## [1.1.0] - 2026-07-12

A visual release: a cohesive dark theme drawn from the app's own identity
colors, per-track accents pulled from cover art, and a set of optional,
persisted appearance controls. Everything here defaults to the shipped look, so
nothing changes until you opt in.

### Theme
- **Cohesive dark theme** — built from the icon's own palette (the **Eigengrau** #16161d base and **Kugelblitz** #94b1ff accent): a short blue-tinted neutral ramp for panels/surfaces/widgets, gently rounded corners, and accent-coloured selection and hover. Replaces egui's flat default grey with a calmer, on-brand look.
- **One palette everywhere** — the theme now also drives the custom-painted surfaces that sit outside egui's widget system (seek bar, playlist rows, status text), so the whole window reads as one design instead of themed widgets over grey chrome.
- **Per-track UI accent** — the now-playing bar, current-row marker, and seek fill pick up an accent sampled from the current track's cover art (weighted toward vivid, saturated colours and blended toward the brand blue so it never clashes). Art-less or grayscale covers fall back to the pure brand accent.

### Appearance (new — optional & persisted)
A new **🎨 Look** menu in the top bar, saved to `~/.moosik/appearance.json`. Defaults reproduce the original appearance exactly.
- **Spectrum palette** — the visualiser's colour ramp is now selectable: **Classic** (the original rainbow), **Kugelblitz**, **Ice**, **Magma**, **Aurora**, **Mono**, and **Album accent** (a ramp built from the current cover art). Applied consistently across bars, octave bands, line, filled, spectrogram, and waterfall.
- **Text size** — a UI-scale slider (70–160%) that scales all text and chrome via egui's zoom factor. Committed on an explicit **Apply** (with **Reset** to 100%), so dragging the slider doesn't live-resize the menu under the cursor.
- **Accent source** — toggle between the album-art accent and the fixed brand accent for the chrome.

### Spectrum
- **Max FPS defaults to the monitor's refresh rate** — read from the primary display at startup (Windows), clamped to the 1–240 range, falling back to 60 when it can't be determined. Previously a fixed 60 that had to be raised by hand every launch on a high-refresh display.

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

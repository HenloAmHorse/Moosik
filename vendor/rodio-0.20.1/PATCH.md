# Moosik rodio 0.20.1 patch

Starting point: the unmodified published crates.io `rodio-0.20.1.crate`, extracted
from Cargo's verified registry archive. Original MIT and Apache-2.0 license texts
and source are retained. This patch does not upgrade rodio or change its existing
Symphonia `SampleBuffer<i16>` conversion policy.

Archive SHA-256:
`e7ceb6607dd738c99bc8cb28eff249b7cd5c8ec88b9db96c0608c1480d140fb1`.

Runtime changes: `src/decoder/symphonia.rs`, plus the `ReadSeekSource` wrapper
and its two constructors in `decoder/mod.rs`:

* Discover the byte length while restoring the caller's cursor. Some container
  readers require a known size; the old seekable wrapper always returned None.
  Failure to restore the cursor is an error, never an unnoticed EOF position.
* Propagate container positioning errors at open instead of panicking on the
  assumption that a format probe cannot return a seek error.

* Select a registered audio codec, preferring the container default when it is
  supported, and retain that track ID throughout playback and seeking.
* Derive duration from the selected track and convert fractional seconds normally
  instead of taking their reciprocal.
* Continue across other tracks' packets and fully trimmed empty audio packets.
  Only consecutive recoverable decode errors consume the bounded retry budget.
* After seeking, reset the decoder and decode preceding packets for codec history
  instead of skipping their compressed data. Preserve demuxer-supplied pre-roll.
* Convert seek offsets through the track's actual `TimeBase`, rather than treating
  Matroska millisecond timestamps as PCM sample counts.
* When an Opus reader seeks from the known beginning, count decoded audible frames
  instead of accumulating errors from rounded container timestamps. This is an
  exact-output, bounded-memory fallback, not an instantaneous random-access claim;
  long seeks may decode the preceding audio again.
* Correct the existing just-before-end helper to saturate at time zero and keep
  the fraction in its valid range when borrowing across an integer second.

The demuxer is still responsible for providing enough pre-roll and correct packet
trims/timestamps. This consumer cannot invent correct history when a container
reader begins too late. No audio device or hardware test is implied by unit tests.

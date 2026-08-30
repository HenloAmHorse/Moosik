// ---------------------------------------------------------------------------
// Bit-perfect output path
//
// Architecture:
//   decode thread (symphonia, full sample precision, real seeking)
//        │  canonical u32 payloads — left-aligned integers or raw f32 bits,
//        │  exact for every source family the policy admits (see `format`)
//        ▼
//   frame-atomic SPSC ring buffer (rtrb via `frame_ring`, ~1 s)
//        │  whole interleaved frames only, in and out
//        ▼
//   platform backend — pops whole frames, writes the canonical payload to the
//   device's negotiated format with shifts and byte copies (no float multiply,
//   no narrowing), taps the spectrum buffers, counts frames for
//   sample-accurate position:
//     • non-Windows: cpal stream at the exact rate (ALSA hw:/PipeWire).
//     • Windows: WASAPI *exclusive* mode via the wasapi crate — shared mode
//       only ever accepts the mixer's configured rate, so true native-rate
//       output requires exclusive access (same as foobar2000's WASAPI
//       exclusive output). Polling mode is used because event-driven
//       exclusive mode is known to stutter with USB audio class drivers.
//
// The realtime render side never decodes, never allocates on the
// steady-state path, and never blocks: the only lock is a try_lock on the
// session slot, contended only for the instant a track change / seek swaps it.
// ---------------------------------------------------------------------------

pub mod format;
pub mod frame_ring;
pub mod q31;
pub mod state;

#[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
pub mod alsa_dsd;
#[cfg(all(windows, feature = "asio-dsd"))]
pub mod asio_dsd;
#[cfg(not(windows))]
mod cpal_out;
#[cfg(windows)]
mod wasapi_out;

/// True when a native-DSD backend (ASIO on Windows, ALSA on Linux) is
/// compiled in — the gate for the shared ring-feed loop below.
#[cfg(any(
    all(windows, feature = "asio-dsd"),
    all(target_os = "linux", feature = "alsa-dsd"),
))]
pub mod native_dsd {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Duration;

    use super::fault;
    use crate::dsd::dop::DSD_SILENCE;

    /// The DSD idle byte in the driver's own bit order.
    ///
    /// `0x69` is DSD silence with the oldest sample in the most significant
    /// bit. A driver that wants the oldest sample in the LSB needs the same
    /// pattern reversed, which is `0x96` — a different byte. Lead-in, tail and
    /// every gap used a fixed `0x69`, so on an LSB-first driver the silence
    /// Moosik generated was not silence.
    #[inline]
    pub const fn idle_byte(lsb_first: bool) -> u8 {
        if lsb_first {
            DSD_SILENCE.reverse_bits()
        } else {
            DSD_SILENCE
        }
    }

    /// What the feeder is doing, readable from the UI thread.
    ///
    /// The feeder used to return `()`, and every way it could go wrong —
    /// a read error, a partial-frame push, a panic — left the loop by a path
    /// that either set `done` (indistinguishable from a clean end of track) or
    /// set nothing at all (a session priming forever in silence). Both kept
    /// the exact claim.
    ///
    /// Faults are stamped with the generation they were raised in, so a feeder
    /// still unwinding after its session was replaced cannot fault the track
    /// that replaced it.
    #[derive(Debug)]
    pub struct FeedStatus {
        /// The fault and the session it belongs to, in one word.
        ///
        /// These were `code` and `fault_gen`, written in sequence and read in
        /// sequence, which is the torn pair the rest of this module spent
        /// three commits removing: a reader that caught the new code beside
        /// the old generation discarded a live fault, and one that caught the
        /// old code beside the new generation reported a fault the current
        /// feed never had. Two atomics cannot be read together however they
        /// are ordered.
        fault: AtomicU64,
        generation: AtomicU64,
    }

    impl Default for FeedStatus {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FeedStatus {
        pub fn new() -> Self {
            FeedStatus {
                fault: AtomicU64::new(super::stamped::pack(0, fault::NONE)),
                generation: AtomicU64::new(1),
            }
        }

        /// Open a new logical native-DSD session and return its generation.
        /// Retires every fault older than it.
        pub fn begin_generation(&self) -> u64 {
            self.generation.fetch_add(1, Ordering::AcqRel) + 1
        }

        pub fn generation(&self) -> u64 {
            self.generation.load(Ordering::Acquire)
        }

        /// Record `code` against `at`, and only `at`. First one wins within a
        /// session; a record from a superseded one is stale and is replaced.
        pub fn fault_in(&self, at: u64, code: u8) {
            if at != self.generation() {
                return;
            }
            let want = super::stamped::pack(at, code);
            let mut cur = self.fault.load(Ordering::Acquire);
            loop {
                // Monotone in the generation: a feeder still unwinding after
                // its session was replaced must not stamp its own, older
                // generation over the successor's live fault.
                if super::stamped::generation(cur) > at {
                    return;
                }
                if super::stamped::generation(cur) == at
                    && super::stamped::code(cur) != fault::NONE
                {
                    return;
                }
                match self.fault.compare_exchange_weak(
                    cur,
                    want,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return,
                    Err(observed) => cur = observed,
                }
            }
        }

        /// The fault of the session now playing, or `fault::NONE`.
        pub fn fault(&self) -> u8 {
            let v = self.fault.load(Ordering::Acquire);
            if super::stamped::generation(v) == self.generation() {
                super::stamped::code(v)
            } else {
                fault::NONE
            }
        }
    }

    /// Turns a feeder that ended without finishing — including by panic — into
    /// a fault against its own generation.
    struct FeedGuard {
        status: Arc<FeedStatus>,
        generation: u64,
        done: Arc<AtomicBool>,
        clean: bool,
    }

    impl Drop for FeedGuard {
        fn drop(&mut self) {
            if !self.clean {
                self.status
                    .fault_in(self.generation, fault::SESSION_THREAD_FAILED);
            }
            // Whatever happened, release the consumer. A native session whose
            // feeder is gone will never produce another byte, and something
            // has to stop the engine waiting for one.
            self.done.store(true, Ordering::Release);
        }
    }

    /// Why a native-DSD feed stopped early.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FeedFault {
        /// The file could not be read to its declared end.
        SourceRead,
        /// The container declared more audio than the file contains.
        Truncated,
        /// The ring rejected a partial frame — every channel after it would be
        /// rotated and no reader could tell which byte belonged where.
        RingInvariant,
    }

    impl FeedFault {
        pub fn code(self) -> u8 {
            match self {
                FeedFault::SourceRead => fault::SOURCE_READ,
                FeedFault::Truncated => fault::SOURCE_READ,
                FeedFault::RingInvariant => fault::RING_INVARIANT,
            }
        }
    }

    /// Stream a DSD file's bytes into a native-DSD ring: ~24 ms of DSD
    /// silence first (DAC settle), then the audio (bit-reversed per byte if
    /// the device wants oldest-sample-in-LSB; the reader's native order is
    /// MSB-first), then a silence tail so the device idles at DSD zero while
    /// the ring drains. Transport-agnostic — the ASIO and ALSA backends both
    /// feed from this. Mirrors `dop_decode_loop`, minus the packing.
    ///
    /// Returns `Ok(())` only for a genuine, complete end of track. Every other
    /// outcome is a typed fault against `generation`, and none of them is a
    /// clean EOF.
    #[allow(clippy::too_many_arguments)]
    pub fn feed_loop(
        mut reader: crate::dsd::DsdFileReader,
        lsb_first: bool,
        mut prod: super::frame_ring::FrameProducer<u8>,
        done: Arc<AtomicBool>,
        // Set only on the path that reaches the end of the file with nothing
        // wrong. The backend's drain reads it to tell a track that finished
        // from a feeder that gave up — the distinction that decides whether
        // the playlist may advance.
        decode_eof: Arc<AtomicBool>,
        stop: Arc<AtomicBool>,
        status: Arc<FeedStatus>,
        generation: u64,
    ) -> Result<(), FeedFault> {
        let mut guard = FeedGuard {
            status: Arc::clone(&status),
            generation,
            done: Arc::clone(&done),
            clean: false,
        };

        let info = reader.info().clone();
        let ch = info.channels as usize;
        let lead_frames = (info.sample_rate as usize / 8 * 24 / 1000).max(64);
        let idle = idle_byte(lsb_first);

        // Whole byte-frames only, in and out. A partial frame in this ring
        // rotates every channel for the rest of the session, and the callback
        // has no way to detect which byte belongs where.
        //
        // `Interrupted` means the session was torn down and is not a fault;
        // `Torn` is one.
        enum Push {
            Ok,
            Interrupted,
            Torn,
        }
        let push_all = |bytes: &[u8],
                        prod: &mut super::frame_ring::FrameProducer<u8>,
                        stop: &AtomicBool|
         -> Push {
            let mut off = 0usize;
            while off < bytes.len() {
                if stop.load(Ordering::Relaxed) {
                    return Push::Interrupted;
                }
                match prod.push_frames(&bytes[off..]) {
                    Err(_) => return Push::Torn,
                    Ok(0) => std::thread::sleep(Duration::from_millis(5)),
                    Ok(n) => off += n * prod.channels(),
                }
            }
            Push::Ok
        };

        // A container that declares more audio than the file holds is not a
        // shorter complete file. Running off the end of one used to be
        // indistinguishable from a track ending — at a gapless boundary,
        // completely invisible.
        if info.is_truncated() {
            status.fault_in(generation, FeedFault::Truncated.code());
            guard.clean = true;
            return Err(FeedFault::Truncated);
        }

        let silence = vec![idle; lead_frames * ch];
        match push_all(&silence, &mut prod, &stop) {
            Push::Ok => {}
            Push::Interrupted => {
                guard.clean = true;
                return Ok(());
            }
            Push::Torn => {
                status.fault_in(generation, FeedFault::RingInvariant.code());
                guard.clean = true;
                return Err(FeedFault::RingInvariant);
            }
        }

        let expect_frames = info.total_frames();
        let mut got_frames: u64 = 0;
        let mut buf = vec![0u8; 4096 * ch];
        loop {
            if stop.load(Ordering::Relaxed) {
                guard.clean = true;
                return Ok(());
            }
            let n = match reader.read_frames(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    // Not an end of file. Breaking here pushed the tail and
                    // set `done`, so a mid-track I/O error arrived at the UI
                    // as a track that had simply finished.
                    crate::mlog!("[native-dsd] read error: {e}");
                    status.fault_in(generation, FeedFault::SourceRead.code());
                    guard.clean = true;
                    return Err(FeedFault::SourceRead);
                }
            };
            got_frames += n as u64;
            let chunk = &mut buf[..n * ch];
            if lsb_first {
                for b in chunk.iter_mut() {
                    *b = b.reverse_bits();
                }
            }
            match push_all(chunk, &mut prod, &stop) {
                Push::Ok => {}
                Push::Interrupted => {
                    guard.clean = true;
                    return Ok(());
                }
                Push::Torn => {
                    status.fault_in(generation, FeedFault::RingInvariant.code());
                    guard.clean = true;
                    return Err(FeedFault::RingInvariant);
                }
            }
        }

        // The reader stopped early against what the container promised.
        if got_frames < expect_frames {
            crate::mlog!(
                "[native-dsd] short read: {got_frames} of {expect_frames} declared byte-frames"
            );
            status.fault_in(generation, FeedFault::Truncated.code());
            guard.clean = true;
            return Err(FeedFault::Truncated);
        }

        if !stop.load(Ordering::Relaxed) {
            let _ = push_all(&silence, &mut prod, &stop);
        }
        // The one path that reaches here having played the whole file.
        decode_eof.store(true, Ordering::Release);
        guard.clean = true;
        Ok(())
    }
}
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder as SymDecoder, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;

use crate::dsd::dop::DopFileStream;
use crate::spectrum::{SampleBuf, StereoBuf, DEFAULT_FFT_SIZE};

use format::{
    canon_from_f32, canon_from_i16, canon_from_i24, canon_from_i32, canon_from_signed,
    canon_from_unsigned, canon_to_f32, low_bits_are_clean, ChannelLayout, PcmKind, SourceFormat,
};
use frame_ring::{channel_ring, FrameConsumer, FrameProducer};
use state::TransformDescription;
pub use state::{EndpointIdentity, OutputFormat, StreamKey, Transport};

// ---------------------------------------------------------------------------
// Realtime fault reporting
// ---------------------------------------------------------------------------

/// Integrity faults, as fixed codes a realtime thread can publish with one
/// atomic store.
///
/// A realtime callback cannot format a string, take the logging mutex or touch
/// the filesystem — 1.4.1 did all three from inside the WASAPI render loop and
/// lengthened the very stalls it was trying to report. It cannot allocate an
/// error type either. So the audio side publishes a `u8`, and a non-realtime
/// poller turns it into words.
///
/// The first fault of a session wins. A dropout followed by a write error is
/// still a session that stopped being exact at the dropout, and the first
/// cause is the one worth reading.
pub mod fault {
    pub const NONE: u8 = 0;
    pub const UNDERRUN: u8 = 1;
    /// The ring held a non-multiple of the channel count.
    ///
    /// Fatal, and it was not. A dropout is a hole in the audio and the audio
    /// resumes; a torn ring means the samples at the head can no longer be
    /// attributed to a channel, so everything after it is played on the wrong
    /// one — permanently, silently, with the track continuing. There is no
    /// resynchronisation that is not a guess, which is why nothing here
    /// attempts one, and a fault that cannot be recovered from is not a
    /// recoverable fault.
    pub const RING_INVARIANT: u8 = 2;
    pub const DECODE_ERROR: u8 = 3;
    pub const SOURCE_READ: u8 = 4;
    pub const SOURCE_FORMAT_CHANGE: u8 = 5;
    pub const BACKEND_WRITE: u8 = 6;
    pub const BACKEND_SPACE: u8 = 7;
    pub const BACKEND_RESET: u8 = 8;
    pub const CALLBACK_LOCK_MISS: u8 = 9;
    /// The output never began producing audio within its bounded window.
    pub const PRIMING_TIMEOUT: u8 = 10;
    /// A decode/feed thread stopped without finishing its track.
    pub const SESSION_THREAD_FAILED: u8 = 11;
    /// The backend thread ended unexpectedly; the stream can carry nothing more.
    pub const BACKEND_DEAD: u8 = 12;
    /// A sample outside the proven value-exact range reached the conversion.
    pub const VALUE_EXACT_VIOLATION: u8 = 13;
    /// A shared-mixer track's audio ran out well short of its declared length.
    ///
    /// A heuristic, and named as one. `rodio`'s decoder has no failure
    /// channel — it yields `None` for a corrupt frame exactly as it does for
    /// the end of the file — so this is not a report from the decoder. It is
    /// an inference from the only thing observable: how much of the track was
    /// actually played. It was previously published as `SOURCE_READ`, which
    /// claims a specific thing happened that nothing observed.
    ///
    /// Fatal, because whatever the cause, the track did not play, and a track
    /// that did not play must not advance a playlist into itself.
    pub const SHARED_ENDED_EARLY: u8 = 15;
    /// The driver reported that it missed its own callback deadline.
    ///
    /// Distinct from `UNDERRUN`, which it used to be reported as. An underrun
    /// is this process failing to keep a ring full and is survivable: the
    /// track goes on and loses its claim. A driver telling us it overran is
    /// the layer below saying it could not do its job, and on DSD — where the
    /// DAC is locked to a bitstream — there is no meaningful "carry on" after
    /// it. The two needed different codes because they now have different
    /// consequences.
    pub const BACKEND_OVERLOAD: u8 = 14;

    /// Whether a fault ends the session or only ends its *claim*.
    ///
    /// The two were one thing, and the conflation cost the listener a
    /// playlist. A dropout, a missed callback lock and an off-grid sample are
    /// losses of integrity: the route stops being what the badge says it is,
    /// the badge has to say so, and the track goes on playing — none of them
    /// is a reason to stop the music or to refuse to advance at the end of it.
    /// A decode error, a read error, a dead backend, a session that never
    /// primed **and a torn ring** are the other kind: there is no usable audio
    /// after them, so the session is over and the playlist must not treat that
    /// as a track having played.
    ///
    /// The torn ring was on the wrong side of that line, and this doc listed
    /// it there long after the code was corrected. A hole in the audio is
    /// survivable because the audio resumes; a ring holding a non-multiple of
    /// the channel count is not, because the samples at its head can no longer
    /// be attributed to a channel and everything after them plays on the wrong
    /// one — for the rest of the track, silently, with nothing to hear except
    /// a stereo image that has come apart.
    ///
    /// Before this split, one dropout anywhere in a track made `finish_clean`
    /// convert the end of that track into a failure, and a failure never
    /// advances — so a single momentary underrun silently stopped the
    /// playlist at the end of the track it happened in.
    pub const fn is_fatal(code: u8) -> bool {
        !matches!(code, NONE | UNDERRUN | CALLBACK_LOCK_MISS | VALUE_EXACT_VIOLATION)
    }

    /// The typed fault a realtime code stands for.
    ///
    /// The audio side publishes a `u8` because that is all a thread holding a
    /// hard deadline may do; this is where it becomes a value the session
    /// state understands. `None` means no fault.
    pub fn to_reason(code: u8) -> Option<super::state::FaultReason> {
        use super::state::FaultReason as R;
        Some(match code {
            UNDERRUN => R::Underrun,
            RING_INVARIANT => R::TornFrame,
            DECODE_ERROR => R::DecodeError,
            SOURCE_READ => R::SourceRead,
            SOURCE_FORMAT_CHANGE => R::SourceFormatChange,
            BACKEND_WRITE => R::BackendWrite,
            BACKEND_SPACE => R::BackendSpace,
            BACKEND_RESET => R::BackendReset,
            CALLBACK_LOCK_MISS => R::CallbackLockMiss,
            PRIMING_TIMEOUT => R::PrimingTimeout,
            SESSION_THREAD_FAILED => R::SessionThreadFailed,
            BACKEND_DEAD => R::BackendDead,
            VALUE_EXACT_VIOLATION => R::ValueExactViolation,
            BACKEND_OVERLOAD => R::BackendOverload,
            SHARED_ENDED_EARLY => R::SharedEndedEarly,
            _ => return None,
        })
    }

    /// The user-facing reason. Persistent: it ends up on the status line and
    /// stays there for the rest of the session.
    pub fn describe(code: u8) -> &'static str {
        match code {
            UNDERRUN => "a dropout — the device played silence that was not in the file",
            RING_INVARIANT => {
                "a torn frame in the output ring — the channels can no longer be told apart"
            }
            DECODE_ERROR => "a decode error mid-track",
            SOURCE_READ => "a read error mid-track",
            SOURCE_FORMAT_CHANGE => "the source changed format mid-stream",
            BACKEND_WRITE => "the audio device rejected a write",
            BACKEND_SPACE => "the audio device stopped reporting buffer space",
            BACKEND_RESET => "the driver reset the stream",
            CALLBACK_LOCK_MISS => "the render callback could not reach the session in time",
            PRIMING_TIMEOUT => "the output never started producing audio",
            SESSION_THREAD_FAILED => "the decoder thread stopped unexpectedly",
            BACKEND_DEAD => "the audio backend stopped unexpectedly",
            VALUE_EXACT_VIOLATION =>
                "a sample outside the proven value-exact range reached the output",
            BACKEND_OVERLOAD => "the driver missed its own callback deadline",
            SHARED_ENDED_EARLY =>
                "the audio ran out well before the end of the track — the file may be damaged",
            _ => "an output integrity fault",
        }
    }
}

// ---------------------------------------------------------------------------
// Device detection
// ---------------------------------------------------------------------------

/// Standard PCM rates probed against each device. 705_600 is the DoP carrier
/// for DSD256 (2.8224 MHz × 4 ÷ 16) — not a "standard" PCM rate on its own,
/// but worth showing in the device picker since a DAC accepting it can take
/// DSD256 over DoP.
pub const PROBE_RATES: [u32; 9] =
    [44_100, 48_000, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000, 705_600];

#[derive(Clone)]
pub struct DeviceCaps {
    pub name: String,
    /// Subset of PROBE_RATES the device accepts.
    /// On Windows this is probed in *exclusive* mode — the device's real
    /// capabilities, not the shared mixer's configured format.
    pub rates: Vec<u32>,
    /// Sample-format labels, e.g. "16i", "24i", "32i", "32f".
    pub formats: Vec<String>,
    pub max_channels: u16,
    pub is_default: bool,
}

impl DeviceCaps {
    pub fn supports_rate(&self, sr: u32) -> bool {
        self.rates.contains(&sr)
    }

    /// One-line capability summary for the device picker UI.
    pub fn summary(&self) -> String {
        let rates = if self.rates.is_empty() {
            "no standard rates".to_string()
        } else {
            let lo = *self.rates.first().unwrap();
            let hi = *self.rates.last().unwrap();
            if lo == hi { fmt_khz(lo) } else { format!("{}–{}", fmt_khz(lo), fmt_khz(hi)) }
        };
        format!("{} · {} · {}ch max", rates, self.formats.join("/"), self.max_channels)
    }
}

fn fmt_khz(sr: u32) -> String {
    if sr.is_multiple_of(1000) { format!("{} kHz", sr / 1000) }
    else { format!("{:.1} kHz", sr as f64 / 1000.0) }
}

/// Probe all output devices on a background thread (device enumeration can
/// block for hundreds of ms per device). The receiver yields one final Vec
/// when the scan completes.
pub fn spawn_device_scan() -> mpsc::Receiver<Vec<DeviceCaps>> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("bp-device-scan".into())
        .spawn(move || { let _ = tx.send(probe_devices()); })
        .ok();
    rx
}

/// Resolve the endpoint a fresh open would attach to, without opening it.
///
/// Every reuse decision starts here rather than from the stream already
/// running, so a Windows default-endpoint change forces a reopen instead of
/// silently continuing on the old device.
pub fn resolve_endpoint(device_name: Option<&str>) -> Option<state::EndpointIdentity> {
    #[cfg(windows)]
    { wasapi_out::resolve_endpoint(device_name) }
    #[cfg(not(windows))]
    { cpal_out::resolve_endpoint(device_name) }
}

fn probe_devices() -> Vec<DeviceCaps> {
    #[cfg(windows)]
    { wasapi_out::probe_devices() }
    #[cfg(not(windows))]
    { cpal_out::probe_devices() }
}

// ---------------------------------------------------------------------------
// Decode preparation (synchronous, so open/format errors surface immediately)
// ---------------------------------------------------------------------------

pub struct Prepared {
    format:   Box<dyn FormatReader>,
    decoder:  Box<dyn SymDecoder>,
    track_id: u32,
    pub sample_rate: u32,
    /// Source bit depth if the container declares it (e.g. 16/24 for FLAC).
    /// Only a hint — `source.kind` is what the decoder actually produced.
    pub bits_per_sample: Option<u32>,
    /// What the decoder actually produces, established by decoding rather than
    /// by trusting a header. This is what the output format is negotiated
    /// against.
    pub source: SourceFormat,
    /// The packet decoded during `prepare` to establish the family, already in
    /// canonical form. Handed to the ring ahead of everything else so the
    /// probe costs no audio.
    primed: Vec<u32>,
    /// Where the container actually seeked to, which is not always where it was
    /// asked to go. Callers move the UI and the time base to *this*, never to
    /// the requested position.
    pub seeked_to: Duration,
    /// The file this was prepared from.
    ///
    /// Carried across a gapless boundary so the successor's value-exactness
    /// can be looked up and proved under *its own* identity. Without it the
    /// boundary had no way to name the new track, so the outgoing track's
    /// verdict stayed applied to it.
    pub path: PathBuf,
}

/// Append one decoded packet to `out` in canonical form, and report the family
/// the decoder actually produced.
///
/// The family comes from the buffer symphonia hands back, not from the
/// container header. A 24-bit FLAC decodes into `S32` storage while declaring
/// 24 bits, and some WAVs declare nothing at all. Steering the output policy
/// from `bits_per_sample` alone meant `None` was treated as "24-bit or
/// unknown" — a guess, and on a 32-bit source a wrong one that cost the low
/// eight bits of every sample.
fn append_canonical(decoded: &AudioBufferRef<'_>, out: &mut Vec<u32>) -> PcmKind {
    macro_rules! interleave {
        ($buf:expr, $conv:expr, $kind:expr) => {{
            let b = $buf;
            let ch = b.spec().channels.count().max(1);
            let frames = b.frames();
            out.reserve(frames * ch);
            for fr in 0..frames {
                for c in 0..ch {
                    out.push($conv(b.chan(c)[fr]));
                }
            }
            $kind
        }};
    }
    use symphonia::core::sample::{i24, u24};
    match decoded {
        AudioBufferRef::U8(b) =>
            interleave!(b, |v: u8| canon_from_unsigned(v as u32, 8), PcmKind::Integer { valid_bits: 8 }),
        AudioBufferRef::U16(b) =>
            interleave!(b, |v: u16| canon_from_unsigned(v as u32, 16), PcmKind::Integer { valid_bits: 16 }),
        AudioBufferRef::U24(b) =>
            interleave!(b, |v: u24| canon_from_unsigned(v.0, 24), PcmKind::Integer { valid_bits: 24 }),
        AudioBufferRef::U32(b) =>
            interleave!(b, |v: u32| canon_from_unsigned(v, 32), PcmKind::Integer { valid_bits: 32 }),
        AudioBufferRef::S8(b) =>
            interleave!(b, |v: i8| canon_from_signed(v as i32, 8), PcmKind::Integer { valid_bits: 8 }),
        AudioBufferRef::S16(b) =>
            interleave!(b, |v: i16| canon_from_i16(v), PcmKind::Integer { valid_bits: 16 }),
        AudioBufferRef::S24(b) =>
            interleave!(b, |v: i24| canon_from_i24(v.0), PcmKind::Integer { valid_bits: 24 }),
        AudioBufferRef::S32(b) =>
            interleave!(b, |v: i32| canon_from_i32(v), PcmKind::Integer { valid_bits: 32 }),
        AudioBufferRef::F32(b) =>
            interleave!(b, |v: f32| canon_from_f32(v), PcmKind::Float32),
        // Narrowed on the way in, and marked as what it is. `Float64` has no
        // exact device format, so this canonical form never reaches an exact
        // route — the policy refuses to open one.
        AudioBufferRef::F64(b) =>
            interleave!(b, |v: f64| canon_from_f32(v as f32), PcmKind::Float64),
    }
}


/// Last-resort family when a seek lands past the last decodable packet: read
/// it off the codec parameters instead of failing the open.
fn kind_from_params(params: &symphonia::core::codecs::CodecParameters) -> Option<PcmKind> {
    use symphonia::core::sample::SampleFormat as SF;
    match params.sample_format {
        Some(SF::U8) | Some(SF::S8) => Some(PcmKind::Integer { valid_bits: 8 }),
        Some(SF::U16) | Some(SF::S16) => Some(PcmKind::Integer { valid_bits: 16 }),
        Some(SF::U24) | Some(SF::S24) => Some(PcmKind::Integer { valid_bits: 24 }),
        Some(SF::U32) | Some(SF::S32) => Some(PcmKind::Integer { valid_bits: 32 }),
        Some(SF::F32) => Some(PcmKind::Float32),
        Some(SF::F64) => Some(PcmKind::Float64),
        None => params.bits_per_sample.and_then(|b| {
            (1..=32).contains(&b).then_some(PcmKind::Integer { valid_bits: b as u8 })
        }),
    }
}

/// Reconcile a container's declared bit depth with the decoder's storage width.
///
/// Nothing is narrowed to `u8` before it is checked. `bits_per_sample` is a
/// `u32` straight out of a container, so casting first made 256 look like 0 and
/// 257 like 1 — a nonsense declaration became a plausible one, and the plausible
/// one then steered the output format.
fn declared_valid_bits(
    declared: Option<u32>,
    storage_bits: u8,
) -> Result<Option<u8>, state::FailureReason> {
    match declared {
        None => Ok(None),
        Some(0) => Err(state::FailureReason::SourceParse(
            "the source declares zero valid bits per sample".into(),
        )),
        Some(d) if d > storage_bits as u32 => Err(state::FailureReason::SourceParse(format!(
            "the source declares {d} valid bits but decodes into {storage_bits}-bit storage"
        ))),
        // Provably `1..=storage_bits`, and `storage_bits` is at most 32.
        Some(d) => Ok(Some(d as u8)),
    }
}

/// The precision a packet actually carries.
///
/// A declared width is believed only when the samples honour it. A source that
/// says 24 bits and puts content below the declared point has more precision
/// than it admits, and the conservative answer — the storage width — is the one
/// that can only ever demand a *wider* output container.
fn effective_precision(storage_bits: u8, declared: Option<u8>, canon: &[u32]) -> u8 {
    match declared {
        Some(d) if d < storage_bits && canon.iter().all(|&c| low_bits_are_clean(c, d)) => d,
        Some(d) if d >= storage_bits => storage_bits,
        Some(_) => storage_bits,
        None => storage_bits,
    }
}

/// Canonicalize one decoded packet, validating everything before any of it
/// becomes visible to the ring.
///
/// One implementation, used by `prepare` for the first packet and by the decode
/// loop for every packet after it. They used to disagree: `prepare` checked the
/// declared precision and later packets checked only the integer/float family,
/// so a clean priming packet could establish a 24-bit plan and the next packet
/// could put content in the low eight bits and reach a 24-bit writer intact.
///
/// `plan` is the format the device was negotiated for. `None` means this is the
/// probe packet and the plan is being established. On any failure `out` is left
/// exactly as it was found: a packet that fails validation publishes nothing.
fn canonicalize_packet(
    decoded: &AudioBufferRef<'_>,
    plan: Option<&SourceFormat>,
    declared: Option<u32>,
    out: &mut Vec<u32>,
) -> Result<SourceFormat, state::FailureReason> {
    let spec = *decoded.spec();
    let channels = spec.channels.count();
    let layout = ChannelLayout(spec.channels.bits());
    if channels == 0 {
        return Err(state::FailureReason::Decode(
            "a packet declared zero channels".into(),
        ));
    }

    let mark = out.len();
    let storage_kind = append_canonical(decoded, out);
    let published = &out[mark..];

    let fail = |e: state::FailureReason, out: &mut Vec<u32>| {
        out.truncate(mark);
        Err(e)
    };

    let storage_bits = match storage_kind {
        PcmKind::Integer { valid_bits } => valid_bits,
        PcmKind::Float32 | PcmKind::Float64 => 32,
    };

    let kind = match storage_kind {
        PcmKind::Integer { .. } => {
            let declared_bits = match declared_valid_bits(declared, storage_bits) {
                Ok(d) => d,
                Err(e) => return fail(e, out),
            };
            PcmKind::Integer {
                valid_bits: effective_precision(storage_bits, declared_bits, published),
            }
        }
        other => other,
    };

    let found = SourceFormat {
        kind,
        sample_rate: spec.rate,
        channels: channels as u16,
        layout,
    };

    if let Some(plan) = plan {
        // A stream that changes rate, layout or family mid-file cannot be
        // carried by the device that was opened for the old one, and treating
        // it as a clean end of file hid that entirely.
        if found.sample_rate != plan.sample_rate {
            return fail(
                state::FailureReason::Decode(format!(
                    "the source changed from {} to {} mid-stream",
                    fmt_khz(plan.sample_rate),
                    fmt_khz(found.sample_rate)
                )),
                out,
            );
        }
        if found.channels != plan.channels {
            return fail(
                state::FailureReason::Decode(format!(
                    "the source changed from {} to {} channels mid-stream",
                    plan.channels, found.channels
                )),
                out,
            );
        }
        if found.layout != plan.layout {
            return fail(
                state::FailureReason::Decode(format!(
                    "the source changed channel layout mid-stream ({} to {})",
                    plan.layout, found.layout
                )),
                out,
            );
        }
        if found.kind.family() != plan.kind.family() {
            return fail(
                state::FailureReason::Decode(format!(
                    "the source changed from {} to {} mid-stream",
                    plan.kind.describe(),
                    found.kind.describe()
                )),
                out,
            );
        }
        // The precision the device was opened for is a ceiling. A packet that
        // needs more bits than the negotiated container holds must be caught
        // here — one frame later it is already truncated on the wire.
        if let (PcmKind::Integer { valid_bits: want }, PcmKind::Integer { valid_bits: have }) =
            (found.kind, plan.kind)
            && want > have
        {
            return fail(
                state::FailureReason::Decode(format!(
                    "the source needs {want} valid bits but the output was negotiated for {have}"
                )),
                out,
            );
        }
    }

    Ok(found)
}
/// Open `path` with symphonia, optionally seek to `start`, and return a ready
/// decode pipeline plus the stream's native rate, channels and decoded family.
pub fn prepare(path: &Path, start: Duration) -> Result<Prepared, state::FailureReason> {
    use state::FailureReason as FR;

    let file = File::open(path).map_err(|e| FR::SourceOpen(format!("open failed: {e}")))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .map_err(|e| FR::SourceParse(format!("unrecognized format: {e}")))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| FR::SourceParse("no audio track".into()))?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| FR::SourceParse("unknown sample rate".into()))?;
    let channels = params
        .channels
        .map(|c| c.count() as u16)
        .ok_or_else(|| FR::SourceParse("unknown channel layout".into()))?;
    let bits_per_sample = params.bits_per_sample;

    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| FR::SourceParse(format!("no decoder: {e}")))?;

    // A failed seek is reported, and the position actually reached is returned
    // rather than assumed.
    //
    // The result used to be discarded with `let _`, so a container that refused
    // the seek carried on decoding from wherever it happened to be while the
    // UI, the spectrum and the elapsed-time base all moved to the position the
    // listener asked for. The audio and everything describing it disagreed, and
    // nothing anywhere reported a problem.
    let mut seeked_to = Duration::ZERO;
    if start > Duration::ZERO {
        let to = format
            .seek(
                SeekMode::Coarse,
                SeekTo::Time {
                    time: Time::from(start.as_secs_f64()),
                    track_id: Some(track_id),
                },
            )
            .map_err(|e| FR::Seek(format!("seek to {:.2}s failed: {e}", start.as_secs_f64())))?;
        seeked_to = Duration::from_secs_f64(to.actual_ts as f64 / sample_rate.max(1) as f64);
        decoder.reset();
    }

    // Decode one packet up front so the source family is measured rather than
    // guessed. It costs one packet of latency at open and is the only way to
    // learn that a "24-bit" FLAC arrives in 32-bit slots.
    let mut primed: Vec<u32> = Vec::new();
    let mut probed_format: Option<SourceFormat> = None;
    let mut reached_end = false;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                reached_end = true;
                break;
            }
            // Not an end of file. Symphonia asks for the decoder to be rebuilt
            // and the track list re-examined; treating it as EOF meant the
            // track simply stopped, silently, wherever the reset happened.
            Err(SymError::ResetRequired) => {
                return Err(FR::SourceRead(
                    "the container asked for a decoder reset while opening; \
                     this file needs re-examination rather than being played from here"
                        .into(),
                ));
            }
            Err(e) => return Err(FR::SourceRead(format!("read failed: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(d) => {
                let found = canonicalize_packet(&d, None, bits_per_sample, &mut primed)?;
                if primed.is_empty() {
                    continue;
                }
                probed_format = Some(found);
                break;
            }
            // A damaged packet is missing audio, not a formatting detail. It
            // used to be skipped in silence and the session then began claiming
            // to be exact, having already lost a packet of the source.
            Err(SymError::DecodeError(e)) => {
                return Err(FR::Decode(format!(
                    "the first audio packet is damaged: {e}"
                )));
            }
            Err(SymError::ResetRequired) => {
                return Err(FR::SourceRead(
                    "the decoder asked to be reset while opening".into(),
                ));
            }
            Err(e) => return Err(FR::Decode(format!("decode failed: {e}"))),
        }
    }

    // A seek past the last decodable packet is a legitimate position, not a
    // broken file — fall back to the declared parameters rather than refusing
    // to open a track the listener just scrubbed to the end of.
    let source = match probed_format {
        Some(f) => f,
        None if reached_end => {
            let kind = kind_from_params(&params).ok_or_else(|| {
                FR::SourceParse("could not establish the source sample format".into())
            })?;
            SourceFormat {
                kind,
                sample_rate,
                channels,
                layout: params
                    .channels
                    .map(|c| ChannelLayout(c.bits()))
                    .unwrap_or_default(),
            }
        }
        None => return Err(FR::SourceRead("no audio packets in this track".into())),
    };

    Ok(Prepared {
        format,
        decoder,
        track_id,
        sample_rate,
        bits_per_sample,
        source,
        primed,
        seeked_to,
        path: path.to_path_buf(),
    })
}
/// Decode a whole track and decide whether every sample is exactly
/// representable in Q1.31.
///
/// A complete decode, not a sample: a track can sit exactly on the integer grid
/// for ten minutes and then contain one value that does not, and a verdict from
/// a priming packet would be the same kind of claim-from-insufficient-evidence
/// this release exists to remove.
///
/// Runs on a worker thread. `cancel` is checked between packets so a track
/// change does not leave a scan grinding through a file nobody is listening to.
pub fn scan_track_q31(
    path: &Path,
    cancel: &AtomicBool,
) -> Result<q31::Q31Verdict, state::FailureReason> {
    let mut prep = prepare(path, Duration::ZERO)?;
    let mut verdict = q31::Q31Verdict::new();
    let channels = prep.source.channels;

    if !matches!(prep.source.kind, PcmKind::Float32) {
        // Only a Float32 source has anything to prove here.
        verdict.value_exact = false;
        return Ok(verdict);
    }

    let mut frames: u64 = 0;
    let primed = std::mem::take(&mut prep.primed);
    q31::scan_packet(&primed, channels, frames, &mut verdict);
    frames += (primed.len() / channels.max(1) as usize) as u64;

    let mut canon: Vec<u32> = Vec::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(state::FailureReason::Configuration("scan cancelled".into()));
        }
        let packet = match prep.format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(state::FailureReason::SourceRead(e.to_string())),
        };
        if packet.track_id() != prep.track_id {
            continue;
        }
        let decoded = match prep.decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => return Err(state::FailureReason::Decode(e.to_string())),
        };
        canon.clear();
        canonicalize_packet(
            &decoded,
            Some(&prep.source),
            prep.bits_per_sample,
            &mut canon,
        )?;
        q31::scan_packet(&canon, channels, frames, &mut verdict);
        frames += (canon.len() / channels.max(1) as usize) as u64;
    }
    Ok(verdict)
}
// ---------------------------------------------------------------------------
// Shared render-side state (used by both backends)
// ---------------------------------------------------------------------------

/// What the UI tick should do about a session's terminal state.
///
/// A reducer rather than a branch inside the tick, because this is the
/// decision the whole `Completion` distinction exists to make and it needs to
/// be checkable without a device, a window or a playlist.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TickAction {
    /// Still playing. Leave everything alone.
    Nothing,
    /// The source ended cleanly: advance per the loop mode.
    Advance,
    /// The source ended cleanly but the sleep timer asked to stop here.
    StopAtEndOfTrack,
    /// The session failed. Stop once, stay visible, reopen nothing.
    Halt(u8),
}

/// Decide what a terminal state means for playback.
///
/// The rule in one place: **only** `CleanEof` reaches `Advance`. A failure
/// reaching it is what turned a file that fails deterministically into an
/// unbounded reopen loop — under Repeat One the "next" track is the same file,
/// so the failure, the advance and the failure again ran as fast as the device
/// could be opened.
pub fn on_completion(completion: Completion, sleep_end_of_track: bool) -> TickAction {
    // A failure is checked first and on its own, so no later branch can turn
    // one into an advance by accident.
    if let Some(reason) = completion.failure() {
        return TickAction::Halt(reason);
    }
    if !completion.may_advance() {
        return TickAction::Nothing;
    }
    if sleep_end_of_track {
        TickAction::StopAtEndOfTrack
    } else {
        TickAction::Advance
    }
}

/// How a playing session ended, if it has.
///
/// This replaces a single `finished: AtomicBool`, and the distinction it draws
/// is the whole point. Every fatal path — a read error, a decode error, a
/// backend that died, a thread that panicked, a driver reset — used to set
/// that one flag, because the flag's real job was "stop waiting for audio".
/// The auto-advance then read it as "the track ended" and started the next
/// one, which under Repeat One is the *same* file. A file that fails
/// deterministically therefore reopened forever: a user's log shows the same
/// track opened hundreds of times in a row.
///
/// Only `CleanEof` may advance a playlist. A `Failed` session stops once,
/// stays visible, and is never reopened by anything except the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Completion {
    /// Still playing, or still draining.
    Running,
    /// The source reached its end and every frame it produced has been played.
    CleanEof,
    /// The session stopped because something went wrong. Carries the fault
    /// code and the generation it belonged to.
    Failed { reason: u8, generation: u64 },
}

impl Completion {
    /// Whether playback is over, for any reason. What "stop waiting" means.
    pub fn is_over(self) -> bool {
        !matches!(self, Completion::Running)
    }

    /// Whether the playlist may advance, or Repeat may restart the track.
    /// **Only** a clean end of file earns this.
    pub fn may_advance(self) -> bool {
        matches!(self, Completion::CleanEof)
    }

    pub fn failure(self) -> Option<u8> {
        match self {
            Completion::Failed { reason, .. } => Some(reason),
            _ => None,
        }
    }
}

/// Wire codes for `Completion`, so it can live in an atomic on a realtime path.
pub(crate) mod done_code {
    pub const RUNNING: u8 = 0;
    pub const CLEAN_EOF: u8 = 1;
    pub const FAILED: u8 = 2;
}

/// A one-byte code and the generation it belongs to, in a single `u64`.
///
/// Both facts a stamped value carries have to be read together or the pair is
/// meaningless — a reader that sees the new code beside the old generation
/// concludes the value is stale and discards a live fault; one that sees the
/// old code beside the new generation reports a fault the current session
/// never had. Two atomics cannot be read together, however they are ordered,
/// so they become one.
///
/// The generation occupies the top 56 bits. At one generation per track, seek
/// and gapless boundary, wrapping needs more state changes than a machine will
/// perform, and a wrap would only mean a stale value is briefly treated as
/// current rather than any memory being unsafe.
pub(crate) use stamped::{
    code as stamped_code, generation as stamped_generation, pack as stamped_pack,
};

mod stamped {
    /// Pack `generation` and `code` into one word.
    #[inline]
    pub(crate) const fn pack(generation: u64, code: u8) -> u64 {
        (generation << 8) | code as u64
    }
    #[inline]
    pub(crate) const fn generation(v: u64) -> u64 {
        v >> 8
    }
    #[inline]
    pub(crate) const fn code(v: u64) -> u8 {
        (v & 0xFF) as u8
    }
}
/// A generation and a running count, in a single word.
///
/// The same problem `stamped` solves, for the counters rather than the codes.
/// A count that is reset when a session begins and added to by the render
/// thread cannot be a bare `AtomicU64`: the render thread reads the session,
/// is descheduled, a new session is installed and the counter zeroed, and the
/// old thread's `fetch_add` lands on the new session's total. The listener's
/// position jumps by whatever the previous track had left in flight.
///
/// 24 bits of generation and 40 of count. Forty bits is sixteen hours at
/// 768 kHz and the count saturates rather than wrapping into the generation;
/// twenty-four is sixteen million track changes in one run of the process.
/// Neither is reachable, and both are checked.
pub(crate) mod counted {
    pub(crate) const COUNT_BITS: u32 = 40;
    pub(crate) const COUNT_MASK: u64 = (1u64 << COUNT_BITS) - 1;
    pub(crate) const GEN_MASK: u64 = (1u64 << 24) - 1;

    #[inline]
    pub(crate) const fn pack(generation: u64, count: u64) -> u64 {
        ((generation & GEN_MASK) << COUNT_BITS) | (count & COUNT_MASK)
    }
    #[inline]
    pub(crate) const fn generation(v: u64) -> u64 {
        v >> COUNT_BITS
    }
    #[inline]
    pub(crate) const fn count(v: u64) -> u64 {
        v & COUNT_MASK
    }
    /// Whether `v` belongs to `at`. The tag is the low bits of the
    /// generation, so this is the same comparison the writer makes.
    #[inline]
    pub(crate) const fn belongs_to(v: u64, at: u64) -> bool {
        generation(v) == (at & GEN_MASK)
    }
}

/// Generation numbers, allocated once and never handed out twice.
///
/// Both clocks used to invent their own successor — the playback clock by
/// `play + 1` and the decode clock by `fetch_add` on its own word — which
/// meant the two counters advanced independently and could arrive at the same
/// number. They did, in the one schedule that matters: A playing, B queued so
/// the decode clock is one ahead, then a seek or a new track calls
/// `begin_generation`, which took the *playback* clock and added one. That is
/// B's number. Everything B's decode thread had left to say — its end of
/// source, its rounding evidence, a fault it had parked, a boundary it was
/// midway through publishing — was addressed to a generation that now named
/// the track that had replaced it, and every stamp check in this file waved it
/// through, because the stamps genuinely matched.
///
/// One counter for the process removes the arithmetic. A number is allocated,
/// used by exactly one session, and never seen again; a replacement is
/// therefore greater than every generation that has ever existed, which
/// includes both clocks, without anyone having to compare them.
pub(crate) mod gen_alloc {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Starts at one so that zero stays available as "no generation" — the
    /// stamp a cleared slot carries, which must never match a live session.
    static NEXT: AtomicU64 = AtomicU64::new(1);

    /// A generation no session in this process holds or has held.
    #[inline]
    pub(crate) fn next() -> u64 {
        NEXT.fetch_add(1, Ordering::Relaxed)
    }
}

/// A way to park a thread *inside* a transition, for tests.
///
/// Every race this module is written against lives in a window a few
/// instructions wide, between deciding what to write and writing it. A test
/// that rendezvouses on either side of a call does not enter that window and
/// so does not test it: it passes against the broken design as readily as
/// against the repaired one, which makes it worse than no test at all.
///
/// The hook fires once, from inside `Shared::transition`, at the instant the
/// old value has been read and the compare-exchange has not yet run. It is
/// keyed to one `Shared` so that tests running in parallel cannot consume each
/// other's, and it is taken rather than copied so a single arming fires
/// exactly once.
#[cfg(test)]
pub(crate) mod window {
    use std::sync::{Mutex, OnceLock};

    /// Where in the code a hook fires. There is more than one window worth
    /// entering, and a hook armed for one must not be consumed by another.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Site {
        /// Inside `Shared::transition`, after the load and before the
        /// compare-exchange.
        Transition,
        /// Inside `Shared::fault_decoding`, after the decoder has established
        /// that its track is not the one playing and before it parks the
        /// fault.
        DeferFault,
        /// Between two stages of publishing a boundary. Publication is
        /// ordered — the ownership claim, then metadata, then the pending
        /// count, then the generation, then the frame that claims it — and
        /// each gap is a place the render and UI threads can be let loose to
        /// prove the order holds. Stage `n` fires after step `n` has
        /// completed; stage 0 fires after the claim and before the queue is
        /// touched.
        BoundaryPublish(u8),
        /// In a decode loop, after the queued successor has been taken off
        /// the queue and before the rollover claims the decode clock. This is
        /// where a real superseded decoder sits.
        Successor,
        /// In `note_decode_eof`, after the liveness check and before the
        /// compare-exchange that publishes it — which is exactly the gap a
        /// check-then-store leaves open.
        DecodeEof,
        /// In `note_off_grid`, after the liveness check and before the
        /// compare-exchange that publishes the count.
        OffGrid,
        /// Between two field reads inside `evidence_at`. Stage `n` fires
        /// after field `n` has been read, so a test can publish into the gap
        /// and prove the read notices.
        Evidence(u8),
        /// Inside a writer, after its store has landed and before the
        /// publication window closes. This is the instant the old
        /// store-then-announce design could not represent: the field has
        /// changed and nothing has said so. 1 is a recoverable record, 2 a
        /// terminal one, 3 a rounding count, 4 the reclamation in
        /// `begin_generation`.
        Publish(u8),
        /// At the top of `begin_generation`, immediately before it contends
        /// for the publication lock. A test releases a parked publisher on
        /// this rather than on a sleep.
        ResetContend,
        /// In a renderer, after the outer `paused` read and before it contends
        /// for the session slot. This is the gap a pause and a new session can
        /// both land in, and the only place a test can stand to force it.
        Render,
    }

    type Hook = Box<dyn Fn() + Send + Sync>;
    static ARMED: OnceLock<Mutex<Vec<(u64, Site, Hook)>>> = OnceLock::new();

    fn cell() -> &'static Mutex<Vec<(u64, Site, Hook)>> {
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Arm a one-shot hook for the `Shared` with this id, at this site.
    pub fn arm(id: u64, site: Site, f: impl Fn() + Send + Sync + 'static) {
        cell().lock().unwrap().push((id, site, Box::new(f)));
    }

    /// An id no other `Shared` in this process holds.
    ///
    /// One allocator for all three backends, because the registry is one map
    /// keyed by `(id, site)` and the backends now share a site. Three private
    /// counters each starting at one meant a PCM `Shared` and an ASIO one both
    /// answering to id 3, and a renderer taking the hook armed for a different
    /// backend's test — which parks two threads on barriers nobody will reach.
    pub fn next_id() -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Run and consume the hook armed for `id` at `site`, if there is one.
    pub fn fire(id: u64, site: Site) {
        // Taken out from under the lock before it runs: a hook exists in order
        // to park the thread that fires it, and parking while holding this
        // would deadlock whoever is meant to release it.
        let taken = {
            let mut g = cell().lock().unwrap();
            g.iter()
                .position(|(k, s, _)| *k == id && *s == site)
                .map(|i| g.remove(i).2)
        };
        if let Some(f) = taken {
            f();
        }
    }
}

/// Everything the UI reads about one audio generation, read as one thing.
///
/// The four facts used to be four separate getters behind a fifth call that
/// asked whether the generation was still current. That is four chances for
/// the device to cross a gapless boundary — after the check, or between any
/// two of the reads — and the fields that came back could then describe two
/// different tracks: an off-grid count from the incoming one beside a
/// recoverable code from the outgoing one, with nothing anywhere able to
/// notice. The check narrowed the window; it did not close it, because the
/// window is *between the reads* and no check placed before them can be about
/// what happens after them.
///
/// Every field here is read by its stamp, which settles *whose* evidence each
/// one is: asking about a generation that has already ended returns that
/// generation's own evidence while it survives and `NONE`/zero once its slots
/// have been reclaimed — never another track's. It is not possible to assemble
/// one of these out of two tracks.
///
/// Stamping is not the whole of it, because it says nothing about *when*.
/// Three atomic loads separated by a publication are three instants however
/// well each one is labelled, and the account they produce can be one that
/// never held — most damagingly a terminal code beside an empty recoverable
/// slot, which reads as a track that died with nothing having gone wrong,
/// when the dropout landed a microsecond after that field was read.
///
/// The fields are therefore read under a publication window: every writer
/// announces that it is *about to* change something before it changes it, and
/// announces again when it has finished. See [`stable_evidence`] for the
/// protocol and [`Evidence::settled`] for what the answer is worth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evidence {
    /// The generation every field below was read against.
    pub generation: u64,
    /// Samples the running Q1.31 conversion could not represent exactly.
    pub off_grid: u64,
    /// The recoverable integrity loss, or `fault::NONE`. The route stopped
    /// being what it claimed and the music continued.
    pub revoked: u8,
    /// The terminal fault, or `fault::NONE`. Fatal only.
    pub fatal: u8,
    /// Whether the last attempt read a quiet instant.
    ///
    /// `true` means no publication was in flight across that attempt's loads,
    /// so what they saw held together. `false` means every attempt was
    /// straddled by a writer and the fields below are the *union* of what was
    /// observed — never less than the truth, possibly more recent in one field
    /// than another, and never a basis for granting a claim.
    ///
    /// The distinction exists for exactly one caller. A value-exact label is
    /// granted on "no sample was rounded", and a rounding count that is being
    /// published *right now* has already been announced but not yet stored: a
    /// reader that answered zero there would upgrade a track on the strength
    /// of a number it was in the middle of being told.
    pub settled: bool,
}

impl Evidence {
    /// Nothing recorded against `at` — the identity of [`Evidence::fold`].
    #[inline]
    pub(crate) fn none(at: u64) -> Self {
        Evidence {
            generation: at,
            off_grid: 0,
            revoked: fault::NONE,
            fatal: fault::NONE,
            settled: false,
        }
    }

    /// Everything either of these two reads saw.
    ///
    /// Sound because each field is monotone within a generation: the off-grid
    /// count only grows, and the two codes are first-wins, so a later read can
    /// only add. What it protects against is the opposite — a *later* read
    /// seeing less, which happens when the generation's slots are reclaimed
    /// between two attempts. The union keeps what was already observed rather
    /// than letting the reclamation swallow it.
    ///
    /// `settled` is *not* unioned; it is taken from `other`, which is always
    /// the newer attempt. It describes the read, not the session.
    #[inline]
    pub(crate) fn fold(self, other: Evidence) -> Self {
        debug_assert_eq!(self.generation, other.generation);
        Evidence {
            generation: self.generation,
            off_grid: self.off_grid.max(other.off_grid),
            revoked: if self.revoked != fault::NONE {
                self.revoked
            } else {
                other.revoked
            },
            fatal: if self.fatal != fault::NONE {
                self.fatal
            } else {
                other.fatal
            },
            settled: other.settled,
        }
    }
}

/// The window a writer publishes inside.
///
/// Two counters, not one. A single counter bumped *after* the store is not a
/// protocol at all: the writer's change is visible before the counter moves,
/// so a reader can load the counter, read a field the writer has already
/// changed, read the counter again unchanged, and conclude it saw one instant.
/// That is the shape this replaced, and it failed exactly where it mattered —
/// a dropout and the write failure after it, both stored, neither announced,
/// read as a track that died with nothing wrong.
///
/// `begin` moves *before* any field changes and `end` moves after. A reader
/// that observes `begin == end`, then reads, then observes `begin` unchanged,
/// has read across an interval in which no writer was inside its window.
///
/// Multi-writer safe without any mutual exclusion between writers, which is
/// the requirement that rules out an odd/even single counter: two overlapping
/// publications leave that counter even in the middle of the second one. Here
/// `begin` and `end` are independent `fetch_add`s and the invariant is only
/// ever `begin >= end`, with equality meaning nobody is inside.
///
/// Nothing here blocks, allocates, logs or spins: a writer pays two relaxed-
/// ordered increments, which is what makes it usable from a realtime callback.
#[derive(Debug, Default)]
pub(crate) struct PublishWindow {
    begin: AtomicU64,
    end: AtomicU64,
}

impl PublishWindow {
    pub(crate) fn new() -> Self {
        Self {
            begin: AtomicU64::new(0),
            end: AtomicU64::new(0),
        }
    }

    /// Announce a publication that is about to change a field.
    ///
    /// Must be called *before* the store or compare-exchange, and paired with
    /// [`PublishWindow::finish`] on every path out — including the paths where
    /// the write turns out not to happen, because a reader cannot tell the
    /// difference between a writer that decided not to write and one that has
    /// not written yet.
    #[inline]
    pub(crate) fn start(&self) {
        self.begin.fetch_add(1, Ordering::AcqRel);
    }

    /// The publication is over, whether or not it changed anything.
    #[inline]
    pub(crate) fn finish(&self) {
        self.end.fetch_add(1, Ordering::Release);
    }

    /// The `begin` count if nobody is inside the window, else `None`.
    ///
    /// `end` is loaded first on purpose. `begin` never decreases and is never
    /// below `end`, so `end_first <= end(t) <= begin(t) = begin_second`; when
    /// the two loads are equal, the counts were equal at the instant `begin`
    /// was loaded and no writer was inside.
    #[inline]
    fn quiet(&self) -> Option<u64> {
        let end = self.end.load(Ordering::Acquire);
        let begin = self.begin.load(Ordering::Acquire);
        (begin == end).then_some(begin)
    }

    /// Whether any publication has started since `mark`.
    #[inline]
    fn undisturbed_since(&self, mark: u64) -> bool {
        self.begin.load(Ordering::Acquire) == mark
    }

    /// Open the window, and close it when the returned guard drops.
    ///
    /// A guard rather than a pair of calls because these writers are
    /// compare-exchange loops with several early returns — the generation is
    /// no longer live, the record is already taken, no slot is free — and a
    /// window left open on one of those paths would make every later read
    /// unsettled forever.
    #[inline]
    pub(crate) fn publish(&self) -> Publishing<'_> {
        self.start();
        Publishing(self)
    }
}

/// Holds a publication window open for its lifetime.
pub(crate) struct Publishing<'a>(&'a PublishWindow);

impl Drop for Publishing<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.finish();
    }
}

/// How many times to try for a quiet instant before answering from the union.
///
/// Bounded, because this runs on the UI thread once per frame and a reader
/// that spun until the writers stopped would be a reader that can be starved
/// by a busy device. An answer is always produced; what varies is whether it
/// is marked [`Evidence::settled`].
pub(crate) const EVIDENCE_ATTEMPTS: usize = 8;

/// Read evidence under the publication window, and keep everything seen.
///
/// Each attempt reads the fields whether or not it expects to be coherent, and
/// every attempt is folded in — so an attempt that loses its race still
/// contributes what it saw. That matters in both directions: a losing attempt
/// may be the only one that sees a rounding count before the generation is
/// reclaimed, and a *later* attempt may be the only one that sees a record a
/// parked writer had stored but not yet announced.
///
/// The answer is the union. `settled` says whether the last attempt read a
/// quiet instant; when it did, the fields also describe one, subject to the
/// union above never under-reporting.
///
/// There is no ordering trick here and no claim that nothing publishes after a
/// fatal record. Both were wrong: a decode thread can revoke, and the ALSA and
/// ASIO callbacks can revoke, after the session has already been failed.
pub(crate) fn stable_evidence(
    window: &PublishWindow,
    at: u64,
    read: impl Fn() -> Evidence,
) -> Evidence {
    let mut folded = Evidence::none(at);
    for _ in 0..EVIDENCE_ATTEMPTS {
        let mark = window.quiet();
        let mut snap = read();
        snap.settled = match mark {
            Some(m) => window.undisturbed_since(m),
            None => false,
        };
        folded = folded.fold(snap);
        if folded.settled {
            return folded;
        }
    }
    folded
}

/// A `Shared` that a test outside this module can drive.
///
/// `Shared` and its writers are private, and deliberately so — nothing outside
/// this file may publish evidence. But the one caller that *grants* a claim on
/// the strength of evidence lives in `main.rs`, and the property worth proving
/// is that it refuses to grant one while a publication is in flight. These are
/// the three verbs that test needs, and nothing else.
#[cfg(test)]
pub(crate) mod test_shared {
    use super::{window, Shared};
    use std::sync::{Arc, Barrier};

    pub(crate) fn new() -> Arc<Shared> {
        Arc::new(Shared::new())
    }

    pub(crate) fn begin(sh: &Arc<Shared>) -> u64 {
        sh.begin_generation()
    }

    pub(crate) fn note_off_grid(sh: &Arc<Shared>, at: u64, n: u64) {
        sh.note_off_grid(at, n);
    }

    /// Park the next `note_off_grid` inside its publication window, after the
    /// window has opened and before the count is stored.
    pub(crate) fn park_off_grid(sh: &Arc<Shared>, inside: Arc<Barrier>, release: Arc<Barrier>) {
        window::arm(sh.hook_id, window::Site::OffGrid, move || {
            inside.wait();
            release.wait();
        });
    }
}

/// A gapless hand-off: where it happens, and what begins there.
///
/// Built on the decode thread at the moment of the swap — after the next
/// track has passed the same compatibility check the queue applied — and
/// applied by the UI thread in one step once the device has played past
/// `frames`. Partial application is what the single struct exists to prevent.
#[derive(Clone, Debug)]
pub struct Boundary {
    /// Cumulative frames pushed since `start()` at which the swap happens.
    pub frames: u64,
    /// The generation that begins here. A fault raised before it belongs to
    /// the track that ended.
    pub generation: u64,
    /// What the new track actually is — not the carrier it travels on.
    pub source: state::MediaSource,
    /// What is being done to it, if anything.
    pub transform: TransformDescription,
    /// The route plan the open stream is carrying it under.
    pub plan: PayloadPlan,
    /// The policy in force when it was queued, so a policy change mid-track
    /// cannot retroactively relabel a stream that was opened under the old one.
    pub policy: state::OutputPolicy,
    /// The file that begins here, so its own value-exactness can be looked up
    /// or proved rather than inheriting its predecessor's verdict.
    pub path: PathBuf,
    /// Whether the track beginning here ends on a half-full DoP carrier frame.
    ///
    /// Carried across because the *next* hand-off has to know it. It was
    /// recorded only when a DoP session opened, so after one gapless
    /// hand-off it described whichever track had opened the stream and every
    /// boundary after that was judged on the wrong file's frame count.
    pub odd_tail: bool,
}

/// One playing track (or seek segment): the ring consumer plus the flags
/// shared with its decode thread.
struct Session {
    /// The generation this session belongs to.
    ///
    /// A session is installed *after* the generation that owns it has been
    /// opened, and the previous session stays in the slot until the moment of
    /// the swap. In that window the render thread is looking at the outgoing
    /// session while `Shared` already reports the incoming generation, and
    /// every verdict it drew — this track has ended, this track has failed —
    /// landed on the wrong one. A decode thread that finished its track
    /// microseconds before the next `start()` therefore failed the track that
    /// was starting, instantly, with `SessionThreadFailed`.
    ///
    /// Every conclusion the render thread reaches is now stamped with this
    /// rather than with whatever `Shared` happens to say, so a session can
    /// only ever speak about itself.
    generation: u64,
    cons: FrameConsumer<u32>,
    decode_done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

pub(crate) struct Shared {
    session: Mutex<Option<Session>>,
    paused: AtomicBool,
    /// **The** playback state: which generation is playing and how it ended,
    /// in one word.
    ///
    /// These were two atomics — `session_gen` and a stamped `completion` — and
    /// every transition was a check against one followed by a write to the
    /// other. That is not a transition; it is two, with a window between them,
    /// and the window is exactly wide enough for the thing it was meant to
    /// prevent. A decode thread from track A reads the generation, finds its
    /// own, and is descheduled. `start()` installs track B. The A thread wakes
    /// and writes its failure — stamped, correctly, with a generation that is
    /// now current, because it read it before it stopped being A's. Track B
    /// fails before its first sample, blaming a file it never opened.
    ///
    /// Installing a generation and ending one are now the same kind of
    /// operation on the same word, each a compare-exchange, each ordered
    /// against every other. A writer that lost the race cannot write: the
    /// compare-exchange it retries sees a generation that is not its own and
    /// it gives up, which is what a superseded session should do.
    play: AtomicU64,
    /// The decode generation whose source ran out cleanly, stamped.
    ///
    /// Set by the decode thread when it reaches the end of its source without
    /// anything going wrong. The render thread promotes it to `CleanEof` once
    /// the ring has actually drained — the decoder finishing is not the same
    /// event as the device having played what it produced.
    ///
    /// Stamped rather than a bare flag, because the decoder can be a track
    /// ahead: a plain `true` left by the outgoing track was read by the render
    /// thread as "the session is over" while a boundary for the incoming one
    /// was still queued and unplayed, which ended the playlist entry early and
    /// skipped the track that was about to start.
    decode_eof: AtomicU64,
    /// Boundaries pushed by the decoder and not yet crossed by the device.
    ///
    /// The render thread may not take the `boundaries` lock — it has a
    /// deadline — so the one fact it needs from that queue lives here. A drain
    /// that completes while this is non-zero would end a session that still
    /// has a track to play.
    pending_boundaries: AtomicU64,
    /// The frame at which the queued track begins, readable without a lock.
    ///
    /// `u64::MAX` when nothing is queued. This is what lets the *render*
    /// thread cross the boundary, which is the only thread that knows when the
    /// device gets there.
    ///
    /// It was crossed by the UI thread, on a frame tick, in
    /// `take_reached_boundary`. Between the device actually arriving and the
    /// UI noticing, up to a frame of audio played under the previous track's
    /// generation: a dropout in the first sixteen milliseconds of a gapless
    /// successor was recorded against its predecessor, which had already
    /// finished, and was discarded as stale. Worse, a single callback deep
    /// enough to span the boundary emitted both tracks' audio under one
    /// generation and there was no instant at which anything could be said
    /// about the second.
    next_boundary_frame: AtomicU64,
    /// The generation that begins at `next_boundary_frame`.
    next_boundary_gen: AtomicU64,
    /// Boundaries the render thread has crossed and the UI has not yet
    /// consumed the metadata for. The UI reads the queue only when this is
    /// non-zero, so it can never describe a track the device has not reached.
    crossed_boundaries: AtomicU64,
    /// Frames handed to the device since the last `start()`, stamped with the
    /// generation they were handed under — see `counted`.
    ///
    /// A bare counter was wrong in the same way everything else here was: the
    /// render thread reads its session, is descheduled, a new session is
    /// installed and the counter zeroed, and the old callback's `fetch_add`
    /// credits the new track with the previous one's frames. The position
    /// jumps forward at every track change by however much was in flight.
    frames_played: AtomicU64,
    /// Gapless: the next track's decode pipeline, queued by the UI thread and
    /// picked up by the decode thread the instant the current file ends (only
    /// ever holds a track with the same rate/channels as the open stream).
    next: Mutex<Option<Prepared>>,
    /// Gapless for DSD: the next DoP stream, same role as `next` but for the
    /// DoP decode loop (only ever holds a DSD track at the same carrier
    /// rate/channels as the open stream).
    next_dop: Mutex<Option<DopFileStream>>,
    /// Where one track ends and the next begins, and everything the UI needs
    /// to describe the track that begins.
    ///
    /// A bare frame count was not enough. The UI rolled the *title* over at
    /// the boundary and left the source, the transform and the fidelity
    /// describing the track that had just ended — so a 24-bit track following
    /// a 16-bit one, or a processed rung following an exact one, kept the
    /// previous track's badge for the rest of its life.
    boundaries: Mutex<std::collections::VecDeque<Boundary>>,
    /// True while a DoP (DSD) session is active. The ring then carries bare
    /// 16-bit DSD payloads rather than audio samples, so the spectrum tap is
    /// not fed from it — those bits are not a waveform.
    dop_active: AtomicBool,
    /// The first realtime integrity fault of this session, as a `fault::` code.
    ///
    /// Written by realtime threads with a single relaxed compare-exchange and
    /// read by the UI poller. There is no string, no allocation and no lock on
    /// the audio side, which is the whole reason it is a `u8`.
    ///
    /// Meaningless without the generation beside it, so the two share one
    /// word — see `stamped`. It is this session's fault only if that
    /// generation still equals `session_gen`.
    fault: AtomicU64,
    /// The first *recoverable* loss of integrity of this session, stamped.
    ///
    /// Kept apart from `fault` because the two are different claims and the
    /// worse one must not erase the earlier one. A session that dropped out
    /// and then had its device reject a write ends as a write failure — that
    /// is what stopped it — but it also dropped out, and the badge, the panel
    /// and the log all have to keep saying so. With one slot and first-wins,
    /// the dropout suppressed the write error and the session reported the
    /// wrong reason for stopping; with one slot and last-wins, the write error
    /// erased the evidence that anything had been lost before it.
    revoked: AtomicU64,
    /// The window every evidence publication happens inside.
    ///
    /// Not evidence itself: it is how a reader tells whether the fields it
    /// read describe one instant. It was a single counter bumped *after* each
    /// store, which is not a protocol — the store is visible before the
    /// counter moves, so a reader could see a changed field with an unchanged
    /// count and call it a snapshot.
    evidence: PublishWindow,
    /// The track the *decoder* is working on.
    ///
    /// Advanced by every start and seek, and by pushing a gapless boundary.
    /// Ahead of `session_gen` exactly while a boundary is queued but unplayed.
    ///
    /// A decode fault for a track that has not started playing must not be
    /// shown against the track that is: it waits in `deferred_fault` until the
    /// device reaches it. Attributing it immediately reported an error in
    /// track B while the listener was still hearing track A.
    decode_gen: AtomicU64,
    /// A fault raised by the decoder for a track the device has not reached,
    /// stamped with the *decode* generation it belongs to.
    deferred_fault: AtomicU64,
    /// Render calls that could not reach the session slot in time. Expected
    /// during a track swap (the UI thread holds the lock for an instant);
    /// a symptom the rest of the time.
    lock_misses: AtomicU64,
    /// Count of render calls that could not fill the device buffer while a
    /// track was genuinely playing — i.e. the decoder fell behind or this
    /// thread was descheduled past the buffer depth. Each one is a stretch of
    /// silence the DAC played, so this is the only direct evidence of a
    /// dropout the process can produce: the test suite cannot observe one, and
    /// by the time a listener hears it there is nothing left to inspect.
    underruns: AtomicU64,
    /// What the open device format can carry, so the decode thread can apply
    /// the same reuse rule the UI thread applied when it queued a track.
    ///
    /// The gapless queue checks compatibility when the track is *queued*; this
    /// is the check at the moment of the swap, against the format that was
    /// actually negotiated. Without it a queue that outlived its check could
    /// hand the render side a wider source than the open container, and every
    /// sample of the second track would be truncated at a boundary the
    /// listener is not even supposed to notice.
    out_valid_bits: AtomicU32,
    out_integer: AtomicBool,
    /// Set once the backend thread has ended, for any reason.
    ///
    /// Separate from the session's fault: a session fault is cleared by a new
    /// session, a dead backend never is. Reusing a stream whose render thread
    /// has exited plays silence with nobody left to report it.
    backend_dead: AtomicBool,
    /// The payload plan, as a `PayloadPlan::code`. Read by the decode thread.
    plan: AtomicU8,
    /// Samples that were not exactly representable in Q1.31 and had to take
    /// the rounded path — **one slot per track in flight**.
    ///
    /// Two slots, because the decoder can be one track ahead of the device and
    /// never two: the gapless queue holds a single track. The decoder writes
    /// the slot its own generation selects; the UI reads the slot the *playing*
    /// generation selects. A single counter meant the incoming track's
    /// rounding was attributed to the outgoing one, which was still audible.
    ///
    /// Zero is a precondition for the value-exact label, never a proof: it
    /// covers what has played, and the claim is about the whole track. The
    /// complete scan supplies the rest, and both must agree.
    q31_off_grid: [AtomicU64; 2],
    /// The policy the stream was opened under, as an `OutputPolicy::code`.
    ///
    /// Carried on the stream rather than read live, so a policy change
    /// mid-track cannot retroactively relabel a route that was negotiated
    /// under the previous one. The new policy applies from the next open.
    policy: AtomicU8,
    /// Set when a session is installed, cleared the first time the ring hands
    /// over a full buffer.
    ///
    /// A seek tears the session down and builds a new one, so the ring is empty
    /// for however long the decoder needs to reach the new position. The render
    /// thread asks for samples in that window and gets none — which is a gap in
    /// the audio, but an expected one at a point the listener just asked to jump
    /// to, not the mid-track glitch the dropout counter exists to report. Every
    /// dropout counted so far has been one of these, which makes the counter
    /// noise rather than a signal.
    priming: AtomicBool,
    /// Frames the device has asked for since priming began.
    ///
    /// Priming had no bound at all: a decoder that never produced a frame left
    /// the device playing silence indefinitely with the session still claiming
    /// to be exact and nothing anywhere to explain it. Past
    /// `PRIMING_LIMIT_SECS` worth of frames the wait is not a seek any more,
    /// and it is reported as a fault.
    priming_frames: AtomicU64,
    /// The negotiated output rate, so the priming bound above is a duration
    /// rather than a buffer count. Zero until a device is negotiated.
    out_rate: AtomicU32,
    /// The device buffer, in frames. The audio still to be played when the
    /// ring runs dry.
    out_buffer_frames: AtomicU32,
    /// True between the ring running dry and the device having played what it
    /// was already given.
    draining: AtomicBool,
    /// Latched once the device has been given enough calls to have played out
    /// what it was holding.
    ///
    /// Separate from `draining` because the two answer different questions and
    /// the answers arrive at different times. "The device has finished playing
    /// out" is a fact about the hardware; "the session may end" additionally
    /// requires that no track is queued behind it. Collapsing them meant a
    /// drain that completed while a gapless boundary was still uncrossed
    /// simply switched itself off and was gone: the successor was never told
    /// the device had drained, so it never ended, and the playlist stopped on
    /// the last track of every gapless run.
    drain_satisfied: AtomicBool,
    /// Frames the device has asked for since the drain began.
    drain_frames: AtomicU64,
    /// When the drain began, in milliseconds since this stream opened.
    ///
    /// The frame ceiling is not a bound. It counts frames the device *asks
    /// for*, so a backend that stops calling back never reaches it: WASAPI and
    /// CPAL both had a drain that could sit open for the life of the process,
    /// with the track neither advancing nor failing and nothing anywhere
    /// saying why. The native backends were given a wall clock in the previous
    /// pass; this is the same bound for the generic ones.
    drain_started_ms: AtomicU64,
    /// This stream's epoch, for the field above.
    opened_at: Instant,
    /// Identifies this `Shared` to `window::fire`. Tests only.
    #[cfg(test)]
    hook_id: u64,
}

/// How long a drain may take before it is called finished anyway.
///
/// A device buffer is milliseconds; this is orders of magnitude longer. It
/// exists so that a driver which stops asking for audio cannot leave a track
/// permanently "almost finished", never advancing and never failing.
const DRAIN_LIMIT_SECS: u64 = 5;

/// How long a session may sit priming before the silence is called a fault.
///
/// Generous on purpose: a deep seek into a FLAC without a seek table is a real
/// wait the listener asked for. Nothing legitimate takes longer than this.
const PRIMING_LIMIT_SECS: u64 = 10;

impl Shared {
    fn new() -> Self {
        Shared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            // Zero: nothing is playing until `start()` allocates a
            // generation. Starting on one meant the first allocation collided
            // with a clock that already held it.
            play: AtomicU64::new(stamped::pack(0, done_code::RUNNING)),
            decode_eof: AtomicU64::new(stamped::pack(0, 0)),
            pending_boundaries: AtomicU64::new(0),
            next_boundary_frame: AtomicU64::new(u64::MAX),
            next_boundary_gen: AtomicU64::new(0),
            crossed_boundaries: AtomicU64::new(0),
            frames_played: AtomicU64::new(counted::pack(1, 0)),
            next: Mutex::new(None),
            next_dop: Mutex::new(None),
            boundaries: Mutex::new(std::collections::VecDeque::new()),
            dop_active: AtomicBool::new(false),
            fault: AtomicU64::new(stamped::pack(0, fault::NONE)),
            revoked: AtomicU64::new(stamped::pack(0, fault::NONE)),
            evidence: PublishWindow::new(),
            decode_gen: AtomicU64::new(0),
            deferred_fault: AtomicU64::new(stamped::pack(0, fault::NONE)),
            lock_misses: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            // Zero until a device format is negotiated: a `Shared` with no
            // stream behind it refuses every gapless continuation, which is the
            // safe direction.
            out_valid_bits: AtomicU32::new(0),
            out_integer: AtomicBool::new(true),
            backend_dead: AtomicBool::new(false),
            plan: AtomicU8::new(0),
            q31_off_grid: [AtomicU64::new(0), AtomicU64::new(0)],
            policy: AtomicU8::new(state::OutputPolicy::PreferExact.code()),
            priming: AtomicBool::new(false),
            priming_frames: AtomicU64::new(0),
            out_rate: AtomicU32::new(0),
            out_buffer_frames: AtomicU32::new(0),
            draining: AtomicBool::new(false),
            drain_satisfied: AtomicBool::new(false),
            drain_frames: AtomicU64::new(0),
            drain_started_ms: AtomicU64::new(0),
            opened_at: Instant::now(),
            #[cfg(test)]
            hook_id: window::next_id(),
        }
    }

    /// The generation currently playing.
    ///
    /// The playback clock and the terminal state are the same word, so this is
    /// never read separately from the state it belongs to by anything that
    /// needs both.
    #[inline]
    fn generation(&self) -> u64 {
        stamped::generation(self.play.load(Ordering::Acquire))
    }

    /// The generation the decoder is producing.
    #[inline]
    fn decode_generation(&self) -> u64 {
        self.decode_gen.load(Ordering::Acquire)
    }

    /// Record `n` off-grid samples against `at`.
    ///
    /// Two slots, found by their stamp rather than by an index computed from
    /// the generation. The index used to be `generation & 1`, which was only
    /// sound while the two clocks advanced in lockstep and differed by exactly
    /// one. They are allocated independently now — a seek between two tracks
    /// consumes a number nobody plays — so the playing and queued generations
    /// can share a parity, and the queued track's `note_off_grid(g, 0)` would
    /// have wiped the evidence of the track still sounding.
    ///
    /// Two is still enough. At most one boundary is ever queued, so at most
    /// two generations are live at once, and a slot held by neither is free.
    /// A write from a superseded generation finds no slot of its own and no
    /// free slot it is entitled to, and is dropped — which is the point.
    fn note_off_grid(&self, at: u64, n: u64) {
        // Open before the first load, not after the store: a rounding count
        // that is *about to* be published has to be visible as in flight, or a
        // value-exact verdict computed in that instant reads zero and upgrades
        // the track on a number it is in the middle of being told.
        let _publishing = self.evidence.publish();
        let want = counted::pack(at, n);
        'attempt: loop {
            // Only a live generation may write evidence at all.
            if !self.evidence_live(at) {
                return;
            }
            // The slot this generation owns, else one no live generation is
            // using. The value read is kept, because it is what the
            // compare-exchange below is conditional on.
            let mut target = None;
            for slot in &self.q31_off_grid {
                let cur = slot.load(Ordering::Acquire);
                if counted::belongs_to(cur, at) {
                    target = Some((slot, cur));
                    break;
                }
            }
            if target.is_none() {
                let playing = self.generation();
                let decoding = self.decode_generation();
                for slot in &self.q31_off_grid {
                    let cur = slot.load(Ordering::Acquire);
                    if !counted::belongs_to(cur, playing)
                        && !counted::belongs_to(cur, decoding)
                    {
                        target = Some((slot, cur));
                        break;
                    }
                }
            }
            let Some((slot, seen)) = target else { return };
            // The window a validate-then-store leaves open, and the reason
            // this is a compare-exchange.
            //
            // The check above passed while this generation was still the one
            // decoding. A `start()` can land here: the successor becomes the
            // live generation, writes its own zeroed evidence into a slot, and
            // this thread — still holding a raw store and a slot index it
            // computed before any of that — wrote its count over the new
            // track's. The new track then reported the old one's rounding as
            // its own, which is the number its value-exact label is decided
            // by.
            #[cfg(test)]
            window::fire(self.hook_id, window::Site::OffGrid);
            if !self.evidence_live(at) {
                return;
            }
            match slot.compare_exchange(seen, want, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    // Stored, and the window is still open. This is the
                    // instant a reader must not mistake for a quiet one.
                    #[cfg(test)]
                    window::fire(self.hook_id, window::Site::Publish(3));
                    return;
                }
                // Someone else took the slot. Look again: either this
                // generation still has somewhere to write, or it is no longer
                // entitled to write at all.
                Err(_) => continue 'attempt,
            }
        }
    }

    /// Whether `at` is one of the two generations that may still publish.
    #[inline]
    fn evidence_live(&self, at: u64) -> bool {
        at == self.generation() || at == self.decode_generation()
    }

    /// Off-grid samples observed in `at`, or zero if neither slot is its.
    #[inline]
    fn off_grid_in(&self, at: u64) -> u64 {
        for slot in &self.q31_off_grid {
            let v = slot.load(Ordering::Relaxed);
            if counted::belongs_to(v, at) {
                return counted::count(v);
            }
        }
        0
    }

    /// A generation greater than everything either clock is holding.
    ///
    /// True by construction rather than by comparison: the allocator is
    /// monotone and both clocks only ever hold numbers it has already handed
    /// out. The assertion states the property the rest of this file relies on,
    /// so that an allocator that stopped having it would be caught here rather
    /// than by a listener.
    fn fresh_generation(&self) -> u64 {
        let g = gen_alloc::next();
        debug_assert!(
            g > self.generation() && g > self.decode_generation(),
            "a replacement generation must outrank both clocks"
        );
        g
    }

    /// Move the decode clock forward to `g`, never backwards.
    ///
    /// A decode thread unwinding from a superseded track can still be inside
    /// its rollover when a new session opens; a plain store would let it put
    /// its own, older number back over the new one and reopen every window
    /// this file closes.
    fn set_decode_generation(&self, g: u64) {
        let mut cur = self.decode_gen.load(Ordering::Acquire);
        while cur < g {
            match self.decode_gen.compare_exchange_weak(
                cur,
                g,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Advance the generation at a gapless rollover.
    ///
    /// Unlike `begin_generation` this does not re-arm priming: at a gapless
    /// boundary the ring is continuously full, and pretending otherwise would
    /// blind the dropout counter for the whole priming window on every track
    /// change — which is exactly where a dropout is most likely.
    /// The decoder has begun a queued track. Advances the *decode* clock only.
    ///
    /// The device is still playing the previous one, and will be for up to a
    /// full ring. Advancing the playback clock here rolled every visible fact
    /// over early: the badge, the source and the transform described the
    /// incoming track while the outgoing one was still audible.
    /// Advance the decode clock alone, for tests whose subject is the clock
    /// rather than the hand-off.
    ///
    /// It claims through the same compare-exchange the production rollover
    /// uses, so a test cannot advance a clock the production path would have
    /// refused to advance.
    #[cfg(test)]
    fn roll_decode_generation(&self) -> u64 {
        let expected = self.decode_generation();
        let g = self.fresh_generation();
        assert!(
            self.decode_gen
                .compare_exchange(expected, g, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "the decode clock moved under this test"
        );
        self.note_off_grid(g, 0);
        g
    }

    /// Claim the decode clock for a queued successor and publish its
    /// boundary — one transaction, or nothing.
    ///
    /// Two defects, and they are the same defect at two scales.
    ///
    /// The rollover used to mint its generation unconditionally. A decode
    /// thread parked between taking the successor off the queue and rolling
    /// came back after a `start()` had replaced everything, allocated a number
    /// *newer* than the one the new session was using, and stored it as the
    /// decode clock. Every ownership check downstream then compared against
    /// that number and passed, because the superseded thread had made itself
    /// current. A stop flag does not help: it is read before the same window.
    /// The claim is a compare-exchange from the generation the caller believes
    /// it owns, so a thread that has been replaced cannot replace anything.
    ///
    /// And publication used to be four separate stores after one check. A
    /// reset landing between them cleared a queue the publisher then pushed
    /// into, and zeroed a count the publisher then incremented — a boundary
    /// belonging to a dead track, in a live session's queue, with a pending
    /// count that never came back down. The whole of it now happens under the
    /// `boundaries` lock, which `begin_generation` also takes for the whole of
    /// its reset. A publisher is therefore either wholly published before the
    /// reset — and wholly cleared by it — or it takes the lock afterwards,
    /// re-reads the clock it no longer owns, and publishes nothing.
    ///
    /// Returns the new decode generation, or `None` if this caller has been
    /// superseded and should stop.
    fn roll_and_publish(
        &self,
        expected: u64,
        at_frame: u64,
        make: impl FnOnce(u64) -> Boundary,
    ) -> Option<u64> {
        let g = self.fresh_generation();
        // Allocated before the lock and simply discarded if the claim fails.
        // Numbers are cheap and gaps in them mean nothing; what matters is
        // that no two sessions ever share one.
        let Ok(mut q) = self.boundaries.lock() else {
            return None;
        };
        self.decode_gen
            .compare_exchange(expected, g, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        // Claimed. From here to the end of this function nothing else can
        // reset the session, because the reset wants this lock.
        self.publish_claimed(&mut q, g, at_frame, make(g));
        Some(g)
    }

    /// The stores half of the publication transaction, with the claim already
    /// made and the lock already held.
    ///
    /// Factored so that production and the tests that drive publication timing
    /// run the *same* code. They did not: the test seam had its own copy of
    /// these four stores, so every assertion about ordering was an assertion
    /// about the copy, and the production stores after a successful
    /// compare-exchange were never executed by a test at all.
    ///
    /// `q` is passed rather than taken, so the caller's guard is provably the
    /// one held across the whole thing.
    fn publish_claimed(
        &self,
        q: &mut std::collections::VecDeque<Boundary>,
        g: u64,
        at_frame: u64,
        b: Boundary,
    ) {
        debug_assert_eq!(b.generation, g, "the boundary must carry the claimed generation");
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::BoundaryPublish(0));
        // The successor's rounding evidence starts from a clean sheet in its
        // own slot; the outgoing track's stays readable until the device
        // leaves it. The outgoing track's end-of-source stamp stays where it
        // is too — it is stamped with *its* generation, so it is no longer an
        // answer to "has the track now decoding finished".
        self.note_off_grid(g, 0);
        q.push_back(b);
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::BoundaryPublish(1));
        self.pending_boundaries.fetch_add(1, Ordering::AcqRel);
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::BoundaryPublish(2));
        self.next_boundary_gen.store(g, Ordering::Release);
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::BoundaryPublish(3));
        // The frame is the claim the render thread swaps for, so it goes last
        // and it goes with `Release`.
        self.next_boundary_frame.store(at_frame, Ordering::Release);
    }

    /// The *device* has crossed a boundary into the track the decoder started
    /// at `decode_generation`.
    ///
    /// This is where every visible fact rolls over, and where a fault the
    /// decoder raised for this track — before it was audible — becomes the
    /// user's business.
    fn reach_boundary(&self, decode_generation: u64) -> u64 {
        // The frames already played belong to the track that has just ended;
        // the new one starts from nothing. Carried over, the successor's
        // position began wherever its predecessor's had reached.
        let carried = self.frames_played();
        // The drain is deliberately left alone. It was reset here, which threw
        // away the successor's only notice that the device had played out —
        // the drain covers everything still in the device when the ring ran
        // dry, and at a boundary that is the tail of the outgoing track *and*
        // the whole of the incoming one. Clearing it lost the event and the
        // last track of a gapless run never ended.
        self.release_pending();
        // The number the decoder was allocated when it queued this track, not
        // a fresh one. The playback clock catching up to the decode clock is
        // what a gapless boundary *is*, and inventing a third number here left
        // the queued track's evidence, its parked fault and its end-of-source
        // stamp all addressed to a generation that was never installed.
        let g = self.install_generation(decode_generation);
        if g != decode_generation {
            // A `start()` overtook this claim between the swap and here. The
            // new session owns everything from this point; the boundary it
            // replaced has nothing left to say.
            return g;
        }
        // Promote anything the decoder recorded against this track while it
        // was still queued. Claimed rather than read, so that a decoder
        // publishing at this instant and this call cannot both act on it.
        if let Some(code) = self.take_deferred(decode_generation) {
            self.fail(g, code);
        }
        // The counter moves to the new generation without losing its place:
        // `frames` in a `Boundary` is a running total since `start()`, and the
        // UI subtracts the boundary's own frame from it, so the count has to
        // keep running across the swap.
        self.frames_played
            .store(counted::pack(g, carried), Ordering::Relaxed);
        // And complete it, if everything it was waiting for has already
        // happened. Nothing new is produced here: the drain and the
        // end-of-source were both recorded before the device arrived, and this
        // is the first moment at which they are about this track.
        self.try_finish(g);
        g
    }

    /// Finish `at` if every condition for a clean end has already been met.
    ///
    /// Called from both sides of the same race — by the drain when it
    /// completes, and by the boundary when it installs a track the drain has
    /// already covered — because either can be the last to arrive.
    fn try_finish(&self, at: u64) {
        if !self.drain_satisfied.load(Ordering::Acquire) {
            return;
        }
        // A track is still queued: the device has played out, but not out of
        // audio the listener has yet to hear.
        if self.boundary_pending() {
            return;
        }
        if !self.decode_eof_for(at) {
            return;
        }
        self.finish_clean(at);
    }

    /// The decoder reached the end of its source cleanly. Not yet the end of
    /// playback — the ring still holds audio the device has not played.
    /// `at` is a *decode* generation: this is the decoder speaking about the
    /// track it was working on, which may still be queued behind the one
    /// playing.
    fn note_decode_eof(&self, at: u64) {
        let want = stamped::pack(at, 1);
        let mut cur = self.decode_eof.load(Ordering::Acquire);
        loop {
            // Monotone in the generation. A decoder parked between its check
            // and its store came back to find its track replaced, and the
            // store went through anyway: it put *its* generation's answer over
            // the successor's, so `decode_eof_for(successor)` became false and
            // the track that had genuinely finished never ended. The session
            // sat at almost-complete until the drain's wall clock failed it.
            if stamped::generation(cur) > at {
                return;
            }
            if stamped::generation(cur) == at && stamped::code(cur) != 0 {
                return; // already said, and saying it twice says nothing
            }
            if at != self.decode_generation() {
                return;
            }
            // The gap the old check-then-store left open.
            #[cfg(test)]
            window::fire(self.hook_id, window::Site::DecodeEof);
            match self.decode_eof.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Whether the decoder has cleanly finished the track `at` is playing.
    ///
    /// A bare flag could not answer this. The decoder sets it for the track it
    /// just finished, which at a gapless boundary is not the track the device
    /// is on, and the render thread read the outgoing track's flag as leave to
    /// end the *session*.
    #[inline]
    fn decode_eof_for(&self, at: u64) -> bool {
        let v = self.decode_eof.load(Ordering::Acquire);
        stamped::generation(v) == at && stamped::code(v) != 0
    }

    /// Publish a queued track: metadata first, then the claim.
    ///
    /// The order is the contract, and it used to be the other way round. The
    /// realtime descriptor went out first and the `Boundary` was pushed onto
    /// the queue afterwards, so the render thread could cross a boundary whose
    /// metadata did not exist yet, hand the UI a token for it, and leave the
    /// UI popping an empty queue — the token spent, the track's source,
    /// transform and path gone for good.
    ///
    /// Reversed:
    ///
    /// 1. the complete `Boundary` goes on the queue, where nothing realtime
    ///    can see it;
    /// 2. the pending count goes up, so a drain that completes from here on
    ///    knows a track is still queued behind it and refuses to finish the
    ///    session;
    /// 3. the generation is stored;
    /// 4. the frame is stored, with `Release`, and *that* is the claim — it is
    ///    the one word the render thread swaps, so everything above is visible
    ///    to whoever wins the swap.
    ///
    /// Returns false, having published nothing, if the caller has been
    /// superseded.
    /// Publish a boundary for a generation the caller has already claimed.
    ///
    /// The rollover goes through `roll_and_publish`, which claims and
    /// publishes as one step. This exists for callers that already hold the
    /// decode clock and only want the publication half — the test seam below,
    /// and nothing in production.
    #[cfg(test)]
    fn publish_boundary(&self, b: Boundary) -> bool {
        let expected = self.decode_generation();
        if b.generation != expected {
            return false;
        }
        // Claim `expected` with itself: a no-op compare-exchange that still
        // fails if the clock moved, so this shares the production path's
        // rejection rule rather than having one of its own.
        self.roll_in_place(expected, b).is_some()
    }

    /// The publication half of `roll_and_publish`, for a generation already
    /// installed on the decode clock.
    ///
    /// It runs `publish_claimed` — the same stores production runs — under the
    /// same lock, so a test of publication timing is a test of the production
    /// stores and not of a copy of them.
    #[cfg(test)]
    fn roll_in_place(&self, expected: u64, b: Boundary) -> Option<u64> {
        let at_frame = b.frames;
        let Ok(mut q) = self.boundaries.lock() else {
            return None;
        };
        if self.decode_generation() != expected || b.generation != expected {
            return None;
        }
        self.publish_claimed(&mut q, expected, at_frame, b);
        Some(expected)
    }

    /// Queue a boundary with placeholder metadata, for tests that care about
    /// the timing and not about what begins there.
    ///
    /// It goes through `publish_boundary`, so the queue and the descriptor
    /// stay consistent and every test that consumes through
    /// `take_reached_boundary` gets something to consume. A test-only shortcut
    /// that published the descriptor alone would have been a test-only version
    /// of the bug.
    #[cfg(test)]
    fn note_boundary_pushed(&self, at_frame: u64, generation: u64) -> bool {
        self.publish_boundary(Boundary {
            frames: at_frame,
            generation,
            source: state::MediaSource::pcm(SourceFormat {
                kind: PcmKind::Integer { valid_bits: 24 },
                sample_rate: 44_100,
                channels: 2,
                layout: ChannelLayout::UNSPECIFIED,
            }),
            transform: TransformDescription::Identity,
            plan: PayloadPlan::Identity,
            policy: state::OutputPolicy::PreferExact,
            path: PathBuf::from("queued.flac"),
            odd_tail: false,
        })
    }

    /// Consume one crossed boundary's metadata, if the device has crossed one
    /// *and* the metadata for it is here.
    ///
    /// The queue is inspected first, and the token is spent only once there is
    /// something for it to be spent on. Taking the token first and then
    /// finding the queue empty destroyed the only record that a boundary had
    /// been crossed: the count went down, the `None` said nothing had
    /// happened, and the track that began there was never described — no
    /// title, no source, no transform, for the rest of its play.
    ///
    /// The lock is held across the claim so the check and the pop cannot be
    /// separated by another consumer.
    pub(crate) fn take_boundary(&self) -> Option<Boundary> {
        // `try_lock`, because a publisher holding this lock is mid-transaction
        // and there is nothing coherent to consume yet. Blocking here would
        // put the UI thread to sleep behind a decode thread; answering "not
        // yet" costs one frame and is the truth.
        let mut q = self.boundaries.try_lock().ok()?;
        q.front()?;
        if !self.take_crossed() {
            return None;
        }
        q.pop_front()
    }

    /// Give back one pending boundary, without ever going below none.
    ///
    /// `fetch_sub` wrapped. `begin_generation` zeroes this counter, and a
    /// render thread that had already claimed a boundary would then return
    /// one it no longer owed — `0 - 1` is `u64::MAX`, and `boundary_pending`
    /// would have answered "yes, forever" from that moment on, which is a
    /// session that can never finish cleanly again.
    #[inline]
    fn release_pending(&self) {
        let _ = self.pending_boundaries.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |cur| cur.checked_sub(1),
        );
    }

    /// One attempt: three loads, in one order.
    ///
    /// There is no second order. The previous version read the terminal record
    /// first on its last attempt, on the argument that nothing can be
    /// published against a generation after its fatal record — which is not
    /// true of this code. A decode thread unwinding after a write failure, and
    /// both native callbacks, can and do revoke afterwards.
    fn read_evidence(&self, at: u64) -> Evidence {
        let off_grid = self.off_grid_in(at);
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::Evidence(1));
        let revoked = self.revoked_in(at);
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::Evidence(2));
        let fatal = self.fatal_in(at);
        Evidence {
            generation: at,
            off_grid,
            revoked,
            fatal,
            settled: false,
        }
    }

    /// Everything about `at`, read by `at`'s own stamps and as one account.
    /// See [`Evidence`] and [`stable_evidence`].
    fn evidence_at(&self, at: u64) -> Evidence {
        stable_evidence(&self.evidence, at, || self.read_evidence(at))
    }

    /// Whether a queued track is still waiting for the device.
    #[inline]
    fn boundary_pending(&self) -> bool {
        self.pending_boundaries.load(Ordering::Acquire) != 0
    }

    /// Frames until the queued track begins, or `None` if none is queued or
    /// the device is already past it.
    #[inline]
    fn frames_to_boundary(&self, played: u64) -> Option<u64> {
        let at = self.next_boundary_frame.load(Ordering::Acquire);
        if at == u64::MAX {
            return None;
        }
        Some(at.saturating_sub(played))
    }

    /// The device has reached the queued track. Called from the render thread,
    /// and only from there.
    ///
    /// Returns the generation now playing, or `None` if there was nothing
    /// queued — which happens when two callbacks race and the first has
    /// already crossed.
    fn cross_boundary_now(&self) -> Option<u64> {
        // The swap is the claim and the acquire, in that order and in one
        // operation: whoever takes the frame out has synchronised with the
        // release that put it there, so the generation beside it — and the
        // metadata on the queue before that — are both visible. Reading the
        // generation *first*, as this did, read a word the publisher might not
        // have written yet and paired it with a frame from the boundary after.
        if self.next_boundary_frame.swap(u64::MAX, Ordering::AcqRel) == u64::MAX {
            return None;
        }
        let queued = self.next_boundary_gen.load(Ordering::Acquire);
        let g = self.reach_boundary(queued);
        if g != queued {
            // Superseded between the claim and the install. There is no track
            // beginning here for the UI to describe, and minting a token for
            // one would spend the metadata of whatever is on the queue next.
            return Some(g);
        }
        // Only now may the UI describe the new track: it has been played.
        self.crossed_boundaries.fetch_add(1, Ordering::AcqRel);
        Some(g)
    }

    /// Whether the UI may consume one boundary's metadata.
    ///
    /// The UI used to decide this itself, by comparing `frames_played` against
    /// the queue's front — and then install the generation, which is a thing
    /// only the device can know the timing of.
    #[inline]
    fn take_crossed(&self) -> bool {
        let mut cur = self.crossed_boundaries.load(Ordering::Acquire);
        loop {
            if cur == 0 {
                return false;
            }
            match self.crossed_boundaries.compare_exchange_weak(
                cur,
                cur - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Frames handed to the device under the generation that is playing.
    #[inline]
    fn frames_played(&self) -> u64 {
        counted::count(self.frames_played.load(Ordering::Acquire))
    }

    /// Credit `got` frames to `at`, and to `at` alone.
    ///
    /// A callback that read its session and was then descheduled past a track
    /// change must not add its frames to the successor's total. The
    /// compare-exchange is what refuses it.
    fn account_frames(&self, at: u64, got: u64) {
        if got == 0 {
            return;
        }
        let mut cur = self.frames_played.load(Ordering::Acquire);
        loop {
            if !counted::belongs_to(cur, at) {
                return; // superseded: these frames belong to a track that is over
            }
            let next = counted::pack(at, counted::count(cur).saturating_add(got));
            match self.frames_played.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// The ring has run dry with the source finished. The device still holds
    /// what it was last given, so the track is not over yet.
    fn begin_drain(&self) {
        self.drain_frames.store(0, Ordering::Relaxed);
        self.drain_satisfied.store(false, Ordering::Relaxed);
        self.drain_started_ms.store(
            self.opened_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        self.draining.store(true, Ordering::Release);
    }

    /// Whether a drain has been open too long to still be a drain.
    ///
    /// Read from the UI thread, because that is the one still running when the
    /// backend has stopped: a drain is advanced by the device asking for
    /// audio, and a device that has stopped asking gives it nothing to advance
    /// with.
    fn drain_expired(&self) -> bool {
        if !self.draining.load(Ordering::Acquire) {
            return false;
        }
        let began = self.drain_started_ms.load(Ordering::Relaxed);
        let now = self.opened_at.elapsed().as_millis() as u64;
        now.saturating_sub(began) >= DRAIN_LIMIT_SECS * 1_000
    }

    /// End a drain that ran out of time rather than out of audio.
    ///
    /// A failure, not an ending: the device did not play what it was holding,
    /// so the track did not finish and a playlist must not advance on it.
    fn expire_drain(&self) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        self.draining.store(false, Ordering::Release);
        self.fail_now(fault::BACKEND_DEAD);
    }

    /// Advance a drain by the frames the device just asked for, and finish the
    /// session once it has had enough to have played everything it holds.
    ///
    /// Emitting `CleanEof` the instant the *ring* ran dry reported a track as
    /// finished while up to a full device buffer of it was still unplayed, so
    /// the next track began over the tail of the last one — audible, and worse
    /// on a large exclusive buffer. The bound is generous and, more
    /// importantly, is a bound: a driver that stops asking for audio ends the
    /// track rather than leaving it permanently almost-finished.
    fn advance_drain(&self, want_frames: usize) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        let n = self
            .drain_frames
            .fetch_add(want_frames as u64, Ordering::Relaxed)
            + want_frames as u64;
        let buffer = self.out_buffer_frames.load(Ordering::Relaxed) as u64;
        let rate = self.out_rate.load(Ordering::Relaxed) as u64;
        // Twice the device buffer covers the deepest write plus the one in
        // flight; the rate term keeps the bound sane on a driver that reports
        // a tiny or absent buffer size.
        let target = (buffer * 2).max(rate / 20).max(1);
        let ceiling = rate.saturating_mul(DRAIN_LIMIT_SECS).max(target);
        if n >= target || n >= ceiling {
            self.draining.store(false, Ordering::Release);
            // Latched, not consumed. A drain that outran the UI thread must
            // not end a session with a track still queued behind it — the ring
            // empties past a boundary before anyone has popped it — but it
            // must not be *forgotten* either, which is what returning here
            // used to do. `reach_boundary` picks it up.
            self.drain_satisfied.store(true, Ordering::Release);
            self.try_finish(self.generation());
        }
    }

    /// Playback ended, cleanly, having played everything.
    ///
    /// Refused if anything has already gone wrong in this generation: a
    /// session that faulted and then drained did not end cleanly, and letting
    /// the drain overwrite the fault is exactly how a failure became an
    /// advance.
    fn finish_clean(&self, at: u64) {
        // Integrity and completion are separate questions. A track that
        // dropped out still ended; a track that lost its value-exact claim
        // still ended. What the fault costs is the *badge*, which
        // `Presentation` reads directly and which this does not touch.
        // Consulting it here turned one momentary underrun into a playlist
        // that stopped at the end of the track — the fault was rewritten as a
        // failure, and a failure never advances.
        self.transition(at, done_code::CLEAN_EOF, false)
    }

    /// Install `g` as the playing generation, atomically with clearing the
    /// terminal state that belonged to the last one.
    ///
    /// `g` comes from the allocator; this no longer invents it. Deriving it
    /// from the word it was about to overwrite was how a replacement session
    /// could be handed the number a queued decoder was still using.
    ///
    /// Refuses to go backwards, and returns whatever is playing afterwards. A
    /// boundary claimed by the render thread can lose to a `start()` that ran
    /// between the claim and this call; the newer session wins and the caller
    /// is told which one it is.
    fn install_generation(&self, g: u64) -> u64 {
        debug_assert!(g != 0, "zero is the empty stamp, not a generation");
        let want = stamped::pack(g, done_code::RUNNING);
        let mut cur = self.play.load(Ordering::Acquire);
        loop {
            let held = stamped::generation(cur);
            if held >= g {
                return held; // a newer session got here first
            }
            match self.play.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return g,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Move `at` to a terminal state, or refuse.
    ///
    /// One compare-exchange, on the word that also holds the generation. That
    /// is the whole point: a writer whose generation has been superseded
    /// cannot succeed, however long it was descheduled between deciding to
    /// write and writing. `fatal` outranks a clean end already recorded for
    /// the same generation; a clean end never overwrites anything.
    fn transition(&self, at: u64, code: u8, fatal: bool) {
        let want = stamped::pack(at, code);
        let mut cur = self.play.load(Ordering::Acquire);
        // The window: the old value is in hand and nothing has been written.
        // A check-then-store design is wrong from here on, and this is where a
        // test gets to prove it.
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::Transition);
        loop {
            if stamped::generation(cur) != at {
                return; // superseded: this session no longer speaks for anyone
            }
            let held = stamped::code(cur);
            if held != done_code::RUNNING && !(fatal && held == done_code::CLEAN_EOF) {
                return; // already terminal, and this is not a failure outranking it
            }
            match self
                .play
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Playback ended because something went wrong.
    ///
    /// Latching, and it outranks a clean end: once a session has failed no
    /// later drain may quietly turn it back into a track that finished.
    fn fail(&self, at: u64, code: u8) {
        self.raise_fault_in(at, code);
        // Outranks a clean end already recorded for the same generation: a
        // session that drained and *then* faulted did not finish either.
        self.transition(at, done_code::FAILED, true);
    }

    /// Record a loss of integrity that does not end the session.
    ///
    /// A dropout, a missed lock, an off-grid sample: the route stops being
    /// what it claimed, the badge has to say so, and the music keeps playing.
    /// A torn ring is not one of these and is not accepted here — the channels
    /// cannot be told apart afterwards, so there is nothing left to keep
    /// playing that is worth hearing. The distinction is `fault::is_fatal`,
    /// and the debug
    /// assertion is here because calling this with a fatal code would silently
    /// leave a session that has no audio left reporting itself as running.
    #[inline]
    fn revoke(&self, at: u64, code: u8) {
        debug_assert!(
            !fault::is_fatal(code),
            "revoke is for integrity faults; {code} ends the session"
        );
        let _publishing = self.evidence.publish();
        if at != self.generation() {
            return;
        }
        let want = stamped::pack(at, code);
        let mut cur = self.revoked.load(Ordering::Acquire);
        loop {
            // Monotone in the generation, first-wins within one — the same
            // rule as `fault`, on its own record.
            if stamped::generation(cur) > at {
                return;
            }
            if stamped::generation(cur) == at && stamped::code(cur) != fault::NONE {
                return;
            }
            match self
                .revoked
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // Stored, not yet published: the window is what a reader
                    // in the middle of its loads has to see.
                    #[cfg(test)]
                    window::fire(self.hook_id, window::Site::Publish(1));
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// The terminal fault recorded against `at`, or `fault::NONE`.
    ///
    /// Fatal only, and read by the stamp rather than by "whatever is playing
    /// now" — which is what makes it composable with the other three fields.
    #[inline]
    fn fatal_in(&self, at: u64) -> u8 {
        let v = self.fault.load(Ordering::Acquire);
        if stamped::generation(v) == at {
            stamped::code(v)
        } else {
            fault::NONE
        }
    }

    /// The recoverable integrity loss recorded against `at`, if any.
    ///
    /// Survives a later fatal fault: the session ended for the fatal reason
    /// and it also lost its claim earlier, and both are true.
    #[inline]
    fn revoked_in(&self, at: u64) -> u8 {
        let v = self.revoked.load(Ordering::Acquire);
        if stamped::generation(v) == at {
            stamped::code(v)
        } else {
            fault::NONE
        }
    }

    /// The recoverable loss of the session playing now.
    ///
    /// Tests only. Production reads `evidence_at`, which answers about one
    /// named generation; this answers about "now", which is the question that
    /// cannot be composed with another.
    #[cfg(test)]
    #[inline]
    pub(crate) fn revoked(&self) -> u8 {
        self.revoked_in(self.generation())
    }

    /// How this session ended, or `Running`.
    fn completion(&self) -> Completion {
        // One load. The generation and the state it describes cannot disagree,
        // because there is nothing to disagree with.
        let v = self.play.load(Ordering::Acquire);
        let generation = stamped::generation(v);
        match stamped::code(v) {
            done_code::CLEAN_EOF => Completion::CleanEof,
            done_code::FAILED => Completion::Failed {
                reason: self.fault_in(generation),
                generation,
            },
            _ => Completion::Running,
        }
    }

    /// Open a new logical session and return its generation.
    ///
    /// Retires every fault older than it in the same step: a fault stamped
    /// with a superseded generation can never be read back as current, so a
    /// new session, a seek and a gapless rollover all start clean without
    /// anyone having to remember to clear a flag.
    fn begin_generation(&self) -> u64 {
        // The reset side of the publication transaction.
        //
        // Held across all of it, including the two clock installs at the
        // bottom, so that a publisher blocked here re-reads the clock *after*
        // the new generation is in place and finds it no longer owns anything.
        // Taking this lock only to clear the queue left the rest of the reset
        // outside the critical section, which is where the publisher's stores
        // used to land.
        //
        // The hook fires immediately before the lock is contended, which is
        // what lets a test release a parked publisher on a handshake rather
        // than on a sleep: from here this thread either takes the lock or
        // waits for it, and nothing else.
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::ResetContend);
        let mut _reset = self.boundaries.lock();
        self.priming.store(true, Ordering::Relaxed);
        self.priming_frames.store(0, Ordering::Relaxed);
        // Cleared against the generation about to be installed, which is the
        // next one — a bare zero is another track's stamp as far as
        // `off_grid_in` is concerned, and reads as "no evidence" either way.
        // A reclamation is a publication as far as a reader is concerned: the
        // evidence of the outgoing generation is about to stop being readable,
        // and a read straddling it would report less than it had already seen.
        {
            let _publishing = self.evidence.publish();
            self.q31_off_grid[0].store(0, Ordering::Relaxed);
            self.q31_off_grid[1].store(0, Ordering::Relaxed);
            #[cfg(test)]
            window::fire(self.hook_id, window::Site::Publish(4));
        }
        self.draining.store(false, Ordering::Relaxed);
        // The latch, too.
        //
        // It was cleared by `begin_drain` and nowhere else, so a stream that
        // outlived its session — a reuse, which is the whole point of keeping
        // the device open — carried the previous track's "the device has
        // played out" into the next one. A short gapless successor on a reused
        // stream then completed the moment its boundary was crossed, on the
        // strength of a drain that finished before it existed.
        self.drain_satisfied.store(false, Ordering::Relaxed);
        self.drain_frames.store(0, Ordering::Relaxed);
        self.next_boundary_frame.store(u64::MAX, Ordering::Relaxed);
        self.crossed_boundaries.store(0, Ordering::Relaxed);
        self.deferred_fault
            .store(stamped::pack(0, fault::NONE), Ordering::Relaxed);
        self.decode_eof.store(stamped::pack(0, 0), Ordering::Relaxed);
        self.pending_boundaries.store(0, Ordering::Relaxed);
        // And the metadata behind the count.
        //
        // Zeroing the count while leaving the queue full meant a track that
        // had been queued and then replaced left its `Boundary` in place; the
        // next real hand-off pushed behind it, and the UI popped the abandoned
        // track's source, transform and path at the first boundary of the new
        // session. The queue and the count are one fact and are cleared as one
        // — under the lock this whole reset is already holding.
        if let Ok(q) = _reset.as_mut() {
            q.clear();
        }
        // A fresh session: both clocks start together, because nothing is
        // queued and the decoder is not ahead of anything. The number is
        // allocated, so it is greater than anything a decode thread that has
        // not noticed yet might still be holding.
        let g = self.fresh_generation();
        self.set_decode_generation(g);
        self.install_generation(g);
        // The frame counter belongs to the new generation from here.
        //
        // It was stamped only by `start()`, which meant every other way of
        // opening a generation — a seek, a rollover — left the counter
        // carrying the previous one's tag, and `account_frames` then refused
        // every frame the new track played. The position stopped moving.
        self.frames_played
            .store(counted::pack(g, 0), Ordering::Relaxed);
        g
    }

    /// End the session that is playing now, because the backend under it has
    /// failed.
    ///
    /// For producers that outlive sessions — the render thread, the backend —
    /// where "now" is the only answer available. These were published as
    /// integrity faults, which recorded a reason and ended nothing: the
    /// session stayed `Running` on a device that had stopped accepting audio,
    /// and only the drain's later refusal to finish cleanly turned it into
    /// anything at all.
    #[inline]
    fn fail_now(&self, code: u8) {
        debug_assert!(fault::is_fatal(code), "fail_now is for fatal faults");
        self.fail(self.generation(), code);
    }

    /// Publish a fault the decoder raised, against the track it was decoding.
    ///
    /// If that track is the one playing, it is the user's business now. If the
    /// decoder has run ahead into a queued track, it is not: the listener is
    /// still hearing the previous one, and showing an error against it would
    /// name the wrong track. It waits until the device arrives.
    fn fault_decoding(&self, decode_generation: u64, code: u8) {
        if decode_generation != self.decode_generation() {
            return; // a superseded decode thread
        }
        if decode_generation == self.generation() {
            self.fail(decode_generation, code);
            return;
        }
        // The window: this thread has established that its track is queued
        // rather than playing, and has not yet written anything down. The
        // device can cross the boundary from here.
        #[cfg(test)]
        window::fire(self.hook_id, window::Site::DeferFault);
        // Park it against the track it belongs to.
        let want = stamped::pack(decode_generation, code);
        let mut cur = self.deferred_fault.load(Ordering::Acquire);
        loop {
            // Monotone, for the same reason `fault` is. The check at the top
            // of this function reads the decode clock and can be raced: a
            // decoder still unwinding from a superseded track would otherwise
            // stamp its own, older generation over a fault the *queued* track
            // had legitimately recorded — and the queued one is the track that
            // is about to play.
            if stamped::generation(cur) > decode_generation {
                return;
            }
            if stamped::generation(cur) == decode_generation
                && stamped::code(cur) != fault::NONE
            {
                return; // first one wins for this track too
            }
            match self.deferred_fault.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
        // The handshake, and the reason there is one.
        //
        // The check above and the park below are two steps, and the device can
        // cross the boundary between them: this thread looks, finds track A
        // playing, is descheduled, the UI installs B, and only then does the
        // fault land in the slot. `reach_boundary` had already looked at that
        // slot and found it empty, so nobody promoted it — B's decode failure
        // was written down and never read, and B played on as though nothing
        // had happened.
        //
        // Both sides look again after acting, and both claim through the same
        // compare-exchange, so the fault is promoted exactly once however the
        // two are ordered.
        if decode_generation == self.generation()
            && let Some(claimed) = self.take_deferred(decode_generation)
        {
            self.fail(decode_generation, claimed);
        }
    }

    /// Claim the deferred fault belonging to `at`, if there is one.
    ///
    /// Claiming rather than reading is what makes the handshake safe to run
    /// from both sides: the compare-exchange has exactly one winner, so the
    /// fault is published once even when the decoder and the boundary reach it
    /// together.
    fn take_deferred(&self, at: u64) -> Option<u8> {
        let mut cur = self.deferred_fault.load(Ordering::Acquire);
        loop {
            if stamped::generation(cur) != at || stamped::code(cur) == fault::NONE {
                return None;
            }
            let code = stamped::code(cur);
            match self.deferred_fault.compare_exchange_weak(
                cur,
                stamped::pack(at, fault::NONE),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(code),
                Err(observed) => cur = observed,
            }
        }
    }

    /// Publish an integrity fault against `gen`, and only `gen`.
    ///
    /// For per-session producers. A decode thread that is still unwinding
    /// after its session was replaced calls this with the generation it was
    /// born in, the check fails, and its successor is left alone.
    #[inline]
    fn raise_fault_in(&self, at: u64, code: u8) {
        let _publishing = self.evidence.publish();
        if at != self.generation() {
            return;
        }
        let want = stamped::pack(at, code);
        let mut cur = self.fault.load(Ordering::Acquire);
        loop {
            // Monotone in the generation, which is what makes the check above
            // safe to have raced. A writer descheduled between reading the
            // clock and reaching here would otherwise stamp its own, older
            // generation over a fault legitimately raised by the session that
            // replaced it — losing a live fault to a dead one.
            if stamped::generation(cur) > at {
                return;
            }
            // First one wins *within a generation*: a dropout followed by a
            // write error is still a session that stopped being exact at the
            // dropout. A record from a superseded generation is stale and this
            // one replaces it.
            if stamped::generation(cur) == at && stamped::code(cur) != fault::NONE {
                return;
            }
            match self
                .fault
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    #[cfg(test)]
                    window::fire(self.hook_id, window::Site::Publish(2));
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// The terminal fault of the session playing now, or `fault::NONE`.
    ///
    /// **Fatal only.** This used to fall through to the recoverable record
    /// when there was no terminal one, which made it a general-purpose "what
    /// is wrong" accessor — and the UI, reasonably, pushed its answer into the
    /// one first-wins field that holds the reason a session ended. A dropout
    /// therefore took that field, and the backend write failure that actually
    /// ended the track arrived at a slot already occupied and was discarded.
    /// The listener was told the music stopped because of a dropout.
    ///
    /// The combined view still exists for `Completion::Failed`, which needs a
    /// reason and is only ever constructed once the session is genuinely over.
    /// It is private, and it is the only caller.
    #[inline]
    fn fault(&self) -> u8 {
        self.fatal_in(self.generation())
    }

    /// What this session has to answer for: the reason it stopped if it
    /// stopped, otherwise the claim it lost.
    ///
    /// A fatal reason outranks a recoverable one because it is the more
    /// serious statement and it is the one that explains why the audio ended.
    /// The recoverable record is not overwritten by it — `revoked_in` still
    /// answers, and the dropout counter still counts.
    #[inline]
    fn fault_in(&self, at: u64) -> u8 {
        let v = self.fault.load(Ordering::Acquire);
        if stamped::generation(v) == at && stamped::code(v) != fault::NONE {
            return stamped::code(v);
        }
        self.revoked_in(at)
    }

    /// Record that the backend thread has ended.
    ///
    /// Both parts matter, and they are different facts. `backend_dead` refuses
    /// stream reuse. The terminal state releases whoever is waiting for
    /// end-of-track, who would otherwise wait on a stream with nobody behind
    /// it forever — but it releases them with a **failure**, not with an
    /// ending. Publishing this as `finished` is what made a dead backend
    /// indistinguishable from a track that had played to its end, so the
    /// playlist advanced into the same dead stream and did it again.
    fn mark_backend_dead(&self) {
        self.backend_dead.store(true, Ordering::Release);
        self.fail(self.generation(), fault::BACKEND_DEAD);
    }

    /// Whether the negotiated device format can carry a source of `kind`
    /// under the plan this stream was opened for. Mirrors
    /// `StreamKey::can_carry`, on the axis the decode thread can see.
    ///
    /// The plan is the part that was missing. Asked about a Float32 source it
    /// used to answer "no, this is an integer stream" — true of the source,
    /// irrelevant to the question, and fatal at a gapless boundary: the second
    /// Float32 track of a playlist faulted with a format change on a route
    /// that was carrying the first one perfectly.
    fn can_carry_kind(&self, kind: PcmKind) -> bool {
        let plan = PayloadPlan::from_code(self.plan.load(Ordering::Relaxed));
        let payload = plan.device_kind(kind);
        let valid = self.out_valid_bits.load(Ordering::Relaxed);
        let integer = self.out_integer.load(Ordering::Relaxed);
        match payload {
            PcmKind::Integer { valid_bits } => integer && valid_bits as u32 <= valid,
            PcmKind::Float32 => !integer && valid == 32,
            PcmKind::Float64 => false,
        }
    }

}

/// Register the calling thread with MMCSS as a "Pro Audio" task, and leave the
/// task again when the returned guard drops.
///
/// The render threads are ordinary threads running a poll/sleep loop against a
/// device buffer measured in milliseconds. Everything else in this process is
/// also an ordinary thread — including a rayon pool that, during a superlet
/// pre-process, wants every core it can get for minutes at a time. Losing that
/// race by more than the buffer depth means the device plays whatever it has
/// left, which is audible.
///
/// MMCSS is the documented fix and what every WASAPI exclusive-mode host uses:
/// the scheduler guarantees the registered thread a share of the CPU regardless
/// of what else is runnable, instead of the blunt `THREAD_PRIORITY_TIME_CRITICAL`
/// which can starve everything else if the loop ever misbehaves. Failure is
/// silently fine — the thread simply runs at normal priority, exactly as it did
/// before, so a machine with the service disabled loses nothing it had.
#[cfg(windows)]
pub(crate) struct AudioPriority(isize);

#[cfg(windows)]
impl AudioPriority {
    pub(crate) fn claim() -> Self {
        use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
        // UTF-16, NUL-terminated: the task name must match a subkey of the
        // MMCSS Tasks registry key, and "Pro Audio" is the standard one.
        let task: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
        let mut index: u32 = 0;
        let h = unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut index) };
        AudioPriority(h as isize)
    }
}

#[cfg(windows)]
impl Drop for AudioPriority {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Threading::AvRevertMmThreadCharacteristics;
        if self.0 != 0 {
            unsafe { AvRevertMmThreadCharacteristics(self.0 as _) };
        }
    }
}

/// No-op elsewhere: cpal owns its own callback thread and asks the platform for
/// realtime scheduling itself, and Linux's equivalent (SCHED_FIFO) needs
/// privileges this process has no business demanding.
#[cfg(not(windows))]
pub(crate) struct AudioPriority;

#[cfg(not(windows))]
impl AudioPriority {
    pub(crate) fn claim() -> Self { AudioPriority }
}

/// Move up to `want_frames` **whole frames** of canonical payload from the
/// active session into `scratch`, handle pause and end-of-track, feed the
/// spectrum tap, and count frames. Returns the frame count; the caller pads
/// the rest of the device buffer.
///
/// This is the entire realtime render logic, shared by both backends. Three
/// things it deliberately does not do:
///
/// * **No volume.** There is no gain here to apply, exactly or otherwise. An
///   exact route's software gain is unity by construction, not by policy, and
///   the multiply that used to live at the bottom of this function is the
///   reason "bit-perfect at 80%" was expressible at all.
/// * **No partial frames.** The old drain loop popped single samples until the
///   ring ran dry, which could stop between a left and a right sample and
///   shift every channel for the rest of the session.
/// * **No allocation, no logging, no blocking lock.** A missed `try_lock` is
///   counted and, outside the priming window, raised as a fault; it is never
///   waited on.
fn render_frames(
    sh: &Shared,
    scratch: &mut [u32],
    tap: &mut SpectrumTap,
    channels: u16,
    want_frames: usize,
) -> usize {
    let ch = channels.max(1) as usize;
    let mut got = 0usize;

    if !sh.paused.load(Ordering::Relaxed) {
        #[cfg(test)]
        window::fire(sh.hook_id, window::Site::Render);
        match sh.session.try_lock() {
            Ok(mut guard) => {
                // Asked again, now that the slot is in hand.
                //
                // The outer read is a fast path and nothing more: between it
                // and this lock the UI thread can pause and install a new
                // session, and the answer the fast path gave was about a
                // session this callback is no longer the one to play.
                //
                // `pause` stores with `Release`, and the UI thread then takes
                // this same mutex to install the session. Acquiring it here
                // synchronises with that release, so a session installed after
                // a pause can never be reached with `paused` still reading
                // false. The load is `Acquire` for the same pairing rather
                // than for this edge, which the mutex already gives.
                //
                // Nothing is consumed before it: `guard.as_mut()` is the first
                // access to the session and it is second in this chain.
                if !sh.paused.load(Ordering::Acquire)
                    && let Some(sess) = guard.as_mut()
                {
                    // The track being *heard*, which is not `sess.generation`.
                    //
                    // A session survives gapless boundaries — same ring, same
                    // decode thread, new track — so its own generation is the
                    // one the stream opened on and stays there for the rest of
                    // the run. Stamping realtime faults with it meant that
                    // after the first rollover every dropout and every torn
                    // ring was attributed to a track that had already
                    // finished, found stale, and discarded: from the second
                    // track of a gapless album onwards, the render thread
                    // could not report anything at all.
                    //
                    // Reading the clock here is safe for the reason it was not
                    // before: the generation is installed inside this same
                    // lock, so while it is held the session in the slot and
                    // the clock belong to each other. `sess.generation` is the
                    // floor, and the assertion is the standing proof of it.
                    let at = sh.generation();
                    debug_assert!(
                        at >= sess.generation,
                        "the clock cannot be behind the session holding it"
                    );
                    // A ring holding a non-multiple of the channel count has
                    // been torn by something outside `frame_ring`, and there
                    // is no way to know which channel the stray samples belong
                    // to. It ends the track: everything after this point would
                    // play on the wrong channel for the rest of it, silently,
                    // and there is no resynchronisation that is not a guess.
                    if !sess.cons.invariant_holds() {
                        sh.fail(at, fault::RING_INVARIANT);
                    }

                    // Split at the boundary, if this callback spans one.
                    //
                    // A device buffer can be deeper than the tail of a track,
                    // in which case one callback emits the end of A and the
                    // beginning of B. Crossing afterwards — or, as it was,
                    // waiting for the UI thread to notice on its next frame —
                    // meant every sample of B in that buffer was played under
                    // A's generation, and anything that went wrong in it was
                    // recorded against a track that had already finished and
                    // thrown away as stale.
                    //
                    // The device is the only thing that knows when it arrives,
                    // so it is what crosses.
                    let mut at = at;
                    let played = sh.frames_played();
                    match sh.frames_to_boundary(played) {
                        // A's remaining frames, then the swap, then B's.
                        Some(0) => {
                            if let Some(g) = sh.cross_boundary_now() {
                                at = g;
                            }
                        }
                        Some(remaining) if (remaining as usize) < want_frames => {
                            let head = remaining as usize;
                            let a_got = sess.cons.pop_frames(scratch, head);
                            sh.account_frames(at, a_got as u64);
                            got = a_got;
                            if a_got == head {
                                // Every frame of A has been handed over. From
                                // the next sample the listener is in B.
                                if let Some(g) = sh.cross_boundary_now() {
                                    at = g;
                                }
                                let tail = want_frames - head;
                                let b_got =
                                    sess.cons.pop_frames(&mut scratch[head * ch..], tail);
                                sh.account_frames(at, b_got as u64);
                                got += b_got;
                            }
                        }
                        _ => {}
                    }
                    if got == 0 {
                        got = sess.cons.pop_frames(scratch, want_frames);
                        sh.account_frames(at, got as u64);
                    }

                    // Drained and the decoder has finished → track over.
                    let ended = sess.decode_done.load(Ordering::Acquire) && sess.cons.is_empty();
                    if got < want_frames {
                        if ended {
                            *guard = None;
                            // The decoder has stopped and the ring is empty.
                            // *Why* it stopped is the decoder's to say: it
                            // recorded either a clean end of source or a
                            // failure before it left. The drain decides only
                            // when — and "when" is not now: the device still
                            // holds what it was last given.
                            if sh.decode_eof_for(sh.decode_generation()) {
                                sh.begin_drain();
                            } else {
                                // Stopped without reaching the end and without
                                // naming a reason. The session is over and it
                                // did not finish.
                                sh.fail(at, fault::SESSION_THREAD_FAILED);
                            }
                        } else if sh.priming.load(Ordering::Relaxed) {
                            // Still filling after a seek or a fresh start: the
                            // gap is the one the listener asked for — up to a
                            // point. Past the bound the decoder is not slow,
                            // it is not coming, and the silence gets a name.
                            let n = sh
                                .priming_frames
                                .fetch_add(want_frames as u64, Ordering::Relaxed)
                                + want_frames as u64;
                            let rate = sh.out_rate.load(Ordering::Relaxed) as u64;
                            if rate > 0 && n > rate * PRIMING_LIMIT_SECS {
                                sh.priming.store(false, Ordering::Relaxed);
                                sh.fail(at, fault::PRIMING_TIMEOUT);
                            }
                        } else {
                            // Short buffer with the decoder still running: the
                            // caller is about to pad with silence and the device
                            // will play it. Not a boundary, not a pause — a
                            // dropout, and the end of this session's claim.
                            sh.underruns.fetch_add(1, Ordering::Relaxed);
                            sh.revoke(at, fault::UNDERRUN);
                        }
                    } else {
                        // A full buffer means the ring has caught up; anything
                        // short from here is genuine starvation.
                        sh.priming.store(false, Ordering::Relaxed);
                        sh.priming_frames.store(0, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                // A miss that coincides with a pause is the UI thread holding
                // the slot to install a paused session. That is the hand-off
                // working, not a dropout, and counting it turned every paused
                // restart into a revoked exactness claim.
                if !sh.paused.load(Ordering::Acquire) {
                    sh.lock_misses.fetch_add(1, Ordering::Relaxed);
                    // During priming the UI thread is installing a session and
                    // holding the slot for an instant; that is expected and
                    // inaudible. Outside it, the device just played silence.
                    if !sh.priming.load(Ordering::Relaxed) {
                        sh.revoke(sh.generation(), fault::CALLBACK_LOCK_MISS);
                    }
                }
            }
        }
    }

    // With no session installed and a drain in progress, these calls are the
    // device playing out what it already holds. Counting them is what turns
    // "the ring is empty" into "the listener has heard the whole track".
    if sh.session.try_lock().map(|g| g.is_none()).unwrap_or(false) {
        sh.advance_drain(want_frames);
    }

    // A DoP session's payload is raw DSD bits, not a waveform — handing them
    // to the analyser would draw noise.
    if !sh.dop_active.load(Ordering::Relaxed) {
        tap.feed_canonical(&scratch[..got * ch]);
    }
    // Accounted above, per generation, as each part of the buffer was taken.
    got
}
// ---------------------------------------------------------------------------
// Stream facade over the platform backends
// ---------------------------------------------------------------------------

enum Backend {
    #[cfg(not(windows))]
    Cpal(#[allow(dead_code)] cpal_out::Handle),
    #[cfg(windows)]
    Wasapi(#[allow(dead_code)] wasapi_out::Handle),
    /// No device at all.
    ///
    /// Exists so a test can hold the *public* `BpStream` and call the adapter
    /// a caller actually calls. Every other variant owns a device handle, so
    /// without this the only thing reachable from a test is the private helper
    /// underneath — and a delegation that is never executed is a delegation
    /// that can be wrong.
    #[cfg(test)]
    Silent,
}

/// Why an output stream could not be opened.
///
/// The distinction is load-bearing, not cosmetic: it decides whether the
/// fallback ladder may advance. Exactly one variant means "a different format
/// on this endpoint could work", and only that one advances anything.
///
/// The three-variant version this replaces put a genuine format rejection, a
/// device held by another process, a failed `CoCreateInstance` and a thread
/// that would not spawn into one `Device` bucket, so a COM failure looked
/// exactly like a DAC that could not take 352.8 kHz and produced a silent
/// downgrade instead of a diagnosable error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenError {
    /// The request contradicts itself — no device could satisfy it.
    /// `MOOSIK_BP_FORMAT=16i` cannot carry a DoP word on any device ever made.
    Config(String),
    /// A synchronous source failure: opening, parsing, or building the reader.
    /// Every route reads the same file, so another one reaches the same
    /// failure with a less useful message.
    Source(String),
    /// The container refused to seek. Not a device problem and not a licence
    /// to play the track from somewhere else.
    Seek(String),
    /// **The only variant that may advance a ladder.** The endpoint answered
    /// the format query and said no. A different format, or a different rung,
    /// may still be accepted by this same working device.
    DeviceFormat(String),
    /// The endpoint itself is unusable: unplugged, missing, already held
    /// exclusively by another process, or with exclusive mode disabled. The
    /// device is not answering format questions, so trying more formats on it
    /// is pointless.
    DeviceUnavailable(String),
    /// Something inside our own backend failed: COM initialisation, a thread
    /// that would not spawn, a buffer query, a driver call. The next rung runs
    /// the same code and fails the same way.
    Backend(String),
}

impl OpenError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    pub fn source(msg: impl Into<String>) -> Self {
        Self::Source(msg.into())
    }
    pub fn seek(msg: impl Into<String>) -> Self {
        Self::Seek(msg.into())
    }
    pub fn device_format(msg: impl Into<String>) -> Self {
        Self::DeviceFormat(msg.into())
    }
    pub fn device_unavailable(msg: impl Into<String>) -> Self {
        Self::DeviceUnavailable(msg.into())
    }
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }

    /// Whether *this endpoint rejected this format*, so another format could
    /// plausibly be accepted. The single question that may advance a ladder.
    pub fn is_device_limitation(&self) -> bool {
        matches!(self, Self::DeviceFormat(_))
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Config(m)
            | Self::Source(m)
            | Self::Seek(m)
            | Self::DeviceFormat(m)
            | Self::DeviceUnavailable(m)
            | Self::Backend(m) => m,
        }
    }

    /// The session-state reason this failure publishes. One mapping, so the
    /// badge, the log and the routing decision cannot disagree about what
    /// happened.
    pub fn to_reason(&self) -> state::FailureReason {
        use state::FailureReason as R;
        match self {
            Self::Config(m) => R::Configuration(m.clone()),
            Self::Source(m) => R::SourceOpen(m.clone()),
            Self::Seek(m) => R::Seek(m.clone()),
            Self::DeviceFormat(m) => R::DeviceFormatUnsupported(m.clone()),
            Self::DeviceUnavailable(m) => R::DeviceUnavailable(m.clone()),
            Self::Backend(m) => R::BackendDead(m.clone()),
        }
    }

    /// A short kind label for logs.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) => "configuration",
            Self::Source(_) => "source",
            Self::Seek(_) => "seek",
            Self::DeviceFormat(_) => "device format",
            Self::DeviceUnavailable(_) => "device unavailable",
            Self::Backend(_) => "backend",
        }
    }
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(m) => write!(f, "configuration error: {m}"),
            Self::Backend(m) => write!(f, "output backend: {m}"),
            Self::Source(m)
            | Self::Seek(m)
            | Self::DeviceFormat(m)
            | Self::DeviceUnavailable(m) => {
                write!(f, "{m}")
            }
        }
    }
}

impl From<state::FailureReason> for OpenError {
    fn from(r: state::FailureReason) -> Self {
        use state::FailureReason as R;
        match r {
            R::SourceOpen(m) | R::SourceParse(m) | R::SourceRead(m) | R::Decode(m) => {
                Self::Source(m)
            }
            R::Seek(m) => Self::Seek(m),
            R::Configuration(m) | R::UnsupportedExactRepresentation(m) => Self::Config(m),
            R::DeviceFormatUnsupported(m) => Self::DeviceFormat(m),
            R::DeviceUnavailable(m) => Self::DeviceUnavailable(m),
            R::BackendReset(m)
            | R::BackendWrite(m)
            | R::BackendDead(m)
            | R::SessionThreadStart(m)
            | R::IntegrityFault(m) => Self::Backend(m),
        }
    }
}
/// What the ring will carry, and what had to be done to get it there.
///
/// A stream is opened for a *plan*, not just a source: a native integer source
/// and a converted Float32 one can both end up in a 32-bit integer device
/// format, and they are not the same claim, so they never share a stream.
///
/// There is deliberately **one** conversion rung, not two. A "value-exact" and
/// a "processed" rung both request identical `I32` output, so they were never
/// two device negotiations — the first always won, and the runtime guard then
/// faulted any ordinary off-grid file that reached it. Value-exactness is a
/// property of the *material*, discovered while it plays; it is a label on the
/// fidelity, never a route to negotiate for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PayloadPlan {
    /// The decoded representation goes to the device untouched.
    Identity,
    /// Float32 rewritten as Q1.31.
    ///
    /// Exact wherever the sample lands on the lattice, deterministically
    /// rounded where it does not, and labelled Processed until a complete scan
    /// and the running conversion both agree it was exact throughout.
    Q31,
}

/// Walk the ladder for `source` under `policy`, opening with `open`.
///
/// Pure apart from `open`, so the routing rule can be exercised against a
/// device that does not exist — which is the only way to test the rule that
/// matters: an endpoint that accepts 32-bit integer and rejects 32-bit float
/// is the ordinary case, and it is the case that used to fault.
///
/// Exactly one classification advances the ladder. Everything else stops where
/// it happened, so a configuration error, an unreadable file or a failure
/// inside our own backend is reported rather than answered with a lossier
/// route nobody asked for.
pub fn walk_ladder<T>(
    source: PcmKind,
    policy: state::OutputPolicy,
    mut open: impl FnMut(PayloadPlan) -> Result<T, OpenError>,
) -> Result<T, OpenError> {
    let mut last: Option<OpenError> = None;
    for plan in plan_ladder(source, policy) {
        match open(plan) {
            Ok(s) => return Ok(s),
            Err(e) if e.is_device_limitation() => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        OpenError::config(
            "no route was attempted: this policy permits no conversion for this source",
        )
    }))
}
/// The routes to try for `source`, best first, under `policy`.
///
/// Pure, so the ladder can be checked without a device. Every entry is a route
/// the user has permitted; the caller stops at the first one a device accepts,
/// and only a *device* rejection advances it.
///
/// The raw exact candidate set is untouched by any of this:
/// `exact_candidates(Float32)` remains `[32f]` and `Writer::new(Float32, I32)`
/// remains impossible. A Float32 source reaches an integer endpoint only by
/// carrying an explicit, named conversion — never by widening the definition of
/// exact.
pub fn plan_ladder(source: PcmKind, policy: state::OutputPolicy) -> Vec<PayloadPlan> {
    // A 64-bit float source has no exact route to any device format that
    // exists, so offering the identity rung first would spend an open on a
    // configuration error that stops the ladder. Under a policy that permits
    // processing it goes straight to the conversion; under Strict it gets the
    // identity rung precisely so the attempt fails and says why.
    if matches!(source, PcmKind::Float64) {
        return if policy.allows_processed() {
            vec![PayloadPlan::Q31]
        } else {
            vec![PayloadPlan::Identity]
        };
    }
    // HQ means "deliberately use the conversion". It does not try the raw
    // route first and it does not chase a claim it has already declined.
    if matches!(source, PcmKind::Float32) && policy == state::OutputPolicy::HqProcessed {
        return vec![PayloadPlan::Q31];
    }
    let mut ladder = vec![PayloadPlan::Identity];
    if matches!(source, PcmKind::Float32) && policy.allows_processed() {
        ladder.push(PayloadPlan::Q31);
    }
    ladder
}

impl PayloadPlan {
    fn code(self) -> u8 {
        match self {
            PayloadPlan::Identity => 0,
            PayloadPlan::Q31 => 1,
        }
    }
    fn from_code(c: u8) -> Self {
        match c {
            1 => PayloadPlan::Q31,
            _ => PayloadPlan::Identity,
        }
    }

    /// The device-side family this plan produces, which is what the format
    /// policy must be asked about — not the source's own family.
    pub fn device_kind(self, source: PcmKind) -> PcmKind {
        match self {
            PayloadPlan::Identity => source,
            // Both float families reach the device as 32-bit integer. A
            // `Float64` source has already been narrowed to `f32` on its way
            // into the canonical buffer, which is a processing step in its own
            // right and is labelled as one.
            PayloadPlan::Q31 => {
                PcmKind::Integer { valid_bits: 32 }
            }
        }
    }

    /// What this route does to `source`.
    ///
    /// The source matters, and was not being asked for. A 64-bit float source
    /// takes this route too, and was described as "Float32 → Q1.31" — the
    /// second half of what happened, with the narrowing to `f32` that precedes
    /// it left out. That narrowing is where the numbers change.
    pub fn describe(self, source: PcmKind) -> TransformDescription {
        match self {
            PayloadPlan::Identity => TransformDescription::Identity,
            // What the route *is*, always. A stream is never opened on the
            // strength of a claim it has not yet earned, so the honest
            // description at open time is the lossy one. The value-exact label
            // is applied later, to the fidelity, if the whole-track scan and
            // the runtime observation both allow it — and it never changes the
            // stream key, because it never changes the route.
            PayloadPlan::Q31 if matches!(source, PcmKind::Float64) => {
                TransformDescription::Float64ToQ31Processed
            }
            PayloadPlan::Q31 => TransformDescription::FloatToQ31Processed,
        }
    }
}
pub struct BpStream {
    _backend: Backend,
    shared: Arc<Shared>,
    pub sample_rate: u32,
    pub channels: u16,
    /// What the backend actually agreed to, read back from the backend.
    pub output: OutputFormat,
    pub transport: Transport,
    /// Everything a following track must match before this stream may carry it.
    key: StreamKey,
    /// Device the stream was requested on (None = system default). Kept for
    /// display; reuse compares the *resolved* endpoint in `key`, because
    /// "system default" can move to another DAC underneath an open stream.
    pub requested_device: Option<String>,
    /// What the ring carries and how it got there.
    pub plan: PayloadPlan,
    /// True if this stream was negotiated for DoP (DSD): format selection was
    /// restricted to integer PCM of at least 24 bits, the ring carries bare
    /// payloads, and the render thread owns the marker phase.
    pub dop: bool,
}
impl BpStream {
    /// A stream with no device behind it, for testing the evidence adapter.
    ///
    /// Everything the adapter touches is `shared`; the rest is filled with
    /// values that are never read on that path.
    #[cfg(test)]
    pub(crate) fn for_evidence_test(shared: Arc<Shared>, plan: PayloadPlan) -> Self {
        let endpoint = state::EndpointIdentity::from_name("test");
        let output = OutputFormat {
            endpoint: endpoint.clone(),
            sample_rate: 48_000,
            channels: 2,
            layout: ChannelLayout::UNSPECIFIED,
            container_bits: 32,
            valid_bits: 24,
            integer: true,
            buffer_frames: 480,
            label: "24i/32 excl".into(),
        };
        let transport = Transport::WasapiExclusivePcm {
            endpoint: endpoint.clone(),
        };
        BpStream {
            _backend: Backend::Silent,
            shared,
            sample_rate: 48_000,
            channels: 2,
            output,
            transport: transport.clone(),
            key: StreamKey {
                endpoint,
                transport,
                sample_rate: 48_000,
                channels: 2,
                layout: ChannelLayout::UNSPECIFIED,
                source_kind: PcmKind::Integer { valid_bits: 24 },
                payload_kind: PcmKind::Integer { valid_bits: 24 },
                transform: state::TransformDescription::Identity,
                out_container_bits: 32,
                out_valid_bits: 24,
                out_integer: true,
                dop: false,
            },
            requested_device: None,
            plan,
            dop: false,
        }
    }

    /// Open an output stream on `device_name` (None = system default) for
    /// exactly `source`.
    ///
    /// The format is chosen from `source.kind` through the exactness matrix in
    /// [`format::exact_candidates`], which has no last-chance candidate: if no
    /// device format can carry this source without losing a bit, the open
    /// fails rather than negotiating a narrower one. `dop` replaces the source
    /// family with the DoP carrier's requirement — an integer container of at
    /// least 24 bits, so a DoP-aware DAC sees the literal marker bytes rather
    /// than a float encoding or a truncation of them.
    pub fn open(
        device_name: Option<&str>,
        source: SourceFormat,
        plan: PayloadPlan,
        dop: bool,
        sample_buf: SampleBuf,
        stereo_buf: StereoBuf,
    ) -> Result<Self, OpenError> {
        let sample_rate = source.sample_rate;
        let channels = source.channels;
        // The device is negotiated for what the ring will *deliver*. A Float32
        // source on a Q31 plan reaches the driver as 32-bit integer, and asking
        // the format policy about the source instead of the payload is how
        // `I32` would end up inside the raw-Float32 exact set — which it must
        // never be.
        let device_kind = plan.device_kind(source.kind);
        let device_format = SourceFormat { kind: device_kind, ..source };

        // Resolved here, before a device is enumerated or opened, so a request
        // that cannot be satisfied by anything fails as a configuration error
        // rather than as an apparent device limitation.
        #[cfg(windows)]
        let order = wasapi_out::resolve_order(device_kind, dop)?;
        #[cfg(not(windows))]
        format::exact_candidates(device_kind, dop)
            .map(|_| ())
            .map_err(OpenError::config)?;

        let shared = Arc::new(Shared::new());
        shared.dop_active.store(dop, Ordering::Relaxed);
        shared.plan.store(plan.code(), Ordering::Relaxed);
        // The analyser reads what the ring carries, which after a Q31
        // conversion is integer and not float.
        let tap = SpectrumTap::new(channels, device_kind, sample_buf, stereo_buf);

        #[cfg(not(windows))]
        let (handle, output) = cpal_out::open(
            device_name, device_format, dop, Arc::clone(&shared), tap)?;
        #[cfg(not(windows))]
        let (backend, transport) = (
            Backend::Cpal(handle),
            Transport::CpalDirect { endpoint: output.endpoint.clone() },
        );

        #[cfg(windows)]
        let (handle, output) = wasapi_out::open(
            device_name, device_format, order, dop, Arc::clone(&shared), tap)?;
        #[cfg(windows)]
        let (backend, transport) = (
            Backend::Wasapi(handle),
            if dop {
                Transport::WasapiExclusiveDop { endpoint: output.endpoint.clone() }
            } else {
                Transport::WasapiExclusivePcm { endpoint: output.endpoint.clone() }
            },
        );

        shared.out_valid_bits.store(output.valid_bits as u32, Ordering::Relaxed);
        shared.out_integer.store(output.integer, Ordering::Relaxed);
        shared.out_rate.store(output.sample_rate, Ordering::Relaxed);
        shared
            .out_buffer_frames
            .store(output.buffer_frames, Ordering::Relaxed);

        let key = StreamKey {
            endpoint: output.endpoint.clone(),
            transport: transport.clone(),
            sample_rate,
            channels,
            layout: source.layout,
            source_kind: source.kind,
            payload_kind: device_kind,
            transform: plan.describe(source.kind),
            out_container_bits: output.container_bits,
            out_valid_bits: output.valid_bits,
            out_integer: output.integer,
            dop,
        };

        Ok(BpStream {
            _backend: backend,
            shared,
            sample_rate,
            channels,
            output,
            transport,
            key,
            plan,
            requested_device: device_name.map(str::to_owned),
            dop,
        })
    }

    /// Whether this open stream can carry `next` without reopening — and
    /// without narrowing anything. See [`StreamKey::can_carry`].
    pub fn can_carry(&self, next: &StreamKey) -> bool { self.key.can_carry(next) }

    /// Build the key a track described by `source` would need in order to ride
    /// this stream, on the endpoint the caller has **independently resolved**.
    ///
    /// `target` is not read from `self`. Taking it from the open stream is what
    /// made a changed Windows default endpoint invisible: the key was built
    /// from `self.output.endpoint` and then compared back to the same stream,
    /// so the comparison could only ever succeed. The caller resolves the
    /// endpoint the *next* track would open on and passes it in.
    pub fn key_for(&self, target: &EndpointIdentity, source: &SourceFormat) -> StreamKey {
        StreamKey {
            endpoint: target.clone(),
            transport: self.transport.clone(),
            sample_rate: source.sample_rate,
            channels: source.channels,
            layout: source.layout,
            source_kind: source.kind,
            payload_kind: self.plan.device_kind(source.kind),
            transform: self.plan.describe(source.kind),
            out_container_bits: self.output.container_bits,
            out_valid_bits: self.output.valid_bits,
            out_integer: self.output.integer,
            dop: self.dop,
        }
    }


    /// Everything about `at`, read by `at`'s own stamps.
    ///
    /// The one call the UI makes. There is deliberately no "is this still
    /// current" question in it: the answer would be stale the instant it was
    /// given, and every field is keyed by the generation asked for, so a
    /// boundary crossing during this call changes nothing about what comes
    /// back.
    pub fn evidence_at(&self, at: u64) -> Evidence {
        self.shared.evidence_at(at)
    }

    /// Which audio generation the evidence above belongs to.
    ///
    /// The render thread crosses a gapless boundary at the sample, so this
    /// moves before the UI has folded anything. Published so that a poll which
    /// runs ahead of the fold can tell that what it is reading is about a
    /// track the UI does not yet know is playing, and leave it for the tick
    /// that does — rather than charging the incoming track's dropout to the
    /// one that just finished cleanly.
    pub fn audio_generation(&self) -> u64 {
        self.shared.generation()
    }

    // `fault_code`, `revoked_code` and `q31_off_grid` were here.
    //
    // They are gone rather than merely unused. Each answered about "whatever
    // is playing now", so any two of them called together could describe two
    // different tracks, and no check placed before the calls could be about
    // what happened between them. `evidence_at` above is the whole of what
    // they said, read by one generation's stamps — and leaving the parts
    // behind would leave the pattern available to be rebuilt.

    /// Render calls that could not reach the session slot in time.
    pub fn lock_misses(&self) -> u64 { self.shared.lock_misses.load(Ordering::Relaxed) }

    /// Whether the backend thread has ended. A dead backend may never be
    /// reused, whatever the session state says.
    pub fn backend_dead(&self) -> bool {
        self.shared.backend_dead.load(Ordering::Acquire)
    }

    /// Start playing a prepared decode pipeline. Replaces any running session;
    /// the old decode thread is signalled to stop and exits on its own.
    ///
    /// There is no volume argument. An exact route's software gain is unity by
    /// construction — there is no multiply anywhere between the decoder and
    /// the device — so a parameter for it would only be a way to express a
    /// state the route cannot be in.
    pub fn start(&self, prep: Prepared) -> Result<(), state::FailureReason> {
        // The slot is held for the whole of this, and the generation turns
        // inside it.
        //
        // Bumping first and installing afterwards left a window in which the
        // outgoing session sat in the slot while the clock already read the
        // incoming generation — the window the native backends had, and the
        // reason the render thread could not simply trust the clock. Holding
        // the slot closes it, and costs nothing: the render side only ever
        // `try_lock`s, so a missed callback here is a track change, which is
        // what `priming` exists to forgive.
        let mut slot = match self.shared.session.lock() {
            Ok(g) => g,
            // A poisoned slot means a previous holder panicked while it was
            // open. There is no safe session behind it.
            Err(_) => {
                return Err(state::FailureReason::SessionThreadStart(
                    "the output session slot was left poisoned by a panic".into(),
                ));
            }
        };
        // A new logical session. The bump retires the previous session's
        // fault and stamps every producer spawned below, so a decode thread
        // still unwinding from the last track cannot fault this one.
        let generation = self.shared.begin_generation();
        let ring_frames = (self.sample_rate as usize).max(16_384); // ~1 s
        let (prod, cons) = channel_ring::<u32>(self.channels.max(1) as usize, ring_frames);
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        // A fresh session supersedes any gapless queue/boundaries from the last.
        if let Ok(mut g) = self.shared.next.lock() { *g = None; }
        if let Ok(mut b) = self.shared.boundaries.lock() { b.clear(); }

        // The session is installed only after the thread that feeds it exists.
        // `.spawn(..).ok()` discarded the failure, so a thread that could not
        // start left an installed session priming forever: silence, with the
        // diamond lit, and no fault anywhere to explain it.
        let guard = SessionGuard::new(
            Arc::clone(&done),
            Arc::clone(&self.shared),
            generation,
            Arc::clone(&stop),
        );
        {
            let stop = Arc::clone(&stop);
            let shared = Arc::clone(&self.shared);
            std::thread::Builder::new()
                .name("bp-decode".into())
                .spawn(move || {
                    let _guard = guard;
                    let logical = _guard.logical_generation();
                    decode_loop(prep, prod, _guard.done(), stop, shared, generation, logical)
                })
                .map_err(|e| state::FailureReason::SessionThreadStart(
                    format!("decode thread could not start: {e}")))?;
        }

        // `begin_generation` above already stamped and zeroed it.
        // old Session drop signals its thread
        *slot = Some(Session { generation, cons, decode_done: done, stop });
        Ok(())
    }

    /// Start a DoP (DSD) session: streams `stream`'s bare 16-bit DSD payloads
    /// into the ring, where the render thread marks them into carrier frames.
    /// The spectrum tap is skipped for the session's whole lifetime
    /// (`Shared::dop_active`, set at `open()`), and there is no gain anywhere
    /// on the path to reset.
    pub fn start_dop(&self, stream: DopFileStream) -> Result<(), state::FailureReason> {
        debug_assert!(self.dop, "start_dop called on a stream not opened with dop=true");
        // Held for the whole of this — see `start`.
        let mut slot = match self.shared.session.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(state::FailureReason::SessionThreadStart(
                    "the output session slot was left poisoned by a panic".into(),
                ));
            }
        };
        let generation = self.shared.begin_generation();
        let ring_frames = (self.sample_rate as usize).max(16_384); // ~1 s
        let (prod, cons) = channel_ring::<u32>(self.channels.max(1) as usize, ring_frames);
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        if let Ok(mut g) = self.shared.next.lock() { *g = None; }
        if let Ok(mut g) = self.shared.next_dop.lock() { *g = None; }
        if let Ok(mut b) = self.shared.boundaries.lock() { b.clear(); }

        let guard = SessionGuard::new(
            Arc::clone(&done),
            Arc::clone(&self.shared),
            generation,
            Arc::clone(&stop),
        );
        {
            let stop = Arc::clone(&stop);
            let shared = Arc::clone(&self.shared);
            std::thread::Builder::new()
                .name("bp-dop-decode".into())
                .spawn(move || {
                    let _guard = guard;
                    let logical = _guard.logical_generation();
                    dop_decode_loop(stream, prod, _guard.done(), stop, shared, generation, logical)
                })
                .map_err(|e| state::FailureReason::SessionThreadStart(
                    format!("DoP decode thread could not start: {e}")))?;
        }

        // `begin_generation` above already stamped and zeroed it.
        *slot = Some(Session { generation, cons, decode_done: done, stop });
        Ok(())
    }

    /// Queue the next track for gapless continuation. The caller guarantees the
    /// rate and channel count match the open stream (a mismatch would force a
    /// device re-open, so it isn't gapless). The decode thread swaps to it the
    /// instant the current file ends, without a gap or a `finished` signal.
    pub fn queue_next(&self, prep: Prepared) {
        if let Ok(mut g) = self.shared.next.lock() { *g = Some(prep); }
    }

    /// Queue the next DSD track for gapless DoP continuation. The caller
    /// guarantees the carrier rate and channel count match the open stream.
    /// The DoP decode loop swaps to it (carrying the marker phase) when the
    /// current file ends — no device re-open, no gap.
    pub fn queue_next_dop(&self, stream: DopFileStream) {
        if let Ok(mut g) = self.shared.next_dop.lock() { *g = Some(stream); }
    }

    /// Drop a queued-but-not-yet-started gapless track (e.g. the user changed
    /// what plays next). No effect once the decode thread has already begun it.
    /// Clears both the PCM and DoP gapless queues.
    pub fn clear_next(&self) {
        if let Ok(mut g) = self.shared.next.lock() { *g = None; }
        if let Ok(mut g) = self.shared.next_dop.lock() { *g = None; }
    }

    /// If the device has finished playing a gapless track, pop that boundary and
    /// return the cumulative played time at it (so the UI can roll over). The
    /// value is frame-exact — no reliance on possibly-wrong metadata duration.
    /// Metadata for a boundary the *device* has already crossed.
    ///
    /// This used to do the crossing: it compared `frames_played` against the
    /// queue and then installed the playback generation. That put the moment
    /// the audio changes in the hands of a thread that polls on a frame tick,
    /// so up to sixteen milliseconds of the new track played under the old
    /// track's generation — and a callback deep enough to span the boundary
    /// played *all* of it under the old one, with no instant at which anything
    /// could be said about the new.
    ///
    /// The render thread crosses now, at the sample. This reports what was
    /// crossed, and only once the device is past it.
    pub fn take_reached_boundary(&self) -> Option<(Duration, Boundary)> {
        let front = self.shared.take_boundary()?;
        let at = Duration::from_secs_f64(front.frames as f64 / self.sample_rate.max(1) as f64);
        Some((at, front))
    }

    /// Set the policy this stream was opened under. Read back by the decode
    /// thread when it assembles a gapless boundary, so the label a track gets
    /// is the one the route was actually negotiated for.
    pub fn set_policy(&self, policy: state::OutputPolicy) {
        self.shared.policy.store(policy.code(), Ordering::Relaxed);
    }

    /// `Release`, so that the renderer's `Acquire` re-read after it has the
    /// session slot cannot see a stale `false` — see `render_frames`.
    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Release); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Release); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    /// How this session ended. The only thing that may advance a playlist is
    /// `Completion::CleanEof`.
    /// How this session ended — the only thing that may advance a playlist is
    /// `Completion::CleanEof`.
    ///
    /// Polled from the UI thread, which is why the wall clock is checked here.
    pub fn completion(&self) -> Completion {
        if self.shared.drain_expired() {
            self.shared.expire_drain();
        }
        self.shared.completion()
    }

    /// Sample-accurate elapsed time within the current session.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played();
        Duration::from_secs_f64(frames as f64 / self.sample_rate.max(1) as f64)
    }

    /// Short description for the status line, e.g. "192 kHz · stereo · 24i excl".
    pub fn describe(&self) -> String {
        format!("{} · {} · {}",
                fmt_khz(self.sample_rate),
                format::describe_channels(self.channels),
                self.output.label)
    }

    /// The resolved device name.
    pub fn device_name(&self) -> &str { &self.output.endpoint.display }

    /// How many times the render loop has come up short mid-track since this
    /// stream opened. Non-zero means the DAC has played silence it shouldn't
    /// have — see `Shared::underruns`.
    pub fn underruns(&self) -> u64 {
        self.shared.underruns.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Spectrum tap
// ---------------------------------------------------------------------------

/// Mirrors SpectrumSource's batching: accumulate locally, flush to the shared
/// buffers with try_lock once per ~512 samples so the spectrum stays in sync
/// with what the device is actually playing.
struct SpectrumTap {
    channels: u16,
    /// How to read a canonical payload as a number. The tap is the one place
    /// the exact path is deliberately left: these floats go to the analyser
    /// and never come back.
    kind: PcmKind,
    ch_idx: u16,
    pending_l: f32,
    frame_sum: f32,
    sample_buf: SampleBuf,
    stereo_buf: StereoBuf,
    batch: Vec<f32>,
    stereo_batch: Vec<[f32; 2]>,
}

const TAP_BATCH: usize = 512;

impl SpectrumTap {
    fn new(channels: u16, kind: PcmKind, sample_buf: SampleBuf, stereo_buf: StereoBuf) -> Self {
        // Reserve the caps here, on the thread that builds the tap, so no
        // realtime flush ever has to.
        //
        // `reserve` takes an amount *additional to the current length*, not a
        // target capacity. Asking for `CAP - capacity()` therefore did nothing
        // whenever the existing capacity already covered `len + that`, which
        // is the ordinary case for a buffer reused across streams: a tap built
        // over a half-sized buffer left it half-sized, and the first flush
        // past the halfway mark grew it — on the render thread.
        if let Ok(mut v) = sample_buf.lock() {
            let want = MONO_CAP.saturating_sub(v.len());
            v.reserve(want);
            debug_assert!(v.capacity() >= MONO_CAP);
        }
        if let Ok(mut v) = stereo_buf.lock() {
            let want = STEREO_CAP.saturating_sub(v.len());
            v.reserve(want);
            debug_assert!(v.capacity() >= STEREO_CAP);
        }
        Self {
            channels,
            kind,
            ch_idx: 0,
            pending_l: 0.0,
            frame_sum: 0.0,
            sample_buf,
            stereo_buf,
            // One sample may be pushed after the threshold check that would
            // have flushed, so both need room for the check value plus one.
            batch: Vec::with_capacity(TAP_BATCH * 2),
            stereo_batch: Vec::with_capacity(TAP_BATCH * 2),
        }
    }

    /// Observe canonical payloads as floats for the analyser.
    ///
    /// One-way by construction: the conversion reads `canon` and writes only
    /// into the tap's own batches. An `f32` cannot represent a 32-bit integer
    /// payload, so a value that made this trip is no longer exact — which is
    /// fine for a spectrum and would not be fine for anything the device sees.
    fn feed_canonical(&mut self, canon: &[u32]) {
        let ch = self.channels.max(1);
        for &c in canon {
            // Both batches are flushed the moment they are full, so neither can
            // grow past the capacity reserved at construction. Flushing only
            // after the whole slice had been appended meant one oversized
            // callback reallocated both of them — on the render thread.
            if self.batch.len() >= TAP_BATCH || self.stereo_batch.len() >= TAP_BATCH {
                self.flush();
            }
            let f = canon_to_f32(c, self.kind);
            if self.channels == 2 {
                if self.ch_idx == 0 {
                    self.pending_l = f;
                } else {
                    self.stereo_batch.push([self.pending_l, f]);
                }
            }
            // The analyser reads this buffer as one mono stream sampled at the
            // track's rate. Pushing interleaved channels made it a 2× (or N×)
            // sample-and-hold instead: every partial appeared at f/N with a
            // mirror image at Nyquist − f/N, which is why bass showed up again
            // at the top of the display. Average each frame down, exactly as
            // the pre-process decoder does.
            self.frame_sum += f;
            self.ch_idx += 1;
            if self.ch_idx >= ch {
                self.batch.push(self.frame_sum / ch as f32);
                self.frame_sum = 0.0;
                self.ch_idx = 0;
            }
        }
        if self.batch.len() >= TAP_BATCH {
            self.flush();
        }
    }

    /// Hand the batched frames to the analyser without allocating.
    ///
    /// The shared buffers are reserved to their cap at construction, and the
    /// old samples are dropped *before* the new ones are appended, so their
    /// length never exceeds the reserved capacity and `extend_from_slice`
    /// cannot reallocate. Extending first and trimming afterwards briefly
    /// exceeded the cap — which is exactly when `Vec` grows.
    fn flush(&mut self) {
        if !self.batch.is_empty()
            && let Ok(mut v) = self.sample_buf.try_lock()
        {
            let room = MONO_CAP.saturating_sub(self.batch.len());
            if v.len() > room {
                let d = v.len() - room;
                v.drain(0..d);
            }
            v.extend_from_slice(&self.batch);
        }
        self.batch.clear();

        if !self.stereo_batch.is_empty()
            && let Ok(mut v) = self.stereo_buf.try_lock()
        {
            let room = STEREO_CAP.saturating_sub(self.stereo_batch.len());
            if v.len() > room {
                let d = v.len() - room;
                v.drain(0..d);
            }
            v.extend_from_slice(&self.stereo_batch);
        }
        self.stereo_batch.clear();
    }
}
/// How much history the analyser keeps. Reserved up front on both shared
/// buffers so a flush can never grow one.
const MONO_CAP: usize = DEFAULT_FFT_SIZE * 4;
const STEREO_CAP: usize = 8192;

// ---------------------------------------------------------------------------
// Decode thread
// ---------------------------------------------------------------------------

/// What became of a push.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PushOutcome {
    /// Every frame was published.
    Pushed,
    /// A stop was signalled mid-push; the caller returns immediately.
    Stopped,
    /// The data was not a whole number of frames. Nothing was published.
    Invalid,
}

/// Push whole frames into the ring with backpressure, sleeping when it is full.
///
/// A sub-frame remainder is reported rather than spun on. The check used to be
/// a `debug_assert`, so a release build with a partial frame would loop forever
/// pushing zero frames — a silent hang on the decode thread with the render
/// side priming behind it.
fn push_frames_all(data: &[u32], prod: &mut FrameProducer<u32>, stop: &AtomicBool) -> PushOutcome {
    let ch = prod.channels();
    if !data.len().is_multiple_of(ch) {
        return PushOutcome::Invalid;
    }
    let mut off = 0usize;
    while off < data.len() {
        if stop.load(Ordering::Relaxed) {
            return PushOutcome::Stopped;
        }
        match prod.push_frames(&data[off..]) {
            Err(_) => return PushOutcome::Invalid,
            Ok(0) => {
                // The ring holds ~1 s, so this thread spends most of its life
                // here.
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(n) => off += n * ch,
        }
    }
    PushOutcome::Pushed
}

/// Turns a decode thread that ended without finishing into a visible fault.
///
/// A decode loop can stop for three reasons: it finished the track, it was
/// asked to stop, or it panicked. The first two set `done` themselves. The
/// third used to set nothing at all, so the render side kept waiting for
/// samples that would never come — priming forever, silent, still claiming to
/// be exact. This guard runs on unwind as well as on return: if `done` was
/// never set, it sets it and raises a fault, so the waiter is released and the
/// session stops claiming.
struct SessionGuard {
    done: Arc<AtomicBool>,
    shared: Arc<Shared>,
    /// The track the loop is on **now**, not the one it was born on.
    ///
    /// It has to be the current one for two different reasons. A fixed birth
    /// generation is wrong going forward: a decode loop survives gapless
    /// boundaries, so a panic three tracks into a gapless run was stamped with
    /// the generation of the track that opened the stream, found stale, and
    /// discarded — the decoder was gone and nothing anywhere said so. And it
    /// is wrong going backward: a thread still unwinding after its session was
    /// replaced must not fault its successor, which is what a bare "fault
    /// whatever is playing" did.
    ///
    /// Shared with the loop, which stores each rollover into it, so the two
    /// cannot drift.
    generation: Arc<AtomicU64>,
    /// Set when the session was torn down on purpose — a stop, a seek, or a
    /// replacement. Ending early because you were asked to is not a failure,
    /// and reporting it as one made every seek raise a fault.
    stop: Arc<AtomicBool>,
}

impl SessionGuard {
    fn new(
        done: Arc<AtomicBool>,
        shared: Arc<Shared>,
        generation: u64,
        stop: Arc<AtomicBool>,
    ) -> Self {
        SessionGuard {
            done,
            shared,
            generation: Arc::new(AtomicU64::new(generation)),
            stop,
        }
    }
    fn done(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.done)
    }
    /// The cell the decode loop writes each gapless rollover into.
    fn logical_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.generation)
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if self.done.load(Ordering::Acquire) {
            return;
        }
        // Whatever happens, release the render side: it is waiting on this
        // flag and nothing else will ever set it.
        self.done.store(true, Ordering::Release);
        if self.stop.load(Ordering::Acquire) {
            return; // asked to stop; not a failure
        }
        // Ended without finishing and without being asked to — a panic, or a
        // path that forgot to say why. Against the track the loop was on, so a
        // thread still unwinding cannot fault the track that replaced it —
        // and, if that track is queued rather than playing, the fault waits
        // for the device rather than stopping the track the listener is
        // currently hearing.
        self.shared.fault_decoding(
            self.generation.load(Ordering::Acquire),
            fault::SESSION_THREAD_FAILED,
        );
    }
}
fn decode_loop(
    mut prep: Prepared,
    mut prod: FrameProducer<u32>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
    generation: u64,
    // The guard's view of which track this loop is on. Written at every
    // gapless rollover so that a panic is reported against the track that was
    // being decoded rather than the one that opened the stream.
    logical: Arc<AtomicU64>,
) {
    // The generation this loop may fault. It advances at each gapless
    // rollover, so a fault always lands on the track it happened in and never
    // on the one that replaced it.
    let mut generation = generation;
    let ch = prod.channels();
    let mut canon: Vec<u32> = Vec::new();
    // Frames pushed into the ring since `start()`, across every gapless track.
    // A push is FIFO into the ring and then FIFO to the device, so this running
    // total is exactly the play-time boundary between one track and the next.
    let mut frames_pushed: u64 = 0;
    let mut expect = prep.source;
    let plan = PayloadPlan::from_code(shared.plan.load(Ordering::Relaxed));
    let policy = state::OutputPolicy::from_code(shared.policy.load(Ordering::Relaxed));
    let mut guard = q31::Q31Guard::new();

    'track: loop {
        // The packet `prepare` decoded to establish the family is real audio
        // and goes first — the probe costs latency, never samples.
        if !prep.primed.is_empty() {
            canon.clear();
            canon.append(&mut prep.primed);
            if plan == PayloadPlan::Q31 {
                guard.convert(&mut canon, expect.channels);
                // Published so the UI can withhold the value-exact label the
                // moment a sample needs the lossy path, without waiting for
                // the whole-track scan to agree.
                shared.note_off_grid(generation, guard.off_grid());
            }
            match push_frames_all(&canon, &mut prod, &stop) {
                PushOutcome::Pushed => {}
                PushOutcome::Stopped => return,
                PushOutcome::Invalid => {
                    shared.fault_decoding(generation, fault::RING_INVARIANT);
                    break 'track;
                }
            }
            frames_pushed += (canon.len() / ch) as u64;
        }

        // Decode the current file until it ends (or the session is superseded).
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }

            let packet = match prep.format.next_packet() {
                Ok(p) => p,
                Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                // Not an end of file. Symphonia wants the decoder rebuilt;
                // treating it as EOF made the track stop wherever the reset
                // happened, silently, with the session still claiming exact.
                Err(SymError::ResetRequired) => {
                    shared.fault_decoding(generation, fault::SOURCE_FORMAT_CHANGE);
                    break 'track;
                }
                Err(_) => {
                    // A read error mid-track is not a clean end of file either.
                    shared.fault_decoding(generation, fault::SOURCE_READ);
                    break 'track;
                }
            };
            if packet.track_id() != prep.track_id {
                continue;
            }

            let decoded = match prep.decoder.decode(&packet) {
                Ok(d) => d,
                // A damaged packet is missing audio. Skipping it and carrying
                // on is the behaviour of a player that values continuity over
                // truth; this path has already promised the opposite.
                Err(SymError::DecodeError(_)) => {
                    shared.fault_decoding(generation, fault::DECODE_ERROR);
                    break 'track;
                }
                Err(SymError::ResetRequired) => {
                    shared.fault_decoding(generation, fault::SOURCE_FORMAT_CHANGE);
                    break 'track;
                }
                Err(_) => {
                    shared.fault_decoding(generation, fault::DECODE_ERROR);
                    break 'track;
                }
            };

            // Every packet goes through the same validation the probe packet
            // did — rate, channels, layout, family, and the effective precision
            // against the format the device was actually opened for. Nothing is
            // published until all of it passes, so a packet that suddenly needs
            // more bits than the negotiated container holds faults here rather
            // than arriving pre-truncated at the writer.
            canon.clear();
            if let Err(_e) =
                canonicalize_packet(&decoded, Some(&expect), prep.bits_per_sample, &mut canon)
            {
                shared.fault_decoding(generation, fault::SOURCE_FORMAT_CHANGE);
                break 'track;
            }
            // The transform runs here, on the decode thread, after validation
            // and before publication. The value-exact conversion re-checks every
            // sample as it goes: a whole-track scan is a cache, not a licence,
            // and a sample it did not predict must not reach the device.
            if plan == PayloadPlan::Q31 {
                let was_clean = guard.all_on_grid();
                guard.convert(&mut canon, expect.channels);
                // Published so the UI can withhold the value-exact label the
                // moment a sample needs the lossy path, without waiting for
                // the whole-track scan to agree.
                shared.note_off_grid(generation, guard.off_grid());
                // Said once, where it happened. Not a fault — the route was
                // opened as processed and this is that route working — but the
                // exact sample that cost the value-exact label is the first
                // thing anyone asks about afterwards.
                if was_clean
                    && !guard.all_on_grid()
                    && let Some(f) = guard.first_off_grid()
                {
                    crate::mlog!(
                        "bp      Q1.31: frame {} channel {} ({}) needed rounding after {} \
                         clean frames — value-exact is off for this track",
                        f.frame,
                        f.channel,
                        f.reason.describe(),
                        guard.frames_seen()
                    );
                }
            }
            match push_frames_all(&canon, &mut prod, &stop) {
                PushOutcome::Pushed => {}
                PushOutcome::Stopped => return,
                PushOutcome::Invalid => {
                    shared.fault_decoding(generation, fault::RING_INVARIANT);
                    break 'track;
                }
            }
            frames_pushed += (canon.len() / ch) as u64;
        }

        // The current file is exhausted. Continue straight into a queued next
        // track (gapless), else signal completion. The caller has already
        // proven the next track fits the open stream (`StreamKey::can_carry`);
        // it is re-checked here because a queue that outlives its check would
        // otherwise narrow silently at the boundary.
        let next = shared.next.lock().ok().and_then(|mut g| g.take());
        match next {
            Some(next_prep) => {
                if next_prep.source.sample_rate != expect.sample_rate
                    || next_prep.source.channels as usize != ch
                    || next_prep.source.layout != expect.layout
                    || !shared.can_carry_kind(next_prep.source.kind)
                {
                    shared.fault_decoding(generation, fault::SOURCE_FORMAT_CHANGE);
                    break;
                }
                // Everything the UI needs about the new track, assembled
                // here and applied in one step. A new generation begins with
                // it: the track that just ended keeps its own fault, and this
                // one starts clean.
                //
                // The generation advances when the boundary is *pushed*,
                // which leads the device by up to a ring (~1 s). A dropout
                // inside that window is attributed to the incoming track
                // rather than the outgoing one. At a gapless boundary the two
                // are contiguous audio, so the distinction is not observable.
                // The decode clock advances here; the playback clock does not.
                // The device is still inside the outgoing track and will be
                // for up to a full ring.
                //
                // The successor is off the queue and this loop is about to
                // claim the clock for it. A `start()` can land exactly here,
                // which is why the claim below is a compare-exchange from the
                // generation this loop believes it owns rather than an
                // unconditional roll.
                #[cfg(test)]
                window::fire(shared.hook_id, window::Site::Successor);
                let source = state::MediaSource::pcm(next_prep.source);
                let transform = plan.describe(next_prep.source.kind);
                let path = next_prep.path.clone();
                // Claim and publish, as one transaction. It refuses if this
                // thread has been superseded, and there is nothing useful to
                // do about that but stop.
                let Some(rolled) = shared.roll_and_publish(
                    generation,
                    frames_pushed,
                    move |g| Boundary {
                        frames: frames_pushed,
                        generation: g,
                        source,
                        transform,
                        plan,
                        policy,
                        path,
                        odd_tail: false, // PCM has no carrier frame to half-fill
                    },
                ) else {
                    break 'track;
                };
                generation = rolled;
                logical.store(generation, Ordering::Release);
                // A new track is new material, and it gets a new guard. Its
                // evidence slot was cleared by `roll_decode_generation`; the
                // outgoing track's stays readable until the device leaves it,
                // because the outgoing track is what the listener is still
                // hearing.
                guard = q31::Q31Guard::new();
                expect = next_prep.source;
                prep = next_prep;
                continue 'track;
            }
            // Nothing queued: the source is exhausted and nothing went wrong.
            // The one path in this loop that earns an advance.
            None => {
                shared.note_decode_eof(generation);
                break;
            }
        }
    }

    done.store(true, Ordering::Release);
}
/// DoP counterpart of `decode_loop`: pulls the DSD bit stream straight from
/// the file (no symphonia, no sample-format conversion) and pushes bare 16-bit
/// payloads into the ring.
///
/// No marker, and no silence. Both belong to the render thread now.
///
/// The lead-in, the tail and every gap used to be generated here, which meant
/// the *only* silence that carried a valid DoP marker was silence the decoder
/// knew about in advance. A pause, an underrun, the prefill before the first
/// buffer and the drain after the last one all emitted PCM zeros instead —
/// which a DoP DAC reads as loss of lock. Now `DopState` on the render thread
/// generates marked silence wherever payload runs out, so every carrier frame
/// leaving this process is a valid DoP frame whatever the reason for it.
fn dop_decode_loop(
    mut stream: DopFileStream,
    mut prod: FrameProducer<u32>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
    generation: u64,
    // See `decode_loop`: the guard's view of which track is being decoded.
    logical: Arc<AtomicU64>,
) {
    let mut generation = generation;
    let policy = state::OutputPolicy::from_code(shared.policy.load(Ordering::Relaxed));
    const CHUNK_PCM_FRAMES: usize = 4096;
    let ch = prod.channels();
    let mut payload: Vec<u32> = Vec::new();
    // Cumulative PCM frames pushed since start() — the gapless boundary basis,
    // consumed by take_reached_boundary against the device's frames_played.
    let mut frames_pushed: u64 = 0;

    'file: loop {
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            payload.clear();
            let n = match stream.read_payload(CHUNK_PCM_FRAMES, &mut payload) {
                Ok(n) => n,
                Err(_) => {
                    shared.fault_decoding(generation, fault::SOURCE_READ);
                    break 'file;
                }
            };
            if n == 0 {
                break;
            }
            match push_frames_all(&payload, &mut prod, &stop) {
                PushOutcome::Pushed => {}
                PushOutcome::Stopped => return,
                PushOutcome::Invalid => {
                    shared.fault_decoding(generation, fault::RING_INVARIANT);
                    break 'file;
                }
            }
            frames_pushed += n as u64;
        }

        // Current file exhausted — continue into a queued DSD track (gapless),
        // else finish. There is no marker phase to carry: the render thread
        // has been counting frames across the whole stream, so the boundary
        // word alternates for free.
        let next = shared.next_dop.lock().ok().and_then(|mut g| g.take());
        match next {
            Some(next_stream) => {
                let info = next_stream.info().clone();
                // The queue checked this when the track was queued; it is
                // checked again here, against the stream that is actually
                // open, because a queue that outlives its check would
                // otherwise change the layout or run off the end of a
                // truncated file at a point nobody is watching.
                if info.channels as usize != ch
                    || next_stream.carrier_rate() != stream.carrier_rate()
                    || info.layout != stream.info().layout
                {
                    shared.fault_decoding(generation, fault::SOURCE_FORMAT_CHANGE);
                    break;
                }
                if info.is_truncated() {
                    shared.fault_decoding(generation, fault::SOURCE_READ);
                    break;
                }
                // Same claim as the PCM loop, for the same reason: this is
                // where a superseded DoP decoder would otherwise mint a
                // generation newer than the session that replaced it.
                #[cfg(test)]
                window::fire(shared.hook_id, window::Site::Successor);
                let source = state::MediaSource::dsd(
                    SourceFormat {
                        kind: PcmKind::Integer { valid_bits: 1 },
                        sample_rate: info.sample_rate,
                        channels: info.channels as u16,
                        layout: ChannelLayout(info.layout.unwrap_or(0)),
                    },
                    info.rate_label(),
                );
                let path = next_stream.path().to_path_buf();
                let odd_tail = info.total_frames() % 2 == 1;
                let Some(rolled) = shared.roll_and_publish(
                    generation,
                    frames_pushed,
                    move |g| Boundary {
                        frames: frames_pushed,
                        generation: g,
                        source,
                        transform: TransformDescription::Identity,
                        plan: PayloadPlan::Identity,
                        policy,
                        path,
                        odd_tail,
                    },
                ) else {
                    break 'file;
                };
                generation = rolled;
                logical.store(generation, Ordering::Release);
                stream = next_stream;
                continue 'file;
            }
            None => {
                shared.note_decode_eof(generation);
                break;
            }
        }
    }

    done.store(true, Ordering::Release);
}
// ---------------------------------------------------------------------------
// Persisted settings
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
pub struct BpSettings {
    pub enabled: bool,
    /// Selected output device name; None = system default.
    pub device: Option<String>,
    /// Selected ASIO driver for native DSD (Windows, asio-dsd builds).
    /// None = DSD uses DoP. Kept in the settings file on every platform so
    /// the choice survives builds without the feature.
    #[serde(default)]
    pub asio_driver: Option<String>,
    /// Selected ALSA device for native DSD (Linux, alsa-dsd builds).
    /// None = DSD uses DoP. Same cross-platform persistence as asio_driver.
    #[serde(default)]
    pub alsa_dsd_device: Option<String>,
    /// How far the user permits Moosik to go when an exact route is
    /// impossible.
    ///
    /// `serde(default)` so a settings file written by an older build loads
    /// unchanged and lands on the general-purpose default rather than being
    /// rejected or silently reinterpreted.
    #[serde(default)]
    pub policy: state::OutputPolicy,
}

fn settings_path(dir: &Path) -> PathBuf { dir.join("bitperfect.json") }

pub fn load_settings(dir: &Path) -> BpSettings {
    std::fs::read_to_string(settings_path(dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_settings(dir: &Path, s: &BpSettings) {
    let _ = std::fs::create_dir_all(dir);
    if let Ok(json) = serde_json::to_string(s) {
        let _ = std::fs::write(settings_path(dir), json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use format::{DevFmt, DopState, Writer};

    // -----------------------------------------------------------------------
    // Render-side behaviour
    // -----------------------------------------------------------------------

    /// A `Shared` with one session holding `queued` frames, ready to render.
    /// Returns the producer too — dropping it would not end the track, only
    /// `decode_done` does that.
    fn staged(queued: &[u32], channels: usize) -> (Shared, FrameProducer<u32>, Arc<AtomicBool>) {
        let (mut prod, cons) = channel_ring::<u32>(channels, 1024);
        assert_eq!(prod.push_frames(queued).unwrap(), queued.len() / channels);
        let done = Arc::new(AtomicBool::new(false));
        let sh = Shared::new();
        *sh.session.lock().unwrap() = Some(Session {
            generation: sh.generation(),
            cons,
            decode_done: Arc::clone(&done),
            stop: Arc::new(AtomicBool::new(false)),
        });
        (sh, prod, done)
    }

    fn tap() -> SpectrumTap {
        SpectrumTap::new(
            2,
            PcmKind::Integer { valid_bits: 16 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        )
    }

    /// The three ways a render call can end, and the one that means the DAC
    /// played silence it should not have. Without this distinction a dropout
    /// and a track boundary look identical from the outside, which is how a
    /// pop stays invisible to every test in the suite.
    #[test]
    fn a_short_buffer_counts_as_a_dropout_only_when_the_track_is_still_running() {
        let mut scratch = vec![0u32; 256];
        let mut t = tap();

        // Enough frames: no dropout, track still open.
        let (sh, _prod, _done) = staged(&[7; 64], 2);
        assert_eq!(render_frames(&sh, &mut scratch, &mut t, 2, 32), 32);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.fault(), fault::NONE);
        assert_eq!(sh.completion(), Completion::Running);

        // Short, decoder still working → dropout, session stays open, and the
        // session's exactness claim is over.
        let (sh, _prod, _done) = staged(&[7; 16], 2);
        assert_eq!(render_frames(&sh, &mut scratch, &mut t, 2, 32), 8);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 1);
        assert_eq!(sh.revoked(), fault::UNDERRUN);
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "a dropout is evidence, not a reason the session stopped"
        );
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "a dropout does not end the session"
        );
        assert!(sh.session.lock().unwrap().is_some());

        // Short because the track genuinely ended → not a dropout, no fault,
        // and the one terminal state that may advance a playlist.
        let (sh, _prod, done) = staged(&[7; 16], 2);
        sh.note_decode_eof(sh.generation());
        done.store(true, Ordering::Release);
        assert_eq!(render_frames(&sh, &mut scratch, &mut t, 2, 32), 8);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.fault(), fault::NONE);
        assert_eq!(sh.completion(), Completion::CleanEof);
        assert!(sh.completion().may_advance());
        assert!(sh.session.lock().unwrap().is_none());

        // The decoder stopped without reaching the end and without saying why.
        // The session is over, and it did not finish — so nothing advances.
        let (sh, _prod, done) = staged(&[7; 16], 2);
        done.store(true, Ordering::Release);
        assert_eq!(render_frames(&sh, &mut scratch, &mut t, 2, 32), 8);
        assert!(!sh.completion().may_advance(), "{:?}", sh.completion());
        assert!(sh.completion().is_over());

        // Paused: the render call is short by design, silence is correct, and
        // — the part that matters for DoP — no source frame is consumed.
        let (sh, _prod, _done) = staged(&[7; 64], 2);
        sh.paused.store(true, Ordering::Relaxed);
        assert_eq!(render_frames(&sh, &mut scratch, &mut t, 2, 32), 0);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.fault(), fault::NONE);
        sh.paused.store(false, Ordering::Relaxed);
        assert_eq!(
            render_frames(&sh, &mut scratch, &mut t, 2, 32),
            32,
            "a pause must not have eaten the payload"
        );
    }

    /// A renderer already past its `paused` check may not play the session a
    /// pause installed behind it.
    ///
    /// The schedule, forced rather than hoped for: the renderer reads
    /// `paused = false`, parks before the session slot, the UI thread pauses
    /// and installs a new session full of payload, and only then is the
    /// renderer let go. It arrives at a session it was never entitled to play
    /// and must leave it alone.
    ///
    /// A same-thread flag probe cannot show this and neither can a source
    /// scan: the defect is a read that is correct when it happens and stale by
    /// the time it is used.
    #[test]
    fn a_renderer_past_its_check_cannot_play_a_session_paused_behind_it() {
        const SENTINEL: u32 = 0xFEED_FACE;
        const CH: usize = 2;
        const WANT: usize = 128;

        for dop in [false, true] {
            let sh = Arc::new(Shared::new());
            sh.dop_active.store(dop, Ordering::Relaxed);
            // The pause the listener asked for, through the call the UI makes.
            let stream = BpStream::for_evidence_test(Arc::clone(&sh), PayloadPlan::Identity);

            let inside = Arc::new(std::sync::Barrier::new(2));
            let go = Arc::new(std::sync::Barrier::new(2));
            window::arm(sh.hook_id, window::Site::Render, {
                let inside = Arc::clone(&inside);
                let go = Arc::clone(&go);
                move || {
                    inside.wait();
                    go.wait();
                }
            });

            let renderer = {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    let mut scratch = vec![SENTINEL; WANT * CH];
                    let mut t = tap();
                    let got = render_frames(&sh, &mut scratch, &mut t, CH as u16, WANT);
                    (got, scratch)
                })
            };

            // 1–2. Past the outer read, parked before the slot.
            inside.wait();

            // 3–4. Pause, then install the new session and let the slot go.
            // Frame-identifying, so a single stolen frame is visible rather
            // than merely absent: a payload of one repeated value cannot show
            // that the buffer the listener finally hears begins where the
            // track does.
            let payload: Vec<u32> =
                (0..(WANT * CH * 2) as u32).map(|i| 0xA000_0000 | i).collect();
            let (mut prod, cons) = channel_ring::<u32>(CH, 1024);
            prod.push_frames(&payload).unwrap();
            stream.pause();
            let generation = sh.begin_generation();
            *sh.session.lock().unwrap() = Some(Session {
                generation,
                cons,
                decode_done: Arc::new(AtomicBool::new(false)),
                stop: Arc::new(AtomicBool::new(false)),
            });

            // 5. Released into the session it must not touch.
            go.wait();
            let (got, scratch) = renderer.join().unwrap();

            assert_eq!(got, 0, "dop={dop}: a paused session emitted frames");
            assert!(
                scratch.iter().all(|&s| s == SENTINEL),
                "dop={dop}: the buffer was written, so the caller's silence — \
                 PCM zero, or DoP's marked 0x69 — is not what the device got"
            );
            assert_eq!(sh.frames_played(), 0, "dop={dop}: the position moved");
            assert!(!sh.draining.load(Ordering::Relaxed), "dop={dop}: drain began");
            assert_eq!(sh.underruns.load(Ordering::Relaxed), 0, "dop={dop}");
            assert_eq!(sh.revoked(), fault::NONE, "dop={dop}");
            assert_eq!(sh.fault(), fault::NONE, "dop={dop}");
            assert_eq!(sh.lock_misses.load(Ordering::Relaxed), 0, "dop={dop}");

            // Nothing was taken from the ring, and resume plays it.
            stream.resume();
            let mut scratch = vec![SENTINEL; WANT * CH];
            let mut t = tap();
            assert_eq!(
                render_frames(&sh, &mut scratch, &mut t, CH as u16, WANT),
                WANT,
                "dop={dop}: the paused call ate the session's first buffer"
            );
            assert_eq!(
                scratch,
                payload[..WANT * CH],
                "dop={dop}: the listener resumed into the middle of a frame the                  paused call had already taken"
            );
        }
    }

    /// A `try_lock` that missed because of the pause is not a dropout.
    ///
    /// Same schedule, except the UI thread is still holding the slot when the
    /// renderer is released — which is what installing a session looks like
    /// from the callback's side. Counting that as a lock miss, or revoking the
    /// exactness claim for it, turns every paused hand-off into a reported
    /// integrity fault.
    #[test]
    fn a_paused_handoff_is_not_a_lock_miss() {
        let sh = Arc::new(Shared::new());
        let stream = BpStream::for_evidence_test(Arc::clone(&sh), PayloadPlan::Identity);

        let inside = Arc::new(std::sync::Barrier::new(2));
        let go = Arc::new(std::sync::Barrier::new(2));
        window::arm(sh.hook_id, window::Site::Render, {
            let inside = Arc::clone(&inside);
            let go = Arc::clone(&go);
            move || {
                inside.wait();
                go.wait();
            }
        });

        let renderer = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || {
                let mut scratch = vec![0u32; 256];
                let mut t = tap();
                render_frames(&sh, &mut scratch, &mut t, 2, 128)
            })
        };

        inside.wait();
        stream.pause();
        // Held across the release: the renderer's `try_lock` must fail.
        let guard = sh.session.lock().unwrap();
        go.wait();
        // Held until the render call has returned, so the miss is forced
        // rather than raced for. The `Err` arm needs nothing from the slot,
        // and the drain check below it only tries, so the renderer runs to
        // completion against a slot it can never have.
        let got = renderer.join().unwrap();
        drop(guard);

        assert_eq!(got, 0);
        assert_eq!(
            sh.lock_misses.load(Ordering::Relaxed),
            0,
            "the UI thread holding the slot to install a paused session is the \
             hand-off working"
        );
        assert_eq!(sh.revoked(), fault::NONE);
        assert_eq!(sh.fault(), fault::NONE);
    }

    /// Every dropout observed in the field so far arrived within milliseconds
    /// of a seek, which is the ring refilling rather than the device starving.
    /// A counter that fires on both cannot report either.
    #[test]
    fn the_gap_right_after_a_seek_is_not_counted_as_a_dropout() {
        let mut scratch = vec![0u32; 256];
        let mut t = tap();

        // Fresh session, ring not yet filled: short, but expected.
        let (sh, _prod, _done) = staged(&[7; 16], 2);
        sh.priming.store(true, Ordering::Relaxed);
        render_frames(&sh, &mut scratch, &mut t, 2, 32);
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            0,
            "seek gap is not a dropout"
        );
        assert_eq!(sh.fault(), fault::NONE);
        assert!(
            sh.priming.load(Ordering::Relaxed),
            "still priming until a full buffer lands"
        );

        // The ring catches up: priming ends.
        let (sh, mut prod, _done) = staged(&[7; 64], 2);
        sh.priming.store(true, Ordering::Relaxed);
        render_frames(&sh, &mut scratch, &mut t, 2, 32);
        assert!(
            !sh.priming.load(Ordering::Relaxed),
            "a full buffer ends priming"
        );

        // And from there a short buffer is a real dropout again.
        prod.push_frames(&[7; 16]).unwrap();
        render_frames(&sh, &mut scratch, &mut t, 2, 32);
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            1,
            "starvation after priming counts"
        );
        assert_eq!(sh.revoked(), fault::UNDERRUN);
    }

    /// A session that dropped out and then failed reports the failure, and
    /// still remembers the dropout.
    ///
    /// One slot could not hold both. First-wins meant a dropout suppressed the
    /// write error that actually stopped the session, so the track reported
    /// the wrong reason for ending; last-wins would have erased the evidence
    /// that anything had been lost before it. Both are true and both are
    /// kept — the fatal one is the reason, the recoverable one is the
    /// evidence.
    #[test]
    fn a_fatal_fault_outranks_a_dropout_without_erasing_it() {
        let sh = Shared::new();
        let g = sh.generation();
        sh.revoke(g, fault::UNDERRUN);
        assert_eq!(
            sh.revoked(),
            fault::UNDERRUN,
            "on its own, it is the whole of what is wrong"
        );
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "and it is not a reason the session stopped, because it did not —              the fatal getter reporting it here is what let a dropout take the              one slot the real reason needed"
        );
        assert_eq!(sh.completion(), Completion::Running, "and it ends nothing");

        sh.fail_now(fault::BACKEND_WRITE);
        assert_eq!(
            sh.fault(),
            fault::BACKEND_WRITE,
            "the reason the session stopped is the write, not the dropout"
        );
        assert_eq!(
            sh.completion().failure(),
            Some(fault::BACKEND_WRITE),
            "and that is what the playlist is told"
        );
        assert_eq!(
            sh.revoked(),
            fault::UNDERRUN,
            "the dropout is still on the record"
        );

        // Not the other way round: a recoverable fault arriving after a fatal
        // one does not demote the reason.
        sh.revoke(g, fault::CALLBACK_LOCK_MISS);
        assert_eq!(sh.fault(), fault::BACKEND_WRITE);
        assert_eq!(
            sh.revoked(),
            fault::UNDERRUN,
            "and first-wins still holds within the recoverable record"
        );

        // A new session starts clean on both records.
        sh.begin_generation();
        assert_eq!(sh.fault(), fault::NONE);
        assert_eq!(sh.revoked(), fault::NONE);
    }

    /// The first fault of a session is the one worth reading; later ones of
    /// the same class do not overwrite it.
    #[test]
    fn the_first_fault_wins() {
        let sh = Shared::new();
        sh.revoke(sh.generation(), fault::UNDERRUN);
        sh.revoke(sh.generation(), fault::CALLBACK_LOCK_MISS);
        assert_eq!(sh.revoked(), fault::UNDERRUN);
        assert!(fault::describe(fault::UNDERRUN).contains("dropout"));
        // Every code says something specific.
        for code in 1..=9u8 {
            assert_ne!(
                fault::describe(code),
                fault::describe(0),
                "code {code} unnamed"
            );
        }
    }

    /// The spectrum analyser treats `sample_buf` as one mono stream sampled at
    /// the track's rate. If the tap pushes interleaved channels instead, the
    /// stream is really an N× sample-and-hold and every partial gains a mirror
    /// image at Nyquist − f/N — bass reappearing at the top of the display.
    #[test]
    fn the_tap_hands_the_analyser_one_mono_sample_per_frame() {
        let mono = Arc::new(Mutex::new(Vec::new()));
        let stereo = Arc::new(Mutex::new(Vec::new()));
        let mut t = SpectrumTap::new(
            2,
            PcmKind::Integer { valid_bits: 32 },
            Arc::clone(&mono),
            Arc::clone(&stereo),
        );

        // One flush worth of frames, left and right pulled apart so an
        // interleaved push cannot accidentally look like the average.
        let frames = TAP_BATCH;
        let scale = 1i64 << 20;
        let mut canon = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            canon.push(canon_from_i32((i as i64 * scale) as i32));
            canon.push(canon_from_i32(((i as i64 + 2) * scale) as i32));
        }
        t.feed_canonical(&canon);

        let got = mono.lock().unwrap();
        assert_eq!(got.len(), frames, "one sample per frame, not per channel");
        for (i, &v) in got.iter().enumerate() {
            let want = ((i as i64 + 1) * scale) as i32 as f32 / 2_147_483_648.0;
            assert!((v - want).abs() < 1e-6, "frame {i}: {v} vs {want}");
        }
        assert_eq!(stereo.lock().unwrap().len(), frames);
    }

    /// The complete PCM render path — ring drain, spectrum tap and all —
    /// allocates nothing at the largest callback it will ever be given.
    ///
    /// The ASIO callback was covered; this is the other half. The tap could
    /// grow both its own batches and the two shared analyser buffers from
    /// inside a WASAPI or CPAL render callback, because it appended everything
    /// first and trimmed afterwards. Trimming afterwards is exactly when a
    /// `Vec` grows.
    #[test]
    fn the_full_render_and_tap_path_allocates_nothing_at_maximum_callback() {
        const CH: usize = 2;
        // Larger than any real callback, and larger than the tap's batch, so
        // the flush path is exercised many times inside one call.
        const FRAMES: usize = 8192;

        let mono = Arc::new(Mutex::new(Vec::new()));
        let stereo = Arc::new(Mutex::new(Vec::new()));
        let mut tap = SpectrumTap::new(
            CH as u16,
            PcmKind::Integer { valid_bits: 24 },
            Arc::clone(&mono),
            Arc::clone(&stereo),
        );

        let sh = Shared::new();
        let (mut prod, cons) = channel_ring::<u32>(CH, 1 << 17);
        let done = Arc::new(AtomicBool::new(false));
        *sh.session.lock().unwrap() = Some(Session {
            generation: sh.generation(),
            cons,
            decode_done: Arc::clone(&done),
            stop: Arc::new(AtomicBool::new(false)),
        });

        let block: Vec<u32> = (0..FRAMES * CH)
            .map(|i| ((i as u32) << 8) & 0xFFFF_FF00)
            .collect();
        let mut scratch = vec![0u32; FRAMES * CH];

        // The probe must be able to see an allocation, or "zero" means nothing.
        let before = crate::alloc_probe::arm();
        let canary = vec![0u8; 4096];
        assert!(
            crate::alloc_probe::disarm(before) > 0,
            "the allocation probe is not observing allocations"
        );
        drop(canary);

        // Armed **before the first callback**, not after a warm-up.
        //
        // Warming outside the measurement is what hid the defect this test
        // exists to catch: a buffer whose reservation did not actually take
        // grows on its *first* flush and never again, so a warmed measurement
        // reports zero for a path that allocates on the render thread every
        // time a stream opens.
        let mut total = 0usize;
        let before = crate::alloc_probe::arm();
        for _ in 0..16 {
            prod.push_frames(&block).unwrap();
            total += render_frames(&sh, &mut scratch, &mut tap, CH as u16, FRAMES);
        }
        let allocs = crate::alloc_probe::disarm(before);
        assert_eq!(
            allocs, 0,
            "the render + tap path allocated {allocs} time(s)"
        );
        assert!(total > 0, "the measurement must actually have moved audio");

        // And the analyser really was fed, so this is not zero-by-doing-nothing.
        assert!(!mono.lock().unwrap().is_empty());
        assert!(!stereo.lock().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // Decoder fixtures
    // -----------------------------------------------------------------------

    /// Minimal canonical WAV writer. `bits` is 16/24/32 for PCM and 32 for
    /// IEEE float (`float = true`), and `data` is already little-endian.
    fn write_wav(
        path: &Path,
        data: &[u8],
        bits: u16,
        float: bool,
        sample_rate: u32,
        channels: u16,
    ) {
        let block = channels * bits / 8;
        let byte_rate = sample_rate * block as u32;
        let data_len = data.len() as u32;
        let mut out = Vec::with_capacity(44 + data.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&(if float { 3u16 } else { 1u16 }).to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(data);
        std::fs::write(path, out).unwrap();
    }

    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn temp(name: &str) -> Temp {
        Temp(std::env::temp_dir().join(format!("moosik_bp_{}_{name}", std::process::id())))
    }

    /// Run the real pipeline — `prepare`, the decode thread, the frame ring —
    /// and return the canonical payloads that reached the render side.
    fn drain(prep: Prepared) -> Vec<u32> {
        let ch = prep.source.channels.max(1) as usize;
        let (prod, mut cons) = channel_ring::<u32>(ch, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        decode_loop(
            prep,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            shared.generation(),
            Arc::new(AtomicU64::new(shared.generation())),
        );
        assert!(
            done.load(Ordering::Acquire),
            "decode loop must signal completion"
        );
        assert_eq!(
            shared.fault(),
            fault::NONE,
            "a clean file must not raise a fault"
        );
        let mut out = vec![0u32; 1 << 20];
        let mut got = Vec::new();
        loop {
            let n = cons.pop_frames(&mut out, 4096);
            if n == 0 {
                break;
            }
            got.extend_from_slice(&out[..n * ch]);
        }
        got
    }

    /// 16-bit integer WAV, end to end: file → canonical → device bytes,
    /// compared against bytes built from the original samples.
    #[test]
    fn a_16_bit_wav_reaches_the_device_unaltered() {
        let sr = 48_000u32;
        let src: Vec<i16> = (0..2048i32).map(|i| ((i * 37) % 65_536) as i16).collect();
        let mut bytes = Vec::new();
        for s in &src {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let t = temp("16.wav");
        write_wav(&t.0, &bytes, 16, false, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).unwrap();
        assert_eq!(prep.source.kind, PcmKind::Integer { valid_bits: 16 });
        assert_eq!(prep.source.sample_rate, sr);
        assert_eq!(prep.source.channels, 2);

        let canon = drain(prep);
        assert_eq!(canon.len(), src.len());

        let w = Writer::new(PcmKind::Integer { valid_bits: 16 }, DevFmt::I16).unwrap();
        let mut out = vec![0u8; canon.len() * 2];
        w.write(&canon, &mut out);
        assert_eq!(out, bytes, "16-bit WAV did not survive the pipeline");
    }

    /// 24-bit integer WAV. The container declares 24 and symphonia decodes to
    /// 24-bit storage; the negotiated container is 24-in-32, and the low byte
    /// must come out zero rather than carrying rounding dirt.
    #[test]
    fn a_24_bit_wav_reaches_the_device_unaltered() {
        let sr = 96_000u32;
        let src: Vec<i32> = (0..2048)
            .map(|i| (((i as i64 * 4099) % 16_777_216) - 8_388_608) as i32)
            .collect();
        let mut bytes = Vec::new();
        for s in &src {
            bytes.extend_from_slice(&(*s as u32).to_le_bytes()[..3]);
        }
        let t = temp("24.wav");
        write_wav(&t.0, &bytes, 24, false, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).unwrap();
        assert_eq!(prep.source.kind, PcmKind::Integer { valid_bits: 24 });
        let canon = drain(prep);
        assert_eq!(canon.len(), src.len());

        let k = PcmKind::Integer { valid_bits: 24 };
        let mut packed = vec![0u8; canon.len() * 3];
        Writer::new(k, DevFmt::I24)
            .unwrap()
            .write(&canon, &mut packed);
        assert_eq!(
            packed, bytes,
            "packed-24 output differs from the file's own bytes"
        );

        let mut wide = vec![0u8; canon.len() * 4];
        Writer::new(k, DevFmt::I24In32)
            .unwrap()
            .write(&canon, &mut wide);
        for (i, (chunk, &v)) in wide.chunks_exact(4).zip(&src).enumerate() {
            assert_eq!(
                chunk[0], 0,
                "sample {i}: 24-in-32 must leave the low byte clear"
            );
            let want = ((v as i64) * 256) as i32;
            assert_eq!(
                i32::from_le_bytes(chunk.try_into().unwrap()),
                want,
                "sample {i}"
            );
        }

        // A 24-bit source must never be offered a 16-bit container.
        assert!(Writer::new(k, DevFmt::I16).is_err());
    }

    /// 32-bit integer WAV — the case the `f32` transport silently destroyed.
    /// Adjacent samples differ only in their lowest byte.
    #[test]
    fn a_32_bit_wav_keeps_its_lowest_byte() {
        let sr = 44_100u32;
        let src: Vec<i32> = (0..2048i32)
            .map(|i| 0x0123_4500i32.wrapping_add(i))
            .collect();
        let mut bytes = Vec::new();
        for s in &src {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let t = temp("32.wav");
        write_wav(&t.0, &bytes, 32, false, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).unwrap();
        assert_eq!(prep.source.kind, PcmKind::Integer { valid_bits: 32 });
        let canon = drain(prep);
        assert_eq!(canon.len(), src.len());

        let k = PcmKind::Integer { valid_bits: 32 };
        let mut out = vec![0u8; canon.len() * 4];
        Writer::new(k, DevFmt::I32).unwrap().write(&canon, &mut out);
        assert_eq!(out, bytes, "32-bit WAV lost data on the way to the device");

        // Distinct inputs, distinct outputs — the property `f32` could not
        // hold. These samples differ only below the 24th bit, so the old
        // transport collapsed runs of 256 of them onto one value.
        let distinct_in: std::collections::HashSet<i32> = src.iter().copied().collect();
        let distinct_out: std::collections::HashSet<&[u8]> = out.chunks_exact(4).collect();
        assert_eq!(
            distinct_out.len(),
            distinct_in.len(),
            "{} distinct inputs collapsed to {} distinct outputs",
            distinct_in.len(),
            distinct_out.len()
        );

        // And no narrower or float container is on offer.
        assert!(Writer::new(k, DevFmt::I24In32).is_err());
        assert!(Writer::new(k, DevFmt::I24).is_err());
        assert!(Writer::new(k, DevFmt::F32).is_err());
    }

    /// Float32 WAV: the payload is a bit pattern, not a number.
    #[test]
    fn a_float32_wav_reaches_the_device_as_the_same_bits() {
        let sr = 48_000u32;
        let src: Vec<f32> = (0..2048)
            .map(|i| {
                let x = (i as f32 / 2048.0) * 2.0 - 1.0;
                if i % 257 == 0 { -0.0 } else { x }
            })
            .collect();
        let mut bytes = Vec::new();
        for s in &src {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let t = temp("f32.wav");
        write_wav(&t.0, &bytes, 32, true, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).unwrap();
        assert_eq!(prep.source.kind, PcmKind::Float32);
        let canon = drain(prep);
        assert_eq!(canon.len(), src.len());

        let mut out = vec![0u8; canon.len() * 4];
        Writer::new(PcmKind::Float32, DevFmt::F32)
            .unwrap()
            .write(&canon, &mut out);
        assert_eq!(out, bytes, "float payload was altered");
        for (i, (c, s)) in canon.iter().zip(&src).enumerate() {
            assert_eq!(
                *c,
                s.to_bits(),
                "sample {i}: signed zero or NaN payload lost"
            );
        }
        // And it can never be converted to an integer container under a claim.
        assert!(Writer::new(PcmKind::Float32, DevFmt::I32).is_err());
    }

    /// Left and right must not be swapped or shifted, whatever the packet
    /// boundaries look like.
    #[test]
    fn channels_stay_in_their_own_lanes() {
        let sr = 44_100u32;
        // Left is always negative, right always positive: a one-sample shift
        // is visible at a glance rather than only in a diff.
        let mut src: Vec<i16> = Vec::new();
        for i in 0..3000i32 {
            src.push(-1 - (i % 1000) as i16);
            src.push(1 + (i % 1000) as i16);
        }
        let mut bytes = Vec::new();
        for s in &src {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let t = temp("lanes.wav");
        write_wav(&t.0, &bytes, 16, false, sr, 2);

        let canon = drain(prepare(&t.0, Duration::ZERO).unwrap());
        assert_eq!(canon.len(), src.len());
        for (i, &c) in canon.iter().enumerate() {
            let v = (c as i32) >> 16;
            if i % 2 == 0 {
                assert!(v < 0, "sample {i} should be a left channel");
            } else {
                assert!(v > 0, "sample {i} should be a right channel");
            }
        }
    }

    /// A queued gapless track that does not fit the open stream is a typed
    /// integrity failure, not a clean end of file.
    #[test]
    fn an_incompatible_gapless_track_faults_instead_of_narrowing() {
        let a = temp("gap_a.wav");
        let b = temp("gap_b.wav");
        let mut d16 = Vec::new();
        for i in 0..1024i16 {
            d16.extend_from_slice(&i.to_le_bytes());
        }
        write_wav(&a.0, &d16, 16, false, 44_100, 2);
        // Same rate and channel count, different depth — the case that used to
        // ride the open 16-bit stream and truncate.
        let mut d32 = Vec::new();
        for i in 0..1024i32 {
            d32.extend_from_slice(&(i << 12).to_le_bytes());
        }
        write_wav(&b.0, &d32, 32, false, 44_100, 2);

        let prep_a = prepare(&a.0, Duration::ZERO).unwrap();
        let prep_b = prepare(&b.0, Duration::ZERO).unwrap();
        assert_eq!(prep_a.source.kind, PcmKind::Integer { valid_bits: 16 });
        assert_eq!(prep_b.source.kind, PcmKind::Integer { valid_bits: 32 });

        // Stand in for a stream negotiated at 16 valid bits — what the first
        // track needed, and what the second one overflows.
        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(16, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        assert!(shared.can_carry_kind(PcmKind::Integer { valid_bits: 16 }));
        assert!(
            !shared.can_carry_kind(PcmKind::Integer { valid_bits: 32 }),
            "the guard under test must reject this"
        );
        *shared.next.lock().unwrap() = Some(prep_b);
        let (prod, _cons) = channel_ring::<u32>(2, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        decode_loop(
            prep_a,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            shared.generation(),
            Arc::new(AtomicU64::new(shared.generation())),
        );
        assert_eq!(
            shared.fault(),
            fault::SOURCE_FORMAT_CHANGE,
            "a depth change at a gapless boundary must be reported"
        );
    }

    #[test]
    fn prepare_seeks_without_decoding_from_start() {
        let sr = 44_100u32;
        let mut bytes = Vec::new();
        for _ in 0..(sr as usize * 2 * 2) {
            bytes.extend_from_slice(&1000i16.to_le_bytes());
        }
        let t = temp("seek.wav");
        write_wav(&t.0, &bytes, 16, false, sr, 2);

        let prep = prepare(&t.0, Duration::from_secs(1)).unwrap();
        let decoded = drain(prep);
        // Roughly 1 s of stereo should remain (allow one packet of slack
        // for the coarse container seek).
        let expect = sr as usize * 2;
        assert!(
            decoded.len() <= expect + 8192 && decoded.len() >= expect - 8192,
            "got {} samples, expected ≈{expect}",
            decoded.len()
        );
    }

    /// A positive gapless proof: two tracks, uniquely tagged, through the real
    /// decode loop and the real ring.
    ///
    /// Not a "did it crash" test. Every frame carries its track and its index,
    /// so a lost, duplicated or reordered frame is identifiable, and the
    /// boundary the decoder reports has to fall exactly where track A ends.
    #[test]
    fn a_gapless_boundary_loses_no_frame_and_lands_exactly() {
        let sr = 44_100u32;
        // Track A: left negative, right positive, magnitude = frame index.
        // Track B: the same, offset far enough that the two cannot be confused.
        let a_frames = 700usize;
        let b_frames = 500usize;
        let mut a_bytes = Vec::new();
        for i in 0..a_frames {
            a_bytes.extend_from_slice(&(-(i as i16 + 1)).to_le_bytes());
            a_bytes.extend_from_slice(&(i as i16 + 1).to_le_bytes());
        }
        let mut b_bytes = Vec::new();
        for i in 0..b_frames {
            b_bytes.extend_from_slice(&(-(i as i16 + 5001)).to_le_bytes());
            b_bytes.extend_from_slice(&(i as i16 + 5001).to_le_bytes());
        }
        let ta = temp("gapless_a.wav");
        let tb = temp("gapless_b.wav");
        write_wav(&ta.0, &a_bytes, 16, false, sr, 2);
        write_wav(&tb.0, &b_bytes, 16, false, sr, 2);

        let prep_a = prepare(&ta.0, Duration::ZERO).unwrap();
        let prep_b = prepare(&tb.0, Duration::ZERO).unwrap();
        assert_eq!(
            prep_a.source, prep_b.source,
            "the fixtures must be reuse-compatible"
        );

        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(16, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        *shared.next.lock().unwrap() = Some(prep_b);

        let (prod, mut cons) = channel_ring::<u32>(2, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        decode_loop(
            prep_a,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            shared.generation(),
            Arc::new(AtomicU64::new(shared.generation())),
        );
        assert_eq!(
            shared.fault(),
            fault::NONE,
            "a compatible gapless hand-off must not fault"
        );

        let mut got = Vec::new();
        let mut buf = vec![0u32; 8192];
        loop {
            let n = cons.pop_frames(&mut buf, 4096);
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n * 2]);
        }

        // Nothing lost, nothing duplicated.
        assert_eq!(
            got.len(),
            (a_frames + b_frames) * 2,
            "frame count changed across the boundary"
        );

        // Every frame is the one it should be, in order, in its own channel.
        for i in 0..a_frames + b_frames {
            let (want_l, want_r) = if i < a_frames {
                (-(i as i32 + 1), i as i32 + 1)
            } else {
                let k = (i - a_frames) as i32;
                (-(k + 5001), k + 5001)
            };
            assert_eq!((got[i * 2] as i32) >> 16, want_l, "frame {i} left");
            assert_eq!((got[i * 2 + 1] as i32) >> 16, want_r, "frame {i} right");
        }

        // The boundary the decoder published lands exactly at the end of A.
        let boundaries: Vec<Boundary> = shared.boundaries.lock().unwrap().iter().cloned().collect();
        assert_eq!(boundaries.len(), 1, "exactly one boundary");
        assert_eq!(
            boundaries[0].frames, a_frames as u64,
            "the boundary lands on exactly the frame track A ended on"
        );
        // And it carries the new track, not just its position: the UI has no
        // other way to roll source, transform and fidelity over together.
        assert_eq!(boundaries[0].source.format.sample_rate, 44_100);
        assert_eq!(boundaries[0].plan, PayloadPlan::Identity);
        assert!(boundaries[0].transform.is_identity());
        assert!(
            boundaries[0].generation > 1,
            "a gapless boundary opens a new generation so the new track does \\
             not inherit the old one's fault"
        );
    }

    /// The reuse predicate and the decode loop agree: a next track the open
    /// format cannot carry stops at the boundary instead of narrowing.
    #[test]
    fn an_incompatible_next_track_stops_at_the_boundary() {
        let sr = 44_100u32;
        let mut a = Vec::new();
        for i in 0..256i16 {
            a.extend_from_slice(&i.to_le_bytes());
        }
        let mut b = Vec::new();
        for i in 0..256i32 {
            b.extend_from_slice(&(i << 12).to_le_bytes());
        }
        let ta = temp("boundary_a.wav");
        let tb = temp("boundary_b.wav");
        write_wav(&ta.0, &a, 16, false, sr, 2);
        write_wav(&tb.0, &b, 32, false, sr, 2);

        let prep_a = prepare(&ta.0, Duration::ZERO).unwrap();
        let prep_b = prepare(&tb.0, Duration::ZERO).unwrap();

        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(16, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        assert!(
            !shared.can_carry_kind(prep_b.source.kind),
            "the guard under test must reject this"
        );
        *shared.next.lock().unwrap() = Some(prep_b);

        let (prod, mut cons) = channel_ring::<u32>(2, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        decode_loop(
            prep_a,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            shared.generation(),
            Arc::new(AtomicU64::new(shared.generation())),
        );
        assert_eq!(shared.fault(), fault::SOURCE_FORMAT_CHANGE);

        // Track A still arrived in full: refusing the hand-off must not throw
        // away audio that was already correct.
        let mut total = 0usize;
        let mut buf = vec![0u32; 8192];
        loop {
            let n = cons.pop_frames(&mut buf, 4096);
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, 128, "track A must be delivered complete");
    }

    // -----------------------------------------------------------------------
    // Packet validation
    //
    // These drive `canonicalize_packet` — the function both `prepare` and the
    // decode loop call — with buffers built the way a decoder produces them.
    // -----------------------------------------------------------------------

    use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, Channels, SignalSpec};

    /// Build a decoded packet of interleaved `i32` samples.
    fn s32_packet(rate: u32, channels: Channels, frames: &[[i32; 2]]) -> AudioBuffer<i32> {
        let spec = SignalSpec::new(rate, channels);
        let mut buf = AudioBuffer::<i32>::new(frames.len() as u64, spec);
        buf.render_reserved(Some(frames.len()));
        let n = channels.count();
        for c in 0..n {
            let lane = buf.chan_mut(c);
            for (i, fr) in frames.iter().enumerate() {
                lane[i] = fr[c.min(1)];
            }
        }
        buf
    }

    const STEREO: Channels = Channels::FRONT_LEFT.union(Channels::FRONT_RIGHT);

    /// An impossible declared width must be rejected, and must not wrap into a
    /// plausible one on the way.
    ///
    /// `bits_per_sample` is a `u32`; the old code cast it to `u8` before
    /// checking, so 256 became 0 and 257 became 1 — and a 257-bit declaration
    /// then steered the output format as though it meant one bit.
    #[test]
    fn impossible_declared_widths_are_rejected_without_wrapping() {
        for bad in [0u32, 33, 40, 64, 256, 257, 512, u32::MAX] {
            let e = declared_valid_bits(Some(bad), 32)
                .expect_err(&format!("{bad} declared bits should be refused"));
            assert!(
                matches!(e, state::FailureReason::SourceParse(_)),
                "{bad}: {e:?}"
            );
        }
        // ...and the legitimate range still passes, unchanged.
        for good in 1..=32u32 {
            assert_eq!(
                declared_valid_bits(Some(good), 32).unwrap(),
                Some(good as u8)
            );
        }
        // A declaration wider than the storage it decodes into is contradictory.
        assert!(declared_valid_bits(Some(32), 24).is_err());
        assert_eq!(declared_valid_bits(None, 24).unwrap(), None);
    }

    /// A declared width is believed only when the samples honour it.
    #[test]
    fn a_declared_width_is_checked_against_the_samples() {
        let clean: Vec<u32> = [0i32, 1 << 8, -1 << 8, 0x7FFF_FF00u32 as i32]
            .iter()
            .map(|&v| v as u32)
            .collect();
        assert_eq!(effective_precision(32, Some(24), &clean), 24);

        // One dirty sample is enough: the source has more content than it
        // admits, and the conservative answer demands the wider container.
        let mut dirty = clean.clone();
        dirty.push(0x0000_0001);
        assert_eq!(effective_precision(32, Some(24), &dirty), 32);

        assert_eq!(effective_precision(16, None, &clean), 16);
        assert_eq!(effective_precision(24, Some(24), &clean), 24);
    }

    /// The case that used to reach a 24-bit writer intact: a clean probe packet
    /// establishes a 24-bit plan, and a later packet puts content in the low
    /// eight bits.
    ///
    /// The plan is the ceiling. A packet needing more bits than the negotiated
    /// container holds must fault *before* publication, not arrive already
    /// truncated on the wire.
    #[test]
    fn a_later_packet_that_needs_more_bits_faults_before_publication() {
        // Probe: every sample clean at 24 bits, declared 24, stored in 32.
        let probe = s32_packet(44_100, STEREO, &[[1 << 8, 2 << 8], [3 << 8, 4 << 8]]);
        let mut out = Vec::new();
        let plan = canonicalize_packet(&probe.as_audio_buffer_ref(), None, Some(24), &mut out)
            .expect("a clean 24-in-32 packet is valid");
        assert_eq!(plan.kind, PcmKind::Integer { valid_bits: 24 });
        assert_eq!(out.len(), 4);

        // Next packet: content below the declared point, so it really needs 32.
        let dirty = s32_packet(44_100, STEREO, &[[1 << 8, 2 << 8], [3 << 8, 0x0000_0001]]);
        let before = out.len();
        let err = canonicalize_packet(
            &dirty.as_audio_buffer_ref(),
            Some(&plan),
            Some(24),
            &mut out,
        )
        .expect_err("a packet needing 32 bits must not ride a 24-bit plan");
        assert!(matches!(err, state::FailureReason::Decode(_)), "{err:?}");
        assert_eq!(out.len(), before, "a rejected packet must publish nothing");
    }

    /// A valid multi-packet 24-in-32 stream is accepted throughout.
    #[test]
    fn a_consistent_24_in_32_stream_is_accepted() {
        let mut out = Vec::new();
        let first = s32_packet(96_000, STEREO, &[[7 << 8, -7 << 8]]);
        let plan =
            canonicalize_packet(&first.as_audio_buffer_ref(), None, Some(24), &mut out).unwrap();
        assert_eq!(plan.kind, PcmKind::Integer { valid_bits: 24 });

        for k in 1..8i32 {
            let p = s32_packet(96_000, STEREO, &[[k << 8, -(k << 8)], [(k * 3) << 8, 0]]);
            let got =
                canonicalize_packet(&p.as_audio_buffer_ref(), Some(&plan), Some(24), &mut out)
                    .unwrap();
            assert_eq!(got, plan);
        }
        assert_eq!(out.len(), 2 + 7 * 4);
        // Every published sample is a left-aligned 24-bit value.
        assert!(out.iter().all(|&c| low_bits_are_clean(c, 24)));
    }

    /// A packet whose rate, channel count or layout differs from the plan is
    /// rejected before any of it is published.
    #[test]
    fn a_spec_mismatch_is_rejected_before_publication() {
        let first = s32_packet(44_100, STEREO, &[[1 << 8, 2 << 8]]);
        let mut out = Vec::new();
        let plan =
            canonicalize_packet(&first.as_audio_buffer_ref(), None, Some(24), &mut out).unwrap();
        let published = out.len();

        // Different rate.
        let other_rate = s32_packet(48_000, STEREO, &[[1 << 8, 2 << 8]]);
        assert!(
            canonicalize_packet(
                &other_rate.as_audio_buffer_ref(),
                Some(&plan),
                Some(24),
                &mut out
            )
            .is_err()
        );
        assert_eq!(out.len(), published);

        // Different channel count and layout.
        let mono = s32_packet(44_100, Channels::FRONT_LEFT, &[[1 << 8, 0]]);
        assert!(
            canonicalize_packet(&mono.as_audio_buffer_ref(), Some(&plan), Some(24), &mut out)
                .is_err()
        );
        assert_eq!(out.len(), published);

        // Same channel count, different speakers: three channels as L/R/LFE is
        // not three channels as L/R/C.
        let lr_lfe = Channels::FRONT_LEFT
            .union(Channels::FRONT_RIGHT)
            .union(Channels::LFE1);
        let lr_c = Channels::FRONT_LEFT
            .union(Channels::FRONT_RIGHT)
            .union(Channels::FRONT_CENTRE);
        assert_eq!(lr_lfe.count(), lr_c.count());
        let a = s32_packet(44_100, lr_lfe, &[[1 << 8, 2 << 8]]);
        let mut o2 = Vec::new();
        let plan3 = canonicalize_packet(&a.as_audio_buffer_ref(), None, Some(24), &mut o2).unwrap();
        let b = s32_packet(44_100, lr_c, &[[1 << 8, 2 << 8]]);
        let n = o2.len();
        assert!(
            canonicalize_packet(&b.as_audio_buffer_ref(), Some(&plan3), Some(24), &mut o2).is_err(),
            "a layout change must be refused even at the same channel count"
        );
        assert_eq!(o2.len(), n);
    }

    // -----------------------------------------------------------------------
    // Seek
    // -----------------------------------------------------------------------

    /// A seek reports where it actually landed, and a seek beyond the end is a
    /// typed failure rather than a silent decode from wherever the reader was.
    #[test]
    fn a_seek_reports_where_it_landed_and_fails_visibly() {
        let sr = 44_100u32;
        let mut bytes = Vec::new();
        for i in 0..(sr as usize * 2 * 2) {
            bytes.extend_from_slice(&((i % 1000) as i16).to_le_bytes());
        }
        let t = temp("seek_report.wav");
        write_wav(&t.0, &bytes, 16, false, sr, 2);

        // No seek requested: the landing position is the start.
        let p = prepare(&t.0, Duration::ZERO).unwrap();
        assert_eq!(p.seeked_to, Duration::ZERO);

        // A real seek lands near where it was asked to.
        let p = prepare(&t.0, Duration::from_millis(500)).unwrap();
        let landed = p.seeked_to.as_secs_f64();
        assert!((landed - 0.5).abs() < 0.25, "landed at {landed}s");

        // Past the end: reported, not silently ignored. The old code discarded
        // the seek result entirely and decoded from wherever the reader was.
        match prepare(&t.0, Duration::from_secs(600)) {
            Err(state::FailureReason::Seek(_)) => {}
            Err(other) => panic!("expected a typed seek failure, got {other:?}"),
            Ok(p) => {
                // A container that clamps rather than refusing must still
                // report the position it actually reached.
                assert!(
                    p.seeked_to < Duration::from_secs(600),
                    "a clamped seek must not claim the requested position"
                );
            }
        }
    }

    /// A damaged first packet is missing audio, and the open fails rather than
    /// beginning a session that claims to be exact having already lost one.
    #[test]
    fn a_damaged_first_packet_fails_the_open() {
        let sr = 44_100u32;
        let mut bytes = Vec::new();
        for i in 0..2048i16 {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        let t = temp("damaged.wav");
        write_wav(&t.0, &bytes, 16, false, sr, 2);
        // Truncating the declared data chunk mid-frame leaves a reader that
        // cannot produce the audio the header promises.
        let mut raw = std::fs::read(&t.0).unwrap();
        raw.truncate(44 + 7);
        std::fs::write(&t.0, &raw).unwrap();

        match prepare(&t.0, Duration::ZERO) {
            Ok(p) => {
                // Some readers surface this as a short but valid first packet.
                // What must never happen is a full-length claim.
                assert!(
                    p.primed.len() <= 4,
                    "a truncated file yielded a full packet"
                );
            }
            Err(e) => assert!(
                matches!(
                    e,
                    state::FailureReason::Decode(_)
                        | state::FailureReason::SourceRead(_)
                        | state::FailureReason::SourceParse(_)
                ),
                "unexpected error kind: {e:?}"
            ),
        }
    }

    /// A Float32 playlist crosses a gapless boundary on one Q1.31 stream, and
    /// the boundary resets what belongs to the outgoing track.
    ///
    /// Two defects met here. The compatibility check asked whether an integer
    /// stream could carry Float32 and said no, so the hand-off faulted with a
    /// format change; and the guard's off-grid latch was never cleared, so an
    /// off-grid first track withheld the value-exact label from every track
    /// after it for the life of the stream.
    #[test]
    fn a_float_playlist_hands_off_gaplessly_on_one_q31_stream() {
        let sr = 44_100u32;
        // Track A: off-grid, so the latch is set while it plays.
        let mut a = Vec::new();
        for i in 0..512 {
            let v = (i as f32 * 0.001_37).sin() * 0.618_034;
            a.extend_from_slice(&v.to_le_bytes());
        }
        // Track B: exactly on the Q1.31 lattice — a 16-bit master as floats.
        let mut b = Vec::new();
        for i in 0..512i32 {
            let v = ((i % 32_768) - 16_384) as f32 / 32_768.0;
            b.extend_from_slice(&v.to_le_bytes());
        }
        let ta = temp("gapless_q31_a.wav");
        let tb = temp("gapless_q31_b.wav");
        write_wav(&ta.0, &a, 32, true, sr, 2);
        write_wav(&tb.0, &b, 32, true, sr, 2);

        let prep_a = prepare(&ta.0, Duration::ZERO).unwrap();
        let prep_b = prepare(&tb.0, Duration::ZERO).unwrap();
        let a_frames = 256usize; // 512 samples / 2 channels

        let shared = Arc::new(Shared::new());
        // The device: 32-bit integer, which is what the conversion produces.
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        shared.out_rate.store(sr, Ordering::Relaxed);
        shared
            .plan
            .store(PayloadPlan::Q31.code(), Ordering::Relaxed);
        *shared.next.lock().unwrap() = Some(prep_b);

        let (prod, mut cons) = channel_ring::<u32>(2, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        let generation = shared.generation();
        decode_loop(
            prep_a,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            generation,
            Arc::new(AtomicU64::new(generation)),
        );

        // The hand-off happened, and it did not fault.
        assert_eq!(
            shared.fault(),
            fault::NONE,
            "a Q1.31 stream must carry the next Float32 track"
        );
        assert!(
            shared.decode_eof_for(shared.decode_generation()),
            "both tracks played out"
        );

        let boundaries: Vec<Boundary> = shared.boundaries.lock().unwrap().iter().cloned().collect();
        assert_eq!(boundaries.len(), 1, "exactly one boundary");
        assert_eq!(boundaries[0].frames, a_frames as u64);
        assert_eq!(boundaries[0].plan, PayloadPlan::Q31);
        assert_eq!(
            boundaries[0].path, tb.0,
            "the boundary names the track that begins, so its own verdict can be sought"
        );
        assert!(boundaries[0].generation > generation);

        // Both tracks are in the ring, in order.
        let mut got = vec![0u32; 1024];
        assert_eq!(cons.pop_frames(&mut got, 512), 512);

        // Track A needed rounding; track B did not — and because the latch is
        // reset at the boundary, the stream's published count reflects the
        // track that is playing rather than the one that ended.
        assert_eq!(
            shared.off_grid_in(boundaries[0].generation),
            0,
            "an off-grid predecessor must not withhold the label from its successor"
        );
    }

    /// A verdict that says "value-exact" does not survive contact with a
    /// sample that is not.
    ///
    /// The scan is a cache. The conversion is the measurement, and it re-runs
    /// on every packet regardless of what the cache said — so a file rewritten
    /// between the scan and playback costs a label rather than producing a
    /// false one.
    #[test]
    fn a_stale_positive_verdict_is_overruled_by_the_samples() {
        let sr = 44_100u32;
        // What the scan saw: entirely on the Q1.31 lattice.
        let mut clean = Vec::new();
        for i in 0..1024i32 {
            let v = ((i % 32_768) - 16_384) as f32 / 32_768.0;
            clean.extend_from_slice(&v.to_le_bytes());
        }
        let t = temp("stale_verdict.wav");
        write_wav(&t.0, &clean, 32, true, sr, 2);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let verdict = scan_track_q31(&t.0, &cancel).unwrap();
        assert!(verdict.value_exact, "the scan must genuinely have passed");

        // The file is then rewritten with one off-grid sample. The verdict is
        // now stale, and a cache keyed on it would still be positive.
        let mut rewritten = clean.clone();
        let off = f32::from_bits(0x2F80_0000); // 2^-32, below the Q1.31 step
        rewritten[600 * 4..600 * 4 + 4].copy_from_slice(&off.to_le_bytes());
        write_wav(&t.0, &rewritten, 32, true, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).unwrap();
        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        shared
            .plan
            .store(PayloadPlan::Q31.code(), Ordering::Relaxed);
        let (prod, _cons) = channel_ring::<u32>(2, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        let g = shared.generation();
        decode_loop(
            prep,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            g,
            Arc::new(AtomicU64::new(g)),
        );

        // It still plays to the end — a stale verdict costs a label, never the
        // audio.
        assert_eq!(shared.fault(), fault::NONE);
        assert!(shared.decode_eof_for(shared.decode_generation()));
        // And the conversion has contradicted the verdict, which is what the
        // UI reads to withdraw the claim.
        assert!(
            shared.off_grid_in(g) > 0,
            "the samples must overrule the cache"
        );
    }

    /// A real 64-bit float file, decoded by the real decoder.
    ///
    /// Everything about this route was previously asserted from the type
    /// level: a `PcmKind::Float64` handed to `plan_ladder`, a
    /// `TransformDescription` compared to a constant. Nothing had ever put an
    /// `f64` WAV through `prepare` and the decode loop, so nothing had
    /// confirmed that the decoder produces `AudioBufferRef::F64` for one, that
    /// the narrowing arm is the arm that runs, or that the values that come
    /// out the other side are the `f32` of the original doubles.
    #[test]
    fn a_64_bit_float_file_is_narrowed_by_the_real_decoder() {
        // Values chosen so the narrowing is observable: the first three are
        // exactly representable in `f32` and survive; the fourth is not and
        // must come out as the nearest `f32`.
        let doubles: [f64; 8] = [
            0.5,
            -0.25,
            0.125,
            0.1,                       // 0.1 has no exact f32 (or f64) form
            1.0 / 3.0,                 // nor this
            -0.7500000000000001,       // just off a representable value
            2.0f64.powi(-30),
            0.0,
        ];
        let mut bytes = Vec::new();
        for v in doubles {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let t = temp("f64.wav");
        write_wav(&t.0, &bytes, 64, true, 48_000, 2);

        let prep = match prepare(&t.0, Duration::ZERO) {
            Ok(p) => p,
            // Recorded rather than hidden: if the decoder in this build cannot
            // read 64-bit float WAV, the route has no reachable input and the
            // claim about it is about nothing.
            Err(e) => panic!(
                "the decoder must read a 64-bit float WAV for this route to exist: {e}"
            ),
        };
        assert_eq!(
            prep.source.kind,
            PcmKind::Float64,
            "the source family has to survive `prepare` or nothing downstream can act on it"
        );

        let got = drain(prep);
        assert_eq!(got.len(), doubles.len(), "every frame reached the ring");
        for (i, v) in doubles.iter().enumerate() {
            let want = canon_from_f32(*v as f32);
            assert_eq!(
                got[i], want,
                "sample {i}: {v} must arrive as the `f32` nearest it"
            );
        }

        // And the narrowing is real for at least one of them, or the fixture
        // is not exercising anything.
        assert!(
            doubles.iter().any(|v| *v as f32 as f64 != *v),
            "this fixture must contain a value that cannot survive the narrowing"
        );
    }

    /// The whole 64-bit float route: decoder, Q1.31 conversion, bytes, end.
    ///
    /// The fixture above proves the decoder narrows. This runs the same file
    /// through the conversion the processed policies actually put it on, and
    /// reads what the ring received — so what is checked is the payload the
    /// device would have been handed, not an intermediate.
    #[test]
    fn a_64_bit_float_file_reaches_the_ring_as_q31_and_ends_cleanly() {
        // Doubles that are *not* on the Q1.31 lattice even after narrowing, so
        // the conversion has to round and the route cannot claim otherwise.
        let doubles: [f64; 8] = [
            0.1,
            1.0 / 3.0,
            -0.7,
            0.123_456_789,
            -1.0 / 7.0,
            0.987_654_321,
            2.0f64.powi(-33),
            -0.000_000_1,
        ];
        let mut bytes = Vec::new();
        for v in doubles {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let t = temp("f64_q31.wav");
        write_wav(&t.0, &bytes, 64, true, 48_000, 2);

        let prep = prepare(&t.0, Duration::ZERO).expect("a 64-bit float WAV decodes");
        assert_eq!(prep.source.kind, PcmKind::Float64);

        let ch = prep.source.channels.max(1) as usize;
        let (prod, mut cons) = channel_ring::<u32>(ch, 1 << 16);
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared::new());
        // The route the processed policies choose for this source.
        let plan = plan_ladder(PcmKind::Float64, state::OutputPolicy::PreferExact);
        assert_eq!(plan, vec![PayloadPlan::Q31]);
        shared.plan.store(PayloadPlan::Q31.code(), Ordering::Relaxed);
        shared
            .policy
            .store(state::OutputPolicy::PreferExact.code(), Ordering::Relaxed);
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_integer.store(true, Ordering::Relaxed);
        let generation = shared.generation();
        decode_loop(
            prep,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            generation,
            Arc::new(AtomicU64::new(generation)),
        );

        // The bytes that reached the ring.
        let mut got = vec![0u32; doubles.len()];
        let frames = cons.pop_frames(&mut got, doubles.len() / ch);
        assert_eq!(frames, doubles.len() / ch, "every frame arrived");

        // Each one is the Q1.31 of the `f32` nearest the original double —
        // two roundings, in that order, and the payload is the result of both.
        for (i, v) in doubles.iter().enumerate() {
            let narrowed = *v as f32;
            let want = q31::to_q31_processed(narrowed) as u32;
            assert_eq!(
                got[i], want,
                "sample {i}: {v} → f32 {narrowed} → Q1.31"
            );
        }

        // The route ran to the end of the source without faulting: an off-grid
        // sample on a conversion that was opened as processed is that
        // conversion working, not a failure.
        assert!(done.load(Ordering::Acquire));
        assert_eq!(shared.fault(), fault::NONE);
        assert!(shared.decode_eof_for(shared.decode_generation()));

        // And it had to round, or this fixture proves nothing about rounding.
        assert!(
            shared.off_grid_in(generation) > 0,
            "every sample here is off the lattice; the guard must have rounded"
        );
    }

    /// The route a 64-bit float source takes, and what it is called.
    ///
    /// Strict stops rather than opening something quieter; the processed
    /// policies take the conversion — and the description says both halves of
    /// what happened, which it did not: "Float32 → Q1.31" names the second and
    /// omits the narrowing that precedes it, which is where the numbers
    /// actually change.
    #[test]
    fn the_64_bit_float_route_says_both_halves_of_what_it_did() {
        use state::OutputPolicy::*;

        // Strict gets the identity rung precisely so the open fails and says
        // why — there is no exact device format for an `f64`.
        assert_eq!(
            plan_ladder(PcmKind::Float64, StrictExact),
            vec![PayloadPlan::Identity]
        );
        assert!(format::exact_candidates(PcmKind::Float64, false).is_err());

        for policy in [PreferExact, HqProcessed] {
            let ladder = plan_ladder(PcmKind::Float64, policy);
            assert_eq!(
                ladder,
                vec![PayloadPlan::Q31],
                "{policy:?}: one rung, and it is the conversion"
            );
            let described = ladder[0].describe(PcmKind::Float64);
            assert_eq!(described, state::TransformDescription::Float64ToQ31Processed);
            assert!(!described.is_identity());
            let text = described.describe();
            assert!(
                text.contains("Float64") && text.contains("Float32"),
                "the narrowing has to be in the text: {text}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forced schedules
    //
    // Each of these pins the exact interleaving that the code under test was
    // written to survive. Run in sequence they all pass trivially, which is
    // why they are not run in sequence: a `Barrier` holds one thread inside
    // the window while the other steps through it. The point is not that a
    // race is unlikely — it is that the window is closed.
    // -----------------------------------------------------------------------

    /// Track A's decode thread, descheduled mid-failure, must not fail track B.
    ///
    /// The schedule: A reads the generation and finds its own; A stops; B is
    /// installed; A resumes and writes. With the clock and the terminal state
    /// as separate words, A's write landed on B — stamped with a generation
    /// that had become current while A was not looking — and track B failed
    /// before its first sample with an error from a file it had never opened.
    #[test]
    fn a_stale_writer_cannot_fail_the_track_that_replaced_it() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.generation();

        // `entered` releases the main thread once the writer is *inside* the
        // transition; `resume` releases the writer once B has been installed.
        // Rendezvousing on either side of `fail` instead would not test
        // anything: the writer would read the clock after B had arrived, find
        // a generation that is not its own, and decline — which the broken
        // design did correctly too. The bug was only ever reachable from
        // between those two instants.
        let entered = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let entered = Arc::clone(&entered);
            let resume = Arc::clone(&resume);
            window::arm(sh.hook_id, window::Site::Transition, move || {
                entered.wait();
                resume.wait();
            });
        }

        let writer = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.fail(a, fault::DECODE_ERROR))
        };

        // The writer is now holding the pre-transition value of `play`, having
        // already satisfied itself that generation A is current — because it
        // was, when it looked.
        entered.wait();
        let b = sh.begin_generation();
        resume.wait();
        writer.join().unwrap();

        assert_eq!(sh.generation(), b);
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "a superseded session ended the one that replaced it"
        );
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "and it must not have left its fault behind either"
        );

        // B still works: refusing a stale writer is not the same as freezing.
        sh.note_decode_eof(b);
        sh.finish_clean(b);
        assert_eq!(sh.completion(), Completion::CleanEof);
    }

    /// The same window, unsynchronised, many times over.
    ///
    /// The barrier above proves the specific interleaving is safe. This proves
    /// there is no *other* one: whatever order the two threads happen to run
    /// in, a session that has been superseded can neither end its successor
    /// nor fault it.
    #[test]
    fn no_interleaving_lets_a_dead_session_speak_for_a_live_one() {
        for _ in 0..500 {
            let sh = Arc::new(Shared::new());
            let a = sh.generation();

            let writers: Vec<_> = [fault::DECODE_ERROR, fault::SOURCE_READ]
                .into_iter()
                .map(|code| {
                    let sh = Arc::clone(&sh);
                    std::thread::spawn(move || sh.fail(a, code))
                })
                .collect();
            let b = sh.begin_generation();
            for w in writers {
                w.join().unwrap();
            }

            // Either the writers got in before `begin_generation` — in which
            // case B is clean because opening a generation retires everything
            // older — or they got in after and were refused. There is no third
            // outcome, and "B is failed" is not one of them.
            assert_eq!(sh.generation(), b);
            assert_eq!(sh.completion(), Completion::Running);
            assert_eq!(sh.fault(), fault::NONE);
        }
    }

    /// The device may empty the ring before the UI has crossed the boundary.
    ///
    /// The render thread drains on its own deadline and the boundary is popped
    /// by the UI thread on a frame tick, so on any real machine the drain
    /// finishes first — the ring is empty *past* the boundary by definition,
    /// since the boundary is a frame count the device has already gone by.
    /// Ending the session there reported the last gapless track as finished
    /// before it had begun, and the playlist moved on without playing it.
    /// One callback spans A→B, and B's dropout belongs to B.
    ///
    /// A device buffer can be deeper than the tail of a track, so a single
    /// callback emits the end of A and the beginning of B. The crossing used
    /// to be done by the *UI* thread on a frame tick, which means all of that
    /// buffer played under A's generation and there was no instant at which
    /// anything could be said about B — a shortfall in B's half was recorded
    /// against a track that had already finished and thrown away as stale.
    ///
    /// The UI is held back here for the whole test, and never consulted. The
    /// render thread crosses, because the device is the only thing that knows
    /// when it arrives.
    #[test]
    fn a_callback_that_spans_a_boundary_blames_the_track_it_lands_in() {
        const CH: usize = 2;
        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(512, Ordering::Relaxed);
        let a = sh.begin_generation();

        // A has 64 frames left; the callback will ask for 256.
        let (mut prod, cons) = channel_ring::<u32>(CH, 4096);
        prod.push_frames(&vec![7u32; 64 * CH]).unwrap();
        *sh.session.lock().unwrap() = Some(Session {
            generation: a,
            cons,
            decode_done: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });
        sh.priming.store(false, Ordering::Relaxed);

        // The decoder pushed B's boundary at frame 64 and has produced none of
        // it yet, so the second half of the callback finds an empty ring.
        let b = sh.roll_decode_generation();
        sh.note_boundary_pushed(64, b);

        let mut tap = SpectrumTap::new(
            CH as u16,
            PcmKind::Integer { valid_bits: 24 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut scratch = vec![0u32; 256 * CH];
        let got = render_frames(&sh, &mut scratch, &mut tap, CH as u16, 256);

        // A's 64 frames came out; B had nothing to give.
        assert_eq!(got, 64, "only A's remaining frames were available");
        assert_eq!(
            sh.generation(),
            b,
            "the render thread crossed; the UI was never asked"
        );
        assert!(
            sh.underruns.load(Ordering::Relaxed) > 0,
            "the second half of the buffer was silence the device played"
        );
        assert_eq!(
            sh.revoked_in(b),
            fault::UNDERRUN,
            "and it is B's dropout: B is what the listener was hearing"
        );
        assert_eq!(
            sh.revoked_in(a),
            fault::NONE,
            "A played every frame it had and lost nothing"
        );

        // The frames are attributed the same way.
        assert_eq!(
            sh.frames_played(),
            64,
            "B has been credited with nothing, because it produced nothing"
        );

        // And only now may the UI look. It could not have caused any of the
        // above, because it has not run.
        assert!(sh.take_crossed(), "the crossing is there to be consumed");
        assert!(!sh.take_crossed(), "and consumed once");
    }

    /// A reused stream does not inherit the previous track's drain.
    ///
    /// `drain_satisfied` latches "the device has played out what it was
    /// holding". It was cleared by `begin_drain` and nowhere else, so a stream
    /// kept open across tracks — which is the entire point of keeping it
    /// open — carried the last track's latch into the next one. A short
    /// gapless successor then completed the instant its boundary was crossed,
    /// on the strength of a drain that finished before it existed.
    #[test]
    fn a_reused_stream_does_not_inherit_the_last_tracks_drain() {
        let sh = Shared::new();
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(512, Ordering::Relaxed);

        // An old session that ran to its end and drained.
        let old = sh.begin_generation();
        sh.note_decode_eof(old);
        sh.begin_drain();
        for _ in 0..64 {
            sh.advance_drain(512);
        }
        assert_eq!(sh.completion(), Completion::CleanEof);
        assert!(
            sh.drain_satisfied.load(Ordering::Acquire),
            "this test needs the latch set, or it proves nothing"
        );

        // A new track on the same stream, with a short B queued behind it.
        let a = sh.begin_generation();
        assert!(
            !sh.drain_satisfied.load(Ordering::Acquire),
            "a fresh session starts with no drain credit"
        );
        let b = sh.roll_decode_generation();
        sh.note_boundary_pushed(0, b);
        // B's end of source is known before the device gets there — the
        // decoder is a ring ahead, which is the ordinary case.
        sh.note_decode_eof(b);
        assert_ne!(a, b);

        // Crossing alone must not finish B.
        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "B has not been played yet; only its source has been read"
        );
        assert_eq!(on_completion(sh.completion(), false), TickAction::Nothing);

        // B's own drain does.
        sh.begin_drain();
        for _ in 0..64 {
            sh.advance_drain(512);
        }
        assert_eq!(sh.completion(), Completion::CleanEof);
        assert_eq!(on_completion(sh.completion(), false), TickAction::Advance);
    }

    /// A callback descheduled past a track change does not credit the new
    /// track with the old one's frames.
    ///
    /// The schedule: the render thread reads its session and its generation,
    /// stops, a new session is installed and the counter zeroed, and the old
    /// callback resumes and accounts. With a bare counter the new track's
    /// position jumped forward by whatever the previous one had in flight.
    #[test]
    fn frames_from_a_superseded_callback_are_not_credited_to_the_new_track() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();
        sh.account_frames(a, 1_000);
        assert_eq!(sh.frames_played(), 1_000);

        let read = Arc::new(Barrier::new(2));
        let account = Arc::new(Barrier::new(2));
        let old = {
            let sh = Arc::clone(&sh);
            let read = Arc::clone(&read);
            let account = Arc::clone(&account);
            std::thread::spawn(move || {
                // The callback has its session and its generation in hand.
                let mine = a;
                read.wait();
                account.wait();
                // ...and only now does it account for what it emitted.
                sh.account_frames(mine, 512);
            })
        };

        read.wait();
        let b = sh.begin_generation();
        account.wait();
        old.join().unwrap();

        assert_eq!(sh.generation(), b);
        assert_eq!(
            sh.frames_played(),
            0,
            "B has played nothing, and A's frames in flight are not B's"
        );

        // B's own frames still count.
        sh.account_frames(b, 256);
        assert_eq!(sh.frames_played(), 256);
    }

    /// A superseded decoder cannot overwrite the queued track's fault.
    ///
    /// The check at the top of `fault_decoding` reads the decode clock and can
    /// be raced. A decoder still unwinding from a track that has been
    /// superseded would otherwise stamp its own, older generation over a fault
    /// the *queued* track had legitimately recorded — and the queued one is
    /// the track that is about to play.
    #[test]
    fn a_superseded_decoder_cannot_overwrite_a_newer_queued_fault() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        sh.begin_generation();
        let older = sh.roll_decode_generation();

        // The old decoder is inside the window: it has established that its
        // track is queued rather than playing, and has written nothing.
        let entered = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let entered = Arc::clone(&entered);
            let resume = Arc::clone(&resume);
            window::arm(sh.hook_id, window::Site::DeferFault, move || {
                entered.wait();
                resume.wait();
            });
        }
        let stale = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.fault_decoding(older, fault::SOURCE_READ))
        };

        entered.wait();
        // A seek supersedes everything: both clocks restart together, which is
        // also what clears the queue, so the one-boundary invariant holds.
        sh.begin_generation();
        // The new session queues a track of its own, and its decoder records a
        // fault against it.
        let newer = sh.roll_decode_generation();
        sh.note_boundary_pushed(0, newer);
        sh.fault_decoding(newer, fault::DECODE_ERROR);
        assert!(newer > older, "the queued track is the later one");

        // Only now does the superseded decoder get to write.
        resume.wait();
        stale.join().unwrap();

        // The device arrives at the track that is actually next.
        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(
            sh.fault(),
            fault::DECODE_ERROR,
            "the queued track's own fault, not the one a dead decoder wrote over it"
        );
    }

    /// A short gapless successor completes from events already recorded.
    ///
    /// The real schedule, and the one that used to lose the track. The decoder
    /// pushes B's boundary, decodes the whole of a short B, records its end of
    /// source, and stops; the render thread finds the ring dry with the
    /// decoder finished and begins the drain; the drain completes while the
    /// UI has not yet popped the boundary. B is queued, so the session must
    /// not end — but the drain *switched itself off and returned*, so the
    /// notice that the device had played out was gone. Nothing re-armed it,
    /// and the last track of every gapless run sat unfinished forever.
    ///
    /// Nothing below calls `note_decode_eof` or `begin_drain` for B. That is
    /// the point: B has to complete from what the decoder and the device
    /// already did, because in production nobody calls them a second time.
    #[test]
    fn a_short_gapless_successor_finishes_without_being_told_twice() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(512, Ordering::Relaxed);
        let a = sh.begin_generation();

        // The decoder crosses into B and runs out of source. Both facts are
        // recorded by the decode side, once, exactly as `decode_loop` records
        // them.
        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);
        sh.note_decode_eof(b);
        assert_ne!(a, b);

        // The render side: ring dry, decoder finished.
        let gate = Arc::new(Barrier::new(2));
        let render = {
            let sh = Arc::clone(&sh);
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                sh.begin_drain();
                // Far past any bound: if the drain were going to end this
                // session it has every opportunity here.
                for _ in 0..4096 {
                    sh.advance_drain(512);
                }
                gate.wait();
            })
        };

        gate.wait();
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "a track is still queued; the session is not over"
        );
        assert_eq!(
            on_completion(sh.completion(), false),
            TickAction::Nothing,
            "and the playlist must not move"
        );
        render.join().unwrap();

        // The UI catches up. This is the only call left, and it is the one the
        // UI really makes.
        let played = sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(played, b);
        assert_eq!(
            sh.completion(),
            Completion::CleanEof,
            "B played out before the boundary was popped; crossing it is when \
             that becomes true of B"
        );
        assert_eq!(on_completion(sh.completion(), false), TickAction::Advance);
    }

    /// The same schedule with the UI first, which must also work.
    ///
    /// Either side can be last to arrive, so the completion has to be
    /// attempted from both. Here the boundary is crossed before the drain
    /// finishes: B is playing, the drain completes against B, and B ends.
    #[test]
    fn the_boundary_may_also_arrive_before_the_drain_completes() {
        let sh = Shared::new();
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(512, Ordering::Relaxed);
        sh.begin_generation();

        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);
        sh.note_decode_eof(b);

        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "the device has not played out yet"
        );

        sh.begin_drain();
        for _ in 0..64 {
            sh.advance_drain(512);
        }
        assert_eq!(sh.completion(), Completion::CleanEof);
    }

    /// After a rollover, a dropout belongs to the track being heard.
    ///
    /// A session survives gapless boundaries — same ring, same decode thread,
    /// new track — so its own generation is the one the stream opened on and
    /// stays there. Stamping realtime faults with it meant that from the
    /// second track of a gapless album onwards every dropout was attributed to
    /// a track that had already finished, found stale and discarded: the
    /// render thread could not report anything at all.
    #[test]
    fn a_dropout_after_a_rollover_is_stamped_with_the_track_being_heard() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(512, Ordering::Relaxed);
        let a = sh.begin_generation();

        // A session installed for A, which is what the render thread will be
        // holding for the whole of the gapless run.
        let (mut prod, cons) = channel_ring::<u32>(2, 4096);
        prod.push_frames(&vec![0u32; 128]).unwrap();
        *sh.session.lock().unwrap() = Some(Session {
            generation: a,
            cons,
            decode_done: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });
        // Out of priming: a shortfall from here is a genuine dropout.
        sh.priming.store(false, Ordering::Relaxed);

        // The decoder pushes B and the device crosses into it.
        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);

        // The render thread is inside the callback when the boundary is
        // crossed; the barrier holds it there.
        let crossed = Arc::new(Barrier::new(2));
        let go = Arc::new(Barrier::new(2));
        let render = {
            let sh = Arc::clone(&sh);
            let crossed = Arc::clone(&crossed);
            let go = Arc::clone(&go);
            std::thread::spawn(move || {
                crossed.wait();
                go.wait();
                let mut tap = SpectrumTap::new(
                    2,
                    PcmKind::Integer { valid_bits: 24 },
                    Arc::new(Mutex::new(Vec::new())),
                    Arc::new(Mutex::new(Vec::new())),
                );
                let mut scratch = vec![0u32; 2048];
                // Asks for far more than the ring holds: a dropout.
                render_frames(&sh, &mut scratch, &mut tap, 2, 1024);
            })
        };

        crossed.wait();
        let played = sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(played, b);
        go.wait();
        render.join().unwrap();

        assert!(
            sh.underruns.load(Ordering::Relaxed) > 0,
            "the shortfall must have been counted as a dropout"
        );
        assert_eq!(
            sh.revoked_in(b),
            fault::UNDERRUN,
            "and recorded against B, which is the track the listener is hearing"
        );
        assert_eq!(
            sh.revoked_in(a),
            fault::NONE,
            "A finished cleanly and must not carry B's dropout"
        );
        assert_eq!(
            sh.revoked(),
            fault::UNDERRUN,
            "visible, on the current track"
        );
    }

    /// A fault the decoder publishes as the boundary is crossed is seen once.
    ///
    /// The check and the publish are two steps, and the device can cross
    /// between them: the decoder looks, finds A playing, is descheduled, the
    /// UI installs B, and only then does the fault land in the slot —
    /// `reach_boundary` having already looked at that slot and found it empty.
    /// B's decode failure was written down and never read.
    ///
    /// Both sides look again after acting and both claim through the same
    /// compare-exchange, so exactly one of them promotes it.
    #[test]
    fn a_fault_published_as_the_boundary_is_crossed_is_promoted_exactly_once() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        sh.begin_generation();
        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);

        // The decoder is held at exactly the instant that matters: it has
        // looked, found A playing and B queued, and written nothing. Racing
        // the two threads freely does not reach this — the publish almost
        // always completes before the boundary is read, so an unhooked version
        // of this test passes against the broken code.
        let entered = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let entered = Arc::clone(&entered);
            let resume = Arc::clone(&resume);
            window::arm(sh.hook_id, window::Site::DeferFault, move || {
                entered.wait();
                resume.wait();
            });
        }

        let decoder = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.fault_decoding(b, fault::DECODE_ERROR))
        };

        entered.wait();
        // The device arrives at B while the decoder is still inside the
        // window. `reach_boundary` reads the deferred slot and finds nothing,
        // because nothing has been written yet.
        let played = sh.cross_boundary_now().expect("the device crosses");
        resume.wait();
        decoder.join().unwrap();

        assert_eq!(played, b);
        assert_eq!(
            sh.fault(),
            fault::DECODE_ERROR,
            "B's decode failure must not be lost between the check and the publish"
        );
        assert_eq!(
            on_completion(sh.completion(), false),
            TickAction::Halt(fault::DECODE_ERROR)
        );
        // Claimed, so it cannot be promoted a second time against a later
        // track.
        assert_eq!(sh.take_deferred(b), None, "the fault was claimed, once");

        // And the other order, unhooked, must reach the same answer: the
        // decoder publishing before the device arrives is the ordinary case.
        let sh = Arc::new(Shared::new());
        sh.begin_generation();
        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);
        sh.fault_decoding(b, fault::SOURCE_READ);
        assert_eq!(sh.fault(), fault::NONE, "A is not at fault");
        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(sh.fault(), fault::SOURCE_READ);
        assert_eq!(sh.take_deferred(b), None, "promoted once, by the boundary");
    }


    /// A decode thread that panics inside a queued track blames that track.
    ///
    /// The schedule: the loop has rolled over into track B and is decoding it
    /// while the device is still inside track A; B's decode panics; the guard
    /// runs on the unwind. Stamped with the generation the loop was *born*
    /// in — A's — the fault was stale by the time it was written and vanished:
    /// the decoder was gone and nothing anywhere said so. Stamped with
    /// whatever was playing, it stopped A partway through for something wrong
    /// with a file that had not started.
    #[test]
    fn a_panic_in_a_queued_track_waits_for_that_track() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let guard = SessionGuard::new(
            Arc::clone(&done),
            Arc::clone(&sh),
            a,
            Arc::clone(&stop),
        );
        let logical = guard.logical_generation();

        // The loop reaches the gapless boundary and takes up track B.
        let b = sh.roll_decode_generation();
        // Frame 0: the device is standing on the boundary, so the next
        // render call crosses it.
        sh.note_boundary_pushed(0, b);
        logical.store(b, Ordering::Release);

        let gate = Arc::new(Barrier::new(2));
        let panicker = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let _guard = guard;
                gate.wait();
                panic!("the decoder fell over inside track B");
            })
        };

        gate.wait();
        assert!(panicker.join().is_err(), "the thread must actually panic");

        // Track A — the one the listener is hearing — is untouched.
        assert_eq!(sh.generation(), a);
        assert_eq!(sh.fault(), fault::NONE, "A did nothing wrong");
        assert_eq!(sh.completion(), Completion::Running);
        assert_eq!(on_completion(sh.completion(), false), TickAction::Nothing);
        assert!(
            done.load(Ordering::Acquire),
            "and the render side has been released rather than left waiting"
        );

        // The device arrives at B, and now it is B's problem.
        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(sh.fault(), fault::SESSION_THREAD_FAILED);
        assert_eq!(
            on_completion(sh.completion(), false),
            TickAction::Halt(fault::SESSION_THREAD_FAILED)
        );
    }

    /// A guard that outlives its session is silent.
    ///
    /// The other direction of the same rule: a decode thread still unwinding
    /// after a seek or a track change replaced it must not fault its
    /// successor.
    #[test]
    fn a_guard_from_a_replaced_session_faults_nothing() {
        let sh = Arc::new(Shared::new());
        let first = sh.begin_generation();
        let guard = SessionGuard::new(
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&sh),
            first,
            Arc::new(AtomicBool::new(false)),
        );
        let second = sh.begin_generation();
        drop(guard);
        assert_eq!(sh.generation(), second);
        assert_eq!(sh.fault(), fault::NONE);
        assert_eq!(sh.completion(), Completion::Running);
    }

    // -----------------------------------------------------------------------
    // Two clocks
    // -----------------------------------------------------------------------

    /// A stamped value is one word, so no reader can see half of a change.
    #[test]
    fn a_stamped_value_carries_its_generation_in_the_same_word() {
        for g in [0u64, 1, 2, 255, 256, 1 << 20, (1u64 << 55) - 1] {
            for c in [fault::NONE, fault::UNDERRUN, fault::BACKEND_DEAD, 255] {
                let v = stamped::pack(g, c);
                assert_eq!(stamped::generation(v), g, "generation for ({g}, {c})");
                assert_eq!(stamped::code(v), c, "code for ({g}, {c})");
            }
        }
        // Distinct pairs stay distinct — a code cannot be mistaken for part of
        // a generation or the reverse.
        assert_ne!(stamped::pack(1, 0), stamped::pack(0, 1));
        assert_ne!(stamped::pack(2, 1), stamped::pack(1, 2));
    }

    /// The decoder runs ahead; the clock the listener experiences does not.
    ///
    /// Advancing the playback clock when the *decoder* crossed a boundary
    /// rolled every visible fact over up to a full ring early: the badge, the
    /// source and the transform described the incoming track while the
    /// outgoing one was still audible.
    #[test]
    fn the_playback_clock_turns_when_the_device_arrives_not_when_the_decoder_does() {
        let sh = Shared::new();
        let start = sh.begin_generation();
        assert_eq!(sh.generation(), start);
        assert_eq!(
            sh.decode_generation(),
            start,
            "a fresh session starts level"
        );

        // The decoder begins the queued track.
        let pushed = sh.roll_decode_generation();
        assert_eq!(sh.decode_generation(), pushed);
        assert_eq!(
            sh.generation(),
            start,
            "the device is still inside the outgoing track"
        );

        // The device arrives.
        let played = sh.reach_boundary(pushed);
        assert_eq!(sh.generation(), played);
        assert_eq!(played, pushed, "the two clocks meet at the boundary");
    }

    /// A fault in the queued track is not shown against the track playing.
    ///
    /// The decoder can be a whole ring ahead, so a decode error in track B
    /// arrives while the listener is still hearing track A. Reporting it
    /// immediately named the wrong track — and, because a failure halts
    /// playback, stopped A partway through for something wrong with B.
    #[test]
    fn a_fault_in_the_queued_track_waits_until_that_track_plays() {
        let sh = Shared::new();
        sh.begin_generation();
        let playing = sh.generation();

        let queued = sh.roll_decode_generation();
        sh.fault_decoding(queued, fault::DECODE_ERROR);

        // Track A is untouched: still running, still unfaulted.
        assert_eq!(sh.fault(), fault::NONE, "the audible track is not at fault");
        assert_eq!(sh.completion(), Completion::Running);
        assert_eq!(
            on_completion(sh.completion(), false),
            TickAction::Nothing,
            "playback continues to the end of the track that is fine"
        );

        // The device reaches B, and now it is B's problem.
        sh.reach_boundary(queued);
        assert_eq!(sh.fault(), fault::DECODE_ERROR);
        assert_eq!(
            on_completion(sh.completion(), false),
            TickAction::Halt(fault::DECODE_ERROR)
        );
        let _ = playing;
    }

    /// Run one render callback and report how many frames it produced.
    ///
    /// The real seam, not a counter: this is the function the backend calls,
    /// and it is what crosses a boundary.
    fn render_through(sh: &Arc<Shared>, ch: usize, want: usize) -> usize {
        let mut tap = SpectrumTap::new(
            ch as u16,
            PcmKind::Integer { valid_bits: 24 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut scratch = vec![0u32; want * ch];
        render_frames(sh, &mut scratch, &mut tap, ch as u16, want)
    }

    /// A real PCM decode loop, parked between taking its successor off the
    /// queue and claiming the clock for it, publishes nothing into the
    /// session that replaced it.
    ///
    /// This is the production loop, not a stand-in: `prepare`, `decode_loop`,
    /// the frame ring and the real gapless queue. The hook sits exactly where
    /// the loop has committed to a hand-off and has not yet touched anything
    /// shared.
    ///
    /// The rollover used to mint its generation unconditionally, which is what
    /// makes the schedule dangerous rather than merely late: the parked thread
    /// allocated a number *newer* than the replacement session's and stored it
    /// as the decode clock, so every ownership check downstream compared
    /// against the superseded thread's own number and passed. A stop flag
    /// cannot close this — it is read before the same window.
    #[test]
    fn a_parked_pcm_successor_publishes_nothing_into_its_replacement() {
        use std::sync::Barrier;
        const SR: u32 = 48_000;

        // Two short Float32 tracks that can ride one stream.
        let pcm = |seed: i32| -> Vec<u8> {
            let mut v = Vec::new();
            for i in 0..256i32 {
                let s = ((i + seed) % 64) as f32 / 128.0;
                v.extend_from_slice(&s.to_le_bytes());
            }
            v
        };
        let ta = temp("i1_pcm_a.wav");
        let tb = temp("i1_pcm_b.wav");
        write_wav(&ta.0, &pcm(0), 32, true, SR, 2);
        write_wav(&tb.0, &pcm(9), 32, true, SR, 2);
        let prep_a = prepare(&ta.0, Duration::ZERO).expect("A prepares");
        let prep_b = prepare(&tb.0, Duration::ZERO).expect("B prepares");

        let shared = Arc::new(Shared::new());
        // A Float32 device, because the tracks are Float32 and the loop
        // re-checks compatibility at the boundary. Getting this wrong makes
        // the loop fault and break *before* the hook, so the fixture asserts
        // it rather than hanging on a barrier nobody reaches.
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_integer.store(false, Ordering::Relaxed);
        shared.out_rate.store(SR, Ordering::Relaxed);
        assert!(
            shared.can_carry_kind(prep_b.source.kind),
            "the fixture must let B ride A's stream, or the hand-off never happens"
        );
        let a = shared.begin_generation();
        *shared.next.lock().unwrap() = Some(prep_b);

        let reached = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let reached = Arc::clone(&reached);
            let resume = Arc::clone(&resume);
            window::arm(shared.hook_id, window::Site::Successor, move || {
                reached.wait();
                resume.wait();
            });
        }

        let (prod, _cons) = channel_ring::<u32>(2, 1 << 18);
        let done = Arc::new(AtomicBool::new(false));
        let logical = Arc::new(AtomicU64::new(a));
        let decoder = {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            let logical = Arc::clone(&logical);
            std::thread::spawn(move || {
                decode_loop(
                    prep_a,
                    prod,
                    done,
                    Arc::new(AtomicBool::new(false)),
                    shared,
                    a,
                    logical,
                )
            })
        };

        // The loop is holding B and has claimed nothing. A new track starts.
        reached.wait();
        let c = shared.begin_generation();
        assert!(c > a);
        resume.wait();
        decoder.join().expect("the decode thread returns");

        // The clock is C's, and B never became current.
        assert_eq!(
            shared.decode_generation(),
            c,
            "a superseded decoder must not mint a generation of its own"
        );
        assert_eq!(
            logical.load(Ordering::Acquire),
            a,
            "and its own view of which track it is on did not advance"
        );

        // Nothing it had to say reached the new session.
        assert!(
            shared.boundaries.lock().unwrap().is_empty(),
            "no boundary from a track that is no longer queued"
        );
        assert!(!shared.boundary_pending(), "and nothing pending behind C");
        assert!(shared.take_boundary().is_none());
        assert!(
            !shared.decode_eof_for(c),
            "B reaching the end of its source is not C reaching the end of its"
        );
        assert_eq!(shared.fault(), fault::NONE, "and C did not inherit a fault");
        assert_eq!(shared.off_grid_in(c), 0, "nor B's rounding evidence");
        assert!(
            matches!(shared.completion(), Completion::Running),
            "{:?}",
            shared.completion()
        );
    }

    /// The DoP loop, same schedule, same rule.
    ///
    /// The native and DoP routes are the ones a DSD listener is actually on,
    /// and the rollover they share is the same code path with a different
    /// reader in front of it. A repair that only reached the PCM loop would
    /// leave the format this release exists for on the broken side.
    #[test]
    fn a_parked_dop_successor_publishes_nothing_into_its_replacement() {
        use crate::dsd::tests::make_dsf;
        use std::sync::Barrier;

        // Two identical two-channel DSD64 files: one block each, so the
        // reader reaches the end quickly and the hand-off is the interesting
        // part.
        const BLOCK: u32 = 4096;
        let dsf = |fill: u8| -> Vec<u8> {
            let row = vec![vec![fill; BLOCK as usize], vec![!fill; BLOCK as usize]];
            make_dsf(
                2,
                2_822_400,
                1,
                BLOCK as u64 * 8,
                BLOCK,
                std::slice::from_ref(&row),
                None,
            )
        };
        let ta = temp("i1_dop_a.dsf");
        let tb = temp("i1_dop_b.dsf");
        std::fs::write(&ta.0, dsf(0x55)).unwrap();
        std::fs::write(&tb.0, dsf(0x33)).unwrap();

        let stream_a = crate::dsd::dop::open_dop_stream(&ta.0).expect("A opens");
        let stream_b = crate::dsd::dop::open_dop_stream(&tb.0).expect("B opens");
        // The loop re-checks compatibility at the boundary; if the fixture
        // fails it the loop faults and breaks before the hook, and the barrier
        // below would never be reached.
        assert_eq!(stream_a.carrier_rate(), stream_b.carrier_rate());
        assert_eq!(stream_a.info().channels, stream_b.info().channels);
        assert!(!stream_b.info().is_truncated(), "a truncated B never rolls");

        let shared = Arc::new(Shared::new());
        let a = shared.begin_generation();
        *shared.next_dop.lock().unwrap() = Some(stream_b);

        let reached = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let reached = Arc::clone(&reached);
            let resume = Arc::clone(&resume);
            window::arm(shared.hook_id, window::Site::Successor, move || {
                reached.wait();
                resume.wait();
            });
        }

        let (prod, _cons) = channel_ring::<u32>(2, 1 << 18);
        let logical = Arc::new(AtomicU64::new(a));
        let decoder = {
            let shared = Arc::clone(&shared);
            let logical = Arc::clone(&logical);
            std::thread::spawn(move || {
                dop_decode_loop(
                    stream_a,
                    prod,
                    Arc::new(AtomicBool::new(false)),
                    Arc::new(AtomicBool::new(false)),
                    shared,
                    a,
                    logical,
                )
            })
        };

        reached.wait();
        let c = shared.begin_generation();
        assert!(c > a);
        resume.wait();
        decoder.join().expect("the DoP thread returns");

        assert_eq!(
            shared.decode_generation(),
            c,
            "a superseded DoP decoder must not mint a generation of its own"
        );
        assert_eq!(logical.load(Ordering::Acquire), a);
        assert!(shared.boundaries.lock().unwrap().is_empty());
        assert!(!shared.boundary_pending());
        assert!(!shared.decode_eof_for(c));
        assert_eq!(shared.fault(), fault::NONE);
        assert_eq!(shared.off_grid_in(c), 0);
    }

    /// The window the old protocol could not represent.
    ///
    /// The schedule, forced, through `BpStream::evidence_at` — the call a
    /// caller actually makes:
    ///
    /// 1. the reader loads the counter and reads `revoked` as `NONE`;
    /// 2. the recoverable writer completes its compare-exchange and parks
    ///    *before* its publication is announced;
    /// 3. the terminal writer does the same;
    /// 4. the reader reads `fatal` and finds the failure.
    ///
    /// Under a counter bumped *after* each store, both fields had already
    /// changed and the counter had not moved: the reader saw an unchanged
    /// count across its loads and called the result a snapshot — a session
    /// that ended for a reason with nothing having gone wrong before it, which
    /// is the one account this whole read exists to make impossible.
    ///
    /// Under a window opened *before* each store, the reader can see that a
    /// publication is in flight. It does not get a quiet instant here, because
    /// the writers are parked inside their windows for as long as the test
    /// says so; what it gets is the union of what its attempts saw, which
    /// contains both records and is marked unsettled.
    #[test]
    fn a_reader_cannot_straddle_a_publication_that_has_not_announced_itself() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();

        let reader_between_loads = Arc::new(Barrier::new(2));
        let writers_have_stored = Arc::new(Barrier::new(2));
        {
            let one = Arc::clone(&reader_between_loads);
            let two = Arc::clone(&writers_have_stored);
            // Fires after `revoked` has been read and before `fatal` is.
            window::arm(sh.hook_id, window::Site::Evidence(2), move || {
                one.wait();
                two.wait();
            });
        }

        // The stream is built *inside* the reading thread, not moved into it.
        //
        // On Linux `Backend` holds a `cpal::Stream`, which is deliberately
        // neither `Send` nor `Sync`, so a `BpStream` cannot cross a thread
        // boundary there — and the Windows build, where the WASAPI handle is
        // `Send`, compiles a version of this test that does not exist on the
        // other platform. `Arc<Shared>` crosses freely, which is all the
        // adapter needs.
        let for_reader = Arc::clone(&sh);
        let reading = std::thread::spawn(move || {
            BpStream::for_evidence_test(for_reader, PayloadPlan::Q31).evidence_at(a)
        });

        reader_between_loads.wait();
        // Both writers complete their stores while the reader waits between
        // its loads. Neither has closed its window.
        sh.revoke(a, fault::UNDERRUN);
        sh.fail(a, fault::BACKEND_WRITE);
        writers_have_stored.wait();

        let ev = reading.join().expect("the read returns");
        assert_eq!(ev.generation, a);
        assert_eq!(ev.fatal, fault::BACKEND_WRITE, "the reason it ended");
        assert_eq!(
            ev.revoked,
            fault::UNDERRUN,
            "and the dropout stored before it — the account the old protocol \
             lost, because the store was visible and the announcement was not"
        );
    }

    /// Evidence published after the fatal record still arrives.
    ///
    /// The previous design's last attempt read the terminal record first, on
    /// the argument that nothing may be published against a generation after
    /// it. That argument does not hold in this code: `note_off_grid` and
    /// `revoke` both check only that the generation is still live, and a
    /// decode thread unwinding after a write failure is still inside a live
    /// generation. The ordering trick is gone; the fold is what answers.
    #[test]
    fn evidence_published_after_the_fatal_record_still_arrives() {
        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();

        sh.fail(a, fault::BACKEND_WRITE);
        // Both of these are reachable in production after a failure.
        sh.revoke(a, fault::UNDERRUN);
        sh.note_off_grid(a, 7);

        let stream = BpStream::for_evidence_test(Arc::clone(&sh), PayloadPlan::Q31);
        let ev = stream.evidence_at(a);
        assert_eq!(ev.fatal, fault::BACKEND_WRITE);
        assert_eq!(ev.revoked, fault::UNDERRUN, "published after the failure");
        assert_eq!(ev.off_grid, 7, "and so was this");
        assert!(ev.settled, "nothing is in flight, so the read is quiet");
    }

    /// A quiet attempt does not discard what a losing attempt already saw.
    ///
    /// One losing attempt, then one quiet one, with the generation's rounding
    /// evidence reclaimed in between. The quiet attempt reads zero — the slots
    /// belong to the next session now — and returning *it* would report a
    /// track as having rounded nothing when the first attempt had already
    /// counted eleven. Zero off-grid samples is the condition a value-exact
    /// label is granted on, so this is an upgrade produced by a reclamation.
    #[test]
    fn a_quiet_attempt_keeps_what_a_losing_attempt_saw() {
        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();
        sh.note_off_grid(a, 11);
        assert_eq!(sh.evidence_at(a).off_grid, 11, "the fixture records it");

        // One hook, so exactly one attempt is disturbed: it fires after the
        // off-grid load of attempt one, opens and closes a window — which
        // makes that attempt unsettled — and reclaims the slots.
        {
            let sh2 = Arc::clone(&sh);
            window::arm(sh.hook_id, window::Site::Evidence(1), move || {
                let _ = sh2.begin_generation();
            });
        }

        let stream = BpStream::for_evidence_test(Arc::clone(&sh), PayloadPlan::Q31);
        let ev = stream.evidence_at(a);
        assert_eq!(
            ev.off_grid, 11,
            "the count was seen before the reclamation, and a later quiet \
             attempt does not erase what an earlier one observed"
        );
        assert!(
            ev.settled,
            "and the answer is still marked settled, because the last attempt \
             was quiet"
        );
    }

    /// A rounding count that is mid-publication is never read as zero.
    ///
    /// The reason the window opens *before* the store rather than after it. A
    /// value-exact claim is granted on "no sample was rounded"; a count that
    /// has been announced and not yet stored would read as zero, and the label
    /// would be granted on a number the reader was in the middle of being
    /// told. The read comes back unsettled, and the one caller that grants a
    /// claim refuses to act on an unsettled read.
    #[test]
    fn a_rounding_count_in_flight_is_never_a_quiet_zero() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();

        let inside = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        {
            let one = Arc::clone(&inside);
            let two = Arc::clone(&release);
            // Inside `note_off_grid`, after its window has opened and before
            // the count is stored.
            window::arm(sh.hook_id, window::Site::OffGrid, move || {
                one.wait();
                two.wait();
            });
        }

        let writer = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.note_off_grid(a, 5))
        };
        inside.wait();

        let stream = BpStream::for_evidence_test(Arc::clone(&sh), PayloadPlan::Q31);
        let ev = stream.evidence_at(a);
        assert_eq!(ev.off_grid, 0, "the count genuinely is not stored yet");
        assert!(
            !ev.settled,
            "but the read knows it was taken while a publication was in \
             flight, which is what stops a zero here from earning a claim"
        );

        release.wait();
        writer.join().expect("the writer returns");
        let ev = stream.evidence_at(a);
        assert_eq!(ev.off_grid, 5);
        assert!(ev.settled);
    }

    /// `take_boundary` through the stream's own seam, for tests that hold an
    /// `Arc<Shared>` rather than a `BpStream`.
    fn sh_take(sh: &Arc<Shared>) -> Option<Boundary> {
        sh.take_boundary()
    }

    /// A real PCM successor publishes wholly while a reset waits for the lock.
    ///
    /// The success path, in production code: `prepare`, `decode_loop`, the
    /// real gapless queue, and `roll_and_publish` claiming the clock and
    /// running its four stores. The previous version of this test reached a
    /// test-only copy of those stores, so nothing ever executed the production
    /// ones after a successful compare-exchange.
    ///
    /// The reset is released on a handshake rather than a sleep: it fires a
    /// hook immediately before it contends for the publication lock, so by the
    /// time this thread lets the publisher go, the reset has provably reached
    /// the lock and is waiting on it.
    #[test]
    fn a_real_successor_publishes_wholly_while_a_reset_waits() {
        use std::sync::Barrier;
        const SR: u32 = 48_000;

        let pcm = |seed: i32| -> Vec<u8> {
            let mut v = Vec::new();
            for i in 0..256i32 {
                let s = ((i + seed) % 64) as f32 / 128.0;
                v.extend_from_slice(&s.to_le_bytes());
            }
            v
        };
        let ta = temp("j1_pub_a.wav");
        let tb = temp("j1_pub_b.wav");
        write_wav(&ta.0, &pcm(0), 32, true, SR, 2);
        write_wav(&tb.0, &pcm(9), 32, true, SR, 2);
        let prep_a = prepare(&ta.0, Duration::ZERO).expect("A prepares");
        let prep_b = prepare(&tb.0, Duration::ZERO).expect("B prepares");

        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_integer.store(false, Ordering::Relaxed);
        shared.out_rate.store(SR, Ordering::Relaxed);
        assert!(shared.can_carry_kind(prep_b.source.kind));
        let a = shared.begin_generation();
        let b_path = tb.0.clone();
        *shared.next.lock().unwrap() = Some(prep_b);

        // The publisher parks *inside* the transaction, after the claim has
        // succeeded and before the first store.
        let claimed = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        {
            let claimed = Arc::clone(&claimed);
            let release = Arc::clone(&release);
            window::arm(shared.hook_id, window::Site::BoundaryPublish(0), move || {
                claimed.wait();
                release.wait();
            });
        }

        let (prod, _cons) = channel_ring::<u32>(2, 1 << 18);
        let logical = Arc::new(AtomicU64::new(a));
        let decoder = {
            let shared = Arc::clone(&shared);
            let logical = Arc::clone(&logical);
            std::thread::spawn(move || {
                decode_loop(
                    prep_a,
                    prod,
                    Arc::new(AtomicBool::new(false)),
                    Arc::new(AtomicBool::new(false)),
                    shared,
                    a,
                    logical,
                )
            })
        };

        claimed.wait();
        // Mid-transaction: nothing coherent to consume, and the UI does not
        // block waiting for one.
        assert!(sh_take(&shared).is_none());

        // The reset, from its own thread, and released on the handshake it
        // fires immediately before contending for the lock.
        let contending = Arc::new(Barrier::new(2));
        {
            let contending = Arc::clone(&contending);
            window::arm(shared.hook_id, window::Site::ResetContend, move || {
                contending.wait();
            });
        }
        let reset_done = Arc::new(AtomicBool::new(false));
        let resetter = {
            let shared = Arc::clone(&shared);
            let reset_done = Arc::clone(&reset_done);
            std::thread::spawn(move || {
                let g = shared.begin_generation();
                reset_done.store(true, Ordering::Release);
                g
            })
        };
        contending.wait();
        assert!(
            !reset_done.load(Ordering::Acquire),
            "the reset has reached the lock and cannot have passed it: the \
             publisher is holding it"
        );

        release.wait();
        decoder.join().expect("the decode thread returns");
        let c = resetter.join().expect("the reset returns");

        // The publisher won the lock, so it published *wholly* — and the reset
        // then cleared the whole of it. Never half of each.
        assert!(c > a);
        assert_eq!(shared.decode_generation(), c);
        assert!(
            shared.boundaries.lock().unwrap().is_empty(),
            "the reset cleared the metadata"
        );
        assert!(
            !shared.boundary_pending(),
            "and the count, which is the half that used to survive"
        );
        assert_eq!(shared.frames_to_boundary(0), None, "and the descriptor");
        assert!(sh_take(&shared).is_none());
        let _ = b_path;
    }

    /// A boundary crossing during the evidence read cannot mix two tracks.
    ///
    /// The four facts used to be separate getters behind a generation check.
    /// The check narrowed the window and could not close it: the window is
    /// *between the reads*, and this test puts a real crossing there — once
    /// between the off-grid count and the recoverable code, and once between
    /// the recoverable code and the fatal one.
    ///
    /// A is playing, has rounded samples and has dropped out. B is queued,
    /// clean. Whatever the device does mid-read, the four fields that come
    /// back are A's or they are nobody's.
    #[test]
    fn evidence_cannot_be_assembled_out_of_two_tracks() {
        for stage in 1u8..=2 {
            let sh = Arc::new(Shared::new());
            let a = sh.begin_generation();

            // A's own evidence: it rounded, and it dropped out.
            sh.note_off_grid(a, 11);
            sh.revoke(a, fault::UNDERRUN);
            assert_eq!(sh.evidence_at(a).off_grid, 11);

            // B, queued and about to become the playing generation — and it
            // has to *differ* from A in every field, or a mixed read is
            // indistinguishable from a correct one. The first version of this
            // test left B's fatal code empty, which is also A's, so a crossing
            // between the recoverable and fatal reads changed nothing it could
            // see and the mutant survived stage 2.
            let b = sh.roll_decode_generation();
            sh.fault_decoding(b, fault::DECODE_ERROR);
            assert!(sh.note_boundary_pushed(0, b));

            // The device crosses in the middle of the read.
            {
                let cross = Arc::clone(&sh);
                window::arm(sh.hook_id, window::Site::Evidence(stage), move || {
                    cross.cross_boundary_now().expect("the device crosses");
                });
            }

            let ev = sh.evidence_at(a);
            assert_eq!(sh.generation(), b, "stage {stage}: the crossing happened");
            assert_eq!(ev.generation, a);
            assert_eq!(
                ev.off_grid, 11,
                "stage {stage}: A's rounding, read against A"
            );
            assert_eq!(
                ev.revoked,
                fault::UNDERRUN,
                "stage {stage}: A's dropout, read against A"
            );
            assert_eq!(
                ev.fatal,
                fault::NONE,
                "stage {stage}: A did not fail — B's decode error is not A's"
            );

            // And B, asked about separately, answers for itself alone.
            let evb = sh.evidence_at(b);
            assert_eq!(evb.generation, b);
            assert_eq!(evb.off_grid, 0, "stage {stage}: B has rounded nothing");
            assert_eq!(evb.revoked, fault::NONE, "stage {stage}: B has lost nothing");
            assert_eq!(
                evb.fatal,
                fault::DECODE_ERROR,
                "stage {stage}: B's decode error, promoted when the device                  crossed into it — which is what makes A's clean fatal field                  a fact and not a coincidence"
            );
        }
    }

    /// A publisher parked between claiming the clock and touching the queue is
    /// wholly published and then wholly cleared — never half of each.
    ///
    /// Publication used to be one check followed by four separate stores, so a
    /// reset landing between them cleared a queue the publisher then pushed
    /// into and zeroed a count the publisher then incremented: a boundary
    /// belonging to a dead track sitting in a live session's queue, behind a
    /// pending count that never came back down. `boundary_pending` then
    /// answered "a track is still queued" for the rest of the session, which
    /// is a session that can never end cleanly.
    ///
    /// The reset takes the same lock, so the two orderings are the only two
    /// there are, and both are checked here.
    #[test]
    fn a_publisher_and_a_reset_cannot_interleave() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();

        let claimed = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        {
            let claimed = Arc::clone(&claimed);
            let release = Arc::clone(&release);
            window::arm(sh.hook_id, window::Site::BoundaryPublish(0), move || {
                claimed.wait();
                release.wait();
            });
        }

        let publisher = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.note_boundary_pushed(64, sh.roll_decode_generation()))
        };

        claimed.wait();
        // Mid-transaction: the UI must not see a partial hand-off, and it must
        // not block waiting for one either.
        assert!(
            sh.take_boundary().is_none(),
            "a half-published boundary is not something to consume"
        );

        // The reset, from its own thread, so nothing in this one is what
        // serialises the two.
        let reset_done = Arc::new(AtomicBool::new(false));
        let resetter = {
            let sh = Arc::clone(&sh);
            let reset_done = Arc::clone(&reset_done);
            std::thread::spawn(move || {
                let g = sh.begin_generation();
                reset_done.store(true, Ordering::Release);
                g
            })
        };

        // It cannot get in, and this is the assertion that says so.
        //
        // A bounded wait is the only way to observe blocking, and it is what
        // makes the rest of this test deterministic rather than a race: if the
        // reset were able to run here, it would, and every assertion below
        // would then be about a schedule that had already happened.
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            !reset_done.load(Ordering::Acquire),
            "a reset must not land between a publisher's claim and its stores"
        );

        release.wait();
        let published = publisher.join().expect("the publisher returns");
        let c = resetter.join().expect("the reset returns");

        assert!(published, "the publisher won the lock, so it published wholly");
        assert!(c > a);
        assert_eq!(sh.decode_generation(), c);
        assert!(
            sh.boundaries.lock().unwrap().is_empty(),
            "and the reset cleared the whole of it"
        );
        assert!(
            !sh.boundary_pending(),
            "including the count — a pending boundary that outlives its queue \
             is a session that can never finish"
        );
        assert_eq!(
            sh.frames_to_boundary(0),
            None,
            "and the realtime descriptor"
        );
        assert!(sh.take_boundary().is_none());
    }

    /// A stale writer cannot erase its successor's end-of-source.
    ///
    /// `note_decode_eof` was a check followed by a store. A decoder parked
    /// between the two came back to find its track replaced and stored anyway,
    /// putting *its* generation's answer over the successor's — so
    /// `decode_eof_for(successor)` became false and the track that had
    /// genuinely finished never ended. The session sat at almost-complete
    /// until the drain's wall clock failed it, and a failure never advances a
    /// playlist.
    #[test]
    fn a_stale_writer_cannot_erase_the_successors_end_of_source() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let b = sh.begin_generation();

        let validated = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let validated = Arc::clone(&validated);
            let resume = Arc::clone(&resume);
            window::arm(sh.hook_id, window::Site::DecodeEof, move || {
                validated.wait();
                resume.wait();
            });
        }

        let stale = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.note_decode_eof(b))
        };

        // B has passed its liveness check and written nothing.
        validated.wait();
        let c = sh.begin_generation();
        sh.note_decode_eof(c);
        assert!(sh.decode_eof_for(c), "C reached the end of its source");
        resume.wait();
        stale.join().expect("the stale writer returns");

        assert!(
            sh.decode_eof_for(c),
            "and it is still true after the superseded writer woke up"
        );
        assert!(
            !sh.decode_eof_for(b),
            "B's answer is about a track nobody is playing"
        );
    }

    /// A stale writer cannot erase its successor's rounding evidence.
    ///
    /// The same shape, on the counter that decides the value-exact label.
    /// Validation happened once and the store was raw, so a decoder parked
    /// between them wrote its own count into a slot the replacement had
    /// meanwhile claimed — and the new track then reported the old one's
    /// rounding as its own.
    #[test]
    fn a_stale_writer_cannot_erase_the_successors_rounding_evidence() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let b = sh.begin_generation();
        // B owns a slot already, so the parked write below is an update of a
        // slot it holds rather than a claim of a free one.
        sh.note_off_grid(b, 3);
        assert_eq!(sh.off_grid_in(b), 3);

        let validated = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        {
            let validated = Arc::clone(&validated);
            let resume = Arc::clone(&resume);
            window::arm(sh.hook_id, window::Site::OffGrid, move || {
                validated.wait();
                resume.wait();
            });
        }

        let stale = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || sh.note_off_grid(b, 77))
        };

        validated.wait();
        let c = sh.begin_generation();
        sh.note_off_grid(c, 5);
        assert_eq!(sh.off_grid_in(c), 5, "C's own evidence, before B wakes");
        resume.wait();
        stale.join().expect("the stale writer returns");

        assert_eq!(
            sh.off_grid_in(c),
            5,
            "a superseded decoder must not write its rounding into the track \
             that replaced it — that count decides a value-exact label"
        );
        assert_eq!(
            sh.off_grid_in(b),
            0,
            "and B has no slot of its own left to report from"
        );
    }

    /// A failure never advances a playlist, whether or not the track dropped
    /// out on the way to it.
    ///
    /// The completion half of the integrity split. The badge half is asserted
    /// against `OutputSessionState` in `main`; this is the half that decides
    /// whether the next track starts, which is what a listener notices.
    #[test]
    fn a_failure_never_advances_with_or_without_a_dropout_before_it() {
        for before in [None, Some(fault::UNDERRUN)] {
            let sh = Shared::new();
            let g = sh.begin_generation();
            if let Some(code) = before {
                sh.revoke(g, code);
            }
            sh.fail_now(fault::BACKEND_WRITE);
            assert_eq!(
                sh.completion().failure(),
                Some(fault::BACKEND_WRITE),
                "the reason is the write, not the dropout, whatever came first"
            );
            assert_eq!(
                on_completion(sh.completion(), false),
                TickAction::Halt(fault::BACKEND_WRITE),
                "a failure stops, once — it never advances a playlist"
            );
            // And a later clean end cannot turn it back into an advance.
            sh.finish_clean(g);
            assert!(!sh.completion().may_advance());
        }
    }

    /// A recoverable loss costs the claim and not the advance.
    #[test]
    fn a_recoverable_loss_still_earns_the_advance() {
        for code in [
            fault::UNDERRUN,
            fault::CALLBACK_LOCK_MISS,
            fault::VALUE_EXACT_VIOLATION,
        ] {
            assert!(!fault::is_fatal(code), "{code} must be survivable");
            let sh = Shared::new();
            let g = sh.begin_generation();
            sh.revoke(g, code);
            assert_eq!(
                sh.fault(),
                fault::NONE,
                "code {code}: a loss is evidence, not a reason the track stopped"
            );
            sh.note_decode_eof(g);
            sh.finish_clean(g);
            assert_eq!(sh.completion(), Completion::CleanEof, "code {code}");
            assert_eq!(
                on_completion(sh.completion(), false),
                TickAction::Advance,
                "code {code}: a dropout must not strand the playlist"
            );
            assert_eq!(sh.revoked(), code, "and the loss is still on the record");
        }
    }

    /// A decoder parked through a replacement publishes nothing into it.
    ///
    /// The schedule, forced rather than hoped for: A is playing, the decoder
    /// rolls forward to B and queues it, and is then parked — after the
    /// rollover, holding B's generation and everything it is about to say
    /// about it. A `start()` runs while it is parked and replaces both tracks
    /// with C, clearing every slot the parked thread was going to write into.
    /// Only then is it resumed.
    ///
    /// This passed against the broken code for the whole of the previous
    /// phase, because the two clocks invented their own successors: the
    /// decode clock had reached B by adding one to itself, and `start()`
    /// reached C by adding one to the *playback* clock, which was still on A.
    /// C and B were the same number. Every stamp check in this file compared
    /// them, found them equal, and let B's end-of-source, B's rounding
    /// evidence, B's parked fault and B's boundary through — addressed to a
    /// generation that now named a completely different file.
    ///
    /// Allocated generations are what make the check mean anything, so this
    /// asserts the uniqueness as well as the consequences.
    #[test]
    fn a_decoder_parked_through_a_replacement_publishes_nothing_into_it() {
        use std::sync::Barrier;

        let sh = Arc::new(Shared::new());
        let a = sh.begin_generation();

        let queued = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));

        let producer = {
            let sh = Arc::clone(&sh);
            let queued = Arc::clone(&queued);
            let resume = Arc::clone(&resume);
            std::thread::spawn(move || {
                // The decoder has finished A's source and begun the queued
                // track.
                let b = sh.roll_decode_generation();
                queued.wait();
                // Parked here, with B's generation in hand, for as long as the
                // main thread wants.
                resume.wait();
                // Everything B had left to say, in the order a decode thread
                // would say it.
                let published = sh.note_boundary_pushed(64, b);
                sh.note_off_grid(b, 31);
                sh.note_decode_eof(b);
                sh.fault_decoding(b, fault::DECODE_ERROR);
                (b, published)
            })
        };

        queued.wait();
        // `start()` on a new file, while the decoder is parked.
        let c = sh.begin_generation();
        resume.wait();
        let (b, published) = producer.join().expect("the decode thread returns");

        // The consequences first, because they are what a listener would
        // notice; the numbers that cause them are checked at the end.
        assert!(!published, "a superseded decoder may not queue a boundary");
        assert_eq!(sh.off_grid_in(c), 0, "B's rounding is not C's rounding");
        assert_eq!(sh.off_grid_in(b), 0, "and B has no slot of its own left");
        assert!(
            !sh.decode_eof_for(c),
            "B reaching the end of its source is not C reaching the end of its"
        );
        assert_eq!(sh.fault(), fault::NONE, "B's decode error is not C's");
        assert!(
            matches!(sh.completion(), Completion::Running),
            "C is playing: {:?}",
            sh.completion()
        );
        assert!(sh.take_deferred(c).is_none(), "nothing parked against C");

        // Nothing reached the boundary machinery either: no metadata, no
        // pending count, nothing for the UI to consume.
        assert!(!sh.boundary_pending(), "no track is queued behind C");
        assert!(sh.take_boundary().is_none(), "and none is waiting to be described");
        assert_eq!(
            sh.generation(),
            c,
            "and the playing generation is still the one start() installed"
        );

        // And the cause: three sessions, three numbers, each greater than the
        // last. Every refusal above is a stamp comparison, and a stamp
        // comparison is only worth something while a number names one session.
        assert!(
            a < b && b < c,
            "generations are allocated, not derived: {a} {b} {c}"
        );
    }

    /// Publishing a boundary is ordered, and the order survives being cut at
    /// every seam in it.
    ///
    /// The producer is parked after each stage in turn — metadata enqueued,
    /// pending counted, generation stored — while the render thread renders
    /// through the boundary frame and the UI polls for something to describe.
    /// Whatever the cut, the hand-off must roll exactly once, the UI must get
    /// the metadata for it, and the pending count must come back to zero.
    ///
    /// The count is the assertion that catches the original order. Publishing
    /// the realtime descriptor before incrementing the pending count let the
    /// render thread cross a boundary that had not been counted yet: the
    /// release found zero and did nothing, the producer then counted it, and
    /// `boundary_pending` answered "a track is still queued" for the rest of
    /// the session — which is a session that can never end cleanly, so the
    /// last track of every gapless run stops the playlist.
    #[test]
    fn a_boundary_survives_being_cut_at_every_publication_stage() {
        use std::sync::Barrier;
        const CH: usize = 2;

        for stage in 1u8..=3 {
            let sh = Arc::new(Shared::new());
            let a = sh.begin_generation();

            // A real session with 128 frames of A in the ring; the boundary is
            // at frame 64, so one 256-frame callback covers A's tail, the
            // crossing, and the empty ring beyond it.
            let (mut prod, cons) = channel_ring::<u32>(CH, 4096);
            prod.push_frames(&vec![9u32; 128 * CH]).unwrap();
            *sh.session.lock().unwrap() = Some(Session {
                generation: a,
                cons,
                decode_done: Arc::new(AtomicBool::new(false)),
                stop: Arc::new(AtomicBool::new(false)),
            });
            sh.priming.store(false, Ordering::Relaxed);

            let parked = Arc::new(Barrier::new(2));
            let resume = Arc::new(Barrier::new(2));
            {
                let parked = Arc::clone(&parked);
                let resume = Arc::clone(&resume);
                window::arm(sh.hook_id, window::Site::BoundaryPublish(stage), move || {
                    parked.wait();
                    resume.wait();
                });
            }

            let producer = {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    let b = sh.roll_decode_generation();
                    assert!(sh.note_boundary_pushed(64, b));
                    b
                })
            };

            parked.wait();
            // The device and the UI, both running while the producer is
            // halfway through publishing.
            let mut seen = Vec::new();
            let mut played = render_through(&sh, CH, 256);
            if let Some(b) = sh.take_boundary() {
                seen.push(b.generation);
            }
            resume.wait();
            let b = producer.join().expect("the producer returns");

            // And again, now that publication has completed.
            played += render_through(&sh, CH, 256);
            while let Some(got) = sh.take_boundary() {
                seen.push(got.generation);
            }

            assert_eq!(
                seen,
                vec![b],
                "stage {stage}: the hand-off rolls exactly once, and the UI gets it"
            );
            assert_eq!(
                sh.generation(),
                b,
                "stage {stage}: and the playing generation is the queued one"
            );
            assert!(
                !sh.boundary_pending(),
                "stage {stage}: the pending count came back to zero"
            );
            assert_eq!(
                played, 128,
                "stage {stage}: A's 128 frames came out, and nothing was invented past them"
            );
        }
    }

    /// A fault in the track that *is* playing is immediate.
    #[test]
    fn a_fault_in_the_playing_track_is_not_deferred() {
        let sh = Shared::new();
        let g = sh.begin_generation();
        sh.fault_decoding(g, fault::SOURCE_READ);
        assert_eq!(sh.fault(), fault::SOURCE_READ);
        assert!(sh.completion().is_over());
    }

    /// Rounding evidence belongs to the track it was observed in — including
    /// when the two tracks in flight happen to share a parity.
    ///
    /// One counter meant the incoming track's rounding was attributed to the
    /// outgoing one — which is the track still being heard, and the track
    /// whose value-exact label the counter decides. Stamping the counts fixed
    /// the attribution but not the *storage*: the slot was still chosen by
    /// `generation & 1`, which is only a distinct answer for two tracks while
    /// their numbers are adjacent. Generations are allocated now, so anything
    /// that consumes a number without playing it — a seek between two tracks —
    /// leaves the playing and queued generations the same parity, and the
    /// queued track's "start from zero" landed on the playing track's slot.
    #[test]
    fn rounding_evidence_survives_a_same_parity_successor() {
        let sh = Shared::new();
        let a = sh.begin_generation();
        sh.note_off_grid(a, 17);

        // Burn generations until the queued track's number shares a parity
        // with the playing one, as a seek between the two tracks would.
        //
        // How many it takes is not knowable in advance: the allocator is
        // process-wide and the test suite runs in parallel, so another test
        // opening a session between two of these calls shifts the parity. This
        // burnt exactly one and asserted the result — which held on Windows by
        // scheduling luck and failed on Linux, where more tests are compiled
        // and the contention is higher. Rolling until the condition is
        // actually met is what the test needs; it is the condition it is
        // about.
        let mut b = 0u64;
        for _ in 0..64 {
            b = sh.roll_decode_generation();
            if b & 1 == a & 1 {
                break;
            }
        }
        assert_eq!(
            a & 1,
            b & 1,
            "this test is about the case the old slot index could not tell apart"
        );
        assert!(b > a, "the allocator hands out each number once");

        assert_eq!(
            sh.off_grid_in(b),
            0,
            "the incoming track starts from a clean sheet"
        );
        assert_eq!(
            sh.off_grid_in(a),
            17,
            "and the outgoing track's evidence survives while it is audible"
        );

        // The device crosses: the label now depends on B's evidence, which is
        // clean, so an off-grid predecessor does not cost its successor the
        // claim.
        assert!(sh.note_boundary_pushed(0, b));
        sh.cross_boundary_now().expect("the device crosses");
        assert_eq!(sh.off_grid_in(sh.generation()), 0);

        // A superseded decoder has no slot to write into at all.
        sh.note_off_grid(a, 999);
        assert_eq!(
            sh.off_grid_in(a),
            17,
            "a generation that is neither playing nor decoding may not claim a slot"
        );
    }

    // -----------------------------------------------------------------------
    // Terminal lifecycle
    // -----------------------------------------------------------------------

    /// The gate, stated once: only a clean end of file advances anything.
    ///
    /// Every fatal path used to arrive at the UI on the same flag the end of a
    /// track arrives on, because that flag's job was "stop waiting for audio".
    /// The tick read it as "the track ended" and started the next one — which
    /// under Repeat One is the same file. A user's log shows the result: one
    /// deterministically failing track opened hundreds of times in a row.
    #[test]
    fn only_a_clean_end_of_file_advances_playback() {
        assert_eq!(
            on_completion(Completion::Running, false),
            TickAction::Nothing
        );
        assert_eq!(
            on_completion(Completion::CleanEof, false),
            TickAction::Advance
        );
        assert_eq!(
            on_completion(Completion::CleanEof, true),
            TickAction::StopAtEndOfTrack,
            "the sleep timer stops at the end of a track that actually ended"
        );

        // Every fatal code, under both sleep settings, halts and never
        // advances. `NONE` is excluded deliberately: it is not a failure.
        for reason in [
            fault::UNDERRUN,
            fault::RING_INVARIANT,
            fault::DECODE_ERROR,
            fault::SOURCE_READ,
            fault::SOURCE_FORMAT_CHANGE,
            fault::BACKEND_WRITE,
            fault::BACKEND_SPACE,
            fault::BACKEND_RESET,
            fault::CALLBACK_LOCK_MISS,
            fault::PRIMING_TIMEOUT,
            fault::SESSION_THREAD_FAILED,
            fault::BACKEND_DEAD,
            fault::VALUE_EXACT_VIOLATION,
        ] {
            for sleeping in [false, true] {
                let action = on_completion(
                    Completion::Failed {
                        reason,
                        generation: 7,
                    },
                    sleeping,
                );
                assert_eq!(
                    action,
                    TickAction::Halt(reason),
                    "fault {reason} must halt, not advance (sleep = {sleeping})"
                );
                assert_ne!(action, TickAction::Advance);
            }
        }
    }

    /// Repeat One is the loop mode that turns a single advance into an
    /// unbounded one, so it gets its own statement of the same rule.
    ///
    /// The loop mode is not consulted at all on a failure: the decision to
    /// advance is made before any index is chosen, and on a failure it is
    /// never made.
    #[test]
    fn a_failed_track_is_never_reopened_under_any_loop_mode() {
        // Repeat One, Repeat All and Sequential all reach the tick through the
        // same reducer, and the reducer never sees the mode. What it returns
        // on a failure is therefore the same for all three, which is the
        // property that matters: no mode can turn a failure into a reopen.
        let failed = Completion::Failed {
            reason: fault::DECODE_ERROR,
            generation: 3,
        };
        assert_eq!(
            on_completion(failed, false),
            TickAction::Halt(fault::DECODE_ERROR)
        );

        // And a hundred consecutive ticks on the same failed session produce a
        // hundred halts and not one advance — the shape of the log the user
        // sent, inverted.
        for _ in 0..100 {
            assert!(!matches!(on_completion(failed, false), TickAction::Advance));
        }
    }

    /// A session that failed stays failed for as long as it is the session.
    #[test]
    fn a_failure_is_reported_once_and_stays_reported() {
        let sh = Shared::new();
        let g = sh.generation();
        sh.fail(g, fault::SOURCE_READ);

        // Polled repeatedly, as the tick does, it keeps the same answer and
        // never decays into something that would advance.
        for _ in 0..50 {
            let c = sh.completion();
            assert_eq!(c.failure(), Some(fault::SOURCE_READ));
            assert!(!c.may_advance());
        }

        // A second, different fault does not overwrite the first: the reason
        // the user is shown is the reason it went wrong, not the last symptom.
        sh.fail(g, fault::BACKEND_WRITE);
        assert_eq!(sh.completion().failure(), Some(fault::SOURCE_READ));
    }

    /// A seek replaces the session. That must not fault the replacement, and
    /// the outgoing thread must not be able to reach forward into it.
    #[test]
    fn a_seek_replacement_does_not_fault_its_successor() {
        let sh = Arc::new(Shared::new());
        let first = sh.generation();

        // The outgoing session's guard, holding the generation it was born in
        // and the stop flag its replacement set.
        let done = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let guard = SessionGuard::new(Arc::clone(&done), Arc::clone(&sh), first, Arc::clone(&stop));

        // A seek: the new session begins, and the old one is told to stop.
        let second = sh.begin_generation();
        stop.store(true, Ordering::Release);
        drop(guard);

        assert_eq!(
            sh.completion(),
            Completion::Running,
            "an intentional replacement is not a failure"
        );
        assert_eq!(sh.fault(), fault::NONE);
        assert!(
            done.load(Ordering::Acquire),
            "and the render side is still released, or it waits forever"
        );
        let _ = second;
    }

    /// A decode thread that panics *after* its session was replaced faults
    /// nothing.
    ///
    /// This is the ordinary track change, not an exotic race: a decode thread
    /// is torn down by its successor being installed, so its guard runs after
    /// the new generation has already begun.
    #[test]
    fn a_late_panicking_session_cannot_fault_the_one_that_replaced_it() {
        let sh = Arc::new(Shared::new());
        let first = sh.generation();
        let done = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let guard = SessionGuard::new(Arc::clone(&done), Arc::clone(&sh), first, Arc::clone(&stop));

        sh.begin_generation(); // the successor is installed

        // The old thread now panics — `stop` was never set, so this is a
        // genuine abnormal end and not a teardown.
        drop(guard);

        assert_eq!(
            sh.completion(),
            Completion::Running,
            "the successor is untouched"
        );
        assert_eq!(sh.fault(), fault::NONE);

        // The same guard, without a replacement, does fault — so the test
        // above is not passing merely because the guard does nothing.
        let sh2 = Arc::new(Shared::new());
        let g2 = sh2.generation();
        let done2 = Arc::new(AtomicBool::new(false));
        drop(SessionGuard::new(
            Arc::clone(&done2),
            Arc::clone(&sh2),
            g2,
            Arc::new(AtomicBool::new(false)),
        ));
        assert_eq!(
            sh2.completion().failure(),
            Some(fault::SESSION_THREAD_FAILED)
        );
        assert!(done2.load(Ordering::Acquire));
    }

    // -----------------------------------------------------------------------
    // The integer-only endpoint
    // -----------------------------------------------------------------------

    /// The regression from a real SMSL C200 Pro log, at the routing seam.
    ///
    /// The DAC advertises "32-bit" and means 32-bit *integer*: it rejects
    /// `32f` in exclusive mode and accepts `32i`. Under Strict that is a stop.
    /// Under Automatic and HQ it is one rejection followed by one successful
    /// open — **not** a loop, and not a fault.
    #[test]
    fn an_integer_only_endpoint_is_offered_the_conversion_exactly_once() {
        // The endpoint: it answers the format query, and says no to float.
        let integer_only = |plan: PayloadPlan| -> Result<PayloadPlan, OpenError> {
            match plan.device_kind(PcmKind::Float32) {
                PcmKind::Float32 => Err(OpenError::device_format(
                    "\"SMSL C200 Pro\" rejected 32f / 2ch in exclusive mode",
                )),
                _ => Ok(plan),
            }
        };

        // Strict: one attempt, one rejection, and it stops there. It does not
        // try a conversion, because it was told never to transform.
        let mut attempts = Vec::new();
        let r = walk_ladder(PcmKind::Float32, state::OutputPolicy::StrictExact, |p| {
            attempts.push(p);
            integer_only(p)
        });
        assert_eq!(attempts, vec![PayloadPlan::Identity]);
        let e = r.expect_err("Strict must stop on an integer-only endpoint");
        assert!(e.is_device_limitation(), "{e:?}");
        assert!(e.message().contains("rejected 32f"), "{e}");

        // Automatic: the source's own representation first, then the one
        // conversion. Two attempts, and the second succeeds.
        let mut attempts = Vec::new();
        let plan = walk_ladder(PcmKind::Float32, state::OutputPolicy::PreferExact, |p| {
            attempts.push(p);
            integer_only(p)
        })
        .expect("Automatic must find the integer route");
        assert_eq!(attempts, vec![PayloadPlan::Identity, PayloadPlan::Q31]);
        assert_eq!(plan, PayloadPlan::Q31);

        // HQ: straight to the conversion. One attempt, no wasted rejection.
        let mut attempts = Vec::new();
        let plan = walk_ladder(PcmKind::Float32, state::OutputPolicy::HqProcessed, |p| {
            attempts.push(p);
            integer_only(p)
        })
        .expect("HQ must open the conversion");
        assert_eq!(attempts, vec![PayloadPlan::Q31]);
        assert_eq!(plan, PayloadPlan::Q31);
    }

    /// Nothing but a format rejection may advance the ladder, so a failing
    /// endpoint cannot be walked repeatedly.
    #[test]
    fn a_ladder_stops_dead_on_anything_that_is_not_a_format_rejection() {
        for e in [
            OpenError::config("MOOSIK_BP_FORMAT=nonsense"),
            OpenError::source("no such file"),
            OpenError::seek("unseekable"),
            OpenError::device_unavailable("in exclusive use by another application"),
            OpenError::backend("CoCreateInstance failed"),
        ] {
            let mut attempts = 0usize;
            let kind = e.kind();
            let r: Result<(), OpenError> =
                walk_ladder(PcmKind::Float32, state::OutputPolicy::PreferExact, |_| {
                    attempts += 1;
                    Err(e.clone())
                });
            assert_eq!(attempts, 1, "{kind} must not advance the ladder");
            assert_eq!(r.unwrap_err().kind(), kind);
        }
    }

    /// The whole off-grid Float32 fixture plays through Q1.31, start to
    /// finish, with no fault and a clean ending — the second half of the same
    /// regression.
    ///
    /// The guard used to refuse the first off-grid sample and fault the
    /// session, which stopped the track *and*, through the shared `finished`
    /// flag, sent the playlist straight back into it.
    #[test]
    fn an_off_grid_float_track_plays_to_its_end_and_earns_an_advance() {
        let sr = 44_100u32;
        let frames = 8192usize;
        // Off-grid on purpose: irrational phase, and a scale that is not a
        // power of two, so essentially no sample lands on the Q1.31 lattice.
        let mut data = Vec::new();
        for i in 0..frames {
            for ch in 0..2 {
                let t = i as f32 * 0.001_37 + ch as f32 * 0.31;
                data.extend_from_slice(&(t.sin() * 0.618_034).to_le_bytes());
            }
        }
        let t = temp("smsl_offgrid.wav");
        write_wav(&t.0, &data, 32, true, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).expect("the fixture must open");
        assert_eq!(prep.source.kind, PcmKind::Float32);

        let ch = 2usize;
        let (prod, mut cons) = channel_ring::<u32>(ch, 1 << 21);
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared.out_rate.store(sr, Ordering::Relaxed);
        shared
            .plan
            .store(PayloadPlan::Q31.code(), Ordering::Relaxed);
        let generation = shared.generation();
        decode_loop(
            prep,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            generation,
            Arc::new(AtomicU64::new(generation)),
        );

        // Nothing faulted, and the decoder reached the end of the file.
        assert_eq!(
            shared.fault(),
            fault::NONE,
            "an off-grid sample on a permitted conversion is not a fault"
        );
        assert!(done.load(Ordering::Acquire));
        assert!(
            shared.decode_eof_for(shared.decode_generation()),
            "the decoder must have reached the end, not given up"
        );

        // Every frame of the file is in the ring.
        let mut got = vec![0u32; frames * ch];
        let n = cons.pop_frames(&mut got, frames);
        assert_eq!(n, frames, "the whole track must have been published");

        // The rounding happened and is on record, so the claim is withheld —
        // amber Processed, never green.
        assert!(
            shared.off_grid_in(generation) > 0,
            "this fixture is off-grid; the guard must have had to round"
        );

        // And the drain turns it into the one terminal state that advances.
        let mut tap = SpectrumTap::new(
            2,
            PcmKind::Integer { valid_bits: 32 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut scratch = vec![0u32; 512 * ch];
        *shared.session.lock().unwrap() = Some(Session {
            generation: shared.generation(),
            cons,
            decode_done: Arc::clone(&done),
            stop: Arc::new(AtomicBool::new(false)),
        });
        // Drain whatever is left, then one more call to cross the boundary.
        for _ in 0..64 {
            render_frames(&shared, &mut scratch, &mut tap, 2, 512);
        }
        assert_eq!(shared.completion(), Completion::CleanEof);
        assert_eq!(
            on_completion(shared.completion(), false),
            TickAction::Advance,
            "a track that played to its end advances; it does not halt"
        );
    }

    // -----------------------------------------------------------------------
    // Generations, priming and backend death
    // -----------------------------------------------------------------------

    /// A fault belongs to the session that raised it, and dies with it.
    ///
    /// One `Shared` outlives every track, seek and gapless hand-off on a
    /// stream, so "the stream has a fault" was never the same statement as
    /// "this track has a fault". A decode thread finishing its unwind could
    /// fault the track that replaced it, and a new session inherited whatever
    /// its predecessor had raised.
    #[test]
    fn a_fault_belongs_to_the_session_that_raised_it() {
        let sh = Shared::new();
        let first = sh.generation();

        sh.raise_fault_in(first, fault::UNDERRUN);
        assert_eq!(sh.fault(), fault::UNDERRUN);

        // A new session. The old fault is not this session's.
        let second = sh.begin_generation();
        assert_ne!(second, first);
        assert_eq!(sh.fault(), fault::NONE, "a new session starts clean");

        // The old producer, still unwinding, tries to fault. It cannot reach
        // the session that replaced it.
        sh.raise_fault_in(first, fault::DECODE_ERROR);
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "a superseded producer must not fault its successor"
        );

        // The current producer can.
        sh.raise_fault_in(second, fault::BACKEND_WRITE);
        assert_eq!(sh.fault(), fault::BACKEND_WRITE);

        // First one wins *within* a generation.
        sh.raise_fault_in(second, fault::RING_INVARIANT);
        assert_eq!(sh.fault(), fault::BACKEND_WRITE);

        // A gapless rollover is a new session too.
        // A gapless boundary: the decoder starts the next track, and the
        // device arrives at it.
        let pushed = sh.roll_decode_generation();
        let third = sh.reach_boundary(pushed);
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "a rollover does not inherit a fault"
        );
        sh.raise_fault_in(third, fault::RING_INVARIANT);
        assert_eq!(sh.fault(), fault::RING_INVARIANT);
    }

    /// The render thread faults whatever is playing now, because that is when
    /// the device played the silence.
    #[test]
    fn the_render_side_faults_the_session_that_is_playing() {
        let sh = Shared::new();
        sh.begin_generation();
        sh.revoke(sh.generation(), fault::UNDERRUN);
        assert_eq!(sh.revoked(), fault::UNDERRUN);
        sh.begin_generation();
        assert_eq!(sh.revoked(), fault::NONE);
        assert_eq!(sh.fault(), fault::NONE);
    }

    /// Priming is bounded. A decoder that never produces a frame is reported,
    /// not waited on forever.
    #[test]
    fn priming_forever_is_a_fault_not_a_wait() {
        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        // An installed session whose decode thread will never deliver.
        // The session is installed *for* the generation it belongs to, which
        // is why the generation is opened first.
        let generation = sh.begin_generation();
        let (_prod, cons) = channel_ring::<u32>(2, 4096);
        *sh.session.lock().unwrap() = Some(Session {
            generation,
            cons,
            decode_done: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });

        let mut tap = SpectrumTap::new(
            2,
            PcmKind::Integer { valid_bits: 24 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut scratch = vec![0u32; 480 * 2];

        // Under the bound, this is the gap the listener asked for.
        let want = 480usize;
        let calls_per_sec = 48_000 / want;
        for _ in 0..(calls_per_sec * (PRIMING_LIMIT_SECS as usize) / 2) {
            assert_eq!(render_frames(&sh, &mut scratch, &mut tap, 2, want), 0);
        }
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "a long seek is a wait, not a fault"
        );
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);

        // Past it, the silence gets a name.
        for _ in 0..(calls_per_sec * (PRIMING_LIMIT_SECS as usize)) {
            render_frames(&sh, &mut scratch, &mut tap, 2, want);
        }
        assert_eq!(
            sh.fault(),
            fault::PRIMING_TIMEOUT,
            "a session that never fills must be reported"
        );
    }

    /// A track is not over when the ring empties — it is over when the device
    /// has played what it was already given.
    ///
    /// Emitting `CleanEof` the instant the ring ran dry reported a track as
    /// finished while up to a full exclusive buffer of it was still unplayed,
    /// so the next track began over the tail of the last one.
    #[test]
    fn a_track_is_not_finished_until_the_device_has_drained() {
        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(2048, Ordering::Relaxed);

        // A session with a little audio in it, whose decoder has finished.
        let (mut prod, cons) = channel_ring::<u32>(2, 4096);
        let data = vec![7u32; 64];
        prod.push_frames(&data).unwrap();
        let done = Arc::new(AtomicBool::new(true));
        *sh.session.lock().unwrap() = Some(Session {
            generation: sh.generation(),
            cons,
            decode_done: Arc::clone(&done),
            stop: Arc::new(AtomicBool::new(false)),
        });
        sh.note_decode_eof(sh.decode_generation());

        let mut tap = SpectrumTap::new(
            2,
            PcmKind::Integer { valid_bits: 24 },
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut scratch = vec![0u32; 512 * 2];

        // The ring runs dry on the first call, which starts the drain — and
        // the track is *not* finished yet.
        render_frames(&sh, &mut scratch, &mut tap, 2, 512);
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "the device still holds a buffer of audio"
        );

        // 2 × 2048 frames have to pass before it is. At 512 a call that is
        // seven more calls; six must not be enough.
        for _ in 0..6 {
            render_frames(&sh, &mut scratch, &mut tap, 2, 512);
            assert_eq!(sh.completion(), Completion::Running);
        }
        render_frames(&sh, &mut scratch, &mut tap, 2, 512);
        render_frames(&sh, &mut scratch, &mut tap, 2, 512);
        assert_eq!(
            sh.completion(),
            Completion::CleanEof,
            "and then it is finished, once the device has had the whole buffer"
        );

        // The drain is bounded: a device that never stops asking still ends
        // the track rather than leaving it permanently almost-finished.
        assert!(!sh.draining.load(Ordering::Acquire));
    }

    /// A drain does not resurrect a session that failed.
    #[test]
    fn a_failed_session_does_not_drain_into_a_clean_ending() {
        let sh = Arc::new(Shared::new());
        sh.out_rate.store(48_000, Ordering::Relaxed);
        sh.out_buffer_frames.store(16, Ordering::Relaxed);
        let g = sh.generation();
        sh.note_decode_eof(g);
        sh.begin_drain();
        sh.fail(g, fault::BACKEND_WRITE);

        // Plenty of drain, all of it after the failure.
        for _ in 0..64 {
            sh.advance_drain(512);
        }
        assert_eq!(sh.completion().failure(), Some(fault::BACKEND_WRITE));
        assert!(!sh.completion().may_advance());
    }

    /// A dead backend refuses reuse, releases whoever is waiting, and does it
    /// as a **failure**.
    ///
    /// Releasing the waiter with a clean ending is what made a dead backend
    /// indistinguishable from a track that had played to its end, so the
    /// playlist advanced into the same dead stream and did it again.
    #[test]
    fn a_dead_backend_ends_the_session_without_earning_an_advance() {
        let sh = Shared::new();
        assert!(!sh.backend_dead.load(Ordering::Acquire));
        assert_eq!(sh.completion(), Completion::Running);

        sh.mark_backend_dead();
        assert!(sh.backend_dead.load(Ordering::Acquire), "reuse is refused");
        assert!(
            sh.completion().is_over(),
            "nobody is left waiting for audio that is not coming"
        );
        assert!(
            !sh.completion().may_advance(),
            "and a dead backend is not a track that finished"
        );
        assert_eq!(sh.completion().failure(), Some(fault::BACKEND_DEAD));
    }

    /// A failure outranks a drain.
    ///
    /// A session that faulted and then ran out of audio did not end cleanly,
    /// and letting the drain overwrite the fault is precisely how a failure
    /// turned into an advance.
    #[test]
    fn a_drain_after_a_fault_is_still_a_failure() {
        let sh = Shared::new();
        let g = sh.generation();
        sh.fail(g, fault::SOURCE_READ);
        // Everything the clean path would do, after the fact.
        sh.note_decode_eof(g);
        sh.finish_clean(g);
        assert!(!sh.completion().may_advance(), "{:?}", sh.completion());
        assert_eq!(sh.completion().failure(), Some(fault::SOURCE_READ));

        // And the other order: a *fatal* fault raised during the drain still
        // wins, because a failure outranks a clean end already recorded for
        // the same generation.
        let sh = Shared::new();
        let g = sh.generation();
        sh.note_decode_eof(g);
        sh.finish_clean(g);
        assert!(sh.completion().may_advance());
        sh.fail(g, fault::BACKEND_DEAD);
        assert!(!sh.completion().may_advance());
        assert_eq!(sh.completion().failure(), Some(fault::BACKEND_DEAD));
    }

    /// A dropout costs the claim, not the playlist.
    ///
    /// Integrity and completion were one thing, and a track that dropped out
    /// was therefore a track that had failed — so the playlist stopped at the
    /// end of it. A dropout is a real loss and the badge has to show it, but
    /// the track still played to its end and the next one is still owed.
    #[test]
    fn an_integrity_fault_costs_the_badge_and_not_the_next_track() {
        for code in [
            fault::UNDERRUN,
            fault::CALLBACK_LOCK_MISS,
            fault::VALUE_EXACT_VIOLATION,
        ] {
            assert!(!fault::is_fatal(code), "{code} must be survivable");
            let sh = Shared::new();
            let g = sh.generation();
            sh.revoke(g, code);
            // Visible, and it stays visible — on the recoverable record,
            // which is the one it belongs to.
            assert_eq!(sh.revoked(), code);
            assert_eq!(sh.fault(), fault::NONE);
            // Still playing: an integrity fault ends nothing by itself.
            assert_eq!(sh.completion(), Completion::Running);
            // And the end of the track is still the end of the track.
            sh.note_decode_eof(g);
            sh.finish_clean(g);
            assert_eq!(sh.completion(), Completion::CleanEof, "code {code}");
            assert_eq!(
                on_completion(sh.completion(), false),
                TickAction::Advance,
                "code {code}: a dropout must not strand the playlist"
            );
            assert_eq!(sh.revoked(), code, "and the fault is not forgotten");
        }

        // The other class does exactly the opposite, which is the whole point
        // of there being two.
        for code in [
            fault::DECODE_ERROR,
            fault::SOURCE_READ,
            fault::BACKEND_WRITE,
            fault::BACKEND_DEAD,
            fault::PRIMING_TIMEOUT,
            fault::SESSION_THREAD_FAILED,
            fault::BACKEND_OVERLOAD,
            // A torn ring belongs here, not above. The samples at the head can
            // no longer be attributed to a channel, so carrying on plays every
            // channel on the wrong one for the rest of the track — silently,
            // with no resynchronisation that is not a guess.
            fault::RING_INVARIANT,
        ] {
            assert!(fault::is_fatal(code), "{code} must end the session");
            let sh = Shared::new();
            let g = sh.generation();
            sh.fail(g, code);
            sh.note_decode_eof(g);
            sh.finish_clean(g);
            assert_eq!(
                on_completion(sh.completion(), false),
                TickAction::Halt(code),
                "code {code}"
            );
        }
    }

    /// A terminal state belongs to the session that reached it.
    ///
    /// A new session must not inherit its predecessor's ending — neither a
    /// failure that would stop it before it started, nor a clean end that
    /// would advance the playlist past a track that has only just begun.
    #[test]
    fn a_terminal_state_does_not_outlive_its_session() {
        let sh = Shared::new();
        let first = sh.generation();
        sh.fail(first, fault::DECODE_ERROR);
        assert!(sh.completion().is_over());

        let second = sh.begin_generation();
        assert_eq!(
            sh.completion(),
            Completion::Running,
            "a replacement session starts running, whatever killed the last one"
        );

        // And the dead one cannot reach forward.
        sh.fail(first, fault::SOURCE_READ);
        sh.note_decode_eof(first);
        sh.finish_clean(first);
        assert_eq!(sh.completion(), Completion::Running);

        // A clean ending is equally scoped.
        sh.note_decode_eof(second);
        sh.finish_clean(second);
        assert!(sh.completion().may_advance());
        sh.begin_generation();
        assert_eq!(sh.completion(), Completion::Running);
    }

    /// The Q1.31 conversion never faults; it records what it had to round, and
    /// the record is what withholds the value-exact label.
    #[test]
    fn an_off_grid_float_track_plays_and_loses_only_the_claim() {
        let sr = 48_000u32;
        // Deliberately off-grid: 1/3 is not a dyadic rational.
        let mut data = Vec::new();
        for i in 0..2048i32 {
            let v = (i as f32 / 3.0).sin() * 0.37;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let t = temp("q31_offgrid_play.wav");
        write_wav(&t.0, &data, 32, true, sr, 2);

        let prep = prepare(&t.0, Duration::ZERO).expect("an ordinary float file must open");
        let ch = prep.source.channels.max(1) as usize;
        let (prod, mut cons) = channel_ring::<u32>(ch, 1 << 20);
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared::new());
        shared.out_valid_bits.store(32, Ordering::Relaxed);
        shared
            .plan
            .store(PayloadPlan::Q31.code(), Ordering::Relaxed);
        decode_loop(
            prep,
            prod,
            Arc::clone(&done),
            Arc::new(AtomicBool::new(false)),
            Arc::clone(&shared),
            shared.generation(),
            Arc::new(AtomicU64::new(shared.generation())),
        );

        assert!(done.load(Ordering::Acquire), "the track must finish");
        assert_eq!(
            shared.fault(),
            fault::NONE,
            "an off-grid sample on a permitted conversion is not a fault"
        );
        let mut got = vec![0u32; 2048];
        let n = cons.pop_frames(&mut got, 1024);
        assert!(n > 0, "the audio must actually have been published");
        assert!(
            shared.off_grid_in(shared.generation())
                > 0,
            "and the rounding must be on record, so the claim is withheld"
        );
    }

    // -----------------------------------------------------------------------
    // Float32 routing
    // -----------------------------------------------------------------------

    /// Strict never offers a conversion. An integer-only endpoint is a stop,
    /// with a reason, and not a quiet downgrade.
    #[test]
    fn strict_offers_no_conversion_at_all() {
        use state::OutputPolicy::StrictExact;
        assert_eq!(
            plan_ladder(PcmKind::Float32, StrictExact),
            vec![PayloadPlan::Identity]
        );
        for b in [16u8, 24, 32] {
            assert_eq!(
                plan_ladder(PcmKind::Integer { valid_bits: b }, StrictExact),
                vec![PayloadPlan::Identity]
            );
        }
    }

    /// Automatic and HQ offer the conversions, in order: raw first, then the
    /// value-exact attempt, then the rounded one.
    #[test]
    fn the_float_ladder_tries_raw_then_value_exact_then_processed() {
        for policy in [
            state::OutputPolicy::PreferExact,
            state::OutputPolicy::HqProcessed,
        ] {
            let _ = policy;
        }
        // Automatic tries the source's own representation first and falls back
        // to the single integer conversion.
        assert_eq!(
            plan_ladder(PcmKind::Float32, state::OutputPolicy::PreferExact),
            vec![PayloadPlan::Identity, PayloadPlan::Q31]
        );
        // HQ was asked for the conversion. Trying the raw route first, or
        // chasing a claim it has already declined, is not what it means.
        assert_eq!(
            plan_ladder(PcmKind::Float32, state::OutputPolicy::HqProcessed),
            vec![PayloadPlan::Q31]
        );
    }

    /// An integer source is never offered a Float conversion, under any policy.
    #[test]
    fn integer_sources_are_never_offered_a_float_conversion() {
        for policy in [
            state::OutputPolicy::StrictExact,
            state::OutputPolicy::PreferExact,
            state::OutputPolicy::HqProcessed,
        ] {
            for b in 1..=32u8 {
                let l = plan_ladder(PcmKind::Integer { valid_bits: b }, policy);
                assert_eq!(l, vec![PayloadPlan::Identity], "{b}-bit under {policy:?}");
            }
        }
        // A 64-bit float has no exact route to any device format that exists.
        // Under a policy that permits processing it goes straight to the
        // conversion — offering the identity rung first would spend the open
        // on a configuration error, which stops the ladder rather than
        // advancing it, so the conversion would never be reached.
        for policy in [
            state::OutputPolicy::PreferExact,
            state::OutputPolicy::HqProcessed,
        ] {
            assert_eq!(
                plan_ladder(PcmKind::Float64, policy),
                vec![PayloadPlan::Q31],
                "{policy:?}"
            );
        }
        // Strict gets the identity rung precisely so the attempt fails and
        // names the real problem instead of quietly converting.
        assert_eq!(
            plan_ladder(PcmKind::Float64, state::OutputPolicy::StrictExact),
            vec![PayloadPlan::Identity]
        );
        // And the conversion's destination is the same 32-bit integer format
        // the Float32 route uses; the f64→f32 narrowing on the way into the
        // canonical buffer is a processing step of its own, and the route is
        // labelled Processed for both reasons.
        assert_eq!(
            PayloadPlan::Q31.device_kind(PcmKind::Float64),
            PcmKind::Integer { valid_bits: 32 }
        );
    }

    /// The raw exact set is untouched by any of the routing above. This is the
    /// thing the contract forbids solving by widening: `I32` must never appear
    /// in `exact_candidates(Float32)`, and a Float32 writer to an integer
    /// format must remain unconstructible.
    #[test]
    fn the_raw_float_exact_set_is_never_widened() {
        assert_eq!(
            format::exact_candidates(PcmKind::Float32, false).unwrap(),
            vec![format::IDX_F32]
        );
        for dev in [
            format::DevFmt::I16,
            format::DevFmt::I24,
            format::DevFmt::I24In32,
            format::DevFmt::I32,
        ] {
            assert!(
                format::Writer::new(PcmKind::Float32, dev).is_err(),
                "a Float32 writer to {dev:?} must not exist"
            );
        }
        // A forced integer override on a raw Float32 route is refused.
        for i in [
            format::IDX_I16,
            format::IDX_I24,
            format::IDX_I24_32,
            format::IDX_I32,
        ] {
            assert!(format::negotiation_order(PcmKind::Float32, false, Some(i)).is_err());
        }
    }

    /// A conversion changes what the *device* is negotiated for, without
    /// changing what the source is.
    #[test]
    fn a_conversion_negotiates_for_the_payload_not_the_source() {
        assert_eq!(
            PayloadPlan::Identity.device_kind(PcmKind::Float32),
            PcmKind::Float32
        );
        assert_eq!(
            PayloadPlan::Q31.device_kind(PcmKind::Float32),
            PcmKind::Integer { valid_bits: 32 }
        );

        // And the device format that follows is the exact-integer one, so a
        // converted stream still cannot be narrowed.
        let k = PayloadPlan::Q31.device_kind(PcmKind::Float32);
        assert_eq!(
            format::exact_candidates(k, false).unwrap(),
            vec![format::IDX_I32]
        );

        // Only the identity plan is transform-free, and the conversion
        // describes itself as what it is at the moment it opens — processed.
        // A stream is never opened under a value-exact description it has not
        // earned, which is what made two identical `I32` negotiations look
        // like two different rungs.
        assert!(
            PayloadPlan::Identity
                .describe(PcmKind::Float32)
                .is_identity()
        );
        assert!(!PayloadPlan::Q31.describe(PcmKind::Float32).is_identity());
        assert_eq!(
            PayloadPlan::Q31.describe(PcmKind::Float32),
            state::TransformDescription::FloatToQ31Processed
        );
        // And it names the source it was given. A 64-bit float source takes
        // the same route and does not go through the same thing: it is
        // narrowed to `f32` in the decoder first, which the old description
        // left out entirely.
        assert_eq!(
            PayloadPlan::Q31.describe(PcmKind::Float64),
            state::TransformDescription::Float64ToQ31Processed
        );
        assert_ne!(
            PayloadPlan::Q31.describe(PcmKind::Float64).describe(),
            PayloadPlan::Q31.describe(PcmKind::Float32).describe(),
            "the two must not read the same to a listener either"
        );

        // The code round-trips, so a plan read back from the ring's atomic is
        // the plan that was written.
        for p in [PayloadPlan::Identity, PayloadPlan::Q31] {
            assert_eq!(PayloadPlan::from_code(p.code()), p);
        }
    }

    /// A whole-track scan of a real Float32 file that is on the integer grid
    /// says so, and one that is not says why.
    #[test]
    fn a_whole_track_scan_decides_on_the_whole_track() {
        let sr = 48_000u32;
        let cancel = std::sync::atomic::AtomicBool::new(false);

        // On-grid: a 16-bit master exported as float. Every sample is a
        // multiple of 2^-15.
        let mut clean = Vec::new();
        for i in 0..4096i32 {
            let v = ((i % 32_768) - 16_384) as f32 / 32_768.0;
            clean.extend_from_slice(&v.to_le_bytes());
        }
        let t = temp("q31_clean.wav");
        write_wav(&t.0, &clean, 32, true, sr, 2);
        let v = scan_track_q31(&t.0, &cancel).unwrap();
        assert!(v.value_exact, "{}", v.describe());
        assert_eq!(v.samples, 4096);

        // One off-grid sample near the end: the scan must find it, not stop at
        // the first packet.
        let mut dirty = clean.clone();
        let off = f32::from_bits(0x2F80_0000); // 2^-32, below the Q1.31 step
        let at = 4000usize;
        dirty[at * 4..at * 4 + 4].copy_from_slice(&off.to_le_bytes());
        let t2 = temp("q31_dirty.wav");
        write_wav(&t2.0, &dirty, 32, true, sr, 2);
        let v = scan_track_q31(&t2.0, &cancel).unwrap();
        assert!(!v.value_exact, "one bad sample must be enough");
        let f = v.first_failure.expect("the failure must be located");
        assert_eq!(f.frame, (at / 2) as u64);
        assert_eq!(f.channel, (at % 2) as u16);

        // An integer source has nothing to prove and is not claimed either way.
        let mut ints = Vec::new();
        for i in 0..512i16 {
            ints.extend_from_slice(&i.to_le_bytes());
        }
        let t3 = temp("q31_int.wav");
        write_wav(&t3.0, &ints, 16, false, sr, 2);
        assert!(!scan_track_q31(&t3.0, &cancel).unwrap().value_exact);
    }

    /// A cancelled scan stops rather than grinding through a file nobody is
    /// listening to.
    #[test]
    fn a_cancelled_scan_gives_up() {
        let sr = 48_000u32;
        let mut data = Vec::new();
        for i in 0..200_000i32 {
            data.extend_from_slice(&((i % 1000) as f32 / 1024.0).to_le_bytes());
        }
        let t = temp("q31_cancel.wav");
        write_wav(&t.0, &data, 32, true, sr, 2);

        let cancel = std::sync::atomic::AtomicBool::new(true);
        let e =
            scan_track_q31(&t.0, &cancel).expect_err("a cancelled scan must not return a verdict");
        assert!(e.message().contains("cancel"), "{e:?}");
    }

    // -----------------------------------------------------------------------
    // DSD containers
    // -----------------------------------------------------------------------

    /// A DSF file's own bits, through the payload reader and the render-side
    /// marker, and back — compared against the container's normalised DSD
    /// bytes rather than against anything the encoder produced.
    #[test]
    fn a_dsf_file_reaches_the_device_as_its_own_dsd_bits() {
        use crate::dsd::tests::make_dsf;
        let blocks = vec![
            vec![vec![0x01, 0x02, 0x03, 0x04], vec![0x11, 0x12, 0x13, 0x14]],
            vec![vec![0x05, 0x06, 0x07, 0x08], vec![0x15, 0x16, 0x17, 0x18]],
        ];
        let file = make_dsf(2, crate::dsd::DSD64_RATE, 1, 64, 4, &blocks, None);
        let t = temp("fixture.dsf");
        std::fs::write(&t.0, &file).unwrap();

        // DSF stores LSB-first; the reader normalises to MSB-first, so the
        // expected bytes are the fixture's own bytes reversed, built here
        // rather than taken from the reader.
        let mut want: Vec<u8> = Vec::new();
        for row in &blocks {
            for i in 0..row[0].len() {
                for ch in row {
                    want.push(ch[i].reverse_bits());
                }
            }
        }
        assert_eq!(reconstruct_dsd(&t.0, 2), want);
    }

    /// The same for DFF, which is already MSB-first and interleaved.
    #[test]
    fn a_dff_file_reaches_the_device_as_its_own_dsd_bits() {
        use crate::dsd::tests::make_dff;
        let audio: Vec<u8> = (0..64u16).map(|i| (i * 29 + 3) as u8).collect();
        let file = make_dff(2, crate::dsd::DSD64_RATE, b"DSD ", &audio, None);
        let t = temp("fixture.dff");
        std::fs::write(&t.0, &file).unwrap();
        assert_eq!(reconstruct_dsd(&t.0, 2), audio);
    }

    /// Read a DSD file the way the output path does — payloads through the
    /// ring, markers applied at the render boundary — then strip the markers
    /// and return the DSD bytes the device would have received.
    fn reconstruct_dsd(path: &Path, channels: usize) -> Vec<u8> {
        let mut stream = crate::dsd::dop::open_dop_stream(path).unwrap();
        let w = Writer::for_dop(DevFmt::I24).unwrap();
        let mut st = DopState::new();

        let mut payload = Vec::new();
        let mut frames = 0usize;
        loop {
            let before = payload.len();
            let n = stream.read_payload(64, &mut payload).unwrap();
            if n == 0 {
                break;
            }
            assert_eq!(payload.len() - before, n * channels);
            frames += n;
        }
        assert!(
            payload.iter().all(|&p| p <= 0xFFFF),
            "a marker reached the ring"
        );

        let mut bytes = vec![0u8; frames * channels * 3];
        assert_eq!(st.emit(&payload, frames, channels, &w, &mut bytes), frames);

        // Strip markers, check them, and re-interleave the two byte lanes.
        let mut out = Vec::with_capacity(frames * channels * 2);
        for fr in 0..frames {
            let marker = format::DOP_MARKERS[fr % 2];
            let mut older = Vec::with_capacity(channels);
            let mut newer = Vec::with_capacity(channels);
            for c in 0..channels {
                let off = (fr * channels + c) * 3;
                let word = bytes[off] as u32
                    | (bytes[off + 1] as u32) << 8
                    | (bytes[off + 2] as u32) << 16;
                assert_eq!((word >> 16) as u8, marker, "frame {fr} ch {c} marker");
                older.push((word >> 8) as u8);
                newer.push(word as u8);
            }
            out.extend_from_slice(&older);
            out.extend_from_slice(&newer);
        }
        // The tail may be padded with DSD silence to fill the last carrier
        // frame; that padding is documented and is not file content.
        while out.last() == Some(&crate::dsd::dop::DSD_SILENCE) {
            out.pop();
        }
        out
    }

    // -----------------------------------------------------------------------
    // Error typing
    // -----------------------------------------------------------------------

    /// The routing layer decides whether to decimate a DSD file to PCM by
    /// asking this question, so the answer has to be carried, not flattened
    /// into a string somewhere on the way up.
    #[test]
    fn a_configuration_error_is_distinguishable_from_a_device_limitation() {
        let cfg = OpenError::config("MOOSIK_BP_FORMAT=16i excl cannot carry DoP");
        let src = OpenError::source("DSD: unsupported container");
        let dev = OpenError::device_format("\"DAC\" rejected 352.8 kHz in exclusive mode");

        // "Is this the device's fault?" — the question the DSD router asks
        // before applying the legacy automatic-fallback policy.
        assert!(!cfg.is_device_limitation());
        assert!(
            !src.is_device_limitation(),
            "every route reads the same file"
        );
        assert!(dev.is_device_limitation());
        assert!(matches!(cfg, OpenError::Config(_)));
        assert!(matches!(src, OpenError::Source(_)));

        // The message survives, and only the configuration case is labelled --
        // a device limitation is already phrased for the user.
        assert!(cfg.to_string().starts_with("configuration error: "));
        assert_eq!(dev.to_string(), dev.message());
        assert_eq!(src.to_string(), src.message());
        assert!(cfg.message().contains("cannot carry DoP"));
    }

    /// A settings file written by an older build must load unchanged.
    ///
    /// `policy` is new in 1.4.3. Without `serde(default)` an existing
    /// `bitperfect.json` would fail to parse, and `load_settings` swallows a
    /// parse failure into `Default` — so the user's device, ASIO driver and
    /// enabled flag would all silently revert on first launch.
    #[test]
    fn an_older_settings_file_loads_without_losing_anything() {
        let legacy = r#"{
            "enabled": true,
            "device": "SMSL USB DAC",
            "asio_driver": "SMSL USB AUDIO ASIO",
            "alsa_dsd_device": null
        }"#;
        let s: BpSettings =
            serde_json::from_str(legacy).expect("a 1.4.2 settings file must still parse");
        assert!(s.enabled);
        assert_eq!(s.device.as_deref(), Some("SMSL USB DAC"));
        assert_eq!(s.asio_driver.as_deref(), Some("SMSL USB AUDIO ASIO"));
        assert_eq!(
            s.policy,
            state::OutputPolicy::PreferExact,
            "an upgrading user lands on the general-purpose default"
        );

        // ...and a round trip keeps every field, including the new one.
        let mut round = s.clone();
        round.policy = state::OutputPolicy::StrictExact;
        let json = serde_json::to_string(&round).unwrap();
        let back: BpSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.policy, state::OutputPolicy::StrictExact);
        assert_eq!(back.device, round.device);
        assert!(
            json.contains("\"strict\""),
            "the policy is persisted by name: {json}"
        );
    }

    /// A source no device format can carry is refused before a device is
    /// touched, and refused as a configuration problem — trying another device
    /// cannot help, so classifying it as a device limitation would send the
    /// caller looking for one.
    #[test]
    fn an_unrepresentable_source_is_refused_as_configuration() {
        let e = format::exact_candidates(PcmKind::Float64, false).unwrap_err();
        assert!(e.contains("64-bit float"), "{e}");
        assert!(!OpenError::config(e).is_device_limitation());
    }
}

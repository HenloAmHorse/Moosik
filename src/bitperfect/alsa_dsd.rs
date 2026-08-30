// Native DSD output via ALSA — the Linux counterpart of `asio_dsd`.
//
// The kernel exposes native DSD as first-class PCM formats (DSD_U8,
// DSD_U16_LE/BE, DSD_U32_LE/BE) on direct `hw:` devices; snd-usb-audio
// advertises them for the same USB DACs whose Windows drivers do DSD over
// ASIO. No DoP carrier, so the ceiling is whatever the hardware accepts —
// DSD512/DSD1024 included.
//
// Format semantics (established by the kernel + alsa-lib + every player that
// ships this — MPD, HQPlayer, Roon Bridge):
//   • one sample holds 8/16/32 consecutive DSD bits FOR ONE CHANNEL;
//   • within a byte the OLDEST bit is the MSB (same as DoP payloads and
//     DSDIFF — and the same normalized order our DsdFileReader emits);
//   • _BE formats store the oldest byte first in memory, _LE formats are the
//     byte-swapped layout (oldest byte last within each sample);
//   • the ALSA sample rate is in SAMPLES per channel per second, i.e. the
//     DSD bit rate ÷ (8 × bytes-per-sample): DSD64 as DSD_U32_BE runs at
//     88 200 Hz, DSD512 at 705 600 Hz.
//
// So: no bit reversal, byte-swap only for _LE formats. (If some device ever
// plays loud noise, per-byte bit order is the knob — the `lsb_first` plumbing
// from the ASIO path is kept wired for exactly that day.)
//
// Unlike ASIO there is no COM, no callbacks and no global context: a plain
// writer thread owns the PCM handle and blocking-writes packed chunks, which
// also paces it. Pause holds the DAC on the DSD idle pattern (0x69) instead
// of stopping the stream, so the DSD lock never drops — same design as the
// ASIO and DoP paths.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use alsa::device_name::HintIter;
use alsa::pcm::{Access, Format, Frames, HwParams, PCM};
use alsa::{Direction, ValueOr};

use crate::dsd::dop::DSD_SILENCE;

// ---------------------------------------------------------------------------
// Device discovery
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AlsaDevice {
    /// The ALSA device string to open, e.g. "hw:CARD=C200Pro,DEV=0".
    pub id: String,
    /// Human description from the hint, e.g. "SMSL C200Pro, USB Audio".
    pub desc: String,
}

static DEVICE_CACHE: Mutex<Option<Vec<AlsaDevice>>> = Mutex::new(None);

/// Playback-capable direct-hardware PCM devices. Only `hw:` — every plugin
/// layer (plughw/dmix/pulse/pipewire) converts or rejects DSD formats, so
/// native DSD needs the raw device. The scan walks alsa-lib's name hints
/// (what `aplay -L` prints); DSD capability itself is NOT probed here —
/// opening every device just to ask would grab hardware other apps may hold.
/// The honest capability check is the first play, which reports exactly what
/// the device said (the same philosophy as the ASIO driver list).
pub fn list_dsd_devices() -> Vec<AlsaDevice> {
    let mut cache = DEVICE_CACHE.lock().unwrap_or_else(|p| p.into_inner());
    cache.get_or_insert_with(scan_devices).clone()
}

/// Forget the cached device list; the next `list_dsd_devices` rescans.
pub fn rescan_devices() {
    if let Ok(mut c) = DEVICE_CACHE.lock() { *c = None; }
}

fn scan_devices() -> Vec<AlsaDevice> {
    let Ok(iter) = HintIter::new_str(None, "pcm") else { return Vec::new() };
    iter.filter_map(|hint| {
        let name = hint.name?;
        // hw: only (see above); skip capture-only entries (direction None
        // means the hint applies both ways).
        if !name.starts_with("hw:") || hint.direction == Some(Direction::Capture) {
            return None;
        }
        // Hint descriptions are "Card name, device name\nUsage note" — the
        // first line identifies the hardware, the note is boilerplate.
        let desc = hint.desc.unwrap_or_default();
        let desc = desc.lines().next().unwrap_or("").to_string();
        Some(AlsaDevice { id: name, desc })
    }).collect()
}

// ---------------------------------------------------------------------------
// Chunk packing — interleaved byte-frames → the negotiated ALSA layout
// ---------------------------------------------------------------------------

/// Formats in negotiation order: wider samples first (fewer frames per
/// second for the same bit rate), BE before LE (BE is the memcpy layout).
const FORMAT_PREFS: [(Format, usize, bool); 5] = [
    (Format::DSDU32BE, 4, false),
    (Format::DSDU32LE, 4, true),
    (Format::DSDU16BE, 2, false),
    (Format::DSDU16LE, 2, true),
    (Format::DSDU8, 1, false),
];

/// Pack `got_frames` interleaved byte-frames from `scratch` (layout:
/// `[bf0: ch0 ch1 …][bf1: ch0 ch1 …]…`, oldest byte first, MSB-first bits —
/// the DsdFileReader's native order) into `out`, one period of interleaved
/// ALSA frames of `bps`-byte samples. Byte-frames beyond `got_frames` are
/// DSD silence. `swap` reverses the byte order within each sample (_LE).
fn pack_chunk(scratch: &[u8], got_frames: usize, ch: usize, bps: usize, swap: bool, out: &mut [u8]) {
    let frames = out.len() / (ch * bps);
    for f in 0..frames {
        for c in 0..ch {
            for k in 0..bps {
                let bf = f * bps + k; // byte-frame index (time order)
                let byte = if bf < got_frames { scratch[bf * ch + c] } else { DSD_SILENCE };
                let kk = if swap { bps - 1 - k } else { k };
                out[(f * ch + c) * bps + kk] = byte;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The stream
// ---------------------------------------------------------------------------

/// One playing track: ring consumer + its decode thread's flags.
struct Session {
    /// The generation this session belongs to — see the PCM `Session`.
    generation: u64,
    cons: super::frame_ring::FrameConsumer<u8>,
    decode_done: Arc<AtomicBool>,
    /// Set by the feeder only when it reached the end of the file cleanly.
    decode_eof: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

struct Shared {
    session: Mutex<Option<Session>>,
    paused: AtomicBool,
    /// Which track is playing and how it ended, in one word — see the PCM
    /// `Shared::play`. As two words, a period thread descheduled between
    /// reading the clock and writing the state ended the wrong track.
    play: AtomicU64,
    /// True between the ring running dry and the device having played what it
    /// already holds.
    draining: AtomicBool,
    /// Byte-frames actually written to the device since the drain began.
    drain_frames: AtomicU64,
    /// What the device still holds when the ring runs dry, in **byte-frames**.
    ///
    /// It held the negotiated ALSA period, which is a count of PCM frames of
    /// `bps` bytes each — so on `DSD_U32` it was a quarter of the byte-frames
    /// the writer thread deals in, and the drain ended after half a period
    /// instead of two. On DSD that shortfall is the DAC losing its lock on a
    /// bitstream mid-hand-off.
    period_frames: AtomicU64,
    /// When the drain began, in milliseconds since this stream opened, and how
    /// many writes it has seen. The frame target says how much the device must
    /// play; these say how long it may take — without which a device that
    /// stops accepting writes leaves a track permanently almost-finished.
    /// The track a drain in progress belongs to.
    ///
    /// A drain outlives its session slot — that is what a drain *is*, the
    /// device playing out audio after the ring has been given up — so from
    /// the moment it starts there is nothing left to read a generation from.
    /// Everything the drain then says was said about whatever happened to be
    /// playing at that instant, which after a track change is the wrong
    /// track, and during the gap between sessions is generation zero: a
    /// number no session ever holds, so the evidence went nowhere at all.
    drain_gen: AtomicU64,
    drain_started_ms: AtomicU64,
    drain_writes: AtomicU64,
    opened_at: Instant,
    /// XRUNs and suspends the writer recovered from.
    ///
    /// A recovery is not nothing having happened: the device ran out of audio
    /// and played whatever was in its buffer, or the stream was taken away and
    /// handed back. Either way the DAC emitted something that was not in the
    /// file, and this is the only evidence of it the process can produce.
    underruns: AtomicU64,
    /// Byte-frames (1 byte = 8 DSD samples, per channel) delivered as audio.
    frames_played: AtomicU64,
    /// The writer hit an unrecoverable device error (e.g. USB unplug) —
    /// surfaced as end-of-session; the next play re-opens from scratch.
    failed: AtomicBool,
    /// The first *fatal* fault of this session, stamped — the reason it
    /// stopped.
    fault_code: AtomicU64,
    /// The first *recoverable* loss of integrity, stamped.
    revoked: AtomicU64,
    /// The window every evidence publication happens inside. See
    /// `bitperfect::PublishWindow`.
    evidence: crate::bitperfect::PublishWindow,
    /// Identifies this `Shared` to `window::fire`. Tests only.
    #[cfg(test)]
    hook_id: u64,
}

impl Shared {
    /// Playback ended, cleanly, having played everything. Refused if anything
    /// has already gone wrong.
    fn finish_clean(&self, at: u64) {
        // An integrity fault costs the badge, not the track — see
        // `bitperfect::fault::is_fatal`.
        self.transition(at, crate::bitperfect::done_code::CLEAN_EOF, false)
    }

    /// Playback ended because something went wrong. Latching, and it outranks
    /// a clean end.
    fn fail(&self, at: u64, code: u8) {
        if code != crate::bitperfect::fault::NONE {
            self.raise_fault_in(at, code);
        }
        self.transition(at, crate::bitperfect::done_code::FAILED, true);
    }

    /// End the session playing now — for producers that outlive sessions.
    #[inline]
    fn fail_now(&self, code: u8) {
        self.fail(self.generation(), code);
    }

    /// A loss of integrity that does not end the track, kept apart from the
    /// reason it eventually stops.
    ///
    /// One slot could not hold both. First-wins meant a dropout suppressed the
    /// write error that actually ended the session, so the track reported the
    /// wrong reason for stopping; last-wins would have erased the evidence
    /// that anything was lost before it. Both are true and both are kept.
    #[inline]
    fn revoke(&self, at: u64, code: u8) {
        debug_assert!(!super::fault::is_fatal(code), "revoke is for integrity faults");
        // Opened before anything is read, and closed by the guard on every
        // exit — including the ones that decide not to write at all, which a
        // reader cannot tell apart from a write that has not happened yet.
        let _publishing = self.evidence.publish();
        let want = super::stamped_pack(at, code);
        let mut cur = self.revoked.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) > at {
                return;
            }
            if super::stamped_generation(cur) == at
                && super::stamped_code(cur) != super::fault::NONE
            {
                return;
            }
            match self.revoked.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    #[cfg(test)]
                    crate::bitperfect::window::fire(
                        self.hook_id,
                        crate::bitperfect::window::Site::Publish(1),
                    );
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// One attempt at reading this session's evidence.
    ///
    /// Native sessions run no Q1.31 conversion, so that field is always zero;
    /// the other two are the same two records, with the same hazard between
    /// them.
    fn read_evidence(&self, at: u64) -> crate::bitperfect::Evidence {
        let revoked = self.revoked_in(at);
        #[cfg(test)]
        crate::bitperfect::window::fire(
            self.hook_id,
            crate::bitperfect::window::Site::Evidence(2),
        );
        let fatal = self.fatal_in(at);
        crate::bitperfect::Evidence {
            generation: at,
            off_grid: 0,
            revoked,
            fatal,
            settled: false,
        }
    }

    /// The recoverable integrity loss recorded against `at`, if any. Survives
    /// a later fatal fault.
    #[inline]
    fn revoked_in(&self, at: u64) -> u8 {
        let v = self.revoked.load(Ordering::Acquire);
        if super::stamped_generation(v) == at {
            super::stamped_code(v)
        } else {
            super::fault::NONE
        }
    }

    /// The recoverable loss of the session playing now.
    ///
    /// Tests only. Production reads `evidence_at`, which answers about one
    /// named generation; this answers about "now", which is the question that
    /// cannot be composed with another.
    #[cfg(test)]
    #[inline]
    fn revoked(&self) -> u8 {
        self.revoked_in(self.generation())
    }

    fn completion(&self) -> crate::bitperfect::Completion {
        let v = self.play.load(Ordering::Acquire);
        let generation = super::stamped_generation(v);
        match super::stamped_code(v) {
            crate::bitperfect::done_code::CLEAN_EOF => crate::bitperfect::Completion::CleanEof,
            crate::bitperfect::done_code::FAILED => crate::bitperfect::Completion::Failed {
                reason: self.fault_in(generation),
                generation,
            },
            _ => crate::bitperfect::Completion::Running,
        }
    }

    /// Publish an integrity fault against `at`, and only `at`. First one wins.
    #[inline]
    fn raise_fault_in(&self, at: u64, code: u8) {
        let _publishing = self.evidence.publish();
        let want = super::stamped_pack(at, code);
        let mut cur = self.fault_code.load(Ordering::Acquire);
        loop {
            // Monotone in the generation: a period thread still finishing
            // after its session was replaced must not stamp its own, older
            // generation over the successor's live fault.
            if super::stamped_generation(cur) > at {
                return;
            }
            if super::stamped_generation(cur) == at
                && super::stamped_code(cur) != super::fault::NONE
            {
                return;
            }
            match self.fault_code.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    #[cfg(test)]
                    crate::bitperfect::window::fire(
                        self.hook_id,
                        crate::bitperfect::window::Site::Publish(2),
                    );
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// This session's fault, or `fault::NONE`.
    /// The terminal fault of the session playing now — **fatal only**. See
    /// the PCM `Shared::fault`.
    /// The terminal fault recorded against `at`, or `fault::NONE` — fatal
    /// only, and read by the stamp so it composes with the other fields.
    #[inline]
    fn fatal_in(&self, at: u64) -> u8 {
        let v = self.fault_code.load(Ordering::Acquire);
        if super::stamped_generation(v) == at {
            super::stamped_code(v)
        } else {
            super::fault::NONE
        }
    }

    #[inline]
    fn fault(&self) -> u8 {
        self.fatal_in(self.generation())
    }

    /// What this session has to answer for: the reason it stopped if it
    /// stopped, otherwise the claim it lost. A fatal reason outranks a
    /// recoverable one without overwriting it.
    #[inline]
    fn fault_in(&self, at: u64) -> u8 {
        let v = self.fault_code.load(Ordering::Acquire);
        if super::stamped_generation(v) == at
            && super::stamped_code(v) != super::fault::NONE
        {
            return super::stamped_code(v);
        }
        self.revoked_in(at)
    }

    #[inline]
    fn generation(&self) -> u64 {
        super::stamped_generation(self.play.load(Ordering::Acquire))
    }

    /// Frames handed to the device under the session that is playing.
    #[inline]
    fn frames_played(&self) -> u64 {
        crate::bitperfect::counted::count(self.frames_played.load(Ordering::Acquire))
    }

    /// Credit `got` frames to `at`, and to `at` alone — see the PCM
    /// `Shared::account_frames`. A callback descheduled past a track change
    /// must not add its frames to the successor's total.
    fn account_frames(&self, at: u64, got: u64) {
        use crate::bitperfect::counted;
        if got == 0 {
            return;
        }
        let mut cur = self.frames_played.load(Ordering::Acquire);
        loop {
            if !counted::belongs_to(cur, at) {
                return;
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


    /// Move `at` to a terminal state, or refuse — one compare-exchange on the
    /// word that also holds the generation.
    fn transition(&self, at: u64, code: u8, fatal: bool) {
        let want = super::stamped_pack(at, code);
        let mut cur = self.play.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) != at {
                return;
            }
            let held = super::stamped_code(cur);
            if held != crate::bitperfect::done_code::RUNNING
                && !(fatal && held == crate::bitperfect::done_code::CLEAN_EOF)
            {
                return;
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

    /// Begin a new track on this stream, retiring the previous one's fault and
    /// terminal state.
    fn begin_session(&self) -> u64 {
        self.draining.store(false, Ordering::Relaxed);
        self.drain_frames.store(0, Ordering::Relaxed);
        // See the ASIO counterpart: allocated, never derived.
        let next = super::gen_alloc::next();
        let want = super::stamped_pack(next, crate::bitperfect::done_code::RUNNING);
        let mut cur = self.play.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) >= next {
                return super::stamped_generation(cur);
            }
            match self
                .play
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
        // The frame counter belongs to the new session from here.
        //
        // It was stamped by `start_session` alone, so every other way of
        // opening one left the counter carrying the previous session's tag and
        // `account_frames` refused every frame the new track played — the
        // position simply stopped moving. The same defect the PCM path had,
        // and it belongs where the generation is decided.
        self.frames_played.store(
            crate::bitperfect::counted::pack(next, 0),
            Ordering::Relaxed,
        );
                    return next;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// The ring has run dry with the feeder finished; the device still holds a
    /// period or more of audio.
    ///
    /// `at` is the track that is draining — see the ASIO counterpart.
    fn begin_drain(&self, at: u64) {
        self.drain_gen.store(at, Ordering::Release);
        self.drain_frames.store(0, Ordering::Relaxed);
        self.drain_writes.store(0, Ordering::Relaxed);
        self.drain_started_ms.store(
            self.opened_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        self.draining.store(true, Ordering::Release);
    }

    /// Whether a drain has been open too long to still be a drain. Read from
    /// the UI thread; the writer can only count what it manages to write, and
    /// a device that has stopped accepting writes gives it nothing to count.
    fn drain_expired(&self) -> bool {
        if !self.draining.load(Ordering::Acquire) {
            return false;
        }
        let began = self.drain_started_ms.load(Ordering::Relaxed);
        let now = self.opened_at.elapsed().as_millis() as u64;
        now.saturating_sub(began) >= DRAIN_LIMIT.as_millis() as u64
    }

    /// End a drain that ran out of time rather than out of audio. A failure,
    /// not an ending: the device did not play what it was holding.
    fn expire_drain(&self) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        self.draining.store(false, Ordering::Release);
        // Against the draining track, not against whatever is playing when
        // the UI notices. A device that stopped mid-drain failed *that*
        // track, and stamping the failure with the current generation put it
        // on a session that had done nothing wrong — or, in the gap between
        // sessions, on generation zero, which no session ever holds.
        self.fail(
            self.drain_gen.load(Ordering::Acquire),
            crate::bitperfect::fault::BACKEND_DEAD,
        );
    }

    /// Which track a piece of evidence belongs to, given what the writer
    /// managed to establish.
    ///
    /// `session_at` is zero when no session was reached — which is every
    /// period of a drain, because the drain *is* the state of having given the
    /// session up. The draining track is still the track playing, and is what
    /// the evidence is about.
    #[inline]
    fn evidence_at(&self, session_at: u64) -> u64 {
        if session_at != 0 {
            session_at
        } else {
            self.drain_gen.load(Ordering::Acquire)
        }
    }

    /// Advance a drain by byte-frames the device has actually accepted.
    ///
    /// `submitted` is what `writei` took, not what the loop offered it. A
    /// write that failed and recovered moved the drain along on audio the
    /// device never received.
    fn advance_drain(&self, submitted: usize) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        // The track that started the drain.
        let at = self.drain_gen.load(Ordering::Acquire);
        self.drain_writes.fetch_add(1, Ordering::Relaxed);
        let n = self
            .drain_frames
            .fetch_add(submitted as u64, Ordering::Relaxed)
            + submitted as u64;
        let target = (self.period_frames.load(Ordering::Relaxed) * 2).max(2048);
        if n >= target {
            self.draining.store(false, Ordering::Release);
            self.finish_clean(at);
        }
    }
}
/// How long a drain may take before the device is presumed to have stopped.
/// See the ASIO constant of the same name.
const DRAIN_LIMIT: Duration = Duration::from_secs(5);

pub struct AlsaDsdStream {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
    shared: Arc<Shared>,
    pub device_name: String,
    /// The DSD bit rate this stream was negotiated for.
    pub dsd_rate: u32,
    pub channels: u16,
    /// True if the device wants oldest-sample-in-LSB bytes (never today —
    /// the ALSA convention is MSB-first — but the feed loop supports it, so
    /// a device quirk would be a one-line fix here).
    pub lsb_first: bool,
    /// Negotiated ALSA format, for the status line ("DSD_U32_BE").
    pub format_label: &'static str,
}
/// A stream with no device behind it, for the pause-ordering test in `main`.
///
/// `main`'s bring-up adapter is generic over the stream it pauses, and the
/// implementation under test is this type's own `pause`/`resume`. The test
/// needs an `AlsaDsdStream` and cannot open a device, so it gets one whose
/// `shared` is the same device-free fixture the evidence tests use.
#[cfg(test)]
pub(crate) fn for_pause_test() -> AlsaDsdStream {
    AlsaDsdStream::for_evidence_test(tests::shared_for_test())
}

impl AlsaDsdStream {
    /// A stream with no device behind it, for testing the evidence adapter.
    ///
    /// The adapter is the thing under test — `evidence_at` on the public type
    /// a caller actually holds — and it touches nothing but `shared`. `join`
    /// is `None`, so `Drop` sets the stop flag and joins nothing.
    #[cfg(test)]
    fn for_evidence_test(shared: Arc<Shared>) -> Self {
        AlsaDsdStream {
            stop: Arc::new(AtomicBool::new(false)),
            join: None,
            shared,
            device_name: "test".into(),
            dsd_rate: 2_822_400,
            channels: 2,
            lsb_first: false,
            format_label: "DSD_U32_BE",
        }
    }

    /// Open `device` (an ALSA `hw:` string) in native-DSD mode at `dsd_rate`
    /// (the bit rate, e.g. 22 579 200 for DSD512) with `channels` outputs.
    /// Negotiation runs synchronously; the writer thread takes over after.
    pub fn open(device: &str, dsd_rate: u32, channels: u16) -> Result<Self, String> {
        // Non-blocking, so the writer can be *told* to stop.
        //
        // A blocking `writei` waits inside the kernel until the device has
        // room, and a device that has stalled never gets room. `stop` is only
        // read between writes, so `Drop` — which sets it and then joins — had
        // no bound at all: a stalled DAC hung the thread that was closing the
        // stream, which is the UI thread, for as long as the process lived.
        // The write returns `EAGAIN` now and the loop gets to look at `stop`
        // every period instead of once per successful write.
        let pcm = PCM::new(device, Direction::Playback, true).map_err(|e| {
            let hint = if e.errno() == libc_ebusy() {
                " — another app (or the PipeWire/PulseAudio server) holds the device; \
                 native DSD needs exclusive hw: access"
            } else { "" };
            format!("ALSA \"{device}\": open failed: {e}{hint}")
        })?;

        let mut negotiated: Option<(Format, usize, bool, Frames, Frames)> = None;
        let mut errors = String::new();
        for (format, bps, swap) in FORMAT_PREFS {
            let rate = dsd_rate / (8 * bps as u32);
            match try_format(&pcm, format, rate, channels as u32) {
                Ok((period, buffer)) => {
                    crate::mlog!(
                        "[alsa-dsd] \"{device}\": {format} @ {rate} Hz accepted \
                         (period {period} frames, buffer {buffer})"
                    );
                    negotiated = Some((format, bps, swap, period, buffer));
                    break;
                }
                Err(e) => {
                    crate::mlog!("[alsa-dsd] \"{device}\": {format} @ {rate} Hz: {e}");
                    if !errors.is_empty() { errors.push_str("; "); }
                    errors.push_str(&format!("{format}: {e}"));
                }
            }
        }
        let Some((format, bps, swap, period, buffer)) = negotiated else {
            let label = crate::dsd::rate_label(dsd_rate);
            return Err(format!(
                "ALSA \"{device}\": no native DSD format accepted for {label} ({errors}) \
                 — DoP is this device's ceiling"
            ));
        };

        pcm.prepare().map_err(|e| format!("ALSA \"{device}\": prepare failed: {e}"))?;

        let shared = Arc::new(Shared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            play: AtomicU64::new(super::stamped_pack(
                0,
                crate::bitperfect::done_code::RUNNING,
            )),
            draining: AtomicBool::new(false),
            drain_frames: AtomicU64::new(0),
            // What the device still holds when the ring runs dry.
            // Converted here, once, where `bps` is in scope: the writer thread
            // counts byte-frames and this is what it is counting towards.
            period_frames: AtomicU64::new(buffer.max(period) as u64 * bps as u64),
            drain_gen: AtomicU64::new(0),
            drain_started_ms: AtomicU64::new(0),
            drain_writes: AtomicU64::new(0),
            opened_at: Instant::now(),
            underruns: AtomicU64::new(0),
            frames_played: AtomicU64::new(crate::bitperfect::counted::pack(0, 0)),
            failed: AtomicBool::new(false),
            fault_code: AtomicU64::new(0),
            revoked: AtomicU64::new(0),
            evidence: crate::bitperfect::PublishWindow::new(),
            #[cfg(test)]
            hook_id: crate::bitperfect::window::next_id(),
        });
        let stop = Arc::new(AtomicBool::new(false));

        let t_shared = Arc::clone(&shared);
        let t_stop = Arc::clone(&stop);
        let t_device = device.to_owned();
        let ch = channels as usize;
        let join = std::thread::Builder::new()
            .name("bp-alsa-dsd".into())
            .spawn(move || writer_thread(pcm, t_device, ch, bps, swap, period, t_shared, t_stop))
            .map_err(|e| format!("thread spawn failed: {e}"))?;

        Ok(AlsaDsdStream {
            stop,
            join: Some(join),
            shared,
            device_name: device.to_owned(),
            dsd_rate,
            channels,
            lsb_first: false,
            format_label: format_label(format),
        })
    }

    /// Install a new track session (ring consumer + decode-thread flags),
    /// replacing any current one.
    pub fn start_session(
        &self,
        cons: super::frame_ring::FrameConsumer<u8>,
        decode_done: Arc<AtomicBool>,
        decode_eof: Arc<AtomicBool>,
        session_stop: Arc<AtomicBool>,
    ) {
        // The clock turns inside the lock that holds the slot — see the ASIO
        // `start_session` for what the gap between the two cost.
        if let Ok(mut g) = self.shared.session.lock() {
            // `begin_session` stamps the frame counter with the generation
            // it installs.
            let generation = self.shared.begin_session();
            *g = Some(Session {
                generation,
                cons,
                decode_done,
                decode_eof,
                stop: session_stop,
            });
        }
    }

    pub fn stop_session(&self) {
        if let Ok(mut g) = self.shared.session.lock() { *g = None; }
        // A bare zero here undid the stamp `begin_session` had just written:
        // zero is generation zero, which no session holds, so the first frame
        // of the next track was refused and the position never moved.
        self.shared.begin_session();
    }

    /// `Release`, so that the writer's `Acquire` re-read after it has the
    /// session slot cannot see a stale `false` — see `take_session_frames`.
    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Release); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Release); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    /// Everything about `at`, read by `at`'s own stamps.
    ///
    /// Native sessions have no Q1.31 conversion, so that field is always zero.
    /// See `bitperfect::Evidence` for why the four are read as one thing.
    pub fn evidence_at(&self, at: u64) -> crate::bitperfect::Evidence {
        crate::bitperfect::stable_evidence(&self.shared.evidence, at, || {
            self.shared.read_evidence(at)
        })
    }

    /// The audio generation this stream's evidence is currently about.
    pub fn audio_generation(&self) -> u64 {
        self.shared.generation()
    }

    // `revoked_code` was here — see the PCM `BpStream` for why the per-field
    // getters are gone rather than merely unused.

    /// Recovered XRUNs and suspends — silence the DAC played.
    pub fn underruns(&self) -> u64 {
        self.shared.underruns.load(Ordering::Relaxed)
    }

    /// Whether playback is over, for any reason.
    pub fn is_finished(&self) -> bool { self.shared.completion().is_over() }

    /// *How* it ended — only `Completion::CleanEof` may advance a playlist.
    ///
    /// The wall clock is checked here because this is the UI thread: a drain
    /// is advanced by writes, and a device that has stopped taking them cannot
    /// advance one.
    pub fn completion(&self) -> crate::bitperfect::Completion {
        if self.shared.drain_expired() {
            self.shared.expire_drain();
        }
        self.shared.completion()
    }

    /// The feeder stopped because something went wrong.
    pub fn note_feed_failure(&self, code: u8) {
        self.shared.fail_now(code);
    }
    pub fn failed(&self) -> bool { self.shared.failed.load(Ordering::Acquire) }

    /// Sample-accurate elapsed time: byte-frames delivered × 8 samples ÷ rate.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played();
        Duration::from_secs_f64(frames as f64 * 8.0 / self.dsd_rate.max(1) as f64)
    }
}

impl Drop for AlsaDsdStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(j) = self.join.take() { let _ = j.join(); }
    }
}

fn format_label(f: Format) -> &'static str {
    match f {
        Format::DSDU32BE => "DSD_U32_BE",
        Format::DSDU32LE => "DSD_U32_LE",
        Format::DSDU16BE => "DSD_U16_BE",
        Format::DSDU16LE => "DSD_U16_LE",
        _ => "DSD_U8",
    }
}

fn libc_ebusy() -> i32 { 16 } // EBUSY — stable across Linux architectures

/// One full hw/sw-params attempt for `format`. A fresh HwParams is required
/// per attempt — a failed constraint poisons the parameter space.
fn try_format(pcm: &PCM, format: Format, rate: u32, channels: u32) -> Result<(Frames, Frames), String> {
    let hwp = HwParams::any(pcm).map_err(|e| format!("hw query: {e}"))?;
    hwp.set_rate_resample(false).map_err(|e| format!("no-resample: {e}"))?;
    hwp.set_access(Access::RWInterleaved).map_err(|e| format!("interleaved access: {e}"))?;
    hwp.set_format(format).map_err(|e| format!("format: {e}"))?;
    hwp.set_channels(channels).map_err(|e| format!("{channels}ch: {e}"))?;
    hwp.set_rate(rate, ValueOr::Nearest).map_err(|e| format!("rate: {e}"))?;
    let got = hwp.get_rate().map_err(|e| format!("rate readback: {e}"))?;
    if got != rate {
        return Err(format!("rate: device offered {got} Hz for {rate} Hz — native DSD must be exact"));
    }
    // ~12 ms periods at DSD64/U32, proportionally shorter at higher rates —
    // the writer is a plain blocking thread, nothing latency-critical.
    let period = hwp.set_period_size_near(1024.max(rate as Frames / 87), ValueOr::Nearest)
        .map_err(|e| format!("period size: {e}"))?;
    let buffer = hwp.set_buffer_size_near(period * 4).map_err(|e| format!("buffer size: {e}"))?;
    pcm.hw_params(&hwp).map_err(|e| format!("hw commit: {e}"))?;

    let swp = pcm.sw_params_current().map_err(|e| format!("sw query: {e}"))?;
    // Start once nearly full: robust against a slow first ring fill.
    swp.set_start_threshold((buffer - period).max(period))
        .map_err(|e| format!("start threshold: {e}"))?;
    pcm.sw_params(&swp).map_err(|e| format!("sw commit: {e}"))?;
    Ok((period, buffer))
}

/// The writer thread: builds one period of interleaved byte-frames per lap
/// (audio from the session ring; DSD silence while paused/starved/idle),
/// packs it to the negotiated format and blocking-writes it — the device
/// paces the loop. Mirrors the ASIO `fill_buffers` semantics exactly:
/// whole frames only, position counts real audio frames, end-of-track =
/// decode done + ring drained.
#[allow(clippy::too_many_arguments)]
/// What a failed `writei` means.
///
/// Pulled out of the loop because the loop needs a device and this does not.
/// The three cases have nothing in common except where they are noticed: one
/// is the device being busy, one is the device having glitched and come back,
/// and one is the device being gone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WriteOutcome {
    /// No room yet. Not a fault; wait and look at `stop` again.
    NoRoom,
    /// An XRUN or a suspend, recovered in place. The device played something
    /// that was not in the file.
    Recovered,
    /// Unrecoverable: the device is gone or the format was taken away.
    Fatal,
}

/// Classify a write failure. `recovered` is what `snd_pcm_recover` said.
pub(crate) fn classify_write(errno: i32, recovered: bool) -> WriteOutcome {
    if errno == libc_eagain() {
        WriteOutcome::NoRoom
    } else if recovered {
        WriteOutcome::Recovered
    } else {
        WriteOutcome::Fatal
    }
}

/// How long a non-blocking write waits for room before looking at `stop`
/// again. One period is the natural unit; this is a ceiling on how long a
/// teardown can take.
const WAIT_MS: u32 = 100;

/// `EAGAIN`, which a non-blocking `writei` returns when the device has no
/// room. Named rather than written as `-11` because it is not a failure.
fn libc_eagain() -> i32 {
    11
}

/// Record an XRUN or suspend the writer recovered from.
///
/// Recovering is not the same as nothing having happened. An XRUN is the
/// device having run out of audio and played whatever was in its buffer; a
/// suspend is the stream having been taken away and handed back. Either way
/// the DAC emitted something that was not in the file, and recovering in
/// silence left the route still claiming to be exact — which is the one claim
/// it had just stopped being able to make.
pub(crate) fn note_recovered(shared: &Shared, at: u64) {
    shared.underruns.fetch_add(1, Ordering::Relaxed);
    // Against the generation of the period being written, not the clock as it
    // reads when the write finally returns.
    //
    // A `writei` that XRUNs blocks for as long as the recovery takes, and a
    // track change can happen inside that window: reading the clock afterwards
    // attributed the outgoing track's dropout to the incoming one, which had
    // not played a sample. The successor lost its exact claim for something
    // that happened before it started.
    shared.revoke(at, crate::bitperfect::fault::UNDERRUN);
}

/// Take up to `want` bytes of the installed session into `scratch`.
///
/// A function of its own so the paused hand-off can be forced without opening
/// a device: `writer_thread` needs a live `PCM`, and this is the whole of its
/// consumption decision. Returns the session the bytes belong to, or zero if
/// none was reached.
fn take_session_frames(shared: &Shared, scratch: &mut Vec<u8>, want: usize, ch: usize) -> u64 {
    // The session these frames belong to, established while the slot is held.
    // Zero means no session was reached, and nothing is accounted.
    let mut session_at = 0u64;
    // Not a realtime callback — a full lock is fine; the only contender is
    // the UI thread swapping sessions for an instant.
    if !shared.paused.load(Ordering::Relaxed) {
        #[cfg(test)]
        crate::bitperfect::window::fire(shared.hook_id, crate::bitperfect::window::Site::Render);
        // Asked again, now that the slot is in hand.
        //
        // The outer read is a fast path and nothing more: between it and this
        // lock the UI thread can pause and install a new session, and the
        // answer the fast path gave was about a session this writer is no
        // longer the one to play.
        //
        // `pause` stores with `Release`, and the UI thread then takes this
        // same mutex to install the session. Acquiring it here synchronises
        // with that release, so a session installed after a pause can never be
        // reached with `paused` still reading false. Nothing is consumed
        // before it. This lock blocks rather than trying, so there is no miss
        // to misclassify.
        if let Ok(mut guard) = shared.session.lock()
            && !shared.paused.load(Ordering::Acquire)
            && let Some(sess) = guard.as_mut()
        {
            // Whole byte-frames, in one transfer. Popping single bytes until
            // the ring ran dry could stop mid-frame and leave a byte belonging
            // to channel *n* at the head, where the next period read it as
            // channel 0 and rotated every channel from then on.
            // Said about the session in hand, not about whatever the stream
            // reports at this instant.
            let at = sess.generation;
            session_at = at;
            // And it ends the track. This comment used to say a torn ring
            // costs the claim and not the track, which stopped being true when
            // `RING_INVARIANT` became fatal and was never corrected: there is
            // no way to know which channel the bytes at the head belong to, so
            // everything after them would play on the wrong one.
            if !sess.cons.invariant_holds() {
                shared.fail(at, super::fault::RING_INVARIANT);
            }
            scratch.resize(want, DSD_SILENCE);
            let frames = sess.cons.pop_frames(scratch, want / ch);
            scratch.truncate(frames * ch);

            if frames * ch < want
                && sess.decode_done.load(Ordering::Acquire)
                && sess.cons.is_empty()
            {
                // Read before the session is dropped: why the feeder stopped
                // is the feeder's to say, and the drain decides only when. A
                // feeder that gave up without reaching the end of the file did
                // not finish the track, and reporting it as though it had is
                // what advanced the playlist into the same failure again.
                let eof = sess.decode_eof.load(Ordering::Acquire);
                *guard = None;
                if eof {
                    // The device still holds a period or more; the drain
                    // decides when that has been played.
                    shared.begin_drain(at);
                } else {
                    shared.fail(at, crate::bitperfect::fault::SESSION_THREAD_FAILED);
                }
            }
        }
    }
    session_at
}

/// Publishes the end of the ALSA writer however the thread leaves it.
///
/// The counterpart of the ASIO `HostDeath`, and it was missing: a writer that
/// panicked left `failed` clear and no terminal state, so the handle went on
/// looking alive and the next track was handed to a stream with nobody behind
/// it. Playing silence, with the diamond lit.
struct WriterDeath(Arc<Shared>);

impl Drop for WriterDeath {
    fn drop(&mut self) {
        self.0.failed.store(true, Ordering::Release);
        // A failure, not an ending: a writer that stopped did not finish
        // playing the track, and a playlist must never advance on it.
        self.0.fail_now(crate::bitperfect::fault::BACKEND_DEAD);
    }
}

fn writer_thread(
    pcm: PCM,
    device: String,
    ch: usize,
    bps: usize,
    swap: bool,
    period: Frames,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
) {
    // Dropped last, on every exit path including a panic.
    let _death = WriterDeath(Arc::clone(&shared));
    let io = pcm.io_bytes();
    let frames = period as usize;            // ALSA frames per chunk
    let byte_frames = frames * bps;          // byte-frames per chunk
    let frame_bytes = ch * bps;
    let want = byte_frames * ch;
    let mut scratch: Vec<u8> = Vec::with_capacity(want);
    let mut packed = vec![DSD_SILENCE; frames * frame_bytes];

    'outer: while !stop.load(Ordering::Relaxed) {
        scratch.clear();
        let session_at = take_session_frames(&shared, &mut scratch, want, ch);

        let got_frames = scratch.len() / ch;

        pack_chunk(&scratch, got_frames, ch, bps, swap, &mut packed);

        // With no session installed and a drain in progress, these periods are
        // the device playing out what it already holds. Counted below, from
        // what `writei` actually accepted — counting it here, before the
        // write, advanced the drain on audio a failing device never received.
        let draining_now =
            shared.session.lock().map(|g| g.is_none()).unwrap_or(false);

        // How much of `packed` is audio rather than the silence `pack_chunk`
        // pads a short chunk with. The device is handed the whole period
        // either way — the drain counts that, because the drain is about what
        // the device is playing out — but the listener's position is about
        // the file, so it counts only this much of it, and only as the device
        // accepts it.
        let audio_byte_frames = got_frames;
        let mut credited = 0usize;

        let mut off = 0;
        while off < packed.len() {
            if stop.load(Ordering::Relaxed) { break 'outer; }
            match io.writei(&packed[off..]) {
                Ok(n) => {
                    off += n * frame_bytes;
                    // `n` is ALSA frames; every counter here is in byte-frames.
                    //
                    // Credited per accepted write rather than once before the
                    // loop. `writei` on a non-blocking handle takes what it has
                    // room for and no more, so a period is routinely accepted
                    // in pieces — and a period that is never accepted at all,
                    // because the device died partway through, used to have
                    // been credited in full before the first attempt. The
                    // remainder is preserved by `off` and offered again;
                    // nothing rejected is ever counted.
                    //
                    // The drain and the position are not the same count, and
                    // the last chunk of a track is where they differ: it is
                    // part audio and part padding, it is the chunk that gives
                    // the session slot up, and it belongs to both. Gating the
                    // position on `!draining_now` dropped it from the position
                    // entirely — a track's elapsed time stopped a few
                    // milliseconds short of its length, every time.
                    let accepted = (off / frame_bytes) * bps;
                    let audio = accepted.min(audio_byte_frames);
                    if audio > credited {
                        shared.account_frames(session_at, (audio - credited) as u64);
                        credited = audio;
                    }
                    if draining_now {
                        shared.advance_drain(n * bps);
                    }
                }
                // The device has no room yet. Expected on a non-blocking
                // handle, and the reason it is one: this is where the loop
                // gets to notice that it has been asked to stop.
                Err(e) => {
                    // One decision, taken in one place.
                    //
                    // `classify_write` was extracted so the three cases could
                    // be tested without a device, and then the loop went on
                    // deciding them again inline — so what the tests covered
                    // and what the writer did were two implementations that
                    // happened to agree. `errno` and the recovery attempt are
                    // the inputs; everything after is the classification.
                    let errno = e.errno();
                    // Only attempted when the error is not simply "no room":
                    // recovering from `EAGAIN` would be recovering from
                    // nothing.
                    let recovered = errno != libc_eagain()
                        && pcm.try_recover(e, true).is_ok();
                    match classify_write(errno, recovered) {
                        WriteOutcome::NoRoom => {
                            if pcm.wait(Some(WAIT_MS)).is_err() {
                                // A failed wait on a device that is not
                                // accepting audio is not something to retry
                                // into.
                                crate::mlog!(
                                    "[alsa-dsd] \"{device}\": wait failed while writing"
                                );
                                shared.failed.store(true, Ordering::Release);
                                shared.fail_now(crate::bitperfect::fault::BACKEND_WRITE);
                                break 'outer;
                            }
                        }
                        WriteOutcome::Recovered => {
                            // The period being written, and during a drain
                            // there is no session to read that from: the slot
                            // was given up when the ring ran dry, which is
                            // where the drain begins. `session_at` is zero
                            // from that point, and zero is a number no session
                            // ever holds — so an XRUN in the last seconds of
                            // every track was recorded against nothing and the
                            // route went on claiming to be exact through a
                            // dropout the listener had just heard.
                            note_recovered(&shared, shared.evidence_at(session_at));
                            crate::mlog!(
                                "[alsa-dsd] \"{device}\": recovered from {e} — the device \
                                 played something that was not in the file"
                            );
                        }
                        WriteOutcome::Fatal => {
                            crate::mlog!(
                                "[alsa-dsd] \"{device}\": unrecoverable write error: {e}"
                            );
                            shared.failed.store(true, Ordering::Release);
                            shared.fail_now(crate::bitperfect::fault::BACKEND_WRITE);
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
    // PCM drops here — device released.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 2ch scratch with recognizable bytes: byte-frame f carries (0xA0+f) on
    /// ch0 and (0xB0+f) on ch1.
    fn scratch(frames: usize) -> Vec<u8> {
        (0..frames).flat_map(|f| [0xA0 + f as u8, 0xB0 + f as u8]).collect()
    }

    /// A `Shared` with no device behind it.
    ///
    /// The writer loop itself needs an ALSA device and is not exercised here;
    /// what is exercised is every decision it makes, each of which has been
    /// pulled out to somewhere a test can reach.
    /// The public adapter answers with both records when both were stored.
    ///
    /// The schedule the old design could not survive, forced through the type
    /// a caller actually holds. A reader is parked between its two loads; a
    /// recoverable record and then a terminal one are stored, each parked
    /// before its publication window closes; the reader resumes and finishes
    /// its loads.
    ///
    /// Under the old protocol both stores were already visible and neither had
    /// moved the counter, so the reader saw the terminal record beside an
    /// empty recoverable slot and called it a snapshot. Under this one the
    /// window is open from before the first store, so the reader knows it is
    /// not looking at a quiet instant and folds what every attempt saw.
    #[test]
    fn alsa_evidence_holds_together_across_two_writers() {
        use crate::bitperfect::fault;
        use crate::bitperfect::window::{arm, Site};
        use std::sync::Barrier;

        let sh = shared_for_test();
        let at = sh.begin_session();
        let id = sh.hook_id;

        let reader_parked = Arc::new(Barrier::new(2));
        let writers_done = Arc::new(Barrier::new(2));
        {
            let a = Arc::clone(&reader_parked);
            let b = Arc::clone(&writers_done);
            // After the recoverable load and before the terminal one.
            arm(id, Site::Evidence(2), move || {
                a.wait();
                b.wait();
            });
        }

        // Built inside the thread, matching the PCM version: the type a
        // caller holds is what is under test, and building it there costs
        // nothing and cannot depend on the type being `Send`.
        let for_reader = Arc::clone(&sh);
        let reading = std::thread::spawn(move || {
            AlsaDsdStream::for_evidence_test(for_reader).evidence_at(at)
        });

        // The reader is now between its loads, having seen no recoverable
        // record. Both writers store while it waits there.
        reader_parked.wait();
        sh.revoke(at, fault::UNDERRUN);
        sh.fail(at, fault::BACKEND_WRITE);
        writers_done.wait();

        let ev = reading.join().expect("the read returns");
        assert_eq!(ev.generation, at);
        assert_eq!(
            ev.fatal,
            fault::BACKEND_WRITE,
            "the reason the session ended"
        );
        assert_eq!(
            ev.revoked,
            fault::UNDERRUN,
            "and the dropout stored before it — an account with the second and \
             not the first is one that never held"
        );
    }

    /// A record published *after* the fatal one still reaches the reader.
    ///
    /// The previous design's final read took the terminal record first, on the
    /// argument that nothing can be published against a generation after it.
    /// That is not true here: a callback still unwinding, and a decode thread
    /// still finishing, both revoke after the session has been failed.
    #[test]
    fn alsa_evidence_published_after_a_fatal_still_arrives() {
        use crate::bitperfect::fault;
        let sh = shared_for_test();
        let at = sh.begin_session();

        sh.fail(at, fault::BACKEND_WRITE);
        sh.revoke(at, fault::UNDERRUN);

        let stream = AlsaDsdStream::for_evidence_test(Arc::clone(&sh));
        let ev = stream.evidence_at(at);
        assert_eq!(ev.fatal, fault::BACKEND_WRITE);
        assert_eq!(
            ev.revoked,
            fault::UNDERRUN,
            "a session that has already failed can still lose its claim, and \
             the read has to say so"
        );
        assert!(ev.settled, "nothing was in flight, so this is a quiet read");
    }

    /// A writer already past its `paused` check may not play the session a
    /// pause installed behind it.
    ///
    /// The PCM schedule, on this backend, at `take_session_frames` — the whole
    /// of the writer's consumption decision, and the only part of
    /// `writer_thread` that does not need an open device. What `writei` would
    /// have been handed is then packed here rather than reasoned about.
    #[test]
    fn a_writer_past_its_check_cannot_play_a_session_paused_behind_it() {
        const CH: usize = 2;
        const BPS: usize = 4;
        const BYTE_FRAMES: usize = 64;
        let want = BYTE_FRAMES * CH;

        let sh = shared_for_test();
        let inside = Arc::new(std::sync::Barrier::new(2));
        let go = Arc::new(std::sync::Barrier::new(2));
        crate::bitperfect::window::arm(
            sh.hook_id,
            crate::bitperfect::window::Site::Render,
            {
                let inside = Arc::clone(&inside);
                let go = Arc::clone(&go);
                move || {
                    inside.wait();
                    go.wait();
                }
            },
        );

        let writer = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || {
                let mut scratch: Vec<u8> = Vec::with_capacity(want);
                let at = take_session_frames(&sh, &mut scratch, want, CH);
                (at, scratch)
            })
        };

        inside.wait();

        let payload: Vec<u8> = (0..want as u16 * 2).map(|b| 0x40 | (b as u8 & 0x3f)).collect();
        let (mut prod, cons) = crate::bitperfect::frame_ring::channel_ring::<u8>(CH, 1024);
        prod.push_frames(&payload).unwrap();
        sh.paused.store(true, Ordering::Release);
        let generation = sh.begin_session();
        *sh.session.lock().unwrap() = Some(Session {
            generation,
            cons,
            decode_done: Arc::new(AtomicBool::new(false)),
            decode_eof: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });

        go.wait();
        let (session_at, scratch) = writer.join().unwrap();

        assert!(scratch.is_empty(), "a paused session gave up bytes");
        assert_eq!(session_at, 0, "and it was not accounted against a session");
        assert!(!sh.draining.load(Ordering::Relaxed), "the drain began");
        assert_eq!(sh.frames_played(), 0, "the position moved");
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.revoked(), crate::bitperfect::fault::NONE);
        assert_eq!(sh.fault(), crate::bitperfect::fault::NONE);

        // What `writei` would have been handed: an idle period, every byte.
        let mut packed = vec![0xAAu8; BYTE_FRAMES * CH * BPS];
        pack_chunk(&scratch, scratch.len() / CH, CH, BPS, false, &mut packed);
        assert!(
            packed.iter().all(|&b| b == DSD_SILENCE),
            "the period handed to the device was not idle"
        );

        // The ring still holds every byte-frame of it.
        let mut back = vec![0u8; want];
        let n = sh
            .session
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .cons
            .pop_frames(&mut back, BYTE_FRAMES);
        assert_eq!(n, BYTE_FRAMES);
        assert_eq!(back, payload[..want], "the ring was consumed");
    }

    pub(super) fn shared_for_test() -> Arc<Shared> {
        Arc::new(Shared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            // Zero, like the production constructor: a test `Shared` that
            // started on generation one could collide with the allocator's
            // first hand-out, and then `begin_session` returned the number the
            // fixture was already using.
            play: AtomicU64::new(crate::bitperfect::stamped_pack(
                0,
                crate::bitperfect::done_code::RUNNING,
            )),
            draining: AtomicBool::new(false),
            drain_frames: AtomicU64::new(0),
            period_frames: AtomicU64::new(1024),
            drain_gen: AtomicU64::new(0),
            drain_started_ms: AtomicU64::new(0),
            drain_writes: AtomicU64::new(0),
            opened_at: Instant::now(),
            underruns: AtomicU64::new(0),
            frames_played: AtomicU64::new(crate::bitperfect::counted::pack(0, 0)),
            failed: AtomicBool::new(false),
            fault_code: AtomicU64::new(0),
            revoked: AtomicU64::new(0),
            evidence: crate::bitperfect::PublishWindow::new(),
            #[cfg(test)]
            hook_id: crate::bitperfect::window::next_id(),
        })
    }

    /// A blocked device is not a fault, and is where the writer notices that
    /// it has been asked to stop.
    ///
    /// The handle is non-blocking precisely so this case exists. With a
    /// blocking one the thread sat inside the kernel until the device found
    /// room, `stop` was read only between successful writes, and `Drop` — set
    /// the flag, then join — had no bound at all: a stalled DAC hung the
    /// thread closing the stream, which is the UI thread, for as long as the
    /// process lived.
    #[test]
    fn a_device_with_no_room_is_waited_for_and_not_faulted() {
        assert_eq!(classify_write(libc_eagain(), false), WriteOutcome::NoRoom);
        // Even if `recover` claims it handled it, `EAGAIN` is still just "not
        // yet" — the order of these two tests matters.
        assert_eq!(classify_write(libc_eagain(), true), WriteOutcome::NoRoom);

        let sh = shared_for_test();
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "waiting for room ends nothing"
        );
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
    }

    /// A recovered XRUN costs the claim.
    ///
    /// An XRUN during a drain belongs to the track that is draining.
    ///
    /// The drain begins where the session slot is given up, so from that point
    /// the writer has no session to read a generation from and `session_at` is
    /// zero. Zero is a number no session ever holds, so `revoke` compared it
    /// against the live generation, found no match, and dropped the evidence —
    /// and a dropout in the last seconds of a track, which is exactly where a
    /// drain is, left the route still claiming to be exact.
    #[test]
    fn an_xrun_during_a_drain_revokes_the_draining_track() {
        let sh = shared_for_test();
        let at = sh.begin_session();
        // The ring has run dry with the feeder finished: the slot is given up
        // and the device plays out what it holds.
        *sh.session.lock().unwrap() = None;
        sh.begin_drain(at);

        // The writer's view from here on: no session, so nothing to name the
        // track with except the drain's own record.
        let session_at = 0u64;
        assert_eq!(
            sh.evidence_at(session_at),
            at,
            "a drain in progress still knows whose it is"
        );
        note_recovered(&sh, sh.evidence_at(session_at));

        assert_eq!(
            sh.revoked_in(at),
            crate::bitperfect::fault::UNDERRUN,
            "the draining track dropped out, and that is the track it happened to"
        );
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 1);
    }

    /// Recovering is not the same as nothing having happened: the device ran
    /// out of audio and played whatever was in its buffer. It was recovered in
    /// silence, and the route went on reporting itself exact.
    #[test]
    fn a_recovered_xrun_is_recorded_and_costs_the_exact_claim() {
        // 32 = EPIPE, an XRUN. 11 = EAGAIN, which is not one.
        assert_eq!(classify_write(32, true), WriteOutcome::Recovered);
        assert_eq!(classify_write(32, false), WriteOutcome::Fatal);
        assert_eq!(classify_write(19, false), WriteOutcome::Fatal); // ENODEV

        let sh = shared_for_test();
        note_recovered(&sh, sh.generation());
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            1,
            "the only evidence of it the process can produce"
        );
        assert_eq!(sh.revoked(), crate::bitperfect::fault::UNDERRUN);
        assert_eq!(
            sh.fault(),
            crate::bitperfect::fault::NONE,
            "a dropout is evidence, not a reason the track stopped"
        );
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "and the track carries on: a dropout is not the end of it"
        );

        // A second one does not re-report, and does not stop counting.
        note_recovered(&sh, sh.generation());
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 2);
        assert_eq!(sh.revoked(), crate::bitperfect::fault::UNDERRUN);
    }

    /// An XRUN belongs to the period that was being written.
    ///
    /// A `writei` that XRUNs blocks for as long as the recovery takes, and a
    /// track change can happen inside that window. Reading the clock when the
    /// write finally returns attributed the outgoing track's dropout to the
    /// incoming one, which had not played a sample: the successor lost its
    /// exact claim for something that happened before it started.
    #[test]
    fn an_xrun_belongs_to_the_period_that_was_being_written() {
        use std::sync::Barrier;

        let sh = shared_for_test();
        let a = sh.begin_session();

        let writing = Arc::new(Barrier::new(2));
        let recovered = Arc::new(Barrier::new(2));
        let writer = {
            let sh = Arc::clone(&sh);
            let writing = Arc::clone(&writing);
            let recovered = Arc::clone(&recovered);
            std::thread::spawn(move || {
                // The period this chunk belongs to, taken while the session
                // slot was held — which is where the writer takes it.
                let at = a;
                writing.wait();
                recovered.wait();
                // The write has XRUN'd and `snd_pcm_recover` has returned.
                assert_eq!(classify_write(32, true), WriteOutcome::Recovered);
                note_recovered(&sh, at);
            })
        };

        writing.wait();
        // The track changes while the write is blocked inside the kernel.
        let b = sh.begin_session();
        recovered.wait();
        writer.join().unwrap();

        assert_ne!(a, b);
        assert_eq!(sh.generation(), b);
        assert_eq!(
            sh.revoked_in(a),
            crate::bitperfect::fault::UNDERRUN,
            "the dropout belongs to the period that was being written"
        );
        assert_eq!(
            sh.revoked_in(b),
            crate::bitperfect::fault::NONE,
            "B has not played a sample and has lost nothing"
        );
        assert_eq!(
            sh.fault(),
            crate::bitperfect::fault::NONE,
            "and nothing is shown against the track now playing"
        );
    }

    /// A dropout and then a dead device: the reason is the death, the evidence
    /// is the dropout.
    #[test]
    fn a_native_dropout_survives_the_failure_that_ends_the_track() {
        let sh = shared_for_test();
        let at = sh.begin_session();

        note_recovered(&sh, at);
        assert_eq!(sh.revoked(), crate::bitperfect::fault::UNDERRUN);
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "a recovered XRUN does not end the track"
        );

        sh.fail_now(crate::bitperfect::fault::BACKEND_WRITE);
        assert_eq!(
            sh.fault(),
            crate::bitperfect::fault::BACKEND_WRITE,
            "the write is what ended it, and that is the reason reported"
        );
        assert_eq!(
            sh.completion().failure(),
            Some(crate::bitperfect::fault::BACKEND_WRITE)
        );
        assert_eq!(
            sh.revoked(),
            crate::bitperfect::fault::UNDERRUN,
            "and the dropout is still on the record"
        );
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 1);
    }

    /// A writer that dies takes its stream with it.
    ///
    /// The ASIO host thread has had this since Phase C; the ALSA writer did
    /// not. A panic left `failed` clear and no terminal state, so the handle
    /// went on looking alive and the next track was handed to a stream with
    /// nobody behind it — playing silence, with the diamond lit.
    #[test]
    fn a_writer_that_dies_publishes_it_however_it_died() {
        // Ordinary return.
        let sh = shared_for_test();
        drop(WriterDeath(Arc::clone(&sh)));
        assert!(sh.failed.load(Ordering::Acquire));
        assert_eq!(
            sh.completion().failure(),
            Some(crate::bitperfect::fault::BACKEND_DEAD)
        );
        assert!(
            !sh.completion().may_advance(),
            "a writer that stopped did not finish the track"
        );

        // Unwind.
        let sh = shared_for_test();
        let outcome = std::panic::catch_unwind({
            let sh = Arc::clone(&sh);
            move || {
                let _death = WriterDeath(sh);
                panic!("injected unwind in the ALSA writer");
            }
        });
        assert!(outcome.is_err(), "the panic must actually happen");
        assert!(sh.failed.load(Ordering::Acquire));
        assert_eq!(
            sh.completion().failure(),
            Some(crate::bitperfect::fault::BACKEND_DEAD)
        );
    }

    /// The drain waits in byte-frames, which is what the writer counts.
    ///
    /// `period_frames` held the negotiated ALSA period — PCM frames of `bps`
    /// bytes each — while the writer advances the drain in byte-frames. On
    /// `DSD_U32` that is a factor of four, and the drain ended after half a
    /// period instead of two: on DSD, the DAC losing its lock mid-hand-off.
    #[test]
    fn the_alsa_drain_waits_in_byte_frames() {
        let sh = shared_for_test();
        // 4096 byte-frames is what a 1024-frame period holds at DSD_U32.
        sh.period_frames.store(4_096, Ordering::Relaxed);
        let at = sh.begin_session();
        sh.begin_drain(sh.generation());

        sh.advance_drain(4_096);
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "one period is not two"
        );
        sh.advance_drain(4_096);
        assert_eq!(sh.completion(), crate::bitperfect::Completion::CleanEof);
        let _ = at;
    }

    #[test]
    fn pack_u32_be_is_time_order() {
        let s = scratch(8);
        let mut out = vec![0u8; 2 * 2 * 4]; // 2 ALSA frames × 2ch × 4B
        pack_chunk(&s, 8, 2, 4, false, &mut out);
        // Frame 0: ch0 = oldest four ch0 bytes in memory order, then ch1.
        assert_eq!(&out[0..4], &[0xA0, 0xA1, 0xA2, 0xA3]);
        assert_eq!(&out[4..8], &[0xB0, 0xB1, 0xB2, 0xB3]);
        assert_eq!(&out[8..12], &[0xA4, 0xA5, 0xA6, 0xA7]);
        assert_eq!(&out[12..16], &[0xB4, 0xB5, 0xB6, 0xB7]);
    }

    #[test]
    fn pack_u32_le_swaps_bytes_within_sample() {
        let s = scratch(4);
        let mut out = vec![0u8; 1 * 2 * 4];
        pack_chunk(&s, 4, 2, 4, true, &mut out);
        assert_eq!(&out[0..4], &[0xA3, 0xA2, 0xA1, 0xA0]);
        assert_eq!(&out[4..8], &[0xB3, 0xB2, 0xB1, 0xB0]);
    }

    #[test]
    fn pack_u16_and_u8() {
        let s = scratch(2);
        let mut out16 = vec![0u8; 1 * 2 * 2];
        pack_chunk(&s, 2, 2, 2, false, &mut out16);
        assert_eq!(out16, vec![0xA0, 0xA1, 0xB0, 0xB1]);

        let mut out8 = vec![0u8; 2 * 2 * 1];
        pack_chunk(&s, 2, 2, 1, false, &mut out8);
        // U8 is exactly the reader's interleaving, de-/re-interleaved 1:1.
        assert_eq!(out8, vec![0xA0, 0xB0, 0xA1, 0xB1]);
    }

    #[test]
    fn pack_pads_missing_frames_with_dsd_silence() {
        let s = scratch(3); // 3 of 4 byte-frames available
        let mut out = vec![0u8; 1 * 2 * 4];
        pack_chunk(&s, 3, 2, 4, false, &mut out);
        assert_eq!(&out[0..4], &[0xA0, 0xA1, 0xA2, DSD_SILENCE]);
        assert_eq!(&out[4..8], &[0xB0, 0xB1, 0xB2, DSD_SILENCE]);
    }

    #[test]
    fn alsa_rates_for_dsd() {
        // rate = bit rate ÷ (8 × bytes per sample)
        for (dsd, u32_rate) in [
            (2_822_400u32, 88_200u32),   // DSD64
            (5_644_800, 176_400),        // DSD128
            (11_289_600, 352_800),       // DSD256
            (22_579_200, 705_600),       // DSD512
            (45_158_400, 1_411_200),     // DSD1024
        ] {
            assert_eq!(dsd / (8 * 4), u32_rate);
            assert_eq!(dsd / (8 * 1), u32_rate * 4); // DSD_U8 equivalent
        }
    }

    #[test]
    fn device_scan_never_panics() {
        // Containers/CI have no sound hardware — the scan must degrade to an
        // empty list, not error out.
        let _ = scan_devices();
    }

    /// End-to-end through ALSA's `null` device (accepts every format,
    /// discards the data): negotiation, the writer thread, session install,
    /// byte-frame position accounting and finish detection all run for real
    /// — everything except an actual DAC. Skips (passes) on systems whose
    /// alsa-lib can't open `null`.
    #[test]
    fn null_device_end_to_end() {
        let stream = match AlsaDsdStream::open("null", 2_822_400, 2) {
            Ok(s) => s,
            Err(e) => { crate::mlog!("skipping: {e}"); return; }
        };
        assert_eq!(stream.format_label, "DSD_U32_BE"); // first preference
        assert!(!stream.lsb_first);

        // Exactly 0.1 s of DSD64: 35 280 byte-frames × 2ch.
        //
        // Through the frame ring, because that is what the session takes now:
        // a raw `rtrb::Consumer<u8>` cannot express "whole byte-frames only",
        // and a partial frame in this ring rotates every channel for the rest
        // of the session.
        let frames = 35_280usize;
        let (mut prod, cons) = crate::bitperfect::frame_ring::channel_ring::<u8>(2, frames + 16);
        let data: Vec<u8> = (0..frames * 2).map(|i| (i % 251) as u8).collect();
        let mut off = 0usize;
        while off < data.len() {
            match prod.push_frames(&data[off..]) {
                Ok(0) => panic!("the ring was sized to hold the whole clip"),
                Ok(n) => off += n * 2,
                Err(e) => panic!("partial frame rejected: {e:?}"),
            }
        }
        let done = Arc::new(AtomicBool::new(true)); // all data pre-pushed
        // The clip is complete, so this is a clean end of file — the one
        // outcome that may advance a playlist.
        let decode_eof = Arc::new(AtomicBool::new(true));
        let session_stop = Arc::new(AtomicBool::new(false));
        stream.start_session(cons, done, decode_eof, session_stop);

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !stream.is_finished() {
            assert!(std::time::Instant::now() < deadline, "stream never finished");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!stream.failed());
        assert_eq!(
            stream.played(),
            Duration::from_secs_f64(frames as f64 * 8.0 / 2_822_400.0),
            "position must count exactly the byte-frames that were audio"
        );
    }
}

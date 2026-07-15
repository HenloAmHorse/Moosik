// ---------------------------------------------------------------------------
// Bit-perfect output path
//
// Architecture:
//   decode thread (symphonia, full sample precision, real seeking)
//        │  f32 samples, exact for sources up to 24-bit
//        ▼
//   lock-free SPSC ring buffer (rtrb, ~1 s)
//        │
//        ▼
//   platform backend — pops samples, converts f32 → the device's native
//   format with power-of-two scaling (bit-transparent round-trip), taps the
//   spectrum buffers, counts frames for sample-accurate position:
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

#[cfg(not(windows))]
mod cpal_out;
#[cfg(windows)]
mod wasapi_out;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder as SymDecoder, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;

use crate::dsd::dop::{word_to_f32, DopFileStream};
use crate::spectrum::{SampleBuf, StereoBuf, DEFAULT_FFT_SIZE};

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
    pub channels:    u16,
    /// Source bit depth if the container declares it (e.g. 16/24 for FLAC).
    pub bits_per_sample: Option<u32>,
}

/// Open `path` with symphonia, optionally seek to `start`, and return a ready
/// decode pipeline plus the stream's native rate/channels.
pub fn prepare(path: &Path, start: Duration) -> Result<Prepared, String> {
    let file = File::open(path).map_err(|e| format!("open failed: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss,
                &FormatOptions { enable_gapless: true, ..Default::default() },
                &MetadataOptions::default())
        .map_err(|e| format!("unrecognized format: {e}"))?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("no audio track")?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let sample_rate = params.sample_rate.ok_or("unknown sample rate")?;
    let channels = params.channels.map(|c| c.count() as u16).ok_or("unknown channel layout")?;
    let bits_per_sample = params.bits_per_sample;

    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder: {e}"))?;

    if start > Duration::ZERO {
        // Real container-level seek — no decode-and-discard.
        let _ = format.seek(SeekMode::Coarse, SeekTo::Time {
            time: Time::from(start.as_secs_f64()),
            track_id: Some(track_id),
        });
        decoder.reset();
    }

    Ok(Prepared { format, decoder, track_id, sample_rate, channels, bits_per_sample })
}

// ---------------------------------------------------------------------------
// Shared render-side state (used by both backends)
// ---------------------------------------------------------------------------

/// One playing track (or seek segment): the ring consumer plus the flags
/// shared with its decode thread.
struct Session {
    cons: rtrb::Consumer<f32>,
    decode_done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

struct Shared {
    session: Mutex<Option<Session>>,
    paused: AtomicBool,
    /// f32 volume as bits; exactly 1.0 means the multiply is skipped entirely.
    volume_bits: AtomicU32,
    finished: AtomicBool,
    /// Frames handed to the device since the last `start()` — drives the
    /// sample-accurate position readout.
    frames_played: AtomicU64,
    /// Gapless: the next track's decode pipeline, queued by the UI thread and
    /// picked up by the decode thread the instant the current file ends (only
    /// ever holds a track with the same rate/channels as the open stream).
    next: Mutex<Option<Prepared>>,
    /// Gapless for DSD: the next DoP stream, same role as `next` but for the
    /// DoP decode loop (only ever holds a DSD track at the same carrier
    /// rate/channels as the open stream).
    next_dop: Mutex<Option<DopFileStream>>,
    /// Cumulative frame counts (since `start()`) at which one track ends and
    /// the next begins — pushed by the decode thread at each gapless hand-off,
    /// consumed by the UI thread once the device has played past them to roll
    /// the displayed track over.
    boundaries: Mutex<std::collections::VecDeque<u64>>,
    /// True while a DoP (DSD) session is active. The ring then carries packed
    /// DoP words, not audio samples: volume is hard-bypassed regardless of
    /// `volume_bits` (any scaling corrupts the marker/bit pattern, not just
    /// precision) and the spectrum tap isn't fed.
    dop_active: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        Shared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            finished: AtomicBool::new(false),
            frames_played: AtomicU64::new(0),
            next: Mutex::new(None),
            next_dop: Mutex::new(None),
            boundaries: Mutex::new(std::collections::VecDeque::new()),
            dop_active: AtomicBool::new(false),
        }
    }
}

/// Fill `scratch` with up to `want` samples from the active session, handle
/// pause / end-of-track, feed the spectrum tap, count frames, apply volume.
/// `scratch` may come back shorter than `want` (pad with silence).
/// This is the entire realtime render logic, shared by both backends.
fn render_samples(
    sh: &Shared,
    scratch: &mut Vec<f32>,
    tap: &mut SpectrumTap,
    channels: u16,
    want: usize,
) {
    scratch.clear();

    if !sh.paused.load(Ordering::Relaxed)
        && let Ok(mut guard) = sh.session.try_lock()
        && let Some(sess) = guard.as_mut()
    {
        while scratch.len() < want {
            match sess.cons.pop() {
                Ok(s) => scratch.push(s),
                Err(_) => break,
            }
        }
        // Drained and the decoder has finished → track over.
        if scratch.len() < want
            && sess.decode_done.load(Ordering::Acquire)
            && sess.cons.is_empty()
        {
            *guard = None;
            sh.finished.store(true, Ordering::Release);
        }
    }

    let dop = sh.dop_active.load(Ordering::Relaxed);

    // A DoP session's "samples" are packed marker+DSD-bit words, not audio —
    // feeding them to the analyzer would just be noise, and any multiply
    // (volume) corrupts the bit pattern the DAC needs to see exactly.
    if !dop {
        tap.feed(scratch);
    }
    sh.frames_played.fetch_add(
        (scratch.len() / channels.max(1) as usize) as u64,
        Ordering::Relaxed);

    if !dop {
        let vol_bits = sh.volume_bits.load(Ordering::Relaxed);
        if vol_bits != 1.0f32.to_bits() {
            let vol = f32::from_bits(vol_bits);
            for s in scratch.iter_mut() { *s *= vol; }
        }
    }
}

// ---------------------------------------------------------------------------
// Stream facade over the platform backends
// ---------------------------------------------------------------------------

enum Backend {
    #[cfg(not(windows))]
    Cpal(#[allow(dead_code)] cpal_out::Handle),
    #[cfg(windows)]
    Wasapi(#[allow(dead_code)] wasapi_out::Handle),
}

pub struct BpStream {
    _backend: Backend,
    shared: Arc<Shared>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Negotiated device format label, e.g. "24i excl".
    pub format_label: String,
    /// Resolved device name (for display).
    pub device_name: String,
    /// Device the stream was requested on (None = system default) —
    /// used to decide whether an open stream can be reused.
    pub requested_device: Option<String>,
    /// True if this stream was negotiated for DoP (DSD): format selection was
    /// restricted to integer PCM (no float, no 16-bit) and volume/the
    /// spectrum tap are hard-bypassed for every session opened on it.
    pub dop: bool,
}

impl BpStream {
    /// Open an output stream on `device_name` (None = system default) at
    /// exactly `sample_rate`/`channels`. `src_bits` (the source's bit depth,
    /// when known) steers the device-format choice so the container is at
    /// least as wide as the source. `dop` forces an integer-only, ≥24-bit
    /// container (never float, never 16-bit) — required so a DoP-aware DAC
    /// sees the literal packed marker/DSD bit pattern, not a float encoding
    /// of it or a truncated one.
    pub fn open(
        device_name: Option<&str>,
        sample_rate: u32,
        channels: u16,
        src_bits: Option<u32>,
        dop: bool,
        sample_buf: SampleBuf,
        stereo_buf: StereoBuf,
    ) -> Result<Self, String> {
        let shared = Arc::new(Shared::new());
        shared.dop_active.store(dop, Ordering::Relaxed);
        let tap = SpectrumTap::new(channels, sample_buf, stereo_buf);

        #[cfg(not(windows))]
        let (handle, format_label, dev_label) = cpal_out::open(
            device_name, sample_rate, channels, src_bits, dop, Arc::clone(&shared), tap)?;
        #[cfg(not(windows))]
        let backend = Backend::Cpal(handle);

        #[cfg(windows)]
        let (handle, format_label, dev_label) = wasapi_out::open(
            device_name, sample_rate, channels, src_bits, dop, Arc::clone(&shared), tap)?;
        #[cfg(windows)]
        let backend = Backend::Wasapi(handle);

        Ok(BpStream {
            _backend: backend,
            shared,
            sample_rate,
            channels,
            format_label,
            device_name: dev_label,
            requested_device: device_name.map(str::to_owned),
            dop,
        })
    }

    /// Start playing a prepared decode pipeline. Replaces any running session;
    /// the old decode thread is signalled to stop and exits on its own.
    pub fn start(&self, prep: Prepared, volume: f32) {
        let ring_cap = (self.sample_rate as usize * self.channels as usize).max(65_536); // ~1 s
        let (prod, cons) = rtrb::RingBuffer::new(ring_cap);
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        // A fresh session supersedes any gapless queue/boundaries from the last.
        if let Ok(mut g) = self.shared.next.lock() { *g = None; }
        if let Ok(mut b) = self.shared.boundaries.lock() { b.clear(); }

        {
            let stop = Arc::clone(&stop);
            let done = Arc::clone(&done);
            let shared = Arc::clone(&self.shared);
            std::thread::Builder::new()
                .name("bp-decode".into())
                .spawn(move || decode_loop(prep, prod, done, stop, shared))
                .ok();
        }

        self.set_volume(volume);
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
        if let Ok(mut g) = self.shared.session.lock() {
            *g = Some(Session { cons, decode_done: done, stop }); // old Session drop signals its thread
        }
    }

    /// Start a DoP (DSD) session: packs and streams `stream`'s DSD data as
    /// pre-encoded 24-bit PCM words. Volume is force-reset to unity and the
    /// spectrum tap is skipped for the session's whole lifetime (`Shared::
    /// dop_active`, set at `open()`) — there is no gapless hand-off (DSD
    /// tracks are never queued onto a stream; each session is one full file).
    pub fn start_dop(&self, stream: DopFileStream) {
        debug_assert!(self.dop, "start_dop called on a stream not opened with dop=true");
        let ring_cap = (self.sample_rate as usize * self.channels as usize).max(65_536); // ~1 s
        let (prod, cons) = rtrb::RingBuffer::new(ring_cap);
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        if let Ok(mut g) = self.shared.next.lock() { *g = None; }
        if let Ok(mut g) = self.shared.next_dop.lock() { *g = None; }
        if let Ok(mut b) = self.shared.boundaries.lock() { b.clear(); }

        {
            let stop = Arc::clone(&stop);
            let done = Arc::clone(&done);
            let shared = Arc::clone(&self.shared);
            std::thread::Builder::new()
                .name("bp-dop-decode".into())
                .spawn(move || dop_decode_loop(stream, prod, done, stop, shared))
                .ok();
        }

        // Belt-and-suspenders: render_samples already ignores volume_bits
        // while dop_active, but force it to unity too so nothing downstream
        // that reads volume_bits directly gets a stale attenuated value.
        self.shared.volume_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
        if let Ok(mut g) = self.shared.session.lock() {
            *g = Some(Session { cons, decode_done: done, stop });
        }
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
    pub fn take_reached_boundary(&self) -> Option<Duration> {
        let played = self.shared.frames_played.load(Ordering::Relaxed);
        let mut b = self.shared.boundaries.lock().ok()?;
        match b.front().copied() {
            Some(front) if played >= front => {
                b.pop_front();
                Some(Duration::from_secs_f64(front as f64 / self.sample_rate.max(1) as f64))
            }
            _ => None,
        }
    }

    /// Stop the current session (decode thread winds down; stream stays open
    /// for instant reuse on the next track at the same rate).
    pub fn stop_session(&self) {
        if let Ok(mut g) = self.shared.session.lock() { *g = None; }
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
    }

    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Relaxed); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Relaxed); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    pub fn is_finished(&self) -> bool { self.shared.finished.load(Ordering::Acquire) }

    /// No-op on a DoP stream — volume is hard-bypassed for DSD (see
    /// `Shared::dop_active`); a scaled sample is not a valid DoP word.
    pub fn set_volume(&self, v: f32) {
        if self.dop { return; }
        self.shared.volume_bits.store(v.to_bits(), Ordering::Relaxed);
    }

    /// Sample-accurate elapsed time within the current session.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played.load(Ordering::Relaxed);
        Duration::from_secs_f64(frames as f64 / self.sample_rate.max(1) as f64)
    }

    /// Short description for the status line, e.g. "192 kHz · 2ch · 24i excl".
    pub fn describe(&self) -> String {
        format!("{} · {}ch · {}", fmt_khz(self.sample_rate), self.channels, self.format_label)
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
    ch_idx: u16,
    pending_l: f32,
    sample_buf: SampleBuf,
    stereo_buf: StereoBuf,
    batch: Vec<f32>,
    stereo_batch: Vec<[f32; 2]>,
}

const TAP_BATCH: usize = 512;

impl SpectrumTap {
    fn new(channels: u16, sample_buf: SampleBuf, stereo_buf: StereoBuf) -> Self {
        Self {
            channels, ch_idx: 0, pending_l: 0.0, sample_buf, stereo_buf,
            batch: Vec::with_capacity(TAP_BATCH * 2),
            stereo_batch: Vec::with_capacity(TAP_BATCH),
        }
    }

    fn feed(&mut self, samples: &[f32]) {
        for &f in samples {
            if self.channels == 2 {
                if self.ch_idx == 0 { self.pending_l = f; }
                else { self.stereo_batch.push([self.pending_l, f]); }
                self.ch_idx = (self.ch_idx + 1) % 2;
            }
            self.batch.push(f);
        }
        if self.batch.len() >= TAP_BATCH {
            if let Ok(mut v) = self.sample_buf.try_lock() {
                v.extend_from_slice(&self.batch);
                const CAP: usize = DEFAULT_FFT_SIZE * 4;
                if v.len() > CAP { let d = v.len() - CAP; v.drain(0..d); }
            }
            self.batch.clear();
            if !self.stereo_batch.is_empty() {
                if let Ok(mut v) = self.stereo_buf.try_lock() {
                    v.extend_from_slice(&self.stereo_batch);
                    const CAP: usize = 8192;
                    if v.len() > CAP { let d = v.len() - CAP; v.drain(0..d); }
                }
                self.stereo_batch.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decode thread
// ---------------------------------------------------------------------------

fn decode_loop(
    mut prep: Prepared,
    mut prod: rtrb::Producer<f32>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
) {
    let mut buf: Option<SampleBuffer<f32>> = None;
    // Frames pushed into the ring since `start()`, across every gapless track.
    // A push is FIFO into the ring and then FIFO to the device, so this running
    // total is exactly the play-time boundary between one track and the next.
    let mut frames_pushed: u64 = 0;

    'track: loop {
        // Decode the current file until it ends (or the session is superseded).
        loop {
            if stop.load(Ordering::Relaxed) { return; }

            let packet = match prep.format.next_packet() {
                Ok(p) => p,
                Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(SymError::ResetRequired) => break,
                Err(e) => { eprintln!("[bit-perfect] read error: {e}"); break; }
            };
            if packet.track_id() != prep.track_id { continue; }

            let decoded = match prep.decoder.decode(&packet) {
                Ok(d) => d,
                Err(SymError::DecodeError(e)) => { eprintln!("[bit-perfect] skipping bad packet: {e}"); continue; }
                Err(e) => { eprintln!("[bit-perfect] decode error: {e}"); break; }
            };

            let spec = *decoded.spec();
            let ch = spec.channels.count().max(1) as u64;
            let needed = decoded.capacity() as u64;
            if buf.as_ref().map(|b| (b.capacity() as u64) < needed * spec.channels.count() as u64)
                  .unwrap_or(true) {
                buf = Some(SampleBuffer::<f32>::new(needed, spec));
            }
            let buf = buf.as_mut().unwrap();
            buf.copy_interleaved_ref(decoded);

            // Push with backpressure — the ring holds ~1 s, so this thread spends
            // most of its life asleep here.
            for &s in buf.samples() {
                loop {
                    if stop.load(Ordering::Relaxed) { return; }
                    match prod.push(s) {
                        Ok(()) => break,
                        Err(_) => std::thread::sleep(Duration::from_millis(5)),
                    }
                }
            }
            frames_pushed += buf.samples().len() as u64 / ch;
        }

        // The current file is exhausted. Continue straight into a queued next
        // track (gapless), else signal completion.
        let next = shared.next.lock().ok().and_then(|mut g| g.take());
        match next {
            Some(next_prep) => {
                if let Ok(mut b) = shared.boundaries.lock() { b.push_back(frames_pushed); }
                prep = next_prep;
                continue 'track;
            }
            None => break,
        }
    }

    done.store(true, Ordering::Release);
}

/// DoP counterpart of `decode_loop`: pulls packed DoP words straight from the
/// DSD file (no symphonia, no sample-format conversion) and pushes their exact
/// f32 encoding into the ring. Gapless-capable — when the file ends it swaps
/// to a queued same-rate DSD track, carrying the marker phase so the boundary
/// word keeps the 0x05/0xFA alternation intact.
fn dop_decode_loop(
    mut stream: DopFileStream,
    mut prod: rtrb::Producer<f32>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
) {
    const CHUNK_PCM_FRAMES: usize = 4096;
    let mut words: Vec<u32> = Vec::new();
    // Cumulative PCM frames pushed since start() — the gapless boundary basis,
    // consumed by take_reached_boundary against the device's frames_played.
    let mut frames_pushed: u64 = 0;

    // Feed the DAC a short lead of DoP-marked silence so it locks into DSD
    // mode before the first audio sample, avoiding a start-of-track transient.
    // ~24 ms is ample for lock and imperceptible as a lead-in.
    let warmup_frames = (stream.carrier_rate() as usize * 24 / 1000).max(64);
    words.clear();
    stream.warmup_silence(warmup_frames, &mut words);
    if push_words(&words, &mut prod, &stop) { return; }
    frames_pushed += warmup_frames as u64;

    'file: loop {
        loop {
            if stop.load(Ordering::Relaxed) { return; }
            words.clear();
            let n = match stream.read_dop(CHUNK_PCM_FRAMES, &mut words) {
                Ok(n) => n,
                Err(e) => { eprintln!("[dsd] read error: {e}"); break 'file; }
            };
            if n == 0 { break; }
            if push_words(&words, &mut prod, &stop) { return; }
            frames_pushed += n as u64;
        }

        // Current file exhausted — continue into a queued DSD track (gapless),
        // else finish. Carry the marker phase so the boundary word alternates.
        let next = shared.next_dop.lock().ok().and_then(|mut g| g.take());
        match next {
            Some(mut next_stream) => {
                next_stream.set_marker_phase(stream.marker_phase());
                if let Ok(mut b) = shared.boundaries.lock() { b.push_back(frames_pushed); }
                stream = next_stream;
                continue 'file;
            }
            None => break,
        }
    }

    // Session over: trail with DoP-marked silence so the DAC stays locked in
    // DSD mode while the ring drains, instead of dropping to plain zeros
    // (PCM silence) at the very end — some DACs pop when they lose DSD lock
    // mid-drain. Mirrors the warm-up at the head.
    if !stop.load(Ordering::Relaxed) {
        let tail_frames = (stream.carrier_rate() as usize * 24 / 1000).max(64);
        words.clear();
        stream.warmup_silence(tail_frames, &mut words);
        if push_words(&words, &mut prod, &stop) { return; }
    }

    done.store(true, Ordering::Release);
}

/// Push DoP words (as f32) into the ring with backpressure. Returns true if a
/// stop was signalled mid-push (caller should return immediately).
fn push_words(words: &[u32], prod: &mut rtrb::Producer<f32>, stop: &AtomicBool) -> bool {
    for &w in words {
        let s = word_to_f32(w);
        loop {
            if stop.load(Ordering::Relaxed) { return true; }
            match prod.push(s) {
                Ok(()) => break,
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Persisted settings
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
pub struct BpSettings {
    pub enabled: bool,
    /// Selected output device name; None = system default.
    pub device: Option<String>,
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

    /// Minimal 16-bit PCM WAV writer (44-byte canonical header).
    fn write_wav(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) {
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * channels as u32 * 2;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for s in samples { out.extend_from_slice(&s.to_le_bytes()); }
        std::fs::write(path, out).unwrap();
    }

    fn drain(prep: Prepared) -> Vec<f32> {
        let (prod, mut cons) = rtrb::RingBuffer::new(4 << 20);
        let done = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared::new());
        decode_loop(prep, prod, Arc::clone(&done), Arc::new(AtomicBool::new(false)), shared);
        assert!(done.load(Ordering::Acquire), "decode loop must signal completion");
        let mut out = Vec::new();
        while let Ok(s) = cons.pop() { out.push(s); }
        out
    }

    #[test]
    fn decode_is_bit_exact_and_complete() {
        let sr = 48_000u32;
        let src: Vec<i16> = (0..sr as usize * 2) // 1 s stereo ramp
            .map(|i| ((i % 65_536) as i64 - 32_768) as i16)
            .collect();
        let path = std::env::temp_dir().join("moosik_bp_test.wav");
        write_wav(&path, &src, sr, 2);

        let prep = prepare(&path, Duration::ZERO).unwrap();
        assert_eq!(prep.sample_rate, sr);
        assert_eq!(prep.channels, 2);

        let decoded = drain(prep);
        assert_eq!(decoded.len(), src.len());
        // i16 → f32 (÷2^15) must be exact, and the I16 writeback (×2^15)
        // must round-trip every value.
        for (&f, &i) in decoded.iter().zip(&src) {
            assert_eq!(f, i as f32 / 32_768.0);
            assert_eq!((f * 32_768.0) as i16, i);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prepare_seeks_without_decoding_from_start() {
        let sr = 44_100u32;
        let src: Vec<i16> = vec![1000; sr as usize * 2 * 2]; // 2 s stereo
        let path = std::env::temp_dir().join("moosik_bp_seek_test.wav");
        write_wav(&path, &src, sr, 2);

        let prep = prepare(&path, Duration::from_secs(1)).unwrap();
        let decoded = drain(prep);
        // Roughly 1 s of stereo should remain (allow one packet of slack
        // for the coarse container seek).
        let expect = sr as usize * 2;
        assert!(decoded.len() <= expect + 8192 && decoded.len() >= expect - 8192,
                "got {} samples, expected ≈{expect}", decoded.len());
        let _ = std::fs::remove_file(&path);
    }
}

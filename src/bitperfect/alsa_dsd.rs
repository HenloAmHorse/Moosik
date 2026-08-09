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
use std::time::Duration;

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
    cons: rtrb::Consumer<u8>,
    decode_done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

struct Shared {
    session: Mutex<Option<Session>>,
    paused: AtomicBool,
    finished: AtomicBool,
    /// Byte-frames (1 byte = 8 DSD samples, per channel) delivered as audio.
    frames_played: AtomicU64,
    /// The writer hit an unrecoverable device error (e.g. USB unplug) —
    /// surfaced as end-of-session; the next play re-opens from scratch.
    failed: AtomicBool,
}

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

impl AlsaDsdStream {
    /// Open `device` (an ALSA `hw:` string) in native-DSD mode at `dsd_rate`
    /// (the bit rate, e.g. 22 579 200 for DSD512) with `channels` outputs.
    /// Negotiation runs synchronously; the writer thread takes over after.
    pub fn open(device: &str, dsd_rate: u32, channels: u16) -> Result<Self, String> {
        let pcm = PCM::new(device, Direction::Playback, false).map_err(|e| {
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
                    eprintln!(
                        "[alsa-dsd] \"{device}\": {format} @ {rate} Hz accepted \
                         (period {period} frames, buffer {buffer})"
                    );
                    negotiated = Some((format, bps, swap, period, buffer));
                    break;
                }
                Err(e) => {
                    eprintln!("[alsa-dsd] \"{device}\": {format} @ {rate} Hz: {e}");
                    if !errors.is_empty() { errors.push_str("; "); }
                    errors.push_str(&format!("{format}: {e}"));
                }
            }
        }
        let Some((format, bps, swap, period, _buffer)) = negotiated else {
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
            finished: AtomicBool::new(false),
            frames_played: AtomicU64::new(0),
            failed: AtomicBool::new(false),
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
        cons: rtrb::Consumer<u8>,
        decode_done: Arc<AtomicBool>,
        session_stop: Arc<AtomicBool>,
    ) {
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
        if let Ok(mut g) = self.shared.session.lock() {
            *g = Some(Session { cons, decode_done, stop: session_stop });
        }
    }

    pub fn stop_session(&self) {
        if let Ok(mut g) = self.shared.session.lock() { *g = None; }
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
    }

    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Relaxed); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Relaxed); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    pub fn is_finished(&self) -> bool { self.shared.finished.load(Ordering::Acquire) }
    pub fn failed(&self) -> bool { self.shared.failed.load(Ordering::Acquire) }

    /// Sample-accurate elapsed time: byte-frames delivered × 8 samples ÷ rate.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played.load(Ordering::Relaxed);
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
    let io = pcm.io_bytes();
    let frames = period as usize;            // ALSA frames per chunk
    let byte_frames = frames * bps;          // byte-frames per chunk
    let frame_bytes = ch * bps;
    let want = byte_frames * ch;
    let mut scratch: Vec<u8> = Vec::with_capacity(want);
    let mut packed = vec![DSD_SILENCE; frames * frame_bytes];

    'outer: while !stop.load(Ordering::Relaxed) {
        scratch.clear();
        // Not a realtime callback — a full lock is fine; the only contender
        // is the UI thread swapping sessions for an instant.
        if !shared.paused.load(Ordering::Relaxed)
            && let Ok(mut guard) = shared.session.lock()
            && let Some(sess) = guard.as_mut()
        {
            while scratch.len() < want {
                match sess.cons.pop() {
                    Ok(b) => scratch.push(b),
                    Err(_) => break,
                }
            }
            if scratch.len() < want
                && sess.decode_done.load(Ordering::Acquire)
                && sess.cons.is_empty()
            {
                *guard = None;
                shared.finished.store(true, Ordering::Release);
            }
        }

        // Whole byte-frames only — a torn frame would swap channels.
        let got_frames = scratch.len() / ch;
        shared.frames_played.fetch_add(got_frames as u64, Ordering::Relaxed);

        pack_chunk(&scratch, got_frames, ch, bps, swap, &mut packed);

        let mut off = 0;
        while off < packed.len() {
            if stop.load(Ordering::Relaxed) { break 'outer; }
            match io.writei(&packed[off..]) {
                Ok(n) => off += n * frame_bytes,
                Err(e) => {
                    // Underrun/suspend recover in place; anything else
                    // (device unplugged, format yanked) ends the stream.
                    if pcm.try_recover(e, true).is_err() {
                        eprintln!("[alsa-dsd] \"{device}\": unrecoverable write error: {e}");
                        shared.failed.store(true, Ordering::Release);
                        shared.finished.store(true, Ordering::Release);
                        break 'outer;
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
            Err(e) => { eprintln!("skipping: {e}"); return; }
        };
        assert_eq!(stream.format_label, "DSD_U32_BE"); // first preference
        assert!(!stream.lsb_first);

        // Exactly 0.1 s of DSD64: 35 280 byte-frames × 2ch.
        let frames = 35_280usize;
        let (mut prod, cons) = rtrb::RingBuffer::new(frames * 2 + 16);
        for i in 0..frames * 2 { prod.push((i % 251) as u8).unwrap(); }
        let done = Arc::new(AtomicBool::new(true)); // all data pre-pushed
        let session_stop = Arc::new(AtomicBool::new(false));
        stream.start_session(cons, done, session_stop);

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

// WASAPI exclusive-mode backend for the bit-perfect path (Windows).
//
// Shared-mode WASAPI (what cpal uses) only accepts the format the Windows
// mixer is configured to — e.g. a DAC set to 384 kHz in the sound control
// panel rejects everything else. Exclusive mode talks to the device driver
// directly, so the device's real capabilities (44.1–384 kHz on a typical
// USB DAC) become available, exactly like foobar2000's WASAPI exclusive
// output.
//
// Polling mode (not event-driven) is used deliberately: event-driven
// exclusive mode is documented to stutter with USB audio class drivers.
//
// All WASAPI objects live on the thread that created them (COM apartment):
// `open()` spawns a render thread that does the entire negotiation and
// reports the result back over a channel before streaming starts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use wasapi::{
    calculate_period_100ns, initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode,
    WasapiError, WaveFormat,
};

use super::{fmt_khz, render_samples, AudioPriority, DeviceCaps, Shared, SpectrumTap, PROBE_RATES};

// HRESULTs from AUDCLNT_ERR (audioclient.h) we want to recognise.
const E_BUFFER_SIZE_NOT_ALIGNED: i32 = 0x8889_0019u32 as i32;
const E_DEVICE_IN_USE: i32 = 0x8889_000Au32 as i32;
const E_EXCLUSIVE_MODE_NOT_ALLOWED: i32 = 0x8889_000Eu32 as i32;

/// How samples are laid out in the device buffer.
#[derive(Clone, Copy, PartialEq)]
enum DevFmt {
    I16,
    /// 24-bit packed in 3 bytes.
    I24,
    /// 24- or 32-valid-bit samples in a 32-bit container (left-aligned).
    I32,
    F32,
}

/// Candidate device formats: (store bits, valid bits, sample type, writer, label).
const CANDIDATES: [(usize, usize, SampleType, DevFmt, &str); 5] = [
    (16, 16, SampleType::Int, DevFmt::I16, "16i excl"),
    (24, 24, SampleType::Int, DevFmt::I24, "24i excl"),
    (32, 24, SampleType::Int, DevFmt::I32, "24i/32 excl"),
    (32, 32, SampleType::Int, DevFmt::I32, "32i excl"),
    (32, 32, SampleType::Float, DevFmt::F32, "32f excl"),
];

/// Candidate order steered by the source bit depth: prefer the narrowest
/// container that holds the source exactly (a 16-bit file on a 16-bit pipe
/// is bit-perfect; padding up to 24/32 is too, but narrower is the classic
/// choice and what drivers support most reliably).
///
/// `dop` forces an integer-only, ≥24-bit container (indices 1-3: I24,
/// 24-in-32, 32i) — the ring carries pre-packed 24-bit DoP words, and a
/// DoP-aware DAC must see that literal integer bit pattern; F32 would send an
/// IEEE-754 encoding instead, and I16 (index 0) is too narrow to hold one.
fn candidate_order(src_bits: Option<u32>, dop: bool) -> Vec<usize> {
    if let Some(i) = forced_candidate() {
        return vec![i];
    }
    if dop {
        // Left alone deliberately. DoP is the one path where the device must
        // see an exact 24-bit word, it is verified working on real DoP DACs
        // with the packed container, and it is not what the Realtek problem
        // below is about.
        return vec![1, 2, 3];
    }
    match src_bits {
        Some(b) if b <= 16 => vec![0, 2, 3, 1, 4],
        Some(b) if b > 24 => vec![3, 4, 2, 1, 0],
        _ => vec![2, 1, 3, 4, 0], // 24-bit or unknown
    }
}

/// `MOOSIK_BP_FORMAT=16i|24i|24i32|32i|32f` pins the device format to one
/// candidate.
///
/// Bisecting a format problem otherwise means a rebuild per guess, and the
/// machine that has the problem is usually not the machine with the compiler.
fn forced_candidate() -> Option<usize> {
    let want = std::env::var("MOOSIK_BP_FORMAT").ok()?;
    let i = match want.trim() {
        "16i" => 0,
        "24i" => 1,
        "24i32" => 2,
        "32i" => 3,
        "32f" => 4,
        other => {
            crate::mlog!("bp      MOOSIK_BP_FORMAT=\"{other}\" not recognised, ignoring");
            return None;
        }
    };
    crate::mlog!("bp      MOOSIK_BP_FORMAT pins the device format to {}", CANDIDATES[i].4);
    Some(i)
}

fn wave_format(c: &(usize, usize, SampleType, DevFmt, &str), rate: u32, channels: u16) -> WaveFormat {
    WaveFormat::new(c.0, c.1, &c.2, rate as usize, channels as usize, None)
}

// ---------------------------------------------------------------------------
// Device probing (exclusive-mode capabilities)
// ---------------------------------------------------------------------------

pub fn probe_devices() -> Vec<DeviceCaps> {
    let _ = initialize_mta();
    let mut out = Vec::new();
    let Ok(enumerator) = DeviceEnumerator::new() else { return out };
    let default_name = enumerator
        .get_default_device(&Direction::Render)
        .and_then(|d| d.get_friendlyname())
        .ok();
    let Ok(collection) = enumerator.get_device_collection(&Direction::Render) else { return out };
    let n = collection.get_nbr_devices().unwrap_or(0);

    for i in 0..n {
        let Ok(device) = collection.get_device_at_index(i) else { continue };
        let Ok(name) = device.get_friendlyname() else { continue };
        let Ok(client) = device.get_iaudioclient() else { continue };

        let mut rates: Vec<u32> = Vec::new();
        let mut formats: Vec<String> = Vec::new();
        for &rate in &PROBE_RATES {
            for c in &CANDIDATES {
                if client
                    .is_supported_exclusive_with_quirks(&wave_format(c, rate, 2))
                    .is_ok()
                {
                    if !rates.contains(&rate) { rates.push(rate); }
                    if !formats.contains(&c.4.to_string()) { formats.push(c.4.to_string()); }
                }
            }
        }
        rates.sort_unstable();

        let max_channels = client
            .get_mixformat()
            .map(|f| f.get_nchannels())
            .unwrap_or(2);

        out.push(DeviceCaps {
            is_default: default_name.as_deref() == Some(name.as_str()),
            name,
            rates,
            formats,
            max_channels,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Render thread
// ---------------------------------------------------------------------------

/// Keeps the render thread alive; dropping stops the stream and joins.
pub struct Handle {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() { let _ = j.join(); }
    }
}

/// Open a WASAPI exclusive stream; returns the keep-alive handle, the
/// negotiated format label, and the resolved device name.
pub fn open(
    device_name: Option<&str>,
    sample_rate: u32,
    channels: u16,
    src_bits: Option<u32>,
    dop: bool,
    shared: Arc<Shared>,
    tap: SpectrumTap,
) -> Result<(Handle, String, String), String> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let stop_t = Arc::clone(&stop);
    let dev_name = device_name.map(str::to_owned);

    let join = std::thread::Builder::new()
        .name("bp-wasapi-render".into())
        .spawn(move || render_thread(dev_name, sample_rate, channels, src_bits, dop, shared, tap, stop_t, tx))
        .map_err(|e| format!("thread spawn failed: {e}"))?;

    match rx.recv() {
        Ok(Ok((label, dev))) => Ok((Handle { stop, join: Some(join) }, label, dev)),
        Ok(Err(e)) => { let _ = join.join(); Err(e) }
        Err(_) => { let _ = join.join(); Err("output thread died during setup".into()) }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_thread(
    device_name: Option<String>,
    sample_rate: u32,
    channels: u16,
    src_bits: Option<u32>,
    dop: bool,
    shared: Arc<Shared>,
    mut tap: SpectrumTap,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<(String, String), String>>,
) {
    let _ = initialize_mta();

    let setup = setup_stream(device_name.as_deref(), sample_rate, channels, src_bits, dop);
    let (client, render, fmt, dev_fmt, label, dev_label) = match setup {
        Ok(s) => s,
        Err(e) => { let _ = tx.send(Err(e)); return; }
    };

    let blockalign = fmt.get_blockalign() as usize;
    let period_hns = client
        .get_device_period()
        .map(|(d, _)| d)
        .unwrap_or(100_000); // 10 ms fallback
    let sleep = Duration::from_millis(((period_hns / 10_000) / 2).max(1) as u64);

    // Prefill the device buffer with silence, then go.
    if let Ok(avail) = client.get_available_space_in_frames() {
        let _ = render.write_to_device(avail as usize, &vec![0u8; avail as usize * blockalign], None);
    }
    if let Err(e) = client.start_stream() {
        let _ = tx.send(Err(format!("stream start failed: {e}")));
        return;
    }
    crate::mlog!(
        "bp      streaming \"{dev_label}\" as {label}: {} Hz, {}ch, blockalign {blockalign}, \
         period {:.2} ms",
        fmt.get_samplespersec(), fmt.get_nchannels(), period_hns as f64 / 10_000.0,
    );
    let _ = tx.send(Ok((label.clone(), dev_label)));

    // Held for the life of the loop: this thread has a hard deadline the rest
    // of the process (notably the rayon pre-process pool) does not.
    let _prio = AudioPriority::claim();

    let mut scratch: Vec<f32> = Vec::with_capacity(16_384);
    let mut bytes: Vec<u8> = Vec::with_capacity(65_536);

    // Per-write detail, opt-in. What the device is actually being handed, and
    // whether the ring is keeping up with it — the two things a "sounds wrong"
    // report cannot distinguish from the outside. The first writes carry the
    // interesting evidence, so they are logged in full and the steady state is
    // sampled after that.
    let trace = std::env::var("MOOSIK_BP_TRACE").is_ok_and(|v| v != "0");
    let mut iters: u64 = 0;
    let mut last_underruns = 0u64;
    let mut peak = 0.0f32;

    loop {
        if stop.load(Ordering::Relaxed) { break; }
        let avail = match client.get_available_space_in_frames() {
            Ok(a) => a as usize,
            Err(e) => { crate::mlog!("[bit-perfect] wasapi error: {e}"); break; }
        };
        if avail > 0 {
            let want = avail * channels as usize;
            render_samples(&shared, &mut scratch, &mut tap, channels, want);
            let got = scratch.len();
            fill_bytes(&mut bytes, &scratch, want, dev_fmt);
            if trace {
                // The sample values themselves. If these are sane and the audio
                // is not, the fault is past this point — in the container or the
                // driver, not in what was decoded.
                for &s in scratch.iter() { peak = peak.max(s.abs()); }
                iters += 1;
                if iters <= 20 || iters.is_multiple_of(500) {
                    crate::mlog!(
                        "bp      write #{iters}: avail {avail}f, want {want}, got {got}, \
                         {} bytes ({} expected), peak {peak:.4}",
                        bytes.len(), avail * blockalign,
                    );
                    peak = 0.0;
                }
            }
            if let Err(e) = render.write_to_device(avail, &bytes, None) {
                crate::mlog!("[bit-perfect] wasapi write error: {e}");
                break;
            }
            // Always reported, trace or not: a dropout is the one thing that is
            // audible and otherwise leaves no record at all.
            let u = shared.underruns.load(Ordering::Relaxed);
            if u != last_underruns {
                crate::mlog!("bp      {} dropout(s) so far", u);
                last_underruns = u;
            }
        }
        std::thread::sleep(sleep);
    }

    crate::mlog!(
        "bp      stream closed after {iters} writes, {} dropout(s)",
        shared.underruns.load(Ordering::Relaxed)
    );
    let _ = client.stop_stream();
}

type StreamSetup = (
    wasapi::AudioClient,
    wasapi::AudioRenderClient,
    WaveFormat,
    DevFmt,
    String,
    String,
);

fn setup_stream(
    device_name: Option<&str>,
    sample_rate: u32,
    channels: u16,
    src_bits: Option<u32>,
    dop: bool,
) -> Result<StreamSetup, String> {
    let enumerator = DeviceEnumerator::new()
        .map_err(|e| format!("device enumeration failed: {e}"))?;
    let device = match device_name {
        None => enumerator
            .get_default_device(&Direction::Render)
            .map_err(|e| format!("no default output device: {e}"))?,
        Some(n) => {
            let collection = enumerator
                .get_device_collection(&Direction::Render)
                .map_err(|e| format!("device enumeration failed: {e}"))?;
            let count = collection.get_nbr_devices().unwrap_or(0);
            let mut found = None;
            for i in 0..count {
                if let Ok(d) = collection.get_device_at_index(i)
                    && d.get_friendlyname().map(|fname| fname == n).unwrap_or(false)
                {
                    found = Some(d);
                    break;
                }
            }
            found.ok_or_else(|| format!("device \"{n}\" not found (unplugged?)"))?
        }
    };
    let dev_label = device.get_friendlyname().unwrap_or_else(|_| "?".into());

    let mut client = device.get_iaudioclient()
        .map_err(|e| format!("audio client failed: {e}"))?;

    crate::mlog!(
        "bp      negotiating \"{dev_label}\": {sample_rate} Hz, {channels}ch, src_bits {:?}, dop {dop}",
        src_bits
    );

    // Negotiate the device format in exclusive mode.
    let mut chosen: Option<(WaveFormat, DevFmt, &str)> = None;
    for idx in candidate_order(src_bits, dop) {
        let c = &CANDIDATES[idx];
        match client.is_supported_exclusive_with_quirks(&wave_format(c, sample_rate, channels)) {
            Ok(accepted) => {
                // What the device agreed to, read back from the format it
                // returned rather than assumed from the candidate. The writer
                // below lays out bytes from the *candidate*, so if these ever
                // disagree the device is fed a frame of the wrong width — which
                // sounds like noise playing at the wrong speed, and is the first
                // thing to check when it does.
                crate::mlog!(
                    "bp      accepted {} → {} Hz, {}ch, {} bits, blockalign {}",
                    c.4,
                    accepted.get_samplespersec(),
                    accepted.get_nchannels(),
                    accepted.get_bitspersample(),
                    accepted.get_blockalign(),
                );
                let want_align = (c.0 / 8) * channels as usize;
                if accepted.get_samplespersec() != sample_rate
                    || accepted.get_nchannels() != channels
                    || accepted.get_blockalign() as usize != want_align
                {
                    crate::mlog!(
                        "bp      MISMATCH requested {sample_rate} Hz/{channels}ch/align {want_align} \
                         — refusing rather than writing frames of the wrong width"
                    );
                    continue;
                }
                chosen = Some((accepted, c.3, c.4));
                break;
            }
            Err(e) => crate::mlog!("bp      rejected {}: {e}", c.4),
        }
    }
    let (fmt, dev_fmt, label) = chosen.ok_or_else(|| {
        // Build a helpful error: which standard rates DOES the device take?
        let rates: Vec<String> = PROBE_RATES.iter().copied()
            .filter(|&r| CANDIDATES.iter().any(|c| {
                client.is_supported_exclusive_with_quirks(&wave_format(c, r, channels)).is_ok()
            }))
            .map(fmt_khz)
            .collect();
        let dop_note = if dop { " (DoP requires a ≥24-bit integer format)" } else { "" };
        format!(
            "\"{dev_label}\" rejected {} / {channels}ch{dop_note} in exclusive mode (accepts: {})",
            fmt_khz(sample_rate),
            if rates.is_empty() { "no standard rates — is another app holding the device?".into() }
            else { rates.join(", ") },
        )
    })?;

    // Period aligned to 128 bytes (required by e.g. Intel HDA devices).
    let (def_period, _min_period) = client.get_device_period()
        .map_err(|e| format!("period query failed: {e}"))?;
    let period = client
        .calculate_aligned_period_near(def_period, Some(128), &fmt)
        .unwrap_or(def_period);

    let mode = StreamMode::PollingExclusive {
        period_hns: period,
        buffer_duration_hns: 8 * period,
    };

    if let Err(e) = client.initialize_client(&fmt, &Direction::Render, &mode) {
        match &e {
            WasapiError::Windows(werr) if werr.code().0 == E_BUFFER_SIZE_NOT_ALIGNED => {
                // Standard recovery: query the aligned size, redo with a
                // fresh client (the failed one is unusable).
                let frames = client.get_buffer_size()
                    .map_err(|e2| format!("alignment recovery failed: {e2}"))?;
                let aligned = calculate_period_100ns(frames as i64, fmt.get_samplespersec() as i64);
                client = device.get_iaudioclient()
                    .map_err(|e2| format!("audio client failed: {e2}"))?;
                let mode = StreamMode::PollingExclusive {
                    period_hns: aligned,
                    buffer_duration_hns: 8 * aligned,
                };
                client.initialize_client(&fmt, &Direction::Render, &mode)
                    .map_err(|e2| format!("exclusive init failed after alignment fix: {e2}"))?;
            }
            WasapiError::Windows(werr) if werr.code().0 == E_DEVICE_IN_USE => {
                return Err(format!(
                    "\"{dev_label}\" is already in exclusive use by another application"));
            }
            WasapiError::Windows(werr) if werr.code().0 == E_EXCLUSIVE_MODE_NOT_ALLOWED => {
                return Err(format!(
                    "exclusive mode is disabled for \"{dev_label}\" — enable \
                     'Allow applications to take exclusive control' in the \
                     device's Properties → Advanced tab"));
            }
            _ => return Err(format!("exclusive init failed: {e}")),
        }
    }

    let render = client.get_audiorenderclient()
        .map_err(|e| format!("render client failed: {e}"))?;

    Ok((client, render, fmt, dev_fmt, label.to_string(), dev_label))
}

/// Convert f32 samples to device bytes, padding with silence up to
/// `total_samples`. Power-of-two scaling keeps integer sources
/// bit-transparent (16-bit ↔ ×2^15, 24-bit → 24/32-bit container via ×2^23 /
/// ×2^31, both exact). Rust float→int `as` casts saturate at full scale.
fn fill_bytes(bytes: &mut Vec<u8>, samples: &[f32], total_samples: usize, fmt: DevFmt) {
    bytes.clear();
    for i in 0..total_samples {
        let v = samples.get(i).copied().unwrap_or(0.0);
        match fmt {
            DevFmt::I16 => {
                bytes.extend_from_slice(&((v * 32_768.0) as i16).to_le_bytes());
            }
            DevFmt::I24 => {
                let s = ((v as f64 * 8_388_608.0) as i32).clamp(-8_388_608, 8_388_607);
                bytes.extend_from_slice(&s.to_le_bytes()[..3]);
            }
            DevFmt::I32 => {
                bytes.extend_from_slice(&((v as f64 * 2_147_483_648.0) as i32).to_le_bytes());
            }
            DevFmt::F32 => {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

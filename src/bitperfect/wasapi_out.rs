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
/// The only containers that can carry a DoP word: integer, at least 24 bits.
///
/// DoP hides the DSD stream inside PCM samples whose top byte alternates
/// `0x05`/`0xFA` so the DAC can recognise it. A narrower container truncates
/// that marker and a float container re-encodes the bit pattern entirely; in
/// both cases the DAC sees ordinary PCM noise at DSD rates.
const DOP_CANDIDATES: [usize; 3] = [1, 2, 3];

/// Resolve the device-format order for this stream, or say why the request is
/// impossible — before any device is enumerated, opened, or negotiated with.
pub fn resolve_order(src_bits: Option<u32>, dop: bool) -> Result<Vec<usize>, super::OpenError> {
    let forced = forced_candidate().map_err(super::OpenError::config)?;
    candidate_policy(src_bits, dop, forced).map_err(super::OpenError::config)
}

/// Which device formats to try, in order — pure, so it can be tested across
/// every combination without touching the environment or a device.
///
/// `forced` comes from `MOOSIK_BP_FORMAT`. It used to be applied before the
/// DoP branch, which let the override reach a device with a container DoP
/// cannot survive: an operator debugging a PCM problem would set `16i`, play a
/// DSD file, and the DAC would receive truncated words while the route still
/// reported success. An override is a debugging aid, not a licence to violate
/// the format contract, so DoP now validates it rather than obeying it.
fn candidate_policy(
    src_bits: Option<u32>,
    dop: bool,
    forced: Option<usize>,
) -> Result<Vec<usize>, String> {
    if dop {
        return match forced {
            None => Ok(DOP_CANDIDATES.to_vec()),
            Some(i) if DOP_CANDIDATES.contains(&i) => Ok(vec![i]),
            // Refused before negotiation, and never quietly downgraded to the
            // default order: an operator who pinned a format needs to be told
            // it was impossible, not handed a different one.
            Some(i) => Err(format!(
                "MOOSIK_BP_FORMAT={} cannot carry DoP — it needs an integer \
                 container of at least 24 bits (24i, 24i32 or 32i)",
                CANDIDATES[i].4,
            )),
        };
    }
    if let Some(i) = forced {
        return Ok(vec![i]);
    }
    Ok(match src_bits {
        Some(b) if b <= 16 => vec![0, 2, 3, 1, 4],
        Some(b) if b > 24 => vec![3, 4, 2, 1, 0],
        _ => vec![2, 1, 3, 4, 0], // 24-bit or unknown
    })
}

/// Parse one `MOOSIK_BP_FORMAT` value.
///
/// `MOOSIK_BP_FORMAT=16i|24i|24i32|32i|32f` pins the device format to one
/// candidate: bisecting a format problem otherwise means a rebuild per guess,
/// and the machine that has the problem is usually not the machine with the
/// compiler. Split out from the environment read so the rejection path can be
/// tested without mutating a process-global.
fn parse_forced(value: &str) -> Result<usize, String> {
    match value.trim() {
        "16i" => Ok(0),
        "24i" => Ok(1),
        "24i32" => Ok(2),
        "32i" => Ok(3),
        "32f" => Ok(4),
        // Ignoring a typo meant an operator who pinned a format quietly got a
        // different one. A misspelled override is a mistake to report, not a
        // preference to discard.
        other => Err(format!(
            "MOOSIK_BP_FORMAT=\"{other}\" is not a device format \
             (expected 16i, 24i, 24i32, 32i or 32f)"
        )),
    }
}

fn forced_candidate() -> Result<Option<usize>, String> {
    let Ok(want) = std::env::var("MOOSIK_BP_FORMAT") else { return Ok(None) };
    let i = parse_forced(&want)?;
    // "requests", not "pins": DoP policy may still reject this below, and the
    // log used to announce a format that was then never used.
    crate::mlog!("bp      MOOSIK_BP_FORMAT requests {}", CANDIDATES[i].4);
    Ok(Some(i))
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
    order: Vec<usize>,
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
        .spawn(move || render_thread(dev_name, sample_rate, channels, order, dop, shared, tap, stop_t, tx))
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
    order: Vec<usize>,
    dop: bool,
    shared: Arc<Shared>,
    mut tap: SpectrumTap,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<(String, String), String>>,
) {
    let _ = initialize_mta();

    let setup = setup_stream(device_name.as_deref(), sample_rate, channels, order, dop);
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

    // The loop body below does no logging: no `mlog!`, no formatting for a
    // log, no logging mutex, no logging I/O.
    //
    // This thread holds MMCSS "Pro Audio" and has a hard deadline; `mlog!`
    // formats into a String, takes a global mutex, and writes *and flushes* to
    // disk. 1.4.1 called it from inside this loop, including on every change
    // to the dropout count -- which is to say, precisely when the thread was
    // already late. A diagnostic that lengthens the stall it is measuring is
    // worse than no diagnostic. Everything interesting is accumulated in plain
    // `Copy` locals and reported after the loop, once the priority claim has
    // been dropped.
    //
    // This is the WASAPI half only. The ASIO driver callbacks and the CPAL
    // error callback still log synchronously; those need a bounded lock-free
    // breadcrumb transport, which is a design task rather than a hotfix.
    let trace = std::env::var("MOOSIK_BP_TRACE").is_ok_and(|v| v != "0");
    let mut writes: u64 = 0;
    let mut short_writes: u64 = 0;
    let mut min_avail = usize::MAX;
    let mut max_avail = 0usize;
    let mut peak = 0.0f32;
    // Kept as a value, not a message: moving the error costs nothing, while
    // formatting it here would allocate on the thread we are protecting.
    let mut space_err: Option<WasapiError> = None;
    let mut write_err: Option<WasapiError> = None;

    loop {
        if stop.load(Ordering::Relaxed) { break; }
        let avail = match client.get_available_space_in_frames() {
            Ok(a) => a as usize,
            Err(e) => {
                space_err = Some(e);
                break;
            }
        };
        if avail > 0 {
            let want = avail * channels as usize;
            render_samples(&shared, &mut scratch, &mut tap, channels, want);
            let got = scratch.len();
            fill_bytes(&mut bytes, &scratch, want, dev_fmt);
            if trace {
                writes += 1;
                if got < want { short_writes += 1; }
                min_avail = min_avail.min(avail);
                max_avail = max_avail.max(avail);
                for &s in scratch.iter() { peak = peak.max(s.abs()); }
            }
            if let Err(e) = render.write_to_device(avail, &bytes, None) {
                write_err = Some(e);
                break;
            }
        }
        std::thread::sleep(sleep);
    }

    // Order matters here. The deadline is released first, then the stream is
    // stopped and every device handle dropped, and only then does anything
    // format a string or touch the filesystem. Logging while the exclusive
    // stream was still started meant a slow or contended logger could keep the
    // DAC held open while `Handle::drop` waited on this thread to join.
    drop(_prio);
    let stop_err = client.stop_stream().err();
    drop(render);
    drop(client);

    if let Some(e) = space_err {
        crate::mlog!("[bit-perfect] wasapi error: {e}");
    }
    if let Some(e) = write_err {
        crate::mlog!("[bit-perfect] wasapi write error: {e}");
    }
    if trace {
        crate::mlog!(
            "bp      trace: {writes} writes, {short_writes} short,              avail {}..{} frames, peak {peak:.4}",
            if min_avail == usize::MAX { 0 } else { min_avail },
            max_avail,
        );
    }
    if let Some(e) = stop_err {
        crate::mlog!("[bit-perfect] wasapi stop error: {e}");
    }
    crate::mlog!(
        "bp      stream closed and device released, {} dropout(s)",
        shared.underruns.load(Ordering::Relaxed)
    );
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
    order: Vec<usize>,
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
        "bp      negotiating \"{dev_label}\": {sample_rate} Hz, {channels}ch, dop {dop},          order {:?}",
        order.iter().map(|&i| CANDIDATES[i].4).collect::<Vec<_>>(),
    );

    // Negotiate the device format in exclusive mode. The order arrived already
    // validated -- see `resolve_order`, which runs before this thread exists.
    let mut chosen: Option<(WaveFormat, DevFmt, &str)> = None;
    for idx in order {
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

#[cfg(test)]
mod tests {
    use super::*;

    const I16: usize = 0;
    const I24: usize = 1;
    const I24_32: usize = 2;
    const I32: usize = 3;
    const F32: usize = 4;

    /// The policy is exercised directly rather than through the environment:
    /// `MOOSIK_BP_FORMAT` is process-global, and tests run in parallel.
    #[test]
    fn a_forced_format_can_never_break_dop() {
        // Without an override, DoP keeps its verified order: packed 24 first,
        // then the two wider integer containers.
        assert_eq!(candidate_policy(Some(24), true, None).unwrap(), vec![I24, I24_32, I32]);

        // The three containers that can carry a DoP word are honoured.
        for &ok in &[I24, I24_32, I32] {
            assert_eq!(candidate_policy(Some(24), true, Some(ok)).unwrap(), vec![ok],
                       "{} should be forceable for DoP", CANDIDATES[ok].4);
        }

        // The two that cannot are refused before any device is opened, and are
        // *not* quietly replaced with the default order — an operator who
        // pinned a format has to be told it was impossible.
        for &bad in &[I16, F32] {
            let err = candidate_policy(Some(24), true, Some(bad))
                .err()
                .unwrap_or_else(|| panic!("{} must not be usable for DoP", CANDIDATES[bad].4));
            assert!(err.contains(CANDIDATES[bad].4), "{err}");
        }
    }

    /// The override still applies unchanged on the PCM path, and the PCM
    /// ordering itself is untouched by the DoP fix.
    #[test]
    fn pcm_ordering_is_unchanged() {
        assert_eq!(candidate_policy(Some(16), false, None).unwrap(), vec![I16, I24_32, I32, I24, F32]);
        assert_eq!(candidate_policy(Some(24), false, None).unwrap(), vec![I24_32, I24, I32, F32, I16]);
        assert_eq!(candidate_policy(None,     false, None).unwrap(), vec![I24_32, I24, I32, F32, I16]);
        assert_eq!(candidate_policy(Some(32), false, None).unwrap(), vec![I32, F32, I24_32, I24, I16]);
        for i in 0..CANDIDATES.len() {
            assert_eq!(candidate_policy(Some(24), false, Some(i)).unwrap(), vec![i]);
        }
    }

    /// A value nobody recognises is a mistake to report. It used to be logged
    /// and discarded, which handed the operator a different format than the one
    /// they pinned and no indication that it had happened.
    #[test]
    fn an_unrecognised_override_is_an_error_not_a_shrug() {
        for good in ["16i", "24i", "24i32", "32i", "32f", "  24i32  "] {
            assert!(parse_forced(good).is_ok(), "{good} should parse");
        }
        for bad in ["", "24", "24bit", "I24", "32", "float", "24i-32"] {
            let e = parse_forced(bad).unwrap_err();
            assert!(e.contains("is not a device format"), "{bad}: {e}");
        }
    }

    /// The two containers DoP is allowed to use must put the same 24-bit word
    /// on the wire: packed into three bytes, or left-aligned in four. A DAC
    /// recognises DoP by the `0x05`/`0xFA` marker in the top byte, so where
    /// that byte lands is the whole contract.
    #[test]
    fn both_dop_containers_carry_the_same_word() {
        // 0x05ABCD — a DoP word whose marker byte is 0x05.
        let word: i32 = 0x05_AB_CD;
        let v = word as f32 / 8_388_608.0;

        let mut bytes = Vec::new();
        fill_bytes(&mut bytes, &[v], 1, DevFmt::I24);
        assert_eq!(bytes, vec![0xCD, 0xAB, 0x05], "packed 24-bit, little-endian");

        fill_bytes(&mut bytes, &[v], 1, DevFmt::I32);
        assert_eq!(bytes, vec![0x00, 0xCD, 0xAB, 0x05], "24 bits left-aligned in 32");

        // The marker byte is the most significant one in both layouts.
        assert_eq!(*bytes.last().unwrap(), 0x05);
    }
}

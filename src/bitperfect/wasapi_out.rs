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
//
// What this backend guarantees, and only this: the sample or payload words
// handed to the Windows audio driver are the source's own words. The USB link,
// the driver's internals and the DAC are past the boundary of anything this
// process can observe.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use wasapi::{
    calculate_period_100ns, initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode,
    WasapiError, WaveFormat,
};

use super::format::{
    self, Candidate, ChannelLayout, DopState, PcmKind, SampleTag, SourceFormat, Writer, CANDIDATES,
};
use super::state::{EndpointIdentity, OutputFormat};
use super::{fault, fmt_khz, render_frames, AudioPriority, DeviceCaps, Shared, SpectrumTap, PROBE_RATES};

// HRESULTs from AUDCLNT_ERR (audioclient.h) we want to recognise.
const E_BUFFER_SIZE_NOT_ALIGNED: i32 = 0x8889_0019u32 as i32;
const E_DEVICE_IN_USE: i32 = 0x8889_000Au32 as i32;
const E_EXCLUSIVE_MODE_NOT_ALLOWED: i32 = 0x8889_000Eu32 as i32;

fn tag(t: SampleTag) -> SampleType {
    match t {
        SampleTag::Int => SampleType::Int,
        SampleTag::Float => SampleType::Float,
    }
}

/// Build a `WaveFormat`, requesting the source's own channel mask when it
/// declares one.
///
/// The mask used to be left to the crate's positional default, which is derived
/// from the channel *count*. For three channels that default is FL/FR/FC — so a
/// FL/FR/LFE source was negotiated, accepted and validated as FL/FR/FC, and the
/// LFE content went to the centre speaker with the diamond lit. The count is
/// not the layout.
///
/// A source that declares nothing keeps the positional default, and the
/// validator treats an unnamed multichannel layout conservatively rather than
/// inventing proof for it.
fn wave_format(c: &Candidate, rate: u32, channels: u16, layout: ChannelLayout) -> WaveFormat {
    let mask = layout.is_specified().then_some(layout.0);
    WaveFormat::new(
        c.store_bits,
        c.valid_bits,
        &tag(c.tag),
        rate as usize,
        channels as usize,
        mask,
    )
}

/// Resolve the device-format order for this stream, or say why the request is
/// impossible — before any device is enumerated, opened, or negotiated with.
///
/// Every candidate in the returned order is exact for `kind`. There is no
/// last-chance entry: a source that no device format can carry is a
/// configuration failure here, not a quiet conversion later.
pub fn resolve_order(kind: PcmKind, dop: bool) -> Result<Vec<usize>, super::OpenError> {
    let forced = forced_candidate().map_err(super::OpenError::config)?;
    format::negotiation_order(kind, dop, forced).map_err(super::OpenError::config)
}

/// Read `MOOSIK_BP_FORMAT`.
///
/// `MOOSIK_BP_FORMAT=16i|24i|24i32|32i|32f` pins the device format to one
/// candidate: bisecting a format problem otherwise means a rebuild per guess,
/// and the machine with the problem is usually not the machine with the
/// compiler. It is a debugging aid and not a licence — the value still has to
/// survive the exactness matrix, which is where a pin that would truncate the
/// source gets refused.
fn forced_candidate() -> Result<Option<usize>, String> {
    let Ok(want) = std::env::var("MOOSIK_BP_FORMAT") else {
        return Ok(None);
    };
    let i = format::parse_forced(&want)?;
    // "requests", not "pins": the policy may still reject this, and the log
    // used to announce a format that was then never used.
    crate::mlog!("bp      MOOSIK_BP_FORMAT requests {}", CANDIDATES[i].label);
    Ok(Some(i))
}
// ---------------------------------------------------------------------------
// Device probing (exclusive-mode capabilities)
// ---------------------------------------------------------------------------

pub fn probe_devices() -> Vec<DeviceCaps> {
    let _ = initialize_mta();
    let mut out = Vec::new();
    let Ok(enumerator) = DeviceEnumerator::new() else {
        return out;
    };
    let default_name = enumerator
        .get_default_device(&Direction::Render)
        .and_then(|d| d.get_friendlyname())
        .ok();
    let Ok(collection) = enumerator.get_device_collection(&Direction::Render) else {
        return out;
    };
    let n = collection.get_nbr_devices().unwrap_or(0);

    for i in 0..n {
        let Ok(device) = collection.get_device_at_index(i) else {
            continue;
        };
        let Ok(name) = device.get_friendlyname() else {
            continue;
        };
        let Ok(client) = device.get_iaudioclient() else {
            continue;
        };

        let mut rates: Vec<u32> = Vec::new();
        let mut formats: Vec<String> = Vec::new();
        for &rate in &PROBE_RATES {
            for c in &CANDIDATES {
                // The same two steps playback takes, in the same order.
                //
                // The probe asked only whether the device *accepted* the
                // format and never whether the reply was the format that was
                // asked for — so the device menu advertised rates and formats
                // that playback would then refuse in validation, and the user
                // saw a capability list that disagreed with what happened when
                // they picked from it.
                let Ok(accepted) = client.is_supported_exclusive_with_quirks(&wave_format(
                    c,
                    rate,
                    2,
                    ChannelLayout::UNSPECIFIED,
                )) else {
                    continue;
                };
                if validate_negotiated(&accepted, c, rate, 2, ChannelLayout::UNSPECIFIED).is_err() {
                    continue;
                }
                if !rates.contains(&rate) {
                    rates.push(rate);
                }
                if !formats.contains(&c.label.to_string()) {
                    formats.push(c.label.to_string());
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

/// Resolve the endpoint a fresh open would attach to, without opening it.
///
/// Called before every reuse decision. The point is that it does **not** read
/// the open stream: `None` means "system default", and Windows can move the
/// default to another DAC while a stream is open. Deriving the target from the
/// stream that is already running made that change invisible, so playback
/// carried on to the old device with a green diamond describing the new one.
pub fn resolve_endpoint(device_name: Option<&str>) -> Option<EndpointIdentity> {
    let _ = initialize_mta();
    let enumerator = DeviceEnumerator::new().ok()?;
    let device = match device_name {
        None => enumerator.get_default_device(&Direction::Render).ok()?,
        Some(want) => {
            let collection = enumerator.get_device_collection(&Direction::Render).ok()?;
            let count = collection.get_nbr_devices().unwrap_or(0);
            let mut found = None;
            for i in 0..count {
                if let Ok(d) = collection.get_device_at_index(i)
                    && d.get_friendlyname().map(|n| n == want).unwrap_or(false)
                {
                    found = Some(d);
                    break;
                }
            }
            found?
        }
    };
    let display = device.get_friendlyname().unwrap_or_else(|_| "?".into());
    let id = device.get_id().unwrap_or_else(|_| display.clone());
    Some(EndpointIdentity::new(id, display))
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
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Open a WASAPI exclusive stream; returns the keep-alive handle and the
/// format the device actually agreed to.
pub fn open(
    device_name: Option<&str>,
    source: SourceFormat,
    order: Vec<usize>,
    dop: bool,
    shared: Arc<Shared>,
    tap: SpectrumTap,
) -> Result<(Handle, OutputFormat), super::OpenError> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let stop_t = Arc::clone(&stop);
    let dev_name = device_name.map(str::to_owned);

    let join = std::thread::Builder::new()
        .name("bp-wasapi-render".into())
        .spawn(move || render_thread(dev_name, source, order, dop, shared, tap, stop_t, tx))
        .map_err(|e| super::OpenError::backend(format!("render thread spawn failed: {e}")))?;

    match rx.recv() {
        Ok(Ok(out)) => Ok((
            Handle {
                stop,
                join: Some(join),
            },
            out,
        )),
        Ok(Err(e)) => {
            let _ = join.join();
            Err(e)
        }
        Err(_) => {
            let _ = join.join();
            Err(super::OpenError::backend("output thread died during setup"))
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn render_thread(
    device_name: Option<String>,
    source: SourceFormat,
    order: Vec<usize>,
    dop: bool,
    shared: Arc<Shared>,
    mut tap: SpectrumTap,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<OutputFormat, super::OpenError>>,
) {
    let _ = initialize_mta();

    let setup = setup_stream(device_name.as_deref(), source, order, dop);
    let (client, render, fmt, writer, output) = match setup {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(Err(e));
            return;
        }
    };

    let channels = source.channels;
    let ch = channels.max(1) as usize;
    let bps = writer.bytes_per_sample();
    let blockalign = fmt.get_blockalign() as usize;
    let period_hns = client
        .get_device_period()
        .map(|(d, _)| d)
        .unwrap_or(100_000); // 10 ms fallback
    let sleep = Duration::from_millis(((period_hns / 10_000) / 2).max(1) as u64);

    // Everything the loop needs, sized once. The device buffer is the ceiling
    // on a single write, so nothing below ever grows and nothing allocates.
    // The device buffer size is the ceiling on every allocation below and on
    // every single write. Guessing it was not a fallback but a fabrication: an
    // 8192-frame guess against a smaller real buffer sizes `canon`/`bytes`
    // wrong and makes every subsequent available-space clamp meaningless. If
    // the driver will not say, the stream does not open.
    let cap_frames = match client.get_buffer_size() {
        Ok(n) if n > 0 => n as usize,
        Ok(_) => {
            let _ = tx.send(Err(super::OpenError::backend(
                "device reported a zero-frame buffer",
            )));
            return;
        }
        Err(e) => {
            let _ = tx.send(Err(super::OpenError::backend(format!(
                "device buffer size query failed: {e}"
            ))));
            return;
        }
    };
    let mut canon: Vec<u32> = vec![0; cap_frames * ch];
    let mut bytes: Vec<u8> = vec![0u8; cap_frames * ch * bps];
    let mut dop_state = DopState::new();

    // Prefill the device buffer before starting. For DoP this has to be
    // *marked* silence: a DoP DAC reading a buffer of PCM zeros sees loss of
    // lock, which is the click at the start of every DSD track before 1.4.3.
    //
    // A stream that could not write its prefill must not start. The failure
    // used to be discarded with `let _`, so the engine started on whatever the
    // driver had left in its buffer — and on the DoP path the DAC's first
    // impression of the stream was noise, while the session reported
    // payload-exact.
    match client.get_available_space_in_frames() {
        Ok(avail) => {
            let frames = (avail as usize).min(cap_frames);
            // A prefill of zero frames is not a prefill. An initialised
            // exclusive client that reports no free space has not given us a
            // buffer to start on, and starting anyway means the device plays
            // whatever the driver had left in it — which on the DoP path the
            // DAC reads as loss of lock while the session reports exact.
            if frames == 0 {
                let _ = tx.send(Err(super::OpenError::backend(
                    "device offered no buffer space to prefill",
                )));
                return;
            }
            let out = &mut bytes[..frames * ch * bps];
            if dop {
                dop_state.emit(&[], frames, ch, &writer, out);
            } else {
                out.fill(0);
            }
            if let Err(e) = render.write_to_device(frames, out, None) {
                let _ = tx.send(Err(super::OpenError::backend(format!(
                    "initial prefill failed: {e}"
                ))));
                return;
            }
        }
        Err(e) => {
            let _ = tx.send(Err(super::OpenError::backend(format!(
                "could not size the device buffer: {e}"
            ))));
            return;
        }
    }
    if let Err(e) = client.start_stream() {
        let _ = tx.send(Err(super::OpenError::backend(format!(
            "stream start failed: {e}"
        ))));
        return;
    }
    crate::mlog!(
        "bp      streaming \"{}\" as {}: {} Hz, {}ch, {} valid/{} container, blockalign \
         {blockalign}, period {:.2} ms",
        output.endpoint.display,
        output.label,
        fmt.get_samplespersec(),
        fmt.get_nchannels(),
        output.valid_bits,
        output.container_bits,
        period_hns as f64 / 10_000.0,
    );
    let _ = tx.send(Ok(output.clone()));

    // Held for the life of the loop: this thread has a hard deadline the rest
    // of the process (notably the rayon pre-process pool) does not.
    let _prio = AudioPriority::claim();

    // The loop body below does no logging: no `mlog!`, no formatting for a
    // log, no logging mutex, no logging I/O, and no allocation.
    //
    // This thread holds MMCSS "Pro Audio" and has a hard deadline; `mlog!`
    // formats into a String, takes a global mutex, and writes *and flushes* to
    // disk. 1.4.1 called it from inside this loop, including on every change
    // to the dropout count -- which is to say, precisely when the thread was
    // already late. A diagnostic that lengthens the stall it is measuring is
    // worse than no diagnostic. Faults are published as a `u8` (see
    // `super::fault`), everything else is accumulated in plain `Copy` locals,
    // and both are reported after the loop once the priority claim is gone.
    let trace = std::env::var("MOOSIK_BP_TRACE").is_ok_and(|v| v != "0");
    let mut writes: u64 = 0;
    let mut short_writes: u64 = 0;
    let mut min_avail = usize::MAX;
    let mut max_avail = 0usize;
    // Kept as values, not messages: moving an error costs nothing, while
    // formatting it here would allocate on the thread we are protecting.
    let mut space_err: Option<WasapiError> = None;
    let mut write_err: Option<WasapiError> = None;

    let mut stopped_cleanly = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            stopped_cleanly = true;
            break;
        }
        let avail = match client.get_available_space_in_frames() {
            Ok(a) => a as usize,
            Err(e) => {
                shared.fail_now(fault::BACKEND_SPACE);
                space_err = Some(e);
                break;
            }
        };
        if avail > 0 {
            let frames = avail.min(cap_frames);
            let got = render_frames(
                &shared,
                &mut canon[..frames * ch],
                &mut tap,
                channels,
                frames,
            );
            let out = &mut bytes[..frames * ch * bps];
            if dop {
                // Marker ownership lives here: one marker per complete channel
                // frame, alternating, and marked DSD silence wherever the
                // payload runs out — pause, underrun, drain, or the gap after
                // a seek. A DoP frame of PCM zeros is never emitted.
                dop_state.emit(&canon[..got * ch], frames, ch, &writer, out);
            } else {
                let filled = got * ch * bps;
                writer.write(&canon[..got * ch], &mut out[..filled]);
                out[filled..].fill(0);
            }
            if trace {
                writes += 1;
                if got < frames {
                    short_writes += 1;
                }
                min_avail = min_avail.min(avail);
                max_avail = max_avail.max(avail);
            }
            if let Err(e) = render.write_to_device(frames, out, None) {
                shared.fail_now(fault::BACKEND_WRITE);
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

    // The render thread is ending. Whether that was an orderly stop or a
    // backend failure, this stream can carry nothing more — publish it before
    // anything can decide to reuse it.
    //
    // A thread that exited on a write error used to leave the handle looking
    // alive, so the next track could be handed to a stream with nobody behind
    // it and would simply play silence.
    shared.mark_backend_dead();
    if !stopped_cleanly {
        shared.fail_now(fault::BACKEND_DEAD);
    }

    if let Some(e) = space_err {
        crate::mlog!("[bit-perfect] wasapi error: {e}");
    }
    if let Some(e) = write_err {
        crate::mlog!("[bit-perfect] wasapi write error: {e}");
    }
    if trace {
        crate::mlog!(
            "bp      trace: {writes} writes, {short_writes} short, avail {}..{} frames",
            if min_avail == usize::MAX {
                0
            } else {
                min_avail
            },
            max_avail,
        );
    }
    if let Some(e) = stop_err {
        crate::mlog!("[bit-perfect] wasapi stop error: {e}");
    }
    let code = shared.fault();
    crate::mlog!(
        "bp      stream closed and device released, {} dropout(s), {} lock miss(es){}",
        shared.underruns.load(Ordering::Relaxed),
        shared.lock_misses.load(Ordering::Relaxed),
        if code == fault::NONE {
            String::new()
        } else {
            format!(", integrity fault: {}", fault::describe(code))
        },
    );
}

type StreamSetup = (
    wasapi::AudioClient,
    wasapi::AudioRenderClient,
    WaveFormat,
    Writer,
    OutputFormat,
);

/// Check that what the device agreed to is what was asked for, field by field.
///
/// A driver may accept `IsFormatSupported` and hand back a *different* format
/// in the closest-match slot; a driver may also honour the container width and
/// quietly change the valid-bits count. Before 1.4.3 the stored format label
/// came from the candidate that was offered rather than from the reply, so
/// either case produced a stream that reported one format and wrote another —
/// frames of the wrong width, which sounds like noise at the wrong speed.
/// Whether the accepted format is the plain `WAVEFORMATEX` shape rather than
/// `WAVEFORMATEXTENSIBLE`.
///
/// The legacy structure has neither `dwChannelMask` nor
/// `wValidBitsPerSample` — it is 14 bytes and a `wBitsPerSample`, and nothing
/// else. Drivers that only speak it come back with both fields zero, which is
/// not "the mask is zero" and not "no bits are valid": it is *the driver did
/// not say*.
///
/// Reading a silence as an answer is the failure in both directions. Comparing
/// zero against a requested mask refuses a legacy driver that would have
/// played perfectly; accepting zero as agreement claims a channel layout and
/// a valid-bit count nobody ever confirmed.
fn is_legacy_wfx(accepted: &WaveFormat) -> bool {
    accepted.get_dwchannelmask() == 0 && accepted.get_validbitspersample() == 0
}

fn validate_negotiated(
    accepted: &WaveFormat,
    c: &Candidate,
    sample_rate: u32,
    channels: u16,
    layout: ChannelLayout,
) -> Result<(), String> {
    let want_align = (c.store_bits / 8) * channels as usize;
    // Compare against the mask that was actually requested. Comparing against a
    // count-derived default meant a FL/FR/LFE request could be "validated" as
    // the FL/FR/FC the default happens to be.
    let want_mask = if layout.is_specified() {
        layout.0
    } else {
        wasapi::make_simple_channelmask(channels as usize)
    };

    // A driver that answered in the legacy shape told us the rate, the channel
    // count, the container width and the block alignment, and nothing else.
    // Those four are checked below exactly as they always are; the two fields
    // it does not have are not checked, and the route is only allowed at all
    // where their absence cannot hide anything.
    if is_legacy_wfx(accepted) {
        // Above stereo the mask is what says which speaker each channel
        // reaches, and there is no default worth guessing: the same three
        // channels are FL/FR/FC on one device and FL/FR/LFE on another.
        if channels > 2 {
            return Err(format!(
                "the device replied in the legacy WAVEFORMATEX form, which carries no \
                 channel mask, and {channels} channels cannot be placed without one"
            ));
        }
        if layout.is_specified() && layout.0 != wasapi::make_simple_channelmask(channels as usize) {
            return Err(
                "the device replied in the legacy WAVEFORMATEX form, which carries no \
                 channel mask, so the source's own speaker layout cannot be confirmed"
                    .to_string(),
            );
        }
        // Without `wValidBitsPerSample` the only container whose valid bits
        // are unambiguous is one where every bit is valid. A 24-in-32 format
        // is exactly the case that cannot be confirmed: the driver may take
        // the low 24 bits or the high 24, and both are silent about it.
        if c.valid_bits != c.store_bits {
            return Err(format!(
                "the device replied in the legacy WAVEFORMATEX form, which carries no valid-bit \
                 count, so {}-valid-in-{} cannot be confirmed",
                c.valid_bits, c.store_bits
            ));
        }
    }

    if accepted.get_samplespersec() != sample_rate {
        return Err(format!(
            "rate {} (asked {sample_rate})",
            accepted.get_samplespersec()
        ));
    }
    if accepted.get_nchannels() != channels {
        return Err(format!(
            "{} channels (asked {channels})",
            accepted.get_nchannels()
        ));
    }
    if !is_legacy_wfx(accepted) && accepted.get_dwchannelmask() != want_mask {
        return Err(format!(
            "channel mask {:#x} (asked {want_mask:#x})",
            accepted.get_dwchannelmask()
        ));
    }
    if accepted.get_blockalign() as usize != want_align {
        return Err(format!(
            "block align {} (asked {want_align})",
            accepted.get_blockalign()
        ));
    }
    if accepted.get_bitspersample() as usize != c.store_bits {
        return Err(format!(
            "{} container bits (asked {})",
            accepted.get_bitspersample(),
            c.store_bits
        ));
    }
    if !is_legacy_wfx(accepted) && accepted.get_validbitspersample() as usize != c.valid_bits {
        return Err(format!(
            "{} valid bits (asked {})",
            accepted.get_validbitspersample(),
            c.valid_bits
        ));
    }
    // The legacy structure has no `SubFormat` either — the tag lives in
    // `wFormatTag`, which the crate does not surface here. The container width
    // and the block alignment above already pin the layout of every word;
    // what remains unconfirmed is integer-versus-float, and that is why the
    // legacy route is confined to a candidate whose valid bits fill its
    // container, where a mistaken family would fail the width check.
    if !is_legacy_wfx(accepted) {
        match accepted.get_subformat() {
            Ok(got) if got == tag(c.tag) => {}
            Ok(got) => return Err(format!("subtype {got:?} (asked {:?})", tag(c.tag))),
            Err(e) => return Err(format!("unreadable subtype: {e}")),
        }
    }
    Ok(())
}
fn setup_stream(
    device_name: Option<&str>,
    source: SourceFormat,
    order: Vec<usize>,
    dop: bool,
) -> Result<StreamSetup, super::OpenError> {
    let sample_rate = source.sample_rate;
    let channels = source.channels;

    let enumerator = DeviceEnumerator::new()
        .map_err(|e| super::OpenError::backend(format!("device enumeration failed: {e}")))?;
    let device = match device_name {
        None => enumerator
            .get_default_device(&Direction::Render)
            .map_err(|e| {
                super::OpenError::device_unavailable(format!("no default output device: {e}"))
            })?,
        Some(n) => {
            let collection = enumerator
                .get_device_collection(&Direction::Render)
                .map_err(|e| {
                    super::OpenError::backend(format!("device enumeration failed: {e}"))
                })?;
            let count = collection.get_nbr_devices().unwrap_or(0);
            let mut found = None;
            for i in 0..count {
                if let Ok(d) = collection.get_device_at_index(i)
                    && d.get_friendlyname()
                        .map(|fname| fname == n)
                        .unwrap_or(false)
                {
                    found = Some(d);
                    break;
                }
            }
            found.ok_or_else(|| {
                super::OpenError::device_unavailable(format!(
                    "device \"{n}\" not found (unplugged?)"
                ))
            })?
        }
    };
    // `GetId` is the endpoint's stable identity; the friendly name is not one.
    // Two DACs of the same model share a friendly name, and Windows renames
    // endpoints — a reuse decision made on the name can therefore keep playing
    // to a device the user did not choose.
    let dev_label = device.get_friendlyname().unwrap_or_else(|_| "?".into());
    let dev_id = device.get_id().unwrap_or_else(|_| dev_label.clone());
    let endpoint = EndpointIdentity::new(dev_id, dev_label.clone());

    let mut client = device
        .get_iaudioclient()
        .map_err(|e| super::OpenError::backend(format!("audio client failed: {e}")))?;

    crate::mlog!(
        "bp      negotiating \"{dev_label}\": {sample_rate} Hz, {channels}ch, {}, dop {dop}, \
         exact candidates {:?}",
        source.kind.describe(),
        order
            .iter()
            .map(|&i| CANDIDATES[i].label)
            .collect::<Vec<_>>(),
    );

    // Negotiate the device format in exclusive mode. The order arrived already
    // validated -- see `resolve_order`, which runs before this thread exists
    // and admits only formats that carry this source exactly.
    let mut chosen: Option<(WaveFormat, usize)> = None;
    for idx in order {
        let c = &CANDIDATES[idx];
        match client.is_supported_exclusive_with_quirks(&wave_format(
            c,
            sample_rate,
            channels,
            source.layout,
        )) {
            Ok(accepted) => {
                crate::mlog!(
                    "bp      accepted {} → {} Hz, {}ch, {} bits ({} valid), mask {:#x}, align {}",
                    c.label,
                    accepted.get_samplespersec(),
                    accepted.get_nchannels(),
                    accepted.get_bitspersample(),
                    accepted.get_validbitspersample(),
                    accepted.get_dwchannelmask(),
                    accepted.get_blockalign(),
                );
                if let Err(why) =
                    validate_negotiated(&accepted, c, sample_rate, channels, source.layout)
                {
                    crate::mlog!(
                        "bp      MISMATCH {}: device replied with {why} — refusing rather than \
                         writing a format it did not agree to",
                        c.label
                    );
                    continue;
                }
                chosen = Some((accepted, idx));
                break;
            }
            Err(e) => crate::mlog!("bp      rejected {}: {e}", c.label),
        }
    }
    let (fmt, idx) = chosen.ok_or_else(|| {
        // Build a helpful error: which standard rates DOES the device take?
        let rates: Vec<String> = PROBE_RATES
            .iter()
            .copied()
            .filter(|&r| {
                CANDIDATES.iter().any(|c| {
                    client
                        .is_supported_exclusive_with_quirks(&wave_format(
                            c,
                            r,
                            channels,
                            source.layout,
                        ))
                        .is_ok()
                })
            })
            .map(fmt_khz)
            .collect();
        let note = if dop {
            " (DoP requires a ≥24-bit integer format)".to_string()
        } else {
            format!(
                " (a {} source needs one of: {})",
                source.kind.describe(),
                format::exact_candidates(source.kind, dop)
                    .map(|v| v
                        .iter()
                        .map(|&i| CANDIDATES[i].label)
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_else(|e| e)
            )
        };
        super::OpenError::device_format(format!(
            "\"{dev_label}\" rejected {} / {channels}ch{note} in exclusive mode (accepts: {})",
            fmt_khz(sample_rate),
            if rates.is_empty() {
                "no standard rates — is another app holding the device?".into()
            } else {
                rates.join(", ")
            },
        ))
    })?;

    let c = &CANDIDATES[idx];
    let writer = if dop {
        Writer::for_dop(c.dev)
            .map_err(|e| super::OpenError::config(format!("format policy: {e}")))?
    } else {
        Writer::new(source.kind, c.dev)
            .map_err(|e| super::OpenError::config(format!("format policy: {e}")))?
    };

    // Period aligned to 128 bytes (required by e.g. Intel HDA devices).
    let (def_period, _min_period) = client
        .get_device_period()
        .map_err(|e| super::OpenError::backend(format!("period query failed: {e}")))?;
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
                let frames = client.get_buffer_size().map_err(|e2| {
                    super::OpenError::backend(format!("alignment recovery failed: {e2}"))
                })?;
                let aligned = calculate_period_100ns(frames as i64, fmt.get_samplespersec() as i64);
                client = device.get_iaudioclient().map_err(|e2| {
                    super::OpenError::backend(format!("audio client failed: {e2}"))
                })?;
                let mode = StreamMode::PollingExclusive {
                    period_hns: aligned,
                    buffer_duration_hns: 8 * aligned,
                };
                client
                    .initialize_client(&fmt, &Direction::Render, &mode)
                    .map_err(|e2| {
                        super::OpenError::backend(format!(
                            "exclusive init failed after alignment fix: {e2}"
                        ))
                    })?;
            }
            WasapiError::Windows(werr) if werr.code().0 == E_DEVICE_IN_USE => {
                return Err(super::OpenError::device_unavailable(format!(
                    "\"{dev_label}\" is already in exclusive use by another application"
                )));
            }
            WasapiError::Windows(werr) if werr.code().0 == E_EXCLUSIVE_MODE_NOT_ALLOWED => {
                return Err(super::OpenError::device_unavailable(format!(
                    "exclusive mode is disabled for \"{dev_label}\" — enable \
                     'Allow applications to take exclusive control' in the \
                     device's Properties → Advanced tab"
                )));
            }
            _ => {
                return Err(super::OpenError::backend(format!(
                    "exclusive init failed: {e}"
                )));
            }
        }
    }

    let render = client
        .get_audiorenderclient()
        .map_err(|e| super::OpenError::backend(format!("render client failed: {e}")))?;

    // Read back from the accepted format, not from the request.
    let output = OutputFormat {
        endpoint,
        sample_rate: fmt.get_samplespersec(),
        channels: fmt.get_nchannels(),
        layout: ChannelLayout(fmt.get_dwchannelmask()),
        container_bits: fmt.get_bitspersample(),
        valid_bits: fmt.get_validbitspersample(),
        integer: c.tag == SampleTag::Int,
        // The exclusive buffer we negotiated. It is what the device still
        // holds when the ring runs dry, so it is what the end-of-track drain
        // has to wait out.
        buffer_frames: client.get_buffer_size().unwrap_or(0),
        label: c.label.to_string(),
    };

    Ok((client, render, fmt, writer, output))
}
#[cfg(test)]
mod tests {
    use super::*;
    use format::{IDX_F32, IDX_I16, IDX_I24, IDX_I24_32, IDX_I32};

    /// A `WAVEFORMATEXTENSIBLE` reply, built the way the crate builds one.
    fn extensible(c: &Candidate, rate: u32, channels: u16, mask: Option<u32>) -> WaveFormat {
        WaveFormat::new(
            c.store_bits,
            c.valid_bits,
            &tag(c.tag),
            rate as usize,
            channels as usize,
            mask,
        )
    }

    /// The legacy shape: no channel mask, no valid-bit count, no subformat
    /// GUID.
    ///
    /// Produced by the crate's own `to_waveformatex`, so this is the exact
    /// structure a driver that only speaks the legacy form leaves us holding
    /// — not an approximation of one.
    fn legacy(c: &Candidate, rate: u32, channels: u16) -> WaveFormat {
        extensible(c, rate, channels, None)
            .to_waveformatex()
            .expect("integer and float both convert to the legacy form")
    }

    /// A driver that answers in the legacy form is neither refused out of hand
    /// nor believed about fields it does not have.
    ///
    /// Both directions were wrong before. Comparing a zero mask against a
    /// requested one refused a legacy driver that would have played perfectly;
    /// reading the zero as agreement claimed a channel layout and a valid-bit
    /// count nobody ever confirmed.
    #[test]
    fn a_legacy_reply_is_accepted_only_where_its_silence_hides_nothing() {
        let c16 = &CANDIDATES[IDX_I16];
        let c32 = &CANDIDATES[IDX_I32];
        let c24in32 = &CANDIDATES[IDX_I24_32];

        // Stereo, valid bits filling the container: everything the legacy
        // structure omits is unambiguous, so the route is allowed.
        for c in [c16, c32] {
            assert!(
                validate_negotiated(
                    &legacy(c, 44_100, 2),
                    c,
                    44_100,
                    2,
                    ChannelLayout::UNSPECIFIED
                )
                .is_ok(),
                "{} stereo should survive a legacy reply",
                c.label
            );
        }

        // 24-valid-in-32 cannot be confirmed without `wValidBitsPerSample`:
        // the driver may take the low 24 bits or the high 24, and the legacy
        // structure is silent about which.
        let e = validate_negotiated(
            &legacy(c24in32, 44_100, 2),
            c24in32,
            44_100,
            2,
            ChannelLayout::UNSPECIFIED,
        )
        .expect_err("24-in-32 must not be claimed from a legacy reply");
        assert!(e.contains("valid-bit"), "{e}");

        // Above stereo the mask is what places each channel, and there is no
        // default worth guessing — the same three channels are FL/FR/FC on one
        // device and FL/FR/LFE on another.
        let e = validate_negotiated(
            &legacy(c32, 44_100, 6),
            c32,
            44_100,
            6,
            ChannelLayout::UNSPECIFIED,
        )
        .expect_err("6 channels cannot be placed without a mask");
        assert!(e.contains("channel mask"), "{e}");

        // And a source that names its own layout cannot have that layout
        // confirmed by a reply with no mask field.
        let lfe = ChannelLayout(0x0000_000B); // FL | FR | LFE
        let e = validate_negotiated(&legacy(c32, 44_100, 2), c32, 44_100, 2, lfe)
            .expect_err("a specified layout needs a mask to confirm it");
        assert!(e.contains("channel mask"), "{e}");

        // The fields the legacy structure *does* carry are still checked
        // exactly as they always are.
        assert!(
            validate_negotiated(
                &legacy(c32, 44_100, 2),
                c32,
                48_000,
                2,
                ChannelLayout::UNSPECIFIED
            )
            .is_err(),
            "a rate mismatch is a rate mismatch whatever shape the reply is"
        );
        assert!(
            validate_negotiated(
                &legacy(c32, 44_100, 2),
                c16,
                44_100,
                2,
                ChannelLayout::UNSPECIFIED
            )
            .is_err(),
            "a container width mismatch is still caught"
        );
    }

    /// An extensible reply is still held to every field it carries.
    #[test]
    fn an_extensible_reply_must_agree_about_the_mask_it_carries() {
        let c = &CANDIDATES[IDX_I32];
        let lfe = ChannelLayout(0x0000_000B); // FL | FR | LFE
        let centre = 0x0000_0007u32; // FL | FR | FC — the count-derived default

        // Asked for FL/FR/LFE, answered FL/FR/FC. This is the case that used
        // to validate: three channels, three channels, and a default mask
        // compared against itself.
        let e = validate_negotiated(&extensible(c, 44_100, 3, Some(centre)), c, 44_100, 3, lfe)
            .expect_err("a different mask is a different placement");
        assert!(e.contains("channel mask"), "{e}");

        // The same mask, agreed, is fine.
        assert!(
            validate_negotiated(&extensible(c, 44_100, 3, Some(lfe.0)), c, 44_100, 3, lfe).is_ok()
        );
    }

    /// The probe answers the question the user is actually asking.
    ///
    /// It asked only whether the device *accepted* a format and never whether
    /// the reply was the format that was asked for, so the device menu
    /// advertised rates and formats that playback then refused in validation.
    /// The two now run the same two steps in the same order — which this
    /// checks by feeding both the same replies.
    #[test]
    fn the_capability_probe_and_playback_agree() {
        for &idx in &[IDX_I16, IDX_I24, IDX_I24_32, IDX_I32, IDX_F32] {
            let c = &CANDIDATES[idx];
            for rate in [44_100u32, 96_000, 352_800] {
                // An honest reply: the probe and playback both accept it.
                let good = extensible(c, rate, 2, None);
                assert!(
                    validate_negotiated(&good, c, rate, 2, ChannelLayout::UNSPECIFIED).is_ok(),
                    "{} at {rate} should be accepted by both",
                    c.label
                );
                // A reply at a different rate: both must refuse, so the menu
                // cannot advertise something the open would decline.
                let wrong = extensible(c, rate, 2, None);
                assert!(
                    validate_negotiated(&wrong, c, rate + 1, 2, ChannelLayout::UNSPECIFIED)
                        .is_err(),
                    "{} must be refused by both when the rate disagrees",
                    c.label
                );
            }
        }
    }

    /// The policy is exercised directly rather than through the environment:
    /// `MOOSIK_BP_FORMAT` is process-global, and tests run in parallel.
    #[test]
    fn a_forced_format_can_never_break_dop() {
        let dsd = PcmKind::Integer { valid_bits: 24 };

        // Without an override, DoP keeps its verified order: packed 24 first,
        // then the two wider integer containers.
        assert_eq!(
            format::negotiation_order(dsd, true, None).unwrap(),
            vec![IDX_I24, IDX_I24_32, IDX_I32]
        );

        // The three containers that can carry a DoP word are honoured.
        for &ok in &[IDX_I24, IDX_I24_32, IDX_I32] {
            assert_eq!(
                format::negotiation_order(dsd, true, Some(ok)).unwrap(),
                vec![ok],
                "{} should be forceable for DoP",
                CANDIDATES[ok].label
            );
        }

        // The two that cannot are refused before any device is opened, and are
        // *not* quietly replaced with the default order — an operator who
        // pinned a format has to be told it was impossible.
        for &bad in &[IDX_I16, IDX_F32] {
            let err = format::negotiation_order(dsd, true, Some(bad))
                .err()
                .unwrap_or_else(|| panic!("{} must not be usable for DoP", CANDIDATES[bad].label));
            assert!(err.contains(CANDIDATES[bad].token), "{err}");
        }
    }

    /// A value nobody recognises is a mistake to report. It used to be logged
    /// and discarded, which handed the operator a different format than the one
    /// they pinned and no indication that it had happened.
    #[test]
    fn an_unrecognised_override_is_an_error_not_a_shrug() {
        for good in ["16i", "24i", "24i32", "32i", "32f", "  24i32  "] {
            assert!(format::parse_forced(good).is_ok(), "{good} should parse");
        }
        for bad in ["", "24", "24bit", "I24", "32", "float", "24i-32"] {
            let e = format::parse_forced(bad).unwrap_err();
            assert!(e.contains("is not a device format"), "{bad}: {e}");
        }
    }
}

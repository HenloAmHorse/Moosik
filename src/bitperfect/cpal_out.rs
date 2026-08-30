// cpal backend for the bit-perfect path (Linux/macOS).
//
// Opens a stream at exactly the requested rate/channels on the chosen device,
// in a format that carries the source exactly. On Linux, picking a direct ALSA
// `hw:` device in the picker avoids the PipeWire/Pulse mixing shims.
//
// What this backend does **not** do is claim exactness. cpal asks for a format;
// what happens after that belongs to ALSA, PipeWire, Pulse, `dmix` or
// CoreAudio, and none of them report back through an interface this process can
// read. A request is not a measurement, so `ActiveRoute::CpalDirect` reports
// `Integrity::Unknown` and the status line says so. That is a narrower claim
// than 1.4.2 made and a truthful one.

use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::format::{self, ChannelLayout, DevFmt, DopState, PcmKind, SourceFormat, Writer};
use super::state::{EndpointIdentity, OutputFormat};
use super::{fault, fmt_khz, render_frames, DeviceCaps, Shared, SpectrumTap, PROBE_RATES};

/// The largest callback, in frames, this backend will fill without allocating.
///
/// 16 384 frames is roughly 340 ms at 48 kHz — far beyond any sane callback
/// size, and small enough that reserving it costs 64 KB per channel.
const CALLBACK_CEILING_FRAMES: usize = 16_384;

/// Keeps the cpal stream alive; dropping it stops playback.
pub struct Handle(#[allow(dead_code)] cpal::Stream);

fn fmt_name(f: cpal::SampleFormat) -> &'static str {
    match f {
        cpal::SampleFormat::F32 => "32f",
        cpal::SampleFormat::F64 => "64f",
        cpal::SampleFormat::I32 => "32i",
        cpal::SampleFormat::I16 => "16i",
        cpal::SampleFormat::U16 => "16u",
        _ => "?",
    }
}

pub fn probe_devices() -> Vec<DeviceCaps> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());
    let mut out = Vec::new();
    let Ok(devices) = host.output_devices() else { return out };
    for dev in devices {
        let Ok(name) = dev.name() else { continue };
        let Ok(configs) = dev.supported_output_configs() else { continue };
        let configs: Vec<_> = configs.collect();
        if configs.is_empty() { continue; }

        let mut rates: Vec<u32> = PROBE_RATES.iter().copied()
            .filter(|&r| configs.iter().any(|c|
                c.min_sample_rate().0 <= r && r <= c.max_sample_rate().0))
            .collect();
        rates.sort_unstable();

        let mut formats: Vec<String> = Vec::new();
        for c in &configs {
            let label = fmt_name(c.sample_format()).to_string();
            if !formats.contains(&label) { formats.push(label); }
        }
        let max_channels = configs.iter().map(|c| c.channels()).max().unwrap_or(0);

        out.push(DeviceCaps {
            is_default: default_name.as_deref() == Some(name.as_str()),
            name, rates, formats, max_channels,
        });
    }
    out
}

/// Resolve the endpoint a fresh open would attach to, without opening it.
///
/// cpal exposes no stable device identifier, so the name is the identity here
/// and reuse across a renamed or replaced default device is correspondingly
/// weaker. That weakness is recorded rather than hidden: the route is
/// Unverified on this platform anyway.
pub fn resolve_endpoint(device_name: Option<&str>) -> Option<EndpointIdentity> {
    find_device(device_name)
        .ok()
        .and_then(|d| d.name().ok())
        .map(EndpointIdentity::from_name)
}

fn find_device(name: Option<&str>) -> Result<cpal::Device, super::OpenError> {
    let host = cpal::default_host();
    match name {
        None => host
            .default_output_device()
            .ok_or_else(|| super::OpenError::device_unavailable("no default output device")),
        Some(n) => host
            .output_devices()
            .map_err(|e| super::OpenError::backend(format!("device enumeration failed: {e}")))?
            .find(|d| d.name().map(|dn| dn == n).unwrap_or(false))
            .ok_or_else(|| {
                super::OpenError::device_unavailable(format!(
                    "device \"{n}\" not found (unplugged?)"
                ))
            }),
    }
}
/// The cpal formats that can carry `kind` without losing a bit, best first.
///
/// cpal has no packed-24 format, so a 17-to-32-bit integer source needs `I32`.
/// A canonical payload is left-aligned, which makes `I32` exact for *every*
/// integer width — the container is simply wider than the content, and the low
/// bits are the zeros the source already had.
///
/// There is no fallback entry. 1.4.2 ranked formats by preference and took the
/// best available, so a 24-bit file on a device offering only `I16` negotiated
/// `I16` and truncated; `F64` was ranked *above* `I16`, which meant a
/// 32-bit-integer source could be rounded through a float and still be called
/// bit-perfect.
fn exact_formats(kind: PcmKind, dop: bool) -> Vec<cpal::SampleFormat> {
    use cpal::SampleFormat as S;
    if dop {
        // A DoP word is 24 integer bits with the marker in the top byte.
        return vec![S::I32];
    }
    match kind {
        PcmKind::Integer { valid_bits } if valid_bits <= 16 => vec![S::I16, S::I32],
        PcmKind::Integer { valid_bits } if valid_bits <= 32 => vec![S::I32],
        PcmKind::Float32 => vec![S::F32],
        _ => Vec::new(),
    }
}

/// Open a cpal output stream; returns the keep-alive handle and the format that
/// was requested and accepted.
pub fn open(
    device_name: Option<&str>,
    source: SourceFormat,
    dop: bool,
    shared: Arc<Shared>,
    mut tap: SpectrumTap,
) -> Result<(Handle, OutputFormat), super::OpenError> {
    let sample_rate = source.sample_rate;
    let channels = source.channels;
    let device = find_device(device_name)?;
    let dev_label = device.name().unwrap_or_else(|_| "?".into());

    let configs: Vec<_> = device
        .supported_output_configs()
        .map_err(|e| super::OpenError::backend(format!("config query failed: {e}")))?
        .collect();

    let wanted = exact_formats(source.kind, dop);
    let format = wanted
        .iter()
        .copied()
        .find(|&want| {
            configs.iter().any(|c| {
                c.channels() == channels
                    && c.min_sample_rate().0 <= sample_rate
                    && sample_rate <= c.max_sample_rate().0
                    && c.sample_format() == want
            })
        })
        .ok_or_else(|| {
            let rates: Vec<String> = PROBE_RATES
                .iter()
                .copied()
                .filter(|&r| {
                    configs
                        .iter()
                        .any(|c| c.min_sample_rate().0 <= r && r <= c.max_sample_rate().0)
                })
                .map(fmt_khz)
                .collect();
            let need = if wanted.is_empty() {
                format!(
                    "no output format can carry {} exactly",
                    source.kind.describe()
                )
            } else {
                format!(
                    "a {} source needs one of: {}",
                    source.kind.describe(),
                    wanted
                        .iter()
                        .map(|&f| fmt_name(f))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            // The endpoint answered the format query and said no. The one
            // classification that may advance a fallback ladder.
            super::OpenError::device_format(format!(
                "\"{dev_label}\" doesn\'t accept {} / {channels}ch ({need}); supports: {}",
                fmt_khz(sample_rate),
                if rates.is_empty() {
                    "none of the standard rates".into()
                } else {
                    rates.join(", ")
                }
            ))
        })?;

    let dev_fmt = match format {
        cpal::SampleFormat::I16 => DevFmt::I16,
        cpal::SampleFormat::I32 => DevFmt::I32,
        cpal::SampleFormat::F32 => DevFmt::F32,
        other => {
            return Err(super::OpenError::config(format!(
                "{} is not an exact output format",
                fmt_name(other)
            )));
        }
    };
    let writer = if dop {
        Writer::for_dop(dev_fmt)
            .map_err(|e| super::OpenError::config(format!("format policy: {e}")))?
    } else {
        Writer::new(source.kind, dev_fmt)
            .map_err(|e| super::OpenError::config(format!("format policy: {e}")))?
    };

    let config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Callback-local state, sized once. cpal does not promise a maximum buffer
    // size up front, so the scratch grows on the first oversized callback and
    // never again; steady state allocates nothing.
    let ch = channels.max(1) as usize;
    // The largest callback this stream will service without allocating.
    // Documented rather than discovered: cpal does not promise a maximum, so
    // the ceiling is stated and a violation is reported.
    let mut canon: Vec<u32> = vec![0; CALLBACK_CEILING_FRAMES * ch];
    let mut dop_state = DopState::new();
    let err_shared = Arc::clone(&shared);

    let stream = device
        .build_output_stream_raw(
            &config,
            format,
            move |data: &mut cpal::Data, _: &cpal::OutputCallbackInfo| {
                // No allocation here, ever. A callback larger than the reserved
                // ceiling is a fault and a buffer of silence, not a `resize` on
                // the audio thread — growing a `Vec` inside a realtime callback
                // is a malloc, and on the wrong day a page fault.
                let frames = (data.len() / ch).min(CALLBACK_CEILING_FRAMES);
                if data.len() / ch > CALLBACK_CEILING_FRAMES {
                    shared.fail_now(fault::BACKEND_WRITE);
                    data.bytes_mut().fill(0);
                    return;
                }
                let got = render_frames(
                    &shared,
                    &mut canon[..frames * ch],
                    &mut tap,
                    channels,
                    frames,
                );
                write_frames(
                    data,
                    &canon[..got * ch],
                    got,
                    frames,
                    ch,
                    dop,
                    &writer,
                    &mut dop_state,
                );
            },
            move |e| {
                // The only notice cpal gives that the stream has died. Raising
                // a fault alone left the handle looking alive, so the next
                // track was handed a stream with nobody behind it and played
                // silence. Backend death and end-of-audio are published too,
                // so reuse is refused and the engine stops waiting.
                crate::mlog!("bp      cpal stream error: {e}");
                err_shared.fail_now(fault::BACKEND_RESET);
                err_shared.mark_backend_dead();
            },
            None,
        )
        .map_err(|e| super::OpenError::backend(format!("stream open failed: {e}")))?;
    stream
        .play()
        .map_err(|e| super::OpenError::backend(format!("stream start failed: {e}")))?;

    let bits = match format {
        cpal::SampleFormat::I16 => 16u16,
        _ => 32,
    };
    let output = OutputFormat {
        endpoint: EndpointIdentity::from_name(dev_label),
        sample_rate,
        channels,
        layout: ChannelLayout(source.layout.0),
        container_bits: bits,
        valid_bits: bits,
        integer: format != cpal::SampleFormat::F32,
        // cpal chooses the buffer itself and does not report it, so the drain
        // falls back to its rate-derived bound.
        buffer_frames: 0,
        label: format!("{} direct", fmt_name(format)),
    };
    Ok((Handle(stream), output))
}
/// Write canonical payloads into the device buffer, padding the rest.
///
/// Typed slices rather than raw bytes, so the packing is endian-independent —
/// and for DoP, so the marker phase advances once per *frame* whatever the
/// channel count. Silence is real silence for PCM and marked DSD silence for
/// DoP: a DoP DAC handed a buffer of PCM zeros drops lock.
#[allow(clippy::too_many_arguments)]
fn write_frames(
    data: &mut cpal::Data,
    canon: &[u32],
    got_frames: usize,
    total_frames: usize,
    channels: usize,
    dop: bool,
    writer: &Writer,
    st: &mut DopState,
) {
    if dop {
        let Some(out) = data.as_slice_mut::<i32>() else {
            data.bytes_mut().fill(0);
            return;
        };
        for fr in 0..total_frames {
            let marker = st.next_marker();
            for c in 0..channels {
                let payload = if fr < got_frames {
                    canon[fr * channels + c] & 0xFFFF
                } else {
                    format::DSD_SILENCE_PAYLOAD
                };
                // 24-bit DoP word left-aligned in the 32-bit container, exactly
                // as the WASAPI 24-in-32 writer lays it out.
                out[fr * channels + c] = ((marker | payload) << 8) as i32;
            }
        }
        return;
    }

    let n = got_frames * channels;
    match writer.dev() {
        DevFmt::I16 => {
            if let Some(out) = data.as_slice_mut::<i16>() {
                for (o, &c) in out[..n].iter_mut().zip(canon) {
                    *o = ((c as i32) >> 16) as i16;
                }
                out[n..].fill(0);
            }
        }
        DevFmt::I32 | DevFmt::I24In32 => {
            if let Some(out) = data.as_slice_mut::<i32>() {
                for (o, &c) in out[..n].iter_mut().zip(canon) {
                    *o = c as i32;
                }
                out[n..].fill(0);
            }
        }
        DevFmt::F32 => {
            if let Some(out) = data.as_slice_mut::<f32>() {
                for (o, &c) in out[..n].iter_mut().zip(canon) {
                    *o = f32::from_bits(c);
                }
                out[n..].fill(0.0);
            }
        }
        // Never selected: `exact_formats` does not offer packed 24 on cpal.
        DevFmt::I24 => data.bytes_mut().fill(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::SampleFormat as S;

    /// The 1.4.2 ranking took the best *available* format, so a 24-bit source
    /// on an I16-only device negotiated I16 and truncated, and a 32-bit integer
    /// source would sooner be rounded through F64 than refused. Exactness is
    /// not a preference order.
    #[test]
    fn no_cpal_format_narrows_or_crosses_families() {
        assert_eq!(
            exact_formats(PcmKind::Integer { valid_bits: 16 }, false),
            vec![S::I16, S::I32]
        );
        assert_eq!(
            exact_formats(PcmKind::Integer { valid_bits: 24 }, false),
            vec![S::I32]
        );
        assert_eq!(
            exact_formats(PcmKind::Integer { valid_bits: 32 }, false),
            vec![S::I32]
        );
        assert_eq!(exact_formats(PcmKind::Float32, false), vec![S::F32]);

        // No float route for integers, no integer route for floats, nothing at
        // all for a source no format can carry.
        for b in 1..=32u8 {
            let got = exact_formats(PcmKind::Integer { valid_bits: b }, false);
            assert!(
                !got.contains(&S::F32) && !got.contains(&S::F64),
                "{b}-bit reached a float"
            );
            if b > 16 {
                assert!(!got.contains(&S::I16), "{b}-bit reached I16");
            }
        }
        assert!(
            !exact_formats(PcmKind::Float32, false)
                .iter()
                .any(|f| f.is_int())
        );
        assert!(exact_formats(PcmKind::Float64, false).is_empty());

        // DoP needs the 32-bit integer container whatever the source looks like.
        assert_eq!(
            exact_formats(PcmKind::Integer { valid_bits: 16 }, true),
            vec![S::I32]
        );
    }
}

// cpal backend for the bit-perfect path (Linux/macOS).
//
// Opens a stream at exactly the requested rate/channels on the chosen device.
// On Linux, picking a direct ALSA `hw:` device in the picker bypasses the
// PipeWire/Pulse resampling shims entirely.

use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::{fmt_khz, render_samples, DeviceCaps, Shared, SpectrumTap, PROBE_RATES};

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

fn find_device(name: Option<&str>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    match name {
        None => host.default_output_device()
            .ok_or_else(|| "no default output device".to_string()),
        Some(n) => host.output_devices()
            .map_err(|e| format!("device enumeration failed: {e}"))?
            .find(|d| d.name().map(|dn| dn == n).unwrap_or(false))
            .ok_or_else(|| format!("device \"{n}\" not found (unplugged?)")),
    }
}

/// Format preference: F32 is lossless for our pipeline, I32 holds 24-bit
/// exactly, F64 is wider than the source can use, I16 truncates hi-res.
fn format_rank(f: cpal::SampleFormat) -> u8 {
    match f {
        cpal::SampleFormat::F32 => 0,
        cpal::SampleFormat::I32 => 1,
        cpal::SampleFormat::F64 => 2,
        cpal::SampleFormat::I16 => 3,
        _ => u8::MAX, // never selected
    }
}

/// Open a cpal output stream; returns the keep-alive handle, the negotiated
/// format label, and the resolved device name.
pub fn open(
    device_name: Option<&str>,
    sample_rate: u32,
    channels: u16,
    _src_bits: Option<u32>,
    shared: Arc<Shared>,
    mut tap: SpectrumTap,
) -> Result<(Handle, String, String), String> {
    let device = find_device(device_name)?;
    let dev_label = device.name().unwrap_or_else(|_| "?".into());

    let configs: Vec<_> = device.supported_output_configs()
        .map_err(|e| format!("config query failed: {e}"))?
        .collect();

    let format = configs.iter()
        .filter(|c| c.channels() == channels
            && c.min_sample_rate().0 <= sample_rate
            && sample_rate <= c.max_sample_rate().0
            && format_rank(c.sample_format()) != u8::MAX)
        .min_by_key(|c| format_rank(c.sample_format()))
        .map(|c| c.sample_format())
        .ok_or_else(|| {
            let rates: Vec<String> = PROBE_RATES.iter().copied()
                .filter(|&r| configs.iter().any(|c|
                    c.min_sample_rate().0 <= r && r <= c.max_sample_rate().0))
                .map(fmt_khz)
                .collect();
            format!("\"{dev_label}\" doesn't accept {} / {channels}ch (supports: {})",
                    fmt_khz(sample_rate),
                    if rates.is_empty() { "none of the standard rates".into() }
                    else { rates.join(", ") })
        })?;

    let config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Callback-local state (no allocation on the steady-state path).
    let mut scratch: Vec<f32> = Vec::with_capacity(16_384);

    let stream = device.build_output_stream_raw(
        &config,
        format,
        move |data: &mut cpal::Data, _: &cpal::OutputCallbackInfo| {
            render_samples(&shared, &mut scratch, &mut tap, channels, data.len());
            write_data(data, &scratch);
        },
        |e| eprintln!("[bit-perfect] stream error: {e}"),
        None,
    ).map_err(|e| format!("stream open failed: {e}"))?;
    stream.play().map_err(|e| format!("stream start failed: {e}"))?;

    Ok((Handle(stream), fmt_name(format).to_string(), dev_label))
}

/// Convert f32 samples to the device's native format. Power-of-two scaling
/// keeps integer sources bit-transparent: i16 → f32 (÷2^15) → i16 (×2^15) and
/// 24-bit → f32 (÷2^23) → i32 (×2^31 = ×2^8 on the integer) are both exact.
/// Rust float→int `as` casts saturate, which handles the +full-scale edge.
fn write_data(data: &mut cpal::Data, samples: &[f32]) {
    let n = samples.len();
    match data.sample_format() {
        cpal::SampleFormat::F32 => {
            if let Some(out) = data.as_slice_mut::<f32>() {
                out[..n].copy_from_slice(samples);
                out[n..].fill(0.0);
            }
        }
        cpal::SampleFormat::F64 => {
            if let Some(out) = data.as_slice_mut::<f64>() {
                for (o, &s) in out[..n].iter_mut().zip(samples) { *o = s as f64; }
                out[n..].fill(0.0);
            }
        }
        cpal::SampleFormat::I32 => {
            if let Some(out) = data.as_slice_mut::<i32>() {
                for (o, &s) in out[..n].iter_mut().zip(samples) {
                    *o = (s as f64 * 2_147_483_648.0) as i32;
                }
                out[n..].fill(0);
            }
        }
        cpal::SampleFormat::I16 => {
            if let Some(out) = data.as_slice_mut::<i16>() {
                for (o, &s) in out[..n].iter_mut().zip(samples) {
                    *o = (s * 32_768.0) as i16;
                }
                out[n..].fill(0);
            }
        }
        _ => data.bytes_mut().fill(0),
    }
}

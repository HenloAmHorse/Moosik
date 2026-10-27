//! A whole track as the chord model hears it: decoded, mixed to one channel
//! and resampled to 22050 Hz with a windowed-sinc filter.

use rayon::prelude::*;
use rodio::Source;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// The longest track the chord model takes. The network holds the whole
/// track's activations at once (measured: 286 MB at peak for three minutes), so this
/// keeps it under about 1.5 GB; a two-hour mix would otherwise need over 10 GB.
pub const MAX_SECONDS: u32 = 15 * 60;

/// Mono samples at `super::SR`. `cancel` stops the decode, as does a track
/// longer than `MAX_SECONDS`.
pub fn load(path: &Path, cancel: &AtomicBool) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("the chord model could not open the file ({e})"))?;
    let dec = rodio::Decoder::new(std::io::BufReader::new(file)).map_err(|e| format!("the chord model could not decode the file ({e})"))?;
    let rate = dec.sample_rate();
    let ch = (dec.channels() as usize).max(1);
    let mut mono = Vec::new();
    let limit = MAX_SECONDS as usize * rate as usize;
    let (mut sum, mut got) = (0f32, 0usize);
    for s in dec {
        sum += s as f32 / 32_768.0;
        got += 1;
        if got == ch {
            mono.push(sum / ch as f32);
            (sum, got) = (0.0, 0);
            if mono.len() % (1 << 18) == 0 && cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            if mono.len() > limit {
                return Err(format!("the track is longer than {} minutes, too long for the chord model", MAX_SECONDS / 60));
            }
        }
    }
    Ok(resample(&mono, rate, super::SR))
}

/// Band-limited resampling: a Blackman-windowed sinc, 32 zero crossings each
/// side, cut off just below the lower Nyquist frequency.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let fc = ratio.min(1.0) * 0.95; // of the input Nyquist
    let half = (32.0 / fc).ceil() as isize;
    let n = (x.len() as f64 * ratio).floor() as usize;
    let tap = |d: f64| {
        let u = d / half as f64; // -1..1
        let w = 0.42 + 0.5 * (std::f64::consts::PI * u).cos() + 0.08 * (2.0 * std::f64::consts::PI * u).cos();
        let a = std::f64::consts::PI * fc * d;
        fc * w * if a.abs() < 1e-12 { 1.0 } else { a.sin() / a }
    };
    // Output i sits at input position i·m/l, so its fractional part takes
    // only l values: with few enough, the filter is tabulated once per phase.
    let g = gcd(from, to);
    let (l, m) = ((to / g) as usize, (from / g) as usize);
    let table: Option<Vec<f64>> = (l <= 4096).then(|| {
        (0..l).flat_map(|p| (-half + 1..=half).map(move |j| (p, j))).map(|(p, j)| tap(p as f64 / l as f64 - j as f64)).collect()
    });
    let width = 2 * half as usize;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let (c, p) = ((i * m / l) as isize, i * m % l);
            let mut acc = 0.0f64;
            for k in (c - half + 1).max(0)..=(c + half).min(x.len() as isize - 1) {
                let h = match &table {
                    Some(t) => t[p * width + (k - c + half - 1) as usize],
                    None => tap(i as f64 / ratio - k as f64),
                };
                acc += x[k as usize] as f64 * h;
            }
            acc as f32
        })
        .collect()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_keeps_its_pitch_and_level_through_resampling() {
        let f = 440.0;
        let x: Vec<f32> = (0..44_100).map(|i| (std::f64::consts::TAU * f * i as f64 / 44_100.0).sin() as f32).collect();
        let y = resample(&x, 44_100, 22_050);
        assert_eq!(y.len(), 22_050);
        // Away from the edges it is the same sine at the new rate.
        let err = (2000..20_000).map(|i| (y[i] as f64 - (std::f64::consts::TAU * f * i as f64 / 22_050.0).sin()).abs()).fold(0.0, f64::max);
        assert!(err < 2e-3, "{err}");
    }

    #[test]
    fn content_above_the_new_nyquist_is_removed() {
        let x: Vec<f32> = (0..48_000).map(|i| (std::f64::consts::TAU * 15_000.0 * i as f64 / 48_000.0).sin() as f32).collect();
        let y = resample(&x, 48_000, 22_050);
        let rms = (y[2000..20_000].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 18_000.0).sqrt();
        assert!(rms < 1e-3, "{rms}");
    }
}
